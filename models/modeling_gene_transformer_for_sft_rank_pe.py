# coding=utf-8
"""
Gene Transformer Model for Stage2 SFT
Refactored to align with Show-o VQ-VAE paradigm + Qwen SFT best practices.

Key design:
1. No cell_embedder, no gene_embedder. Gene tokens are standard discrete tokens.
2. Embedding table resized for gene codebook tokens.
3. Weak rank signal injection via 2-layer MLP (no periodic PE).
4. Standard 1D RoPE only; omni 4D attention mask for gene bidirectional region.
5. SFT: standard NTP loss only (gene acts as full condition, no masking, no gene loss).
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from pathlib import Path
from transformers import AutoConfig

from .misc import next_token_prediction
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .modules import DiffusionHeadConfig, ModulatedAttentionBlock, RMSNorm
from .sc_omni_attention import omni_attn_mask_vectorized


def load_special_tokens_config(config_path: str = None) -> Dict:
    if config_path is None:
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)


class RankSignalInjector(nn.Module):
    """
    Weak rank signal injector: no PE, no embedding table, only continuous scalar MLP.
    x_i = E_token + alpha * MLP(log(1 + r_i) / log(1 + max_rank))
    """
    def __init__(self, hidden_size: int, max_rank: int = 1200):
        super().__init__()
        self.max_rank = max_rank
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, ranks: torch.Tensor) -> torch.Tensor:
        feat = torch.log1p(ranks.float()) / math.log(1 + self.max_rank)
        feat = feat.unsqueeze(-1)
        in_dtype = self.mlp[0].weight.dtype
        feat = feat.to(dtype=in_dtype)
        out = self.mlp(feat)
        return out * self.alpha.to(dtype=out.dtype)


class GenePredictionHead(nn.Module):
    def __init__(self, hidden_size, gene_vocab_size):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, gene_vocab_size, bias=True)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.norm_final(x)
        logits = self.linear(x)
        return logits


class GeneTransformer(nn.Module):
    def __init__(
        self,
        llm_vocab_size: int,
        llm_model_path: str,
        load_from_showo: bool = False,
        config_dict: Dict[str, Any] = None,
        special_tokens_ids: Dict[str, int] = None,
    ):
        super().__init__()

        assert config_dict is not None, "必须传入 config_dict!"
        assert special_tokens_ids is not None, "必须传入 special_tokens_ids 字典!"

        self.model_config = config_dict.get('model', {})
        self.dataset_config = config_dict.get('dataset', {})
        self.training_config = config_dict.get('training', {})
        self.optimizer_config = config_dict.get('optimizer', {})
        self.data_config = config_dict.get('data', {})

        self.hidden_size = self.model_config.get('hidden_size', 3584)
        self.num_diffusion_layers = self.model_config.get('num_diffusion_layers', 10)
        self.dropout_rate = self.model_config.get('dropout_rate', 0.0)
        self.use_flash_attn = self.model_config.get('use_flash_attn', True)
        self.attention_type = self.model_config.get('attention_type', 'omni')
        self.load_from_showo = load_from_showo
        # 与 stage1 对齐：gene 段位置控制（NoPE / Bucketed RoPE）
        self.disable_gene_position_ids = bool(self.model_config.get('disable_gene_position_ids', False))
        self.gene_position_buckets = int(self.model_config.get('gene_position_buckets', 64))
        if self.gene_position_buckets < 1:
            self.gene_position_buckets = 1

        self.max_genes = self.dataset_config.get('max_genes', 1200)
        self.clusters_per_gene = self.dataset_config.get('clusters_per_gene', 64)
        self.gene_vocab_size = self.dataset_config.get('gene_vocab_size', 0)

        # 加载 scGPT vocab size（输出头预测 scGPT gene id）
        scgpt_vocab_path = self.data_config.get('scgpt_gene_vocab', '')
        if scgpt_vocab_path and Path(scgpt_vocab_path).exists():
            with open(scgpt_vocab_path, 'r') as f:
                scgpt_vocab = json.load(f)
            self.scgpt_vocab_size = max(int(v) for v in scgpt_vocab.values()) + 1
        else:
            self.scgpt_vocab_size = 0

        self.weight_decay = self.optimizer_config.get('weight_decay', 1e-4)
        self.clip_grad_norm = self.optimizer_config.get('clip_grad_norm', 1.0)
        self.gradient_checkpointing = self.training_config.get('gradient_checkpointing', True)
        self.mtp_mask_prob_eps = float(self.training_config.get('mtp_mask_prob_eps', 0.1))

        print(f"✅ 从配置加载模型参数:")
        print(f"   - hidden_size: {self.hidden_size}")
        print(f"   - num_diffusion_layers: {self.num_diffusion_layers}")
        print(f"   - attention_type: {self.attention_type}")
        print(f"   - weight_decay: {self.weight_decay}")
        print(f"   - clip_grad_norm: {self.clip_grad_norm}")
        print(f"   - gradient_checkpointing: {self.gradient_checkpointing}")
        print(f"   - disable_gene_position_ids: {self.disable_gene_position_ids}")
        print(f"   - gene_position_buckets: {self.gene_position_buckets}")

        self.sog_id = special_tokens_ids.get('sog_id')
        self.eog_id = special_tokens_ids.get('eog_id')
        self.mask_gene_id = special_tokens_ids.get('mask_gene_id')

        self._logged_mask_info = False

        # ==========================================
        # 1. 加载 LLM 骨干
        # ==========================================
        llm_model_path = llm_model_path or self.model_config.get('llm_model_path')
        local_files_only = self.model_config.get('local_files_only', True)

        llm_config = AutoConfig.from_pretrained(llm_model_path, local_files_only=local_files_only)
        llm_config.use_cache = False

        from .qwen2 import Qwen2ForCausalLM
        if self.load_from_showo:
            self.showo = Qwen2ForCausalLM(llm_config)
        else:
            self.showo = Qwen2ForCausalLM.from_pretrained(
                llm_model_path,
                config=llm_config,
                attn_implementation='sdpa',
                local_files_only=local_files_only,
            )

        self.showo.config.use_cache = False
        if hasattr(self.showo.model, 'config'):
            self.showo.model.config.use_cache = False

        if self.gradient_checkpointing:
            self.showo.enable_input_require_grads()
            self.showo.gradient_checkpointing_enable()
            print("✅ Gradient Checkpointing 已启用")

        # ==========================================
        # 2. 独立 gene embedding（完全解耦）
        # ==========================================
        self.text_vocab_size = llm_vocab_size or self.showo.config.vocab_size
        codebook_dir = self.data_config.get('codebook_dir', '/root/wanghaoran/zxy/project/sc_showo/run/gene_codebook_data')
        prefer_compact = bool(self.data_config.get('prefer_compact_codebook', True))
        self._build_gene_vocab_and_embed(codebook_dir, prefer_compact=prefer_compact)
        self.gene_mask_id = self.gene_vocab_size
        self.mask_gene_id = self.text_vocab_size + self.gene_mask_id

        proj_mid = self.hidden_size * 4
        self.gene_proj = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, proj_mid),
            nn.GELU(),
            nn.Linear(proj_mid, proj_mid),
            nn.GELU(),
            nn.Linear(proj_mid, self.hidden_size),
        )

        # ==========================================
        # 3. Rank Signal 注入器
        # ==========================================
        self.rank_signal_injector = RankSignalInjector(self.hidden_size, max_rank=self.max_genes)

        # ==========================================
        # 4. 离散扩散预测头（保留结构，SFT 阶段不计算 loss）
        # ==========================================
        qwen_config = self.showo.config
        real_head_dim = qwen_config.hidden_size // qwen_config.num_attention_heads

        print(f"\n🔧 Diffusion Head 配置对齐 Qwen2.5 底座:")
        print(f"   - Qwen hidden_size: {qwen_config.hidden_size}")
        print(f"   - Qwen num_attention_heads: {qwen_config.num_attention_heads}")
        print(f"   - Qwen intermediate_size: {qwen_config.intermediate_size}")
        print(f"   - head_dim: {real_head_dim}")

        self.diffusion_head_config = DiffusionHeadConfig(
            hidden_size=qwen_config.hidden_size,
            num_attention_heads=qwen_config.num_attention_heads,
            head_dim=real_head_dim,
            intermediate_size=qwen_config.intermediate_size,
            num_layers=self.num_diffusion_layers,
            attention_dropout=self.dropout_rate,
        )

        self.diffusion_head_a = nn.ModuleList([
            ModulatedAttentionBlock(self.diffusion_head_config, layer_idx)
            for layer_idx in range(self.num_diffusion_layers)
        ])
        self.diffusion_head_b = GenePredictionHead(qwen_config.hidden_size, self.scgpt_vocab_size)

        self.reset_parameters()
        self.register_buffer('dummy_float', torch.tensor(0.0), persistent=False)
        self.model = self.showo

    def _build_gene_vocab_and_embed(self, codebook_dir: str, prefer_compact: bool = True):
        compact_tensor_path = Path(codebook_dir) / 'codebook_tensor_compact.pt'
        compact_map_path = Path(codebook_dir) / 'gene_to_compact_idx.json'
        full_tensor_path = Path(codebook_dir) / 'codebook_tensor.pt'
        nclusters_path = Path(codebook_dir) / 'gene_nclusters.json'
        if not nclusters_path.exists():
            print(f"⚠️  codebook 文件未找到，将随机初始化 gene_embed_tokens")
            if self.gene_vocab_size <= 0:
                self.gene_vocab_size = int(self.dataset_config.get('max_genes', 1200)) * int(self.clusters_per_gene)
            if self.gene_vocab_size <= 0:
                self.gene_vocab_size = int(self.dataset_config.get('max_genes', 1200)) * int(self.clusters_per_gene)
            self.gene_embed_tokens = nn.Embedding(self.gene_vocab_size + 1, 768)
            self.gene_embed_tokens.weight.requires_grad = False
            return

        use_compact = bool(prefer_compact and compact_tensor_path.exists() and compact_map_path.exists())
        if use_compact:
            codebook_tensor = torch.load(compact_tensor_path, map_location='cpu', weights_only=True)
            with open(compact_map_path, 'r') as f:
                gene_to_compact_idx = {int(k): int(v) for k, v in json.load(f).items()}
            print(f"✅ 使用 compact codebook: {compact_tensor_path.name}")
        elif full_tensor_path.exists():
            codebook_tensor = torch.load(full_tensor_path, map_location='cpu', weights_only=True)
            gene_to_compact_idx = None
            print(f"✅ 使用 full codebook: {full_tensor_path.name}")
        else:
            print(f"⚠️  codebook tensor 不存在，将随机初始化 gene_embed_tokens")
            self.gene_embed_tokens = nn.Embedding(self.gene_vocab_size + 1, 768)
            self.gene_embed_tokens.weight.requires_grad = False
            return

        with open(nclusters_path, 'r') as f:
            nclusters = {int(k): int(v) for k, v in json.load(f).items()}

        valid_genes = sorted([gid for gid, k in nclusters.items() if k > 0])
        # 保留所有有效基因，不再截断到 num_genes
        self.scgpt_to_local = {gid: idx for idx, gid in enumerate(valid_genes)}
        self.local_to_scgpt = valid_genes
        actual_gene_vocab_size = len(valid_genes) * self.clusters_per_gene

        if actual_gene_vocab_size != self.gene_vocab_size:
            print(f"⚠️  gene_vocab_size 调整: {self.gene_vocab_size} -> {actual_gene_vocab_size}")
            self.gene_vocab_size = actual_gene_vocab_size

        self.gene_embed_tokens = nn.Embedding(self.gene_vocab_size + 1, 768)
        with torch.no_grad():
            for local_id, scgpt_id in enumerate(valid_genes):
                k = nclusters[scgpt_id]
                for c in range(k):
                    flat_id = local_id * self.clusters_per_gene + c
                    if use_compact:
                        compact_idx = gene_to_compact_idx.get(scgpt_id, None)
                        if compact_idx is None:
                            continue
                        self.gene_embed_tokens.weight[flat_id] = codebook_tensor[compact_idx, c]
                    else:
                        self.gene_embed_tokens.weight[flat_id] = codebook_tensor[scgpt_id, c]
        self.gene_embed_tokens.weight.requires_grad = False
        print(f"✅ 初始化 gene_embed_tokens: vocab={self.gene_vocab_size}, base={len(valid_genes)} genes x {self.clusters_per_gene} clusters, frozen")

        # 保存文本 embedding 层引用，避免 LoRA 注入后 PeftModel 改变访问路径
        self.text_embed_tokens = self.showo.model.embed_tokens

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def reset_parameters(self):
        for m in self.gene_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.rank_signal_injector.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _to_model_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.dummy_float.dtype)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        modality_positions: Optional[torch.LongTensor] = None,
        gene_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        gene_labels: Optional[torch.LongTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        data_type: Optional[List[str]] = None,
        gene_embeddings: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        b, seq_len = input_ids.shape
        device = input_ids.device

        # 1. 解耦文本 / 基因 token embedding
        text_mask = input_ids < self.text_vocab_size
        gene_mask = ~text_mask
        input_embeds = torch.empty(b, seq_len, self.hidden_size, dtype=self.text_embed_tokens.weight.dtype, device=device)
        if text_mask.any():
            input_embeds[text_mask] = self.text_embed_tokens(input_ids[text_mask])
        if gene_mask.any():
            if gene_embeddings is not None:
                # 与 stage1 对齐：优先按 modality_positions 的真实 gene span 对齐；mask 位强制用 mask token embedding
                mask_token_768 = self.gene_embed_tokens.weight[self.gene_mask_id].unsqueeze(0)  # [1, 768]
                mask_token_768 = F.normalize(mask_token_768, p=2, dim=-1)
                mask_token_768 = torch.nan_to_num(mask_token_768, nan=0.0)
                mask_proj = self.gene_proj(mask_token_768).squeeze(0)  # [hidden]

                for i in range(b):
                    if modality_positions is not None:
                        g_start = int(modality_positions[i, 0, 0].item())
                        g_len = int(modality_positions[i, 0, 1].item())
                        g_end = min(g_start + g_len, seq_len)
                        gene_pos = torch.arange(g_start, g_end, device=device, dtype=torch.long)
                    else:
                        gene_pos = torch.where(gene_mask[i])[0]

                    n_pos = int(gene_pos.numel())
                    if n_pos <= 0:
                        continue

                    n_emb = int(gene_embeddings[i].shape[0])
                    n_use = min(n_pos, n_emb)
                    if n_use <= 0:
                        continue
                    gene_pos = gene_pos[:n_use]

                    gene_emb_i = gene_embeddings[i, :n_use].to(mask_token_768.dtype)
                    gene_emb_i = F.normalize(gene_emb_i, p=2, dim=-1)
                    gene_emb_i = torch.nan_to_num(gene_emb_i, nan=0.0)
                    proj_i = self.gene_proj(gene_emb_i)

                    masked_local = (input_ids[i, gene_pos] == self.mask_gene_id)
                    if masked_local.any():
                        proj_i[masked_local] = mask_proj

                    input_embeds[i, gene_pos] = proj_i
            else:
                gene_ids_flat = input_ids[gene_mask] - self.text_vocab_size
                gene_embeds_768 = self.gene_embed_tokens(gene_ids_flat)
                gene_embeds_768 = F.normalize(gene_embeds_768, p=2, dim=-1)
                gene_embeds_768 = torch.nan_to_num(gene_embeds_768, nan=0.0)
                input_embeds[gene_mask] = self.gene_proj(gene_embeds_768)

        if attention_mask is not None:
            attention_mask = self._to_model_dtype(attention_mask)

        # 2. 基因区域注入弱 rank signal（当 gene position 启用时）
        if (not self.disable_gene_position_ids) and modality_positions is not None:
            for i in range(b):
                g_start = modality_positions[i, 0, 0].item()
                g_len = modality_positions[i, 0, 1].item()
                if g_len > 0:
                    ranks = torch.arange(g_len, device=device, dtype=torch.float32)
                    rank_signal = self.rank_signal_injector(ranks)
                    input_embeds[i, g_start:g_start + g_len] = input_embeds[i, g_start:g_start + g_len] + rank_signal

        # 3. Omni-Attention 4D mask
        combined_modalities = []
        for i in range(b):
            mods = []
            if modality_positions is not None:
                g_start = modality_positions[i, 0, 0].item()
                g_len = modality_positions[i, 0, 1].item()
                if g_len > 0:
                    mods.append((g_start, g_len))
            combined_modalities.append(mods)

        omni_mask_binary = omni_attn_mask_vectorized(
            B=b,
            LEN=seq_len,
            modalities=combined_modalities,
            device=device,
            inverted=False,
        ).bool()

        if attention_mask is not None:
            omni_mask_binary &= attention_mask.unsqueeze(1).unsqueeze(2).bool()

        dtype = input_embeds.dtype
        omni_mask_4d = torch.zeros((b, 1, seq_len, seq_len), dtype=dtype, device=device)
        omni_mask_4d.masked_fill_(~omni_mask_binary, torch.finfo(dtype).min)

        if not self._logged_mask_info:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"🎭 Omni-Attention Mask: 文本=Causal, Gene=Bidirectional")
            self._logged_mask_info = True

        # 4. position_ids 规范化：纯 1D RoPE
        if position_ids is not None and not isinstance(position_ids, torch.Tensor):
            if isinstance(position_ids, list):
                max_len = max(x.shape[-1] for x in position_ids)
                padded = []
                for x in position_ids:
                    if x.shape[-1] < max_len:
                        pad = torch.zeros(x.shape[0], max_len - x.shape[-1], dtype=x.dtype, device=x.device)
                        x = torch.cat([x, pad], dim=-1)
                    padded.append(x)
                position_ids = torch.stack(padded, dim=0)
            else:
                position_ids = torch.tensor(position_ids, device=device)

        if position_ids is not None and position_ids.dim() == 3 and position_ids.shape[1] == 2:
            # [B, 2, S] -> 纯 1D RoPE 使用第一维位置
            position_ids_for_model = position_ids[:, 0, :].long()
        elif position_ids is not None and position_ids.dim() == 3 and position_ids.shape[0] == 2:
            # [2, B, S]
            position_ids_for_model = position_ids[0, :, :].long()
        else:
            position_ids_for_model = position_ids

        # 与 stage1 一致的 gene 段位置策略
        if modality_positions is not None:
            if position_ids_for_model is None:
                position_ids_for_model = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, -1).long()
            elif position_ids_for_model.dim() == 1:
                position_ids_for_model = position_ids_for_model.unsqueeze(0).expand(b, -1).clone().long()
            else:
                position_ids_for_model = position_ids_for_model.clone().long()

            for i in range(b):
                g_start = int(modality_positions[i, 0, 0].item())
                g_len = int(modality_positions[i, 0, 1].item())
                if g_len <= 0:
                    continue
                g_end = min(g_start + g_len, position_ids_for_model.shape[1])
                span = g_end - g_start
                if span <= 0:
                    continue

                if self.disable_gene_position_ids:
                    position_ids_for_model[i, g_start:g_end] = 0
                else:
                    num_buckets = min(self.gene_position_buckets, span)
                    local = torch.arange(span, device=device, dtype=torch.long)
                    bucket = torch.div(local * num_buckets, span, rounding_mode='floor')
                    position_ids_for_model[i, g_start:g_end] = g_start + bucket

        # 5. LLM 前向
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=omni_mask_4d,
            position_ids=position_ids_for_model,
            output_hidden_states=True,
            **kwargs
        )

        logits = outputs.logits
        last_hidden_states = outputs.hidden_states[-1]

        # 6. 扩散头（前向计算保持计算图活跃，但 SFT 不取 loss）
        if hasattr(self, 'diff_proj'):
            last_hidden_states = self.diff_proj(last_hidden_states)

        dummy_adaln = torch.zeros(b, self.diffusion_head_config.hidden_size, device=device, dtype=last_hidden_states.dtype)
        for layer in self.diffusion_head_a:
            last_hidden_states = layer(
                hidden_states=last_hidden_states,
                adaln_input=dummy_adaln,
                attention_mask=omni_mask_4d,
                position_ids=position_ids_for_model,
                modality_positions=modality_positions,
            )[0]

        gene_logits = self.diffusion_head_b(last_hidden_states)

        # 7. Loss 计算：
        # - stage2(SFT): 计算 NTP
        # - stage1_understanding(配对理解): 计算 NTP
        # - stage1_generation(配对生成): 只算 MTP(gene masked token prediction)
        loss_ntp = (logits * 0.0).sum()
        loss_gene = (gene_logits * 0.0).sum()

        if labels is not None:
            batch_size = input_ids.shape[0]
            if data_type is None:
                data_type = ['stage2'] * batch_size

            ntp_labels = labels.clone()
            for i in range(batch_size):
                if modality_positions is not None:
                    g_start = modality_positions[i, 0, 0].item()
                    g_len = modality_positions[i, 0, 1].item()
                    if g_len > 0:
                        gene_end = min(g_start + g_len + 1, ntp_labels.shape[1])
                        ntp_labels[i, max(g_start - 1, 0):gene_end] = -100

            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = ntp_labels[..., 1:].contiguous()
            n_vocab = self.showo.config.vocab_size

            raw_ntp_loss = F.cross_entropy(
                shift_logits.view(-1, n_vocab),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='none',
            ).view(shift_labels.shape)

            stage2_cfg = self.training_config.get('stage2', {})
            stage2_lambda_ntp = stage2_cfg.get('lambda_ntp', 1.0)

            sample_mask = torch.tensor(
                [dt in {'stage2', 'stage1_understanding'} for dt in data_type],
                device=shift_labels.device,
                dtype=torch.bool,
            )
            valid_mask = (shift_labels != -100) & sample_mask.unsqueeze(1)
            if valid_mask.any():
                loss_ntp = loss_ntp + raw_ntp_loss[valid_mask].mean() * stage2_lambda_ntp

            # Stage1 paired generation 仅贡献 MTP(gene) loss
            stage2_lambda_gene = stage2_cfg.get('lambda_gene', 1.0)
            mtp_num = torch.tensor(0.0, device=device, dtype=torch.float32)
            mtp_den = torch.tensor(0.0, device=device, dtype=torch.float32)

            for i, dt in enumerate(data_type):
                if dt != 'stage1_generation':
                    continue
                if modality_positions is None:
                    continue

                g_start = modality_positions[i, 0, 0].item()
                g_len = modality_positions[i, 0, 1].item()
                if g_len <= 0:
                    continue

                labels_i_full = labels[i]
                actual_g_len = min(g_len, labels_i_full.shape[0] - g_start)
                if actual_g_len <= 0:
                    continue

                gene_logits_i = gene_logits[i, g_start:g_start + actual_g_len].float()
                mtp_labels_i = labels_i_full[g_start:g_start + actual_g_len]
                valid_i = (mtp_labels_i != -100)
                if not valid_i.any():
                    continue

                safe_labels_i = mtp_labels_i.clone()
                safe_labels_i[(safe_labels_i != -100) & ((safe_labels_i < 0) | (safe_labels_i >= self.scgpt_vocab_size))] = 0

                per_token_loss = F.cross_entropy(
                    gene_logits_i,
                    safe_labels_i,
                    reduction='none',
                )

                t_i = float(t[i].item()) if t is not None else 1.0
                p_mask = max(self.mtp_mask_prob_eps, min(t_i, 0.999999))

                mtp_num = mtp_num + (per_token_loss[valid_i] / p_mask).sum()
                mtp_den = mtp_den + float(actual_g_len)

            if mtp_den.item() > 0:
                loss_gene = loss_gene + (mtp_num / mtp_den).to(input_embeds.dtype) * stage2_lambda_gene

        detailed_losses = {
            'ntp_sft': 0.0, 'ntp_stage1': 0.0,
            'gene_sft': 0.0, 'gene_stage1': 0.0
        }

        counts = {'stage2': 0, 'stage1_understanding': 0, 'stage1_generation': 0}
        if labels is not None and data_type is not None:
            for i, dt in enumerate(data_type):
                if dt not in counts:
                    continue
                counts[dt] += 1

                if dt in ['stage2', 'stage1_understanding']:
                    sample_shift_logits = shift_logits[i]
                    sample_shift_labels = shift_labels[i]
                    valid_mask_i = sample_shift_labels != -100
                    if valid_mask_i.any():
                        item_loss = F.cross_entropy(
                            sample_shift_logits[valid_mask_i],
                            sample_shift_labels[valid_mask_i],
                            reduction='mean'
                        ).item()
                        if dt == 'stage2':
                            detailed_losses['ntp_sft'] += item_loss
                        else:
                            detailed_losses['ntp_stage1'] += item_loss

                if dt == 'stage1_generation' and modality_positions is not None:
                    g_start = modality_positions[i, 0, 0].item()
                    g_len = modality_positions[i, 0, 1].item()
                    if g_len > 0:
                        actual_g_len = min(g_len, labels[i].shape[0] - g_start)
                        if actual_g_len > 0:
                            gene_logits_i = gene_logits[i, g_start:g_start + actual_g_len].float()
                            mtp_labels_i = labels[i, g_start:g_start + actual_g_len]
                            mtp_mask_i = mtp_labels_i != -100
                            if mtp_mask_i.any():
                                safe_labels_i = mtp_labels_i.clone()
                                safe_labels_i[(safe_labels_i < 0) | (safe_labels_i >= self.scgpt_vocab_size)] = 0
                                item_gene = F.cross_entropy(
                                    gene_logits_i[mtp_mask_i],
                                    safe_labels_i[mtp_mask_i],
                                    reduction='mean'
                                ).item()
                                detailed_losses['gene_stage1'] += item_gene

            if counts['stage2'] > 0:
                detailed_losses['ntp_sft'] /= counts['stage2']
                detailed_losses['gene_sft'] = 0.0
            if counts['stage1_understanding'] > 0:
                detailed_losses['ntp_stage1'] /= counts['stage1_understanding']
            if counts['stage1_generation'] > 0:
                detailed_losses['gene_stage1'] /= counts['stage1_generation']

        return logits, loss_ntp, loss_gene, detailed_losses
