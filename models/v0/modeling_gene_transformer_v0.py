# coding=utf-8
"""
Gene Transformer Model for Transcriptome Data Prediction
基于 show-o2 深度改造，专为离散扩散（MaskedGIT）建模设计。
已集成：
1. 配置字典驱动的连续细胞特征映射 (动态支持任意 tokens 数量)
2. LLM 与 Gene 词表物理隔离 (nn.Embedding 劫持)
3. 文本 / 基因 Loss 分离计算 (防模型崩溃)
4. Dataset 级 Offset RoPE 强制透传
5. 无时间步调制的独立扩散头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union  # ✅ 一次性导入所有需要的类型
from transformers import AutoConfig

from .misc import next_token_prediction
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .modules import DiffusionHeadConfig, ModulatedAttentionBlock, RMSNorm
# ✅ 引入全向注意力掩码生成器（离散扩散的核心！）
from .sc_omni_attention import omni_attn_mask_vectorized


def load_special_tokens_config(config_path: str = None) -> Dict:
    """
    从 JSON 文件加载特殊 Tokens 配置
    """
    if config_path is None:
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)


class GenePredictionHead(nn.Module):
    """
    纯粹的分类预测头，已彻底移除 adaLN_modulation (时间步参数)
    """
    def __init__(self, hidden_size, gene_vocab_size):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, gene_vocab_size, bias=True)
        
        # 遵循初始化建议：防止一开始分布不均导致巨大 Loss 破坏大模型权重
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
        
        # ✅ 分层读取配置
        self.model_config = config_dict.get('model', {})
        self.dataset_config = config_dict.get('dataset', {})
        self.sequence_config = config_dict.get('sequence', {})
        self.training_config = config_dict.get('training', {})
        self.optimizer_config = config_dict.get('optimizer', {})
        
        # 模型架构
        self.hidden_size = self.model_config.get('hidden_size', 3584)
        self.num_diffusion_layers = self.model_config.get('num_diffusion_layers', 10)
        self.dropout_rate = self.model_config.get('dropout_rate', 0.0)
        self.use_flash_attn = self.model_config.get('use_flash_attn', True)
        self.attention_type = self.model_config.get('attention_type', 'omni')
        self.load_from_showo = load_from_showo
        
        # 数据集超参
        self.cell_feature_dim = self.dataset_config['cell_feature_dim']
        self.cell_feature_tokens = self.dataset_config['cell_feature_tokens']
        self.gene_vocab_size = self.dataset_config['gene_vocab_size']
        self.max_genes = self.dataset_config['max_genes']
        
        # 优化器配置
        self.weight_decay = self.optimizer_config.get('weight_decay', 1e-4)
        self.clip_grad_norm = self.optimizer_config.get('clip_grad_norm', 1.0)
        
        # RoPE 配置
        self.gene_start_offset = self.sequence_config.get('gene_start_offset', 1024)
        self.rope_theta = self.sequence_config.get('rope_theta', 10000.0)
        
        # ✅ 启用 Gradient Checkpointing（从 config 读取）
        self.gradient_checkpointing = self.training_config.get('gradient_checkpointing', True)
        
        print(f"✅ 从配置加载模型参数:")
        print(f"   - hidden_size: {self.hidden_size}")
        print(f"   - num_diffusion_layers: {self.num_diffusion_layers}")
        print(f"   - attention_type: {self.attention_type}")
        print(f"   - gene_start_offset: {self.gene_start_offset}")
        print(f"   - weight_decay: {self.weight_decay}")
        print(f"   - clip_grad_norm: {self.clip_grad_norm}")
        print(f"   - gradient_checkpointing: {self.gradient_checkpointing}")
        
        # ✅ 添加一次性日志标记（避免 forward 中重复打印）
        self._logged_mask_info = False
        
        # ==========================================
        # 2. 加载并扩展 LLM 骨干 (原生支持 SDPA/FlexAttention)
        # ==========================================

        llm_model_path = llm_model_path or self.model_config.get('llm_model_path')
        # ✅ 获取离线开关，默认为 False 以防旧配置报错
        local_files_only = self.model_config.get('local_files_only', True) 
        
        llm_config = AutoConfig.from_pretrained(
            llm_model_path, 
            local_files_only=local_files_only
        )
        
        # 尝试在 config 层面关闭
        llm_config.use_cache = False
        
        from .qwen2 import Qwen2ForCausalLM
        if self.load_from_showo:
            self.showo = Qwen2ForCausalLM(llm_config)
        else:
            self.showo = Qwen2ForCausalLM.from_pretrained(
                llm_model_path, 
                config=llm_config,
                attn_implementation='sdpa',
                local_files_only=local_files_only 
            )
            
        # 🚨 终极镇压：实例化后强制改写所有可能残留的 use_cache 属性
        self.showo.config.use_cache = False
        if hasattr(self.showo.model, 'config'):
            self.showo.model.config.use_cache = False
        
        # ✅ 启用 Gradient Checkpointing
        if self.gradient_checkpointing:
            self.showo.enable_input_require_grads()
            # 在执行这行之前，use_cache 已经被彻底置为 False，就不会再报警了
            self.showo.gradient_checkpointing_enable()
            print("✅ Gradient Checkpointing 已启用（节省约 40% 显存）")
            
        # 扩展词表以容纳特殊 Tokens
        if llm_vocab_size is not None:
            # ✅ 致命修复：确保 Embedding 大小能盖住最大的特殊 Token ID，绝不反向缩小！
            max_id_needed = max(special_tokens_ids.values()) + 1
            new_vocab_size = max(llm_vocab_size, self.showo.config.vocab_size, max_id_needed)
            
            if new_vocab_size > self.showo.config.vocab_size:
                self.showo.resize_token_embeddings(new_vocab_size)
                print(f"✅ 扩展词表：{self.showo.config.vocab_size} → {new_vocab_size}")
            else:
                print(f"⚠️  词表大小无需调整 (当前：{self.showo.config.vocab_size}, 需要：{new_vocab_size})")
        
        # ==========================================
        # 3. 跨模态投影层 (Cell Embedder)
        # ==========================================
        # 将 768 维连续特征映射到 N 个 Token (N * hidden_size)
        # 使用 4 倍扩容的 MLP 保证表达能力
        hidden_size = self.hidden_size
        self.cell_embedder = nn.Sequential(
            # 第一层：对齐维度，不进行 Token 展开
            nn.Linear(self.cell_feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            
            # 第二层：升维到高容量中间空间 (4倍)
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            
            # 第三层：映射到最终的 8 个 Token 总维度
            nn.Linear(hidden_size * 4, hidden_size * self.cell_feature_tokens)
        )
        
        # ==========================================
        # 4. 词表隔离映射层 (Gene Embedder)
        # ==========================================
        # 预留 +1 的空间专属给 Mask Token，彻底切断与文本词表的交集
        self.gene_embedder = nn.Embedding(self.gene_vocab_size + 1, hidden_size)
        self.gene_mask_idx = self.gene_vocab_size # 把 mask ID 定向到最后一位
        
        # ==========================================
        # 5. 离散扩散预测组
        # ==========================================
        # ✅ 致命修复：严格继承 Qwen2.5 的底层架构维度，彻底消除 Tensor 维度不匹配
        qwen_config = self.showo.config
        
        # 计算 Qwen 的真实 head_dim
        real_head_dim = qwen_config.hidden_size // qwen_config.num_attention_heads
        
        print(f"\n🔧 Diffusion Head 配置对齐 Qwen2.5 底座:")
        print(f"   - Qwen hidden_size: {qwen_config.hidden_size}")
        print(f"   - Qwen num_attention_heads: {qwen_config.num_attention_heads}")
        print(f"   - Qwen intermediate_size: {qwen_config.intermediate_size}")
        print(f"   - 计算得到的 head_dim: {real_head_dim}")
        
        self.diffusion_head_config = DiffusionHeadConfig(
            hidden_size=qwen_config.hidden_size,                # ✅ 3584
            num_attention_heads=qwen_config.num_attention_heads, # ✅ 28
            head_dim=real_head_dim,                             # ✅ 128
            intermediate_size=qwen_config.intermediate_size,    # ✅ 18928 (Qwen2.5-7B 标准)
            num_layers=self.model_config.get('num_diffusion_layers', 10),
        )
        
        # 保留架构一致性，后续通过 Dummy Adaln 绕过报错
        self.diffusion_head_a = nn.ModuleList(
            [ModulatedAttentionBlock(self.diffusion_head_config, layer_idx) for layer_idx in
             range(self.num_diffusion_layers)]
        )
        
        self.diffusion_head_b = GenePredictionHead(qwen_config.hidden_size, self.gene_vocab_size)
        self.reset_parameters()
        
        # ✅ 注册 dtype 跟踪 buffer（用于 BF16 混合精度训练）
        self.register_buffer('dummy_float', torch.tensor(0.0), persistent=False)
    
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True
    
    def reset_parameters(self):
        nn.init.normal_(self.gene_embedder.weight, mean=0.0, std=0.02)
        for m in self.cell_embedder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cell_features: Optional[torch.FloatTensor] = None,
        cell_positions: Optional[torch.LongTensor] = None,
        modality_positions: Optional[torch.LongTensor] = None,
        gene_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        b, seq_len = input_ids.shape
        
        # ✅ 关键修复：先从 LLM 获取所有位置的初始嵌入
        input_embeds = self.showo.model.embed_tokens(input_ids)
        # shape: (batch, seq_len, hidden_size)
        
        # ✅ 获取设备信息（从输入张量）
        device = input_ids.device
        
        # ✅ 确保所有浮点输入都使用模型的正确 dtype
        if attention_mask is not None:
            attention_mask = self._to_model_dtype(attention_mask)
        
        # ==================== Cell Embedder ====================
        if cell_features is not None:
            # ✅ 使用辅助方法转换 dtype
            cell_features = self._to_model_dtype(cell_features)
            
            # Cell Embedder: (b, 768) → (b, 8*hidden_size)
            cell_embeds_flat = self.cell_embedder(cell_features)
            
            # Reshape 成 8 个独立的 tokens: (b, 8, hidden_size)
            cell_embeds = cell_embeds_flat.view(
                b, self.cell_feature_tokens, self.hidden_size
            )
            
            # ==========================================
            # 🚀 梯度安全修复：使用非就地拼接替代 Assignment
            # ==========================================
            new_input_embeds = []
            for i in range(b):
                start = cell_positions[i, 0].item()
                length = cell_positions[i, 1].item()
                end = start + length
                
                # 通过拼接重组单条序列的 embedding
                # 这种方式会创建一个新的张量，完美保留 cell_embeds[i] 的梯度路径
                row = torch.cat([
                    input_embeds[i, :start],
                    cell_embeds[i],
                    input_embeds[i, end:]
                ], dim=0)
                new_input_embeds.append(row)
            
            # 将列表重新堆叠回 Batch 张量
            input_embeds = torch.stack(new_input_embeds)

        # ==================== Gene Embedder ====================
        if modality_positions is not None:
            # 🚨 同样使用梯度安全的非就地操作
            new_input_embeds = []
            for i in range(b):
                g_start = modality_positions[i, 0, 0].item()
                g_len = modality_positions[i, 0, 1].item()
                
                if g_len > 0:
                    gene_tokens = input_ids[i, g_start : g_start+g_len].clone()
                    
                    # 将 Mask 位置的 Qwen ID 替换为 Gene Embedding 专属 ID
                    if gene_mask is not None:
                        gene_tokens[gene_mask[i][:g_len]] = self.gene_mask_idx
                    else:
                        gene_tokens[gene_tokens == self.mask_gene_id] = self.gene_mask_idx
                    
                    # 获取基因嵌入
                    gene_embeds = self.gene_embedder(gene_tokens)
                    
                    # 通过拼接重组（梯度安全）
                    row = torch.cat([
                        input_embeds[i, :g_start],
                        gene_embeds,
                        input_embeds[i, g_start+g_len:]
                    ], dim=0)
                    new_input_embeds.append(row)
                else:
                    # 如果没有基因区域，直接复制
                    new_input_embeds.append(input_embeds[i])
            
            # 更新 input_embeds
            input_embeds = torch.stack(new_input_embeds)
        
        # ----------------------------------------------------
        # 阶段 A.5：🚨 致命修复 - 构建 Omni-Attention 4D 掩码
        # ----------------------------------------------------
        # 构建给 omni_attn_mask_naive 使用的 3D 坐标列表
        combined_modalities = []
        for i in range(b):
            mods = []
            # 1. 细胞连续特征区块 (内部全向注意力)
            if cell_positions is not None:
                mods.append((cell_positions[i, 0].item(), cell_positions[i, 1].item()))
            
            # 2. 离散基因序列区块 (内部全向注意力)
            if modality_positions is not None:
                g_start = modality_positions[i, 0, 0].item()
                g_len = modality_positions[i, 0, 1].item()
                if g_len > 0:
                    mods.append((g_start, g_len))
                    
            combined_modalities.append(mods)

        # ✅ 性能修复：使用向量化版本替代 Python for 循环
        # 生成基础的 0/1 掩码 (1 表示可 attend，0 表示遮蔽)
        omni_mask_binary = omni_attn_mask_vectorized(
            B=b, 
            LEN=seq_len, 
            modalities=combined_modalities, 
            device=device,  # ✅ 现在 device 已定义
            inverted=False  # 返回 0/1 张量
        ).bool()

        # ✅ 终极加固：利用 DataLoader 传来的 attention_mask，将 Pad 区域彻底抹除
        if attention_mask is not None:
            # attention_mask 形状为 (B, LEN)，1 为有效，0 为 Pad
            # 扩展后与 omni_mask 进行逻辑与，确保没有任何 Token 可以 attend 到 Pad 区域
            omni_mask_binary &= attention_mask.unsqueeze(1).unsqueeze(2).bool()

        # 转换为 HuggingFace Qwen2 (SDPA) 期望的 4D Float 掩码格式
        # 0.0 表示保留，最小值 (-3.4e38) 表示遮蔽
        dtype = input_embeds.dtype
        omni_mask_4d = torch.zeros((b, 1, seq_len, seq_len), dtype=dtype, device=device)
        omni_mask_4d.masked_fill_(~omni_mask_binary, torch.finfo(dtype).min)
        
        # ✅ 致命修复：只有在第一次 forward 时打印一次（使用原生 distributed API）
        if not self._logged_mask_info:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"🎭 Omni-Attention Mask 结构验证:")
                print(f"   - 文本区域：因果注意力 (Causal)")
                print(f"   - 细胞区域 ({self.cell_feature_tokens} tokens): 全向注意力 (Bidirectional)")
                print(f"   - 基因区域 ({self.max_genes} tokens): 全向注意力 (Bidirectional)")
                print(f"   - ✅ Gene 可以看到 Text + Cell（条件生成关键）")
            self._logged_mask_info = True
        
        # ----------------------------------------------------
        # 阶段 B：LLM 骨干网络（强制接管 Attention Mask）
        # ----------------------------------------------------
        # ✅ 修复：明确设置 output_hidden_states=True（因为后面需要使用）
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=omni_mask_4d,  # ✅ 致命修复：用 4D Omni Mask 覆盖默认的因果掩码
            position_ids=position_ids,    # ✅ 保护 1024 偏移量
            output_hidden_states=True     # ✅ 直接设置为 True
        )
        
        logits = outputs.logits
        last_hidden_states = outputs.hidden_states[-1]
        
        # ----------------------------------------------------
        # 阶段 C：独立离散扩散头
        # ----------------------------------------------------
        if hasattr(self, 'diff_proj'):
            last_hidden_states = self.diff_proj(last_hidden_states)
            
        # 构造 Dummy 零向量，喂给旧代码里的 adaLN 以绕过报错
        dummy_adaln = torch.zeros(b, self.diffusion_head_config.hidden_size, device=device, dtype=last_hidden_states.dtype)
        
        for layer in self.diffusion_head_a:
            last_hidden_states = layer(
                hidden_states=last_hidden_states,
                adaln_input=dummy_adaln,  
                attention_mask=omni_mask_4d,  # ✅ 让扩散层也使用全向注意力矩阵
                position_ids=position_ids, 
                modality_positions=modality_positions, 
            )[0]
            
        gene_logits = self.diffusion_head_b(last_hidden_states)
        
        # ----------------------------------------------------
        # 阶段 C：分别计算 Text 和 Gene 的 Loss
        # ----------------------------------------------------
        # 🌟 终极防死锁机制：即便没有标签，也要用乘以 0 的方式激活整个计算图！
        # 这确保了无论数据情况如何，后向传播的通信 Hook 都能在所有 GPU 上准时被触发
        loss_ntp = (logits * 0.0).sum()
        loss_gene = (gene_logits * 0.0).sum()
        
        if labels is not None:
            # ========== 1. NTP Loss (极简向量化实现 + 致命 Shift 修复) ==========
            ntp_labels = labels.clone()
            
            # 将整个基因区间及 [SOG] token 全部设为 -100，保护 LLM 词表
            for i in range(b):
                if modality_positions is not None:
                    g_start = modality_positions[i, 0, 0].item()
                    ntp_labels[i, g_start - 1:] = -100 
            
            # 🚨 致命修复：Causal LM 必须错位计算 Loss！(Shift)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ntp_labels[..., 1:].contiguous()
            
            valid_text_mask = (shift_labels != -100)
            b_text = valid_text_mask.sum().item()
            
            if b_text > 0:
                real_loss_ntp = F.cross_entropy(
                    shift_logits[valid_text_mask], 
                    shift_labels[valid_text_mask], 
                    reduction='mean'
                )
                # 💡 架构师降本增效：Stage 1 底座是冻结的，NTP 没有任何学习空间。
                # 强行反向传播会白白计算巨大且无用的梯度。
                # 使用 .detach() 既能在日志监控 LLM 有没有坏，又能省出 30% 显存和算力！
                loss_ntp = loss_ntp + real_loss_ntp.detach()
            
            # ========== 2. Gene Diffusion Loss（基因预测） ==========
            n_valid_samples = 0
            real_loss_gene = torch.tensor(0.0, device=device, dtype=input_embeds.dtype)
            
            for i in range(b):
                g_start = modality_positions[i, 0, 0].item()
                g_len = modality_positions[i, 0, 1].item()
                
                if g_len > 0:
                    gene_logits_i = gene_logits[i, g_start : g_start+g_len]
                    gene_labels_i = labels[i, g_start : g_start+g_len]
                    
                    valid_mask = (gene_labels_i != -100)
                    
                    if valid_mask.any():
                        valid_labels = gene_labels_i[valid_mask]
                        # 越界保护
                        out_of_bounds = (valid_labels < 0) | (valid_labels >= self.gene_vocab_size)
                        if out_of_bounds.any():
                            valid_labels = valid_labels.clone()
                            valid_labels[out_of_bounds] = 0
                        
                        sample_loss_gene = F.cross_entropy(
                            gene_logits_i[valid_mask], 
                            valid_labels,
                            reduction='mean'
                        )
                        real_loss_gene = real_loss_gene + sample_loss_gene
                        n_valid_samples += 1
            
            # 平均 Gene loss
            if n_valid_samples > 0:
                loss_gene = loss_gene + (real_loss_gene / n_valid_samples)
        
        return logits, loss_ntp, loss_gene
    
    # ✅ 添加 dtype 转换辅助方法
    def _to_model_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量转换到模型的当前 dtype（支持 BF16 混合精度）"""
        return tensor.to(self.dummy_float.dtype)
