# coding=utf-8
"""
Gene Transformer Model for Stage2 
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
            
            # 第二层：升维到高容量中间空间 (4 倍)
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
        
        # ✅ 关键修复：让 self.model 指向 self.showo，适配 forward 中的调用
        self.model = self.showo
    
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
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cell_features: Optional[torch.FloatTensor] = None,
        cell_positions: Optional[torch.LongTensor] = None,
        modality_positions: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        data_type: Optional[List[str]] = None, # 🔴 新增：用于识别 batch 里的任务类型
        **kwargs
    ):
        # ----------------------------------------------------
        # 阶段 A：特征投影与 Embeddings 融合 (修复 embed_tokens 访问)
        # ----------------------------------------------------
        # ✅ 关键修复：Qwen2ForCausalLM 的结构是 self.showo.model.embed_tokens
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        if cell_features is not None and cell_positions is not None:
            # 投影 768d 细胞特征到 LLM 空间
            # ✅ 将 [B, 28672] 重新展开为 [B, 8, 3584]，以匹配 LLM 的维度要求
            cell_proj = self.cell_embedder(cell_features).view(
                cell_features.shape[0], 
                self.cell_feature_tokens,  # ✅ FIXED!
                -1
            )
            for i in range(input_ids.shape[0]):
                c_start = cell_positions[i, 0].item()
                c_len = cell_positions[i, 1].item()
                if c_len > 0:
                    inputs_embeds[i, c_start : c_start + c_len] = cell_proj[i]

        # ----------------------------------------------------
        # 阶段 B：SFT 适配版掩码生成 (关键修改)
        # ----------------------------------------------------
        # ✅ 修复：正确调用 omni_attn_mask_vectorized
        # 需要从 modality_positions 中提取 modalities 信息
        B = input_ids.shape[0]
        LEN = input_ids.shape[1]
        
        # 将 modality_positions 转换为函数需要的格式
        # modality_positions shape: [B, 1, 2] -> [[(start, len)], ...]
        modalities = []
        for i in range(B):
            # 提取第 i 个样本的模态信息
            start = modality_positions[i, 0, 0].item()
            length = modality_positions[i, 0, 1].item()
            modalities.append([(start, length)])
        
        # 使用向量化版本生成掩码
        mask_4d = omni_attn_mask_vectorized(
            B=B, 
            LEN=LEN, 
            modalities=modalities, 
            device=input_ids.device,
            inverted=True  # SDPA 需要 inverted mask
        )
        
        # 转换掩码 dtype 以匹配 Qwen2
        mask_4d = self._to_model_dtype(mask_4d)

        # ----------------------------------------------------
        # 阶段 C：大模型骨干前向传播
        # ----------------------------------------------------
        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=mask_4d, # 传入我们自定义的全向/因果混合掩码
            position_ids=position_ids,
            **kwargs
        )
        hidden_states = outputs[0]
        # ✅ 关键修复：lm_head 也在 model 里面
        logits = self.model.lm_head(hidden_states)

        # ----------------------------------------------------  
        # 阶段 D：双轨 Loss 计算 (样本级精准路由)
        # ----------------------------------------------------
        loss_ntp = torch.tensor(0.0, device=logits.device)
        loss_gene = torch.tensor(0.0, device=logits.device)

        if labels is not None:
            # 🚀 核心修复：创建一个专属于 NTP 的纯净 labels 张量
            # 必须把基因区域强制刷成 -100，防止 LLM 去预测基因 ID
            ntp_labels = labels.clone()
            for i in range(input_ids.shape[0]):
                g_start = modality_positions[i, 0, 0].item()
                g_len = modality_positions[i, 0, 1].item()
                if g_len > 0:
                    ntp_labels[i, g_start : g_start + g_len] = -100

            # 1. 计算 NTP Loss (使用净化后的 ntp_labels)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ntp_labels[..., 1:].contiguous() # 👈 改用 ntp_labels
            
            n_vocab = self.showo.config.vocab_size
            
            # [B, L-1]
            raw_ntp_loss = F.cross_entropy(
                shift_logits.view(-1, n_vocab), 
                shift_labels.view(-1), 
                ignore_index=-100, 
                reduction='none'
            ).view(shift_labels.shape) 

            # 🚀 样本级路由逻辑保持不变
            if data_type is not None:
                for i in range(input_ids.shape[0]):
                    sample_ntp = raw_ntp_loss[i]
                    valid_mask = shift_labels[i] != -100 # 此时 valid_mask 里绝对没有基因了！
                    
                    if valid_mask.any():
                        avg_sample_ntp = sample_ntp[valid_mask].mean()
                        if data_type[i] == 'stage1':
                            loss_ntp = loss_ntp + avg_sample_ntp.detach()
                        else:
                            loss_ntp = loss_ntp + avg_sample_ntp

                loss_ntp = loss_ntp / input_ids.shape[0]


                # 2. 基因 Loss 计算 (逻辑维持你的现状，非常稳健)
                # ✅ 修复：使用 self.diffusion_head_b
                gene_logits = self.diffusion_head_b(hidden_states)
                real_loss_gene = 0.0
                n_valid_gene = 0
                
                for i in range(input_ids.shape[0]):
                    g_start = modality_positions[i, 0, 0].item()
                    g_len = modality_positions[i, 0, 1].item()
                    
                    if g_len > 0:
                        gene_logits_i = gene_logits[i, g_start : g_start + g_len]
                        gene_labels_i = labels[i, g_start : g_start + g_len]
                        
                        # 仅对被 Mask 的基因（即 labels 里的真实 ID）算 Loss
                        valid_mask = gene_labels_i != -100
                        if valid_mask.any():
                            # 基因 Loss 在两阶段都贡献梯度，维持 Embedder 上限
                            real_loss_gene += F.cross_entropy(
                                gene_logits_i[valid_mask], 
                                gene_labels_i[valid_mask]
                            )
                            n_valid_gene += 1
                
                if n_valid_gene > 0:
                    loss_gene = real_loss_gene / n_valid_gene

        return logits, loss_ntp, loss_gene

    
    # ✅ 添加 dtype 转换辅助方法
    def _to_model_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量转换到模型的当前 dtype（支持 BF16 混合精度）"""
        return tensor.to(self.dummy_float.dtype)
