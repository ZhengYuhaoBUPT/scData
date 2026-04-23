# coding=utf-8
"""
SFT Dataset for Multimodal Single-Cell Conversation
专为 SFT 阶段设计，支持多轮对话、ChatML 状态机掩码与离散基因扩散。
兼容 Show-o2 MixedDataLoader。
"""

import collections
import json
import os
import time
import math
import random
import bisect
import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import anndata as ad
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class SFTDataset(Dataset):
    """
    SFT 多模态对齐数据集 - Map-style 全内存极速版
    """
    
    def __init__(self,
                 feature_dir: str,
                 json_paths: List[str],  
                 scgpt_gene_vocab: str,
                 text_tokenizer: Any,
                 config_dict: Dict[str, Any],
                 special_tokens_ids: Dict[str, int],
                 max_seq_len: int = 2048,
                 accelerator=None):
        super().__init__()
        
        self.dataset_config = config_dict.get('dataset', {})
        self.max_genes = self.dataset_config.get('max_genes', 1200)
        self.offset = self.dataset_config.get('offset', 1024)
        self.cell_feature_tokens = self.dataset_config.get('cell_feature_tokens', 8)
        # 确保使用 2048 或从 config 中读取的最大长度
        self.max_seq_len = self.dataset_config.get('max_seq_len', max_seq_len)
        
        # 扩散采样配置
        self.mask_min_ratio = self.dataset_config.get('mask_min_ratio', 1e-5)
        self.mask_max_ratio = self.dataset_config.get('mask_max_ratio', 0.9999)
        
        # 分布式环境
        self.accelerator = accelerator
        self.num_replicas = accelerator.num_processes if accelerator else 1
        self.rank = accelerator.process_index if accelerator else 0
        
        # Tokenizer & 特殊 Tokens
        self.text_tokenizer = text_tokenizer
        self.soc_id = special_tokens_ids.get('soc_id')
        self.eoc_id = special_tokens_ids.get('eoc_id')
        self.sog_id = special_tokens_ids.get('sog_id')
        self.eog_id = special_tokens_ids.get('eog_id')
        self.mask_gene_id = special_tokens_ids.get('mask_gene_id')
        
        self.bos_id = getattr(text_tokenizer, 'bos_token_id', None)
        self.pad_id = text_tokenizer.pad_token_id if getattr(text_tokenizer, 'pad_token_id', None) is not None else 151643
        
        # ========== 1. 加载并建立 JSON O(1) 索引 ==========
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main: print("\n" + "="*60 + "\n🧠 初始化 SFTDataset...\n" + "="*60)
        
        self.qa_map = {}
        for jp in json_paths:
            if is_main: print(f"📥 加载对话文件: {jp}")
            with open(jp, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.qa_map[item['id']] = item['conversations']
                    
        if is_main: print(f"✅ 共加载 {len(self.qa_map):,} 条对话记录")
        
        # ========== 2. 扫描并分配 H5AD 文件 (纯 h5py 极速版) ==========
        h5ad_files = sorted(Path(feature_dir).glob("*.h5ad"))
        total_cells_all_files = 0
        file_cell_ranges = []
        
        for h5_path in h5ad_files:
            with h5py.File(h5_path, 'r') as f:
                n_cells = f['X'].shape[0]
            start_idx = total_cells_all_files
            total_cells_all_files += n_cells
            file_cell_ranges.append((start_idx, start_idx + n_cells, h5_path))
            
        usable_samples = (total_cells_all_files // self.num_replicas) * self.num_replicas
        cells_per_rank = usable_samples // self.num_replicas
        my_start = self.rank * cells_per_rank
        my_end = my_start + cells_per_rank
        
        # ========== 3. 加载本进程对应的 Chunk ==========
        self.data_blocks = []
        self.cumulative_sizes = [0]
        
        for range_start, range_end, h5_path in file_cell_ranges:
            overlap_start = max(range_start, my_start)
            overlap_end = min(range_end, my_end)
            if overlap_start >= overlap_end:
                continue
                
            local_start = overlap_start - range_start
            local_end = overlap_end - range_start
            n_cells_to_load = overlap_end - overlap_start
            
            # 使用 ad.read_h5ad(backed='r') 极速提取 obs 元数据
            adata_temp = ad.read_h5ad(h5_path, backed='r')
            obs_df = adata_temp.obs.iloc[local_start:local_end].copy()
            adata_temp.file.close()
            
            with h5py.File(h5_path, 'r') as h5_file:
                X_array = h5_file['X'][local_start:local_end].astype(np.float32)
                rank_array = h5_file['obsm']['rank'][local_start:local_end].astype(np.int32)
                log1p_array = h5_file['obsm']['rank_log1p'][local_start:local_end].astype(np.float32)
            
            self.data_blocks.append({
                'X': X_array,
                'rank': rank_array,
                'log1p': log1p_array,
                'obs': obs_df.to_dict('list'),
                'cell_ids': obs_df['cell_id'].tolist() if 'cell_id' in obs_df else obs_df.index.tolist()
            })
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + n_cells_to_load)
            
        self.total_cells = self.cumulative_sizes[-1]
        
        if is_main:
            print(f"✅ 分布式加载完成: 每卡 {self.total_cells:,} 个细胞")
            
        # ========== 4. 加载 scGPT 词汇表 ==========
        with open(scgpt_gene_vocab, 'r') as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)

    def __len__(self) -> int:
        return self.total_cells

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # ==========================================
        # 1. 寻址与数据提取
        # ==========================================
        block_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[block_idx]
        block = self.data_blocks[block_idx]
        
        cell_features = block['X'][local_idx]
        rank_seq = block['rank'][local_idx]
        log1p_values = block['log1p'][local_idx]
        cell_id = block['cell_ids'][local_idx]
        
        # ==========================================
        # 2. 动态 Caption 生成 (彻底与 Config 解耦，安全提取)
        # ==========================================
        if "SRX" in str(cell_id):
            # ArchS4: 静默模式，只给通用 Prompt
            system_prompt = "You are an AI assistant analyzing RNA-seq data and its corresponding gene sequence."
        else:
            # Census: 智能从当前 h5ad 的 obs 中探查可用的 metadata 字段
            obs_dict = block['obs']
            
            # 内部安全获取函数，绝对不会抛出 IndexError 或 KeyError
            def safe_get(keys, index, default='unknown'):
                for k in keys:
                    if k in obs_dict and index < len(obs_dict[k]):
                        val = obs_dict[k][index]
                        if val is not None and str(val).lower() != 'nan' and str(val).strip() != '':
                            return str(val)
                return default

            tissue = safe_get(['tissue_definition', 'tissue', 'tissue_name'], local_idx, 'unknown tissue')
            disease = safe_get(['disease_definition', 'disease', 'disease_name'], local_idx, 'healthy condition')
            sex = safe_get(['sex_name', 'sex'], local_idx, 'unknown sex')
            dev_stage = safe_get(['development_stage', 'stage_name', 'stage'], local_idx, 'unknown')
            
            system_prompt = f"This is a single-cell sample from the {tissue} of a {disease} human {sex} at {dev_stage} developmental stage."
            
        # ==========================================
        # 3. 核心：ChatML 多轮状态机序列拼接
        # ==========================================
        conversations = self.qa_map.get(cell_id, [])
        if not conversations:
            # Fallback for missing QA
            conversations = [{"from": "human", "value": "Describe this cell (with its gene sequence)."}, 
                             {"from": "gpt", "value": "It is a biological sample."}]
            
        input_ids = []
        text_labels = []
        
        # BOS
        if self.bos_id is not None:
            input_ids.append(self.bos_id)
            text_labels.append(-100)
            
        # System Block
        sys_tokens = self.text_tokenizer.encode(f"<|im_start|>system\n{system_prompt}<|im_end|>\n", add_special_tokens=False)
        input_ids.extend(sys_tokens)
        text_labels.extend([-100] * len(sys_tokens))
        
        # 准备特征 Tokens
        cell_tokens = [self.soc_id] + [self.pad_id] * self.cell_feature_tokens + [self.eoc_id]
        scgpt_token_ids = rank_seq.tolist()
        gene_tokens = [self.sog_id] + scgpt_token_ids + [self.eog_id]
        
        cell_pos_start = 0
        mod_pos_start = 0
        
        is_first_user = True
        
        for conv in conversations:
            if conv['from'] == 'human':
                # 语义锚点替换
                text = conv['value'].replace('<image>', 'given this cell (with its gene sequence)')
                u_tokens = self.text_tokenizer.encode(f"<|im_start|>user\n{text}<|im_end|>\n", add_special_tokens=False)
                input_ids.extend(u_tokens)
                text_labels.extend([-100] * len(u_tokens))
                
                # 📍 在第一轮 User 提问结束时，注入双向多模态区块
                if is_first_user:
                    cell_pos_start = len(input_ids) + 1  # 记录 SOC 之后的绝对位置
                    input_ids.extend(cell_tokens)
                    text_labels.extend([-100] * len(cell_tokens))
                    
                    mod_pos_start = len(input_ids) + 1   # 记录 SOG 之后的绝对位置
                    input_ids.extend(gene_tokens)
                    text_labels.extend([-100] * len(gene_tokens))
                    
                    is_first_user = False
                    
            elif conv['from'] == 'gpt':
                # 切分 Assistant 的前缀，只计算回答内容的 Loss
                prefix = "<|im_start|>assistant\n"
                content = f"{conv['value']}<|im_end|>\n"
                
                p_tokens = self.text_tokenizer.encode(prefix, add_special_tokens=False)
                c_tokens = self.text_tokenizer.encode(content, add_special_tokens=False)
                
                input_ids.extend(p_tokens + c_tokens)
                # 前缀不计入 Loss，实际内容计入
                text_labels.extend([-100] * len(p_tokens) + c_tokens)

        # 截断保护
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            text_labels = text_labels[:self.max_seq_len]

        # ==========================================
        # 4. 基因区域：Log-Normal 扩散采样掩码
        # ==========================================
        valid_positions = [idx for idx, gid in enumerate(scgpt_token_ids) if gid < self.vocab_size]
        
        # Log-Normal 采样 t
        normal_sample = torch.randn(1).item()
        t_raw = math.exp(0.5 + 1.0 * normal_sample)
        t = 1.0 / (1.0 + math.exp(-t_raw + 0.5))
        
        mask_ratio = t * (self.mask_max_ratio - self.mask_min_ratio) + self.mask_min_ratio
        n_mask = max(1, int(len(valid_positions) * mask_ratio))
        mask_positions = random.sample(valid_positions, n_mask)
        
        # 生成双轨张量
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        text_labels_tensor = torch.tensor(text_labels, dtype=torch.long)
        gene_labels_tensor = torch.full_like(input_ids_tensor, -100)
        
        # 遍历选中的 Mask 位置进行替换和打标
        for pos in mask_positions:
            abs_pos = mod_pos_start + pos  # 绝对坐标
            if abs_pos < len(input_ids_tensor):  # 防截断越界
                original_gene_id = input_ids_tensor[abs_pos].item()
                input_ids_tensor[abs_pos] = self.mask_gene_id
                # 仅对非零基因计算 Loss
                if original_gene_id != 0:
                    gene_labels_tensor[abs_pos] = original_gene_id
                    
        # 完美融合：文本 Loss 和 基因 Loss 在物理位置上绝对互斥
        final_labels_tensor = torch.where(gene_labels_tensor != -100, gene_labels_tensor, text_labels_tensor)
        
        # 生成原始的顺序 Position IDs
        position_ids_tensor = torch.arange(len(input_ids_tensor), dtype=torch.long)
        
        # 应用 offset 逻辑
        if mod_pos_start < len(position_ids_tensor):
            # 🚀 核心修复：让基因区块以及其后的所有问答文本，统统继承 offset
            position_ids_tensor[mod_pos_start:] += self.offset
            
        # ==========================================
        # 5. 组装返回 Dict (100% 适配 MixedDataLoader)
        # ==========================================
        return {
            'input_ids': input_ids_tensor,
            'position_ids': position_ids_tensor,
            'attention_mask': torch.ones_like(input_ids_tensor),
            'labels': final_labels_tensor,
            'cell_features': torch.tensor(cell_features, dtype=torch.bfloat16), 
            'log1p': torch.tensor(log1p_values, dtype=torch.float32),
            'cell_positions': torch.tensor([cell_pos_start, self.cell_feature_tokens], dtype=torch.long),
            'modality_positions': torch.tensor([[mod_pos_start, self.max_genes]], dtype=torch.long),
            'gene_mask': (input_ids_tensor[mod_pos_start:mod_pos_start+self.max_genes] == self.mask_gene_id),
            'non_zero_mask': torch.tensor(rank_seq != 0, dtype=torch.bool),
            'offset_rope': torch.tensor(0, dtype=torch.long),
            'data_type': ['stage2']  # MixedDataLoader 期望 list of strings
        }
        
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """与 GeneDataset 完全一致的 Collate 函数"""
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
                
        for k, v in batched.items():
            if k == 'data_type':
                from itertools import chain
                batched[k] = list(chain.from_iterable(v))
            elif isinstance(v[0], torch.Tensor):
                if v[0].dim() == 1:
                    if k in ('cell_features', 'cell_positions', 'gene_mask', 'non_zero_mask', 'log1p'):
                        batched[k] = torch.stack(v, dim=0)
                    elif k == 'labels':
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=-100)
                    elif k == 'input_ids':
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=151643) # Qwen pad
                    elif k == 'position_ids':
                        # 🔴 显式处理：确保 padding 不破坏 RoPE 偏移量
                        # 使用 0 padding（标准做法），因为 attention_mask 会屏蔽这些位置
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=0)
                    else:
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=0)
                elif v[0].dim() == 0:
                    batched[k] = torch.stack(v, dim=0)
                elif v[0].dim() == 2:
                    if k == 'modality_positions':
                        batched[k] = torch.stack(v, dim=0)
                else:
                    batched[k] = torch.stack(v, dim=0)
        return dict(batched)