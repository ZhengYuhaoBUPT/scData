# coding=utf-8
"""
Cell-Only SFT Dataset for Multimodal Single-Cell Conversation Ablation.
100% 复用了原先跑通的 gene_sft_dataset_no_metadata_prompt.py 架构：
- 内存驻留 (Memory-resident) 加速 __getitem__，杜绝 HDF5 读盘死锁。
- 恢复原版的多卡数据手动切片与长度补齐 (Padding)，杜绝 NCCL 假死。
- 完全剥离 Gene Tokens，仅保留 Cell Features，适配 CW 消融实验。
"""

import collections
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import anndata as ad
import h5py
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src_ablation_cw.eval.common_eval_utils import read_cell_features


class CWSFTCellOnlyDataset(Dataset):
    def __init__(self,
                 feature_dir: str,
                 json_paths: List[str],
                 text_tokenizer: Any,
                 config_dict: Dict[str, Any],
                 special_tokens_ids: Dict[str, int],
                 accelerator=None,
                 curriculum_stage: str = None,
                 data_type_tag: str = "stage2",
                 append_image_tag: bool = True):
        super().__init__()
        
        self.dataset_config = config_dict.get('dataset', {})
        self.max_seq_len = int(self.dataset_config.get("max_seq_len", 1024))
        self.cell_feature_tokens = int(self.dataset_config.get("cell_feature_tokens", 8))
        self.cell_feature_dim = int(self.dataset_config.get("cell_feature_dim", 768))
        
        self.data_type_tag = data_type_tag
        self.append_image_tag = append_image_tag
        self.curriculum_stage = curriculum_stage
        
        self.accelerator = accelerator
        self.num_replicas = accelerator.num_processes if accelerator else 1
        self.rank = accelerator.process_index if accelerator else 0
        
        self.text_tokenizer = text_tokenizer
        self.soc_id = special_tokens_ids.get('soc_id')
        self.eoc_id = special_tokens_ids.get('eoc_id')
        
        self.bos_id = getattr(text_tokenizer, 'bos_token_id', None)
        # Qwen2 默认 Pad
        self.pad_id = text_tokenizer.pad_token_id if getattr(text_tokenizer, 'pad_token_id', None) is not None else 151643
        
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        # ========== 1. 加载对话 JSON ==========
        self.qa_map = {}
        for jp in json_paths:
            if is_main: print(f"📥 加载对话文件: {jp}")
            with open(jp, 'r') as f:
                data = json.load(f)
                for item in data:
                    cell_id = str(item['id'])
                    if cell_id not in self.qa_map:
                        self.qa_map[cell_id] = []
                    self.qa_map[cell_id].append(item['conversations'])
        if is_main: print(f"✅ Unique cell ids in QA map: {len(self.qa_map):,}")
        
        # ========== 2. 扫描 H5AD 文件获取全局尺寸 ==========
        h5ad_files = sorted(Path(feature_dir).glob("*.h5ad"))
        total_cells_all_files = 0
        file_cell_ranges = []
        
        for h5_path in h5ad_files:
            try:
                with h5py.File(h5_path, 'r') as f:
                    if "X" in f:
                        n_cells = f['X'].shape[0]
                    elif "obs/_index" in f:
                        n_cells = len(f["obs/_index"])
                    else:
                        continue
                start_idx = total_cells_all_files
                total_cells_all_files += n_cells
                file_cell_ranges.append((start_idx, start_idx + n_cells, h5_path))
            except Exception as e:
                pass
                
        usable_samples = (total_cells_all_files // self.num_replicas) * self.num_replicas
        cells_per_rank = usable_samples // self.num_replicas if self.num_replicas > 0 else total_cells_all_files
        my_start = self.rank * cells_per_rank
        my_end = my_start + cells_per_rank
        
        # ========== 3. 将本进程数据加载到 CPU 内存 ==========
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
            
            adata_temp = ad.read_h5ad(h5_path, backed='r')
            obs_df = adata_temp.obs.iloc[local_start:local_end].copy()
            
            # CW 特征对齐: 统一按 X_scFM > X_pca > X 读取
            X_array = read_cell_features(adata_temp, local_start, local_end)
            adata_temp.file.close()
            
            self.data_blocks.append({
                'X': X_array,
                'cell_ids': obs_df['cell_id'].tolist() if 'cell_id' in obs_df else obs_df.index.tolist()
            })
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + n_cells_to_load)
            
        self.total_cells = self.cumulative_sizes[-1]
        
        # ========== 4. 建立有效索引并防止 NCCL 死锁 ==========
        self.valid_indices = []
        self._build_valid_indices(is_main)

    def _build_valid_indices(self, is_main: bool) -> None:
        qa_keys = set(self.qa_map.keys())
        all_valid_indices = []

        for block_idx, block in enumerate(self.data_blocks):
            for local_idx, cell_id in enumerate(block['cell_ids']):
                cell_id_str = str(cell_id)
                if cell_id_str in qa_keys:
                    conv_count = len(self.qa_map.get(cell_id_str, []))
                    conv_count = max(1, conv_count)
                    for conv_idx in range(conv_count):
                        all_valid_indices.append((block_idx, local_idx, conv_idx))

        local_count = len(all_valid_indices)

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            count_tensor = torch.tensor(local_count, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.long)
            all_counts = [torch.zeros_like(count_tensor) for _ in range(world_size)]
            dist.all_gather(all_counts, count_tensor)
            counts_list = [int(c.item()) for c in all_counts]

            max_count = max(counts_list) if counts_list else local_count
            if local_count < max_count and local_count > 0:
                rnd = random.Random(42 + rank)
                pad_indices = [all_valid_indices[rnd.randrange(local_count)] for _ in range(max_count - local_count)]
                all_valid_indices.extend(pad_indices)
                
            self.valid_indices = all_valid_indices
            if is_main:
                print(f"📊 各卡分布: {counts_list} -> Padding 至 {max_count}")
        else:
            self.valid_indices = all_valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        valid_item = self.valid_indices[idx]
        block_idx, local_idx, conv_choice = valid_item
        block = self.data_blocks[block_idx]
        
        raw_feature = block['X'][local_idx]
        cell_id = block['cell_ids'][local_idx]
        
        # 🚨 严格保持 1 维向量，彻底删除 expand(8)
        cell_feature = torch.from_numpy(raw_feature)
        if cell_feature.dim() == 1:
            if cell_feature.shape[0] < self.cell_feature_dim:
                pad_size = self.cell_feature_dim - cell_feature.shape[0]
                cell_feature = torch.nn.functional.pad(cell_feature, (0, pad_size))
            elif cell_feature.shape[0] > self.cell_feature_dim:
                cell_feature = cell_feature[:self.cell_feature_dim]
        elif cell_feature.dim() == 2:
            token_count, feat_dim = cell_feature.shape
            if token_count < self.cell_feature_tokens:
                pad_tokens = self.cell_feature_tokens - token_count
                cell_feature = torch.nn.functional.pad(cell_feature, (0, 0, 0, pad_tokens))
            elif token_count > self.cell_feature_tokens:
                cell_feature = cell_feature[:self.cell_feature_tokens]
            if feat_dim < self.cell_feature_dim:
                pad_dim = self.cell_feature_dim - feat_dim
                cell_feature = torch.nn.functional.pad(cell_feature, (0, pad_dim))
            elif feat_dim > self.cell_feature_dim:
                cell_feature = cell_feature[:, :self.cell_feature_dim]
        else:
            raise ValueError(f"Unsupported cell feature ndim={cell_feature.dim()} for cell_id={cell_id}")

        # ==========================================
        # 2. 对话拼接 (原汁原味 ChatML 模板解析)
        # ==========================================
        cell_id_str = str(cell_id)
        conv_bank = self.qa_map.get(cell_id_str, [])
        conversations = conv_bank[conv_choice % len(conv_bank)] if conv_bank else []
            
        input_ids = []
        text_labels = []
        
        if self.bos_id is not None:
            input_ids.append(self.bos_id)
            text_labels.append(-100)
            
        # 🚨 系统提示词：必须加 allowed_special="all"
        system_prompt = "You are a helpful assistant."
        sys_str = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        sys_tokens = self.text_tokenizer.encode(sys_str, add_special_tokens=False)
        input_ids.extend(sys_tokens)
        text_labels.extend([-100] * len(sys_tokens))
        
        cell_tokens = [self.soc_id] + [self.pad_id] * self.cell_feature_tokens + [self.eoc_id]
        cell_start_pos = 0
        is_first_user = True
        
        for conv in conversations:
            role = conv['from']
            # 🚨 数据洗澡：不管 JSON 多脏，在这里彻底消灭 LLaMA 残留毒素！
            val = conv.get('value', "")
            val = val.replace("[INST]", "").replace("[/INST]", "")
            val = val.replace("<<SYS>>", "").replace("<</SYS>>", "")
            val = val.replace("<s>", "").replace("</s>", "").strip()
            
            # 兼容 key
            if role == 'human':
                role = 'user'
            elif role == 'gpt':
                role = 'assistant'
                
            if role == 'user':
                if is_first_user:
                    # 第一轮用户对话：注入图像占位符和细胞特征
                    val = val.replace("<image>", "").strip()
                    if self.append_image_tag and val:
                        val = "\n" + val

                    # 提取纯 header，加 allowed_special="all" 防切碎
                    u_header = self.text_tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
                    input_ids.extend(u_header)
                    text_labels.extend([-100] * len(u_header))
                    
                    # 注入细胞 Tokens
                    soc_pos = len(input_ids)
                    input_ids.extend(cell_tokens)
                    text_labels.extend([-100] * len(cell_tokens))
                    cell_start_pos = soc_pos + 1
                    
                    # 提问正文
                    u_body = self.text_tokenizer.encode(f"{val}<|im_end|>\n", add_special_tokens=False)
                    input_ids.extend(u_body)
                    text_labels.extend([-100] * len(u_body))
                    
                    is_first_user = False
                else:
                    # 后续多轮用户对话
                    u_str = f"<|im_start|>user\n{val}<|im_end|>\n"
                    u_ids = self.text_tokenizer.encode(u_str, add_special_tokens=False)
                    input_ids.extend(u_ids)
                    text_labels.extend([-100] * len(u_ids))
                    
            elif role == 'assistant':
                # 🚨 SFT 掩码黄金法则：隔离 header，只算正文的 Loss！
                a_header = self.text_tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
                a_body = self.text_tokenizer.encode(f"{val}<|im_end|>\n", add_special_tokens=False)
                
                # Header 不算 loss (-100)
                input_ids.extend(a_header)
                text_labels.extend([-100] * len(a_header))
                
                # Body 是我们的预测目标，拷贝到 text_labels
                input_ids.extend(a_body)
                text_labels.extend(a_body)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            text_labels = text_labels[:self.max_seq_len]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(text_labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor, dtype=torch.bool)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask,
            "cell_features": cell_feature,
            "cell_positions": torch.tensor([cell_start_pos, self.cell_feature_tokens], dtype=torch.long),
            "data_type": self.data_type_tag,
        }

def cw_cell_only_collate(batch):
    batched = collections.defaultdict(list)
    for data in batch:
        for k, v in data.items():
            batched[k].append(v)

    out = {}
    for k, v in batched.items():
        if k == 'data_type':
            out[k] = v
            continue
        if isinstance(v[0], torch.Tensor):
            if v[0].dim() == 1:
                if k in ('cell_features', 'cell_positions'):
                    # 栈叠后，[768] 自动变成 [B, 768]
                    out[k] = torch.stack(v, dim=0)
                elif k == 'labels':
                    out[k] = pad_sequence(v, batch_first=True, padding_value=-100)
                elif k == 'input_ids':
                    out[k] = pad_sequence(v, batch_first=True, padding_value=151643)
                else:
                    out[k] = pad_sequence(v, batch_first=True, padding_value=0)
            else:
                out[k] = torch.stack(v, dim=0)
    return out