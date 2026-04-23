# coding=utf-8
"""
Mixed Dataloader for Show-o2 Stage 2 Training (Concatenation + Subset Version)
架构特点：
1. 绝对纯净：底层 Dataset 无需知道白名单的存在，彻底解耦。
2. 调度层拦截：在 DataLoader 层读取 Whitelist，使用 torch.utils.data.Subset 进行零拷贝过滤。
3. 物理合并：Subset 过滤后的 Stage 1 数据与 SFT 数据通过 ConcatDataset 直接拼接打乱。
"""

import os
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

root = '/root/wanghaoran/zxy/project/sc_showo'
if root not in sys.path:
    sys.path.insert(0,root)
    
# 请根据实际情况调整 import 路径
from src.datasets.gene_dataset import GeneDataset
from src.datasets.gene_sft_dataset import SFTDataset

def create_mixed_dataloader(
    config_dict: dict,
    special_tokens_ids: dict,
    text_tokenizer,
    accelerator=None,
    **kwargs
) -> DataLoader:
    data_config = config_dict.get('data', {})
    training_config = config_dict.get('training', {})
    dataset_config = config_dict.get('dataset', {})
    
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    
    if is_main:
        print("\n" + "="*80)
        print("🚀 [MixedLoader] 正在构建混合训练数据流 (洁净架构版)...")
        print("="*80)

    # ==========================================
    # 1. 初始化纯净的 SFT 数据集 (Stage 2)
    # ==========================================
    if is_main: print("\n🟡 正在初始化 SFTDataset (Stage 2 多模态对话数据)...")
    sft_dataset = SFTDataset(
        feature_dir=data_config.get('sft_feature_dir'),
        json_paths=data_config.get('sft_json_paths'),
        scgpt_gene_vocab=data_config.get('scgpt_gene_vocab'),
        text_tokenizer=text_tokenizer,
        config_dict=config_dict,
        special_tokens_ids=special_tokens_ids,
        max_seq_len=dataset_config.get('max_seq_len', 2048),
        accelerator=accelerator
    )

    # ==========================================
    # 2. 初始化纯净的 Gene 数据集 (Stage 1)
    # ==========================================
    if is_main: print("\n🟢 正在初始化 GeneDataset (Stage 1 单模态对齐数据)...")
    stage1_dataset = GeneDataset(
        feature_dir=data_config.get('feature_dir'),
        lmdb_base_dir=data_config.get('lmdb_base_dir'),
        scgpt_gene_vocab=data_config.get('scgpt_gene_vocab'),
        text_tokenizer=text_tokenizer,
        config_dict=config_dict,
        special_tokens_ids=special_tokens_ids,
        max_seq_len=dataset_config.get('max_seq_len', 2048),
        load_metadata=data_config.get('load_metadata', True),
        accelerator=accelerator
    )

    # ==========================================
    # 3. 在调度层进行白名单过滤 (Zero-copy Subset)
    # ==========================================
    whitelist_path = data_config.get('stage1_whitelist_json')
    if whitelist_path and os.path.exists(whitelist_path):
        if is_main: print(f"\n🛡️ [MixedLoader] 正在应用白名单过滤: {whitelist_path}")
        with open(whitelist_path, 'r') as f:
            whitelist = json.load(f)
            
        valid_indices = []
        # 遍历 stage1_dataset 的块，寻找符合白名单的绝对坐标
        for block_idx, block in enumerate(stage1_dataset.data_blocks):
            db_name = Path(stage1_dataset.lmdb_paths[block_idx]).stem
            allowed_keys = set(whitelist.get(db_name, []))
            
            if not allowed_keys:
                continue
                
            start_offset = stage1_dataset.cumulative_sizes[block_idx]
            for local_idx, key in enumerate(block['lmdb_keys']):
                if str(key) in allowed_keys:
                    valid_indices.append(start_offset + local_idx)
                    
        # 使用 PyTorch 原生的 Subset 生成过滤后的数据集
        filtered_stage1 = Subset(stage1_dataset, valid_indices)
        if is_main: 
            print(f"   - 过滤前单卡 Stage 1 样本数: {len(stage1_dataset):,}")
            print(f"   - 过滤后单卡 Stage 1 样本数: {len(filtered_stage1):,}")
    else:
        if is_main: print("\n⚠️ [MixedLoader] 未找到白名单，将混合全量 Stage 1 数据。")
        filtered_stage1 = stage1_dataset

    # ==========================================
    # 4. 物理合并 (ConcatDataset)
    # ==========================================
    combined_dataset = ConcatDataset([sft_dataset, filtered_stage1])
    
    if is_main:
        print("\n" + "="*80)
        print("🎯 数据混合完毕！")
        print(f"   - 单卡 SFT 数据量: {len(sft_dataset):,} 个")
        print(f"   - 单卡 Stage1 抽样量: {len(filtered_stage1):,} 个")
        print(f"   - 单卡 总混合容量: {len(combined_dataset):,} 个")
        print("="*80 + "\n")

    # ==========================================
    # 5. 构建原生 DataLoader
    # ==========================================
    batch_size = training_config.get('batch_size_per_gpu', training_config.get('batch_size', 16))
    num_workers = training_config.get('num_workers', 4)

    mixed_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=num_workers,
        collate_fn=SFTDataset.collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    return mixed_loader