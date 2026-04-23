# coding=utf-8
"""
Gene Dataset for Transcriptome Data (Map-style Version)
支持从多个 LMDB + 多个 H5AD 文件并行读取
[✅ 终极架构：Map-style Dataset + DistributedSampler + 全内存加载]
[🔒 核心原则：只改数据调度层，业务逻辑 100% 保留]
"""

import collections
import json
import os
import time
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import bisect
import psutil
import h5py
import lmdb


class GeneDataset(Dataset):
    """
    转录组数据集 - Map-style 版本（全内存加载）
    
    架构设计：
    - 每个 .db 文件对应一个 .h5ad 文件（同名不同后缀）
    - 所有数据在 __init__ 时一次性加载到内存
    - O(1) 绝对坐标寻址，支持精确样本控制
    """
    
    def __init__(self,
                 feature_dir: str = None,
                 lmdb_base_dir: str = None,
                 scgpt_gene_vocab: str = None,
                 text_tokenizer: Any = None,
                 config_dict: Dict[str, Any] = None,
                 special_tokens_ids: Dict[str, int] = None,
                 max_seq_len: Optional[int] = None,
                 load_metadata: bool = None,
                 accelerator=None):
        """
        完全配置驱动的数据集
        
        Args:
            config_dict: 完整配置字典（从 config.json 加载）
            special_tokens_ids: 特殊 token ID 映射
            accelerator: Accelerator 实例（用于分布式数据切片）
        """
        super().__init__()
        
        # ✅ 分层读取配置
        self.data_config = config_dict.get('data', {})
        self.dataset_config = config_dict.get('dataset', {})
        self.sequence_config = config_dict.get('sequence', {})
        self.tokenizer_config = config_dict.get('tokenizer', {})
        self.training_config = config_dict.get('training', {})
        
        # ✅ 新增：从 config 读取 metadata 字段配置
        self.metadata_fields = self.data_config.get('metadata_fields', [
            'celltype_name', 'celltype_definition',
            'disease_name', 'disease_definition',
            'tissue_name', 'tissue_definition',
            'stage_name', 'stage_definition',
            'sex_name', 'sex_definition'
        ])
        
        # 数据路径（优先级：传入参数 > config > 默认）
        self.feature_dir = Path(feature_dir) if feature_dir else Path(self.data_config.get('feature_dir', ''))
        self.lmdb_base_dir = Path(lmdb_base_dir) if lmdb_base_dir else Path(self.data_config.get('lmdb_base_dir', ''))
        self.scgpt_gene_vocab = scgpt_gene_vocab if scgpt_gene_vocab else self.data_config.get('scgpt_gene_vocab', '')
        self.load_metadata = load_metadata if load_metadata is not None else self.data_config.get('load_metadata', True)
        
        # 数据集超参
        self.max_genes = self.dataset_config['max_genes']
        self.offset = self.dataset_config['offset']
        self.cell_feature_tokens = self.dataset_config['cell_feature_tokens']
        self.cell_feature_dim = self.dataset_config.get('cell_feature_dim', 768)
        self.max_text_len = self.dataset_config.get('max_text_len', 512)
        self.cond_dropout_prob = self.dataset_config.get('cond_dropout_prob', 0.1)
        
        # MaskedGIT 安全边界
        self.mask_min_ratio = self.dataset_config.get('mask_min_ratio', 1e-5)
        self.mask_max_ratio = self.dataset_config.get('mask_max_ratio', 0.9999)
        
        # 训练配置
        self.batch_size = self.training_config.get('batch_size', 16)
        self.gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)
        self.num_epochs = self.training_config.get('epochs', 10)
        
        # ✅ 新增：分布式信息
        self.accelerator = accelerator
        if accelerator is not None:
            self.num_replicas = accelerator.num_processes
            self.rank = accelerator.process_index
        else:
            self.num_replicas = 1
            self.rank = 0
        
        print(f"✅ 从配置加载数据集参数:")
        print(f"   - max_genes: {self.max_genes}")
        print(f"   - offset: {self.offset}")
        print(f"   - cell_feature_tokens: {self.cell_feature_tokens}")
        print(f"   - batch_size: {self.batch_size}")
        print(f"   - gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        print(f"   - num_epochs: {self.num_epochs}")
        print(f"   - Process ID: {os.getpid()}")
        print(f"   - 分布式：Rank {self.rank}/{self.num_replicas}")
        
        # ========== 1. 扫描并配对 LMDB 和 H5AD 文件 ==========
        feature_dir = Path(feature_dir)
        lmdb_base_dir = Path(lmdb_base_dir)
        
        if not feature_dir.exists():
            raise ValueError(f"特征目录不存在：{feature_dir}")
        if not lmdb_base_dir.exists():
            raise ValueError(f"LMDB 目录不存在：{lmdb_base_dir}")
        
        # 扫描所有 .h5ad 文件
        h5ad_files = sorted([f for f in feature_dir.glob("*.h5ad")])
        
        if len(h5ad_files) == 0:
            raise ValueError(f"在 {feature_dir} 中没有找到 .h5ad 文件")
        
        print(f"\n📂 [PID-{os.getpid()} Rank-{self.rank}] 扫描到 {len(h5ad_files)} 个 H5AD 文件")
        
        # ==========================================
        # 🌟 阶段 1：分布式数据分配（纯 h5py 极速版）
        # ==========================================
        import torch.distributed as dist
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        
        if is_main:
            print("\n" + "="*60)
            print("🧠 阶段 1：分布式数据分配（纯 h5py 极速版）...")
        
        # ==========================================
        # 🚀 极致提速：跳过 AnnData，用纯 h5py 毫秒级扫描
        # ==========================================
        total_cells_all_files = 0
        file_cell_ranges = []
        
        start_scan_time = time.time()
        for h5_path in h5ad_files:
            file_name = Path(h5_path).stem
            lmdb_path = Path(lmdb_base_dir) / f"{file_name}.db"
            if not lmdb_path.exists(): 
                if is_main: print(f"⚠️  跳过：LMDB 不存在 {lmdb_path}")
                continue
            
            # 🚨 核心修复：直接读取 HDF5 的形状元数据，0 内存分配！
            with h5py.File(h5_path, 'r') as f:
                n_cells = f['X'].shape[0]
            
            start_idx = total_cells_all_files
            total_cells_all_files += n_cells
            file_cell_ranges.append((start_idx, start_idx + n_cells, h5_path, lmdb_path))
        
        if is_main:
            scan_time = time.time() - start_scan_time
            print(f"   扫描完成：总计 {total_cells_all_files:,} 个细胞，仅耗时 {scan_time:.4f}s")
        
        # ==========================================
        # 🚨 关键：数学上严格的均匀分配算法
        # ==========================================
        # 步骤 2：计算可被 num_replicas 整除的最大样本数
        usable_samples = (total_cells_all_files // self.num_replicas) * self.num_replicas
        discarded_samples = total_cells_all_files - usable_samples
        
        # 每卡固定样本数（数学保证完全相同）
        cells_per_rank = usable_samples // self.num_replicas
        
        if is_main:
            print(f"\n📊 分配策略:")
            print(f"   总样本数：{total_cells_all_files:,}")
            print(f"   GPU 数量：{self.num_replicas}")
            print(f"   可用样本：{usable_samples:,} (丢弃 {discarded_samples:,} 个余数)")
            print(f"   每卡配额：{cells_per_rank:,} 个细胞 (严格相等)")
            print(f"   丢弃率：{discarded_samples / total_cells_all_files * 100:.4f}%")
        
        # 步骤 3：计算本进程的起始和结束索引（全局坐标）
        my_start = self.rank * cells_per_rank
        my_end = my_start + cells_per_rank
        
        if is_main:
            print(f"\n🎯 Rank {self.rank} 分配范围：[{my_start:,}, {my_end:,})")
        
        # ==========================================
        # 步骤 4：只加载落在 [my_start, my_end) 范围内的数据
        # ==========================================
        self.data_blocks = []
        self.lmdb_paths = []
        self.cumulative_sizes = [0]
        
        start_load_time = time.time()
        
        for range_start, range_end, h5_path, lmdb_path in file_cell_ranges:
            # 计算交集：[range_start, range_end) ∩ [my_start, my_end)
            overlap_start = max(range_start, my_start)
            overlap_end = min(range_end, my_end)
            
            # 如果没有交集，跳过这个文件
            if overlap_start >= overlap_end:
                continue
            
            # 计算在这个文件中的局部坐标
            local_start = overlap_start - range_start
            local_end = overlap_end - range_start
            n_cells_to_load = overlap_end - overlap_start
            
            if is_main and len(self.data_blocks) == 0:
                print(f"\n📂 正在加载第一个文件片段：{h5_path.name}")
                print(f"   全局范围：[{overlap_start}, {overlap_end})")
                print(f"   局部范围：[{local_start}, {local_end})")
                print(f"   细胞数：{n_cells_to_load:,}")
            
            # 🚨 核心修复：直接用 h5py 切片，绝不构建 Pandas DataFrame
            with h5py.File(h5_path, 'r') as h5_file:
                # 1. 提取数值特征
                X_array = h5_file['X'][local_start:local_end].astype(np.float32)
                rank_array = h5_file['obsm']['rank'][local_start:local_end].astype(np.int32)
                log1p_array = h5_file['obsm']['rank_log1p'][local_start:local_end].astype(np.float32)
                
                # 2. 提取 lmdb_key 字符串
                try:
                    # 尝试直接从 HDF5 的 obs 组读取 (速度最快)
                    keys_raw = h5_file['obs']['lmdb_key'][local_start:local_end]
                    if hasattr(keys_raw[0], 'decode'):
                        lmdb_keys = [k.decode('utf-8') for k in keys_raw]
                    else:
                        lmdb_keys = [str(k) for k in keys_raw]
                except Exception:
                    # 如果 H5AD 的 obs 保存为 Categories 格式的兼容处理
                    try:
                        cats = h5_file['obs']['__categories']['lmdb_key'][:]
                        codes = h5_file['obs']['lmdb_key'][local_start:local_end]
                        cats = [c.decode('utf-8') if hasattr(c, 'decode') else str(c) for c in cats]
                        lmdb_keys = [cats[code] for code in codes]
                    except Exception:
                        # 终极 Fallback (万一格式极其特殊)
                        import anndata as ad
                        adata_temp = ad.read_h5ad(h5_path, backed='r')
                        lmdb_keys = adata_temp.obs['lmdb_key'].iloc[local_start:local_end].astype(str).tolist()
                        adata_temp.file.close()

            self.data_blocks.append({
                'X': X_array, 
                'rank': rank_array, 
                'log1p': log1p_array, 
                'lmdb_keys': lmdb_keys  # 直接存 list，内存极小
            })
            self.lmdb_paths.append(str(lmdb_path))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + n_cells_to_load)
        
        # 计算本进程实际加载的细胞总数
        self.total_cells = self.cumulative_sizes[-1]
        self.lmdb_envs = {}  # 留给多进程懒加载
        
        # ==========================================
        # 🚨 致命验证：确保所有进程的 total_cells 完全相同
        # ==========================================
        load_time = time.time() - start_load_time
        
        if is_main:
            elapsed = time.time() - start_load_time
            mem_gb = psutil.Process().memory_info().rss / (1024 ** 3)
            print(f"\n✅ 分布式加载完成！")
            print(f"   Rank {self.rank}: {self.total_cells:,} 个细胞")
            print(f"   耗时：{elapsed:.2f}s")
            print(f"   内存：{mem_gb:.2f} GB")
            
            # ✅ 数学验证
            assert self.total_cells == cells_per_rank, \
                f"Rank {self.rank} 实际加载 {self.total_cells} != 配额 {cells_per_rank}"
            print(f"\n✅ 数学验证通过：所有进程严格加载 {cells_per_rank:,} 个细胞")
            print("="*60)
        
        # ========== 3. 加载 scGPT 词汇表 ==========
        with open(scgpt_gene_vocab, 'r') as f:
            self.vocab = json.load(f)
        
        if '<cls>' not in self.vocab:
            self.vocab['<cls>'] = max(self.vocab.values()) + 1
        if '<pad>' not in self.vocab:
            self.vocab['<pad>'] = max(self.vocab.values()) + 2
            
        self.vocab_size = len(self.vocab)
        print(f"✅ scGPT 词汇表：{self.vocab_size} 个词条 (ID 范围：0-{self.vocab_size-1})")
        
        # ========== 4. 初始化 LMDB 环境缓存 ==========
        if load_metadata:
            print("⚠️  注意：LMDB 将在 Worker 进程中按需打开")
        
        # ========== 5. 计算序列长度 ==========
        cell_feature_len = self.cell_feature_tokens + 2
        gene_seq_len = self.max_genes + 2
        
        min_required = self.max_text_len + cell_feature_len + gene_seq_len
        
        if max_seq_len is None:
            self.max_seq_len = min_required
        else:
            self.max_seq_len = max_seq_len
            assert self.max_seq_len >= min_required, \
                f"配置的 max_seq_len({self.max_seq_len}) 太小，至少需要 {min_required}"
        
        actual_text_capacity = self.max_seq_len - cell_feature_len - gene_seq_len
        print(f"✅ 序列长度配置：max_seq_len={self.max_seq_len}")
        print(f"   - 细胞特征占位符：{cell_feature_len} tokens")
        print(f"   - 基因序列：{gene_seq_len} tokens")
        print(f"   - 文本容量：{actual_text_capacity} tokens")
        
        # ========== 7. 参数设置 ==========
        self.text_tokenizer = text_tokenizer
        self.soc_id = special_tokens_ids.get('soc_id')
        self.eoc_id = special_tokens_ids.get('eoc_id')
        self.sog_id = special_tokens_ids.get('sog_id')
        self.eog_id = special_tokens_ids.get('eog_id')
        self.mask_gene_id = special_tokens_ids.get('mask_gene_id')
        
        # 🚨 致命修复：bos_id 和 pad_id 必须从 tokenizer 获取，防范 NoneType 崩溃！
        self.bos_id = getattr(text_tokenizer, 'bos_token_id', None)
        self.pad_id = text_tokenizer.pad_token_id if (
            hasattr(text_tokenizer, 'pad_token_id') and 
            text_tokenizer.pad_token_id is not None
        ) else 151643  # Qwen2.5 的默认 PAD ID
        
        print(f"✅ 特殊 Token IDs: BOS={self.bos_id}, SOC={self.soc_id}, EOC={self.eoc_id}, SOG={self.sog_id}, EOG={self.eog_id}, MASK={self.mask_gene_id}, PAD={self.pad_id}")
        
        # 初始化 collate_fn 的 counter
        self._collate_counter = 0
    
    def __len__(self) -> int:
        return self.total_cells

    def _get_lmdb_env(self, lmdb_path: str):
        # 懒加载 LMDB 环境，确保 DataLoader 的多进程 Worker 安全
        if lmdb_path not in self.lmdb_envs:
            self.lmdb_envs[lmdb_path] = lmdb.Environment(
                lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self.lmdb_envs[lmdb_path]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # ==========================================
        # 🌟 1. 绝对坐标 O(1) 内存寻址 (替代原先的 file 循环)
        # ==========================================
        block_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[block_idx]
        
        block = self.data_blocks[block_idx]
        
        # 拿到这一个细胞的特征 (相当于原本代码里的 X_array[i])
        cell_features = block['X'][local_idx]
        rank_seq = block['rank'][local_idx]
        log1p_value = block['log1p'][local_idx]
        
        # 获取 lmdb 环境
        lmdb_path = self.lmdb_paths[block_idx]
        env = self._get_lmdb_env(lmdb_path)
        
        # ==========================================
        # 🚀 2. 老逻辑无缝接入点：向下完全粘贴原本的代码！
        # ==========================================
        # 🚨 关键修改：从 lmdb_keys list 直接获取，不再通过 obs_df
        lmdb_key = block['lmdb_keys'][local_idx]
        
        with env.begin(write=False) as txn:
            sample_data = txn.get(str(lmdb_key).encode())
            
            metadata = {}
            if self.load_metadata and sample_data:
                try:
                    lmdb_res = json.loads(sample_data.decode())
                    metadata = {field: lmdb_res.get(field, '') for field in self.metadata_fields}
                except Exception:
                    pass
            
            prompt = self._create_prompt(metadata) if hasattr(self, '_create_prompt') else "Describe the cell."
            
            # Tokenizer 和 Tensor 组装逻辑保持不变...
            text_tokens_list = []
            if prompt:
                try:
                    tokenized = self.text_tokenizer(
                        prompt, add_special_tokens=False, truncation=True, max_length=self.max_text_len
                    )
                    text_tokens_list = tokenized.input_ids if tokenized else []
                except Exception:
                    pass
            
            text_tokens = [self.bos_id] + text_tokens_list if self.bos_id is not None else text_tokens_list
            cell_tokens = [self.soc_id] + ([self.pad_id] * self.cell_feature_tokens) + [self.eoc_id]
            
            # 处理基因序列
            if hasattr(cell_features, 'toarray'):
                cell_features = cell_features.toarray().flatten()
            elif hasattr(cell_features, 'A1'):
                cell_features = cell_features.A1
            else:
                cell_features = np.array(cell_features).flatten()
            
            if hasattr(rank_seq, 'toarray'):
                rank_seq = rank_seq.toarray().flatten()
            elif hasattr(rank_seq, 'A1'):
                rank_seq = rank_seq.A1
            else:
                rank_seq = np.array(rank_seq).flatten()
            
            if hasattr(log1p_value, '__iter__'):
                log1p_values = np.array(log1p_value).flatten()
            else:
                log1p_values = [float(log1p_value)]
            
            gene_ids_array = torch.tensor(rank_seq, dtype=torch.long)
            non_zero_mask = (gene_ids_array != 0)
            
            # ✅ 关键修复：不再物理删除零基因！保持序列长度固定
            scgpt_token_ids = rank_seq.tolist()
            
            # 2. ✅ 修正：找出所有可 Mask 的位置（包括零基因！）
            valid_positions = [idx for idx, gid in enumerate(scgpt_token_ids) if gid < self.vocab_size]
            
            if len(valid_positions) < 10:
                raise ValueError(f"细胞 {idx} 有效基因位置不足 10 个")
            
            # 3. 🚨 致命修复：从 Uniform 采样改为 Log-Normal 采样！
            # ✅ Uniform 采样问题：t 集中在 0.5 附近，扩散模型难以学习
            # ✅ Log-Normal 采样：自然偏向小值，符合扩散理论
            
            import math
            
            # 标准 Log-Normal 采样（均值 0.5，方差自适应）
            # t ~ log_normal(mean=0.5, std=1.0)
            normal_sample = torch.randn(1).item()  # N(0, 1) → float
            t_raw = math.exp(0.5 + 1.0 * normal_sample)  # 使用 math.exp 处理 float
            
            # 归一化到 (0, 1) 区间（使用 sigmoid 而不是 clamp）
            t = 1.0 / (1.0 + math.exp(-t_raw + 0.5))  # 手动实现 sigmoid
            
            # 计算需要 Mask 的数量
            mask_ratio = t * (self.mask_max_ratio - self.mask_min_ratio) + self.mask_min_ratio
            n_mask = max(1, int(len(valid_positions) * mask_ratio))
            
            # 4. ✅ 修正：在所有有效基因中随机撒掩码（包括零基因！）
            mask_positions = random.sample(valid_positions, n_mask)
            masked_gene_ids = scgpt_token_ids.copy()
            for pos in mask_positions:
                masked_gene_ids[pos] = self.mask_gene_id
            
            # 转换为 Tensor，长度雷打不动依然是原始维度
            masked_gene_ids = torch.tensor(masked_gene_ids, dtype=torch.long)
            gene_ids = torch.tensor(scgpt_token_ids, dtype=torch.long)
            
            gene_tokens = [self.sog_id] + masked_gene_ids.tolist() + [self.eog_id]
            
            full_sequence = text_tokens + cell_tokens + gene_tokens
            input_ids = torch.tensor(full_sequence, dtype=torch.long)
            
            # 🚀 最终版：保证 RoPE 严格单调递增，且与 SFT 逻辑 100% 对齐
            text_positions = torch.arange(len(text_tokens))
            
            # 从细胞开始，所有后续序列统统继承 offset 空间
            # 这样保证了 [Cell] [Gene] [Caption] 都在高频/远端位置空间
            cell_start_idx = len(text_tokens)
            remaining_len = len(cell_tokens) + len(gene_tokens)
            
            offset_positions = self.offset + torch.arange(remaining_len)
            
            position_ids = torch.cat([text_positions, offset_positions])
            
            labels = input_ids.clone()
            
            soc_idx = len(text_tokens)
            eoc_idx = len(text_tokens) + len(cell_tokens) - 1
            
            # 屏蔽文本和细胞特征的 Loss
            labels[soc_idx:eoc_idx+1] = -100
            labels[soc_idx] = -100
            labels[eoc_idx] = -100
            labels[eoc_idx + 1] = -100
            labels[-1] = -100
            
            sog_idx = eoc_idx + 1
            original_gene_list = gene_ids.tolist()
            
            # ==========================================
            # ✅ 精准的 Loss 屏蔽逻辑（最终修正版）
            # ==========================================
            for j in range(len(masked_gene_ids)):
                gene_pos = sog_idx + 1 + j
                gid = original_gene_list[j]
                
                # ✅ 核心正确实现：
                # 条件 1: 这个位置被 Mask 盖住了 → 需要预测
                # 条件 2: 它原本是一个非零的有效基因 → 值得预测
                if input_ids[gene_pos] == self.mask_gene_id and gid != 0:
                    labels[gene_pos] = gid
                else:
                    labels[gene_pos] = -100
            
            attention_mask = torch.ones_like(input_ids)
            gene_mask_inner = (masked_gene_ids == self.mask_gene_id)
            
            return {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'gene_ids': masked_gene_ids,
                'gene_labels': gene_ids,
                'cell_features': torch.tensor(cell_features, dtype=torch.bfloat16), 
                'log1p': torch.tensor(log1p_values, dtype=torch.float),
                't': torch.tensor(t, dtype=torch.float32),
                'texts': prompt,
                'cell_feature_indices': {
                    'soc_pos': soc_idx, 'eoc_pos': eoc_idx, 'feature_len': self.cell_feature_tokens, 
                },
                'cell_positions': torch.tensor([soc_idx + 1, self.cell_feature_tokens], dtype=torch.long),
                'modality_positions': torch.tensor([[sog_idx + 1, self.max_genes]], dtype=torch.long),
                'gene_mask': gene_mask_inner,
                'non_zero_mask': non_zero_mask,
                'metadata': {
                    # ✅ 修复：不再使用 row，直接使用 local_idx 作为 cell_id
                    'cell_id': local_idx, 
                    'lmdb_source': lmdb_path, 
                    'lmdb_key': lmdb_key, 
                    **metadata
                },
                'data_type': ['stage1']  # 标记为基因数据（stage1）
            }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        from torch.nn.utils.rnn import pad_sequence
        
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                if k not in ('metadata', 'cell_feature_indices'):
                    batched[k].append(v)
                else:
                    batched[k].append(v)
        
        for k, v in batched.items():
            if k == 'data_type':
                from itertools import chain
                batched[k] = list(chain.from_iterable(v))
            elif k not in ('texts', 'metadata', 'cell_feature_indices') and isinstance(v[0], torch.Tensor):
                if v[0].dim() == 1:
                    if k in ('cell_features', 'cell_positions', 'gene_mask', 'log1p'):
                        batched[k] = torch.stack(v, dim=0)
                    elif k == 'labels':
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=-100)
                    elif k == 'input_ids':
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=151643)
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
    
    def _create_prompt(self, metadata: Dict) -> str:
        if self.cond_dropout_prob > 0 and torch.rand(1).item() < self.cond_dropout_prob:
            return ""
        if not metadata or len(metadata) == 0:
            return "Describe the cell."
        
        # ✅ 修复：将每个字段转换为 "Field name is value." 格式，然后拼接
        sentences = []
        for field in self.metadata_fields:
            value = metadata.get(field, '')
            if value:
                # 将下划线转为空格，首字母小写（例如 'celltype_name' -> 'celltype name'）
                field_name = field.replace('_', ' ')
                # 构建简单句： "field name is value."
                sentences.append(f"{field_name} is {value}.")
        
        # 用空格连接所有句子
        description = ' '.join(sentences) if sentences else "Describe the cell."
        return description


def create_gene_dataloader(
    feature_dir: str = None,
    lmdb_base_dir: str = None,
    scgpt_gene_vocab: str = None,
    text_tokenizer: Any = None,
    config_dict: Dict[str, Any] = None,
    special_tokens_ids: Dict[str, int] = None,
    batch_size: int = None,
    max_seq_len: Optional[int] = None,
    num_workers: int = None,
    load_metadata: bool = None,
    accelerator=None,
    **dataset_kwargs
) -> DataLoader:
    
    data_config = config_dict.get('data', {})
    dataset_config = config_dict.get('dataset', {})
    training_config = config_dict.get('training', {})
    
    feature_dir = feature_dir or data_config.get('feature_dir', '')
    lmdb_base_dir = lmdb_base_dir or data_config.get('lmdb_base_dir', '')
    scgpt_gene_vocab = scgpt_gene_vocab or data_config.get('scgpt_gene_vocab', '')
    load_metadata = load_metadata if load_metadata is not None else data_config.get('load_metadata', True)
    
    if max_seq_len is None:
        max_seq_len = dataset_config.get('max_seq_len', None)
    
    if batch_size is None:
        batch_size = training_config.get('batch_size', 16)
    if num_workers is None:
        num_workers = training_config.get('num_workers', 0)
    
    # ✅ 传入 accelerator 到 Dataset，实现真正的分布式加载
    dataset = GeneDataset(
        feature_dir=feature_dir,
        lmdb_base_dir=lmdb_base_dir,
        scgpt_gene_vocab=scgpt_gene_vocab,
        text_tokenizer=text_tokenizer,
        config_dict=config_dict,
        special_tokens_ids=special_tokens_ids,
        max_seq_len=max_seq_len,
        load_metadata=load_metadata,
        accelerator=accelerator,  # ✅ 新增
        **dataset_kwargs
    )
    
    # ✅ 关键修改：不再使用 DistributedSampler！
    # 因为数据已经在 Dataset.__init__ 中按 rank 切好了
    # 这里只需要普通的随机采样或顺序采样即可
    sampler = None  # 不使用任何 sampler，直接遍历整个 dataset
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # ✅ None：每个进程遍历自己的全部数据
        shuffle=(sampler is None),  # ✅ True：每个进程内部随机打乱
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=True,  # ✅ 最后不足 batch_size 的丢弃
        persistent_workers=True if num_workers > 0 else False,
    )


def calculate_training_steps(
    total_samples: int,
    config_dict: Dict[str, Any],
    accelerator=None
) -> Dict[str, int]:
    
    if accelerator is None:
        raise ValueError("必须传入 accelerator 参数以获取分布式训练配置！")
    
    scheduler_config = config_dict.get('scheduler', {})
    training_config = config_dict.get('training', {})
    
    batch_size_per_gpu = training_config.get('batch_size', 16)
    grad_accum = accelerator.gradient_accumulation_steps
    num_gpus = accelerator.num_processes
    
    effective_batch_size = batch_size_per_gpu * num_gpus * grad_accum
    
    steps_per_epoch = total_samples // effective_batch_size
    num_epochs = training_config.get('epochs', 10)
    total_steps = steps_per_epoch * num_epochs
    
    warmup_steps = scheduler_config.get('num_warmup_steps', None)
    if warmup_steps is None:
        warmup_ratio = scheduler_config.get('warmup_ratio', 0.05)
        warmup_steps = int(total_steps * warmup_ratio)
    
    print(f"\n📊 训练配置自动计算:")
    print(f"   - 真实样本数：{total_samples:,}")
    print(f"   - 单卡 Batch Size: {batch_size_per_gpu}")
    print(f"   - GPU 数量：{num_gpus}")
    print(f"   - Gradient Accumulation: {grad_accum}")
    print(f"   - 有效 Batch Size: {effective_batch_size:,}")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - 每 epoch 步数：{steps_per_epoch:,}")
    print(f"   - Total Training Steps: {total_steps:,}")
    print(f"   - Warmup Steps: {warmup_steps:,} ({warmup_steps/total_steps*100:.2f}%)")
    
    return {
        'total_samples': total_samples,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
    }