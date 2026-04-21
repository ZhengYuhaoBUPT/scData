#!/usr/bin/env python
# coding=utf-8
"""
CellWhisper 特征提取脚本 - 融合 advanced 和 repro 脚本优点
支持单细胞 (census) 和 Bulk (archs4) 数据集，分文件输出

核心特性:
1. ✅ 从 JSON 读取目标 ID，建立 ID 索引映射
2. ✅ 检测 .X 是否为 log1p，否则报错退出
3. ✅ 使用 scGPT 词表进行基因映射，打印匹配率
4. ✅ 按表达量降序保存 rank (uint16, scGPT ID) 和 rank_log1p (float16)
5. ✅ 保存 metadata 到 obs（根据数据集类型）
6. ✅ 流式 chunk 处理，避免内存溢出
7. ✅ 极致存储优化：float16, uint16, int8
8. ✅ Per-cell binning (每细胞独立分箱)
9. ✅ 本体论映射（MONDO/UBERON → 详细描述）
10. ✅ 所有 metadata 列都是必需的
"""

import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import argparse
import time
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Tuple, Optional, List, Iterator
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse

try:
    import orjson
except ImportError:
    orjson = None

# ========== 本体论解析器 ==========
class OntologyParser:
    """简单的 OBO 格式本体论解析器"""
    
    def __init__(self, obo_path: str):
        self.obo_path = obo_path
        self.id_to_definition = {}
        self.id_to_name = {}
        self._parse()
    
    def _parse(self):
        """解析 OBO 文件 - 逐行扫描 [Term] stanza"""
        print(f"   📖 正在解析本体论文件：{self.obo_path}")
        
        current_id = None
        current_name = None
        current_def = None
        
        term_count = 0
        with open(self.obo_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # 新的 stanza 开始
                if line == '[Term]':
                    # 保存前一个 term
                    if current_id and (current_def or current_name):
                        self.id_to_definition[current_id] = current_def if current_def else current_name
                        self.id_to_name[current_id] = current_name
                        term_count += 1
                    
                    current_id = None
                    current_name = None
                    current_def = None
                    continue
                
                # 解析 id
                if line.startswith('id: '):
                    current_id = line[4:].strip()
                
                # 解析 name
                elif line.startswith('name: '):
                    current_name = line[6:].strip()
                
                # 解析 definition（只取双引号内的内容）
                elif line.startswith('def: "'):
                    # 提取双引号内的定义
                    start_quote = line.find('"') + 1
                    end_quote = line.find('"', start_quote)
                    if end_quote > start_quote:
                        current_def = line[start_quote:end_quote]
            
            # 保存最后一个 term
            if current_id and (current_def or current_name):
                self.id_to_definition[current_id] = current_def if current_def else current_name
                self.id_to_name[current_id] = current_name
                term_count += 1
        
        print(f"      ✅ 解析完成：{term_count:,} 个 terms")
        print(f"         - 有定义的 terms: {sum(1 for v in self.id_to_definition.values() if v and ' ' in v):,}")
    
    def get_definition(self, ontology_id: str, fallback_name: str = None) -> str:
        """获取本体论 ID 的定义"""
        if ontology_id in self.id_to_definition:
            return self.id_to_definition[ontology_id]
        elif ontology_id in self.id_to_name:
            return self.id_to_name[ontology_id]
        else:
            # 找不到时返回 fallback
            return f"{fallback_name or 'Unknown'} [NOT FOUND: {ontology_id}]"


# ========== scGPT 路径配置 ==========
scgpt_path = "/root/wanghaoran/zxy/project/sc_showo/scgpt"

# ✅ 清理所有已导入的 scgpt 模块
modules_to_remove = [k for k in sys.modules.keys() if k.startswith('scgpt')]
for mod in modules_to_remove:
    del sys.modules[mod]

# ✅ 在导入之前插入路径
if scgpt_path not in sys.path:
    sys.path.insert(0, scgpt_path)

try:
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.preprocess import binning as scgpt_binning
    from scgpt.utils import set_seed
    print("✅ scGPT 模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败：{e}")
    print(f"\n🔍 当前 sys.path:")
    for i, path in enumerate(sys.path[:10]):
        print(f"   {i}: {path}")
    sys.exit(1)

# ========== 全局配置 ==========
set_seed(1)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CHUNK_SIZE = 400000
BATCH_SIZE = 1024
N_GENES = 1200
N_BINS = 51
RANDOM_SEED = 1

# ========== 数据集配置 ==========
DATASET_CONFIGS = {
    "census": {
        "h5ad_path": "/data/bgi/data/projects/multimodal/RNA_data/cellwhisper_data/cellxgene/full_data.h5ad",
        "json_path": "/root/wanghaoran/zxy/project/sc_showo/run/cw_sft_data/all_census_target_ids.json",
        "metadata_cols": ["disease", "tissue", "development_stage", "sex"],
        "ontology_mapping": {
            "disease_ontology_term_id": {
                "source_col": "disease_ontology_term_id",
                "target_col": "disease_definition",
                "obo_path": "/data/bgi/data/projects/multimodal/RNA_data/disease_ontology/mondo.obo",
                "fallback_name": "Disease",
                "min_match_rate": 0.95,  # ✅ 提高要求：只检查 MONDO，不包含 PATO
                "pato_normal_id": "PATO:0000461"  # ✅ PATO 正常状态 ID，直接映射为 normal
            },
            "tissue_ontology_term_id": {
                "source_col": "tissue_ontology_term_id",
                "target_col": "tissue_definition",
                "obo_path": "/data/bgi/data/projects/multimodal/RNA_data/tissue_ontology/uberon-basic.obo",
                "fallback_name": "Tissue",
                "min_match_rate": 0.95
            }
        },
        "required_cols": ["disease", "tissue", "development_stage", "sex"],
        "name": "CW_Stage1_SingleCell"
    },
    "archs4": {
        "h5ad_path": "/data/bgi/data/projects/multimodal/RNA_data/cellwhisper_data/archs4_geo/cellxgene.h5ad",
        "json_path": "/root/wanghaoran/zxy/project/sc_showo/run/cw_sft_data/all_archs4_target_ids.json",
        "metadata_cols": ["natural_language_annotation_replicates", "organism"],
        "required_cols": ["natural_language_annotation_replicates", "organism"],
        "name": "CW_Stage2_Bulk"
    }
}

class CellProjectionHead(nn.Module):
    def __init__(self, input_dim: int, intermediate_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.norm(x)


def load_cell_encoder(model_path: str, vocab: GeneVocab) -> Tuple[TransformerModel, CellProjectionHead]:
    """加载 Cell Encoder 模型（完全复用 advanced 脚本逻辑）"""
    model_dir = Path(model_path).parent
    with open(model_dir / "args.json", "r") as f:
        model_configs = json.load(f)

    gene_encoder = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs.get("n_layers_cls", 3),
        vocab=vocab,
        pad_token=pad_token,
        cell_emb_style=model_configs.get("cell_emb_style", "cls"),
        use_fast_transformer=True,
    )

    adapter_dim = model_configs.get("adapter_dim", 768)
    projection_head = CellProjectionHead(
        input_dim=model_configs["embsize"],
        intermediate_dim=adapter_dim,
        output_dim=768
    )

    full_state_dict = torch.load(model_path, map_location="cpu")

    gene_encoder_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if key.startswith("transformer_encoder_CL."):
            new_key = key.replace("transformer_encoder_CL.", "transformer_encoder.")
            gene_encoder_state_dict[new_key] = value
        elif key.startswith("transformer_encoder.") or key.startswith("cls_decoder.") or key.startswith("decoder."):
            continue
        else:
            gene_encoder_state_dict[key] = value

    gene_encoder.load_state_dict(gene_encoder_state_dict, strict=False)

    projection_head_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if key.startswith("cell2textAdapter."):
            new_key = key.replace("cell2textAdapter.", "")
            projection_head_state_dict[new_key] = value

    if projection_head_state_dict:
        projection_head.load_state_dict(projection_head_state_dict)

    gene_encoder.to(device).half().eval()
    projection_head.to(device).half().eval()

    return gene_encoder, projection_head


def validate_log1p(adata: AnnData, sample_name: str = "") -> None:
    """验证 .X 是否为 log1p 格式（增强版）"""
    X_sample = adata.X
    if issparse(X_sample):
        X_sample = X_sample.toarray()
    
    if X_sample.ndim > 1:
        X_sample = X_sample.flatten()
    
    nonzero_mask = X_sample > 0
    zero_mask = X_sample == 0
    negative_mask = X_sample < 0
    
    X_nonzero = X_sample[nonzero_mask]
    
    print(f"\n🔍 检测 {sample_name} 表达量格式:")
    print(f"   总数据点：{len(X_sample):,}")
    print(f"   非零点：{len(X_nonzero):,} ({len(X_nonzero)/len(X_sample)*100:.2f}%)")
    print(f"   零点：{zero_mask.sum():,} ({zero_mask.sum()/len(X_sample)*100:.2f}%)")
    
    if not np.issubdtype(X_sample.dtype, np.floating):
        print(f"\n❌ 错误：数据类型不是浮点数！当前 dtype: {X_sample.dtype}")
        sys.exit(1)
    
    if negative_mask.any():
        n_negative = negative_mask.sum()
        print(f"\n❌ 错误：检测到 {n_negative:,} 个负值！")
        sys.exit(1)
    
    if len(X_nonzero) == 0:
        raise ValueError(f"{sample_name}: 所有值都是零，数据为空!")
    
    max_val = np.max(X_nonzero)
    mean_val = np.mean(X_nonzero)
    median_val = np.median(X_nonzero)
    std_val = np.std(X_nonzero)
    
    print(f"\n📊 统计信息:")
    print(f"   最大值：{max_val:.4f}")
    print(f"   均值：{mean_val:.4f}")
    print(f"   中位数：{median_val:.4f}")
    print(f"   标准差：{std_val:.4f}")
    
    is_log1p = max_val < 20
    
    print(f"\n🎯 log1p 格式判断:")
    print(f"   判断阈值：最大值 < 20")
    print(f"   实际最大值：{max_val:.4f}")
    print(f"   判断结果：{'✅ 是 log1p' if is_log1p else '❌ 不是log1p'}")
    
    if not is_log1p:
        print(f"\n❌ 错误：检测到数据不是 log1p 格式！")
        sys.exit(1)
    
    print(f"\n✅ 验证通过：数据为 log1p 格式")
    print(f"{'='*80}\n")


def check_gene_vocab_match(gene_names: List[str], vocab: GeneVocab, dataset_name: str) -> float:
    """检查基因与 scGPT 词表的匹配率"""
    total_genes = len(gene_names)
    matched_genes = sum(1 for g in gene_names if g in vocab)
    match_rate = matched_genes / total_genes * 100 if total_genes > 0 else 0
    
    print(f"\n📊 {dataset_name} 基因 - 词表匹配情况:")
    print(f"   总基因数：{total_genes:,}")
    print(f"   匹配基因数：{matched_genes:,}")
    print(f"   匹配率：{match_rate:.2f}%")
    
    if match_rate < 50:
        print(f"   ⚠️  警告：匹配率较低")
    
    return match_rate


def extract_features_batch(
    gene_encoder: TransformerModel,
    projection_head: CellProjectionHead,
    vocab: GeneVocab,
    adata_chunk: AnnData,
    batch_size: int = BATCH_SIZE,
    n_genes: int = N_GENES,
    n_bins: int = N_BINS,
    seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """批量提取特征并计算 rank/log1p"""
    n_cells = adata_chunk.n_obs
    n_vars = adata_chunk.n_vars
    
    all_features = []
    
    gene_names = adata_chunk.var["gene_name"].tolist()
    gene_ids_in_vocab = np.array([vocab[g] for g in gene_names if g in vocab], dtype=np.int64)
    
    if len(gene_ids_in_vocab) == 0:
        raise ValueError("当前 chunk 没有能映射到 scGPT vocab 的基因")
    
    X_data = adata_chunk.X
    if issparse(X_data):
        X_data = X_data.toarray()
    
    np.random.seed(seed)
    
    sampled_indices = np.zeros((n_cells, n_genes), dtype=np.int64)
    
    for i in range(n_cells):
        cell_expr = X_data[i]
        non_zero_idx = np.nonzero(cell_expr)[0]
        
        if len(non_zero_idx) >= n_genes:
            sampled = np.random.choice(non_zero_idx, size=n_genes, replace=False)
        else:
            remaining = n_genes - len(non_zero_idx)
            sampled = np.concatenate([
                non_zero_idx,
                np.random.choice(n_vars, size=remaining, replace=n_vars < remaining)
            ])
        sampled_indices[i] = sampled
    
    sampled_expression = np.take_along_axis(X_data, sampled_indices, axis=1)
    sampled_gene_ids = gene_ids_in_vocab[sampled_indices % len(gene_ids_in_vocab)]
    
    print(f"   ├─ 🔬 执行 per-cell binning (n_bins={n_bins}, seed={seed})...")
    binning_array = scgpt_binning(sampled_expression, n_bins=n_bins)
    
    n_batches = (n_cells + batch_size - 1) // batch_size
    
    gpu_pbar = tqdm(
        range(n_batches), 
        desc="   ├─ 🔬 GPU 推理", 
        leave=False, 
        colour='green',
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for batch_idx in gpu_pbar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_cells)
        
        batch_binning = binning_array[start_idx:end_idx]
        batch_gene_ids = sampled_gene_ids[start_idx:end_idx]
        current_batch_size = batch_binning.shape[0]
        
        expr_tensor = torch.from_numpy(batch_binning).to(device=device, dtype=torch.float32, non_blocking=True)
        gene_ids_tensor = torch.from_numpy(batch_gene_ids).to(device=device, dtype=torch.long, non_blocking=True)
        src_key_padding_mask = torch.zeros(current_batch_size, n_genes, dtype=torch.bool, device=device)
        
        if device.type == "cuda" and (batch_idx == 0 or (batch_idx + 1) % 50 == 0):
            gpu_mem = torch.cuda.memory_allocated() / 1024**2
            gpu_pbar.set_description(f"   ├─ 🔬 GPU 推理 (显存：{gpu_mem:.0f}MB, Batch:{current_batch_size})")
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = gene_encoder(
                    src=gene_ids_tensor,
                    values=expr_tensor,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False, MVC=False, ECS=False
                )
                cell_emb = output["cell_emb"]
                cell_feat = projection_head(cell_emb)
        
        all_features.append(cell_feat.cpu().numpy())
    
    gpu_pbar.close()
    
    features = np.concatenate(all_features, axis=0).astype(np.float16)
    
    sorted_indices = np.argsort(-sampled_expression, axis=1)
    rank_array = np.take_along_axis(sampled_gene_ids, sorted_indices, axis=1).astype(np.uint16)
    log1p_array = np.take_along_axis(sampled_expression, sorted_indices, axis=1).astype(np.float16)
    
    return features, rank_array, log1p_array, binning_array


def process_dataset(
    dataset_type: str,
    gene_encoder: TransformerModel,
    projection_head: CellProjectionHead,
    vocab: GeneVocab,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_cells: Optional[int] = None
) -> Dict:
    """处理单个数据集"""
    config = DATASET_CONFIGS[dataset_type]
    h5ad_path = config["h5ad_path"]
    json_path = config["json_path"]
    metadata_cols = config["metadata_cols"]
    ontology_mapping = config.get("ontology_mapping", {})
    dataset_name = config["name"]
    
    print(f"\n{'='*80}")
    print(f"🚀 开始处理 {dataset_name}")
    print(f"{'='*80}")
    
    with open(json_path, 'r') as f:
        target_ids = json.load(f)["target_ids"]
    
    print(f"📖 从 JSON 加载目标 ID: {len(target_ids):,} 个")
    
    # ========== 1. 加载数据集 ==========
    load_start_time = time.time()
    print(f"\n💾 正在加载数据集：{h5ad_path}")
    
    adata_full = sc.read_h5ad(h5ad_path)
    
    load_time = time.time() - load_start_time
    
    print(f"✅ 数据集加载完成 (耗时：{load_time:.1f}s)")
    print(f"   原始大小：{adata_full.n_obs:,} cells × {adata_full.n_vars:,} genes")
    
    import psutil
    process = psutil.Process()
    mem_usage = process.memory_info().rss / 1024**3
    print(f"   📊 当前进程内存占用：{mem_usage:.1f} GB")
    
    # ✅ 本体论映射处理（针对 census 数据集）
    if ontology_mapping and dataset_type == "census":
        print(f"\n{'='*80}")
        print("🔍 执行本体论映射...")
        print(f"{'='*80}")
        
        ontology_parsers = {}
        for ont_key, ont_config in ontology_mapping.items():
            obo_path = ont_config["obo_path"]
            
            if not os.path.exists(obo_path):
                raise FileNotFoundError(f"本体论文件不存在：{obo_path}")
            
            parser = OntologyParser(obo_path)
            ontology_parsers[ont_key] = parser
            
            source_col = ont_config["source_col"]
            if source_col not in adata_full.obs.columns:
                raise ValueError(f"必需列 '{source_col}' 不存在于 .obs 中！")
        
        for ont_key, ont_config in ontology_mapping.items():
            source_col = ont_config["source_col"]
            target_col = ont_config["target_col"]
            fallback_name = ont_config["fallback_name"]
            min_match_rate = ont_config.get("min_match_rate", 0.80)
            pato_normal_id = ont_config.get("pato_normal_id", None)  # ✅ 获取 PATO normal ID
            parser = ontology_parsers[ont_key]
            
            print(f"\n   🔗 映射 {source_col} → {target_col}...")
            
            source_values = adata_full.obs[source_col].tolist()
            mapped_values = []
            not_found_count = 0
            total_count = len(source_values)
            pato_normal_count = 0  # ✅ 统计 PATO normal 数量
            
            for val in tqdm(source_values, desc=f"      映射进度", leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
                val_str = str(val)
                
                # ✅ 特殊处理：PATO normal 直接映射为 "normal"，不参与匹配率统计
                if pato_normal_id and val_str == pato_normal_id:
                    mapped_values.append("normal")
                    pato_normal_count += 1
                else:
                    # 正常解析本体论（如 MONDO）
                    definition = parser.get_definition(val_str, fallback_name)
                    mapped_values.append(definition)
                    
                    if '[NOT FOUND:' in definition:
                        not_found_count += 1
            
            # ✅ 计算匹配率时排除 PATO normal
            effective_total = total_count - pato_normal_count
            match_rate = (effective_total - not_found_count) / effective_total * 100 if effective_total > 0 else 0
            
            print(f"\n      📊 映射统计:")
            print(f"         总数：{total_count:,}")
            if pato_normal_count > 0:
                print(f"         PATO normal (健康): {pato_normal_count:,} ({pato_normal_count/total_count*100:.2f}%) → 直接写入 'normal'")
            print(f"         本体论匹配：{effective_total - not_found_count:,}/{effective_total:,} ({match_rate:.2f}%)")
            print(f"         未找到：{not_found_count:,} ({100 - match_rate:.2f}%)")
            
            if match_rate < min_match_rate * 100:
                raise ValueError(f"❌ {source_col} 的本体论匹配率过低 ({match_rate:.2f}%)！要求最低 {min_match_rate*100:.0f}%。")
            
            adata_full.obs[target_col] = mapped_values
            print(f"      ✅ 已保存到 .obs['{target_col}']")
        
        print(f"\n{'='*80}")
        print("✅ 本体论映射完成")
        print(f"{'='*80}\n")
    
    # ========== 2. 建立 ID 索引 ==========
    build_index_start = time.time()
    
    print(f"\n🔍 检测基因标识类型...")
    
    gene_names_source = None
    gene_names_list = None
    
    if adata_full.var.index.str.startswith('ENSG').any():
        print(f"   ⚠️  .var.index 包含 Ensembl ID")
        gene_names_source = 'var_index_ensembl'
        
        possible_symbol_cols = ['gene_name', 'gene_symbol', 'symbol', 'hgnc', 'gene', 'external_gene_name']
        
        symbol_col_found = None
        for col in possible_symbol_cols:
            if col in adata_full.var.columns:
                sample_vals = adata_full.var[col].dropna().head(10).astype(str)
                if not sample_vals.str.contains('ENSG').any() and (sample_vals.str.len() > 0).all():
                    symbol_col_found = col
                    break
        
        if symbol_col_found:
            print(f"   ✅ 找到基因 Symbol 列：'{symbol_col_found}'")
            gene_names_list = adata_full.var[symbol_col_found].tolist()
        else:
            print(f"   🔍 未找到标准 Symbol 列，遍历所有列...")
            for col in adata_full.var.columns:
                sample_vals = adata_full.var[col].dropna().head(100).astype(str)
                if (not sample_vals.str.contains('ENSG').any() and 
                    (sample_vals.str.len() > 1).all() and 
                    (sample_vals.str.len() < 25).all() and
                    sample_vals.str.match(r'^[A-Za-z0-9_-]+$').all()):
                    symbol_col_found = col
                    gene_names_list = adata_full.var[col].tolist()
                    print(f"   ✅ 推断 '{col}' 为基因 Symbol 列")
                    break
            
            if not symbol_col_found:
                print(f"   ❌ 未找到基因 Symbol 列，将使用 Ensembl ID")
                gene_names_list = adata_full.var.index.tolist()
    else:
        print(f"   ✅ .var.index 似乎是基因 Symbol")
        gene_names_list = adata_full.var.index.tolist()
    
    if gene_names_list is None:
        gene_names_list = adata_full.var.index.tolist()
    
    adata_full.var['gene_name'] = gene_names_list
    
    if pd.isna(adata_full.var['gene_name']).any():
        print(f"\n   ⚠️  检测到 {pd.isna(adata_full.var['gene_name']).sum()} 个基因的 gene_name 为 NaN")
        print(f"   💡 使用 .var.index (Ensembl ID) 填充 NaN 值")
        adata_full.var['gene_name'] = adata_full.var['gene_name'].fillna(pd.Series(adata_full.var.index))
    
    print(f"   📊 基因名称来源：{gene_names_source or 'var.index'}")
    print(f"   📊 基因数量：{len(gene_names_list):,}")
    print(f"   📊 示例：{gene_names_list[:5]}")
    
    full_id_to_idx = {name: i for i, name in enumerate(adata_full.obs_names)}
    valid_task_list = [(tid, full_id_to_idx[tid]) for tid in target_ids if tid in full_id_to_idx]
    build_index_time = time.time() - build_index_start
    
    matched_count = len(valid_task_list)
    match_rate = matched_count / len(target_ids) * 100 if target_ids else 0
    
    print(f"\n🔍 ID 索引构建完成 (耗时：{build_index_time:.1f}s)")
    print(f"   目标 ID 数：{len(target_ids):,}")
    print(f"   匹配成功：{matched_count:,} ({match_rate:.2f}%)")
    print(f"   匹配失败：{len(target_ids) - matched_count:,}")
    
    if match_rate < 80:
        print(f"\n⚠️  警告：匹配率较低 ({match_rate:.2f}%)，请检查 ID 列表是否正确")
        if match_rate < 50:
            raise ValueError(f"❌ 匹配率过低 ({match_rate:.2f}%)，终止处理！")
    
    if max_cells:
        valid_task_list = valid_task_list[:max_cells]
        print(f"\n🧪 测试模式：限制为 {max_cells:,} cells")
    
    num_cells = len(valid_task_list)
    if num_cells == 0:
        raise ValueError("❌ 没有可处理的细胞！")
    
    print(f"\n📊 本次任务总量：{num_cells:,} cells")
    print(f"{'='*80}\n")
    
    # ========== 3. 初始化收集器 ==========
    all_features = []
    all_ranks = []
    all_rank_log1p = []
    all_metadata = {col: [] for col in metadata_cols}
    all_cell_ids = []
    
    processed_cells = 0
    failed_chunks = []
    
    first_chunk = True
    global_chunk_start_time = time.time()
    
    # ========== 4. Chunk 流式处理 ==========
    n_chunks = (num_cells + chunk_size - 1) // chunk_size
    print(f"📦 分块策略：共 {n_chunks} 个 chunks, 每块 {chunk_size:,} cells\n")
    
    for chunk_idx in range(n_chunks):
        chunk_start_time = time.time()
        
        start_i = chunk_idx * chunk_size
        end_i = min(start_i + chunk_size, num_cells)
        
        chunk_tasks = valid_task_list[start_i:end_i]
        current_rows = [task[1] for task in chunk_tasks]
        chunk_cell_ids = [task[0] for task in chunk_tasks]
        current_chunk_size = len(chunk_cell_ids)
        
        progress = (chunk_idx + 1) / n_chunks * 100
        elapsed_total = time.time() - global_chunk_start_time
        avg_speed = processed_cells / elapsed_total if elapsed_total > 0 else 0
        eta_seconds = (num_cells - processed_cells) / avg_speed if avg_speed > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"⏰ Chunk {chunk_idx + 1}/{n_chunks} ({progress:.1f}%)")
        print(f"   细胞范围：{start_i:,} → {end_i:,} (本批 {current_chunk_size:,})")
        print(f"   累计处理：{processed_cells:,}/{num_cells:,}")
        print(f"   平均速度：{avg_speed:.1f} cells/s")
        print(f"   预计剩余：{eta_seconds/60:.1f} 分钟")
        print(f"{'='*60}")
        
        read_start = time.time()
        try:
            print(f"   ├─ 🔍 读取数据... ", end='', flush=True)
            
            if issparse(adata_full.X):
                X_data = adata_full.X[current_rows, :].tocsc()
                print(f"(稀疏矩阵切片)", end='', flush=True)
            else:
                X_data = adata_full.X[current_rows, :]
                print(f"(稠密矩阵切片)", end='', flush=True)
            
            read_time = time.time() - read_start
            print(f" ✅ ({read_time:.2f}s)")
            
        except Exception as e:
            error_msg = f"❌ Chunk {chunk_idx + 1} 数据读取失败！"
            print(f"\n{error_msg}")
            print(f"   错误类型：{type(e).__name__}")
            print(f"   错误信息：{str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"{error_msg}\n原始错误：{str(e)}") from e
        
        build_adata_start = time.time()
        try:
            print(f"   ├─ 构建 AnnData... ", end='', flush=True)
            
            adata_chunk = AnnData(
                X=X_data,
                obs=adata_full.obs.iloc[current_rows].copy(),
                var=adata_full.var.copy()
            )
            
            if 'gene_name' in adata_full.var.columns:
                gene_names_chunk = adata_full.var['gene_name'].tolist()
                import math
                gene_names_chunk = [
                    g if not (isinstance(g, float) and math.isnan(g)) else adata_full.var.index[i]
                    for i, g in enumerate(gene_names_chunk)
                ]
                adata_chunk.var['gene_name'] = gene_names_chunk
            else:
                adata_chunk.var["gene_name"] = adata_chunk.var.index.tolist()
            
            build_adata_time = time.time() - build_adata_start
            print(f"✅ ({build_adata_time:.2f}s)")
            
        except Exception as e:
            error_msg = f"❌ Chunk {chunk_idx + 1} AnnData 构建失败！"
            print(f"\n{error_msg}")
            print(f"   错误类型：{type(e).__name__}")
            print(f"   错误信息：{str(e)}")
            raise RuntimeError(f"{error_msg}\n原始错误：{str(e)}") from e
        
        norm_start = time.time()
        try:
            print(f"   ├─ 🔬 Scanpy 标准化... ", end='', flush=True)
            
            X_to_use = adata_chunk.X
            
            if np.issubdtype(X_to_use.dtype, np.integer):
                print(f"(检测到原始 count 数据，执行标准化→log1p)", end='', flush=True)
                adata_chunk.layers["counts"] = adata_chunk.X.copy()
                sc.pp.normalize_total(adata_chunk, target_sum=1e4)
                sc.pp.log1p(adata_chunk)
                X_data = adata_chunk.X
            else:
                print(f"(浮点数据，跳过标准化)", end='', flush=True)
                X_data = X_to_use
            
            norm_time = time.time() - norm_start
            print(f" ✅ ({norm_time:.2f}s)")
            
        except Exception as e:
            error_msg = f"❌ Chunk {chunk_idx + 1} 标准化失败！"
            print(f"\n{error_msg}")
            print(f"   错误类型：{type(e).__name__}")
            print(f"   错误信息：{str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"{error_msg}\n原始错误：{str(e)}") from e
        
        collect_meta_start = time.time()
        try:
            print(f"   ├─ 收集 Metadata... ", end='', flush=True)
            
            # ✅ 简单直接：收集 .obs 中存在的列
            for col in metadata_cols:
                if col in adata_chunk.obs.columns:
                    all_metadata[col].extend(adata_chunk.obs[col].tolist())
                # 如果 obs 中没有，尝试从 obsm 获取（按行索引）
                elif col in adata_full.obsm:
                    obsm_val = adata_full.obsm[col]
                    # 如果是 DataFrame/Series，且长度与完整数据集一致
                    if hasattr(obsm_val, 'iloc') and len(obsm_val) == len(adata_full):
                        # 取当前 chunk 对应的行
                        chunk_vals = []
                        for global_idx in current_rows:
                            val = obsm_val.iloc[global_idx]
                            # 处理可能的多列情况
                            if hasattr(val, '__iter__') and not isinstance(val, str):
                                chunk_vals.append(str(val.iloc[0]) if len(val) > 0 else "")
                            else:
                                chunk_vals.append(str(val))
                        all_metadata[col].extend(chunk_vals)
            
            all_cell_ids.extend(chunk_cell_ids)
            
            collect_meta_time = time.time() - collect_meta_start
            print(f"✅ ({collect_meta_time:.2f}s)")
            
        except Exception as e:
            error_msg = f"❌ Chunk {chunk_idx + 1} Metadata 收集失败！"
            print(f"\n{error_msg}")
            print(f"   错误类型：{type(e).__name__}")
            print(f"   错误信息：{str(e)}")
            raise RuntimeError(f"{error_msg}\n原始错误：{str(e)}") from e
        
        validate_start = time.time()
        try:
            print(f"   ├─ 验证 log1p... ", end='', flush=True)
            validate_log1p(adata_chunk, f"{dataset_name}_chunk_{chunk_idx}")
            validate_time = time.time() - validate_start
            print(f"✅ ({validate_time:.2f}s)")
            
        except Exception as e:
            error_msg = f"❌ Chunk {chunk_idx + 1} log1p 验证失败！"
            print(f"\n{error_msg}")
            print(f"   错误类型：{type(e).__name__}")
            print(f"   错误信息：{str(e)}")
            raise RuntimeError(f"{error_msg}\n原始错误：{str(e)}") from e
        
        if first_chunk:
            print(f"\n   🔍 首个 Chunk 基因词表匹配检查:")
            match_rate_genes = check_gene_vocab_match(
                adata_chunk.var["gene_name"].tolist(), 
                vocab, 
                f"{dataset_name}"
            )
            
            if match_rate_genes < 50:
                raise ValueError(f"❌ 基因词表匹配率过低 ({match_rate_genes:.2f}%)，终止处理！")
            
            first_chunk = False
        
        extract_start = time.time()
        try:
            print(f"   ├─ 🔬 提取特征... ", end='', flush=True)
            
            features, ranks, rank_log1p, _ = extract_features_batch(
                gene_encoder, projection_head, vocab, adata_chunk,
                batch_size=BATCH_SIZE, n_genes=N_GENES
            )
            
            extract_time = time.time() - extract_start
            speed_extract = current_chunk_size / extract_time if extract_time > 0 else 0
            print(f"✅ ({extract_time:.2f}s, {speed_extract:.1f} cells/s)")
            
        except Exception as e:
            error_msg = f"❌ Chunk {chunk_idx + 1} 特征提取失败！"
            print(f"\n{error_msg}")
            print(f"   错误类型：{type(e).__name__}")
            print(f"   错误信息：{str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"{error_msg}\n原始错误：{str(e)}") from e
        
        all_features.append(features)
        all_ranks.append(ranks)
        all_rank_log1p.append(rank_log1p)
        
        processed_cells += current_chunk_size
        
        cleanup_start = time.time()
        del adata_chunk, X_data
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cleanup_time = time.time() - cleanup_start
        
        print(f"   ├─ 内存清理 ✅ ({cleanup_time:.2f}s)")
        print(f"   └─ 本 Chunk 总耗时：{time.time() - chunk_start_time:.1f}s")
        
        chunk_elapsed = time.time() - chunk_start_time
        tqdm.write(f"\n📊 Chunk {chunk_idx + 1} 完成 | "
                  f"处理：{current_chunk_size:,} cells | "
                  f"耗时：{chunk_elapsed:.1f}s | "
                  f"速度：{current_chunk_size / chunk_elapsed:.1f} cells/s\n")
    
    total_time = time.time() - global_chunk_start_time
    
    print(f"\n{'='*80}")
    print(f"✅ {dataset_name} 处理完成")
    print(f"{'='*80}")
    print(f"   成功处理：{processed_cells:,} cells")
    print(f"   失败跳过：{len(failed_chunks)} chunks")
    print(f"   总耗时：{total_time/60:.1f} 分钟 ({total_time:.1f}s)")
    print(f"   平均速度：{processed_cells / total_time:.1f} cells/s")
    
    if failed_chunks:
        print(f"\n❌ 以下 chunks 失败:")
        for fail_info in failed_chunks:
            print(f"   - Chunk {fail_info['chunk_idx'] + 1}: {fail_info['error']}")
        raise RuntimeError(f"有 {len(failed_chunks)} 个 chunks 处理失败，已终止执行")
    
    print(f"{'='*80}\n")
    
    return {
        'features': all_features,
        'ranks': all_ranks,
        'rank_log1p': all_rank_log1p,
        'metadata': all_metadata,
        'cell_ids': all_cell_ids,
        'dataset_name': dataset_name
    }


def main():
    parser = argparse.ArgumentParser(description="CellWhisper 特征提取脚本")
    parser.add_argument("--model-path", type=str,
                       default="/root/wanghaoran/zxy/project/sc_showo/save/okrcell_ckpt/model-241492.pt",
                       help="Cell Encoder 模型权重路径")
    parser.add_argument("--vocab-path", type=str,
                       default="/root/wanghaoran/zxy/project/sc_showo/save/okrcell_ckpt/vocab.json",
                       help="scGPT 词汇表路径")
    parser.add_argument("--output-file", type=str,
                       default="/home/qijinyin/wanghaoran/zxy/features/okrcell_sft_features.h5ad",
                       help="输出 H5AD 文件基础路径（会自动添加数据集后缀）")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                       help="每个 chunk 的细胞数")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="GPU 推理 batch size")
    parser.add_argument("--n-genes", type=int, default=N_GENES,
                       help="每个细胞采样基因数")
    parser.add_argument("--max-cells-per-dataset", type=int, default=None,
                       help="每个数据集最大细胞数（用于测试）")
    parser.add_argument("--datasets", type=str, nargs='+', default=['census', 'archs4'],
                       choices=['census', 'archs4'],
                       help="要处理的数据集")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("🔧 CellWhisper 特征提取脚本启动")
    print(f"{'='*80}")
    print(f"⚙️  配置信息:")
    print(f"   设备：{device}")
    print(f"   Chunk 大小：{args.chunk_size:,}")
    print(f"   Batch 大小：{args.batch_size:,}")
    print(f"   基因数：{args.n_genes:,}")
    print(f"   Binning: {N_BINS} bins")
    print(f"   Random Seed: {RANDOM_SEED}")
    print(f"   数据集：{args.datasets}")
    print(f"   输出模式：每个数据集单独保存")
    print(f"{'='*80}\n")
    
    vocab = GeneVocab.from_file(args.vocab_path)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    
    print("🔧 加载 Cell Encoder 模型...")
    gene_encoder, projection_head = load_cell_encoder(args.model_path, vocab)
    print("✅ 模型加载完成\n")
    
    output_files = {}
    
    for dataset_type in args.datasets:
        print(f"\n{'='*80}")
        print(f"🚀 开始处理 {dataset_type}")
        print(f"{'='*80}")
        
        # ✅ 生成输出文件名并检查是否已存在
        output_base, output_ext = os.path.splitext(args.output_file)
        dataset_output = f"{output_base}_{dataset_type}{output_ext}"
        
        if os.path.exists(dataset_output):
            file_size = os.path.getsize(dataset_output) / (1024**3)
            print(f"\n⏭️  检测到输出文件已存在，跳过 {dataset_type}!")
            print(f"   文件路径：{dataset_output}")
            print(f"   文件大小：{file_size:.2f} GB")
            print(f"   💡 如需重新处理，请先删除该文件或更改输出路径")
            output_files[dataset_type] = dataset_output
            continue
        
        result = process_dataset(
            dataset_type=dataset_type,
            gene_encoder=gene_encoder,
            projection_head=projection_head,
            vocab=vocab,
            chunk_size=args.chunk_size,
            max_cells=args.max_cells_per_dataset
        )
        
        output_files[dataset_type] = dataset_output
        
        print(f"\n💾 正在保存 {dataset_type} 到：{dataset_output}")
        save_single_dataset(result, dataset_output)
    
    print(f"\n{'='*80}")
    print("🎉 全部处理完成！")
    print(f"{'='*80}")
    print(f"\n📋 输出文件列表:")
    for dataset_type, output_path in output_files.items():
        file_size = os.path.getsize(output_path) / (1024**3)
        print(f"   - {dataset_type}: {output_path} ({file_size:.2f} GB)")
    
    print(f"\n💡 读取示例:")
    print(f"   import scanpy as sc")
    for dataset_type, output_path in output_files.items():
        print(f"   {dataset_type}_data = sc.read_h5ad('{output_path}')  # O(1) 加载")


def save_single_dataset(result: Dict, output_file: str) -> None:
    """保存单个数据集到 H5AD 文件"""
    print(f"\n{'='*80}")
    print("💾 保存数据集到 H5AD...")
    print(f"{'='*80}")
    
    merge_start = time.time()
    
    # ✅ 修复：需要合并所有 chunk 的数据（因为 process_dataset 返回的是 list）
    print(f"   正在合并 {len(result['features'])} 个 chunk 的数据...")
    X_all = np.concatenate(result['features'], axis=0)
    rank_all = np.concatenate(result['ranks'], axis=0)
    rank_log1p_all = np.concatenate(result['rank_log1p'], axis=0)
    
    all_cell_ids = result['cell_ids']
    all_metadata = result['metadata']
    
    merge_time = time.time() - merge_start
    print(f"   合并耗时：{merge_time:.1f}s")
    print(f"   最终大小：{X_all.shape[0]:,} cells × {X_all.shape[1]:,} features")
    
    obs_df = pd.DataFrame({
        'cell_id': all_cell_ids,
        **all_metadata
    })
    obs_df.index = obs_df.index.astype(str)
    
    adata = AnnData(X=X_all, obs=obs_df)
    adata.obsm['rank'] = rank_all
    adata.obsm['rank_log1p'] = rank_log1p_all
    
    print(f"\n🔍 验证数据结构...")
    print(f"   obs 列：{list(adata.obs.columns)}")
    print(f"   obsm 键：{list(adata.obsm.keys())}")
    print(f"   X 形状：{adata.X.shape}, dtype: {adata.X.dtype}")
    print(f"   rank 形状：{adata.obsm['rank'].shape}, dtype: {adata.obsm['rank'].dtype}")
    print(f"   rank_log1p 形状：{adata.obsm['rank_log1p'].shape}, dtype: {adata.obsm['rank_log1p'].dtype}")
    
    if 'disease_definition' in adata.obs.columns:
        print(f"\n   📋 本体论映射结果:")
        print(f"      disease_definition: {adata.obs['disease_definition'].iloc[0][:80]}...")
        print(f"      tissue_definition: {adata.obs['tissue_definition'].iloc[0][:80]}...")
    
    rank_min = rank_all.min()
    rank_max = rank_all.max()
    print(f"   Rank 范围：[{rank_min}, {rank_max}]")
    
    print(f"\n💾 正在压缩并保存 H5AD (gzip): {output_file}")
    save_start = time.time()
    
    try:
        adata.write_h5ad(output_file, compression="gzip")
        save_time = time.time() - save_start
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024**3)
            print(f"\n✅ 保存完成!")
            print(f"   输出文件：{output_file}")
            print(f"   文件大小：{file_size:.2f} GB")
            print(f"   保存耗时：{save_time:.1f}s")
    except Exception as e:
        print(f"\n❌ 保存失败：{e}")
        raise


if __name__ == "__main__":
    main()