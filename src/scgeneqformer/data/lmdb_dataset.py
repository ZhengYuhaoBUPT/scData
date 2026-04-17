#!/usr/bin/env python3
# coding: utf-8

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import lmdb
import numpy as np
import torch


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def open_lmdb(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=path.is_dir(),
        max_readers=64,
    )


def build_gene_mappings(pathway_json: Path, lmdb_vocab_json: Path) -> Tuple[List[str], Dict[str, int], Dict[int, int]]:
    pathway_data = load_json(pathway_json)
    lmdb_vocab = load_json(lmdb_vocab_json)
    genes = pathway_data["pathway_genes_list"]
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    gene_to_lmdb_id = {gene: int(lmdb_vocab[gene]) for gene in genes}
    lmdb_id_to_target_idx = {lmdb_id: gene_to_idx[gene] for gene, lmdb_id in gene_to_lmdb_id.items()}
    return genes, gene_to_lmdb_id, lmdb_id_to_target_idx


def record_to_pathway_vector(record: Dict, lmdb_id_to_target_idx: Dict[int, int], num_genes: int) -> np.ndarray:
    vector = np.zeros(num_genes, dtype=np.float32)
    for lmdb_id, value in zip(record["gene_ids"], record["log1p_x"]):
        idx = lmdb_id_to_target_idx.get(lmdb_id)
        if idx is not None:
            vector[idx] = float(value)
    return vector


def iter_cell_keys(txn: lmdb.Transaction, max_cells: Optional[int] = None) -> Iterable[str]:
    cursor = txn.cursor()
    yielded = 0
    for key, _value in cursor:
        if key.startswith(b"-") or key in {b"__len__", b"num_samples"}:
            continue
        yield key.decode()
        yielded += 1
        if max_cells is not None and yielded >= max_cells:
            break


def load_cells_by_keys(
    lmdb_path: Path,
    cell_keys: List[str],
    lmdb_id_to_target_idx: Dict[int, int],
    num_genes: int,
) -> Tuple[torch.Tensor, List[Dict[str, str]]]:
    vectors: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []
    env = open_lmdb(lmdb_path)
    with env.begin(write=False) as txn:
        for cell_key in cell_keys:
            raw = txn.get(cell_key.encode())
            if raw is None:
                continue
            record = json.loads(raw)
            vectors.append(record_to_pathway_vector(record, lmdb_id_to_target_idx, num_genes))
            metadata.append(
                {
                    "cell_key": cell_key,
                    "cell_id": str(record.get("cell_id", "")),
                    "celltype_name": record.get("celltype_name", ""),
                    "tissue_name": record.get("tissue_name", ""),
                    "disease_name": record.get("disease_name", ""),
                }
            )
    env.close()
    return torch.from_numpy(np.stack(vectors, axis=0)).float(), metadata


def load_first_n_cells(
    lmdb_path: Path,
    lmdb_id_to_target_idx: Dict[int, int],
    num_genes: int,
    max_cells: int,
) -> Tuple[List[str], torch.Tensor, List[Dict[str, str]]]:
    cell_keys: List[str] = []
    vectors: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []
    env = open_lmdb(lmdb_path)
    with env.begin(write=False) as txn:
        for cell_key in iter_cell_keys(txn, max_cells=max_cells):
            raw = txn.get(cell_key.encode())
            if raw is None:
                continue
            record = json.loads(raw)
            cell_keys.append(cell_key)
            vectors.append(record_to_pathway_vector(record, lmdb_id_to_target_idx, num_genes))
            metadata.append(
                {
                    "cell_key": cell_key,
                    "cell_id": str(record.get("cell_id", "")),
                    "celltype_name": record.get("celltype_name", ""),
                    "tissue_name": record.get("tissue_name", ""),
                    "disease_name": record.get("disease_name", ""),
                }
            )
    env.close()
    return cell_keys, torch.from_numpy(np.stack(vectors, axis=0)).float(), metadata


def load_topk_json(topk_json: Path) -> Dict:
    return load_json(topk_json)


def collect_topk_cell_keys(topk_data: Dict) -> List[str]:
    keys = set()
    for items in topk_data["topk"].values():
        for item in items:
            keys.add(item["cell_key"])
    return sorted(keys)


def build_static_gene_embeddings_from_cell_features(
    topk_data: Dict,
    genes: List[str],
    cell_key_to_feature: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    hidden_dim = next(iter(cell_key_to_feature.values())).numel()
    sums = torch.zeros(len(genes), hidden_dim, dtype=torch.float32)
    counts = torch.zeros(len(genes), dtype=torch.int32)

    for gene, items in topk_data["topk"].items():
        row = gene_to_idx[gene]
        for item in items:
            feature = cell_key_to_feature.get(item["cell_key"])
            if feature is None:
                continue
            sums[row] += feature.float()
            counts[row] += 1

    denom = counts.clamp(min=1).unsqueeze(1).float()
    return sums / denom, counts


def build_static_prototypes_from_topk(
    topk_data: Dict,
    lmdb_path: Path,
    genes: List[str],
    lmdb_id_to_target_idx: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    sums = torch.zeros(len(genes), len(genes), dtype=torch.float32)
    counts = torch.zeros(len(genes), dtype=torch.int32)
    selected_by_cell: Dict[str, List[str]] = {}
    for gene, items in topk_data["topk"].items():
        for item in items:
            selected_by_cell.setdefault(item["cell_key"], []).append(gene)

    env = open_lmdb(lmdb_path)
    with env.begin(write=False) as txn:
        for cell_key, gene_list in selected_by_cell.items():
            raw = txn.get(cell_key.encode())
            if raw is None:
                continue
            record = json.loads(raw)
            vector = torch.from_numpy(record_to_pathway_vector(record, lmdb_id_to_target_idx, len(genes)))
            for gene in gene_list:
                idx = gene_to_idx[gene]
                sums[idx] += vector
                counts[idx] += 1
    env.close()
    denom = counts.clamp(min=1).unsqueeze(1).float()
    return sums / denom, counts


def collect_topk_cell_refs(topk_data: Dict) -> List[Tuple[str, str]]:
    refs = set()
    for items in topk_data["topk"].values():
        for item in items:
            refs.add((item.get("shard", ""), item["cell_key"]))
    return sorted(refs)


def make_cell_ref(shard: str, cell_key: str) -> str:
    return f"{shard}::{cell_key}"


def load_cells_by_refs(
    lmdb_root: Path,
    cell_refs: List[Tuple[str, str]],
    lmdb_id_to_target_idx: Dict[int, int],
    num_genes: int,
) -> Tuple[List[str], torch.Tensor, List[Dict[str, str]]]:
    by_shard: Dict[str, List[str]] = {}
    for shard, cell_key in cell_refs:
        by_shard.setdefault(shard, []).append(cell_key)

    ref_keys: List[str] = []
    vectors: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []
    for shard, keys in by_shard.items():
        env = open_lmdb(lmdb_root / shard)
        with env.begin(write=False) as txn:
            for cell_key in keys:
                raw = txn.get(cell_key.encode())
                if raw is None:
                    continue
                record = json.loads(raw)
                ref_key = make_cell_ref(shard, cell_key)
                ref_keys.append(ref_key)
                vectors.append(record_to_pathway_vector(record, lmdb_id_to_target_idx, num_genes))
                metadata.append({
                    "ref_key": ref_key,
                    "shard": shard,
                    "cell_key": cell_key,
                    "cell_id": str(record.get("cell_id", "")),
                    "celltype_name": record.get("celltype_name", ""),
                    "tissue_name": record.get("tissue_name", ""),
                    "disease_name": record.get("disease_name", ""),
                })
        env.close()
    return ref_keys, torch.from_numpy(np.stack(vectors, axis=0)).float(), metadata


def build_static_gene_embeddings_from_cell_features_refs(
    topk_data: Dict,
    genes: List[str],
    ref_to_feature: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    hidden_dim = next(iter(ref_to_feature.values())).numel()
    sums = torch.zeros(len(genes), hidden_dim, dtype=torch.float32)
    counts = torch.zeros(len(genes), dtype=torch.int32)

    for gene, items in topk_data["topk"].items():
        row = gene_to_idx[gene]
        for item in items:
            ref_key = make_cell_ref(item.get("shard", ""), item["cell_key"])
            feature = ref_to_feature.get(ref_key)
            if feature is None:
                continue
            sums[row] += feature.float()
            counts[row] += 1

    denom = counts.clamp(min=1).unsqueeze(1).float()
    return sums / denom, counts
