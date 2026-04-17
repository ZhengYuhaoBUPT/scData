#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import lmdb
import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-shard Q-Former-style prototype from per-shard topk results."
    )
    parser.add_argument(
        "--topk-json",
        type=str,
        default="/data/bgi/data/projects/multimodal/zyh/scData/outputs/pathway_static_pipeline/per_shard_topk/split_task1_writer1.db.topk.json",
    )
    parser.add_argument(
        "--lmdb-path",
        type=str,
        default="/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText/split_task1_writer1.db",
    )
    parser.add_argument(
        "--pathway-json",
        type=str,
        default="/data/bgi/data/projects/multimodal/zyh/datasets/pathway/pathway_anchor_genes.json",
    )
    parser.add_argument(
        "--lmdb-vocab",
        type=str,
        default="/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText/config/lmdb_vocab.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/bgi/data/projects/multimodal/zyh/scData/outputs/qformer_shard_prototype",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def build_gene_mappings(pathway_data: Dict, lmdb_vocab: Dict) -> Tuple[List[str], Dict[str, int], Dict[int, int]]:
    genes = pathway_data["pathway_genes_list"]
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    gene_to_lmdb_id = {gene: int(lmdb_vocab[gene]) for gene in genes}
    lmdb_id_to_target_idx = {lmdb_id: gene_to_idx[gene] for gene, lmdb_id in gene_to_lmdb_id.items()}
    return genes, gene_to_lmdb_id, lmdb_id_to_target_idx


def to_target_vector(record: Dict, lmdb_id_to_target_idx: Dict[int, int], n_genes: int) -> np.ndarray:
    vec = np.zeros(n_genes, dtype=np.float32)
    for lmdb_id, value in zip(record["gene_ids"], record["log1p_x"]):
        idx = lmdb_id_to_target_idx.get(lmdb_id)
        if idx is not None:
            vec[idx] = float(value)
    return vec


def build_static_prototypes(
    topk_data: Dict,
    env: lmdb.Environment,
    genes: List[str],
    lmdb_id_to_target_idx: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    n_genes = len(genes)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    selected_by_cell: Dict[str, List[str]] = defaultdict(list)
    for gene, items in topk_data["topk"].items():
        for item in items:
            selected_by_cell[item["cell_key"]].append(gene)

    sums = np.zeros((n_genes, n_genes), dtype=np.float32)
    counts = np.zeros(n_genes, dtype=np.int32)

    with env.begin(write=False) as txn:
        for cell_key, gene_list in selected_by_cell.items():
            raw = txn.get(cell_key.encode())
            if raw is None:
                continue
            record = json.loads(raw)
            vec = to_target_vector(record, lmdb_id_to_target_idx, n_genes)
            for gene in gene_list:
                idx = gene_to_idx[gene]
                sums[idx] += vec
                counts[idx] += 1

    prototypes = sums / np.maximum(counts[:, None], 1)
    return prototypes, counts


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, key_value, key_value, need_weights=False)
        query = self.norm1(query + attn_out)
        ffn_out = self.ffn(query)
        return self.norm2(query + ffn_out)


class PathwayQFormerPrototype(nn.Module):
    def __init__(self, num_genes: int, hidden_dim: int, num_queries: int, num_heads: int, query_init: torch.Tensor):
        super().__init__()
        self.static_projector = nn.Linear(num_genes, hidden_dim)
        self.expr_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_tokens = nn.Parameter(query_init.clone())
        self.block = CrossAttentionBlock(hidden_dim, num_heads)

    def forward(self, static_prototypes: torch.Tensor, cell_expr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        static_emb = self.static_projector(static_prototypes)
        expr_feat = self.expr_mlp(cell_expr.unsqueeze(-1))
        dynamic_tokens = static_emb.unsqueeze(0) + expr_feat
        queries = self.query_tokens.unsqueeze(0).expand(cell_expr.size(0), -1, -1)
        query_out = self.block(queries, dynamic_tokens)
        return static_emb, query_out


def iter_cell_keys(txn: lmdb.Transaction, max_cells: int) -> Iterable[str]:
    cursor = txn.cursor()
    yielded = 0
    for key, _value in cursor:
        if key.startswith(b"-") or key in {b"__len__", b"num_samples"}:
            continue
        yield key.decode()
        yielded += 1
        if yielded >= max_cells:
            break


def build_cell_batch(
    env: lmdb.Environment,
    lmdb_id_to_target_idx: Dict[int, int],
    n_genes: int,
    max_cells: int,
) -> Tuple[List[str], torch.Tensor, List[Dict[str, str]]]:
    cell_keys: List[str] = []
    vectors: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []
    with env.begin(write=False) as txn:
        for cell_key in iter_cell_keys(txn, max_cells=max_cells):
            raw = txn.get(cell_key.encode())
            if raw is None:
                continue
            record = json.loads(raw)
            cell_keys.append(cell_key)
            vectors.append(to_target_vector(record, lmdb_id_to_target_idx, n_genes))
            metadata.append(
                {
                    "cell_key": cell_key,
                    "cell_id": str(record.get("cell_id", "")),
                    "celltype_name": record.get("celltype_name", ""),
                    "tissue_name": record.get("tissue_name", ""),
                    "disease_name": record.get("disease_name", ""),
                }
            )
    return cell_keys, torch.from_numpy(np.stack(vectors, axis=0)), metadata


def make_query_init(pathway_data: Dict, genes: List[str], static_prototypes: torch.Tensor, hidden_dim: int) -> Tuple[List[str], torch.Tensor]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    pathway_names = list(pathway_data["pathway_to_genes"].keys())
    pathway_vectors = []
    for pathway in pathway_names:
        members = [gene_to_idx[g] for g in pathway_data["pathway_to_genes"][pathway] if g in gene_to_idx]
        if members:
            pathway_vectors.append(static_prototypes[members].mean(dim=0))
        else:
            pathway_vectors.append(torch.zeros(static_prototypes.size(1), dtype=static_prototypes.dtype))

    query_base = torch.stack(pathway_vectors, dim=0)
    query_projector = nn.Linear(query_base.size(1), hidden_dim)
    nn.init.xavier_uniform_(query_projector.weight)
    nn.init.zeros_(query_projector.bias)
    query_projector = query_projector.to(dtype=query_base.dtype)
    return pathway_names, query_projector(query_base)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topk_data = load_json(Path(args.topk_json))
    pathway_data = load_json(Path(args.pathway_json))
    lmdb_vocab = load_json(Path(args.lmdb_vocab))
    genes, gene_to_lmdb_id, lmdb_id_to_target_idx = build_gene_mappings(pathway_data, lmdb_vocab)

    env = open_lmdb(Path(args.lmdb_path))
    prototypes_np, topk_counts = build_static_prototypes(topk_data, env, genes, lmdb_id_to_target_idx)
    prototypes = torch.from_numpy(prototypes_np).float()

    pathway_names, query_init = make_query_init(pathway_data, genes, prototypes, args.hidden_dim)
    model = PathwayQFormerPrototype(
        num_genes=len(genes),
        hidden_dim=args.hidden_dim,
        num_queries=len(pathway_names),
        num_heads=args.num_heads,
        query_init=query_init,
    )
    model.eval()

    cell_keys, cell_expr, cell_metadata = build_cell_batch(
        env=env,
        lmdb_id_to_target_idx=lmdb_id_to_target_idx,
        n_genes=len(genes),
        max_cells=args.max_cells,
    )
    cell_expr = cell_expr.float()
    env.close()

    static_emb_chunks = []
    query_chunks = []
    with torch.no_grad():
        for start in range(0, cell_expr.size(0), args.batch_size):
            batch_expr = cell_expr[start:start + args.batch_size]
            static_emb, query_out = model(prototypes, batch_expr)
            if not static_emb_chunks:
                static_emb_chunks.append(static_emb.cpu())
            query_chunks.append(query_out.cpu())

    query_tokens = torch.cat(query_chunks, dim=0)
    static_emb = static_emb_chunks[0]

    torch.save(
        {
            "genes": genes,
            "gene_to_lmdb_id": gene_to_lmdb_id,
            "pathway_names": pathway_names,
            "topk_counts": torch.from_numpy(topk_counts),
            "static_prototypes_4366d": prototypes,
            "static_gene_embeddings": static_emb,
            "cell_keys": cell_keys,
            "cell_metadata": cell_metadata,
            "cell_query_tokens": query_tokens,
            "sample_ratio": topk_data.get("sample_ratio"),
            "source_topk_json": args.topk_json,
            "source_lmdb_path": args.lmdb_path,
        },
        output_dir / "qformer_shard_prototype.pt",
    )

    with (output_dir / "qformer_shard_prototype_metadata.json").open("w") as f:
        json.dump(
            {
                "num_genes": len(genes),
                "num_pathways": len(pathway_names),
                "hidden_dim": args.hidden_dim,
                "max_cells": len(cell_keys),
                "cell_query_tokens_shape": list(query_tokens.shape),
                "static_gene_embeddings_shape": list(static_emb.shape),
                "static_prototypes_shape": list(prototypes.shape),
                "sample_ratio": topk_data.get("sample_ratio"),
                "source_topk_json": args.topk_json,
                "source_lmdb_path": args.lmdb_path,
                "note": "This is an untrained Q-Former-style prototype for tensor construction and shape validation.",
            },
            f,
            indent=2,
        )

    print(f"Saved prototype bundle: {output_dir / 'qformer_shard_prototype.pt'}")
    print(f"Static prototypes shape: {tuple(prototypes.shape)}")
    print(f"Static gene embeddings shape: {tuple(static_emb.shape)}")
    print(f"Cell query tokens shape: {tuple(query_tokens.shape)}")


if __name__ == "__main__":
    main()
