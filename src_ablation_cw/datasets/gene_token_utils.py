# coding=utf-8
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


@lru_cache(maxsize=4)
def load_static_gene_bundle(ckpt_path: str):
    payload = torch.load(str(ckpt_path), map_location="cpu")
    genes = list(payload["genes"])
    static_gene_embeddings = payload["static_gene_embeddings_768d"].float().cpu()
    gene_to_scgpt_id = {g: int(v) for g, v in payload["gene_to_scgpt_id"].items()}
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    scgpt_id_to_idx = {
        int(gene_to_scgpt_id[g]): gene_to_idx[g]
        for g in genes
        if g in gene_to_scgpt_id
    }
    return {
        "genes": genes,
        "static_gene_embeddings": static_gene_embeddings,
        "gene_to_scgpt_id": gene_to_scgpt_id,
        "gene_to_idx": gene_to_idx,
        "scgpt_id_to_idx": scgpt_id_to_idx,
    }


def build_gene_sequence_from_rank(
    rank_ids: np.ndarray,
    scgpt_id_to_idx: Dict[int, int],
    static_gene_embeddings: torch.Tensor,
    gene_input_tokens: int,
    rank_order: str = "asc",
) -> torch.Tensor:
    rank_ids = np.asarray(rank_ids).reshape(-1)
    if rank_order == "asc":
        ordered_ids = rank_ids[::-1]
    else:
        ordered_ids = rank_ids

    selected_gene_indices: List[int] = []
    for gid in ordered_ids:
        idx = scgpt_id_to_idx.get(int(gid))
        if idx is None:
            continue
        selected_gene_indices.append(idx)
        if len(selected_gene_indices) >= gene_input_tokens:
            break

    hidden_dim = int(static_gene_embeddings.shape[1])
    out = torch.zeros(gene_input_tokens, hidden_dim, dtype=torch.float32)
    if selected_gene_indices:
        picked = static_gene_embeddings[selected_gene_indices]
        out[: picked.shape[0]] = picked
    return out


def build_pathway_embeddings_from_static_gene_ckpt(
    pathway_json_path: str,
    static_gene_ckpt_path: str,
    num_queries: int,
) -> Tuple[List[str], torch.Tensor, List[int]]:
    pathway_payload = json.loads(Path(pathway_json_path).read_text())
    bundle = load_static_gene_bundle(static_gene_ckpt_path)
    genes = bundle["genes"]
    static_gene_embeddings = bundle["static_gene_embeddings"]
    gene_to_idx = bundle["gene_to_idx"]

    pathway_to_genes = pathway_payload["pathway_to_genes"]
    pathway_names: List[str] = []
    pathway_embeddings: List[torch.Tensor] = []
    pathway_gene_counts: List[int] = []

    for pathway_name, pathway_genes in list(pathway_to_genes.items())[:num_queries]:
        idxs = [gene_to_idx[g] for g in pathway_genes if g in gene_to_idx]
        if idxs:
            pathway_vec = static_gene_embeddings[idxs].mean(dim=0)
            pathway_gene_counts.append(len(idxs))
        else:
            pathway_vec = torch.zeros(static_gene_embeddings.shape[1], dtype=torch.float32)
            pathway_gene_counts.append(0)
        pathway_names.append(pathway_name)
        pathway_embeddings.append(pathway_vec)

    if len(pathway_embeddings) != num_queries:
        raise ValueError(f"Expected {num_queries} pathways, got {len(pathway_embeddings)} from {pathway_json_path}")

    return pathway_names, torch.stack(pathway_embeddings, dim=0), pathway_gene_counts
