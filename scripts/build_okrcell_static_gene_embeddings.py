#!/usr/bin/env python
# coding=utf-8

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_INPUT_H5ADS = [
    "/home/qijinyin/wanghaoran/zxy/features/okrcell_sft_features_census.h5ad",
    "/home/qijinyin/wanghaoran/zxy/features/okrcell_sft_features_archs4.h5ad",
]

DEFAULT_REFERENCE_STATIC_CKPT = (
    "/data/bgi/data/projects/multimodal/zyh/scData/outputs/training_runs/"
    "scgene_qformer_compact_cellfeat_rankaux_top128_1layer_v1/"
    "scgene_qformer_cellfeat_checkpoint.pt"
)

DEFAULT_OUTPUT_PATH = (
    "/data/bgi/data/projects/multimodal/zyh/scData/outputs/static_gene_embeddings/"
    "okrcell_top100_static_gene_embeddings.pt"
)


@dataclass
class TopKItem:
    score: float
    tie_breaker: int
    cell_id: str
    dataset_name: str
    feature: np.ndarray

    def heap_entry(self):
        return (self.score, self.tie_breaker, self.cell_id, self.dataset_name, self.feature)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build static gene embeddings from OKR-cell H5AD features. "
            "For each target gene, keep the global top-K cells ranked by rank_log1p, "
            "then average their 768-d cell features."
        )
    )
    parser.add_argument(
        "--input-h5ad",
        nargs="+",
        default=DEFAULT_INPUT_H5ADS,
        help="One or more OKR-cell feature H5AD files.",
    )
    parser.add_argument(
        "--reference-static-ckpt",
        default=DEFAULT_REFERENCE_STATIC_CKPT,
        help=(
            "Existing static checkpoint used as the whitelist and gene->scGPT-id mapping "
            "(for example the current 2925-gene checkpoint)."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output .pt checkpoint path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top cells kept per gene.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of cells read per chunk from each H5AD.",
    )
    parser.add_argument(
        "--save-topk-json",
        default="",
        help="Optional JSON path for exporting per-gene top-K records.",
    )
    parser.add_argument(
        "--max-cells-per-file",
        type=int,
        default=0,
        help="Optional cap for smoke testing. Use 0 to scan the full file.",
    )
    return parser.parse_args()


def load_reference_gene_bundle(path: str) -> Dict:
    payload = torch.load(path, map_location="cpu")
    genes = list(payload["genes"])
    gene_to_scgpt_id = {str(g): int(v) for g, v in payload["gene_to_scgpt_id"].items()}
    missing_ids = [g for g in genes if g not in gene_to_scgpt_id]
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} genes missing scGPT ids in reference checkpoint.")

    scgpt_id_to_gene_idx = {gene_to_scgpt_id[g]: i for i, g in enumerate(genes)}
    return {
        "genes": genes,
        "gene_to_scgpt_id": gene_to_scgpt_id,
        "scgpt_id_to_gene_idx": scgpt_id_to_gene_idx,
    }


def update_gene_heap(
    heap: List[Tuple],
    score: float,
    tie_breaker: int,
    cell_id: str,
    dataset_name: str,
    feature_row: np.ndarray,
    top_k: int,
) -> None:
    key = (score, tie_breaker)
    if len(heap) < top_k or key > heap[0][:2]:
        item = TopKItem(
            score=score,
            tie_breaker=tie_breaker,
            cell_id=cell_id,
            dataset_name=dataset_name,
            feature=feature_row.copy(),
        )
        entry = item.heap_entry()
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
        else:
            heapq.heapreplace(heap, entry)


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def process_h5ad_file(
    h5ad_path: str,
    scgpt_id_to_gene_idx: Dict[int, int],
    top_k: int,
    chunk_size: int,
    heaps: List[List[Tuple]],
    tie_breaker_start: int,
    max_cells_per_file: int,
) -> Tuple[int, int]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    if "rank" not in adata.obsm or "rank_log1p" not in adata.obsm:
        raise ValueError(f"{h5ad_path} is missing obsm['rank'] or obsm['rank_log1p'].")
    if "cell_id" not in adata.obs.columns:
        raise ValueError(f"{h5ad_path} is missing obs['cell_id'].")

    dataset_name = Path(h5ad_path).stem
    total_cells = int(adata.n_obs)
    if max_cells_per_file > 0:
        total_cells = min(total_cells, int(max_cells_per_file))
    matched_events = 0
    tie_breaker = tie_breaker_start

    pbar = tqdm(
        range(0, total_cells, chunk_size),
        desc=f"scan {dataset_name}",
        unit="chunk",
    )

    for start in pbar:
        end = min(start + chunk_size, total_cells)

        rank_chunk = np.asarray(adata.obsm["rank"][start:end], dtype=np.int64)
        expr_chunk = np.asarray(adata.obsm["rank_log1p"][start:end], dtype=np.float32)
        feat_chunk = np.asarray(adata.X[start:end], dtype=np.float16)
        cell_ids = adata.obs["cell_id"].iloc[start:end].tolist()

        local_hits = 0
        for row_idx in range(rank_chunk.shape[0]):
            rank_row = rank_chunk[row_idx]
            expr_row = expr_chunk[row_idx]
            feature_row = feat_chunk[row_idx]
            cell_id = str(cell_ids[row_idx])

            for pos in range(rank_row.shape[0]):
                gene_idx = scgpt_id_to_gene_idx.get(int(rank_row[pos]))
                if gene_idx is None:
                    continue
                score = float(expr_row[pos])
                update_gene_heap(
                    heaps[gene_idx],
                    score=score,
                    tie_breaker=tie_breaker,
                    cell_id=cell_id,
                    dataset_name=dataset_name,
                    feature_row=feature_row,
                    top_k=top_k,
                )
                tie_breaker += 1
                matched_events += 1
                local_hits += 1

        pbar.set_postfix(cells=end, matched=matched_events, local_hits=local_hits)

    return matched_events, tie_breaker


def finalize_outputs(
    genes: List[str],
    heaps: List[List[Tuple]],
    top_k: int,
) -> Tuple[torch.Tensor, List[int], Dict[str, List[Dict[str, object]]]]:
    hidden_dim = 768
    static_gene_embeddings = torch.zeros(len(genes), hidden_dim, dtype=torch.float32)
    topk_counts: List[int] = []
    topk_records: Dict[str, List[Dict[str, object]]] = {}

    for gene_idx, gene_name in enumerate(genes):
        entries = sorted(heaps[gene_idx], key=lambda x: (x[0], x[1]), reverse=True)
        topk_counts.append(len(entries))

        if entries:
            features = np.stack([entry[4] for entry in entries], axis=0).astype(np.float32)
            static_gene_embeddings[gene_idx] = torch.from_numpy(features.mean(axis=0))

        topk_records[gene_name] = [
            {
                "score": float(entry[0]),
                "cell_id": str(entry[2]),
                "dataset_name": str(entry[3]),
            }
            for entry in entries[:top_k]
        ]

    return static_gene_embeddings, topk_counts, topk_records


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    bundle = load_reference_gene_bundle(args.reference_static_ckpt)
    genes = bundle["genes"]
    gene_to_scgpt_id = bundle["gene_to_scgpt_id"]
    scgpt_id_to_gene_idx = bundle["scgpt_id_to_gene_idx"]

    print("Loaded reference gene bundle")
    print(f"  genes: {len(genes)}")
    print(f"  mapped scGPT ids: {len(scgpt_id_to_gene_idx)}")
    print(f"  top_k: {args.top_k}")
    print(f"  chunk_size: {args.chunk_size}")

    heaps: List[List[Tuple]] = [[] for _ in genes]
    total_matched_events = 0
    tie_breaker = 0

    for h5ad_path in args.input_h5ad:
        print(f"\nProcessing {h5ad_path}")
        matched_events, tie_breaker = process_h5ad_file(
            h5ad_path=h5ad_path,
            scgpt_id_to_gene_idx=scgpt_id_to_gene_idx,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            heaps=heaps,
            tie_breaker_start=tie_breaker,
            max_cells_per_file=args.max_cells_per_file,
        )
        total_matched_events += matched_events

    static_gene_embeddings, topk_counts, topk_records = finalize_outputs(
        genes=genes,
        heaps=heaps,
        top_k=args.top_k,
    )

    covered_genes = sum(1 for count in topk_counts if count > 0)
    full_genes = sum(1 for count in topk_counts if count >= args.top_k)

    payload = {
        "model_class": "static_gene_embedding_from_okrcell_topk",
        "genes": genes,
        "gene_to_scgpt_id": gene_to_scgpt_id,
        "static_gene_embeddings_768d": static_gene_embeddings,
        "topk_counts": topk_counts,
        "config": {
            "input_h5ad": list(args.input_h5ad),
            "reference_static_ckpt": args.reference_static_ckpt,
            "top_k": int(args.top_k),
            "chunk_size": int(args.chunk_size),
            "feature_source": "adata.X",
            "gene_rank_source": "adata.obsm['rank']",
            "gene_score_source": "adata.obsm['rank_log1p']",
            "selection_rule": (
                "For each gene, keep global top-K cells by rank_log1p among appearances "
                "inside per-cell top-1200 ranked genes, then average their 768-d cell features."
            ),
        },
        "summary": {
            "num_genes": len(genes),
            "covered_genes": covered_genes,
            "full_topk_genes": full_genes,
            "total_matched_events": int(total_matched_events),
        },
    }

    ensure_parent_dir(args.output)
    torch.save(payload, args.output)

    print("\nSaved static gene embedding checkpoint")
    print(f"  output: {args.output}")
    print(f"  embedding_shape: {tuple(static_gene_embeddings.shape)}")
    print(f"  covered_genes: {covered_genes}/{len(genes)}")
    print(f"  full_topk_genes: {full_genes}/{len(genes)}")
    print(f"  min_count: {min(topk_counts) if topk_counts else 0}")
    print(f"  max_count: {max(topk_counts) if topk_counts else 0}")

    if args.save_topk_json:
        ensure_parent_dir(args.save_topk_json)
        with open(args.save_topk_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "top_k": int(args.top_k),
                    "num_genes": len(genes),
                    "topk_counts": topk_counts,
                    "gene_topk_records": topk_records,
                },
                f,
                ensure_ascii=False,
            )
        print(f"  topk_json: {args.save_topk_json}")


if __name__ == "__main__":
    main()
