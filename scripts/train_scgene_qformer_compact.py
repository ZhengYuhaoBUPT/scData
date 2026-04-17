#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import lmdb
import msgpack
import numpy as np
import torch

PROJECT_ROOT = Path("/data/bgi/data/projects/multimodal/zyh/scData")
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scgeneqformer.data.lmdb_dataset import build_static_gene_embeddings_from_cell_features_refs, collect_topk_cell_refs, load_json, load_topk_json
from scgeneqformer.models.cell_encoder import encode_pathway_vectors_to_cell_features, load_cell_encoder
from scgeneqformer.models.gene_qformer import GeneQFormerModel
from scgeneqformer.train.trainer import run_reconstruction_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scGeneQFormer on compact 8-shard scGPT-id LMDB data.")
    parser.add_argument("--topk-json", type=str, default=str(PROJECT_ROOT / "outputs/scgpt_8shards_topk/merged_topk.json"))
    parser.add_argument("--lmdb-root", type=str, default="/home/qijinyin/wanghaoran/zxy/features/per_gene_feat/whitelist_lmdb_8shards_compact")
    parser.add_argument("--pathway-json", type=str, default="/data/bgi/data/projects/multimodal/zyh/datasets/pathway/pathway_anchor_genes.json")
    parser.add_argument("--scgpt-path", type=str, default="/root/wanghaoran/zxy/project/sc_showo/scgpt")
    parser.add_argument("--encoder-model-path", type=str, default="/root/wanghaoran/zxy/project/sc_showo/save/okrcell_ckpt/model-241492.pt")
    parser.add_argument("--encoder-vocab-path", type=str, default="/data/bgi/data/projects/multimodal/zyh/datasets/CellwText/scgpt/vocab.json")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs/training_runs/scgene_qformer_compact_run1"))
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-train-cells", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def iter_cell_keys(txn: lmdb.Transaction) -> Iterable[str]:
    cursor = txn.cursor()
    for key, _value in cursor:
        if key.startswith(b"-") or key in {b"__len__", b"num_samples"}:
            continue
        yield key.decode()


def decode_record(raw: bytes) -> Dict:
    return msgpack.unpackb(raw, raw=False)


def record_to_pathway_vector(record: Dict, scgpt_id_to_target_idx: Dict[int, int], num_genes: int) -> np.ndarray:
    vector = np.zeros(num_genes, dtype=np.float32)
    for scgpt_id, value in zip(record["scgpt_ids"], record["log1p_x"]):
        idx = scgpt_id_to_target_idx.get(int(scgpt_id))
        if idx is not None:
            vector[idx] = float(value)
    return vector


def load_cells_by_refs_compact(
    lmdb_root: Path,
    cell_refs: List[Tuple[str, str]],
    scgpt_id_to_target_idx: Dict[int, int],
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
                record = decode_record(raw)
                ref_key = f"{shard}::{cell_key}"
                ref_keys.append(ref_key)
                vectors.append(record_to_pathway_vector(record, scgpt_id_to_target_idx, num_genes))
                metadata.append({"ref_key": ref_key, "shard": shard, "cell_key": cell_key})
        env.close()
    return ref_keys, torch.from_numpy(np.stack(vectors, axis=0)).float(), metadata


def load_first_n_cells_from_root_compact(
    lmdb_root: Path,
    scgpt_id_to_target_idx: Dict[int, int],
    num_genes: int,
    max_cells: int,
) -> Tuple[List[str], torch.Tensor, List[Dict[str, str]]]:
    ref_keys: List[str] = []
    vectors: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []
    for shard_path in sorted(lmdb_root.glob('shard_*.db')):
        env = open_lmdb(shard_path)
        with env.begin(write=False) as txn:
            for cell_key in iter_cell_keys(txn):
                raw = txn.get(cell_key.encode())
                if raw is None:
                    continue
                record = decode_record(raw)
                ref_key = f"{shard_path.name}::{cell_key}"
                ref_keys.append(ref_key)
                vectors.append(record_to_pathway_vector(record, scgpt_id_to_target_idx, num_genes))
                metadata.append({"ref_key": ref_key, "shard": shard_path.name, "cell_key": cell_key})
                if len(ref_keys) >= max_cells:
                    env.close()
                    return ref_keys, torch.from_numpy(np.stack(vectors, axis=0)).float(), metadata
        env.close()
    return ref_keys, torch.from_numpy(np.stack(vectors, axis=0)).float(), metadata


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topk_data = load_topk_json(Path(args.topk_json))
    pathway_data = load_json(Path(args.pathway_json))
    scgpt_vocab = load_json(Path(args.encoder_vocab_path))

    all_genes = pathway_data["pathway_genes_list"]
    zero_topk_genes = [gene for gene in all_genes if len(topk_data["topk"].get(gene, [])) == 0]
    missing_scgpt_genes = [gene for gene in all_genes if gene not in scgpt_vocab]
    genes = [gene for gene in all_genes if len(topk_data["topk"].get(gene, [])) > 0 and gene in scgpt_vocab]
    dropped_genes = sorted(set(zero_topk_genes) | set(missing_scgpt_genes))
    topk_data = {**topk_data, "topk": {gene: topk_data["topk"][gene] for gene in genes}}

    gene_to_scgpt_id = {gene: int(scgpt_vocab[gene]) for gene in genes}
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    scgpt_id_to_target_idx = {scgpt_id: gene_to_idx[gene] for gene, scgpt_id in gene_to_scgpt_id.items()}
    num_genes = len(genes)

    print(f"Keeping {num_genes}/{len(all_genes)} genes for training")
    print(f"Dropped zero-topk genes: {len(zero_topk_genes)}")
    print(f"Dropped missing-scgpt genes: {len(missing_scgpt_genes)} -> {missing_scgpt_genes[:20]}")

    gene_encoder, projection_head, _loaded_vocab, binning_fn = load_cell_encoder(
        model_path=args.encoder_model_path,
        vocab_path=args.encoder_vocab_path,
        scgpt_path=args.scgpt_path,
        device=device,
    )

    topk_cell_refs = collect_topk_cell_refs(topk_data)
    topk_ref_keys, topk_expr, _topk_metadata = load_cells_by_refs_compact(
        lmdb_root=Path(args.lmdb_root),
        cell_refs=topk_cell_refs,
        scgpt_id_to_target_idx=scgpt_id_to_target_idx,
        num_genes=num_genes,
    )
    topk_features = encode_pathway_vectors_to_cell_features(
        pathway_expr=topk_expr,
        genes=genes,
        gene_to_scgpt_id=gene_to_scgpt_id,
        gene_encoder=gene_encoder,
        projection_head=projection_head,
        binning_fn=binning_fn,
        device=device,
        batch_size=args.encoder_batch_size,
    )
    ref_to_feature = {ref_key: topk_features[i] for i, ref_key in enumerate(topk_ref_keys)}
    static_gene_embeddings, topk_counts = build_static_gene_embeddings_from_cell_features_refs(topk_data, genes, ref_to_feature)

    train_cell_keys, train_expr, train_metadata = load_first_n_cells_from_root_compact(
        lmdb_root=Path(args.lmdb_root),
        scgpt_id_to_target_idx=scgpt_id_to_target_idx,
        num_genes=num_genes,
        max_cells=args.max_train_cells,
    )

    model = GeneQFormerModel(
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_genes=num_genes,
    )
    train_result = run_reconstruction_training(
        model=model,
        static_gene_embeddings=static_gene_embeddings,
        cell_expr=train_expr,
        num_epochs=args.num_epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        device=device,
        max_steps=args.max_steps,
    )

    model = model.to(device).eval()
    with torch.no_grad():
        query_tokens, recon = model(static_gene_embeddings.to(device), train_expr.to(device))
    query_tokens = query_tokens.cpu()
    recon = recon.cpu()

    torch.save(
        {
            "genes": genes,
            "gene_to_scgpt_id": gene_to_scgpt_id,
            "topk_counts": topk_counts,
            "static_gene_embeddings_768d": static_gene_embeddings,
            "train_cell_keys": train_cell_keys,
            "train_cell_metadata": train_metadata,
            "query_tokens": query_tokens,
            "reconstructed_expr": recon,
            "train_history": train_result["history"],
            "step_history": train_result.get("step_history", []),
            "global_step": train_result.get("global_step"),
            "config": vars(args),
        },
        output_dir / "scgene_qformer_run.pt",
    )
    with (output_dir / "scgene_qformer_run_metadata.json").open("w") as f:
        json.dump(
            {
                "num_genes": num_genes,
                "num_topk_cells": len(topk_ref_keys),
                "num_train_cells": len(train_cell_keys),
                "static_gene_embeddings_shape": list(static_gene_embeddings.shape),
                "query_tokens_shape": list(query_tokens.shape),
                "reconstructed_expr_shape": list(recon.shape),
                "train_history": train_result["history"],
                "num_step_records": len(train_result.get("step_history", [])),
                "global_step": train_result.get("global_step"),
                "source_topk_json": args.topk_json,
                "source_lmdb_root": args.lmdb_root,
                "num_dropped_zero_topk_genes": len(zero_topk_genes),
                "dropped_zero_topk_genes": zero_topk_genes,
                "num_dropped_missing_scgpt_genes": len(missing_scgpt_genes),
                "dropped_missing_scgpt_genes": missing_scgpt_genes,
                "num_dropped_total_genes": len(dropped_genes),
            },
            f,
            indent=2,
        )
    print(f"Saved training bundle to {output_dir}")
    print(f"Static gene embeddings shape: {tuple(static_gene_embeddings.shape)}")
    print(f"Query tokens shape: {tuple(query_tokens.shape)}")


if __name__ == "__main__":
    main()
