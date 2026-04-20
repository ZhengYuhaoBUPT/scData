#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from scgeneqformer.models.gene_qformer import PathwayCellFeatureQFormer
from scgeneqformer.train.trainer import run_cell_feature_training_with_rank_aux


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cell-feature-input pathway Q-Former with rank auxiliary loss on compact 8-shard data.")
    parser.add_argument("--topk-json", type=str, default=str(PROJECT_ROOT / "outputs/scgpt_8shards_topk/merged_topk.json"))
    parser.add_argument("--lmdb-root", type=str, default="/home/qijinyin/wanghaoran/zxy/features/per_gene_feat/whitelist_lmdb_8shards_compact")
    parser.add_argument("--pathway-json", type=str, default="/data/bgi/data/projects/multimodal/zyh/datasets/pathway/pathway_anchor_genes.json")
    parser.add_argument("--scgpt-path", type=str, default="/root/wanghaoran/zxy/project/sc_showo/scgpt")
    parser.add_argument("--encoder-model-path", type=str, default="/root/wanghaoran/zxy/project/sc_showo/save/okrcell_ckpt/model-241492.pt")
    parser.add_argument("--encoder-vocab-path", type=str, default="/data/bgi/data/projects/multimodal/zyh/datasets/CellwText/scgpt/vocab.json")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs/training_runs/scgene_qformer_compact_cellfeat_run1"))
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--top-rank-genes", type=int, default=256)
    parser.add_argument("--rank-loss-weight", type=float, default=0.2)
    parser.add_argument("--max-train-cells", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def open_lmdb(path: Path) -> lmdb.Environment:
    return lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False, subdir=path.is_dir(), max_readers=64)


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


def load_cells_by_refs_compact(lmdb_root: Path, cell_refs: List[Tuple[str, str]], scgpt_id_to_target_idx: Dict[int, int], num_genes: int):
    by_shard: Dict[str, List[str]] = {}
    for shard, cell_key in cell_refs:
        by_shard.setdefault(shard, []).append(cell_key)
    ref_keys, vectors, metadata = [], [], []
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


def load_first_n_cells_from_root_compact(lmdb_root: Path, scgpt_id_to_target_idx: Dict[int, int], num_genes: int, max_cells: int):
    ref_keys, vectors, metadata = [], [], []
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
                    return ref_keys, torch.from_numpy(np.stack(vectors, axis=0)).float(), metadata
        env.close()
    return ref_keys, torch.from_numpy(np.stack(vectors, axis=0)).float(), metadata


def build_pathway_embeddings(pathway_to_genes: Dict[str, List[str]], kept_genes: List[str], static_gene_embeddings: torch.Tensor):
    gene_to_idx = {gene: idx for idx, gene in enumerate(kept_genes)}
    pathway_names, pathway_embeddings, pathway_gene_counts = [], [], []
    for pathway_name, genes in pathway_to_genes.items():
        idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if not idxs:
            continue
        pathway_names.append(pathway_name)
        pathway_gene_counts.append(len(idxs))
        pathway_embeddings.append(static_gene_embeddings[idxs].mean(dim=0))
    return pathway_names, torch.stack(pathway_embeddings, dim=0), pathway_gene_counts


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
    topk_ref_keys, topk_expr, _topk_metadata = load_cells_by_refs_compact(Path(args.lmdb_root), topk_cell_refs, scgpt_id_to_target_idx, num_genes)
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

    pathway_names, pathway_embeddings, pathway_gene_counts = build_pathway_embeddings(
        pathway_data["pathway_to_genes"],
        genes,
        static_gene_embeddings,
    )
    print(f"Built {len(pathway_names)} pathway embeddings")

    train_cell_keys, train_expr, train_metadata = load_first_n_cells_from_root_compact(Path(args.lmdb_root), scgpt_id_to_target_idx, num_genes, args.max_train_cells)
    train_cell_features = encode_pathway_vectors_to_cell_features(
        pathway_expr=train_expr,
        genes=genes,
        gene_to_scgpt_id=gene_to_scgpt_id,
        gene_encoder=gene_encoder,
        projection_head=projection_head,
        binning_fn=binning_fn,
        device=device,
        batch_size=args.encoder_batch_size,
    )

    model = PathwayCellFeatureQFormer(
        hidden_dim=args.hidden_dim,
        num_queries=len(pathway_names),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        out_dim=args.hidden_dim,
    )
    train_result = run_cell_feature_training_with_rank_aux(
        model=model,
        pathway_embeddings=pathway_embeddings,
        cell_features=train_cell_features,
        cell_expr=train_expr,
        num_epochs=args.num_epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        device=device,
        max_steps=args.max_steps,
        rank_topk=args.top_rank_genes,
        rank_loss_weight=args.rank_loss_weight,
    )

    model = model.to(device).eval()
    with torch.no_grad():
        query_tokens, recon = model(pathway_embeddings.to(device), train_cell_features.to(device))
    query_tokens = query_tokens.cpu()
    recon = recon.cpu()

    train_bundle = {
        "genes": genes,
        "pathway_names": pathway_names,
        "pathway_gene_counts": pathway_gene_counts,
        "gene_to_scgpt_id": gene_to_scgpt_id,
        "topk_counts": topk_counts,
        "static_gene_embeddings_768d": static_gene_embeddings,
        "pathway_embeddings_768d": pathway_embeddings,
        "train_cell_keys": train_cell_keys,
        "train_cell_metadata": train_metadata,
        "train_cell_features_768d": train_cell_features,
        "train_cell_expr": train_expr,
        "query_tokens": query_tokens,
        "reconstructed_cell_features": recon,
        "train_history": train_result["history"],
        "step_history": train_result.get("step_history", []),
        "global_step": train_result.get("global_step"),
        "config": vars(args),
    }
    torch.save(train_bundle, output_dir / "scgene_qformer_cellfeat_run.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": type(model).__name__,
        "config": vars(args),
        "genes": genes,
        "pathway_names": pathway_names,
        "pathway_gene_counts": pathway_gene_counts,
        "gene_to_scgpt_id": gene_to_scgpt_id,
        "topk_counts": topk_counts,
        "static_gene_embeddings_768d": static_gene_embeddings,
        "pathway_embeddings_768d": pathway_embeddings,
        "global_step": train_result.get("global_step"),
        "train_history": train_result["history"],
        "rank_aux_head_state_dict": train_result.get("rank_aux_head_state_dict"),
    }
    torch.save(checkpoint, output_dir / "scgene_qformer_cellfeat_checkpoint.pt")

    lightweight_checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": type(model).__name__,
        "config": vars(args),
        "pathway_embeddings_768d": pathway_embeddings,
        "pathway_names": pathway_names,
        "pathway_gene_counts": pathway_gene_counts,
        "global_step": train_result.get("global_step"),
    }
    torch.save(lightweight_checkpoint, output_dir / "scgene_qformer_cellfeat_checkpoint_light.pt")
    with (output_dir / "scgene_qformer_cellfeat_run_metadata.json").open("w") as f:
        json.dump(
            {
                "num_genes": num_genes,
                "num_pathways": len(pathway_names),
                "num_topk_cells": len(topk_ref_keys),
                "num_train_cells": len(train_cell_keys),
                "top_rank_genes": args.top_rank_genes,
                "rank_loss_weight": args.rank_loss_weight,
                "static_gene_embeddings_shape": list(static_gene_embeddings.shape),
                "pathway_embeddings_shape": list(pathway_embeddings.shape),
                "train_cell_features_shape": list(train_cell_features.shape),
                "train_cell_expr_shape": list(train_expr.shape),
                "query_tokens_shape": list(query_tokens.shape),
                "reconstructed_cell_features_shape": list(recon.shape),
                "train_history": train_result["history"],
                "num_step_records": len(train_result.get("step_history", [])),
                "global_step": train_result.get("global_step"),
                "source_topk_json": args.topk_json,
                "source_lmdb_root": args.lmdb_root,
                "num_dropped_zero_topk_genes": len(zero_topk_genes),
                "num_dropped_missing_scgpt_genes": len(missing_scgpt_genes),
                "dropped_missing_scgpt_genes": missing_scgpt_genes,
            },
            f,
            indent=2,
        )
    print(f"Saved training bundle to {output_dir}")
    print(f"Saved checkpoint to {output_dir / 'scgene_qformer_cellfeat_checkpoint.pt'}")
    print(f"Saved lightweight checkpoint to {output_dir / 'scgene_qformer_cellfeat_checkpoint_light.pt'}")
    print(f"Pathway embeddings shape: {tuple(pathway_embeddings.shape)}")
    print(f"Train cell features shape: {tuple(train_cell_features.shape)}")
    print(f"Train cell expr shape: {tuple(train_expr.shape)}")
    print(f"Query tokens shape: {tuple(query_tokens.shape)}")


if __name__ == "__main__":
    main()
