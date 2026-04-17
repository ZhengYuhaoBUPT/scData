#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/data/bgi/data/projects/multimodal/zyh/scData")
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scgeneqformer.data.lmdb_dataset import (
    build_static_gene_embeddings_from_cell_features_refs,
    collect_topk_cell_refs,
    load_cells_by_refs,
    load_first_n_cells,
    load_json,
    load_topk_json,
)
from scgeneqformer.models.cell_encoder import encode_pathway_vectors_to_cell_features, load_cell_encoder
from scgeneqformer.models.gene_qformer import GeneQFormerModel
from scgeneqformer.train.trainer import run_reconstruction_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a first-pass scGeneQFormer prototype.")
    parser.add_argument("--topk-json", type=str, default=str(PROJECT_ROOT / "outputs/pathway_static_pipeline/merged_topk.json"))
    parser.add_argument("--lmdb-path", type=str, default="/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText/split_task1_writer1.db")
    parser.add_argument("--lmdb-root", type=str, default="/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText")
    parser.add_argument("--pathway-json", type=str, default="/data/bgi/data/projects/multimodal/zyh/datasets/pathway/pathway_anchor_genes.json")
    parser.add_argument("--lmdb-vocab", type=str, default="/data/bgi/data/projects/multimodal/zyh/datasets/CellwText/vocab/gene_vocab.json")
    parser.add_argument("--scgpt-path", type=str, default="/root/wanghaoran/zxy/project/sc_showo/scgpt")
    parser.add_argument("--encoder-model-path", type=str, default="/root/wanghaoran/zxy/project/sc_showo/save/okrcell_ckpt/model-241492.pt")
    parser.add_argument("--encoder-vocab-path", type=str, default="/data/bgi/data/projects/multimodal/zyh/datasets/CellwText/scgpt/vocab.json")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs/training_runs/scgene_qformer_run1"))
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-train-cells", type=int, default=64)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--max-topk-cells", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_two_stage_scgpt_mapping(genes, gene_to_lmdb_id, scgpt_vocab):
    missing = []
    gene_to_scgpt_id = {}
    for gene in genes:
        if gene in scgpt_vocab:
            gene_to_scgpt_id[gene] = int(scgpt_vocab[gene])
        else:
            missing.append(gene)
    return gene_to_scgpt_id, missing


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topk_data = load_topk_json(Path(args.topk_json))
    pathway_data = load_json(Path(args.pathway_json))
    lmdb_vocab = load_json(Path(args.lmdb_vocab))
    all_genes = pathway_data["pathway_genes_list"]
    genes = [gene for gene in all_genes if len(topk_data["topk"].get(gene, [])) > 0]
    dropped_genes = [gene for gene in all_genes if len(topk_data["topk"].get(gene, [])) == 0]
    topk_data = {**topk_data, "topk": {gene: topk_data["topk"][gene] for gene in genes}}
    gene_to_lmdb_id = {gene: int(lmdb_vocab[gene]) for gene in genes}
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    lmdb_id_to_target_idx = {lmdb_id: gene_to_idx[gene] for gene, lmdb_id in gene_to_lmdb_id.items()}
    num_genes = len(genes)
    print(f"Keeping {num_genes}/{len(all_genes)} genes with at least one top-k cell; dropped {len(dropped_genes)} zero-cell genes")

    gene_encoder, projection_head, scgpt_vocab, binning_fn = load_cell_encoder(
        model_path=args.encoder_model_path,
        vocab_path=args.encoder_vocab_path,
        scgpt_path=args.scgpt_path,
        device=device,
    )
    gene_to_scgpt_id, missing_scgpt_genes = build_two_stage_scgpt_mapping(genes, gene_to_lmdb_id, scgpt_vocab)
    print(f"Two-stage vocab mapping: {len(gene_to_scgpt_id)}/{len(genes)} genes found in scGPT vocab")
    if missing_scgpt_genes:
        print(f"Missing scGPT genes (first 20): {missing_scgpt_genes[:20]}")

    topk_cell_refs = collect_topk_cell_refs(topk_data)
    if args.max_topk_cells is not None:
        print(f"Ignoring --max-topk-cells={args.max_topk_cells}; using all per-gene top-k cells for kept genes")
    topk_ref_keys, topk_expr, _topk_metadata = load_cells_by_refs(
        lmdb_root=Path(args.lmdb_root),
        cell_refs=topk_cell_refs,
        lmdb_id_to_target_idx=lmdb_id_to_target_idx,
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

    train_cell_keys, train_expr, train_metadata = load_first_n_cells(
        lmdb_path=Path(args.lmdb_path),
        lmdb_id_to_target_idx=lmdb_id_to_target_idx,
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
            "gene_to_lmdb_id": gene_to_lmdb_id,
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
                "source_lmdb_path": args.lmdb_path,
                "num_dropped_zero_topk_genes": len(dropped_genes),
                "dropped_zero_topk_genes": dropped_genes,
            },
            f,
            indent=2,
        )
    print(f"Saved training bundle to {output_dir}")
    print(f"Static gene embeddings shape: {tuple(static_gene_embeddings.shape)}")
    print(f"Query tokens shape: {tuple(query_tokens.shape)}")


if __name__ == "__main__":
    main()
