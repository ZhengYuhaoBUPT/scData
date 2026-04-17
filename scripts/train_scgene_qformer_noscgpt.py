#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path('/data/bgi/data/projects/multimodal/zyh/scData')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scgeneqformer.data.lmdb_dataset import (
    build_gene_mappings,
    build_static_prototypes_from_topk,
    load_first_n_cells,
    load_topk_json,
)
from scgeneqformer.models.prototype_qformer import PrototypeQFormerModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train scGeneQFormer without scGPT/vocab dependency.')
    parser.add_argument('--topk-json', type=str, default=str(PROJECT_ROOT / 'outputs/pathway_static_pipeline/per_shard_topk/split_task1_writer1.db.topk.json'))
    parser.add_argument('--lmdb-path', type=str, default='/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText/split_task1_writer1.db')
    parser.add_argument('--pathway-json', type=str, default='/data/bgi/data/projects/multimodal/zyh/datasets/pathway/pathway_anchor_genes.json')
    parser.add_argument('--lmdb-vocab', type=str, default='/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText/config/lmdb_vocab.json')
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'outputs/training_runs/scgene_qformer_noscgpt_run1'))
    parser.add_argument('--hidden-dim', type=int, default=768)
    parser.add_argument('--num-queries', type=int, default=50)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-topk-cells', type=int, default=None)
    parser.add_argument('--max-train-cells', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def run_training(model, static_prototypes, cell_expr, num_epochs, batch_size, learning_rate, device):
    model = model.to(device)
    static_prototypes = static_prototypes.to(device)
    cell_expr = cell_expr.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        steps = 0
        for start in range(0, cell_expr.size(0), batch_size):
            batch = cell_expr[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            _static, _queries, recon = model(static_prototypes, batch)
            loss = torch.nn.functional.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1
        history.append({'epoch': epoch + 1, 'loss': epoch_loss / max(steps, 1)})
    return history


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topk_data = load_topk_json(Path(args.topk_json))
    genes, gene_to_lmdb_id, lmdb_id_to_target_idx = build_gene_mappings(Path(args.pathway_json), Path(args.lmdb_vocab))
    if args.max_topk_cells is not None:
        for gene, items in topk_data['topk'].items():
            topk_data['topk'][gene] = items[:args.max_topk_cells]

    static_prototypes, topk_counts = build_static_prototypes_from_topk(
        topk_data=topk_data,
        lmdb_path=Path(args.lmdb_path),
        genes=genes,
        lmdb_id_to_target_idx=lmdb_id_to_target_idx,
    )
    train_cell_keys, train_expr, train_metadata = load_first_n_cells(
        lmdb_path=Path(args.lmdb_path),
        lmdb_id_to_target_idx=lmdb_id_to_target_idx,
        num_genes=len(genes),
        max_cells=args.max_train_cells,
    )

    model = PrototypeQFormerModel(
        num_genes=len(genes),
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    history = run_training(
        model=model,
        static_prototypes=static_prototypes,
        cell_expr=train_expr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )

    model = model.to(device).eval()
    with torch.no_grad():
        static_gene_embeddings, query_tokens, recon = model(static_prototypes.to(device), train_expr.to(device))
    static_gene_embeddings = static_gene_embeddings.cpu()
    query_tokens = query_tokens.cpu()
    recon = recon.cpu()

    torch.save(
        {
            'genes': genes,
            'gene_to_lmdb_id': gene_to_lmdb_id,
            'topk_counts': topk_counts,
            'static_prototypes_4366d': static_prototypes,
            'static_gene_embeddings_768d': static_gene_embeddings,
            'train_cell_keys': train_cell_keys,
            'train_cell_metadata': train_metadata,
            'query_tokens': query_tokens,
            'reconstructed_expr': recon,
            'train_history': history,
            'config': vars(args),
        },
        output_dir / 'scgene_qformer_noscgpt_run.pt',
    )
    with (output_dir / 'scgene_qformer_noscgpt_run_metadata.json').open('w') as f:
        json.dump(
            {
                'num_genes': len(genes),
                'num_train_cells': len(train_cell_keys),
                'static_prototypes_shape': list(static_prototypes.shape),
                'static_gene_embeddings_shape': list(static_gene_embeddings.shape),
                'query_tokens_shape': list(query_tokens.shape),
                'reconstructed_expr_shape': list(recon.shape),
                'train_history': history,
                'source_topk_json': args.topk_json,
                'source_lmdb_path': args.lmdb_path,
                'note': 'No scGPT vocab or pretrained encoder is used in this pipeline.',
            },
            f,
            indent=2,
        )
    print(f'Saved no-scGPT training bundle to {output_dir}')
    print(f'Static gene embeddings shape: {tuple(static_gene_embeddings.shape)}')
    print(f'Query tokens shape: {tuple(query_tokens.shape)}')


if __name__ == '__main__':
    main()
