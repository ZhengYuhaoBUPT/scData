#!/usr/bin/env python3
# coding=utf-8

import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch


def main():
    h5ad_path = Path('/data/bgi/data/projects/multimodal/zxy/sft_eval_data/tab_sap.h5ad')
    eval_json_path = Path('/data/bgi/data/projects/multimodal/RNA_data/cellwhisper_data/sft_data/conversations/tabula_sapiens_conversations.json')
    ckpt_path = Path('/data/bgi/data/projects/multimodal/zyh/scData/outputs/training_runs/scgene_qformer_compact_cellfeat_rankaux_top128_1layer_v1/scgene_qformer_cellfeat_checkpoint.pt')
    output_path = Path('/data/bgi/data/projects/multimodal/zyh/scData/outputs/eval_prep/tab_sap_top1200_intersection.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eval_items = json.loads(eval_json_path.read_text())
    target_cell_ids = [str(item['id']) for item in eval_items]
    target_cell_id_set = set(target_cell_ids)

    payload = torch.load(str(ckpt_path), map_location='cpu')
    train_genes = [str(g) for g in payload['genes']]
    gene_to_static_idx = {g: i for i, g in enumerate(train_genes)}

    adata = ad.read_h5ad(str(h5ad_path), backed='r')
    obs_ids = [str(x) for x in adata.obs.index.tolist()]
    cell_to_row = {cid: i for i, cid in enumerate(obs_ids) if cid in target_cell_id_set}

    if 'gene_symbol' in adata.var.columns:
        eval_gene_names = [str(x) for x in adata.var['gene_symbol'].tolist()]
    else:
        eval_gene_names = [str(x) for x in adata.var.index.tolist()]

    mapped_cols = []
    mapped_gene_names = []
    mapped_static_idx = []
    for col_idx, gene_name in enumerate(eval_gene_names):
        static_idx = gene_to_static_idx.get(gene_name)
        if static_idx is None:
            continue
        mapped_cols.append(col_idx)
        mapped_gene_names.append(gene_name)
        mapped_static_idx.append(static_idx)

    mapped_cols = np.asarray(mapped_cols, dtype=np.int64)
    mapped_static_idx = np.asarray(mapped_static_idx, dtype=np.int64)

    found_cell_ids = [cid for cid in target_cell_ids if cid in cell_to_row]
    found_rows = np.asarray([cell_to_row[cid] for cid in found_cell_ids], dtype=np.int64)
    missing_cells = [cid for cid in target_cell_ids if cid not in cell_to_row]

    sub = adata.X[found_rows][:, mapped_cols]
    if hasattr(sub, 'toarray'):
        sub = sub.toarray()
    sub = np.asarray(sub, dtype=np.float32)
    if sub.ndim != 2:
        sub = sub.reshape(len(found_cell_ids), len(mapped_cols))

    items = []
    for cell_id, values in zip(found_cell_ids, sub):
        order = np.argsort(values)[::-1][:1200]
        top_gene_names = [mapped_gene_names[i] for i in order]
        top_static_gene_indices = mapped_static_idx[order].astype(int).tolist()
        top_h5ad_col_indices = mapped_cols[order].astype(int).tolist()
        top_expr_values = values[order].astype(float).tolist()
        num_positive_expr = int((values > 0).sum())

        items.append({
            'cell_id': cell_id,
            'top_gene_names': top_gene_names,
            'top_static_gene_indices': top_static_gene_indices,
            'top_h5ad_col_indices': top_h5ad_col_indices,
            'top_expr_values': top_expr_values,
            'num_top_genes': len(top_gene_names),
            'num_positive_expr_in_overlap': num_positive_expr,
            'pad_count_if_used_as_1200_tokens': max(0, 1200 - len(top_gene_names)),
        })

    output = {
        'source_h5ad': str(h5ad_path),
        'source_eval_json': str(eval_json_path),
        'source_static_gene_ckpt': str(ckpt_path),
        'num_eval_items': len(eval_items),
        'num_target_cells': len(target_cell_ids),
        'num_found_cells': len(items),
        'num_missing_cells': len(missing_cells),
        'missing_cells': missing_cells,
        'num_train_genes': len(train_genes),
        'num_eval_genes': int(adata.shape[1]),
        'num_overlap_genes': int(len(mapped_cols)),
        'top_k': 1200,
        'rank_order': 'desc',
        'items': items,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False))
    print('output_path', output_path)
    print('num_found_cells', len(items))
    print('num_missing_cells', len(missing_cells))
    print('num_overlap_genes', len(mapped_cols))
    if items:
        first = items[0]
        print('first_cell_id', first['cell_id'])
        print('first_num_top_genes', first['num_top_genes'])
        print('first_first_10_genes', first['top_gene_names'][:10])


if __name__ == '__main__':
    main()
