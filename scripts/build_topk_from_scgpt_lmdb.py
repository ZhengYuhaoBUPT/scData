#!/usr/bin/env python3
# coding: utf-8

import argparse
import heapq
import json
from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
import msgpack
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build per-shard top-k cells for pathway genes from scGPT-format LMDB.')
    parser.add_argument('--lmdb-path', type=str, required=True)
    parser.add_argument('--pathway-json', type=str, default='/data/bgi/data/projects/multimodal/zyh/datasets/pathway/pathway_anchor_genes.json')
    parser.add_argument('--scgpt-vocab', type=str, default='/data/bgi/data/projects/multimodal/zyh/datasets/CellwText/scgpt/vocab.json')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--max-cells', type=int, default=None)
    return parser.parse_args()


def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    lmdb_path = Path(args.lmdb_path)
    shard_name = lmdb_path.name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{shard_name}.topk.json'

    pathway = load_json(args.pathway_json)
    scgpt_vocab = load_json(args.scgpt_vocab)
    genes = pathway['pathway_genes_list']
    gene_to_scgpt_id = {g: int(scgpt_vocab[g]) for g in genes if g in scgpt_vocab}
    target_id_to_gene = {v: k for k, v in gene_to_scgpt_id.items()}
    missing_genes = [g for g in genes if g not in gene_to_scgpt_id]

    heaps: Dict[str, List[Tuple[float, str, int]]] = {g: [] for g in genes}
    processed_cells = 0

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False, subdir=True, max_readers=64)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, raw in tqdm(cursor, desc=f'topk {shard_name}', unit='cell', mininterval=10.0):
            cell_key = key.decode()
            obj = msgpack.unpackb(raw, raw=False)
            scgpt_ids = obj['scgpt_ids']
            expr = obj['log1p_x']
            for idx, scgpt_id in enumerate(scgpt_ids):
                gene = target_id_to_gene.get(scgpt_id)
                if gene is None:
                    continue
                score = float(expr[idx])
                item = (score, cell_key, idx)
                heap = heaps[gene]
                if len(heap) < args.top_k:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)
            processed_cells += 1
            if args.max_cells is not None and processed_cells >= args.max_cells:
                break
    env.close()

    result = {
        'shard': shard_name,
        'top_k': args.top_k,
        'processed_cells': processed_cells,
        'pathway_gene_count': len(genes),
        'matched_gene_count': len(gene_to_scgpt_id),
        'missing_genes': missing_genes,
        'topk': {
            gene: [
                {
                    'score': score,
                    'shard': shard_name,
                    'cell_key': cell_key,
                    'gene_slot_index': slot_idx,
                }
                for score, cell_key, slot_idx in sorted(items, key=lambda x: (-x[0], x[1]))
            ]
            for gene, items in heaps.items()
        },
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'saved {output_path}')
    print(f'processed_cells={processed_cells}')
    print(f'matched_gene_count={len(gene_to_scgpt_id)}')
    print(f'missing_genes={missing_genes}')


if __name__ == '__main__':
    main()
