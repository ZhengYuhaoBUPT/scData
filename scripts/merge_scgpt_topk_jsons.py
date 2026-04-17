#!/usr/bin/env python3
# coding: utf-8

import argparse
import glob
import heapq
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merge per-shard scGPT topk JSON files into global topk per gene.')
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-json', type=str, required=True)
    parser.add_argument('--top-k', type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(glob.glob(str(input_dir / '*.json')))
    if not input_paths:
        raise FileNotFoundError(f'No json files found in {input_dir}')

    merged_heaps: Dict[str, List[Tuple[float, str, str, int]]] = {}
    total_processed_cells = 0
    all_missing = set()
    pathway_gene_count = None

    for path in input_paths:
        with open(path, 'r') as f:
            data = json.load(f)
        total_processed_cells += int(data.get('processed_cells', 0))
        all_missing.update(data.get('missing_genes', []))
        if pathway_gene_count is None:
            pathway_gene_count = int(data['pathway_gene_count'])
        for gene, records in data['topk'].items():
            heap = merged_heaps.setdefault(gene, [])
            for record in records:
                score = float(record['score'])
                shard = str(record['shard'])
                cell_key = str(record['cell_key'])
                slot_idx = int(record['gene_slot_index'])
                item = (score, shard, cell_key, slot_idx)
                if len(heap) < args.top_k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)

    result = {
        'top_k': args.top_k,
        'num_source_files': len(input_paths),
        'processed_cells': total_processed_cells,
        'pathway_gene_count': pathway_gene_count,
        'missing_genes': sorted(all_missing),
        'topk': {
            gene: [
                {
                    'score': score,
                    'shard': shard,
                    'cell_key': cell_key,
                    'gene_slot_index': slot_idx,
                }
                for score, shard, cell_key, slot_idx in sorted(items, key=lambda x: (-x[0], x[1], x[2]))
            ]
            for gene, items in sorted(merged_heaps.items())
        },
    }

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    counts = [len(v) for v in result['topk'].values()]
    print(f'saved {output_json}')
    print(f'num_source_files={len(input_paths)}')
    print(f'processed_cells={total_processed_cells}')
    print(f'pathway_gene_count={pathway_gene_count}')
    print(f'genes_with_any_hits={sum(c > 0 for c in counts)}')
    print(f'genes_with_100_hits={sum(c == args.top_k for c in counts)}')
    print(f'missing_genes={sorted(all_missing)}')


if __name__ == '__main__':
    main()
