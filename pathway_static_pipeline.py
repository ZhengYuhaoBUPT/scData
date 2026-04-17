#!/usr/bin/env python3
# coding: utf-8

import argparse
import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import lmdb
import numpy as np
import torch
from tqdm import tqdm


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def open_lmdb(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
        max_readers=256,
    )


def iter_records(txn: lmdb.Transaction, max_cells: Optional[int]) -> Iterable[Tuple[str, Dict]]:
    cursor = txn.cursor()
    seen = 0
    for key, value in cursor:
        if key.startswith(b"-") or key in {b"__len__", b"num_samples"}:
            continue
        yield key.decode(), json.loads(value)
        seen += 1
        if max_cells is not None and seen >= max_cells:
            break


def keep_sample(cell_key: str, sample_ratio: float, sample_seed: int) -> bool:
    if sample_ratio >= 1.0:
        return True
    payload = f"{sample_seed}:{cell_key}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    threshold = int(sample_ratio * ((1 << 64) - 1))
    return value <= threshold


def load_targets(pathway_json: Path, lmdb_vocab_path: Path) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    pathway_data = load_json(pathway_json)
    lmdb_vocab = load_json(lmdb_vocab_path)
    genes = pathway_data["pathway_genes_list"]
    gene_to_lmdb_id: Dict[str, int] = {}
    lmdb_id_to_gene: Dict[int, str] = {}
    for gene in genes:
        if gene not in lmdb_vocab:
            raise KeyError(f"Missing target gene in vocab: {gene}")
        lmdb_id = int(lmdb_vocab[gene])
        gene_to_lmdb_id[gene] = lmdb_id
        lmdb_id_to_gene[lmdb_id] = gene
    return genes, gene_to_lmdb_id, lmdb_id_to_gene


def find_shards(lmdb_root: Path, shard_name: Optional[str], limit_shards: Optional[int]) -> List[Path]:
    shards = sorted(p for p in lmdb_root.glob("*.db") if p.is_dir())
    if shard_name is not None:
        shards = [p for p in shards if p.name == shard_name]
    if limit_shards is not None:
        shards = shards[:limit_shards]
    if not shards:
        raise FileNotFoundError("No LMDB shards matched the request.")
    return shards


def serialize_topk(topk: Dict[str, List[Tuple[float, str, str, int]]]) -> Dict:
    return {
        gene: [
            {
                "score": score,
                "shard": shard,
                "cell_key": cell_key,
                "gene_slot_index": gene_slot_index,
            }
            for score, shard, cell_key, gene_slot_index in items
        ]
        for gene, items in topk.items()
    }


def deserialize_topk(data: Dict) -> Dict[str, List[Tuple[float, str, str, int]]]:
    return {
        gene: [
            (float(item["score"]), item["shard"], item["cell_key"], int(item["gene_slot_index"]))
            for item in items
        ]
        for gene, items in data.items()
    }


def shard_topk(args: argparse.Namespace) -> None:
    lmdb_root = Path(args.lmdb_root)
    output_dir = Path(args.output_dir)
    shards = find_shards(lmdb_root, args.shard_name, args.limit_shards)
    genes, _gene_to_lmdb_id, lmdb_id_to_gene = load_targets(Path(args.pathway_json), Path(args.lmdb_vocab))
    target_lmdb_ids = set(lmdb_id_to_gene.keys())

    per_shard_dir = output_dir / "per_shard_topk"
    per_shard_dir.mkdir(parents=True, exist_ok=True)

    for shard_path in shards:
        out_path = per_shard_dir / f"{shard_path.name}.topk.json"
        if out_path.exists() and not args.overwrite:
            print(f"skip existing shard result: {out_path}")
            continue

        heaps: Dict[str, List[Tuple[float, str, str, int]]] = {gene: [] for gene in genes}
        processed_cells = 0

        env = open_lmdb(shard_path)
        with env.begin(write=False) as txn:
            for cell_key, record in tqdm(
                iter_records(txn, max_cells=args.max_cells_per_shard),
                desc=f"shard-topk {shard_path.name}",
                unit="cell",
                mininterval=10.0,
            ):
                if not keep_sample(cell_key, args.sample_ratio, args.sample_seed):
                    continue
                for idx, lmdb_id in enumerate(record["gene_ids"]):
                    if lmdb_id not in target_lmdb_ids:
                        continue
                    score = float(record["log1p_x"][idx])
                    gene = lmdb_id_to_gene[lmdb_id]
                    item = (score, shard_path.name, cell_key, idx)
                    heap = heaps[gene]
                    if len(heap) < args.top_k:
                        heapq.heappush(heap, item)
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, item)
                processed_cells += 1
        env.close()

        result = {
            "shard": shard_path.name,
            "top_k": args.top_k,
            "sample_ratio": args.sample_ratio,
            "sample_seed": args.sample_seed,
            "processed_cells": processed_cells,
            "topk": serialize_topk({gene: sorted(items, key=lambda x: (-x[0], x[2])) for gene, items in heaps.items()}),
        }
        save_json(out_path, result)
        print(f"saved shard topk: {out_path}")


def merge_topk(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    per_shard_dir = output_dir / "per_shard_topk"
    merged_path = output_dir / "merged_topk.json"

    genes, gene_to_lmdb_id, _lmdb_id_to_gene = load_targets(Path(args.pathway_json), Path(args.lmdb_vocab))
    heaps: Dict[str, List[Tuple[float, str, str, int]]] = {gene: [] for gene in genes}

    shard_files = sorted(per_shard_dir.glob("*.topk.json"))
    if not shard_files:
        raise FileNotFoundError(f"No per-shard topk files found under {per_shard_dir}")

    total_cells = 0
    used_files: List[str] = []
    for shard_file in shard_files:
        data = load_json(shard_file)
        total_cells += int(data.get("processed_cells", 0))
        used_files.append(shard_file.name)
        shard_topk = deserialize_topk(data["topk"])
        for gene, items in shard_topk.items():
            heap = heaps[gene]
            for item in items:
                if len(heap) < args.top_k:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

    merged = {gene: sorted(items, key=lambda x: (-x[0], x[1], x[2])) for gene, items in heaps.items()}
    save_json(
        merged_path,
        {
            "top_k": args.top_k,
            "processed_cells": total_cells,
            "source_files": used_files,
            "gene_to_lmdb_id": gene_to_lmdb_id,
            "topk": serialize_topk(merged),
        },
    )
    print(f"saved merged topk: {merged_path}")


def build_prototypes(args: argparse.Namespace) -> None:
    lmdb_root = Path(args.lmdb_root)
    output_dir = Path(args.output_dir)
    merged_path = output_dir / "merged_topk.json"
    merged = load_json(merged_path)

    genes = list(merged["gene_to_lmdb_id"].keys())
    gene_to_lmdb_id = {gene: int(v) for gene, v in merged["gene_to_lmdb_id"].items()}
    topk_results = deserialize_topk(merged["topk"])

    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    lmdb_id_to_target_idx = {lmdb_id: gene_to_idx[gene] for gene, lmdb_id in gene_to_lmdb_id.items()}

    selected_by_shard: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for gene, items in topk_results.items():
        for _score, shard_name, cell_key, _gene_slot_index in items:
            selected_by_shard[shard_name][cell_key].append(gene)

    sums = np.zeros((len(genes), len(genes)), dtype=np.float32)
    counts = np.zeros(len(genes), dtype=np.int32)

    shards = find_shards(lmdb_root, None, None)
    for shard_path in shards:
        wanted = selected_by_shard.get(shard_path.name)
        if not wanted:
            continue
        env = open_lmdb(shard_path)
        with env.begin(write=False) as txn:
            for cell_key in tqdm(sorted(wanted.keys()), desc=f"prototype {shard_path.name}", unit="cell", mininterval=10.0):
                raw = txn.get(cell_key.encode())
                if raw is None:
                    continue
                record = json.loads(raw)
                vector = np.zeros(len(genes), dtype=np.float32)
                for lmdb_id, value in zip(record["gene_ids"], record["log1p_x"]):
                    idx = lmdb_id_to_target_idx.get(lmdb_id)
                    if idx is not None:
                        vector[idx] = float(value)
                for gene in wanted[cell_key]:
                    row = gene_to_idx[gene]
                    sums[row] += vector
                    counts[row] += 1
        env.close()

    prototypes = sums / np.maximum(counts[:, None], 1)
    torch.save(
        {
            "genes": genes,
            "gene_to_lmdb_id": gene_to_lmdb_id,
            "prototype_4366d": torch.from_numpy(prototypes),
            "topk_counts": torch.from_numpy(counts),
        },
        output_dir / "pathway_gene_static_prototypes.pt",
    )
    save_json(
        output_dir / "prototype_metadata.json",
        {
            "top_k": int(merged["top_k"]),
            "processed_cells": int(merged["processed_cells"]),
            "prototype_shape": list(prototypes.shape),
            "num_incomplete": int((counts < int(merged["top_k"])).sum()),
        },
    )
    print(f"saved prototypes: {output_dir / 'pathway_gene_static_prototypes.pt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable shard-wise pathway static prototype pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lmdb-root", type=str, default="/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText")
    common.add_argument("--pathway-json", type=str, default="/data/bgi/data/projects/multimodal/zyh/datasets/pathway/pathway_anchor_genes.json")
    common.add_argument("--lmdb-vocab", type=str, default="/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText/config/lmdb_vocab.json")
    common.add_argument("--output-dir", type=str, default="/data/bgi/data/projects/multimodal/zyh/scData/outputs/pathway_static_pipeline")
    common.add_argument("--top-k", type=int, default=100)

    p1 = subparsers.add_parser("shard-topk", parents=[common])
    p1.add_argument("--shard-name", type=str, default=None)
    p1.add_argument("--limit-shards", type=int, default=None)
    p1.add_argument("--max-cells-per-shard", type=int, default=None)
    p1.add_argument("--sample-ratio", type=float, default=1.0)
    p1.add_argument("--sample-seed", type=int, default=42)
    p1.add_argument("--overwrite", action="store_true")
    p1.set_defaults(func=shard_topk)

    p2 = subparsers.add_parser("merge-topk", parents=[common])
    p2.set_defaults(func=merge_topk)

    p3 = subparsers.add_parser("build-prototypes", parents=[common])
    p3.set_defaults(func=build_prototypes)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
