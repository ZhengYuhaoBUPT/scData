# coding=utf-8
"""
Bidirectional Stage 1 Dataset for Unified Multimodal Model (Understanding + Generation)
Fully decoupled version: no h5ad, reads from cluster LMDB directly.
"""

import bisect
import collections
import json
import math
import os
import random
import hashlib
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from .cluster_feature_loader import ClusterFeatureLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MetadataFormatter:
    """Inline copy from src_ablation_cw.datasets.metadata_formatter."""

    def __init__(
        self,
        celltype_question_weight: float = 0.65,
        field_dropout_prob: float = 0.15,
        short_prob: float = 0.10,
        qa_prob: float = 0.40,
    ):
        self.celltype_weight = celltype_question_weight
        self.field_dropout_prob = field_dropout_prob
        self.short_prob = short_prob
        self.qa_prob = qa_prob
        assert short_prob + qa_prob <= 1.0, "short_prob + qa_prob must <= 1.0"

        self.caption_templates = [
            "This sample represents a {celltype} derived from {tissue}. "
            "The cell is at the {stage} stage and associated with {disease}. Sex: {sex}.",
            "Cell type: {celltype}\nTissue: {tissue}\nStage: {stage}\nCondition: {disease}\nSex: {sex}",
            "The cell is characterized as {celltype}, originating in the {tissue}. "
            "It corresponds to the {stage} stage with {disease} condition. Sex is {sex}.",
            "Sample characteristics include {celltype} ({tissue}, {stage}, {disease}, {sex}).",
            "The sample originates from {tissue} and shows {disease} condition. "
            "The cell type is {celltype}, at the {stage} stage, from a {sex} donor.",
            "{celltype} ({tissue}, {stage})",
        ]

        self.celltype_questions = [
            "What is the cell type of this sample?",
            "Identify the cell type.",
            "Which cell type does this sample belong to?",
            "What kind of cell is this?",
            "Can you tell me the cell type?",
            "Please classify this cell.",
        ]

        self.meta_questions = {
            "tissue": [
                "What tissue does this cell come from?",
                "Which tissue is this cell derived from?",
                "What is the tissue origin?",
            ],
            "disease": [
                "What is the disease condition?",
                "What condition is associated with this cell?",
                "Describe the disease state.",
            ],
            "stage": [
                "What developmental stage is this cell at?",
                "What stage does this cell belong to?",
                "What is the developmental stage?",
            ],
            "sex": [
                "What is the sex of the donor?",
                "What sex is this sample from?",
                "Is this from a male or female donor?",
            ],
        }

        self.short_templates = [
            "This cell is a {celltype}.",
            "This cell is an {celltype}.",
            "The cell type is {celltype}.",
            "It is a {celltype}.",
            "This sample represents a {celltype}.",
        ]

        self.short_questions = [
            "Describe the cell briefly.",
            "What is this cell?",
            "",
        ]

    def _clean_metadata(self, metadata: Dict) -> Dict:
        cleaned = {}
        for k, v in metadata.items():
            if v is None:
                continue
            s = str(v).strip()
            if not s or s.lower() in ("unknown", "nan", "none", ""):
                continue
            if k.endswith("_name"):
                base = k.replace("_name", "")
                cleaned[base] = s
            elif k.endswith("_definition"):
                base = k.replace("_definition", "")
                if base not in cleaned:
                    cleaned[base] = s
        return cleaned

    def format(self, metadata: Dict, force_mode: str = None) -> tuple:
        meta = self._clean_metadata(metadata)
        if not meta or "celltype" not in meta:
            return "Describe the cell.", "This is a cell sample."

        if force_mode == "celltype_qa":
            return self._build_qa_celltype(meta)
        elif force_mode == "meta":
            if random.random() < 0.5:
                return self._build_caption(meta)
            else:
                return self._build_qa(meta)

        r = random.random()
        if r < self.short_prob:
            return self._build_short(meta)
        elif r < self.short_prob + self.qa_prob:
            return self._build_qa(meta)
        else:
            return self._build_caption(meta)

    def format_caption_only(self, metadata: Dict) -> tuple:
        meta = self._clean_metadata(metadata)
        if not meta or "celltype" not in meta:
            return "Describe the cell.", "This is a cell sample."
        return self._build_caption(meta)

    def _build_short(self, meta: Dict) -> tuple:
        celltype = meta.get("celltype", "cell")
        ans = random.choice(self.short_templates).format(celltype=celltype)
        ans = self._fix_a_an(ans)
        q = random.choice(self.short_questions)
        return q, ans

    def _build_qa_celltype(self, meta: Dict) -> tuple:
        q = random.choice(self.celltype_questions)
        a = meta["celltype"]
        templates = [
            "{answer}.",
            "The cell type is {answer}.",
            "It is {answer}.",
            "This cell is {answer}.",
            "{answer}",
        ]
        ans = random.choice(templates).format(answer=a)
        ans = self._fix_a_an(ans)
        return q, ans

    def _build_qa(self, meta: Dict) -> tuple:
        other_fields = [k for k in meta.keys() if k != "celltype"]
        if random.random() < self.celltype_weight or not other_fields:
            return self._build_qa_celltype(meta)
        else:
            field = random.choice(other_fields)
            q_pool = self.meta_questions.get(field, [f"What is the {field}?"])
            q = random.choice(q_pool)
            a = meta[field]
            field_name = field

        templates = [
            "{answer}.",
            "The {field} is {answer}.",
            "It is {answer}.",
            "This cell has {answer} as its {field}.",
            "{answer}",
        ]
        ans = random.choice(templates).format(answer=a, field=field_name)
        ans = self._fix_a_an(ans)
        return q, ans

    def _build_caption(self, meta: Dict) -> tuple:
        active = {k: v for k, v in meta.items() if k == "celltype" or random.random() >= self.field_dropout_prob}

        if random.random() < 0.35:
            fields = list(active.items())
            random.shuffle(fields)
            parts = []
            for k, v in fields:
                if k == "celltype":
                    parts.append(f"the cell type is {v}")
                elif k == "tissue":
                    parts.append(f"derived from {v}")
                elif k == "disease":
                    parts.append(f"associated with {v}")
                elif k == "stage":
                    parts.append(f"at the {v} stage")
                elif k == "sex":
                    parts.append(f"from a {v} donor")
            if parts:
                if len(parts) == 1:
                    sentence = f"The sample {parts[0]}."
                elif len(parts) == 2:
                    sentence = f"The sample {parts[0]} and {parts[1]}."
                else:
                    sentence = f"The sample {', '.join(parts[:-1])}, and {parts[-1]}."
                sentence = self._fix_a_an(sentence)
                return "Describe the cell.", sentence

        template = random.choice(self.caption_templates)
        filled = template.format(
            celltype=active.get("celltype", "unknown cell type"),
            tissue=active.get("tissue", "unknown tissue"),
            disease=active.get("disease", "healthy"),
            stage=active.get("stage", "unknown stage"),
            sex=active.get("sex", "unknown"),
        )
        filled = self._fix_a_an(filled)
        return "Describe the cell.", filled

    @staticmethod
    def _fix_a_an(text: str) -> str:
        import re
        text = re.sub(r"\ba\s+([aeiouAEIOU])", r"an \1", text)
        text = re.sub(r"\ban\s+([^aeiouAEIOU\s])", r"a \1", text)
        return text


class BidirectionalStage1Dataset(Dataset):
    """
    Bidirectional Stage 1 Dataset - reads from cluster LMDB (_cluster.db).
    - 50% understanding: [Gene] -> [Text]
    - 50% generation:    [Text] -> [Gene]
    """

    def __init__(
        self,
        cluster_lmdb_dir: str = None,
        caption_lmdb_dir: str = None,
        codebook_dir: str = None,
        scgpt_gene_vocab: str = None,
        text_tokenizer: Any = None,
        config_dict: Dict[str, Any] = None,
        special_tokens_ids: Dict[str, int] = None,
        max_seq_len: Optional[int] = None,
        accelerator=None,
        understanding_ratio: float = 0.5,
        skip_load_data: bool = False,
    ):
        super().__init__()

        self.data_config = config_dict.get("data", {}) if config_dict else {}
        self.dataset_config = config_dict.get("dataset", {}) if config_dict else {}
        self.training_config = config_dict.get("training", {}) if config_dict else {}

        self.cluster_lmdb_dir = Path(cluster_lmdb_dir) if cluster_lmdb_dir else Path(self.data_config.get("cluster_lmdb_dir", ""))
        self.caption_lmdb_dir = Path(caption_lmdb_dir) if caption_lmdb_dir else Path(self.data_config.get("lmdb_base_dir", ""))
        self.codebook_dir = Path(codebook_dir) if codebook_dir else Path(self.data_config.get("codebook_dir", ""))
        self.scgpt_gene_vocab = scgpt_gene_vocab if scgpt_gene_vocab else self.data_config.get("scgpt_gene_vocab", "")

        self.max_genes = self.dataset_config.get("max_genes", 1200)
        self.max_text_len = self.dataset_config.get("max_text_len", 512)
        self.cond_dropout_prob = float(self.dataset_config.get("conditional_dropout_prob", self.dataset_config.get("cond_dropout_prob", 0.2)))
        self.mask_min_ratio = self.dataset_config.get("mask_min_ratio", 0.01)
        self.mask_max_ratio = self.dataset_config.get("mask_max_ratio", 0.99)
        self.understanding_ratio = understanding_ratio
        self.clusters_per_gene = int(self.dataset_config.get("clusters_per_gene", 64))

        self.accelerator = accelerator
        if accelerator is not None:
            self.num_replicas = accelerator.num_processes
            self.rank = accelerator.process_index
        else:
            self.num_replicas = 1
            self.rank = 0

        print(f"[BidirectionalStage1DS] initialized:")
        print(f"   - max_genes: {self.max_genes}")
        print(f"   - understanding_ratio: {self.understanding_ratio}")
        print(f"   - generation_ratio: {1 - self.understanding_ratio}")
        print(f"   - Rank {self.rank}/{self.num_replicas}")

        # Build compact gene vocab from codebook metadata
        self._build_gene_vocab()

        # Text tokenizer & special tokens
        self.text_tokenizer = text_tokenizer
        self.sog_id = special_tokens_ids.get("sog_id")
        self.eog_id = special_tokens_ids.get("eog_id")
        self.pad_id = text_tokenizer.pad_token_id if getattr(text_tokenizer, "pad_token_id", None) is not None else 151643
        self.bos_id = getattr(text_tokenizer, "bos_token_id", None)
        self.text_vocab_size = getattr(text_tokenizer, "vocab_size", 151936)
        # Mask gene id lives at the end of gene vocab
        self.mask_gene_id = self.text_vocab_size + self.gene_vocab_size

        gene_seq_len = self.max_genes + 2  # SOG + genes + EOG
        min_required = self.max_text_len + gene_seq_len
        if max_seq_len is None:
            self.max_seq_len = min_required
        else:
            self.max_seq_len = max_seq_len
            assert self.max_seq_len >= min_required, f"max_seq_len({self.max_seq_len}) too small, need at least {min_required}"

        self.formatter = MetadataFormatter()
        self.caption_envs = {}
        self.use_cell_cls = bool(self.data_config.get("use_cell_cls", False))
        if self.use_cell_cls:
            from .cell_aware_feature_loader import CellAwareFeatureLoader
            self.cell_aware_loader = CellAwareFeatureLoader(
                cluster_lmdb_dir=self.cluster_lmdb_dir,
                codebook_dir=self.codebook_dir,
                clusters_per_gene=self.clusters_per_gene,
                prefer_compact_codebook=bool(self.data_config.get("prefer_compact_codebook", True)),
                readahead=bool(self.data_config.get("cluster_loader_readahead", False)),
                max_readers=int(self.data_config.get("cluster_loader_max_readers", 2048)),
                gene_weight=float(self.data_config.get("cell_cls_gene_weight", 0.5)),
                cell_weight=float(self.data_config.get("cell_cls_cell_weight", 0.5)),
                cell_dropout=float(self.data_config.get("cell_cls_dropout", 0.3)),
            )
            print(f"[BidirectionalStage1DS] ✅ use_cell_cls=True, gene_weight={self.data_config.get('cell_cls_gene_weight', 0.5)}, cell_dropout={self.data_config.get('cell_cls_dropout', 0.3)}")
        else:
            self.feature_loader = ClusterFeatureLoader(
                cluster_lmdb_dir=self.cluster_lmdb_dir,
                readahead=bool(self.data_config.get("cluster_loader_readahead", False)),
                max_readers=int(self.data_config.get("cluster_loader_max_readers", 2048)),
            )
        self.key_cache_dir = Path(self.data_config.get("key_cache_dir", "/tmp/sc_showo_keycache"))
        self.key_cache_dir.mkdir(parents=True, exist_ok=True)
        self._key_offsets_cache = {}
        self._key_bin_handles = {}
        if not skip_load_data:
            self._load_data()

    def _build_gene_vocab(self):
        nclusters_path = self.codebook_dir / "gene_nclusters.json"
        with open(nclusters_path, "r") as f:
            nclusters = {int(k): int(v) for k, v in json.load(f).items()}

        valid_genes = sorted([gid for gid, k in nclusters.items() if k > 0])
        # 保留所有有效基因，不再截断到 max_genes；max_genes 仅控制每个细胞的输入序列长度
        self.scgpt_to_local = {gid: idx for idx, gid in enumerate(valid_genes)}
        self.local_to_scgpt = valid_genes
        self.gene_vocab_size = len(valid_genes) * 64

        # 加载 scGPT vocab 获取 pad_id（用于 gene_labels 中 padding 位置设为 -100）
        scgpt_vocab_path = self.data_config.get("scgpt_gene_vocab", "")
        if scgpt_vocab_path and Path(scgpt_vocab_path).exists():
            with open(scgpt_vocab_path, "r") as f:
                scgpt_vocab = json.load(f)
            self.scgpt_pad_id = scgpt_vocab.get("<pad>", max(int(v) for v in scgpt_vocab.values()) + 1)
        else:
            self.scgpt_pad_id = 0

        print(f"[BidirectionalStage1DS] compact gene vocab: {len(valid_genes)} genes x 64 = {self.gene_vocab_size} tokens")

    @staticmethod
    def _stable_celltype_id(celltype_name: str) -> int:
        name = (celltype_name or "").strip().lower()
        if not name:
            return 0
        h = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, byteorder="big", signed=False) & ((1 << 63) - 1)

    @staticmethod
    def sample_negative_indices_excluding_celltype(
        global_celltype_ids: torch.LongTensor,
        anchor_global_idx: int,
        k: int,
        candidate_indices: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        if global_celltype_ids is None or global_celltype_ids.numel() == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long, device=global_celltype_ids.device if global_celltype_ids is not None else None)

        device = global_celltype_ids.device
        n = int(global_celltype_ids.shape[0])
        all_idx = torch.arange(n, device=device, dtype=torch.long)
        if candidate_indices is None:
            pool = all_idx
        else:
            pool = candidate_indices.to(device=device, dtype=torch.long)

        if pool.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        anchor_global_idx = int(anchor_global_idx)
        anchor_celltype = global_celltype_ids[anchor_global_idx]
        valid = (pool != anchor_global_idx) & (global_celltype_ids[pool] != anchor_celltype)
        pool = pool[valid]

        if pool.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        if int(pool.numel()) <= k:
            return pool

        perm = torch.randperm(int(pool.numel()), device=device)
        return pool[perm[:k]]

    def _get_caption_env(self, lmdb_path: str):
        if lmdb_path not in self.caption_envs:
            self.caption_envs[lmdb_path] = lmdb.Environment(
                lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self.caption_envs[lmdb_path]

    def _cache_paths_for_db(self, db_name: str):
        tag = hashlib.md5(db_name.encode('utf-8')).hexdigest()[:16]
        base = self.key_cache_dir / f"{db_name}.{tag}"
        return base.with_suffix('.keys.bin'), base.with_suffix('.keys.idx.npy')

    def _get_cluster_env(self, db_name: str):
        if self.use_cell_cls:
            return self.cell_aware_loader._base._get_env(db_name)
        return self.feature_loader._get_env(db_name)

    def _build_key_cache_if_needed(self, db_name: str, expected_n: int):
        bin_path, idx_path = self._cache_paths_for_db(db_name)
        if bin_path.exists() and idx_path.exists():
            try:
                offsets = np.load(idx_path, mmap_mode='r')
                if int(offsets.shape[0]) == int(expected_n) + 1:
                    return
            except Exception:
                pass

        env = self._get_cluster_env(db_name)
        offsets = [0]
        cur = 0
        with env.begin(write=False) as txn, open(bin_path, 'wb') as f:
            c = txn.cursor()
            for k, _ in c:
                if k.startswith(b'-'):
                    continue
                b = bytes(k)
                f.write(b)
                cur += len(b)
                offsets.append(cur)
        arr = np.asarray(offsets, dtype=np.uint64)
        np.save(idx_path, arr)
        del offsets
        gc.collect()

    def _get_key_by_pos(self, db_name: str, pos: int) -> bytes:
        if db_name not in self._key_offsets_cache:
            _, idx_path = self._cache_paths_for_db(db_name)
            self._key_offsets_cache[db_name] = np.load(idx_path, mmap_mode='r')
        if db_name not in self._key_bin_handles:
            bin_path, _ = self._cache_paths_for_db(db_name)
            self._key_bin_handles[db_name] = open(bin_path, 'rb', buffering=0)

        offs = self._key_offsets_cache[db_name]
        if pos < 0 or pos + 1 >= offs.shape[0]:
            raise IndexError(f"key position out of range: {pos} for {db_name}")
        start = int(offs[pos])
        end = int(offs[pos + 1])
        fh = self._key_bin_handles[db_name]
        fh.seek(start)
        return fh.read(end - start)

    def _load_data(self):
        import torch.distributed as dist
        is_dist = dist.is_initialized()
        is_main = (not is_dist) or dist.get_rank() == 0

        if is_main:
            print("\n" + "=" * 60)
            print("[BidirectionalStage1DS] Distributed data allocation (global sample split + lazy keys)")
            print("=" * 60)

        cluster_db_files = sorted([f for f in self.cluster_lmdb_dir.glob("*_cluster.db")])
        if len(cluster_db_files) == 0:
            raise ValueError(f"No *_cluster.db files found in {self.cluster_lmdb_dir}")

        all_blocks = []
        total_all = 0
        for cluster_path in cluster_db_files:
            db_name = cluster_path.name.replace("_cluster.db", "")
            caption_path = self.caption_lmdb_dir / f"{db_name}.db"
            if not caption_path.exists():
                if is_main:
                    print(f"[BidirectionalStage1DS] Skip {db_name}: caption LMDB not found at {caption_path}")
                continue

            env = lmdb.Environment(str(cluster_path), readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                n_cells = int(txn.stat().get('entries', 0))
            env.close()

            all_blocks.append({
                "cluster_path": str(cluster_path),
                "cluster_db_name": cluster_path.name,
                "caption_path": str(caption_path),
                "n_cells": n_cells,
                "global_start": total_all,
                "global_end": total_all + n_cells,
            })
            total_all += n_cells

        if total_all <= 0:
            raise ValueError("No usable cells found in cluster DBs")

        base = total_all // self.num_replicas
        rem = total_all % self.num_replicas
        rank_start = self.rank * base + min(self.rank, rem)
        rank_count = base + (1 if self.rank < rem else 0)
        rank_end = rank_start + rank_count

        if is_main:
            print(f"[BidirectionalStage1DS] Total cells(all ranks): {total_all:,}")
            print(f"[BidirectionalStage1DS] Per-rank target ~= {base:,} (+1 for first {rem} ranks)")

        self.data_blocks = []
        self.cumulative_sizes = [0]

        for blk in all_blocks:
            s0, e0 = blk['global_start'], blk['global_end']
            s = max(s0, rank_start)
            e = min(e0, rank_end)
            if e <= s:
                continue
            local_start = s - s0
            local_end = e - s0
            n_local = local_end - local_start
            b = dict(blk)
            b['local_start'] = int(local_start)
            b['local_end'] = int(local_end)
            b['n_local'] = int(n_local)
            self.data_blocks.append(b)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + n_local)

        # one-time key-cache build on rank0, others wait
        if is_main:
            for blk in all_blocks:
                self._build_key_cache_if_needed(blk['cluster_db_name'], blk['n_cells'])
        if is_dist:
            dist.barrier()

        self.total_cells = self.cumulative_sizes[-1]
        print(f"[BidirectionalStage1DS] Rank {self.rank}: cells={self.total_cells:,}, blocks={len(self.data_blocks)}")

    def __len__(self) -> int:
        return self.total_cells

    def _build_flat_gene_tokens(self, scgpt_ids: np.ndarray, cluster_indices: np.ndarray) -> List[int]:
        flat_tokens = []
        for gid, cid in zip(scgpt_ids, cluster_indices):
            local_id = self.scgpt_to_local.get(int(gid))
            if local_id is not None:
                token_id = self.text_vocab_size + (local_id * 64 + int(cid))
            else:
                # fallback: mask gene for unknown genes
                token_id = self.mask_gene_id
            flat_tokens.append(token_id)
        return flat_tokens

    def _sample_masked_gene_ids(self, flat_gene_tokens: List[int]) -> tuple:
        valid_positions = [i for i, tid in enumerate(flat_gene_tokens) if tid != self.mask_gene_id]
        if len(valid_positions) < 10:
            raise ValueError("Not enough valid gene positions")

        low = max(0.01, min(float(self.mask_min_ratio), 0.99))
        high = max(low, min(float(self.mask_max_ratio), 0.99))
        t = random.uniform(low, high)
        mask_ratio = t
        n_mask = max(1, int(len(valid_positions) * mask_ratio))
        mask_positions = random.sample(valid_positions, n_mask)

        masked_gene_ids = flat_gene_tokens.copy()
        for pos in mask_positions:
            masked_gene_ids[pos] = self.mask_gene_id

        gene_mask_inner = torch.tensor(
            [tid == self.mask_gene_id for tid in masked_gene_ids], dtype=torch.bool
        )
        return flat_gene_tokens, masked_gene_ids, gene_mask_inner, t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        block_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[block_idx]
        block = self.data_blocks[block_idx]

        key_pos = int(block['local_start'] + local_idx)
        key = self._get_key_by_pos(block["cluster_db_name"], key_pos)

        if self.use_cell_cls:
            # CellAwareFeatureLoader 返回 record + 融合 embedding + token_ids
            raw = self.cell_aware_loader.get_raw(block["cluster_db_name"], key)
            if raw is None:
                return self.__getitem__((idx + 1) % len(self))
            record, lookup = raw
            fused_emb, token_ids_arr, valid_mask = self.cell_aware_loader._fuse(record, lookup)

            # 用 token_ids 构建 flat_gene_tokens（unknown=-1 时 fallback 到 mask）
            flat_gene_tokens = []
            for tid in token_ids_arr:
                if tid >= 0:
                    flat_gene_tokens.append(int(tid) + self.text_vocab_size)
                else:
                    flat_gene_tokens.append(self.mask_gene_id)

            # 仍保留原始 scgpt_ids 以便构建 gene_labels
            scgpt_ids = record.scgpt_ids
            gene_embeddings = torch.from_numpy(fused_emb).float()
            non_zero_mask = torch.from_numpy(valid_mask.astype(np.bool_))
        else:
            record = self.feature_loader.get(block["cluster_db_name"], key)
            if record is None:
                return self.__getitem__((idx + 1) % len(self))
            scgpt_ids = record.scgpt_ids
            cluster_indices = record.cluster_indices
            flat_gene_tokens = self._build_flat_gene_tokens(scgpt_ids, cluster_indices)
            gene_embeddings = None
            non_zero_mask = torch.tensor(scgpt_ids != 0, dtype=torch.bool)

        # Load metadata from caption LMDB
        caption_env = self._get_caption_env(block["caption_path"])
        metadata = {}
        with caption_env.begin(write=False) as txn:
            sample_data = txn.get(key)
            if sample_data:
                try:
                    lmdb_res = json.loads(sample_data.decode())
                    metadata = {field: lmdb_res.get(field, "") for field in self.data_config.get("metadata_fields", [])}
                except Exception:
                    pass

        is_understanding = random.random() < self.understanding_ratio
        original_gene_tokens, masked_gene_ids, gene_mask_inner, t = self._sample_masked_gene_ids(flat_gene_tokens)


        system_prompt = "You are a helpful assistant."

        if is_understanding:
            return self._build_understanding_sample(
                metadata, original_gene_tokens, masked_gene_ids, gene_mask_inner, t, non_zero_mask,
                local_idx, block["cluster_path"], system_prompt, scgpt_ids, gene_embeddings
            )
        else:
            return self._build_generation_sample(
                metadata, original_gene_tokens, masked_gene_ids, gene_mask_inner, t, non_zero_mask,
                local_idx, block["cluster_path"], system_prompt, scgpt_ids, gene_embeddings
            )

    def _build_understanding_sample(
        self, metadata, original_gene_tokens, masked_gene_ids, gene_mask_inner, t, non_zero_mask,
        local_idx, cluster_path, system_prompt, scgpt_ids, gene_embeddings=None
    ) -> Dict[str, Any]:
        q_text, a_text = self.formatter.format(metadata)

        input_ids = []
        labels = []

        if self.bos_id is not None:
            input_ids.append(self.bos_id)
            labels.append(-100)

        sys_tokens = self.text_tokenizer.encode(
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n", add_special_tokens=False
        )
        input_ids.extend(sys_tokens)
        labels.extend([-100] * len(sys_tokens))

        u_header = self.text_tokenizer.encode(
            "<|im_start|>user\n", add_special_tokens=False
        )
        input_ids.extend(u_header)
        labels.extend([-100] * len(u_header))

        sog_pos = len(input_ids)
        input_ids.append(self.sog_id)
        labels.append(-100)
        gene_start_pos = len(input_ids)
        input_ids.extend(masked_gene_ids)
        labels.extend([-100] * len(masked_gene_ids))
        input_ids.append(self.eog_id)
        labels.append(-100)
        gene_end_pos = len(input_ids) - 1

        if q_text:
            q_body = self.text_tokenizer.encode(f"{q_text}<|im_end|>\n", add_special_tokens=False)
        else:
            q_body = self.text_tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        input_ids.extend(q_body)
        labels.extend([-100] * len(q_body))

        a_header = self.text_tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        a_body = self.text_tokenizer.encode(
            f"{a_text}<|im_end|>\n", add_special_tokens=False
        )
        input_ids.extend(a_header)
        labels.extend([-100] * len(a_header))
        input_ids.extend(a_body)
        labels.extend(a_body)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        seq_len = len(input_ids_tensor)
        position_ids = torch.arange(seq_len, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor)

        for j, pos in enumerate(range(gene_start_pos, gene_end_pos)):
            if input_ids_tensor[pos] == self.mask_gene_id and original_gene_tokens[j] != self.mask_gene_id:
                labels_tensor[pos] = int(scgpt_ids[j])

        # gene_labels 用于 diffusion loss：预测 scGPT gene id，pad 位置设为 -100
        gene_labels_arr = scgpt_ids.copy().astype(np.int64)
        gene_labels_arr[gene_labels_arr == self.scgpt_pad_id] = -100

        celltype_name = str(metadata.get("celltype_name", "") or "")
        celltype_id = self._stable_celltype_id(celltype_name)

        result = {
            "input_ids": input_ids_tensor,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
            "gene_labels": torch.tensor(gene_labels_arr, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.float32),
            "texts": f"{q_text} -> {a_text}",
            "modality_positions": torch.tensor([[gene_start_pos, len(original_gene_tokens)]], dtype=torch.long),
            "gene_mask": gene_mask_inner,
            "non_zero_mask": non_zero_mask,
            "celltype_name": celltype_name,
            "celltype_id": torch.tensor(celltype_id, dtype=torch.long),
            "metadata": {
                "cell_id": local_idx,
                "lmdb_source": cluster_path,
                "mode": "understanding",
            },
            "data_type": ["stage1_understanding"],
        }
        if gene_embeddings is not None:
            result["gene_embeddings"] = gene_embeddings
        return result

    def _build_generation_sample(
        self, metadata, original_gene_tokens, masked_gene_ids, gene_mask_inner, t, non_zero_mask,
        local_idx, cluster_path, system_prompt, scgpt_ids, gene_embeddings=None
    ) -> Dict[str, Any]:
        cond_keep_prob = 1.0 - min(max(self.cond_dropout_prob, 0.0), 1.0)
        if random.random() < cond_keep_prob:
            _, prompt_text = self.formatter.format_caption_only(metadata)
        else:
            prompt_text = ""

        input_ids = []
        labels = []

        if self.bos_id is not None:
            input_ids.append(self.bos_id)
            labels.append(-100)

        sys_tokens = self.text_tokenizer.encode(
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n", add_special_tokens=False
        )
        input_ids.extend(sys_tokens)
        labels.extend([-100] * len(sys_tokens))

        u_header = self.text_tokenizer.encode(
            "<|im_start|>user\n", add_special_tokens=False
        )
        input_ids.extend(u_header)
        labels.extend([-100] * len(u_header))

        if prompt_text:
            p_body = self.text_tokenizer.encode(f"{prompt_text}<|im_end|>\n", add_special_tokens=False)
        else:
            p_body = self.text_tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        input_ids.extend(p_body)
        labels.extend([-100] * len(p_body))

        a_header = self.text_tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        input_ids.extend(a_header)
        labels.extend([-100] * len(a_header))

        gene_start_pos = len(input_ids)
        input_ids.append(self.sog_id)
        labels.append(-100)
        input_ids.extend(masked_gene_ids)
        labels.extend([-100] * len(masked_gene_ids))
        input_ids.append(self.eog_id)
        labels.append(-100)
        gene_end_pos = len(input_ids) - 1

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        seq_len = len(input_ids_tensor)
        position_ids = torch.arange(seq_len, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor)

        for j, pos in enumerate(range(gene_start_pos + 1, gene_end_pos)):
            if input_ids_tensor[pos] == self.mask_gene_id and original_gene_tokens[j] != self.mask_gene_id:
                labels_tensor[pos] = int(scgpt_ids[j])

        # gene_labels 用于 diffusion loss：预测 scGPT gene id，pad 位置设为 -100
        gene_labels_arr = scgpt_ids.copy().astype(np.int64)
        gene_labels_arr[gene_labels_arr == self.scgpt_pad_id] = -100

        celltype_name = str(metadata.get("celltype_name", "") or "")
        celltype_id = self._stable_celltype_id(celltype_name)

        result = {
            "input_ids": input_ids_tensor,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
            "gene_labels": torch.tensor(gene_labels_arr, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.float32),
            "texts": prompt_text if prompt_text else "<unconditional>",
            "modality_positions": torch.tensor([[gene_start_pos + 1, len(original_gene_tokens)]], dtype=torch.long),
            "gene_mask": gene_mask_inner,
            "non_zero_mask": non_zero_mask,
            "celltype_name": celltype_name,
            "celltype_id": torch.tensor(celltype_id, dtype=torch.long),
            "metadata": {
                "cell_id": local_idx,
                "lmdb_source": cluster_path,
                "mode": "generation",
            },
            "data_type": ["stage1_generation"],
        }
        if gene_embeddings is not None:
            result["gene_embeddings"] = gene_embeddings
        return result

    def __del__(self):
        try:
            for fh in self._key_bin_handles.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._key_bin_handles.clear()
            if self.use_cell_cls:
                self.cell_aware_loader.close()
            else:
                self.feature_loader.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['caption_envs'] = {}
        state['_key_offsets_cache'] = {}
        state['_key_bin_handles'] = {}
        if getattr(self, 'use_cell_cls', False):
            state['cell_aware_loader'] = None
        else:
            state['feature_loader'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.caption_envs = {}
        self._key_offsets_cache = {}
        self._key_bin_handles = {}
        if getattr(self, 'use_cell_cls', False):
            from .cell_aware_feature_loader import CellAwareFeatureLoader
            self.cell_aware_loader = CellAwareFeatureLoader(
                cluster_lmdb_dir=self.cluster_lmdb_dir,
                codebook_dir=self.codebook_dir,
                clusters_per_gene=self.clusters_per_gene,
                prefer_compact_codebook=bool(self.data_config.get('prefer_compact_codebook', True)),
                readahead=bool(self.data_config.get('cluster_loader_readahead', False)),
                max_readers=int(self.data_config.get('cluster_loader_max_readers', 2048)),
                gene_weight=float(self.data_config.get('cell_cls_gene_weight', 0.5)),
                cell_weight=float(self.data_config.get('cell_cls_cell_weight', 0.5)),
                cell_dropout=float(self.data_config.get('cell_cls_dropout', 0.3)),
            )
        else:
            self.feature_loader = ClusterFeatureLoader(
                cluster_lmdb_dir=self.cluster_lmdb_dir,
                readahead=bool(self.data_config.get('cluster_loader_readahead', False)),
                max_readers=int(self.data_config.get('cluster_loader_max_readers', 2048)),
            )

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        from torch.nn.utils.rnn import pad_sequence
        from itertools import chain

        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            if k == "data_type":
                batched[k] = list(chain.from_iterable(v))
            elif k not in ("texts", "metadata") and isinstance(v[0], torch.Tensor):
                if v[0].dim() == 1:
                    if k in ("gene_mask", "non_zero_mask"):
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=False)
                    elif k == "labels":
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=-100)
                    elif k == "input_ids":
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=151643)
                    elif k == "position_ids":
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=0)
                    else:
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=0)
                elif v[0].dim() == 2:
                    if k == "modality_positions":
                        batched[k] = torch.stack(v, dim=0)
                    elif k == "gene_embeddings":
                        # gene_embeddings: [L, 768]，需要 pad 到 [B, max_L, 768]
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=0.0)
                    else:
                        batched[k] = torch.stack(v, dim=0)
                elif v[0].dim() == 0:
                    batched[k] = torch.stack(v, dim=0)
                else:
                    batched[k] = torch.stack(v, dim=0)

        return dict(batched)


class WhitelistKeyDataset(Dataset):
    """
    基于白名单 key 列表的 Stage1 数据集包装器。

    复用 BidirectionalStage1Dataset 的所有组件（feature_loader、tokenizer、
    _build_flat_gene_tokens、_sample_masked_gene_ids、_build_*_sample 等），
    但用 key-list 替代 position-based 索引，完全避免全量扫描。
    """

    def __init__(
        self,
        base_dataset: BidirectionalStage1Dataset,
        whitelist_path: str,
        accelerator=None,
    ):
        self.base = base_dataset

        # 读取白名单 JSON
        with open(whitelist_path, "r", encoding="utf-8") as f:
            whitelist = json.load(f)

        # 构建 entries: (cluster_db_name, key_str, caption_path, cluster_path, db_name)
        self.entries = []
        missing_dbs = []
        for db_name, keys in whitelist.items():
            cluster_db_name = f"{db_name}_cluster.db"
            cluster_path = base_dataset.cluster_lmdb_dir / cluster_db_name
            caption_path = base_dataset.caption_lmdb_dir / f"{db_name}.db"

            if not cluster_path.exists():
                missing_dbs.append(db_name)
                continue

            for key in keys:
                self.entries.append((
                    cluster_db_name,
                    key,
                    str(caption_path),
                    str(cluster_path),
                    db_name,
                ))

        if missing_dbs and (accelerator is None or accelerator.is_main_process):
            print(f"[WhitelistKeyDS] Warning: {len(missing_dbs)} DBs not found in cluster_lmdb_dir, skipped")

        # 按 rank 均分
        if accelerator is not None:
            rank = accelerator.process_index
            world_size = accelerator.num_processes
            self.entries = self.entries[rank::world_size]

        self.total_cells = len(self.entries)
        print(f"[WhitelistKeyDS] Rank {getattr(accelerator, 'process_index', 0)}: entries={self.total_cells:,}")

    def __len__(self):
        return self.total_cells

    def __getitem__(self, idx):
        original_idx = idx
        for attempt in range(min(len(self), 5)):
            entry_idx = (original_idx + attempt) % len(self)
            result = self._get_item_inner(entry_idx)
            if result is not None:
                return result
        raise RuntimeError(
            f"WhitelistKeyDataset: 连续 5 个样本读取失败，起始 idx={original_idx}, "
            f"entry=({self.entries[original_idx][4]}/{self.entries[original_idx][1]})"
        )

    def _get_item_inner(self, idx):
        cluster_db_name, key_str, caption_path, cluster_path, db_name = self.entries[idx]
        key_bytes = key_str.encode("utf-8")
        base = self.base

        # 1. 读取 gene record
        try:
            if base.use_cell_cls:
                raw = base.cell_aware_loader.get_raw(cluster_db_name, key_bytes)
                if raw is None:
                    return None
                record, lookup = raw
                fused_emb, token_ids_arr, valid_mask = base.cell_aware_loader._fuse(record, lookup)

                flat_gene_tokens = []
                for tid in token_ids_arr:
                    if tid >= 0:
                        flat_gene_tokens.append(int(tid) + base.text_vocab_size)
                    else:
                        flat_gene_tokens.append(base.mask_gene_id)

                scgpt_ids = record.scgpt_ids
                gene_embeddings = torch.from_numpy(fused_emb).float()
                non_zero_mask = torch.from_numpy(valid_mask.astype(np.bool_))
            else:
                record = base.feature_loader.get(cluster_db_name, key_bytes)
                if record is None:
                    return None
                scgpt_ids = record.scgpt_ids
                cluster_indices = record.cluster_indices
                flat_gene_tokens = base._build_flat_gene_tokens(scgpt_ids, cluster_indices)
                gene_embeddings = None
                non_zero_mask = torch.tensor(scgpt_ids != 0, dtype=torch.bool)
        except Exception as e:
            print(f"[WhitelistKeyDS] Error reading gene record for {db_name}/{key_str}: {e}")
            return None

        # 2. 读取 caption metadata
        metadata = {}
        try:
            caption_env = base._get_caption_env(caption_path)
            if caption_env is not None:
                with caption_env.begin(write=False) as txn:
                    sample_data = txn.get(key_bytes)
                    if sample_data:
                        try:
                            lmdb_res = json.loads(sample_data.decode("utf-8", errors="ignore"))
                            metadata = {
                                field: lmdb_res.get(field, "")
                                for field in base.data_config.get("metadata_fields", [])
                            }
                        except Exception:
                            pass
        except Exception:
            pass

        # 3. mask & build sample
        is_understanding = random.random() < base.understanding_ratio
        try:
            original_gene_tokens, masked_gene_ids, gene_mask_inner, t = \
                base._sample_masked_gene_ids(flat_gene_tokens)
        except Exception:
            return None

        system_prompt = "You are a helpful assistant."

        if is_understanding:
            return base._build_understanding_sample(
                metadata, original_gene_tokens, masked_gene_ids, gene_mask_inner,
                t, non_zero_mask, idx, cluster_path, system_prompt, scgpt_ids, gene_embeddings
            )
        else:
            return base._build_generation_sample(
                metadata, original_gene_tokens, masked_gene_ids, gene_mask_inner,
                t, non_zero_mask, idx, cluster_path, system_prompt, scgpt_ids, gene_embeddings
            )
