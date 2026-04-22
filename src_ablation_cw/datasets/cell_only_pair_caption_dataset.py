# coding=utf-8
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import lmdb
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from src_ablation_cw.datasets.gene_token_utils import (
    build_gene_sequence_from_rank,
    load_static_gene_bundle,
)
from src_ablation_cw.datasets.metadata_formatter import MetadataFormatter


os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


class CellOnlyPairCaptionDataset(Dataset):
    def __init__(
        self,
        h5ad_paths: Sequence[str],
        lmdb_base_dir: Optional[str],
        text_tokenizer: Any,
        config_dict: Dict[str, Any],
        special_tokens_ids: Dict[str, int],
        accelerator=None,
        data_type_tag: str = "stage_pair",
        max_samples: int = 0,
        sample_seed: int = 1,
    ):
        super().__init__()

        self.dataset_config = config_dict.get("dataset", {})
        self.max_seq_len = int(self.dataset_config.get("max_seq_len", 1024))
        self.cell_feature_tokens = int(self.dataset_config.get("cell_feature_tokens", 8))
        self.gene_input_tokens = int(self.dataset_config.get("gene_input_tokens", 1200))
        self.gene_rank_order = str(self.dataset_config.get("gene_rank_order", "desc"))
        self.data_type_tag = str(data_type_tag)
        self.max_samples = int(max_samples)
        self.sample_seed = int(sample_seed)

        model_cfg = config_dict.get("model", {})
        static_gene_ckpt_path = model_cfg.get("static_gene_embedding_ckpt_path")
        if not static_gene_ckpt_path:
            raise ValueError("model.static_gene_embedding_ckpt_path is required for pair dataset")
        bundle = load_static_gene_bundle(static_gene_ckpt_path)
        self.static_gene_embeddings = bundle["static_gene_embeddings"]
        self.scgpt_id_to_idx = bundle["scgpt_id_to_idx"]

        self.text_tokenizer = text_tokenizer
        self.soc_id = special_tokens_ids.get("soc_id")
        self.eoc_id = special_tokens_ids.get("eoc_id")
        self.bos_id = getattr(text_tokenizer, "bos_token_id", None)
        self.pad_id = text_tokenizer.pad_token_id if getattr(text_tokenizer, "pad_token_id", None) is not None else 151643

        self.accelerator = accelerator
        self.num_replicas = accelerator.num_processes if accelerator else 1
        self.rank = accelerator.process_index if accelerator else 0

        self.formatter = MetadataFormatter(
            celltype_question_weight=0.65,
            field_dropout_prob=0.15,
            short_prob=0.10,
            qa_prob=0.40,
        )

        self.h5ad_paths = [str(p) for p in h5ad_paths if p]
        if not self.h5ad_paths:
            raise ValueError("h5ad_paths is empty for pair dataset")
        self.lmdb_base_dir = str(lmdb_base_dir) if lmdb_base_dir else None

        self._adatas: Dict[str, ad.AnnData] = {}
        self._lmdb_envs: Dict[str, lmdb.Environment] = {}
        self.blocks: List[Dict[str, Any]] = []
        self.total_cells = 0
        self._build_index()

        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if is_main:
            num_with_lmdb = sum(1 for x in self.blocks if x.get("lmdb_path"))
            print(
                f"[PairDS] rank={self.rank} total_cells={self.total_cells} max_samples={self.max_samples} "
                f"with_lmdb={num_with_lmdb}"
            )

    def _resolve_lmdb_path(self, h5ad_path: str) -> Optional[str]:
        h5_path = Path(h5ad_path)
        candidates: List[Path] = []
        if self.lmdb_base_dir:
            candidates.append(Path(self.lmdb_base_dir) / f"{h5_path.stem}.db")
        candidates.append(h5_path.with_suffix(".db"))
        candidates.append(h5_path.parent / f"{h5_path.stem}.db")
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _open_adata(self, path: str):
        adata = self._adatas.get(path)
        if adata is None:
            adata = ad.read_h5ad(path, backed="r")
            self._adatas[path] = adata
        return adata

    def _get_lmdb_env(self, lmdb_path: str):
        env = self._lmdb_envs.get(lmdb_path)
        if env is None:
            env = lmdb.Environment(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            self._lmdb_envs[lmdb_path] = env
        return env

    def _build_index(self) -> None:
        file_infos: List[Dict[str, Any]] = []
        total_all = 0
        for h5ad_path in self.h5ad_paths:
            lmdb_path = self._resolve_lmdb_path(h5ad_path)
            adata = ad.read_h5ad(h5ad_path, backed="r")
            try:
                n_cells = int(adata.n_obs)
            finally:
                file_obj = getattr(adata, "file", None)
                if file_obj is not None:
                    try:
                        file_obj.close()
                    except Exception:
                        pass
            file_infos.append({
                "h5ad_path": str(h5ad_path),
                "lmdb_path": lmdb_path,
                "n_cells": n_cells,
                "global_start": total_all,
                "global_end": total_all + n_cells,
            })
            total_all += n_cells

        usable = (total_all // self.num_replicas) * self.num_replicas if self.num_replicas > 0 else total_all
        per_rank = usable // self.num_replicas if self.num_replicas > 0 else total_all
        my_start = self.rank * per_rank
        my_end = my_start + per_rank

        sample_locs: List[Dict[str, Any]] = []
        for info in file_infos:
            overlap_start = max(my_start, info["global_start"])
            overlap_end = min(my_end, info["global_end"])
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - info["global_start"]
            local_end = overlap_end - info["global_start"]
            for row_idx in range(local_start, local_end):
                sample_locs.append({
                    "h5ad_path": info["h5ad_path"],
                    "lmdb_path": info["lmdb_path"],
                    "row_idx": int(row_idx),
                })

        if self.max_samples > 0 and len(sample_locs) > self.max_samples:
            rnd = random.Random(self.sample_seed + self.rank)
            sample_locs = rnd.sample(sample_locs, self.max_samples)

        sample_locs.sort(key=lambda x: (x["h5ad_path"], x["row_idx"]))
        self.blocks = sample_locs
        self.total_cells = len(self.blocks)

    def __len__(self) -> int:
        return self.total_cells

    @staticmethod
    def _clean_obs_metadata(obs_row: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        ignore_keys = {
            "gene_id",
            "lmdb_key",
            "_index",
        }
        for k, v in obs_row.items():
            if k in ignore_keys:
                continue
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            s = str(v).strip()
            if not s or s.lower() in {"nan", "none"}:
                continue
            metadata[k] = v
        return metadata

    def _load_lmdb_metadata(self, lmdb_path: Optional[str], lmdb_key: Optional[str]) -> Dict[str, Any]:
        if not lmdb_path or not lmdb_key:
            return {}
        env = self._get_lmdb_env(lmdb_path)
        with env.begin(write=False) as txn:
            sample_data = txn.get(lmdb_key.encode())
            if not sample_data:
                return {}
            try:
                return json.loads(sample_data.decode())
            except Exception:
                return {}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.blocks[idx]
        adata = self._open_adata(entry["h5ad_path"])
        row_idx = int(entry["row_idx"])

        rank_ids = np.asarray(adata.obsm["rank"][row_idx], dtype=np.int64).reshape(-1)
        cell_feature = build_gene_sequence_from_rank(
            rank_ids=rank_ids,
            scgpt_id_to_idx=self.scgpt_id_to_idx,
            static_gene_embeddings=self.static_gene_embeddings,
            gene_input_tokens=self.gene_input_tokens,
            rank_order=self.gene_rank_order,
        )

        obs_row = adata.obs.iloc[row_idx].to_dict()
        metadata = self._clean_obs_metadata(obs_row)
        lmdb_key = str(obs_row.get("lmdb_key", "")).strip() if "lmdb_key" in obs_row else None
        lmdb_metadata = self._load_lmdb_metadata(entry.get("lmdb_path"), lmdb_key)
        if lmdb_metadata:
            metadata = {**metadata, **lmdb_metadata}

        question, answer = self.formatter.format(metadata)

        input_ids = []
        labels = []

        if self.bos_id is not None:
            input_ids.append(self.bos_id)
            labels.append(-100)

        sys_tokens = self.text_tokenizer.encode(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            add_special_tokens=False,
        )
        input_ids.extend(sys_tokens)
        labels.extend([-100] * len(sys_tokens))

        cell_tokens = [self.soc_id] + [self.pad_id] * self.cell_feature_tokens + [self.eoc_id]
        u_header = self.text_tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        input_ids.extend(u_header)
        labels.extend([-100] * len(u_header))

        soc_pos = len(input_ids)
        input_ids.extend(cell_tokens)
        labels.extend([-100] * len(cell_tokens))
        cell_start_pos = soc_pos + 1

        u_body = self.text_tokenizer.encode(f"{question}<|im_end|>\n", add_special_tokens=False)
        input_ids.extend(u_body)
        labels.extend([-100] * len(u_body))

        a_header = self.text_tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        a_body = self.text_tokenizer.encode(f"{answer}<|im_end|>\n", add_special_tokens=False)
        input_ids.extend(a_header)
        labels.extend([-100] * len(a_header))
        input_ids.extend(a_body)
        labels.extend(a_body)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor, dtype=torch.bool)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask,
            "cell_features": cell_feature,
            "cell_positions": torch.tensor([cell_start_pos, self.cell_feature_tokens], dtype=torch.long),
            "data_type": self.data_type_tag,
        }
