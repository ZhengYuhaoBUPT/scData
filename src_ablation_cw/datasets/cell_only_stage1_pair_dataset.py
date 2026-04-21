# coding=utf-8
"""
Cell-Only Stage1 Pair Dataset.
Converts paired cell metadata into diverse ChatML-format conversations
for autoregressive NTP training (no gene tokens).
"""

import bisect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import anndata as ad
import h5py
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from src_ablation_cw.datasets.metadata_formatter import MetadataFormatter
from src_ablation_cw.datasets.expression_h5ad_registry import ExpressionH5ADRegistry


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class CellOnlyStage1PairDataset(Dataset):
    def __init__(
        self,
        feature_dir: str,
        lmdb_base_dir: str,
        text_tokenizer: Any,
        config_dict: Dict[str, Any],
        special_tokens_ids: Dict[str, int],
        accelerator=None,
    ):
        super().__init__()

        self.data_config = config_dict.get("data", {})
        self.dataset_config = config_dict.get("dataset", {})

        self.feature_dir = Path(feature_dir) if feature_dir else Path(self.data_config.get("feature_dir", ""))
        self.lmdb_base_dir = Path(lmdb_base_dir) if lmdb_base_dir else Path(self.data_config.get("lmdb_base_dir", ""))

        self.max_seq_len = int(self.dataset_config.get("max_seq_len", 1024))
        self.cell_feature_tokens = int(self.dataset_config.get("cell_feature_tokens", 8))
        self.cell_feature_dim = int(self.dataset_config.get("cell_feature_dim", 768))
        self.gene_input_tokens = int(self.dataset_config.get("gene_input_tokens", 1200))
        self.gene_rank_order = str(self.dataset_config.get("gene_rank_order", "asc"))

        model_cfg = config_dict.get("model", {})
        static_gene_ckpt_path = model_cfg.get("static_gene_embedding_ckpt_path")
        if not static_gene_ckpt_path:
            raise ValueError("model.static_gene_embedding_ckpt_path is required for gene-token training")
        bundle = load_static_gene_bundle(static_gene_ckpt_path)
        self.static_gene_embeddings = bundle["static_gene_embeddings"]
        self.scgpt_id_to_idx = bundle["scgpt_id_to_idx"]

        self.accelerator = accelerator
        self.num_replicas = accelerator.num_processes if accelerator else 1
        self.rank = accelerator.process_index if accelerator else 0

        self.text_tokenizer = text_tokenizer
        self.soc_id = special_tokens_ids.get("soc_id")
        self.eoc_id = special_tokens_ids.get("eoc_id")
        self.bos_id = getattr(text_tokenizer, "bos_token_id", None)
        self.pad_id = text_tokenizer.pad_token_id if getattr(text_tokenizer, "pad_token_id", None) is not None else 151643

        self.formatter = MetadataFormatter(
            celltype_question_weight=0.65,
            field_dropout_prob=0.15,
            short_prob=0.10,
            qa_prob=0.40,
        )

        self.lmdb_envs = {}
        self._load_data()

        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main:
            print(f"[Stage1PairDS] Rank {self.rank}: loaded {self.total_cells} cells from {len(self.data_blocks)} blocks.")

    def _load_data(self):
        self.data_blocks = []
        self.lmdb_paths = []
        self.cumulative_sizes = [0]
        self.total_cells = 0

    def _get_lmdb_env(self, lmdb_path: str):
        if lmdb_path not in self.lmdb_envs:
            self.lmdb_envs[lmdb_path] = lmdb.Environment(
                lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self.lmdb_envs[lmdb_path]

    def __len__(self) -> int:
        return self.total_cells

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        block_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[block_idx]
        block = self.data_blocks[block_idx]

        rank_ids = block["rank"][local_idx]
        cell_feature = build_gene_sequence_from_rank(
            rank_ids=rank_ids,
            scgpt_id_to_idx=self.scgpt_id_to_idx,
            static_gene_embeddings=self.static_gene_embeddings,
            gene_input_tokens=self.gene_input_tokens,
            rank_order=self.gene_rank_order,
        )

        lmdb_path = self.lmdb_paths[block_idx]
        env = self._get_lmdb_env(lmdb_path)
        lmdb_key = block["lmdb_keys"][local_idx]

        metadata = {}
        with env.begin(write=False) as txn:
            sample_data = txn.get(str(lmdb_key).encode())
            if sample_data:
                try:
                    metadata = json.loads(sample_data.decode())
                except Exception:
                    pass

        question, answer = self.formatter.format(metadata)

        # Build ChatML tokens (strictly aligned with CWSFTCellOnlyDataset)
        input_ids = []
        labels = []

        if self.bos_id is not None:
            input_ids.append(self.bos_id)
            labels.append(-100)

        system_prompt = "You are a helpful assistant."
        sys_tokens = self.text_tokenizer.encode(
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n",
            add_special_tokens=False,
        )
        input_ids.extend(sys_tokens)
        labels.extend([-100] * len(sys_tokens))

        cell_tokens = [self.soc_id] + [self.pad_id] * self.cell_feature_tokens + [self.eoc_id]
        cell_start_pos = 0

        # User turn
        u_header = self.text_tokenizer.encode(
            "<|im_start|>user\n", add_special_tokens=False
        )
        input_ids.extend(u_header)
        labels.extend([-100] * len(u_header))

        # Inject cell tokens AFTER user header and BEFORE question body
        soc_pos = len(input_ids)
        input_ids.extend(cell_tokens)
        labels.extend([-100] * len(cell_tokens))
        cell_start_pos = soc_pos + 1

        # If question is not empty, encode it before <|im_end|>
        if question:
            u_body_str = f"{question}<|im_end|>\n"
        else:
            u_body_str = "<|im_end|>\n"
        u_body = self.text_tokenizer.encode(u_body_str, add_special_tokens=False)
        input_ids.extend(u_body)
        labels.extend([-100] * len(u_body))

        # Assistant turn
        a_header = self.text_tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        a_body = self.text_tokenizer.encode(
            f"{answer}<|im_end|>\n", add_special_tokens=False
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
        attention_mask = torch.ones_like(input_ids_tensor, dtype=torch.bool)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask,
            "cell_features": cell_feature,
            "cell_positions": torch.tensor([cell_start_pos, self.cell_feature_tokens], dtype=torch.long),
            "data_type": "stage1_pair",
        }
