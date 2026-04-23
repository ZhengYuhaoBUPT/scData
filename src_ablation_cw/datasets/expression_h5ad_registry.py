# coding=utf-8
from __future__ import annotations

import os
from typing import Dict, Sequence, Tuple

import anndata as ad
import numpy as np
import torch

from src_ablation_cw.datasets.gene_token_utils import load_static_gene_bundle


os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


class ExpressionH5ADRegistry:
    def __init__(
        self,
        h5ad_paths: Sequence[str],
        static_gene_ckpt_path: str,
        gene_input_tokens: int = 1200,
        gene_rank_order: str = "desc",
    ):
        if not h5ad_paths:
            raise ValueError("gene_h5ad_paths is empty")
        self.paths = [str(p) for p in h5ad_paths]
        self.gene_input_tokens = int(gene_input_tokens)
        self.gene_rank_order = str(gene_rank_order)

        bundle = load_static_gene_bundle(static_gene_ckpt_path)
        self.static_gene_embeddings: torch.Tensor = bundle["static_gene_embeddings"]
        self.gene_to_idx: Dict[str, int] = bundle["gene_to_idx"]
        self.scgpt_id_to_idx: Dict[int, int] = bundle["scgpt_id_to_idx"]

        self._adatas: Dict[str, ad.AnnData] = {}
        self._cell_to_loc: Dict[str, Tuple[str, int]] = {}
        self._mapped_cols_by_path: Dict[str, np.ndarray] = {}
        self._mapped_gene_idx_by_path: Dict[str, np.ndarray] = {}
        self._has_rank_by_path: Dict[str, bool] = {}
        self._build_index()

    def _open_adata(self, path: str):
        adata = self._adatas.get(path)
        if adata is None:
            adata = ad.read_h5ad(path, backed="r")
            self._adatas[path] = adata
        return adata

    def _close_adata(self, adata) -> None:
        file_obj = getattr(adata, "file", None)
        if file_obj is not None:
            try:
                file_obj.close()
            except Exception:
                pass

    def _build_index(self):
        for path in self.paths:
            adata = ad.read_h5ad(path, backed="r")
            try:
                if "cell_id" in adata.obs.columns:
                    obs_ids = [str(x) for x in adata.obs["cell_id"].tolist()]
                else:
                    obs_ids = [str(x) for x in adata.obs.index.tolist()]
                for i, cid in enumerate(obs_ids):
                    self._cell_to_loc.setdefault(cid, (path, i))

                has_rank = "rank" in adata.obsm.keys()
                self._has_rank_by_path[path] = bool(has_rank)
                if has_rank:
                    continue

                if "gene_name" in adata.var.columns:
                    gene_names = adata.var["gene_name"].astype(str).tolist()
                else:
                    gene_names = [str(x) for x in adata.var.index.tolist()]

                mapped_cols = []
                mapped_gene_idx = []
                for col_idx, gene_name in enumerate(gene_names):
                    static_idx = self.gene_to_idx.get(gene_name)
                    if static_idx is None:
                        continue
                    mapped_cols.append(col_idx)
                    mapped_gene_idx.append(static_idx)
                self._mapped_cols_by_path[path] = np.asarray(mapped_cols, dtype=np.int64)
                self._mapped_gene_idx_by_path[path] = np.asarray(mapped_gene_idx, dtype=np.int64)
            finally:
                self._close_adata(adata)

    def _load_rank_gene_tokens(self, adata, row_idx: int) -> torch.Tensor:
        rank_ids = np.asarray(adata.obsm["rank"][row_idx], dtype=np.int64).reshape(-1)
        if self.gene_rank_order == "asc":
            ordered_ids = rank_ids[::-1]
        else:
            ordered_ids = rank_ids

        selected_gene_indices = []
        for gid in ordered_ids:
            idx = self.scgpt_id_to_idx.get(int(gid))
            if idx is None:
                continue
            selected_gene_indices.append(idx)
            if len(selected_gene_indices) >= self.gene_input_tokens:
                break

        out = torch.zeros(self.gene_input_tokens, self.static_gene_embeddings.shape[1], dtype=torch.float32)
        if selected_gene_indices:
            picked = self.static_gene_embeddings[selected_gene_indices]
            out[: picked.shape[0]] = picked
        return out

    def _load_row_values(self, adata, row_idx: int, cols: np.ndarray) -> np.ndarray:
        row = adata.X[row_idx, cols]
        if hasattr(row, "toarray"):
            row = row.toarray()
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        return row

    def _load_expression_gene_tokens(self, adata, path: str, row_idx: int) -> torch.Tensor:
        cols = self._mapped_cols_by_path[path]
        gene_idx = self._mapped_gene_idx_by_path[path]
        values = self._load_row_values(adata, row_idx, cols)

        if values.size == 0:
            return torch.zeros(self.gene_input_tokens, self.static_gene_embeddings.shape[1], dtype=torch.float32)

        order = np.argsort(values)
        if self.gene_rank_order == "desc":
            order = order[::-1]
        order = order[: self.gene_input_tokens]
        selected_gene_idx = gene_idx[order]

        out = torch.zeros(self.gene_input_tokens, self.static_gene_embeddings.shape[1], dtype=torch.float32)
        picked = self.static_gene_embeddings[selected_gene_idx]
        out[: picked.shape[0]] = picked
        return out

    def has_cell(self, cell_id: str) -> bool:
        return str(cell_id) in self._cell_to_loc

    def get_gene_tokens(self, cell_id: str) -> torch.Tensor:
        loc = self._cell_to_loc.get(str(cell_id))
        if loc is None:
            raise KeyError(f"cell_id not found in gene_h5ad_paths: {cell_id}")
        path, row_idx = loc
        adata = self._open_adata(path)

        if self._has_rank_by_path.get(path, False):
            return self._load_rank_gene_tokens(adata, row_idx)
        return self._load_expression_gene_tokens(adata, path, row_idx)
