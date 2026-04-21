# coding=utf-8
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import anndata as ad
import numpy as np
import torch

from src_ablation_cw.datasets.gene_token_utils import load_static_gene_bundle


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
        self._adatas: Dict[str, ad.AnnData] = {}
        self._cell_to_loc: Dict[str, Tuple[str, int]] = {}
        self._mapped_cols_by_path: Dict[str, np.ndarray] = {}
        self._mapped_gene_idx_by_path: Dict[str, np.ndarray] = {}
        self._build_index()

    def _open_adata(self, path: str):
        adata = self._adatas.get(path)
        if adata is None:
            adata = ad.read_h5ad(path, backed='r')
            self._adatas[path] = adata
        return adata

    def _build_index(self):
        for path in self.paths:
            adata = self._open_adata(path)
            obs_ids = [str(x) for x in adata.obs.index]
            for i, cid in enumerate(obs_ids):
                self._cell_to_loc.setdefault(cid, (path, i))
            gene_names = None
            if 'gene_name' in adata.var.columns:
                gene_names = adata.var['gene_name'].astype(str).tolist()
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

    def _load_row_values(self, adata, row_idx: int, cols: np.ndarray) -> np.ndarray:
        row = adata.X[row_idx, cols]
        if hasattr(row, 'toarray'):
            row = row.toarray()
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        return row

    def get_gene_tokens(self, cell_id: str) -> torch.Tensor:
        loc = self._cell_to_loc.get(str(cell_id))
        if loc is None:
            raise KeyError(f"cell_id not found in gene_h5ad_paths: {cell_id}")
        path, row_idx = loc
        adata = self._open_adata(path)
        cols = self._mapped_cols_by_path[path]
        gene_idx = self._mapped_gene_idx_by_path[path]
        values = self._load_row_values(adata, row_idx, cols)

        if values.size == 0:
            return torch.zeros(self.gene_input_tokens, self.static_gene_embeddings.shape[1], dtype=torch.float32)

        order = np.argsort(values)
        if self.gene_rank_order == 'desc':
            order = order[::-1]
        order = order[: self.gene_input_tokens]
        selected_gene_idx = gene_idx[order]

        out = torch.zeros(self.gene_input_tokens, self.static_gene_embeddings.shape[1], dtype=torch.float32)
        picked = self.static_gene_embeddings[selected_gene_idx]
        out[: picked.shape[0]] = picked
        return out
