# coding=utf-8
"""
Cell-Aware Gene Feature Loader.

复用 ClusterFeatureLoader 读取 LMDB 记录，并在 per-gene codebook embedding 上
融合该细胞的 cls_token（cell-level embedding），权重固定为 0.5。

融合公式：
    fused_emb = 0.5 * gene_codebook_emb + 0.5 * cell_emb
    （cell_emb 广播到每个基因，不做 normalize，保持原始量级）

用法（dataset 侧）：
    from src.datasets.cell_aware_feature_loader import CellAwareFeatureLoader
    loader = CellAwareFeatureLoader(
        cluster_lmdb_dir=..., codebook_dir=..., clusters_per_gene=64
    )
    fused, token_ids, valid_mask = loader.get_fused(db_name, key)
    # fused: np.ndarray float32 [L, 768]
    # token_ids: np.ndarray int64 [L], unknown=-1
    # valid_mask: np.ndarray bool [L]
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .cluster_feature_loader import ClusterFeatureLoader, ClusterFeatureRecord, CodebookLookupResult


class CellAwareFeatureLoader:
    """在 ClusterFeatureLoader 基础上融合 cell cls_token 的加载器。"""

    def __init__(
        self,
        cluster_lmdb_dir: Union[str, Path],
        codebook_dir: Union[str, Path],
        clusters_per_gene: int = 64,
        prefer_compact_codebook: bool = True,
        readahead: bool = False,
        max_readers: int = 2048,
        gene_weight: float = 0.5,
        cell_weight: float = 0.5,
        cell_dropout: float = 0.3,
    ):
        """
        Args:
            gene_weight: per-gene codebook embedding 权重（默认 0.5）
            cell_weight: cell cls_token embedding 权重（默认 0.5）
            cell_dropout: 以该概率随机丢弃 cell_emb（只用 gene_emb），默认 0.3
        """
        self.gene_weight = float(gene_weight)
        self.cell_weight = float(cell_weight)
        self.cell_dropout = float(cell_dropout)
        assert abs(self.gene_weight + self.cell_weight - 1.0) < 1e-6, \
            "gene_weight + cell_weight 必须等于 1.0"

        self._base = ClusterFeatureLoader(
            cluster_lmdb_dir=cluster_lmdb_dir,
            readahead=readahead,
            max_readers=max_readers,
            codebook_dir=codebook_dir,
            clusters_per_gene=clusters_per_gene,
            prefer_compact_codebook=prefer_compact_codebook,
        )

    def close(self):
        self._base.close()

    def _fuse(self, record: ClusterFeatureRecord, lookup: CodebookLookupResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        核心融合逻辑。

        Returns:
            fused_emb:   float32 [L, 768]
            token_ids:   int64   [L], unknown=-1
            valid_mask:  bool    [L]
        """
        L = int(lookup.features_768.shape[0])
        gene_emb = lookup.features_768.astype(np.float32)      # [L, 768]
        cell_emb = record.cls_token.astype(np.float32)          # [768]

        # 防止历史坏样本把 NaN/Inf 带入训练
        gene_emb = np.nan_to_num(gene_emb, nan=0.0, posinf=0.0, neginf=0.0)
        cell_emb = np.nan_to_num(cell_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # cell_dropout：以该概率丢弃 cell_emb，只用 gene_emb（训练时防止作弊）
        if self.cell_dropout > 0 and np.random.rand() < self.cell_dropout:
            cell_emb = np.zeros_like(cell_emb)

        # 广播融合
        fused = self.gene_weight * gene_emb + self.cell_weight * cell_emb
        fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        # 保持和原始 codebook 一致的 dtype/范围，不额外 normalize
        # 如果需要 normalize，可在此加：
        # fused = fused / (np.linalg.norm(fused, axis=-1, keepdims=True) + 1e-8)

        return fused, lookup.token_ids, lookup.valid_mask

    def get(
        self,
        db_name: Union[str, Path],
        key: Union[bytes, str],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        读取单个样本并返回融合后的基因 embedding。

        Returns:
            (fused_emb, token_ids, valid_mask) 或 None（key 不存在）
        """
        rec_lookup = self._base.get_with_codebook(db_name, key)
        if rec_lookup is None:
            return None
        record, lookup = rec_lookup
        return self._fuse(record, lookup)

    def get_raw(
        self,
        db_name: Union[str, Path],
        key: Union[bytes, str],
    ) -> Optional[Tuple[ClusterFeatureRecord, CodebookLookupResult]]:
        """如果需要原始 record + lookup，可从此获取。"""
        return self._base.get_with_codebook(db_name, key)

    def get_many(
        self,
        requests: List[Tuple[Union[str, Path], Union[bytes, str]]],
        num_workers: int = 0,
    ) -> List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """批量读取。"""
        if num_workers <= 0:
            return [self.get(db, key) for db, key in requests]

        # 批量读取原始数据
        raw_results = self._base.get_many(requests, num_workers=num_workers)
        out = []
        for raw in raw_results:
            if raw is None:
                out.append(None)
            else:
                out.append(self._fuse(raw, self._base.codebook.lookup(raw.scgpt_ids, raw.cluster_indices)))
        return out
