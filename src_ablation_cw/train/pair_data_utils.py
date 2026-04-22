# coding=utf-8
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src_ablation_cw.datasets.cell_only_pair_caption_dataset import CellOnlyPairCaptionDataset


def resolve_pair_h5ad_paths(config: Dict[str, Any]) -> List[str]:
    data_cfg = config.get("data", {})
    explicit = [str(x) for x in data_cfg.get("pair_h5ad_paths", []) if x]
    if explicit:
        return explicit
    feature_dir = data_cfg.get("pair_feature_dir") or data_cfg.get("feature_dir")
    if not feature_dir:
        return []
    return [str(p) for p in sorted(Path(feature_dir).glob("*.h5ad"))]


def build_optional_pair_dataset(
    config: Dict[str, Any],
    text_tokenizer: Any,
    special_tokens_ids: Dict[str, int],
    accelerator=None,
    data_type_tag: str = "stage_pair",
    max_samples: int = 0,
    sample_seed: int = 1,
):
    h5ad_paths = resolve_pair_h5ad_paths(config)
    if not h5ad_paths:
        return None

    data_cfg = config.get("data", {})
    lmdb_base_dir = data_cfg.get("pair_lmdb_base_dir") or data_cfg.get("lmdb_base_dir")

    dataset = CellOnlyPairCaptionDataset(
        h5ad_paths=h5ad_paths,
        lmdb_base_dir=lmdb_base_dir,
        text_tokenizer=text_tokenizer,
        config_dict=config,
        special_tokens_ids=special_tokens_ids,
        accelerator=accelerator,
        data_type_tag=data_type_tag,
        max_samples=max_samples,
        sample_seed=sample_seed,
    )
    if len(dataset) == 0:
        return None
    return dataset
