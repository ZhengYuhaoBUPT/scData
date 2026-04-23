#!/usr/bin/env python3
# coding=utf-8
"""Stage2 (CW protocol) on show-o2 cell-only ablation.

- Data: finetune dialogs + sampled pretrain dialogs (num_stage1_samples)
- Model: show-o2 style backbone + omni-attn + 1D RoPE, gene branch removed
- Default: full finetune (CW-aligned training protocol), optional LoRA kept as switch
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_ablation_cw.datasets.metadata_formatter import MetadataFormatter
from src_ablation_cw.datasets.cw_sft_cell_only_dataset import CWSFTCellOnlyDataset, cw_cell_only_collate
from src_ablation_cw.models.modeling_cell_transformer_for_sft_cw import CellTransformerForSFTCW
from src_ablation_cw.train.common import WandbLogger, ensure_dir, load_config, load_state_pt, resolve_resume_path, save_state_pt
from src_ablation_cw.train.pair_data_utils import resolve_pair_h5ad_paths


try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


SPECIAL_TOKENS_IDS = {
    "soc_id": 151669,
    "eoc_id": 151670,
}

def move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved



def build_global_linear_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.0):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(current_step: int):
        step = int(current_step)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def _ensure_path_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x]
    return [str(value)]


def _load_json_list(paths):
    merged = []
    for p in paths:
        with open(p, "r") as f:
            merged.extend(json.load(f))
    return merged


def get_stage2_source_paths(config):
    cw_cfg = config.get("training", {}).get("cw_ablation", {})
    sft_paths = _ensure_path_list(config.get("data", {}).get("sft_json_paths", []))

    default_pretrain = sft_paths[:1]
    default_finetune = sft_paths[1:] if len(sft_paths) > 1 else sft_paths[:1]

    pretrain_paths = _ensure_path_list(cw_cfg.get("stage1_json_paths", default_pretrain))
    finetune_paths = _ensure_path_list(cw_cfg.get("stage2_json_paths", default_finetune))

    if not pretrain_paths or not finetune_paths:
        raise ValueError("Cannot resolve pretrain/finetune json paths from config")
    return pretrain_paths, finetune_paths


def _resolve_lmdb_path(h5ad_path: str, lmdb_base_dir: str | None) -> str | None:
    h5_path = Path(h5ad_path)
    candidates = []
    if lmdb_base_dir:
        candidates.append(Path(lmdb_base_dir) / f"{h5_path.stem}.db")
    candidates.append(h5_path.with_suffix(".db"))
    candidates.append(h5_path.parent / f"{h5_path.stem}.db")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _clean_obs_metadata(obs_row: dict) -> dict:
    metadata = {}
    ignore_keys = {"gene_id", "lmdb_key", "_index"}
    for k, v in obs_row.items():
        if k in ignore_keys or v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        text = str(v).strip()
        if not text or text.lower() in {"nan", "none", "unknown"}:
            continue
        metadata[k] = v
    return metadata


def _close_adata(adata) -> None:
    file_obj = getattr(adata, "file", None)
    if file_obj is not None:
        try:
            file_obj.close()
        except Exception:
            pass


def build_caption_converted_json(
    h5ad_paths,
    lmdb_base_dir: str | None,
    num_celltype_samples: int,
    num_meta_samples: int,
    seed: int,
    rank: int,
) -> str:
    """Convert paired h5ad/lmdb metadata to temporary ChatML-style conversation JSON.

    The output ids use obs['cell_id'] when present, otherwise obs index. This keeps the
    generated conversations compatible with ExpressionH5ADRegistry/gene_h5ad_paths.
    """
    import anndata as ad

    file_infos = []
    total_cells = 0
    for h5ad_path in [str(p) for p in h5ad_paths if p]:
        try:
            adata = ad.read_h5ad(h5ad_path, backed="r")
            n_cells = int(adata.n_obs)
            _close_adata(adata)
        except Exception:
            continue
        lmdb_path = _resolve_lmdb_path(h5ad_path, lmdb_base_dir)
        file_infos.append({
            "h5ad_path": h5ad_path,
            "lmdb_path": lmdb_path,
            "global_start": total_cells,
            "global_end": total_cells + n_cells,
        })
        total_cells += n_cells

    num_total = int(num_celltype_samples) + int(num_meta_samples)
    if total_cells == 0 or num_total <= 0:
        out = Path(tempfile.gettempdir()) / f"cw_stage2_caption_empty_rank{rank}.json"
        with open(out, "w") as f:
            json.dump([], f)
        return str(out)

    num_total = min(num_total, total_cells)
    actual_celltype = min(int(num_celltype_samples), num_total)
    actual_meta = num_total - actual_celltype

    rnd = random.Random(int(seed) + rank)
    sampled_global = sorted(rnd.sample(range(total_cells), num_total))

    formatter = MetadataFormatter()
    lmdb_env_cache = {}

    def _get_env(lmdb_path: str):
        if lmdb_path not in lmdb_env_cache:
            lmdb_env_cache[lmdb_path] = lmdb.Environment(
                lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return lmdb_env_cache[lmdb_path]

    results = []
    sample_idx = 0
    for info in file_infos:
        picked = [
            gidx - info["global_start"]
            for gidx in sampled_global
            if info["global_start"] <= gidx < info["global_end"]
        ]
        if not picked:
            continue

        adata = ad.read_h5ad(info["h5ad_path"], backed="r")
        try:
            obs_df = adata.obs.iloc[np.asarray(picked, dtype=np.int64)]
            obs_records = obs_df.to_dict("records")
            obs_indices = [str(x) for x in obs_df.index.tolist()]
        finally:
            _close_adata(adata)

        for local_pos, obs_row in enumerate(obs_records):
            cell_id = str(obs_row.get("cell_id", obs_indices[local_pos])).strip()
            lmdb_key = str(obs_row.get("lmdb_key", cell_id)).strip()
            metadata = _clean_obs_metadata(obs_row)

            lmdb_path = info.get("lmdb_path")
            if lmdb_path and lmdb_key:
                env = _get_env(lmdb_path)
                with env.begin(write=False) as txn:
                    sample_data = txn.get(lmdb_key.encode())
                    if sample_data:
                        try:
                            lmdb_metadata = json.loads(sample_data.decode())
                            metadata = {**metadata, **lmdb_metadata}
                        except Exception:
                            pass

            force_mode = "celltype_qa" if sample_idx < actual_celltype else "meta"
            q, a = formatter.format(metadata, force_mode=force_mode)
            results.append({
                "id": cell_id,
                "conversations": [
                    {"from": "human", "value": q},
                    {"from": "gpt", "value": a},
                ],
            })
            sample_idx += 1

    out = Path(tempfile.gettempdir()) / f"cw_stage2_caption_rank{rank}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    return str(out)


def build_stage2_mixed_json(pretrain_paths, finetune_paths, caption_path, num_stage1_samples: int, seed: int, rank: int) -> str:
    finetune = _load_json_list(finetune_paths)
    pretrain = _load_json_list(pretrain_paths)

    finetune_ids = {str(x.get("id")) for x in finetune}
    pretrain_candidates = [x for x in pretrain if str(x.get("id")) not in finetune_ids]

    n = min(int(num_stage1_samples), len(pretrain_candidates))
    rnd = random.Random(int(seed))
    sampled = rnd.sample(pretrain_candidates, n) if n > 0 else []

    caption = []
    if caption_path and Path(caption_path).exists():
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = json.load(f)

    mixed = list(finetune) + sampled + caption
    out = Path(tempfile.gettempdir()) / f"cw_stage2_mixed_rank{rank}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(mixed, f, ensure_ascii=False)
    return str(out)


def load_model_weights_only(model, ckpt_path: str):
    rp = resolve_resume_path(ckpt_path)
    if rp is None:
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    state = torch.load(rp, map_location="cpu")
    missing, unexpected = model.load_state_dict(state.get("model", {}), strict=False)
    return rp, list(missing), list(unexpected)


def print_trainable_summary(model):
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = 100.0 * trainable / max(1, total)
    print(f"[Stage2-CW] trainable params: {trainable:,} / {total:,} ({ratio:.4f}%)")


def print_trainable_modules(model, top_k: int = 20):
    from collections import defaultdict
    module_counts = defaultdict(int)
    for name, p in model.named_parameters():
        if p.requires_grad:
            # 取第一层点号前的模块名作为聚合键，例如 "showo.base_model.model.model.layers.0..." -> "showo"
            top_key = name.split(".")[0]
            module_counts[top_key] += p.numel()
    sorted_modules = sorted(module_counts.items(), key=lambda x: x[1], reverse=True)
    print("[Stage2-CW] Trainable params by top-level module:")
    for k, v in sorted_modules[:top_k]:
        print(f"  - {k}: {v:,}")
    if len(sorted_modules) > top_k:
        rest = sum(v for _, v in sorted_modules[top_k:])
        print(f"  - (others): {rest:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "src_ablation_cw/config/config_cw_ablation_cell_only.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    tr_cfg = config.get("training", {})
    cw_cfg = tr_cfg.get("cw_ablation", {})
    opt_cfg = config.get("optimizer", {})
    ckpt_cfg = config.get("checkpoint", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    batch_size = int(tr_cfg.get("batch_size", 1))
    num_workers = int(tr_cfg.get("num_workers", 4))
    grad_accum = int(tr_cfg.get("gradient_accumulation_steps", 1))
    epochs = int(cw_cfg.get("stage2_epochs", tr_cfg.get("epochs", 1)))
    log_steps = int(tr_cfg.get("steps_per_print", 10))
    save_interval = int(config.get("logging", {}).get("save_interval", 500))

    lora_lr = float(cw_cfg.get("stage2_lora_lr", opt_cfg.get("stage2_lr_llm", 2e-5)))
    cell_lr = float(cw_cfg.get("stage2_cell_lr", opt_cfg.get("stage2_lr_embedder", 1e-4)))
    llm_lr = float(cw_cfg.get("stage2_llm_lr", opt_cfg.get("stage2_lr_llm", 2e-5)))
    full_finetune = bool(cw_cfg.get("stage2_full_finetune", True))
    align_cw_single_lr = bool(cw_cfg.get("stage2_align_cw_single_lr", True))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    warmup_ratio = float(config.get("scheduler", {}).get("warmup_ratio", 0.03))
    max_steps = int(cw_cfg.get("stage2_max_steps", 0))

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=("bf16" if torch.cuda.is_available() else "no"),
    )

    logger = None
    if bool(config.get("logging", {}).get("enable_wandb", config.get("logging", {}).get("enable_swanlab", False))):
        logger = WandbLogger(config, accelerator)

    seed = int(tr_cfg.get("seed", 1))
    torch.manual_seed(seed + accelerator.process_index)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["llm_model_path"],
        trust_remote_code=True,
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pretrain_paths, finetune_paths = get_stage2_source_paths(config)
    num_stage1_samples = int(cw_cfg.get("num_stage1_samples", 10000))
    mix_seed = int(cw_cfg.get("stage2_mix_seed", seed))

    caption_json_path = None
    use_pair_caption = bool(cw_cfg.get("stage2_use_pair_caption", True))
    if use_pair_caption:
        pair_h5ad_paths = resolve_pair_h5ad_paths(config)
        pair_lmdb_dir = data_cfg.get("pair_lmdb_base_dir") or data_cfg.get("lmdb_base_dir")
        num_caption_samples = int(cw_cfg.get("num_caption_samples", 20000))
        caption_celltype_ratio = float(cw_cfg.get("caption_celltype_ratio", 0.65))
        if pair_h5ad_paths:
            num_celltype = int(num_caption_samples * caption_celltype_ratio)
            num_meta = num_caption_samples - num_celltype
            caption_json_path = build_caption_converted_json(
                h5ad_paths=pair_h5ad_paths,
                lmdb_base_dir=pair_lmdb_dir,
                num_celltype_samples=num_celltype,
                num_meta_samples=num_meta,
                seed=int(cw_cfg.get("stage2_caption_seed", mix_seed)),
                rank=accelerator.process_index,
            )
            if accelerator.is_main_process:
                print(
                    f"[Stage2-CW] generated caption JSON: {caption_json_path} "
                    f"({num_celltype} celltype + {num_meta} meta)"
                )
        elif accelerator.is_main_process:
            print("[Stage2-CW] Pair h5ad paths not configured; using pretrain+finetune only.")
    elif accelerator.is_main_process:
        print("[Stage2-CW] stage2_use_pair_caption=false; using pretrain+finetune only.")

    mixed_json_path = build_stage2_mixed_json(
        pretrain_paths=pretrain_paths,
        finetune_paths=finetune_paths,
        caption_path=caption_json_path,
        num_stage1_samples=num_stage1_samples,
        seed=mix_seed,
        rank=accelerator.process_index,
    )

    if accelerator.is_main_process:
        print(f"[Stage2-CW] pretrain_paths({len(pretrain_paths)})={pretrain_paths}")
        print(f"[Stage2-CW] finetune_paths({len(finetune_paths)})={finetune_paths}")
        print(f"[Stage2-CW] mixed json={mixed_json_path}, num_stage1_samples={num_stage1_samples}")

    dialog_dataset = CWSFTCellOnlyDataset(
        feature_dir=None,
        json_paths=[mixed_json_path],
        text_tokenizer=tokenizer,
        config_dict=config,
        special_tokens_ids=SPECIAL_TOKENS_IDS,
        accelerator=accelerator,
        curriculum_stage=None,
        data_type_tag="stage2",
        append_image_tag=bool(cw_cfg.get("append_image_tag", True)),
    )

    dataset = dialog_dataset

    if accelerator.is_main_process:
        print(f"[Stage2-CW] dataset_mix mixed_dialog={len(dialog_dataset)} total={len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=cw_cell_only_collate,
        pin_memory=bool(tr_cfg.get("pin_memory", False)),
        drop_last=True,
    )

    model = CellTransformerForSFTCW(config, special_tokens_ids=SPECIAL_TOKENS_IDS)

    init_from = cw_cfg.get("stage2_init_from_stage1")
    if init_from:
        path, miss, unexp = load_model_weights_only(model, init_from)
        if accelerator.is_main_process:
            print(f"[Stage2-CW] loaded init model from: {path}")
            print(f"[Stage2-CW] init missing={len(miss)}, unexpected={len(unexp)}")
    else:
        if accelerator.is_main_process:
            print("[Stage2-CW] WARNING: stage2_init_from_stage1 is not set. "
                  "Cell embedder will be randomly initialized, which usually leads to worse results. "
                  "Consider loading a stage1 checkpoint for best practice.")

    use_lora = bool(cw_cfg.get("stage2_use_lora", False))

    # Guardrail: when LoRA is enabled, full-LLM LR interfaces must be disabled to avoid silent config conflicts.
    if use_lora:
        conflict_keys = []
        if bool(cw_cfg.get("stage2_full_finetune", False)):
            conflict_keys.append("stage2_full_finetune")
        if "stage2_llm_lr" in cw_cfg:
            conflict_keys.append("stage2_llm_lr")
        if bool(cw_cfg.get("stage2_align_cw_single_lr", False)):
            conflict_keys.append("stage2_align_cw_single_lr")
        if conflict_keys:
            raise ValueError(
                "LR config conflict when stage2_use_lora=true. Disable these keys: " + ", ".join(conflict_keys)
            )

    if use_lora:
        model.freeze_llm_backbone()
        for p in model.cell_embedder.parameters():
            p.requires_grad = True
        if model.use_pathway_cell_qformer and model.train_pathway_cell_qformer:
            for p in model.pathway_qformer.parameters():
                p.requires_grad = True
            model.pathway_embeddings.requires_grad = True

        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required for stage2 LoRA but not installed")

        lora_cfg = LoraConfig(
            r=int(cw_cfg.get("lora_r", 8)),
            lora_alpha=int(cw_cfg.get("lora_alpha", 16)),
            lora_dropout=float(cw_cfg.get("lora_dropout", 0.05)),
            bias="none",
            target_modules=cw_cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            task_type="CAUSAL_LM",
        )
        model.showo = get_peft_model(model.showo, lora_cfg)

        lora_params, cell_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                "cell_embedder" in n
                or "direct_token_projector" in n
                or "pathway_qformer" in n
                or "pathway_embeddings" in n
            ):
                cell_params.append(p)
            else:
                lora_params.append(p)

        param_groups = []
        if cell_params:
            param_groups.append({"params": cell_params, "lr": cell_lr, "weight_decay": weight_decay})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay})

    else:
        if full_finetune:
            model.unfreeze_llm_backbone()
        else:
            model.freeze_llm_backbone()

        for module in [model.cell_embedder, model.direct_token_projector]:
            for p in module.parameters():
                p.requires_grad = True
        if model.use_pathway_cell_qformer and model.train_pathway_cell_qformer:
            for p in model.pathway_qformer.parameters():
                p.requires_grad = True
            model.pathway_embeddings.requires_grad = True

        cell_params, llm_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                "cell_embedder" in n
                or "direct_token_projector" in n
                or "pathway_qformer" in n
                or "pathway_embeddings" in n
            ):
                cell_params.append(p)
            else:
                llm_params.append(p)

        param_groups = []
        cell_lr_explicit = ("stage2_cell_lr" in cw_cfg)
        if full_finetune and align_cw_single_lr and (not cell_lr_explicit):
            eff_cell_lr = llm_lr
        else:
            eff_cell_lr = cell_lr
        if accelerator.is_main_process and full_finetune and align_cw_single_lr and cell_lr_explicit:
            print("[LR] stage2_align_cw_single_lr=true but stage2_cell_lr is explicitly set; keep distinct cell_lr.")
        if cell_params:
            param_groups.append({"params": cell_params, "lr": eff_cell_lr, "weight_decay": weight_decay})
        if llm_params:
            param_groups.append({"params": llm_params, "lr": llm_lr, "weight_decay": weight_decay})

    optimizer = AdamW(param_groups)

    if accelerator.is_main_process:
        print(
            f"[Optimizer] use_lora={use_lora} cell_lr={cell_lr:.2e} llm_lr={llm_lr:.2e} lora_lr={lora_lr:.2e} "
            f"full_finetune={full_finetune} align_cw_single_lr={align_cw_single_lr}"
        )
        for i, pg in enumerate(param_groups):
            n_params = sum(p.numel() for p in pg.get("params", []))
            print(f"[Optimizer] group_{i}: lr={pg.get('lr', 0.0):.2e}, wd={pg.get('weight_decay', 0.0):.2e}, params={n_params:,}")

    updates_per_epoch = max(1, math.ceil(len(dataloader) / grad_accum))
    computed_total_steps = max(1, updates_per_epoch * epochs)
    scheduler_cfg = config.get("scheduler", {})
    scheduler_total_steps = int(scheduler_cfg.get("total_steps", 0))
    schedule_by_max_steps = bool(scheduler_cfg.get("schedule_by_max_steps", True))
    if scheduler_total_steps > 0:
        total_steps = scheduler_total_steps
        total_steps_source = "scheduler.total_steps"
    elif schedule_by_max_steps and int(max_steps) > 0:
        total_steps = int(max_steps)
        total_steps_source = "stage*_max_steps"
    else:
        total_steps = int(computed_total_steps)
        total_steps_source = "computed_total_steps"
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))

    scheduler = build_global_linear_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )
    if accelerator.is_main_process:
        print(f"[Scheduler] type=linear_warmup_cosine_decay total_steps={total_steps} source={total_steps_source} warmup_steps={warmup_steps} min_lr_ratio={min_lr_ratio} computed_total_steps={computed_total_steps} max_steps={max_steps}")

    # Dataset is already rank-sharded inside CWSFTCellOnlyDataset.
    # Keep raw dataloader for training to avoid second sharding by accelerate.
    raw_dataloader = dataloader
    model, optimizer, _dataloader_for_ds_init, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    dataloader = raw_dataloader
    if accelerator.is_main_process:
        print(f"[Dataloader] use pre-sharded raw loader: len={len(dataloader)} (avoid double-sharding)")

    if accelerator.is_main_process:
        print(
            f"[Stage2-CW] mode: use_lora={use_lora}, full_finetune={full_finetune}, "
            f"align_cw_single_lr={align_cw_single_lr}"
        )
        print_trainable_summary(accelerator.unwrap_model(model))
        print_trainable_modules(accelerator.unwrap_model(model))

    global_step = 0
    resume_from = cw_cfg.get("stage2_resume_from") or ckpt_cfg.get("resume_from")
    if resume_from:
        gstep, info = load_state_pt(
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            resume_from=resume_from,
            strict=False,
        )
        global_step = gstep
        if accelerator.is_main_process:
            print(f"[Stage2-CW] resumed from {info['state_path']} @ step={global_step}")
            print(f"[Stage2-CW] missing={len(info['missing_keys'])}, unexpected={len(info['unexpected_keys'])}")

    save_root = os.path.join(ckpt_cfg.get("save_dir", "./checkpoints"), "cw_ablation_stage2")
    ensure_dir(save_root)

    model.train()
    t0 = time.time()

    for epoch in range(epochs):
        for batch in dataloader:
            batch = move_batch_to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                _, loss_ntp, _, _ = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    cell_features=batch["cell_features"],
                    cell_positions=batch["cell_positions"],
                    modality_positions=batch.get("modality_positions"),
                    labels=batch["labels"],
                    data_type=batch.get("data_type"),
                )
                accelerator.backward(loss_ntp)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        float(opt_cfg.get("clip_grad_norm", 1.0)),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % log_steps == 0:
                    lrs = ", ".join([f"{pg['lr']:.2e}" for pg in optimizer.param_groups])
                    dt = time.time() - t0
                    print(f"[Stage2-CW] step={global_step} ntp_loss={loss_ntp.item():.4f} lr=[{lrs}] time={dt:.1f}s")
                    if logger is not None:
                        metrics = {"train/ntp_loss": float(loss_ntp.item()), "train/epoch": float(epoch)}
                        for i, pg in enumerate(optimizer.param_groups):
                            metrics[f"train/lr_group_{i}"] = float(pg["lr"])
                        logger.log(metrics, step=global_step)
                    t0 = time.time()

                if accelerator.is_main_process and global_step % save_interval == 0:
                    ck = save_state_pt(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        save_dir=save_root,
                        extra={"stage": "stage2_cw_cell_only", "use_lora": use_lora},
                    )
                    print(f"[Stage2-CW] saved: {ck}")

                if max_steps > 0 and global_step >= max_steps:
                    break

        if max_steps > 0 and global_step >= max_steps:
            break

    if accelerator.is_main_process:
        ck = save_state_pt(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            save_dir=save_root,
            extra={"stage": "stage2_cw_cell_only_final", "use_lora": use_lora},
        )
        print(f"[Stage2-CW] final checkpoint: {ck}")

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
