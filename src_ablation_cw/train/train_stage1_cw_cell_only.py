#!/usr/bin/env python3
# coding=utf-8
"""Stage1 (CW protocol) on show-o2 cell-only ablation.

- Data: pretrain_texts dialog + paired metadata captions (mixed)
- Model: standard Qwen2 + cell embedder, gene branch removed
- Trainability: freeze LLM backbone, train cell condition path only
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_ablation_cw.datasets.cw_sft_cell_only_dataset import CWSFTCellOnlyDataset, cw_cell_only_collate
from src_ablation_cw.models.modeling_cell_transformer_for_sft_cw import CellTransformerForSFTCW
from src_ablation_cw.train.common import WandbLogger, ensure_dir, load_config, load_state_pt, save_state_pt


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

def get_stage1_json_paths(config):
    cw_cfg = config.get("training", {}).get("cw_ablation", {})
    if cw_cfg.get("stage1_json_paths"):
        return cw_cfg["stage1_json_paths"]
    sft_paths = config.get("data", {}).get("sft_json_paths", [])
    if not sft_paths:
        raise ValueError("config.data.sft_json_paths is empty")
    return [sft_paths[0]]


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
    epochs = int(cw_cfg.get("stage1_epochs", tr_cfg.get("epochs", 1)))
    log_steps = int(tr_cfg.get("steps_per_print", 5))
    save_interval = int(config.get("logging", {}).get("save_interval", 500))
    lr = float(cw_cfg.get("stage1_lr", opt_cfg.get("stage2_lr_embedder", 1e-4)))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    warmup_ratio = float(config.get("scheduler", {}).get("warmup_ratio", 0.03))
    max_steps = int(cw_cfg.get("stage1_max_steps", 0))

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

    json_paths = get_stage1_json_paths(config)
    dataset = CWSFTCellOnlyDataset(
        feature_dir=None,
        json_paths=json_paths,
        text_tokenizer=tokenizer,
        config_dict=config,
        special_tokens_ids=SPECIAL_TOKENS_IDS,
        accelerator=accelerator,
        curriculum_stage=None,
        data_type_tag="stage1_understanding",
        append_image_tag=bool(cw_cfg.get("append_image_tag", True)),
    )

    if accelerator.is_main_process:
        print("[Stage1-CW] Using conversation-only dataset with gene_h5ad_paths input; pair/lmdb branch disabled.")

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
    model.freeze_llm_backbone()
    for module in [model.cell_embedder, model.direct_token_projector]:
        for p in module.parameters():
            p.requires_grad = True
    if model.use_pathway_cell_qformer and model.train_pathway_cell_qformer:
        for p in model.pathway_qformer.parameters():
            p.requires_grad = True
        model.pathway_embeddings.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)

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

    global_step = 0
    resume_from = cw_cfg.get("stage1_resume_from") or ckpt_cfg.get("resume_from")
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
            print(f"[Stage1-CW] resumed from {info['state_path']} @ step={global_step}")
            print(f"[Stage1-CW] missing={len(info['missing_keys'])}, unexpected={len(info['unexpected_keys'])}")

    save_root = os.path.join(ckpt_cfg.get("save_dir", "./checkpoints"), "cw_ablation_stage1")
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
                    accelerator.clip_grad_norm_(trainable, float(opt_cfg.get("clip_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % log_steps == 0:
                    lr_cur = optimizer.param_groups[0]["lr"]
                    dt = time.time() - t0
                    print(f"[Stage1-CW] step={global_step} ntp_loss={loss_ntp.item():.4f} lr={lr_cur:.2e} time={dt:.1f}s")
                    if logger is not None:
                        logger.log(
                            {
                                "train/ntp_loss": float(loss_ntp.item()),
                                "train/lr": float(lr_cur),
                                "train/epoch": float(epoch),
                            },
                            step=global_step,
                        )
                    t0 = time.time()

                if accelerator.is_main_process and global_step % save_interval == 0:
                    ck = save_state_pt(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        save_dir=save_root,
                        extra={"stage": "stage1_cw_cell_only"},
                    )
                    print(f"[Stage1-CW] saved: {ck}")

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
            extra={"stage": "stage1_cw_cell_only_final"},
        )
        print(f"[Stage1-CW] final checkpoint: {ck}")

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
