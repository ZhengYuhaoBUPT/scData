#!/usr/bin/env python3
# coding=utf-8
"""
🧬 Stage 1 理解训练脚本 (Understanding-only Training)
仅训练理解:
- 100% 理解模式: [Gene] -> [Text] (Cell Understanding)

"""

import os
import sys
import json
import argparse
import time
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoTokenizer, get_scheduler

# 注入项目路径：默认使用当前仓库根目录；如需原始大项目，可设置 SC_SHOWO_ROOT。
repo_root = Path(__file__).resolve().parents[1]
project_root = Path(os.environ.get("SC_SHOWO_ROOT", repo_root)).expanduser().resolve()
for _path in (project_root, repo_root):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from src.datasets.bidirectional_stage1_dataset_rankloss import BidirectionalStage1Dataset
    from src.models.modeling_gene_transformer_rank_pe import GeneTransformer
    from src.train.utils.utils import SwanLabLogger, save_checkpoint, load_checkpoint, TrainingState
    from src.train.utils.scheduler_utils import build_scheduler
except ModuleNotFoundError:
    from datasets.bidirectional_stage1_dataset_rankloss import BidirectionalStage1Dataset
    from models.modeling_gene_transformer_rank_pe import GeneTransformer
    from train.utils.utils import SwanLabLogger, save_checkpoint, load_checkpoint, TrainingState
    from train.utils.scheduler_utils import build_scheduler
from torch.utils.data import DataLoader, DistributedSampler


def resolve_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json. Defaults to $SC_SHOWO_CONFIG or <repo>/config/config.json.",
    )
    args, _ = parser.parse_known_args()

    candidates = []
    if args.config:
        candidates.append(Path(args.config))
    env_config = os.environ.get("SC_SHOWO_CONFIG")
    if env_config:
        candidates.append(Path(env_config))
    candidates.append(project_root / "config" / "config.json")
    candidates.append(repo_root / "config" / "config.json")

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {p.expanduser()}" for p in candidates)
    raise FileNotFoundError(
        "Cannot find config.json. Pass --config /path/to/config.json or set SC_SHOWO_CONFIG.\n"
        f"Searched:\n{searched}"
    )


def _select_probe_indices(data_types, target: str, max_samples: int):
    idx = [i for i, dt in enumerate(data_types) if target in str(dt)]
    if max_samples > 0:
        idx = idx[:max_samples]
    return idx


def _slice_batch_by_indices(batch: dict, indices: torch.Tensor):
    out = {}
    bsz = int(batch['input_ids'].shape[0])
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == bsz:
            out[k] = v.index_select(0, indices)
        elif isinstance(v, list) and len(v) == bsz:
            idx_list = indices.tolist()
            out[k] = [v[i] for i in idx_list]
    return out


def _deranged_perm(n: int, device: torch.device):
    if n < 2:
        return None
    for _ in range(16):
        perm = torch.randperm(n, device=device)
        if not torch.any(perm == torch.arange(n, device=device)):
            return perm
    return None

def _pad_2d_for_global_gather(x: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    """Pad [B, L] to global max L across ranks before all-gather."""
    if not dist.is_initialized():
        return x
    l = torch.tensor([x.shape[1]], device=x.device, dtype=torch.long)
    dist.all_reduce(l, op=dist.ReduceOp.MAX)
    max_len = int(l.item())
    if x.shape[1] >= max_len:
        return x
    pad = torch.full((x.shape[0], max_len - x.shape[1]), pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def _pad_3d_for_global_gather(x: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """Pad [B, L, D] to global max L across ranks before all-gather."""
    if not dist.is_initialized():
        return x
    l = torch.tensor([x.shape[1]], device=x.device, dtype=torch.long)
    dist.all_reduce(l, op=dist.ReduceOp.MAX)
    max_len = int(l.item())
    if x.shape[1] >= max_len:
        return x
    pad = torch.full((x.shape[0], max_len - x.shape[1], x.shape[2]), pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)



def _answer_only_ntp_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    # Probe-only NTP:
    # - Start from supervised positions (labels != -100)
    # - Remove trailing formatting tokens in assistant tail (e.g., <|im_end|>, newline)
    if logits is None or labels is None:
        return None

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_logits = torch.nan_to_num(shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    shift_labels = labels[..., 1:].contiguous()

    valid = (shift_labels != -100)
    if not valid.any():
        return None

    # 1516xx is Qwen special-token range (<|im_*|> etc.); 198/13 are newline-like tokens.
    bsz = shift_labels.shape[0]
    for i in range(bsz):
        pos = torch.where(valid[i])[0]
        while pos.numel() > 1:
            t = int(shift_labels[i, pos[-1]].item())
            if t >= 151640 or t in (198, 13):
                valid[i, pos[-1]] = False
                pos = pos[:-1]
            else:
                break

    if not valid.any():
        return None

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1))[valid.view(-1)],
        shift_labels.view(-1)[valid.view(-1)],
        reduction='mean',
    )
    return loss


def _compute_matched_mismatched_gap_stage1(model, batch: dict, target: str, max_samples: int):
    data_types = batch.get('data_type', [])
    idx = _select_probe_indices(data_types, target=target, max_samples=max_samples)
    if len(idx) < 2:
        return None

    device = batch['input_ids'].device
    idx_t = torch.tensor(idx, device=device, dtype=torch.long)
    sub = _slice_batch_by_indices(batch, idx_t)
    if 'modality_positions' not in sub:
        return None

    n = sub['input_ids'].shape[0]
    perm = _deranged_perm(n, device=device)
    if perm is None:
        return None

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model_kwargs = dict(
                input_ids=sub['input_ids'],
                attention_mask=sub.get('attention_mask'),
                position_ids=sub.get('position_ids'),
                modality_positions=sub['modality_positions'],
                labels=sub['labels'],
                gene_labels=sub.get('gene_labels'),
                t=sub.get('t'),
                data_type=sub.get('data_type'),
            )
            if 'gene_embeddings' in sub:
                model_kwargs['gene_embeddings'] = sub['gene_embeddings']

            out = model(**model_kwargs)
            matched_logits = out[0] if isinstance(out, (tuple, list)) else None
            matched_ntp = _answer_only_ntp_from_logits(matched_logits, sub.get('labels'))
            if matched_ntp is None:
                return None

            mismatched_ids = sub['input_ids'].clone()
            mismatched_gene_emb = sub['gene_embeddings'].clone() if 'gene_embeddings' in sub else None
            for i in range(n):
                si = int(sub['modality_positions'][i, 0, 0].item())
                li = int(sub['modality_positions'][i, 0, 1].item())
                j = int(perm[i].item())
                sj = int(sub['modality_positions'][j, 0, 0].item())
                lj = int(sub['modality_positions'][j, 0, 1].item())
                if li <= 0 or lj <= 0 or li != lj:
                    return None
                mismatched_ids[i, si:si + li] = sub['input_ids'][j, sj:sj + lj]
                if mismatched_gene_emb is not None:
                    mismatched_gene_emb[i, :li] = sub['gene_embeddings'][j, :lj]

            model_kwargs['input_ids'] = mismatched_ids
            if mismatched_gene_emb is not None:
                model_kwargs['gene_embeddings'] = mismatched_gene_emb
            out_mis = model(**model_kwargs)
            mismatched_logits = out_mis[0] if isinstance(out_mis, (tuple, list)) else None
            mismatched_ntp = _answer_only_ntp_from_logits(mismatched_logits, sub.get('labels'))
            if mismatched_ntp is None:
                return None
    finally:
        if was_training:
            model.train()

    return (mismatched_ntp - matched_ntp).detach()


def _compute_soft_rank_loss_stage1(
    model,
    batch: dict,
    logits: torch.Tensor,
    accelerator: Accelerator,
    rank_k: int,
    tau: float,
    anchors_per_rank: int,
):
    device = batch['input_ids'].device
    local_bsz = int(batch['input_ids'].shape[0])
    if local_bsz <= 0 or rank_k <= 0 or anchors_per_rank <= 0:
        return None, {"anchors": 0, "negs": 0}

    data_types = batch.get('data_type', [])
    local_under = [i for i, dt in enumerate(data_types) if 'understanding' in str(dt)]

    # Collective pool build (all ranks must call)
    local_celltype = batch.get('celltype_id')
    if local_celltype is None:
        local_celltype = torch.zeros(local_bsz, dtype=torch.long, device=device)
    else:
        local_celltype = local_celltype.to(device=device, dtype=torch.long)

    local_mod = batch['modality_positions'][:, 0, :].to(device=device, dtype=torch.long)
    local_under_t = torch.tensor([1 if i in local_under else 0 for i in range(local_bsz)], device=device, dtype=torch.long)

    global_input_ids = accelerator.gather(_pad_2d_for_global_gather(batch['input_ids'].detach(), pad_value=151643))
    global_mod = accelerator.gather(local_mod.detach())
    global_celltype = accelerator.gather(local_celltype.detach())
    global_under = accelerator.gather(local_under_t.detach())

    has_gene_emb = 'gene_embeddings' in batch
    global_gene_emb = None
    if has_gene_emb:
        global_gene_emb = accelerator.gather(_pad_3d_for_global_gather(batch['gene_embeddings'].detach(), pad_value=0.0))

    if len(local_under) == 0:
        return None, {"anchors": 0, "negs": 0}

    # Per-rank anchors: random choose up to anchors_per_rank
    perm = torch.randperm(len(local_under), device=device)
    chosen_local = [local_under[int(i)] for i in perm[:min(anchors_per_rank, len(local_under))].tolist()]

    tau = float(max(tau, 1e-3))
    world_bsz = int(global_input_ids.shape[0])
    global_base = int(accelerator.process_index * local_bsz)

    was_training = model.training
    rank_losses = []
    used_negs = 0

    try:
        for anchor_idx in chosen_local:
            # Positive score from current graph (with grad)
            pos_logits = logits[anchor_idx:anchor_idx + 1]
            pos_labels = batch['labels'][anchor_idx:anchor_idx + 1]
            ce_pos = _answer_only_ntp_from_logits(pos_logits, pos_labels)
            if ce_pos is None or (not torch.isfinite(ce_pos)):
                continue
            s_pos = -ce_pos

            a_start = int(batch['modality_positions'][anchor_idx, 0, 0].item())
            a_len = int(batch['modality_positions'][anchor_idx, 0, 1].item())
            if a_len <= 0:
                continue

            anchor_global_idx = global_base + int(anchor_idx)
            if anchor_global_idx < 0 or anchor_global_idx >= world_bsz:
                continue

            candidate_mask = (global_mod[:, 1] == a_len)
            candidate_idx = torch.where(candidate_mask)[0]
            neg_idx = BidirectionalStage1Dataset.sample_negative_indices_excluding_celltype(
                global_celltype_ids=global_celltype,
                anchor_global_idx=anchor_global_idx,
                k=rank_k,
                candidate_indices=candidate_idx,
            )
            if neg_idx.numel() == 0:
                continue

            neg_scores = []
            model.eval()
            for ni in neg_idx.tolist():
                n_start = int(global_mod[ni, 0].item())
                n_len = int(global_mod[ni, 1].item())
                if n_len != a_len or n_len <= 0:
                    continue

                mis_ids = batch['input_ids'][anchor_idx:anchor_idx + 1].clone()
                mis_ids[0, a_start:a_start + a_len] = global_input_ids[ni, n_start:n_start + n_len]

                mk = dict(
                    input_ids=mis_ids,
                    attention_mask=batch.get('attention_mask')[anchor_idx:anchor_idx + 1] if batch.get('attention_mask') is not None else None,
                    position_ids=batch['position_ids'][anchor_idx:anchor_idx + 1],
                    modality_positions=batch['modality_positions'][anchor_idx:anchor_idx + 1],
                    labels=batch['labels'][anchor_idx:anchor_idx + 1],
                    gene_labels=batch.get('gene_labels')[anchor_idx:anchor_idx + 1] if batch.get('gene_labels') is not None else None,
                    t=batch.get('t')[anchor_idx:anchor_idx + 1] if batch.get('t') is not None else None,
                    data_type=['stage1_understanding'],
                )

                if has_gene_emb and global_gene_emb is not None:
                    mis_ge = batch['gene_embeddings'][anchor_idx:anchor_idx + 1].clone()
                    mis_ge[0, :a_len] = global_gene_emb[ni, :a_len]
                    mk['gene_embeddings'] = mis_ge

                with torch.no_grad():
                    out_mis = model(**mk)
                    mis_logits = out_mis[0] if isinstance(out_mis, (tuple, list)) else None
                    ce_neg = _answer_only_ntp_from_logits(mis_logits, mk['labels'])
                if ce_neg is None or (not torch.isfinite(ce_neg)):
                    continue
                neg_scores.append((-ce_neg).detach())

            if len(neg_scores) == 0:
                continue

            neg_t = torch.stack(neg_scores).to(device=device, dtype=s_pos.dtype)
            delta = torch.clamp((neg_t - s_pos) / tau, min=-20.0, max=20.0)
            r_soft = 1.0 + torch.sigmoid(delta).sum()
            rank_i = torch.log(torch.clamp(r_soft, min=1.0 + 1e-8))
            if torch.isfinite(rank_i):
                rank_losses.append(rank_i)
                used_negs += int(len(neg_scores))
    finally:
        if was_training:
            model.train()

    if len(rank_losses) == 0:
        return None, {"anchors": 0, "negs": used_negs}

    rank_loss = torch.stack(rank_losses).mean()
    return rank_loss, {"anchors": len(rank_losses), "negs": used_negs}




def get_parameter_groups(model, config):
    """一阶段：统一学习率，冻结 LLM，只训练 rank_signal_injector 和 diffusion head"""
    lr = config['optimizer'].get('lr', 1e-4)
    weight_decay = config['optimizer'].get('weight_decay', 1e-4)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    return [{"params": trainable_params, "lr": lr, "weight_decay": weight_decay}]


def main():
    # ==================== 1. 初始化加速器 ====================
    accelerator = Accelerator()

    # ==================== 2. 加载配置 ====================
    config_path = resolve_config_path()
    accelerator.print(f"📄 使用配置: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    training_config = config['training']

    # ==================== 3. 加载 Tokenizer ====================
    model_path = config['model']['llm_model_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ==================== 4. 定义特殊 Tokens ====================
    special_tokens_ids = {
        'sog_id': 151665, 'eog_id': 151666,
        'mask_gene_id': 151667
    }

    # ==================== 5. 初始化模型 (Stage 1: 1D RoPE + 弱 rank signal) ====================
    model = GeneTransformer(
        llm_vocab_size=tokenizer.vocab_size,
        llm_model_path=model_path,
        load_from_showo=False,
        config_dict=config,
        special_tokens_ids=special_tokens_ids
    )
    accelerator.print('\n⚠️  Stage1 使用弱 rank signal + 1D RoPE')

    model.requires_grad_(True)

    # ==================== 6. 初始化双向数据加载器 ====================
    accelerator.print("\n🧬 初始化理解数据加载器...")
    accelerator.print("   模式分布: 100% 理解 [Gene->Text]")

    dataset = BidirectionalStage1Dataset(
        config_dict=config,
        special_tokens_ids=special_tokens_ids,
        text_tokenizer=tokenizer,
        # IMPORTANT:
        # Let Accelerate handle distributed sharding.
        # Passing accelerator into dataset would trigger manual rank split
        # and cause double-sharding after accelerator.prepare(dataloader).
        accelerator=None,
        max_seq_len=config['dataset'].get('max_seq_len', 1800),
        understanding_ratio=1.0,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.get('batch_size', 16),
        sampler=sampler,
        num_workers=training_config.get('num_workers', 4),
        collate_fn=BidirectionalStage1Dataset.collate_fn,
        drop_last=True,
    )

    # ==================== 7. 优化器 ====================
    # 🔧 一阶段冻结 LLM 参数，仅训练 gene 接口与扩散头；若禁用 gene PE，则同步关闭 rank_signal 学习
    disable_gene_pe = bool(config.get('model', {}).get('disable_gene_position_ids', False))
    for name, param in model.named_parameters():
        if "gene_proj" in name or "diffusion_head" in name:
            param.requires_grad = True
        elif "rank_signal_injector" in name:
            param.requires_grad = (not disable_gene_pe)
        else:
            param.requires_grad = False

    if accelerator.is_main_process and disable_gene_pe:
        print("⚠️ disable_gene_position_ids=True: 已关闭 rank_signal_injector 训练与注入")

    param_groups = get_parameter_groups(model, config)

    if accelerator.is_main_process:
        print(f"\n📊 参数组统计:")
        # 计算总参数量和可训练参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"   总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"   可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"   冻结参数量: {frozen_params:,} ({frozen_params/1e9:.2f}B)")
        print(f"   训练比例: {trainable_params/total_params*100:.2f}%")

    optimizer = AdamW(param_groups)

    # ==================== 8. 学习率调度器 ====================
    data_steps = len(dataloader) * training_config['epochs']
    grad_accum = accelerator.gradient_accumulation_steps
    total_steps = (data_steps + grad_accum - 1) // grad_accum
    num_warmup_steps = int(total_steps * config['scheduler']['warmup_ratio'])

    lr_scheduler, scheduler_info = build_scheduler(
        optimizer=optimizer,
        scheduler_config=config['scheduler'],
        computed_total_steps=total_steps,
    )

    # ==================== 9. 指标监控 ====================
    logger = SwanLabLogger(config, accelerator)
    training_state = TrainingState()

    # 准备环境
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, dataloader
    )

    if accelerator.is_main_process:
        print(f"\n📊 训练配置:")
        print(f"   - 数据步数: {data_steps:,}")
        print(f"   - 优化步数: {total_steps:,}")
        print(f"   - 梯度累积步数: {grad_accum}")
        print(f"   - Epochs: {training_config['epochs']}")
        print(f"   - Scheduler: {scheduler_info['type']}")
        print(f"   - Warmup Steps: {scheduler_info['num_warmup_steps']:,}")
        print(f"   - Training Steps: {scheduler_info['num_training_steps']:,}")

    # ==================== 10. Resume 从 Checkpoint 恢复 ====================
    global_step = 0
    start_epoch = 0

    # 优先从配置读取 resume 路径 (支持直接指定到 checkpoint-step-xxx 或 bidirectional_step_xxx)
    resume_path = config.get('checkpoint', {}).get('resume_from')

    # 如果没有配置，自动查找最新的 checkpoint
    if not resume_path:
        save_dir = Path(config['checkpoint']['save_dir'])
        if save_dir.exists():
            # 查找 bidirectional_step_* 目录
            step_dirs = sorted(
                [d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith("bidirectional_step_")],
                key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else 0
            )
            if step_dirs:
                latest_step_dir = step_dirs[-1]
                # 在其中查找 checkpoint-step-* 子目录
                ckpt_dirs = list(latest_step_dir.glob("checkpoint-step-*"))
                if ckpt_dirs:
                    resume_path = str(ckpt_dirs[0])
                else:
                    resume_path = str(latest_step_dir)
                if accelerator.is_main_process:
                    print(f"\n📥 自动发现最新 checkpoint: {resume_path}")

    # 如果 resume_path 指向 bidirectional_step_xxx 而不是 checkpoint-step-xxx，自动进入子目录
    if resume_path:
        resume_path_obj = Path(resume_path)
        if resume_path_obj.name.startswith("bidirectional_step_"):
            ckpt_dirs = list(resume_path_obj.glob("checkpoint-step-*"))
            if ckpt_dirs:
                resume_path = str(ckpt_dirs[0])
                if accelerator.is_main_process:
                    print(f"   -> 自动定位到 checkpoint: {resume_path}")

    # 加载 checkpoint
    if resume_path and Path(resume_path).exists():
        if accelerator.is_main_process:
            print(f"\n📥 正在从 checkpoint 恢复训练: {resume_path}")
            print(f"   (注意: 这将恢复模型、优化器、调度器状态和训练进度)")

        global_step = load_checkpoint(
            model, optimizer, lr_scheduler,
            resume_path, accelerator
        )
        training_state.global_step = global_step

        # 计算应该从哪个 epoch 继续
        steps_per_epoch = len(dataloader)
        start_epoch = global_step // steps_per_epoch

        if accelerator.is_main_process:
            print(f"✅ 恢复成功!")
            print(f"   - 全局步数: {global_step}")
            print(f"   - 起始 epoch: {start_epoch + 1}")
            print(f"   - 剩余优化步数: {total_steps - global_step}")
    else:
        if accelerator.is_main_process:
            print(f"\n🚀 从头开始训练 (未找到 checkpoint)")

    # ==================== 11. 训练循环 ====================
    model.train()

    for epoch in range(start_epoch, training_config['epochs']):
        epoch_start_time = time.time()

        # mm_gap 配置（提前读取，避免 log 时重复访问）
        mm_cfg = config.get('logging', {})
        mm_enabled = bool(mm_cfg.get('mm_gap_enabled', True))
        mm_max_samples = int(mm_cfg.get('mm_gap_max_samples', 8))
        mm_target = str(mm_cfg.get('mm_gap_stage1_target', 'understanding'))

        rank_cfg = config.get('training', {}).get('stage1_rank_loss', {})
        lambda_rank = float(rank_cfg.get('lambda_rank', 0.1))
        rank_tau = float(rank_cfg.get('tau', 0.1))
        rank_k = int(rank_cfg.get('k', 4))
        rank_anchor_per_rank = int(rank_cfg.get('anchor_per_rank', 1))
        rank_interval = int(rank_cfg.get('interval', 1))
        rank_start_step = int(rank_cfg.get('start_step', 0))

        # 🔧 修复：累积统计变量（用于梯度累积 + 步间平均）
        accum_understanding = 0
        accum_generation = 0
        accum_samples = 0
        accum_loss_ntp = 0.0
        accum_loss_gene = 0.0
        accum_loss_rank = 0.0
        accum_total_loss = 0.0
        accum_rank_anchors = 0
        accum_rank_negs = 0
        accum_log_steps = 0

        # mm_gap 多 batch 采样缓冲
        mm_gap_buffer = []
        mm_gap_sample_counter = 0
        mm_gap_sample_interval = 5  # 每 5 个 micro-batch 采样一次

        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                # 🔧 修复：累积 mode 统计
                data_types = batch.get('data_type', [])
                accum_understanding += sum(1 for dt in data_types if 'understanding' in dt)
                accum_generation += sum(1 for dt in data_types if 'generation' in dt)
                accum_samples += len(data_types)

                # 理解-only 自回归训练：仅优化文本 NTP
                model_kwargs = dict(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    position_ids=batch['position_ids'],
                    modality_positions=batch['modality_positions'],
                    labels=batch['labels'],
                    gene_labels=batch.get('gene_labels'),
                    t=batch.get('t'),
                    data_type=batch.get('data_type'),
                )
                if 'gene_embeddings' in batch:
                    model_kwargs['gene_embeddings'] = batch['gene_embeddings']

                logits, loss_ntp, loss_gene = model(**model_kwargs)

                total_loss = loss_ntp

                rank_loss = torch.zeros((), device=accelerator.device, dtype=total_loss.dtype)
                rank_meta = {"anchors": 0, "negs": 0}
                should_rank = (
                    accelerator.sync_gradients
                    and lambda_rank > 0.0
                    and rank_k > 0
                    and rank_anchor_per_rank > 0
                    and global_step >= rank_start_step
                    and (global_step % max(rank_interval, 1) == 0)
                )
                if should_rank:
                    rl, rmeta = _compute_soft_rank_loss_stage1(
                        model=model,
                        batch=batch,
                        logits=logits,
                        accelerator=accelerator,
                        rank_k=rank_k,
                        tau=rank_tau,
                        anchors_per_rank=rank_anchor_per_rank,
                    )
                    if rl is not None and torch.isfinite(rl):
                        rank_loss = rl.to(dtype=total_loss.dtype)
                        total_loss = total_loss + lambda_rank * rank_loss
                        rank_meta = rmeta

                # 分布式一致的 NaN/Inf 防护：任一 rank 非有限则全体跳过该步，避免状态分叉/collective hang
                finite_local = torch.tensor(1.0 if torch.isfinite(total_loss).all() else 0.0, device=accelerator.device)
                finite_global = accelerator.reduce(finite_local, reduction='min')
                if float(finite_global.item()) < 0.5:
                    optimizer.zero_grad()
                    if accelerator.is_main_process:
                        print(f"[WARN] Non-finite loss detected at step={global_step}, skip this update.")
                    continue

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config['optimizer'].get('clip_grad_norm', 1.0))

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # mm_gap 采样（在 accumulate 上下文外，避免干扰梯度状态）
            if mm_enabled:
                mm_gap_sample_counter += 1
                if mm_gap_sample_counter % mm_gap_sample_interval == 0:
                    try:
                        gap = _compute_matched_mismatched_gap_stage1(
                            model, batch, target=mm_target, max_samples=mm_max_samples
                        )
                        if gap is not None:
                            mm_gap_buffer.append(gap.item())
                    except Exception:
                        pass

            if accelerator.sync_gradients:
                accum_loss_ntp += loss_ntp.item()
                accum_loss_gene += loss_gene.item()
                accum_loss_rank += float(rank_loss.item())
                accum_total_loss += total_loss.item()
                accum_rank_anchors += int(rank_meta.get("anchors", 0))
                accum_rank_negs += int(rank_meta.get("negs", 0))
                accum_log_steps += 1

                global_step += 1
                if global_step % config['logging']['log_interval'] == 0:
                    # 🔧 修复：使用累积的 mode 分布统计（与 loss 统计范围一致）
                    understanding_count = accum_understanding
                    generation_count = accum_generation
                    total_samples = accum_samples

                    # 一阶段：统一学习率，只有一个参数组
                    lr = optimizer.param_groups[0]['lr']

                    # 🔧 关键修复：先做步间平均（当前卡），再做全量规约
                    if accum_log_steps > 0:
                        local_avg_ntp = accum_loss_ntp / accum_log_steps
                        local_avg_gene = accum_loss_gene / accum_log_steps
                        local_avg_rank = accum_loss_rank / accum_log_steps
                        local_avg_total = accum_total_loss / accum_log_steps
                    else:
                        local_avg_ntp = 0.0
                        local_avg_gene = 0.0
                        local_avg_rank = 0.0
                        local_avg_total = 0.0

                    avg_ntp_tensor = torch.tensor(local_avg_ntp, device=accelerator.device)
                    avg_gene_tensor = torch.tensor(local_avg_gene, device=accelerator.device)
                    avg_rank_tensor = torch.tensor(local_avg_rank, device=accelerator.device)
                    avg_total_tensor = torch.tensor(local_avg_total, device=accelerator.device)
                    rank_anchor_tensor = torch.tensor(float(accum_rank_anchors), device=accelerator.device)
                    rank_neg_tensor = torch.tensor(float(accum_rank_negs), device=accelerator.device)

                    global_avg_ntp = accelerator.reduce(avg_ntp_tensor, reduction="mean").item()
                    global_avg_gene = accelerator.reduce(avg_gene_tensor, reduction="mean").item()
                    global_avg_rank = accelerator.reduce(avg_rank_tensor, reduction="mean").item()
                    global_avg_total = accelerator.reduce(avg_total_tensor, reduction="mean").item()
                    global_rank_anchors = accelerator.reduce(rank_anchor_tensor, reduction='sum').item()
                    global_rank_negs = accelerator.reduce(rank_neg_tensor, reduction='sum').item()

                    metrics = {
                        "total_loss": global_avg_total,
                        "loss_ntp": global_avg_ntp,
                        "loss_gene": global_avg_gene,
                        "loss_rank": global_avg_rank,
                        "rank_anchor_count": global_rank_anchors,
                        "rank_neg_count": global_rank_negs,
                        "lr": lr,
                        "understanding_samples": understanding_count,
                        "generation_samples": generation_count,
                    }
                    # 使用本窗口内多 batch 累积的 mm_gap 平均值
                    mm_gap = None
                    if mm_enabled:
                        if mm_gap_buffer:
                            local_mm_gap = sum(mm_gap_buffer) / len(mm_gap_buffer)
                            mm_val = torch.tensor(local_mm_gap, device=accelerator.device)
                            mm_valid = torch.tensor(1.0, device=accelerator.device)
                        else:
                            mm_val = torch.tensor(0.0, device=accelerator.device)
                            mm_valid = torch.tensor(0.0, device=accelerator.device)

                        mm_val_sum = accelerator.reduce(mm_val, reduction='sum')
                        mm_valid_sum = accelerator.reduce(mm_valid, reduction='sum')
                        valid_cnt = float(mm_valid_sum.item())
                        if valid_cnt > 0:
                            mm_gap = float((mm_val_sum / mm_valid_sum).item())
                            metrics['matched_vs_mismatched_gap'] = mm_gap

                        # 清空缓冲区，准备下一个窗口
                        mm_gap_buffer = []

                    logger.log(metrics, step=global_step)

                    if accelerator.is_main_process:
                        print(f"🧬 [Step {global_step}] Avg Loss: {global_avg_total:.4f} (NTP: {global_avg_ntp:.4f}, Gene: {global_avg_gene:.4f}, Rank: {global_avg_rank:.4f})")
                        print(f"   模式分布: 理解={understanding_count}, 生成={generation_count} (总计={total_samples})")
                        print(f"   LR: {lr:.1e} (Show-o2: 理解→NTP, 生成→Gene)")
                        if mm_gap is not None:
                            print(f"   Matched-vs-Mismatched gap: {mm_gap:.6f}")

                    # 🔧 修复：重置所有累积统计
                    accum_understanding = 0
                    accum_generation = 0
                    accum_samples = 0
                    accum_loss_ntp = 0.0
                    accum_loss_gene = 0.0
                    accum_loss_rank = 0.0
                    accum_total_loss = 0.0
                    accum_rank_anchors = 0
                    accum_rank_negs = 0
                    accum_log_steps = 0

                # 定期保存
                if global_step % config['logging']['save_interval'] == 0 and accelerator.sync_gradients:
                    save_dir = Path(config['checkpoint']['save_dir']) / f"bidirectional_step_{global_step}"
                    save_checkpoint(model, optimizer, lr_scheduler, global_step, config, output_dir=str(save_dir), accelerator=accelerator)

        epoch_time = time.time() - epoch_start_time
        accelerator.print(f"\n✅ Epoch {epoch+1} 完成，耗时: {epoch_time:.2f}s")

    accelerator.print("\n✅ Stage 1 双向训练结束！")
    accelerator.print(f"   总训练步数: {global_step:,}")

    # 保存最终模型
    final_save_dir = Path(config['checkpoint']['save_dir']) / "bidirectional_final"
    save_checkpoint(model, optimizer, lr_scheduler, global_step, config, output_dir=str(final_save_dir), accelerator=accelerator)
    accelerator.print(f"   最终模型保存至: {final_save_dir}")


if __name__ == "__main__":
    main()
