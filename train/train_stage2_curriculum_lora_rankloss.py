#!/usr/bin/env python3
# coding=utf-8
"""
🎓 Stage 2 两阶段课程学习训练脚本 (LoRA 版本)
训练顺序: EASY (简单单轮) -> COMPLEX (复杂单轮 + 多轮对话)
每个阶段都与 Stage 1 数据自然比例混合

设计原则:
1. Stage 1 使用线性 1D RoPE 完成稳定对齐
2. Stage 2 从 Stage 1 权重继续训练，LLM backbone 使用 LoRA 微调
3. Stage 2 中所有配对数据仍然严格复用 Stage 1 的构造方式，且仅用于生成任务
"""

import os
import sys
import json
import argparse
import math
import time
import collections
import pickle
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

# 注入项目路径：默认使用当前仓库根目录；如需原始大项目，可设置 SC_SHOWO_ROOT。
repo_root = Path(__file__).resolve().parents[1]
project_root = Path(os.environ.get("SC_SHOWO_ROOT", repo_root)).expanduser().resolve()
for _path in (project_root, repo_root):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from src.datasets.gene_sft_dataset_no_metadata_prompt_rankloss import SFTDataset
    from src.datasets.bidirectional_stage1_dataset_rankloss import BidirectionalStage1Dataset, WhitelistKeyDataset
    from src.models.modeling_gene_transformer_for_sft_rank_pe import GeneTransformer
    from src.train.utils.utils import SwanLabLogger, save_checkpoint, load_checkpoint, TrainingState
    from src.train.utils.scheduler_utils import build_scheduler
except ModuleNotFoundError:
    from datasets.gene_sft_dataset_no_metadata_prompt_rankloss import SFTDataset
    from datasets.bidirectional_stage1_dataset_rankloss import BidirectionalStage1Dataset, WhitelistKeyDataset
    from models.modeling_gene_transformer_for_sft_rank_pe import GeneTransformer
    from train.utils.utils import SwanLabLogger, save_checkpoint, load_checkpoint, TrainingState
    from train.utils.scheduler_utils import build_scheduler


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


SPECIAL_TOKENS_IDS = {
    'sog_id': 151665,
    'eog_id': 151666,
    'mask_gene_id': 151667,
}


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
    if logits is None or labels is None:
        return None
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_logits = torch.nan_to_num(shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    shift_labels = labels[..., 1:].contiguous()
    valid = (shift_labels != -100)
    if not valid.any():
        return None
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
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1))[valid.view(-1)],
        shift_labels.view(-1)[valid.view(-1)],
        reduction='mean',
    )


def _compute_matched_mismatched_gap_stage2(model, batch: dict, target: str, max_samples: int):
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
                gene_mask=sub.get('gene_mask'),
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


def _compute_soft_rank_loss_stage2(
    model,
    batch: dict,
    logits: torch.Tensor,
    accelerator: Accelerator,
    rank_k: int,
    tau: float,
    anchors_per_rank: int,
    target: str,
):
    device = batch['input_ids'].device
    local_bsz = int(batch['input_ids'].shape[0])
    if local_bsz <= 0 or rank_k <= 0 or anchors_per_rank <= 0:
        return None, {"anchors": 0, "negs": 0}

    data_types = batch.get('data_type', [])
    local_target = [i for i, dt in enumerate(data_types) if target in str(dt)]

    local_celltype = batch.get('celltype_id')
    if local_celltype is None:
        local_celltype = torch.zeros(local_bsz, dtype=torch.long, device=device)
    else:
        local_celltype = local_celltype.to(device=device, dtype=torch.long)

    local_mod = batch['modality_positions'][:, 0, :].to(device=device, dtype=torch.long)
    local_target_t = torch.tensor([1 if i in local_target else 0 for i in range(local_bsz)], device=device, dtype=torch.long)

    global_input_ids = accelerator.gather(_pad_2d_for_global_gather(batch['input_ids'].detach(), pad_value=151643))
    global_mod = accelerator.gather(local_mod.detach())
    global_celltype = accelerator.gather(local_celltype.detach())
    global_target = accelerator.gather(local_target_t.detach())

    has_gene_emb = 'gene_embeddings' in batch
    global_gene_emb = None
    if has_gene_emb:
        global_gene_emb = accelerator.gather(_pad_3d_for_global_gather(batch['gene_embeddings'].detach(), pad_value=0.0))

    if len(local_target) == 0:
        return None, {"anchors": 0, "negs": 0}

    perm = torch.randperm(len(local_target), device=device)
    chosen_local = [local_target[int(i)] for i in perm[:min(anchors_per_rank, len(local_target))].tolist()]

    tau = float(max(tau, 1e-3))
    world_bsz = int(global_input_ids.shape[0])
    global_base = int(accelerator.process_index * local_bsz)

    was_training = model.training
    rank_losses = []
    used_negs = 0

    try:
        for anchor_idx in chosen_local:
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

            candidate_mask = (global_target == 1) & (global_mod[:, 1] == a_len)
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
                    gene_mask=batch.get('gene_mask')[anchor_idx:anchor_idx + 1] if batch.get('gene_mask') is not None else None,
                    labels=batch['labels'][anchor_idx:anchor_idx + 1],
                    gene_labels=batch.get('gene_labels')[anchor_idx:anchor_idx + 1] if batch.get('gene_labels') is not None else None,
                    t=batch.get('t')[anchor_idx:anchor_idx + 1] if batch.get('t') is not None else None,
                    data_type=[target],
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

    return torch.stack(rank_losses).mean(), {"anchors": len(rank_losses), "negs": used_negs}


def inject_lora(model, lora_config: dict):
    """为 LLM backbone (model.showo) 注入 LoRA"""
    rank = lora_config.get('rank', 8)
    alpha = lora_config.get('alpha', 16)
    dropout = lora_config.get('dropout', 0.05)
    target_modules = lora_config.get('target_modules', ["q_proj", "v_proj"])
    bias = lora_config.get('bias', "none")
    task_type = lora_config.get('task_type', "CAUSAL_LM")

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
        task_type=task_type,
    )
    model.showo = get_peft_model(model.showo, peft_config)
    return model


def get_parameter_groups(model, config):
    """LoRA 版参数组：LoRA 参数 + 其他可训练参数 (gene_proj / rank_signal_injector / diffusion_head)"""
    lora_cfg = config.get('lora', {})
    lr_lora = lora_cfg.get('lr', 1e-4)
    lr_base = lora_cfg.get('cell_lr', config['optimizer'].get('lr', 1e-3))
    weight_decay = config['optimizer'].get('weight_decay', 1e-4)

    lora_params = []
    base_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_' in name:
            lora_params.append(param)
        else:
            base_params.append(param)

    return [
        {"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay},
        {"params": base_params, "lr": lr_base, "weight_decay": weight_decay},
    ]



def stage2_mixed_collate(batch):
    """
    混合 SFT + Stage1 paired 数据的统一 collate。

    关键要求:
    - Stage1 paired 样本的字段与取值不做任何业务改写
    - 仅在 batch 维度上对缺失字段补默认值，保证混合 batch 可堆叠
    - position_ids 统一输出为 [B, 2, S] 或 [B, S]，与当前模型兼容
    """
    batched = collections.defaultdict(list)
    all_keys = set()
    for data in batch:
        all_keys.update(data.keys())

    for data in batch:
        for key in all_keys:
            batched[key].append(data.get(key))

    result = {}
    batch_size = len(batch)

    for key, values in batched.items():
        if key == 'data_type':
            flattened = []
            for value in values:
                if value is None:
                    flattened.append('unknown')
                elif isinstance(value, list):
                    flattened.extend(value)
                else:
                    flattened.append(value)
            result[key] = flattened
            continue

        present_values = [value for value in values if value is not None]
        if not present_values:
            continue

        sample_value = present_values[0]

        if key in ('texts', 'metadata'):
            result[key] = values
            continue

        if not isinstance(sample_value, torch.Tensor):
            result[key] = values
            continue

        if sample_value.dim() == 0:
            filled = [value if value is not None else torch.zeros_like(sample_value) for value in values]
            result[key] = torch.stack(filled, dim=0)
            continue

        if key == 'input_ids':
            filled = [value if value is not None else torch.empty(0, dtype=sample_value.dtype) for value in values]
            result[key] = pad_sequence(filled, batch_first=True, padding_value=151643)
            continue

        if key == 'labels':
            filled = [value if value is not None else torch.empty(0, dtype=sample_value.dtype) for value in values]
            result[key] = pad_sequence(filled, batch_first=True, padding_value=-100)
            continue

        if key == 'position_ids':
            filled = []
            if sample_value.dim() == 2:
                max_len = max((value.shape[1] if value is not None else 0) for value in values)
                for value in values:
                    if value is None:
                        value = torch.zeros(sample_value.shape[0], max_len, dtype=sample_value.dtype)
                    elif value.shape[1] < max_len:
                        pad = torch.zeros(value.shape[0], max_len - value.shape[1], dtype=value.dtype, device=value.device)
                        value = torch.cat([value, pad], dim=1)
                    filled.append(value)
                result[key] = torch.stack(filled, dim=0)  # [B, 2, S]
            else:
                filled = [value if value is not None else torch.empty(0, dtype=sample_value.dtype) for value in values]
                result[key] = pad_sequence(filled, batch_first=True, padding_value=0)
            continue

        if key == 'modality_positions':
            filled = [value if value is not None else torch.zeros_like(sample_value) for value in values]
            result[key] = torch.stack(filled, dim=0)
            continue

        if sample_value.dim() == 1:
            if key in ('gene_labels', 'gene_ids'):
                filled = [value if value is not None else torch.empty(0, dtype=sample_value.dtype) for value in values]
                result[key] = pad_sequence(filled, batch_first=True, padding_value=-100 if key == 'gene_labels' else 0)
            elif key in ('gene_mask', 'non_zero_mask'):
                filled = [value if value is not None else torch.empty(0, dtype=sample_value.dtype) for value in values]
                result[key] = pad_sequence(filled, batch_first=True, padding_value=0)
            else:
                filled = [value if value is not None else torch.empty(0, dtype=sample_value.dtype) for value in values]
                result[key] = pad_sequence(filled, batch_first=True, padding_value=0)
            continue

        if key == 'gene_embeddings':
            filled = [
                value if value is not None else torch.empty((0, sample_value.shape[-1]), dtype=sample_value.dtype)
                for value in values
            ]
            result[key] = pad_sequence(filled, batch_first=True, padding_value=0.0)
        else:
            filled = [value if value is not None else torch.zeros_like(sample_value) for value in values]
            result[key] = torch.stack(filled, dim=0)

    if 'data_type' in result and len(result['data_type']) != batch_size:
        normalized = []
        for value in batched.get('data_type', []):
            if value is None:
                normalized.append('unknown')
            elif isinstance(value, list) and len(value) > 0:
                normalized.append(str(value[0]))
            else:
                normalized.append(str(value))
        result['data_type'] = normalized[:batch_size]

    # 明确保证 position_ids 为 [B, 2, S]，避免 2D RoPE 维度被误置换。
    if 'position_ids' in result and isinstance(result['position_ids'], torch.Tensor):
        pid = result['position_ids']
        if pid.dim() == 3 and pid.shape[0] == 2 and pid.shape[1] == batch_size:
            result['position_ids'] = pid.transpose(0, 1).contiguous()

    return result



def train_stage(
    model,
    optimizer,
    lr_scheduler,
    dataloader,
    accelerator,
    logger,
    training_state,
    config,
    stage_name: str,
    global_step: int,
    start_epoch: int = 0,
):
    """训练一个课程阶段，并输出多 batch + 多卡平均 loss。"""
    model.train()

    accelerator.print(f"\n{'=' * 80}")
    accelerator.print(f"🎓 开始训练阶段: {stage_name}")
    accelerator.print(f"{'=' * 80}\n")

    stage1_rank_cfg = config.get('training', {}).get('stage1_rank_loss', {})
    stage2_rank_cfg = config.get('training', {}).get('stage2_rank_loss', {})
    lambda_rank = float(stage2_rank_cfg.get('lambda_rank', stage1_rank_cfg.get('lambda_rank', 0.1)))
    rank_tau = float(stage2_rank_cfg.get('tau', stage1_rank_cfg.get('tau', 0.1)))
    rank_k = int(stage2_rank_cfg.get('k', stage1_rank_cfg.get('k', 4)))
    rank_anchor_per_rank = int(stage2_rank_cfg.get('anchor_per_rank', stage1_rank_cfg.get('anchor_per_rank', 1)))
    rank_interval = int(stage2_rank_cfg.get('interval', stage1_rank_cfg.get('interval', 1)))
    rank_start_step = int(stage2_rank_cfg.get('start_step', stage1_rank_cfg.get('start_step', 0)))
    rank_target = str(stage2_rank_cfg.get('target', 'stage2'))

    accum_loss_ntp = 0.0
    accum_loss_gene = 0.0
    accum_loss_rank = 0.0
    accum_total_loss = 0.0
    accum_rank_anchors = 0
    accum_rank_negs = 0
    accum_log_steps = 0

    for epoch in range(start_epoch, config['training']['epochs']):
        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                model_kwargs = dict(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    position_ids=batch['position_ids'],
                    modality_positions=batch['modality_positions'],
                    gene_mask=batch.get('gene_mask'),
                    labels=batch['labels'],
                    gene_labels=batch.get('gene_labels'),
                    t=batch.get('t'),
                    data_type=batch['data_type'],
                )
                if 'gene_embeddings' in batch:
                    model_kwargs['gene_embeddings'] = batch['gene_embeddings']
                logits, loss_ntp, loss_gene, detailed_losses = model(**model_kwargs)

                # SFT 模型内部已按 data_type 分别计算，stage2 样本只计 NTP loss
                total_loss = loss_ntp + loss_gene

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
                    rl, rmeta = _compute_soft_rank_loss_stage2(
                        model=model,
                        batch=batch,
                        logits=logits,
                        accelerator=accelerator,
                        rank_k=rank_k,
                        tau=rank_tau,
                        anchors_per_rank=rank_anchor_per_rank,
                        target=rank_target,
                    )
                    if rl is not None and torch.isfinite(rl):
                        rank_loss = rl.to(dtype=total_loss.dtype)
                        total_loss = total_loss + lambda_rank * rank_loss
                        rank_meta = rmeta

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config['optimizer'].get('clip_grad_norm', 1.0))

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

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
                        "lr_lora": optimizer.param_groups[0]['lr'],
                        "lr_base": optimizer.param_groups[1]['lr'],
                        "curriculum_stage": stage_name,
                    }

                    mm_gap = None
                    mm_cfg = config.get('logging', {})
                    mm_enabled = bool(mm_cfg.get('mm_gap_enabled', True))
                    mm_interval = int(mm_cfg.get('mm_gap_interval', config['logging']['log_interval']))
                    mm_max_samples = int(mm_cfg.get('mm_gap_max_samples', 8))
                    mm_target = str(mm_cfg.get('mm_gap_stage2_target', 'stage2'))

                    if mm_enabled and (global_step % max(1, mm_interval) == 0):
                        try:
                            mm_gap_local = _compute_matched_mismatched_gap_stage2(
                                model, batch, target=mm_target, max_samples=mm_max_samples
                            )
                            if mm_gap_local is not None:
                                mm_gap = accelerator.reduce(mm_gap_local.to(accelerator.device), reduction='mean').item()
                                metrics['matched_vs_mismatched_gap'] = mm_gap
                        except Exception as e:
                            if accelerator.is_main_process:
                                print(f"[MM-Gap] Step {global_step} probe failed: {e}")

                    logger.log(metrics, step=global_step)

                    if accelerator.is_main_process:
                        print(
                            f"🎓 [{stage_name}] Step {global_step} | Avg Loss: {global_avg_total:.4f} "
                            f"(NTP: {global_avg_ntp:.4f}, Gene: {global_avg_gene:.4f}, Rank: {global_avg_rank:.4f})"
                        )
                        print(
                            f"   Learning Rates -> LoRA: {optimizer.param_groups[0]['lr']:.1e}, "
                            f"Base: {optimizer.param_groups[1]['lr']:.1e}"
                        )
                        if mm_gap is not None:
                            print(f"   Matched-vs-Mismatched gap: {mm_gap:.6f}")

                    accum_loss_ntp = 0.0
                    accum_loss_gene = 0.0
                    accum_loss_rank = 0.0
                    accum_total_loss = 0.0
                    accum_rank_anchors = 0
                    accum_rank_negs = 0
                    accum_log_steps = 0

                if global_step % config['logging']['save_interval'] == 0:
                    save_dir = Path(config['checkpoint']['save_dir']) / f"{stage_name.lower()}_step_{global_step}"
                    save_checkpoint(model, optimizer, lr_scheduler, global_step, config, output_dir=str(save_dir), accelerator=accelerator)

    return global_step



def build_stage1_dataset(config, tokenizer, accelerator):
    """Stage 2 中所有配对数据严格复用 Stage 1 的理解样本构造方式。"""
    whitelist_path = config['data'].get('stage1_whitelist_json')

    if whitelist_path and os.path.exists(whitelist_path):
        base = BidirectionalStage1Dataset(
            config_dict=config,
            special_tokens_ids=SPECIAL_TOKENS_IDS,
            text_tokenizer=tokenizer,
            accelerator=None,
            max_seq_len=config['dataset'].get('max_seq_len', 1800),
            understanding_ratio=1.0,
            skip_load_data=True,
        )
        return WhitelistKeyDataset(base, whitelist_path, accelerator)

    return BidirectionalStage1Dataset(
        config_dict=config,
        special_tokens_ids=SPECIAL_TOKENS_IDS,
        text_tokenizer=tokenizer,
        accelerator=accelerator,
        max_seq_len=config['dataset'].get('max_seq_len', 1800),
        understanding_ratio=1.0,
    )



def resolve_stage2_resume_path(config, accelerator):
    """
    解析 stage2 resume 路径，仅从 config 中读取 resume_from 配置。
    不自动发现 checkpoint，没有配置则返回 None（从头训练）。
    """
    resume_path = config.get('checkpoint', {}).get('resume_from')

    if resume_path:
        resume_path_obj = Path(resume_path)
        if resume_path_obj.is_dir():
            # 如果指向的是 step 目录，自动定位到其中的 checkpoint 子目录
            if resume_path_obj.name.startswith(('easy_step_', 'complex_step_')):
                ckpt_dirs = list(resume_path_obj.glob('checkpoint-step-*'))
                if ckpt_dirs:
                    resume_path = str(ckpt_dirs[0])
                    if accelerator.is_main_process:
                        print(f"\n📥 从配置恢复 stage2 checkpoint: {resume_path}")
                else:
                    if accelerator.is_main_process:
                        print(f"\n⚠️ 配置的 resume_from 目录下未找到 checkpoint: {resume_path}")
                    resume_path = None
            else:
                if accelerator.is_main_process:
                    print(f"\n📥 从配置恢复 stage2 checkpoint: {resume_path}")
        elif not resume_path_obj.exists():
            if accelerator.is_main_process:
                print(f"\n⚠️ 配置的 resume_from 路径不存在: {resume_path}")
            resume_path = None
    else:
        if accelerator.is_main_process:
            print("\n🚀 未配置 resume_from，Stage 2 将从头开始训练")

    return resume_path



def _resolve_checkpoint_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_file():
        return str(p)
    if p.is_dir():
        candidates = [
            p / 'state.pt',
            p / 'pytorch_model.bin',
            p / 'model.pt',
            p / 'checkpoint.pt',
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return str(c)
    raise FileNotFoundError(f"Cannot resolve checkpoint from: {path_str}")


def _load_stage1_state_dict_compat(ckpt_path: str):
    resolved = _resolve_checkpoint_path(ckpt_path)
    try:
        state = torch.load(resolved, map_location='cpu', weights_only=True)
    except Exception as e:
        if isinstance(e, pickle.UnpicklingError) or 'Weights only load failed' in str(e):
            state = torch.load(resolved, map_location='cpu', weights_only=False)
        else:
            raise

    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    return state, resolved


def main():
    accelerator = Accelerator()

    config_path = resolve_config_path()
    accelerator.print(f"📄 使用配置: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    training_config = config['training']
    model_path = config['model']['llm_model_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = GeneTransformer(
        llm_vocab_size=tokenizer.vocab_size,
        llm_model_path=model_path,
        load_from_showo=False,
        config_dict=config,
        special_tokens_ids=SPECIAL_TOKENS_IDS,
    )
    model.requires_grad_(True)

    # ==========================================
    # 注入 LoRA
    # ==========================================
    lora_cfg = config.get('lora', {})
    model = inject_lora(model, lora_cfg)

    # 冻结 LLM 原始参数，只保留 LoRA + gene_proj + diffusion_head；若禁用 gene PE，则同步关闭 rank_signal
    disable_gene_pe = bool(config.get('model', {}).get('disable_gene_position_ids', False))
    for name, param in model.named_parameters():
        if 'lora_' in name or 'gene_proj' in name or 'diffusion_head' in name:
            param.requires_grad = True
        elif 'rank_signal_injector' in name:
            param.requires_grad = (not disable_gene_pe)
        else:
            param.requires_grad = False

    if accelerator.is_main_process and disable_gene_pe:
        print("⚠️ disable_gene_position_ids=True: Stage2 LoRA 已冻结 rank_signal_injector")

    stage1_ckpt = config['checkpoint'].get('stage1_weights_path')
    if stage1_ckpt:
        try:
            state_dict, resolved_ckpt = _load_stage1_state_dict_compat(stage1_ckpt)
            accelerator.print(f"📥 正在挂载一阶段对齐权重：{resolved_ckpt}")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            accelerator.print(f"✅ Stage1->Stage2 权重加载完成: missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:
                accelerator.print(f"   first_missing: {missing[:10]}")
            if unexpected:
                accelerator.print(f"   first_unexpected: {unexpected[:10]}")
        except Exception as e:
            accelerator.print(f"❌ Stage1 权重加载失败: {e}")
            raise
    else:
        accelerator.print("⚠️ 未发现一阶段权重，将从随机初始化开始")

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n)
        base_params = trainable_params - lora_params
        print(f"\n📊 参数统计:")
        print(f"   总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"   可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"   └─ LoRA 参数: {lora_params:,} ({lora_params/1e6:.2f}M)")
        print(f"   └─ Base 参数 (gene_proj/rank_signal/diffusion): {base_params:,} ({base_params/1e6:.2f}M)")

    accelerator.print("\n🎓 初始化课程学习数据加载器...")

    accelerator.print("\n🧬 加载 Stage 1 理解配对数据（严格复用 Stage 1 理解逻辑）...")
    stage1_dataset = build_stage1_dataset(config, tokenizer, accelerator)
    accelerator.print(f"   Stage 1 paired 数据量: {len(stage1_dataset):,}")

    accelerator.print("\n🟢 EASY: 简单单轮 SFT + Stage1 understanding paired")
    easy_dataset = SFTDataset(
        json_paths=config['data'].get('sft_json_paths'),
        text_tokenizer=tokenizer,
        config_dict=config,
        special_tokens_ids=SPECIAL_TOKENS_IDS,
        max_seq_len=config['dataset'].get('max_seq_len', 2048),
        accelerator=accelerator,
        curriculum_stage='EASY',
    )
    accelerator.print(f"   EASY 数据集: {len(easy_dataset):,}")

    accelerator.print("\n🟡 COMPLEX: 复杂单轮/多轮 SFT + Stage1 understanding paired")
    complex_dataset = SFTDataset(
        json_paths=config['data'].get('sft_json_paths'),
        text_tokenizer=tokenizer,
        config_dict=config,
        special_tokens_ids=SPECIAL_TOKENS_IDS,
        max_seq_len=config['dataset'].get('max_seq_len', 2048),
        accelerator=accelerator,
        curriculum_stage='COMPLEX',
    )
    accelerator.print(f"   COMPLEX 数据集: {len(complex_dataset):,}")

    batch_size = training_config.get('batch_size', 5)
    epochs = training_config.get('epochs', 1)
    grad_accum = accelerator.gradient_accumulation_steps
    world_size = max(1, accelerator.num_processes)

    easy_total_samples = len(easy_dataset) + len(stage1_dataset)
    complex_total_samples = len(complex_dataset) + len(stage1_dataset)

    easy_data_steps_per_epoch = math.ceil(easy_total_samples / batch_size)
    complex_data_steps_per_epoch = math.ceil(complex_total_samples / batch_size)

    easy_optimizer_steps_per_epoch = math.ceil(easy_data_steps_per_epoch / grad_accum)
    complex_optimizer_steps_per_epoch = math.ceil(complex_data_steps_per_epoch / grad_accum)
    easy_steps = easy_optimizer_steps_per_epoch * epochs
    complex_steps = complex_optimizer_steps_per_epoch * epochs
    total_steps = easy_steps + complex_steps

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("📊 课程学习训练计划:")
        print(f"   EASY SFT: {len(easy_dataset):,} samples")
        print(f"   COMPLEX SFT: {len(complex_dataset):,} samples")
        print(f"   Stage1 paired: {len(stage1_dataset):,} samples")
        print(f"   world_size: {world_size}")
        print(f"   梯度累积步数: {grad_accum}")
        print(f"   EASY 每进程数据步/epoch: {easy_data_steps_per_epoch:,}")
        print(f"   COMPLEX 每进程数据步/epoch: {complex_data_steps_per_epoch:,}")
        print(f"   总优化步数(估算): {total_steps:,}")
        print("=" * 80)

    optimizer_type = config['optimizer'].get('type', 'AdamW')
    if accelerator.is_main_process:
        print(f"\n📊 Optimizer Config:")
        print(f"   type: {optimizer_type}")
        if optimizer_type != 'AdamW':
            print(f"   ⚠️  Warning: 当前脚本为 AdamW 专用版本，但配置中 type={optimizer_type}")
            print(f"   ⚠️  将强制使用 AdamW，忽略 type 设置")

    param_groups = get_parameter_groups(model, config)
    optimizer = AdamW(param_groups)

    lr_scheduler, scheduler_info = build_scheduler(
        optimizer=optimizer,
        scheduler_config=config['scheduler'],
        computed_total_steps=total_steps,
    )

    if accelerator.is_main_process:
        print(f"\n📊 Scheduler Config:")
        print(f"   type: {scheduler_info['type']}")
        print(f"   num_training_steps: {scheduler_info['num_training_steps']:,}")
        print(f"   num_warmup_steps: {scheduler_info['num_warmup_steps']:,}")
        if scheduler_info['type'] == 'linear_warmup_cosine_decay':
            print(f"   min_lr_ratio: {scheduler_info['min_lr_ratio']}")

    logger = SwanLabLogger(config, accelerator)
    training_state = TrainingState()

    easy_combined = ConcatDataset([easy_dataset, stage1_dataset])
    complex_combined = ConcatDataset([complex_dataset, stage1_dataset])

    easy_loader = DataLoader(
        easy_combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config.get('num_workers', 4),
        collate_fn=stage2_mixed_collate,
        drop_last=False,
    )
    complex_loader = DataLoader(
        complex_combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config.get('num_workers', 4),
        collate_fn=stage2_mixed_collate,
        drop_last=False,
    )

    model, optimizer, lr_scheduler, easy_loader, complex_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, easy_loader, complex_loader
    )

    global_step = 0
    easy_steps_total = easy_optimizer_steps_per_epoch * training_config['epochs']
    complex_steps_total = complex_optimizer_steps_per_epoch * training_config['epochs']
    resume_path = resolve_stage2_resume_path(config, accelerator)

    if resume_path and Path(resume_path).exists():
        if accelerator.is_main_process:
            print(f"\n📥 正在从 stage2 checkpoint 恢复训练: {resume_path}")
            print("   (注意: 这将恢复模型、优化器、调度器状态和训练进度)")

        global_step = load_checkpoint(model, optimizer, lr_scheduler, resume_path, accelerator)
        training_state.global_step = global_step

        if accelerator.is_main_process:
            print("✅ Stage 2 恢复成功!")
            print(f"   - 全局步数: {global_step}")
            print(f"   - EASY 总步数: {easy_steps_total}")
            print(f"   - COMPLEX 总步数: {complex_steps_total}")
    elif resume_path:
        if accelerator.is_main_process:
            print(f"\n❌ 错误: 配置的 resume_from 路径不存在: {resume_path}")
            print("   请检查 config.json 中的 checkpoint.resume_from 配置")
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    else:
        if accelerator.is_main_process:
            print("\n🚀 Stage 2 从头开始训练 (未配置 resume_from)")

    easy_start_epoch = 0
    complex_start_epoch = 0

    if global_step < easy_steps_total:
        easy_start_epoch = global_step // easy_optimizer_steps_per_epoch
        global_step = train_stage(
            model, optimizer, lr_scheduler, easy_loader, accelerator, logger,
            training_state, config, 'EASY', global_step, start_epoch=easy_start_epoch
        )
        global_step = train_stage(
            model, optimizer, lr_scheduler, complex_loader, accelerator, logger,
            training_state, config, 'COMPLEX', global_step, start_epoch=0
        )
    else:
        complex_completed_steps = global_step - easy_steps_total
        complex_start_epoch = complex_completed_steps // complex_optimizer_steps_per_epoch
        if accelerator.is_main_process:
            print(f"⏭️ EASY 阶段已完成，直接从 COMPLEX epoch {complex_start_epoch + 1} 继续")
        global_step = train_stage(
            model, optimizer, lr_scheduler, complex_loader, accelerator, logger,
            training_state, config, 'COMPLEX', global_step, start_epoch=complex_start_epoch
        )

    accelerator.print("\n✅ 两阶段课程学习训练圆满结束！")
    accelerator.print(f"   总训练步数: {global_step:,}")

    final_save_dir = Path(config['checkpoint']['save_dir']) / 'stage2_curriculum_final'
    save_checkpoint(model, optimizer, lr_scheduler, global_step, config, output_dir=str(final_save_dir), accelerator=accelerator)
    accelerator.print(f"   最终模型保存至: {final_save_dir}")


if __name__ == '__main__':
    main()
