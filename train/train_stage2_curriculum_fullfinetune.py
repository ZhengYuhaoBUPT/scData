#!/usr/bin/env python3
# coding=utf-8
"""
🎓 Stage 2 两阶段课程学习训练脚本 (v2)
训练顺序: EASY (简单单轮) -> COMPLEX (复杂单轮 + 多轮对话)
每个阶段都与 Stage 1 数据自然比例混合

设计原则:
1. Stage 1 使用线性 1D RoPE 完成稳定对齐
2. Stage 2 从 Stage 1 权重继续训练，开启 Gene RoPE
3. Stage 2 中所有配对数据仍然严格复用 Stage 1 的构造方式
"""

import os
import sys
import json
import math
import time
import collections
import torch
from pathlib import Path
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path as PathLib
import json as json_mod

# 注入项目路径
project_root = Path("/root/wanghaoran/zxy/project/sc_showo")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.datasets.gene_sft_dataset_no_metadata_prompt import SFTDataset
from src.datasets.bidirectional_stage1_dataset import BidirectionalStage1Dataset
from src.models.modeling_gene_transformer_for_sft_rank_pe import GeneTransformer
from src.train.utils.utils import SwanLabLogger, save_checkpoint, load_checkpoint, TrainingState
from src.train.utils.scheduler_utils import build_scheduler


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

    with torch.no_grad():
        out = model(
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
        matched_ntp = out[1] if isinstance(out, (tuple, list)) else None
        if matched_ntp is None:
            return None

        mismatched_ids = sub['input_ids'].clone()
        for i in range(n):
            si = int(sub['modality_positions'][i, 0, 0].item())
            li = int(sub['modality_positions'][i, 0, 1].item())
            j = int(perm[i].item())
            sj = int(sub['modality_positions'][j, 0, 0].item())
            lj = int(sub['modality_positions'][j, 0, 1].item())
            if li <= 0 or lj <= 0 or li != lj:
                return None
            mismatched_ids[i, si:si + li] = sub['input_ids'][j, sj:sj + lj]

        out_mis = model(
            input_ids=mismatched_ids,
            attention_mask=sub.get('attention_mask'),
            position_ids=sub.get('position_ids'),
            modality_positions=sub['modality_positions'],
            gene_mask=sub.get('gene_mask'),
            labels=sub['labels'],
            gene_labels=sub.get('gene_labels'),
            t=sub.get('t'),
            data_type=sub.get('data_type'),
        )
        mismatched_ntp = out_mis[1] if isinstance(out_mis, (tuple, list)) else None
        if mismatched_ntp is None:
            return None

    return (mismatched_ntp - matched_ntp).detach()


def get_parameter_groups(model, config):
    """分层学习率参数组"""
    lr_rank_signal = config['optimizer'].get('stage2_lr_rank_signal', 1e-3)
    lr_diffusion = config['optimizer'].get('stage2_lr_diffusion', 1e-4)
    lr_llm = config['optimizer'].get('stage2_lr_llm', 1e-5)
    weight_decay = config['optimizer'].get('weight_decay', 1e-4)

    rank_signal_params = []
    head_params = []
    llm_params = []

    disable_gene_pe = bool(config.get('model', {}).get('disable_gene_position_ids', False))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'rank_signal_injector' in name:
            if disable_gene_pe:
                param.requires_grad = False
                continue
            rank_signal_params.append(param)
        elif 'gene_proj' in name:
            rank_signal_params.append(param)
        elif 'diffusion_head' in name:
            head_params.append(param)
        else:
            llm_params.append(param)

    return [
        {"params": rank_signal_params, "lr": lr_rank_signal, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_diffusion, "weight_decay": weight_decay},
        {"params": llm_params, "lr": lr_llm, "weight_decay": weight_decay},
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

    accum_loss_ntp = 0.0
    accum_loss_gene = 0.0
    accum_total_loss = 0.0
    accum_log_steps = 0

    for epoch in range(start_epoch, config['training']['epochs']):
        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                logits, loss_ntp, loss_gene, detailed_losses = model(
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

                # SFT 模型内部已按 data_type 分别计算，stage2 样本只计 NTP loss
                total_loss = loss_ntp + loss_gene

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config['optimizer'].get('clip_grad_norm', 1.0))

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                accum_loss_ntp += loss_ntp.item()
                accum_loss_gene += loss_gene.item()
                accum_total_loss += total_loss.item()
                accum_log_steps += 1
                global_step += 1

                if global_step % config['logging']['log_interval'] == 0:
                    if accum_log_steps > 0:
                        local_avg_ntp = accum_loss_ntp / accum_log_steps
                        local_avg_gene = accum_loss_gene / accum_log_steps
                        local_avg_total = accum_total_loss / accum_log_steps
                    else:
                        local_avg_ntp = 0.0
                        local_avg_gene = 0.0
                        local_avg_total = 0.0

                    avg_ntp_tensor = torch.tensor(local_avg_ntp, device=accelerator.device)
                    avg_gene_tensor = torch.tensor(local_avg_gene, device=accelerator.device)
                    avg_total_tensor = torch.tensor(local_avg_total, device=accelerator.device)

                    global_avg_ntp = accelerator.reduce(avg_ntp_tensor, reduction="mean").item()
                    global_avg_gene = accelerator.reduce(avg_gene_tensor, reduction="mean").item()
                    global_avg_total = accelerator.reduce(avg_total_tensor, reduction="mean").item()

                    metrics = {
                        "total_loss": global_avg_total,
                        "loss_ntp": global_avg_ntp,
                        "loss_gene": global_avg_gene,
                        "lr_llm": optimizer.param_groups[2]['lr'],
                        "lr_rank_signal": optimizer.param_groups[0]['lr'],
                        "lr_diffusion": optimizer.param_groups[1]['lr'],
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
                            f"(NTP: {global_avg_ntp:.4f}, Gene: {global_avg_gene:.4f})"
                        )
                        print(
                            f"   Learning Rates -> LLM: {optimizer.param_groups[2]['lr']:.1e}, "
                            f"RankSignal: {optimizer.param_groups[0]['lr']:.1e}, "
                            f"Diffusion: {optimizer.param_groups[1]['lr']:.1e}"
                        )
                        if mm_gap is not None:
                            print(f"   Matched-vs-Mismatched gap: {mm_gap:.6f}")

                    accum_loss_ntp = 0.0
                    accum_loss_gene = 0.0
                    accum_total_loss = 0.0
                    accum_log_steps = 0

                if global_step % config['logging']['save_interval'] == 0:
                    save_dir = Path(config['checkpoint']['save_dir']) / f"{stage_name.lower()}_step_{global_step}"
                    save_checkpoint(model, optimizer, lr_scheduler, global_step, config, output_dir=str(save_dir), accelerator=accelerator)

    return global_step



def build_stage1_dataset(config, tokenizer, accelerator):
    """Stage 2 中所有配对数据严格复用 Stage 1 的构造方式。"""
    stage1_full = BidirectionalStage1Dataset(
        config_dict=config,
        special_tokens_ids=SPECIAL_TOKENS_IDS,
        text_tokenizer=tokenizer,
        accelerator=accelerator,
        max_seq_len=config['dataset'].get('max_seq_len', 1800),
        understanding_ratio=0.0,
    )

    whitelist_path = config['data'].get('stage1_whitelist_json')
    if whitelist_path and os.path.exists(whitelist_path):
        with open(whitelist_path, 'r') as f:
            whitelist = json_mod.load(f)
        valid_indices = []
        for block_idx, block in enumerate(stage1_full.data_blocks):
            db_name = PathLib(block['cluster_path']).stem
            if db_name.endswith('_cluster'):
                db_name = db_name[:-len('_cluster')]
            allowed_keys = set(whitelist.get(db_name, []))
            if not allowed_keys:
                continue
            start_offset = stage1_full.cumulative_sizes[block_idx]
            for local_idx, key in enumerate(block['lmdb_keys']):
                kstr = key.decode('utf-8', errors='ignore') if isinstance(key, (bytes, bytearray)) else str(key)
                if kstr in allowed_keys:
                    valid_indices.append(start_offset + local_idx)
        return Subset(stage1_full, valid_indices)

    return stage1_full



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


def main():
    accelerator = Accelerator()

    config_path = project_root / "config/config.json"
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
    if bool(config.get('model', {}).get('disable_gene_position_ids', False)):
        for name, param in model.named_parameters():
            if 'rank_signal_injector' in name:
                param.requires_grad = False
        accelerator.print("⚠️ disable_gene_position_ids=True: Stage2 已冻结 rank_signal_injector")

    stage1_ckpt = config['checkpoint'].get('stage1_weights_path')
    if stage1_ckpt and os.path.exists(stage1_ckpt):
        accelerator.print(f"📥 正在挂载一阶段对齐权重：{stage1_ckpt}")
        state_dict = torch.load(stage1_ckpt, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict) and 'model' in state_dict and isinstance(state_dict['model'], dict):
            state_dict = state_dict['model']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"✅ Stage1->Stage2 权重加载完成: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            accelerator.print(f"   first_missing: {missing[:10]}")
        if unexpected:
            accelerator.print(f"   first_unexpected: {unexpected[:10]}")
    else:
        accelerator.print("⚠️ 未发现一阶段权重，将从随机初始化开始")

    accelerator.print("\n🎓 初始化课程学习数据加载器...")

    accelerator.print("\n🧬 加载 Stage 1 配对数据（严格复用 Stage 1 逻辑）...")
    stage1_dataset = build_stage1_dataset(config, tokenizer, accelerator)
    accelerator.print(f"   Stage 1 paired 数据量: {len(stage1_dataset):,}")

    accelerator.print("\n🟢 EASY: 简单单轮 SFT + Stage1 paired")
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

    accelerator.print("\n🟡 COMPLEX: 复杂单轮/多轮 SFT + Stage1 paired")
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

    # 注意：accelerate.prepare 后 DataLoader 会被按进程切分。
    # 这里先按“每进程”估算步数，避免 scheduler 用全局步数导致学习率计划过慢。
    easy_total_samples = len(easy_dataset) + len(stage1_dataset)
    complex_total_samples = len(complex_dataset) + len(stage1_dataset)

    easy_global_data_steps_per_epoch = math.ceil(easy_total_samples / batch_size)
    complex_global_data_steps_per_epoch = math.ceil(complex_total_samples / batch_size)

    easy_data_steps_per_epoch = math.ceil(easy_global_data_steps_per_epoch / world_size)
    complex_data_steps_per_epoch = math.ceil(complex_global_data_steps_per_epoch / world_size)

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

    # =====================================================
    # AdamW 优化器配置 (原脚本专用)
    # =====================================================
    optimizer_type = config['optimizer'].get('type', 'AdamW')
    if accelerator.is_main_process:
        print(f"\n📊 Optimizer Config:")
        print(f"   type: {optimizer_type}")
        if optimizer_type != 'AdamW':
            print(f"   ⚠️  Warning: 当前脚本为 AdamW 专用版本，但配置中 type={optimizer_type}")
            print(f"   ⚠️  将强制使用 AdamW，忽略 type 设置")

    param_groups = get_parameter_groups(model, config)
    optimizer = AdamW(param_groups)

    # 使用新的 scheduler 工具构建学习率调度器
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
        # 配置了 resume_from 但路径不存在
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
