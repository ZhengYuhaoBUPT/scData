#!/usr/bin/env python3
# coding=utf-8
"""
🚀 Stage 2: Multimodal Instruction Tuning (SFT) + Gene Diffusion 混合训练脚本
训练特性：
1. 权重继承：自动加载一阶段对齐后的 Embedder 与 Diffusion Head 权重。
2. 全量解冻：激活 LLM (Qwen2.5) 权重，通过分层学习率进行指令对齐。
3. 双轨 Loss：同时计算 SFT NTP Loss 与基因重建 MTP Loss，保持生成上限。
4. 极致加速：集成 DeepSpeed ZeRO-2/3 与 BF16 混合精度。
"""

import os
import sys
import json
import time
import math
import torch
from pathlib import Path
from typing import Dict, Any
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoTokenizer, get_scheduler

# 1. 注入项目路径
project_root = Path("/root/wanghaoran/zxy/project/sc_showo")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.datasets.mixed_dataloader import create_mixed_dataloader
from src.models.modeling_gene_transformer_for_sft import GeneTransformer
from src.train.utils.utils import SwanLabLogger, save_checkpoint, TrainingState

def get_parameter_groups(model, config):
    """
    🛠️ 核心黑科技：构建分层学习率参数组
    根据 config.json 分别为 Embedder, Diffusion Head 和 LLM 设置不同的更新强度
    """
    lr_embedder = config['optimizer'].get('stage2_lr_embedder', 1e-3)
    lr_diffusion = config['optimizer'].get('stage2_lr_diffusion', 1e-4)
    lr_llm = config['optimizer'].get('stage2_lr_llm', 1e-5)
    weight_decay = config['optimizer'].get('weight_decay', 1e-4)

    # 分组提取
    embedder_params = [p for n, p in model.named_parameters() if "cell_embedder" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "diffusion_head" in n and p.requires_grad]  # ✅ 同时匹配 a 和 b
    llm_params = [p for n, p in model.named_parameters() if "model." in n and p.requires_grad]

    param_groups = [
        {"params": embedder_params, "lr": lr_embedder, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_diffusion, "weight_decay": weight_decay},
        {"params": llm_params, "lr": lr_llm, "weight_decay": weight_decay},
    ]
    return param_groups

def main():
    # ==================== 1. 初始化加速器 ====================
    # ✅ 修复：完全从 accelerate_configs/*.yaml 读取配置
    # 包括 gradient_accumulation_steps, mixed_precision, deepspeed_config 等
    accelerator = Accelerator()

    # ==================== 2. 加载配置 ====================
    config_path = project_root / "config/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # 提取常用配置子项，方便后续使用
    training_config = config['training']

    # ==================== 3. 加载 Tokenizer ====================
    model_path = config['model']['llm_model_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ==================== 4. 定义特殊 Tokens ====================
    special_tokens_ids = {
        'soc_id': 151669, 'eoc_id': 151670, 
        'sog_id': 151665, 'eog_id': 151666, 
        'mask_gene_id': 151667
    }

    # ==================== 5. 初始化模型 ====================
    model = GeneTransformer(
        llm_vocab_size=tokenizer.vocab_size,
        llm_model_path=model_path,
        load_from_showo=False,
        config_dict=config,
        special_tokens_ids=special_tokens_ids
    )
    
    # 🔓 关键点：全面解冻。如果你显存紧张，可以只解冻最后几层
    model.requires_grad_(True)
    
    # ==================== 6. 挂载一阶段权重 ====================
    stage1_ckpt = config['checkpoint'].get('stage1_weights_path')
    if stage1_ckpt and os.path.exists(stage1_ckpt):
        accelerator.print(f"📥 正在挂载一阶段对齐权重：{stage1_ckpt}")
        # 这里建议使用 safe_load 或处理非 strict 加载（如果你改了模块名）
        state_dict = torch.load(stage1_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False) 
    else:
        accelerator.print("⚠️ 未发现一阶段权重，将从随机初始化开始二阶段（不推荐）")

    # ==================== 7. 构建数据流 ====================
    mixed_loader = create_mixed_dataloader(
        config_dict=config,
        special_tokens_ids=special_tokens_ids,
        text_tokenizer=tokenizer,
        accelerator=accelerator
    )

    # ==================== 8. 优化器与调度器 ====================
    param_groups = get_parameter_groups(model, config)
    
    # 🔍 关键修复：在 accelerator.prepare() 之前统计参数！
    if accelerator.is_main_process:
        print(f"\n📊 参数组分配统计 (准备前):")
        total_tensors = 0
        total_params = 0
        
        for i, group in enumerate(param_groups):
            n_tensors = len(group['params'])
            n_params = sum(p.numel() for p in group['params'])
            total_tensors += n_tensors
            total_params += n_params
            
            # 转换为更易读的单位（M = 百万）
            n_params_million = n_params / 1e6
            
            print(f"   Group {i}:")
            print(f"      - 张量数量：{n_tensors:,}")
            print(f"      - 参数量：{n_params:,} ({n_params_million:.2f}M)")
            print(f"      - 学习率：{group['lr']:.1e}")
        
        print(f"   {'─'*50}")
        print(f"   总计:")
        print(f"      - 张量总数：{total_tensors:,}")
        print(f"      - 参数总量：{total_params:,} ({total_params/1e9:.2f}B)")
        
        # 🔴 关键验证：确保有 3 个参数组且都不为空
        if len(param_groups) != 3:
            raise ValueError(f"❌ 参数组数量异常：期望 3 个，实际 {len(param_groups)} 个")
        
        # 检查每个参数组是否有实际参数
        group_names = ["Embedder", "Diffusion Head", "LLM"]
        for i, (group, name) in enumerate(zip(param_groups, group_names)):
            n_params = sum(p.numel() for p in group['params'])
            if n_params == 0:
                raise ValueError(f"❌ {name} 参数组为空！检查模块名是否正确")
        
        print(f"\n✅ 所有参数组都已正确加载！")
    
    optimizer = AdamW(param_groups)

    # ==================== 修复：正确计算训练步数 ====================
    # 使用 dataset 长度而不是 dataloader 长度，避免 ConcatDataset 在分布式环境下的长度异常
    batch_size = training_config.get('batch_size', 5)
    dataset_len = len(mixed_loader.dataset)  # 获取底层 dataset 的实际长度
    steps_per_epoch = dataset_len // batch_size
    total_steps = steps_per_epoch * config['training']['epochs'] // accelerator.gradient_accumulation_steps
    num_warmup_steps = int(total_steps * config['scheduler']['warmup_ratio'])

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    # ==================== 9. 指标监控 ====================
    logger = SwanLabLogger(config, accelerator)
    training_state = TrainingState()

    # 🔴 关键：先保存参数组的参数量统计（prepare 后会被分片）
    param_group_stats = []
    for i, group in enumerate(param_groups):
        n_params = sum(p.numel() for p in group['params'])
        param_group_stats.append(n_params)
    
    # 准备环境
    model, optimizer, mixed_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, mixed_loader, lr_scheduler
    )
    
    # 🔴 现在可以安全地打印 prepare 后的统计（虽然数字是分片的）
    if accelerator.is_main_process:
        print(f"\n📊 训练配置概览:")
        print(f"   - 每 GPU Batch Size: {batch_size}")
        print(f"   - 每 GPU 每 Epoch Steps: {steps_per_epoch:,}")
        print(f"   - Gradient Accumulation Steps: {accelerator.gradient_accumulation_steps}")
        print(f"   - Warmup Steps ({config['scheduler']['warmup_ratio']*100:.0f}%): {num_warmup_steps:,}")
        print(f"   - 预估总步数: {total_steps:,}")
        print(f"   - 总 Epochs: {config['training']['epochs']}")
        print(f"   - Mixed Precision: {accelerator.mixed_precision}")
        print(f"   - Distributed Type: {accelerator.distributed_type}")
        print(f"   - 模型总参数量: {sum(param_group_stats):,} ({sum(param_group_stats)/1e9:.2f}B)")
        print(f"   - ZeRO-2 已启用")
    
    # ==================== 10. 训练循环 ====================
    global_step = 0  # ✅ 关键修复：在使用前初始化！
    
    accelerator.print(f"🔥 Stage 2 训练启动！总步数: {total_steps:,} | Warmup步数: {num_warmup_steps:,} | Epochs: {config['training']['epochs']}")

    for epoch in range(config['training']['epochs']):
        model.train()
        for step, batch in enumerate(mixed_loader):
            with accelerator.accumulate(model):
                # 转发
                # 注意：labels 已经在 dataset 里处理好了，包含了 NTP 和 MTP 的混合标记
                logits, loss_ntp, loss_gene = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    position_ids=batch['position_ids'],
                    cell_features=batch['cell_features'],
                    cell_positions=batch['cell_positions'],
                    modality_positions=batch['modality_positions'],
                    labels=batch['labels'],
                    data_type=batch['data_type'] # 🔴 路由核心：传递任务类型
                )

                # Loss 融合：可以根据实验微调权重（建议 1:1 或 1:0.5）
                total_loss = loss_ntp + loss_gene

                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config['optimizer'].get('clip_grad_norm', 1.0))
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ==================== 11. 日志记录 ====================
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % config['logging']['log_interval'] == 0:
                    metrics = {
                        "total_loss": total_loss.item(),
                        "loss_ntp": loss_ntp.item(),
                        "loss_gene": loss_gene.item(),
                        "lr_llm": optimizer.param_groups[2]['lr'],
                        "lr_embedder": optimizer.param_groups[0]['lr'],
                        "lr_diffusion": optimizer.param_groups[1]['lr']  # ✅ 新增：Diffusion Head 学习率
                    }
                    logger.log(metrics, step=global_step)
                    
                    if accelerator.is_main_process:
                        print(f"Step {global_step} | Loss: {total_loss.item():.4f} (NTP: {loss_ntp.item():.4f}, Gene: {loss_gene.item():.4f})")
                        print(f"   Learning Rates -> LLM: {optimizer.param_groups[2]['lr']:.1e}, Embedder: {optimizer.param_groups[0]['lr']:.1e}, Diffusion: {optimizer.param_groups[1]['lr']:.1e}")

            # ==================== 12. 定期保存 ====================
            if global_step % config['logging']['save_interval'] == 0 and accelerator.sync_gradients:
                save_dir = Path(config['checkpoint']['save_dir']) / f"step_{global_step}"
                save_checkpoint(model, optimizer, lr_scheduler, global_step, config, output_dir=str(save_dir), accelerator=accelerator)

    accelerator.print("✅ Stage 2 训练圆满结束！")

if __name__ == "__main__":
    main()