#!/usr/bin/env python3
"""
🚀 Stage 1: Cell Embedder + Diffusion Head 训练脚本（Accelerate + DeepSpeed 加速版）

训练目标：
1. 训练 Cell Embedder 将 768 维细胞特征映射到 LLM 语义空间
2. 训练 Diffusion Head 预测被 mask 的基因 ID
3. 同时监控 loss_ntp 和 loss_gene，确保对齐质量

冻结策略：
- LLM (Qwen2.5) 完全冻结
- 只训练 Cell Embedder 和 Diffusion Head

加速特性：
- ✅ 支持 DeepSpeed ZeRO-2 优化器分片
- ✅ BF16 混合精度训练
- ✅ 多 GPU 数据并行
- ✅ 梯度累积自动管理
"""

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, get_scheduler

# 添加项目路径
project_root = Path("/root/wanghaoran/zxy/project/sc_showo")
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

# ✅ 修复：明确导入本地 datasets 模块的子模块
from src.datasets.gene_dataset import GeneDataset, create_gene_dataloader, calculate_training_steps
from src.models.modeling_gene_transformer import GeneTransformer
from src.train.utils.utils import SwanLabLogger, save_checkpoint, TrainingState


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer_params(model, config):
    """
    获取优化器参数（排除 LayerNorm 和 Bias 的权重衰减）
    """
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config['optimizer']['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    return optimizer_grouped_parameters


def main():
    # ==================== 1. 加载配置 ====================
    config_path = project_root / "config" / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # ==================== 2. 初始化 Accelerator ====================
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        mixed_precision=config['training'].get('mixed_precision', 'bf16'),
        log_with=None,  
        project_dir=config['logging']['swanlab_dir'],
        # 🚨 致命修正：禁止主进程分发！让 7 张卡各自独立执行 DataLoader！
        dispatch_batches=False,  
        split_batches=False      
    )
    
    # ✅ 添加配置验证（移到 accelerator 初始化之后）
    if accelerator.is_main_process:
        print("\n📋 配置验证:")
        print(f"   - config.json log_interval: {config['logging'].get('log_interval', '未设置')}")
        print(f"   - config.json save_interval: {config['logging'].get('save_interval', '未设置')}")
        print(f"   - optimizer lr: {config['optimizer'].get('lr', '未设置')}")
        print(f"   - optimizer weight_decay: {config['optimizer'].get('weight_decay', '未设置')}")
        print(f"   - training batch_size: {config['training'].get('batch_size', '未设置')}")
        print(f"   - training epochs: {config['training'].get('epochs', '未设置')}")
    
    # ✅ 打印加速配置信息
    if accelerator.state.deepspeed_plugin is not None:
        if accelerator.is_main_process:
            print("="*80)
            print("🚀 DeepSpeed 加速配置")
            print("="*80)
            print(f"  - Zero Stage: {accelerator.state.deepspeed_plugin.zero_stage}")
            print(f"  - Gradient Accumulation Steps: {accelerator.state.deepspeed_plugin.gradient_accumulation_steps}")
            print(f"  - Mixed Precision: {accelerator.mixed_precision}")
            # ✅ 修复：从 ds_config 字典中获取 train_micro_batch_size_per_gpu
            ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
            micro_batch = ds_config.get('train_micro_batch_size_per_gpu', 'auto')
            
            # ✅ 关键修复：优先从 ds_config 读取，如果没有则从 config.json读取
            steps_per_print = ds_config.get('steps_per_print')
            if steps_per_print is None or steps_per_print == float('inf'):
                # DeepSpeed 没有设置或为 inf，降级到 config.json
                steps_per_print = config['training'].get('steps_per_print', 10)
            
            print(f"  - Train Micro Batch Size Per GPU: {micro_batch}")
            print(f"  - Steps Per Print: {steps_per_print}")
            print("="*80)
    
    # 设置种子
    set_seed(config['training']['seed'])
    
    # ==================== 3. 加载 Tokenizer ====================
    if accelerator.is_main_process:
        print("\n📥 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['llm_model_path'],
        trust_remote_code=True,
        use_fast=True,
        local_files_only=config['model'].get('local_files_only', True)
    )
    
    # ✅ 从 config 读取并添加 special tokens
    special_tokens_dict = config['tokenizer']['special_tokens']
    
    if accelerator.is_main_process:
        print(f"\n📋 Special Tokens Dict: {special_tokens_dict}")
    
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens_dict.values())})
    
    # ✅ 将 token 字符串转换为 IDs（GeneDataset 需要的是 IDs）
    # 注意：键名必须是 'sog_id', 'eog_id' 等格式
    special_tokens_ids = {}
    for name, token in special_tokens_dict.items():
        # name: "sog" -> key: "sog_id"
        token_id = tokenizer.convert_tokens_to_ids(token)
        special_tokens_ids[f"{name}_id"] = token_id
        
        if accelerator.is_main_process:
            print(f"   {name}: '{token}' -> ID: {token_id}")
    
    if accelerator.is_main_process:
        print(f"✅ Tokenizer 加载成功，词表大小：{len(tokenizer)}")
        print(f"   添加了 {num_added_tokens} 个特殊 tokens")
        print(f"   Special Tokens IDs: {special_tokens_ids}")
    
    # ==================== 4. 创建 Dataset ====================
    if accelerator.is_main_process:
        print("\n📊 创建 Dataset...")
    
    # ✅ 关键修改：传入 accelerator 参数，实现真正的分布式数据加载
    dataloader = create_gene_dataloader(
        config_dict=config,
        special_tokens_ids=special_tokens_ids,
        text_tokenizer=tokenizer,
        accelerator=accelerator,  # ✅ 新增：让 Dataset 根据 rank 切分数据
    )
    
    # ✅ 从 DataLoader 中获取真实的 Dataset（用于计算训练步数）
    dataset = dataloader.dataset
    
    # 🚨 致命修复：恢复全局样本数概念！
    # 因为现在的 dataset 是 Sharded 模式（只包含当前 GPU 的 1/N 数据），
    # 但 calculate_training_steps 内部还会除以 GPU 数量。
    # 所以必须把单卡样本数乘回 num_processes，还原出真实的 Global Total Samples！
    if hasattr(dataset, 'total_cells'):
        # ✅ 关键：单卡样本数 × GPU 数量 = 全局样本数
        estimated_samples = dataset.total_cells * accelerator.num_processes
        if accelerator.is_main_process:
            print(f"📊 ShardedDataset - 单卡样本：{dataset.total_cells:,}")
            print(f"   → 还原全局估算：{estimated_samples:,} 个细胞 (×{accelerator.num_processes})")
    else:
        # Fallback
        estimated_samples = len(dataset) * accelerator.num_processes
        if accelerator.is_main_process:
            print(f"📊 ShardedDataset - 单卡样本：{len(dataset):,}")
            print(f"   → 还原全局估算：{estimated_samples:,} 个细胞 (×{accelerator.num_processes})")
    
    training_info = calculate_training_steps(estimated_samples, config, accelerator)
    total_steps = training_info['total_steps']
    warmup_steps = training_info['warmup_steps']
    
    if accelerator.is_main_process:
        print(f"\n🔥 实际训练配置:")
        print(f"   - 全局样本数：{estimated_samples:,}")
        print(f"   - 单卡 Batch Size: {config['training']['batch_size']}")
        print(f"   - GPU 数量：{accelerator.num_processes}")
        print(f"   - Gradient Accumulation: {accelerator.gradient_accumulation_steps}")
        print(f"   - 有效 Batch Size: {config['training']['batch_size'] * accelerator.num_processes * accelerator.gradient_accumulation_steps}")
        print(f"   - 每 epoch 步数：{training_info['steps_per_epoch']:,}")
        print(f"📈 总训练步数：{total_steps:,} ({training_info['steps_per_epoch']:,} steps/epoch × {config['training']['epochs']} epochs)")
        print(f"🔥 Warmup 步数：{warmup_steps:,} ({warmup_steps/total_steps*100:.2f}%)")
    
    # ==================== 5. 创建 Model ====================
    if accelerator.is_main_process:
        print("\n🏗️  创建 Model...")
    
    # ✅ 修复：显式关闭 use_cache 以适配梯度检查点
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(config['model']['llm_model_path'])
    model_config.use_cache = False
    
    model = GeneTransformer(
        llm_vocab_size=len(tokenizer),
        llm_model_path=config['model']['llm_model_path'],
        load_from_showo=config['model']['load_from_showo'],
        config_dict=config,
        special_tokens_ids=special_tokens_ids,
    )
    
    # ❄️ 冻结策略：Stage 1 只训练 Embedder 和 Diffusion Head
    if accelerator.is_main_process:
        print("\n❄️  冻结 LLM 参数（Stage 1 只训练 Embedder + Diffusion Head）...")
        print("   可训练模块:")
        print("     - cell_embedder (细胞特征投影)")
        print("     - gene_embedder (基因词表嵌入)")
        print("     - diffusion_head_a (扩散注意力层)")
        print("     - diffusion_head_b (基因预测头)")
        print("   冻结模块:")
        print("     - showo (Qwen2.5-7B 骨干网络)")
    
    # ✅ 致命修复：完全冻结 LLM（showo 的所有参数）
    for name, param in model.showo.named_parameters():
        param.requires_grad = False
    
    # ✅ 显式设置需要训练的模块（确保这些模块的 requires_grad=True）
    for name, param in model.cell_embedder.named_parameters():
        param.requires_grad = True
    
    for name, param in model.gene_embedder.named_parameters():
        param.requires_grad = True
    
    for name, param in model.diffusion_head_a.named_parameters():
        param.requires_grad = True
    
    for name, param in model.diffusion_head_b.named_parameters():
        param.requires_grad = True
    
    # ✅ 关键修复：ZeRO-2 不需要 Meta Tensor，可以立即检查参数状态！
    if accelerator.is_main_process:
        print(f"\n✅ Model 创建成功")
        print(f"\n🔍 检查参数冻结状态...")
        
        frozen_params = 0
        trainable_params = 0
        
        # 统计冻结参数（LLM）
        for name, param in model.showo.named_parameters():
            if not param.requires_grad:
                frozen_params += param.numel()
        
        # 统计可训练参数（Embedder + Diffusion Head）
        for module_name, module in [
            ('cell_embedder', model.cell_embedder),
            ('gene_embedder', model.gene_embedder),
            ('diffusion_head_a', model.diffusion_head_a),
            ('diffusion_head_b', model.diffusion_head_b),
        ]:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    trainable_params += param.numel()
        
        print(f"   ❄️  冻结参数 (LLM): {frozen_params:,} ({frozen_params/1e9:.2f}B)")
        print(f"   🔥 可训练参数：{trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"   📊 可训练比例：{trainable_params / (frozen_params + trainable_params) * 100:.2f}%")
        
        # 📊 ZeRO-2 显存预估（7 GPUs, BF16 混合精度）：
        #   - Qwen2.5-7B 参数 (BF16): ~14GB
        #   - ZeRO-2 梯度分片：~2GB
        #   - ZeRO-2 优化器状态分片 (AdamW): ~12GB
        #   - 静态总占用：~28GB
        #   - 剩余显存：80GB - 28GB = 52GB（足够承载激活值）
        #   → 可尝试 batch_size=4 或更高
        print(f"\n💡 ZeRO-2 优势：相比 ZeRO-3，训练速度提升 2-3 倍，显存占用更低")
    
    # 等待所有进程完成预加载
    accelerator.wait_for_everyone()
    
    # ==================== 6. 优化器和调度器 ====================
    if accelerator.is_main_process:
        print("\n🔧 初始化优化器和调度器...")
    
    optimizer_params = get_optimizer_params(model, config)
    
    optimizer = AdamW(
        optimizer_params,
        lr=config['optimizer']['lr'],
        betas=tuple(config['optimizer']['betas']),
        eps=config['optimizer']['eps'],
    )
    
    # 线性预热 + 余弦衰减
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    if accelerator.is_main_process:
        print(f"✅ 优化器：AdamW (lr={config['optimizer']['lr']}, wd={config['optimizer']['weight_decay']})")
        print(f"✅ 调度器：Linear Warmup + Cosine Decay")
    
    # ==================== 7. 准备 Accelerator ====================
    if accelerator.is_main_process:
        print("\n🔄 准备 Accelerator (DeepSpeed ZeRO-2 将在此初始化真实权重)...")
    
    # ✅ 修复：同时准备 model 和 optimizer/scheduler（但不包括 dataloader！）
    # 🚨 致命修正：绝对不要把 dataloader 传给 prepare！
    # 防止 Accelerate 画蛇添足给你强行注入 DistributedSampler 导致数据二次切分


# 🚨 终极补丁：显式告知 DeepSpeed 微批次大小，彻底接管 Accelerate 底层配置
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config['training']['batch_size']
    
    # 依然不传入 dataloader，保护我们的原生切片
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    
    # ✅ dataloader 保持原样，因为它已经在 Dataset.__init__ 中按 rank 切好了
    
    # ==================== 9. 初始化 Logger ====================
    logger = SwanLabLogger(config, accelerator)
    training_state = TrainingState()
    
    # ==================== 10. 加载 Checkpoint（可选） ====================
    start_epoch = 0
    global_step = 0
    
    if config['checkpoint'].get('resume_from'):
        resume_path = config['checkpoint']['resume_from']
        if accelerator.is_main_process:
            print(f"\n📥 从 checkpoint 恢复：{resume_path}")
        
        global_step = load_checkpoint(
            model, optimizer, lr_scheduler,
            resume_path, accelerator
        )
        training_state.global_step = global_step
        start_epoch = global_step // training_info['steps_per_epoch']
        
        if accelerator.is_main_process:
            print(f"✅ 恢复到 Epoch {start_epoch}, Step {global_step}")
    
    # ==================== 11. 训练循环 ====================
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("🎯 开始训练...")
        print("="*80)
    
    model.train()
    
    # ==========================================
    # 🌟 终极护盾：点火前的强制集结号 (Global Barrier)
    # 确保所有卡的 120GB 内存特征完全载入后，同一微秒起跑
    # ==========================================
    if accelerator.is_main_process:
        print("\n⏳ 正在等待所有 GPU 节点完成数据预加载...")
    
    accelerator.wait_for_everyone()  # 🚨 强制同步！所有人到齐后才准放行
    
    if accelerator.is_main_process:
        print("✅ 全军集结完毕！正式点火，进入第一轮 Forward！\n")
    # ==========================================

    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start_time = time.time()
        
        # ✅ 新增：维护上一个日志打点的时间和步数（用于准确计算吞吐率）
        if epoch == start_epoch and global_step == 0:
            last_log_time = time.time()
            last_log_step = global_step
        
        epoch_loss = 0.0
        epoch_loss_ntp = 0.0
        epoch_loss_gene = 0.0
        n_batches = 0
        
        # ✅ 原生 PyTorch 迭代（每个进程遍历自己已被静态切分的全部数据）
        for step, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            # 🚨 手动将 Tensor 移动到当前 GPU (因为绕过了 accelerate.prepare)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(accelerator.device)
            
# 🚨 调试探针：每个进程打印前 5 个样本的输入内容
            if accelerator.is_local_main_process and step < 5:
                print(f"\n{'='*80}")
                print(f"🔍 [Rank {accelerator.process_index}] Step {step} - 检查输入数据 (双语解码版)")
                print(f"{'='*80}")
                
                # ✅ 1. 建立基因词表反向映射字典
                id_to_gene = {v: k for k, v in dataset.vocab.items()}
                
                # ✅ 2. 混合解码器：优先查基因词表，查不到（如 mask_gene_id=151667）则去查 LLM 词表
                def decode_gene(idx):
                    return id_to_gene.get(idx, tokenizer.convert_ids_to_tokens(idx))

                # 打印文本 tokens（前 20 个）
                input_ids = batch['input_ids'][0]  # 取第一个样本
                text_len = batch['cell_positions'][0, 0].item()  # 文本区域长度
                
                print(f"\n📝 文本区域 (前 20 tokens):")
                print(f"   Input IDs:   {input_ids[:min(20, text_len)].tolist()}")
                print(f"   Text Tokens: {tokenizer.convert_ids_to_tokens(input_ids[:min(20, text_len)].tolist())}")
                
                # 打印基因 tokens（前 30 个）
                sog_idx = batch['modality_positions'][0, 0, 0].item()
                gene_ids = input_ids[sog_idx+1:sog_idx+31]  # 跳过 SOG token
                
                print(f"\n🧬 基因区域 (前 30 tokens):")
                print(f"   Gene IDs:    {gene_ids.tolist()}")
                print(f"   Gene Tokens: {[decode_gene(idx) for idx in gene_ids.tolist()]}")
                
                # 打印标签（前 30 个基因位置）
                labels = batch['labels'][0, sog_idx+1:sog_idx+31]
                print(f"\n🎯 基因 Labels (前 30 个位置):")
                print(f"   Labels:        {labels.tolist()}")
                # ✅ 极其硬核：直接把真实的 Target Label 也反编译成基因名给你看！
                print(f"   Target Tokens: {[decode_gene(idx) if idx != -100 else 'IGNORE' for idx in labels.tolist()]}")
                print(f"   (其中 -100 / IGNORE 表示不计算 Loss)")
                
                # 统计全局（1200 个基因）的有效 Loss 位置
                global_gene_mask = batch['gene_mask'][0]
                global_non_zero = batch['non_zero_mask'][0]
                
                valid_for_loss_global = (batch['labels'][0, sog_idx+1:] != -100).sum().item()
                masked_and_nonzero_global = (global_gene_mask & global_non_zero).sum().item()
                print(f"\n📊 全局统计:")
                print(f"   总基因数：{global_gene_mask.shape[0]}")
                print(f"   全局被 Mask 的基因数：{global_gene_mask.sum().item()}")
                print(f"   全局非零表达基因数：{global_non_zero.sum().item()}")
                print(f"   全局既被 Mask 又非零（计算 Loss）：{masked_and_nonzero_global}")
                
                # 统计前 30 个基因的局部信息
                gene_mask_local = batch['gene_mask'][0][:30]
                non_zero_mask_local = batch['non_zero_mask'][0][:30]
                valid_for_loss_local = (labels != -100).sum().item()
                masked_and_nonzero_local = (gene_mask_local & non_zero_mask_local).sum().item()
                print(f"\n📊 局部统计 (前 30 个基因):")
                print(f"   被 Mask 的基因数：{gene_mask_local.sum().item()}")
                print(f"   非零表达的基因数：{non_zero_mask_local.sum().item()}")
                print(f"   既被 Mask 又非零（计算 Loss）：{masked_and_nonzero_local}")
                print(f"   Labels 中有效位置（!= -100）: {valid_for_loss_local}")
                
                print(f"\n{'='*80}\n")
            
            # 👇 下面紧接着原本的 with accelerator.accumulate(model): ...
            with accelerator.accumulate(model):
                # 1. Forward pass (绝对不要在这里加任何 dist.barrier!)
                logits, loss_ntp, loss_gene = model(
                    input_ids=batch['input_ids'],
                    position_ids=batch['position_ids'],
                    attention_mask=batch.get('attention_mask'),
                    cell_features=batch['cell_features'],
                    cell_positions=batch['cell_positions'],
                    modality_positions=batch['modality_positions'],
                    gene_mask=batch['gene_mask'],
                    labels=batch['labels'],
                )
                
                # ✅ 损失加权计算
                total_loss = (
                    config['training'].get('lambda_ntp', 1.0) * loss_ntp + 
                    config['training'].get('lambda_gene', 1.0) * loss_gene
                )
                
                # 2. Backward pass
                accelerator.backward(total_loss)
                
                # 3. 梯度裁剪 (依赖 accelerate 自动处理上下文)
                if accelerator.sync_gradients and config['optimizer'].get('clip_grad_norm') is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad_norm'])
                
                # 4. Optimizer & Scheduler Step + 🚨 致命缺漏：Zero Grad!
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 更新单 batch 的 Loss (用于 epoch 统计，必须用 .item() 防内存泄漏)
            epoch_loss += total_loss.item()
            epoch_loss_ntp += loss_ntp.item()
            epoch_loss_gene += loss_gene.item()
            n_batches += 1
            
            # ==========================================
            # 📊 全局日志与 Checkpoint 触发逻辑
            # ==========================================
            # 只有在真正进行了梯度更新（sync_gradients）后，才执行 step ++
            if accelerator.sync_gradients:
                global_step += 1
                
                # 📝 1. 打印日志
                if global_step % config['logging'].get('log_interval', 10) == 0:
                    # ✅ 关键修复：统计真实的时间窗口和样本处理量
                    current_time = time.time()
                    time_elapsed = current_time - last_log_time
                    
                    # 真实走过的全局步数（Global Step）
                    steps_passed = global_step - last_log_step
                    
                    # 真实消化掉的总样本数（考虑梯度累积和多卡）
                    total_samples_processed = (
                        steps_passed * 
                        config['training']['batch_size'] * 
                        accelerator.num_processes * 
                        accelerator.gradient_accumulation_steps
                    )
                    
                    # 真实的 Global Speed（samples/s）
                    samples_per_sec = total_samples_processed / time_elapsed if time_elapsed > 0 else 0
                    
                    # ✅ 更新打点
                    last_log_time = current_time
                    last_log_step = global_step
                    
                    # ✅ 使用 accelerate 原生的 reduce 规避 gather 维度崩溃问题
                    avg_loss = accelerator.reduce(total_loss, reduction="mean").item()
                    avg_ntp = accelerator.reduce(loss_ntp, reduction="mean").item()
                    avg_gene = accelerator.reduce(loss_gene, reduction="mean").item()
                    
                    if accelerator.is_main_process:
                        current_lr = lr_scheduler.get_last_lr()[0]
                        print(f"[Epoch {epoch+1}/{config['training']['epochs']}] "
                              f"Step {global_step}/{total_steps} | "
                              f"Loss: {avg_loss:.4f} (NTP: {avg_ntp:.4f}, Gene: {avg_gene:.4f}) | "
                              f"LR: {current_lr:.2e} | "
                              f"Speed: {samples_per_sec:.2f} samples/s")
                        
                        logger.log({
                            "train/total_loss": avg_loss,
                            "train/loss_ntp": avg_ntp,
                            "train/loss_gene": avg_gene,
                            "train/lr": current_lr,
                            "train/samples_per_sec": samples_per_sec,
                            "step": global_step,
                            "epoch": epoch,
                        }, step=global_step)

                # 💾 2. 保存 Checkpoint
                # 这里可以放一个 wait_for_everyone，因为它在 accumulate 外部，不会阻碍反向传播
                if global_step > 0 and global_step % config['checkpoint'].get('save_interval', 500) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_checkpoint(
                            model, optimizer, lr_scheduler, global_step, config,
                            output_dir=config['checkpoint']['save_dir'],
                            accelerator=accelerator,
                            save_total_limit=config['checkpoint'].get('save_total_limit', 3),
                        )
                        print(f"💾 Checkpoint saved at step {global_step}")
                
                # 🧪 3. 采样测试
                if global_step > 0 and global_step % config['callbacks'].get('sample_interval', 10000) == 0:
                    if accelerator.is_main_process:
                        print(f"🧪 Running sampling test at step {global_step}...")
                        # TODO: 实现 generate_samples 函数
        # Epoch 结束时的逻辑保持不变
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0
        avg_epoch_loss_ntp = epoch_loss_ntp / n_batches if n_batches > 0 else 0
        avg_epoch_loss_gene = epoch_loss_gene / n_batches if n_batches > 0 else 0
        
        if accelerator.is_main_process:
            print(f"\n{'='*80}")
            print(f"✅ Epoch {epoch+1}/{config['training']['epochs']} 完成")
            print(f"   平均 Loss: {avg_epoch_loss:.4f} (NTP: {avg_epoch_loss_ntp:.4f}, Gene: {avg_epoch_loss_gene:.4f})")
            print(f"   用时：{epoch_time/60:.2f} 分钟")
            print(f"{'='*80}\n")
        
        # 更新训练状态
        training_state.update(avg_epoch_loss, epoch, global_step)
    
    # ==================== 12. 保存最终 Checkpoint ====================
    if accelerator.is_main_process:
        final_ckpt_path = save_checkpoint(
            model, optimizer, lr_scheduler, global_step, config,
            output_dir=config['checkpoint']['save_dir'] + "/final",
            accelerator=accelerator,
        )
        print(f"\n🎉 训练完成！最终 checkpoint: {final_ckpt_path}")
    
    # 结束 Logger
    logger.finish()
    
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("🎊 Stage 1 训练完成！")
        print("="*80)


if __name__ == "__main__":
    main()