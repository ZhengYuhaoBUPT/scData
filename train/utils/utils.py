"""
🛠️ Training Utilities: Logging & Checkpointing

功能：
1. SwanLab 日志记录
2. Checkpoint 保存/加载
3. 训练状态跟踪
"""

import os
import json
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def _filter_trainable_state_dict(unwrapped_model: torch.nn.Module, full_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, p in unwrapped_model.named_parameters() if p.requires_grad}
    trainable_prefixes = tuple(name + "." for name in trainable_names)

    out = {}
    for k, v in full_state_dict.items():
        if k in trainable_names:
            out[k] = v
            continue
        if k.startswith(trainable_prefixes):
            out[k] = v
    return out


def _maybe_get_state_dict(model: torch.nn.Module, accelerator: Optional[Any]) -> Dict[str, torch.Tensor]:
    if accelerator is not None:
        return accelerator.get_state_dict(model)
    return model.state_dict()


class SwanLabLogger:
    """
    📊 SwanLab 日志记录器
    
    封装 SwanLab API，提供统一的日志接口
    """
    
    def __init__(self, config: Dict[str, Any], accelerator=None):
        """
        Args:
            config: 完整配置字典
            accelerator: Accelerator 实例（用于分布式训练）
        """
        self.config = config
        self.accelerator = accelerator
        self.is_main_process = accelerator.is_main_process if accelerator else True
        
        if self.is_main_process:
            try:
                import swanlab
                
                # 初始化 SwanLab
                swanlab_dir = config['logging'].get('swanlab_dir', './logs/swanlab')
                os.makedirs(swanlab_dir, exist_ok=True)
                
                # 构建实验名称
                experiment_name = config['logging'].get('experiment_name')
                if not experiment_name:
                    from datetime import datetime
                    experiment_name = f"stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                swanlab.init(
                    project=config['logging'].get('project', 'sc_show_o'),
                    experiment_name=experiment_name,
                    description=config.get('comment', {}).get('description', 'SC-Show-O Training'),
                    logdir=swanlab_dir,
                    config={
                        'model': config['model'],
                        'dataset': config['dataset'],
                        'training': config['training'],
                        'optimizer': config['optimizer'],
                        'logging': config['logging'],
                    }
                )
                
                self.swanlab = swanlab
                print(f"✅ SwanLab 初始化成功：{config['logging'].get('project', 'sc_show_o')} (exp: {experiment_name})")
                
            except ImportError:
                print("⚠️  未安装 swanlab，请运行：pip install swanlab")
                self.swanlab = None
        else:
            self.swanlab = None
    
    def log(self, metrics: Dict[str, Any], step: int):
        """
        记录训练指标
        
        Args:
            metrics: 指标字典，如 {"train/loss": 0.5, "train/lr": 1e-4}
            step: 全局步数
        """
        if self.swanlab and self.is_main_process:
            sanitized = {}
            for k, v in metrics.items():
                if isinstance(v, str):
                    continue
                if torch.is_tensor(v):
                    if v.numel() == 1:
                        sanitized[k] = v.item()
                    else:
                        continue
                else:
                    sanitized[k] = v
            self.swanlab.log(sanitized, step=step)
    
    def finish(self):
        """结束实验"""
        if self.swanlab and self.is_main_process:
            self.swanlab.finish()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_step: int,
    config: Dict[str, Any],
    output_dir: str,
    accelerator: Optional[Any] = None,
    save_total_limit: int = 5,
) -> str:
    """
    💾 保存检查点
    """
    if accelerator is not None and not accelerator.is_main_process:
        return

    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
    else:
        unwrapped_model = model

    checkpoint_cfg = config.get('checkpoint', {})
    save_format = str(checkpoint_cfg.get('save_format', 'pytorch')).lower()
    if save_format != 'pytorch':
        raise ValueError(f"Only save_format='pytorch' is supported now, got: {save_format}")

    save_trainable_only = bool(checkpoint_cfg.get('save_trainable_only', True))
    save_optimizer_state = bool(checkpoint_cfg.get('save_optimizer_state', True))
    save_scheduler_state = bool(checkpoint_cfg.get('save_scheduler_state', True))

    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_path = checkpoint_dir / f"checkpoint-step-{global_step}"
    save_path.mkdir(parents=True, exist_ok=True)

    full_state = _maybe_get_state_dict(model, accelerator)
    if save_trainable_only:
        model_state = _filter_trainable_state_dict(unwrapped_model, full_state)
        model_state_type = 'trainable_only'
    else:
        model_state = full_state
        model_state_type = 'full'

    payload = {
        'model': model_state,
        'model_state_type': model_state_type,
        'global_step': global_step,
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }

    if save_optimizer_state:
        payload['optimizer'] = optimizer.state_dict()
    if save_scheduler_state:
        payload['scheduler'] = lr_scheduler.state_dict()

    torch.save(payload, save_path / 'state.pt')

    metadata = {
        'global_step': global_step,
        'model_state_type': model_state_type,
        'save_trainable_only': save_trainable_only,
        'save_optimizer_state': save_optimizer_state,
        'save_scheduler_state': save_scheduler_state,
        'config': {
            'batch_size': config['training']['batch_size'],
            'lr': config['optimizer']['lr'],
            'epochs': config['training']['epochs'],
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(save_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.name.startswith('checkpoint-step-')],
        key=lambda x: int(x.name.split('-')[-1])
    )

    while len(checkpoints) > save_total_limit:
        oldest = checkpoints.pop(0)
        shutil.rmtree(oldest)
        print(f"🗑️  删除旧 checkpoint: {oldest}")

    print(f"💾 Checkpoint 已保存：{save_path} (model_state={model_state_type}, keys={len(model_state)})")
    return str(save_path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
    accelerator: Optional[Any] = None,
) -> int:
    """
    📥 加载检查点
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在：{checkpoint_path}")

    state_file = checkpoint_path / "state.pt"
    if not state_file.exists():
        raise FileNotFoundError(f"状态文件不存在：{state_file}")

    state_dict = torch.load(state_file, map_location='cpu', weights_only=True)

    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        missing, unexpected = unwrapped_model.load_state_dict(state_dict['model'], strict=False)

        if accelerator.is_main_process:
            print(f"   ℹ️ 模型权重加载完成: missing={len(missing)}, unexpected={len(unexpected)}")

        if 'optimizer' in state_dict:
            try:
                optimizer.load_state_dict(state_dict['optimizer'])
                if accelerator.is_main_process:
                    print("   ✅ Optimizer 状态已恢复")
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"   ⚠️  Optimizer 恢复失败: {e}")

        if 'scheduler' in state_dict:
            try:
                lr_scheduler.load_state_dict(state_dict['scheduler'])
                if accelerator.is_main_process:
                    print("   ✅ Scheduler 状态已恢复")
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"   ⚠️  Scheduler 恢复失败: {e}")

        global_step = state_dict['global_step']
    else:
        missing, unexpected = model.load_state_dict(state_dict['model'], strict=False)
        if 'optimizer' in state_dict:
            optimizer.load_state_dict(state_dict['optimizer'])
        if 'scheduler' in state_dict:
            lr_scheduler.load_state_dict(state_dict['scheduler'])
        global_step = state_dict['global_step']
        print(f"   ℹ️ 模型权重加载完成: missing={len(missing)}, unexpected={len(unexpected)}")

    print(f"📥 Checkpoint 已加载：{checkpoint_path} (step={global_step})")
    return global_step


class TrainingState:
    """
    📋 训练状态跟踪器
    
    用于记录和恢复训练进度
    """
    
    def __init__(self):
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
    
    def update(self, loss: float, epoch: int, global_step: int):
        """更新状态"""
        self.global_step = global_step
        self.epoch = epoch
        self.training_history.append({
            'step': global_step,
            'loss': loss,
            'epoch': epoch,
        })
        
        if loss < self.best_loss:
            self.best_loss = loss
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history[-100:],  # 只保留最近 100 条
        }
    
    def from_dict(self, state_dict: Dict[str, Any]):
        """从字典加载"""
        self.global_step = state_dict['global_step']
        self.epoch = state_dict['epoch']
        self.best_loss = state_dict['best_loss']
        self.training_history = state_dict.get('training_history', [])