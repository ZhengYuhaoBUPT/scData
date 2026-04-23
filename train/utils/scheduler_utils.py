"""
Scheduler Utilities

统一的学习率调度器工具，支持：
- linear: 线性 warmup + 线性 decay
- linear_warmup_cosine_decay: 线性 warmup + cosine decay
"""

import math
from typing import Dict, Any, Tuple


SUPPORTED_SCHEDULER_TYPES = {"linear", "linear_warmup_cosine_decay"}


class ConfigurableLRScheduler:
    """
    轻量级 scheduler，不依赖 optimizer 必须继承 torch.optim.Optimizer。

    只要求 optimizer 暴露：
    - param_groups
    - state_dict()
    - load_state_dict()
    """

    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = -1

    def step(self):
        self.last_epoch += 1
        factor = self.lr_lambda(self.last_epoch)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.base_lrs = list(state_dict["base_lrs"])
        self.last_epoch = int(state_dict["last_epoch"])
        factor = self.lr_lambda(self.last_epoch)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor


def resolve_scheduler_config(scheduler_config: Dict[str, Any], computed_total_steps: int) -> Dict[str, Any]:
    """
    解析 scheduler 配置，处理优先级和参数校验。
    """
    scheduler_type = scheduler_config.get("type", "linear")
    if scheduler_type not in SUPPORTED_SCHEDULER_TYPES:
        raise ValueError(
            f"Unsupported scheduler type: {scheduler_type}. "
            f"Supported types: {SUPPORTED_SCHEDULER_TYPES}"
        )

    num_training_steps = scheduler_config.get("num_training_steps")
    if num_training_steps is not None:
        num_training_steps = int(num_training_steps)
    else:
        num_training_steps = computed_total_steps

    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be > 0, got {num_training_steps}")

    num_warmup_steps = scheduler_config.get("num_warmup_steps")
    if num_warmup_steps is not None:
        num_warmup_steps = int(num_warmup_steps)
    else:
        warmup_ratio = scheduler_config.get("warmup_ratio", 0.0)
        num_warmup_steps = int(num_training_steps * warmup_ratio)

    if not (0 <= num_warmup_steps <= num_training_steps):
        raise ValueError(
            f"num_warmup_steps must be in [0, num_training_steps], "
            f"got {num_warmup_steps} (num_training_steps={num_training_steps})"
        )

    min_lr_ratio = scheduler_config.get("min_lr_ratio", 0.1)
    if scheduler_type == "linear_warmup_cosine_decay":
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    else:
        min_lr_ratio = 0.0

    return {
        "type": scheduler_type,
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
        "min_lr_ratio": min_lr_ratio,
    }


def build_scheduler(
    optimizer,
    scheduler_config: Dict[str, Any],
    computed_total_steps: int,
) -> Tuple[ConfigurableLRScheduler, Dict[str, Any]]:
    """
    构建学习率调度器。
    """
    info = resolve_scheduler_config(scheduler_config, computed_total_steps)
    scheduler_type = info["type"]
    num_training_steps = info["num_training_steps"]
    num_warmup_steps = info["num_warmup_steps"]
    min_lr_ratio = info["min_lr_ratio"]

    if scheduler_type == "linear":
        lr_lambda = _get_linear_lambda(num_warmup_steps, num_training_steps)
    elif scheduler_type == "linear_warmup_cosine_decay":
        lr_lambda = _get_cosine_lambda(num_warmup_steps, num_training_steps, min_lr_ratio)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    scheduler = ConfigurableLRScheduler(optimizer, lr_lambda)
    return scheduler, info


def _get_linear_lambda(num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 1.0 - progress)

    return lr_lambda


def _get_cosine_lambda(num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float):
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda
