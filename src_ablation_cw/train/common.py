# coding=utf-8
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_resume_path(resume_from: Optional[str]) -> Optional[str]:
    if not resume_from:
        return None
    p = Path(resume_from)
    if p.is_dir():
        state_pt = p / "state.pt"
        if state_pt.exists():
            return str(state_pt)
    if p.is_file():
        return str(p)
    return None


def save_state_pt(
    accelerator,
    model,
    optimizer,
    scheduler,
    global_step: int,
    save_dir: str,
    extra: Optional[Dict[str, Any]] = None,
):
    ckpt_dir = Path(save_dir) / f"checkpoint-step-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "global_step": int(global_step),
        "model": accelerator.get_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "extra": extra or {},
    }
    accelerator.save(state, str(ckpt_dir / "state.pt"))
    return str(ckpt_dir)


def load_state_pt(
    model,
    optimizer,
    scheduler,
    resume_from: str,
    map_location: str = "cpu",
    strict: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    state_path = resolve_resume_path(resume_from)
    if state_path is None:
        raise FileNotFoundError(f"resume path not found: {resume_from}")

    state = torch.load(state_path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state.get("model", {}), strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    extra = {
        "missing_keys": list(missing) if missing is not None else [],
        "unexpected_keys": list(unexpected) if unexpected is not None else [],
        "state_path": state_path,
        "extra": state.get("extra", {}),
    }
    return int(state.get("global_step", 0)), extra
