import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def random_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    step: int,
    seed: int,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "seed": seed,
        "random_state": random_state(),
    }
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    restore_random_state(payload["random_state"])
    return payload

