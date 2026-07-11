import logging
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from bepsy.config import ExperimentConfig
from bepsy.data.records import SubjectBatch
from bepsy.model.world import BEPSY
from bepsy.objectives import multi_objective_loss


@dataclass(frozen=True)
class EpochResult:
    loss: float
    batches: int
    examples: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_schedule(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class Trainer:
    def __init__(self, model: BEPSY, config: ExperimentConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.train.mixed_precision and device.type == "cuda")
        self.logger = logging.getLogger(__name__)

    def configure_schedule(self, batches: int) -> None:
        self.scheduler = cosine_schedule(
            self.optimizer, batches * self.config.train.epochs, self.config.train.warmup_fraction
        )

    def train_epoch(self, loader: DataLoader[SubjectBatch], epoch: int) -> EpochResult:
        self.model.train()
        total = 0.0
        examples = 0
        batches = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            enabled = self.config.train.mixed_precision and self.device.type == "cuda"
            with torch.autocast(device_type=self.device.type, enabled=enabled):
                output = self.model(batch.values, batch.times, batch.mask)
                losses = multi_objective_loss(
                    self.model,
                    output,
                    batch.values,
                    batch.mask,
                    batch.regimes,
                    batch.archives,
                    self.config.loss,
                    warm_start=epoch < self.config.train.warm_start_epochs,
                )
            self.scaler.scale(losses.total).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total += float(losses.total.detach()) * batch.values.shape[0]
            examples += batch.values.shape[0]
            batches += 1
        return EpochResult(total / max(1, examples), batches, examples)

    @torch.no_grad()
    def validate(self, loader: DataLoader[SubjectBatch]) -> EpochResult:
        self.model.eval()
        total = 0.0
        examples = 0
        batches = 0
        with torch.enable_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(batch.values, batch.times, batch.mask)
                losses = multi_objective_loss(
                    self.model,
                    output,
                    batch.values,
                    batch.mask,
                    batch.regimes,
                    batch.archives,
                    self.config.loss,
                )
                total += float(losses.total.detach()) * batch.values.shape[0]
                examples += batch.values.shape[0]
                batches += 1
        return EpochResult(total / max(1, examples), batches, examples)

