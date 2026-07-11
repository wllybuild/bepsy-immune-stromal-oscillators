from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    modalities: int
    compartments: int
    sequence_length: int
    workers: int


@dataclass(frozen=True)
class ModelConfig:
    latent_per_compartment: int
    slow_width: int
    fast_width: int
    gate_width: int
    attention_heads: int
    classes: int


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    epochs: int
    warm_start_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_fraction: float
    mixed_precision: bool
    sindy_interval: int


@dataclass(frozen=True)
class LossConfig:
    data: float
    hamiltonian: float
    radial: float
    sparse_coupling: float
    terminal_consistency: float
    bifurcation: float


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: int
    bootstrap_resamples: int
    rolling_window_days: int


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    loss: LossConfig
    evaluation: EvaluationConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)
    return ExperimentConfig(
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        loss=LossConfig(**raw["loss"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
    )

