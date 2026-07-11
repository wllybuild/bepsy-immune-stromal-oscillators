from dataclasses import dataclass

import torch
from torch import Tensor, nn


class BifurcationClassifier(nn.Module):
    def __init__(self, compartments: int, width: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(compartments * 4, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Linear(width, classes),
        )

    def forward(self, radius: Tensor, phase: Tensor, lyapunov: Tensor, frequency: Tensor) -> Tensor:
        features = torch.cat(
            (radius, torch.sin(phase), lyapunov.expand_as(radius), frequency.expand_as(radius)), dim=-1
        )
        return self.network(features)


@dataclass(frozen=True)
class SparseEquation:
    coefficients: Tensor
    active: Tensor


class SparseEquationReadout(nn.Module):
    def __init__(self, compartments: int, threshold: float = 1e-3) -> None:
        super().__init__()
        self.compartments = compartments
        self.threshold = threshold
        self.coefficients = nn.Parameter(torch.zeros(compartments, 7, 2))

    def library(self, radius: Tensor, phase: Tensor) -> Tensor:
        return torch.stack(
            (
                torch.ones_like(radius),
                radius,
                radius.square(),
                radius.pow(3),
                torch.sin(phase),
                torch.cos(phase),
                radius * torch.sin(phase),
            ),
            dim=-1,
        )

    def forward(self, radius: Tensor, phase: Tensor) -> Tensor:
        return torch.einsum("...nl,nlo->...no", self.library(radius, phase), self.coefficients)

    def prune(self) -> SparseEquation:
        active = self.coefficients.abs() >= self.threshold
        return SparseEquation(self.coefficients * active, active)


class TerminalArchiveDiscriminator(nn.Module):
    def __init__(self, state_size: int, archives: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, state_size),
            nn.SiLU(),
            nn.Linear(state_size, archives),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state)

