from dataclasses import dataclass

import torch
from torch import Tensor

from bepsy.config import LossConfig
from bepsy.model.world import BEPSY, BEPSYOutput


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    data: Tensor
    hamiltonian: Tensor
    radial: Tensor
    sparse_coupling: Tensor
    terminal_consistency: Tensor
    bifurcation: Tensor


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    squared = (prediction - target).square() * mask
    return squared.sum() / mask.sum().clamp_min(1)


def multi_objective_loss(
    model: BEPSY,
    output: BEPSYOutput,
    values: Tensor,
    mask: Tensor,
    regimes: Tensor,
    archives: Tensor,
    weights: LossConfig,
    warm_start: bool = False,
) -> LossBreakdown:
    data = masked_mse(output.reconstruction, values, mask)
    hamiltonian = output.energy.square().mean() + output.potential.mean()
    radius = output.latent[..., 0]
    derivative = radius[:, 1:] - radius[:, :-1]
    normal = model.generator.normal_form(radius[:, :-1])[..., 0]
    radial = (derivative - normal).square().mean()
    sparse_coupling = model.generator.coupling.base.abs().mean()
    terminal_consistency = torch.nn.functional.cross_entropy(output.archive_logits, archives)
    bifurcation = torch.nn.functional.cross_entropy(output.regime_logits, regimes)
    total = (
        weights.data * data
        + weights.hamiltonian * hamiltonian
        + weights.radial * radial
        + weights.sparse_coupling * sparse_coupling
        + weights.terminal_consistency * terminal_consistency
        + weights.bifurcation * bifurcation
    )
    if warm_start:
        total = weights.hamiltonian * hamiltonian + weights.radial * radial
    return LossBreakdown(total, data, hamiltonian, radial, sparse_coupling, terminal_consistency, bifurcation)

