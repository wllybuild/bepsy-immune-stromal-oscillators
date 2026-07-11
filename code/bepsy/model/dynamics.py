from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DynamicsTerms:
    derivative: Tensor
    conservative: Tensor
    dissipative: Tensor
    coupling: Tensor
    energy: Tensor
    potential: Tensor


class HamiltonianEnergy(nn.Module):
    def __init__(self, state_size: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )
        self.quadratic = nn.Parameter(torch.ones(state_size))

    def forward(self, state: Tensor) -> Tensor:
        learned = self.network(state)
        anchored = 0.5 * (self.quadratic.abs() * state.square()).sum(dim=-1, keepdim=True)
        return learned + anchored


class DissipationPotential(nn.Module):
    def __init__(self, state_size: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, width),
            nn.Softplus(),
            nn.Linear(width, width),
            nn.Softplus(),
            nn.Linear(width, 1),
            nn.Softplus(),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state)


class AdaptiveCoupling(nn.Module):
    def __init__(self, compartments: int, context_size: int) -> None:
        super().__init__()
        self.compartments = compartments
        self.base = nn.Parameter(torch.zeros(compartments, compartments))
        self.context = nn.Sequential(
            nn.Linear(context_size, compartments * compartments),
            nn.Tanh(),
        )
        self.epsilon_one = nn.Parameter(torch.tensor(0.01))
        self.epsilon_two = nn.Parameter(torch.tensor(0.0))

    def matrix(self, context: Tensor) -> Tensor:
        adaptive = self.context(context).reshape(*context.shape[:-1], self.compartments, self.compartments)
        matrix = torch.nn.functional.softplus(self.base + adaptive)
        identity = torch.eye(self.compartments, device=matrix.device, dtype=matrix.dtype)
        return matrix * (1 - identity)

    def forward(self, radius: Tensor, phase: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        matrix = self.matrix(context)
        phase_delta = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        radius_delta = radius.unsqueeze(-2) - radius.unsqueeze(-1)
        phase_force = (matrix * torch.sin(-phase_delta)).sum(dim=-1)
        radius_force = (matrix * radius_delta).sum(dim=-1)
        derivative = self.epsilon_one * torch.sin(phase_delta + self.epsilon_two)
        force = torch.stack((radius_force, phase_force), dim=-1)
        return force, derivative


class StuartLandauField(nn.Module):
    def __init__(self, compartments: int) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(compartments))
        self.omega = nn.Parameter(torch.linspace(0.2, 1.0, compartments))
        self.lyapunov = nn.Parameter(torch.ones(compartments))
        self.gamma = nn.Parameter(torch.zeros(compartments))

    def forward(self, radius: Tensor) -> Tensor:
        radial = self.mu * radius - self.lyapunov * radius.pow(3)
        angular = self.omega + self.gamma * radius.square()
        return torch.stack((radial, angular), dim=-1)


class PseudoHamiltonianGenerator(nn.Module):
    def __init__(self, compartments: int, context_size: int, width: int) -> None:
        super().__init__()
        self.compartments = compartments
        self.state_size = compartments * 2
        self.energy = HamiltonianEnergy(self.state_size, width)
        self.potential = DissipationPotential(self.state_size, width)
        self.coupling = AdaptiveCoupling(compartments, context_size)
        self.normal_form = StuartLandauField(compartments)

    def symplectic(self, gradient: Tensor) -> Tensor:
        paired = gradient.reshape(*gradient.shape[:-1], self.compartments, 2)
        rotated = torch.stack((paired[..., 1], -paired[..., 0]), dim=-1)
        return rotated.flatten(start_dim=-2)

    def forward(self, state: Tensor, context: Tensor, create_graph: bool = True) -> DynamicsTerms:
        state = state.requires_grad_(True)
        energy = self.energy(state)
        potential = self.potential(state)
        energy_gradient = torch.autograd.grad(energy.sum(), state, create_graph=create_graph)[0]
        potential_gradient = torch.autograd.grad(potential.sum(), state, create_graph=create_graph)[0]
        conservative = self.symplectic(energy_gradient)
        norm = potential_gradient.norm(dim=-1, keepdim=True)
        dissipative = -(norm.square() * potential_gradient) / (norm + 1e-6)
        structured = state.reshape(*state.shape[:-1], self.compartments, 2)
        forces, _ = self.coupling(structured[..., 0], structured[..., 1], context)
        normal = self.normal_form(structured[..., 0])
        coupling = forces.flatten(start_dim=-2)
        derivative = conservative + dissipative + coupling + normal.flatten(start_dim=-2)
        return DynamicsTerms(derivative, conservative, dissipative, coupling, energy, potential)

    def step(self, state: Tensor, context: Tensor, delta: Tensor) -> Tensor:
        first = self.forward(state, context).derivative
        midpoint = state + 0.5 * delta[..., None] * first
        second = self.forward(midpoint, context).derivative
        return state + delta[..., None] * second


def complex_order_parameter(radius: Tensor, phase: Tensor) -> tuple[Tensor, Tensor]:
    real = (radius * torch.cos(phase)).mean(dim=-1)
    imaginary = (radius * torch.sin(phase)).mean(dim=-1)
    magnitude = torch.sqrt(real.square() + imaginary.square())
    angle = torch.atan2(imaginary, real)
    return magnitude, angle
