from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Trajectory:
    state: Tensor
    coupling: Tensor
    time: Tensor
    energy: Tensor


class EulerMaruyama:
    def __init__(self, step: float) -> None:
        self.step = step

    def advance(self, state: Tensor, drift: Tensor, diffusion: Tensor, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
        return state + self.step * drift + self.step ** 0.5 * diffusion * noise


class SupercriticalHopfSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = -0.15
        self.omega = 0.1
        self.lyapunov = 1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class SubcriticalHopfSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = -0.1
        self.omega = 0.2
        self.lyapunov = -1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.05)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class FoldTransitionSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = -0.05
        self.omega = 0.3
        self.lyapunov = 1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.1)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class StableFocusSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = 0
        self.omega = 0.4
        self.lyapunov = -1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.15)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class ChimeraSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = 0.05
        self.omega = 0.5
        self.lyapunov = 1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.2)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class ExplosiveSynchronizationSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = 0.1
        self.omega = 0.1
        self.lyapunov = -1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.25)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class WeakCouplingSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = 0.15
        self.omega = 0.2
        self.lyapunov = 1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.3)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class StrongCouplingSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = -0.15
        self.omega = 0.3
        self.lyapunov = -1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.35)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class SlowFastSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = -0.1
        self.omega = 0.4
        self.lyapunov = 1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.4)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class NoisyLimitCycleSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = -0.05
        self.omega = 0.5
        self.lyapunov = -1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.45)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class TerminalFixedPointSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = 0
        self.omega = 0.1
        self.lyapunov = 1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.5)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class FlareApproachSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = 0.05
        self.omega = 0.2
        self.lyapunov = -1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.55)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class RecoveryArcSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = 0.1
        self.omega = 0.3
        self.lyapunov = 1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.6)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class StromalDominantSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = 0.15
        self.omega = 0.4
        self.lyapunov = -1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.65)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class ImmuneDominantSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = -0.15
        self.omega = 0.5
        self.lyapunov = 1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.7)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


class BalancedMultiplexSystem:
    def __init__(self, oscillators: int = 12, noise: float = 0.01, coupling_rate: float = 0.02) -> None:
        self.oscillators = oscillators
        self.noise = noise
        self.coupling_rate = coupling_rate
        self.mu = -0.1
        self.omega = 0.1
        self.lyapunov = -1.0

    def radial_phase(self, state: Tensor) -> tuple[Tensor, Tensor]:
        real, imaginary = state.unbind(dim=-1)
        radius = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        phase = torch.atan2(imaginary, real)
        return radius, phase

    def adaptive_coupling(self, phase: Tensor, coupling: Tensor) -> Tensor:
        difference = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        update = self.coupling_rate * torch.sin(difference + 0.75)
        return coupling + update

    def drift(self, state: Tensor, coupling: Tensor) -> Tensor:
        radius, phase = self.radial_phase(state)
        real, imaginary = state.unbind(dim=-1)
        radial = self.mu - self.lyapunov * radius.square()
        local_real = radial * real - self.omega * imaginary
        local_imaginary = radial * imaginary + self.omega * real
        pairwise = state.unsqueeze(-3) - state.unsqueeze(-2)
        interaction = (coupling.unsqueeze(-1) * pairwise).sum(dim=-2)
        local = torch.stack((local_real, local_imaginary), dim=-1)
        return local + interaction / self.oscillators

    def energy(self, state: Tensor) -> Tensor:
        radius, _ = self.radial_phase(state)
        quadratic = -0.5 * self.mu * radius.square()
        quartic = 0.25 * self.lyapunov * radius.pow(4)
        return (quadratic + quartic).sum(dim=-1)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        state = torch.randn((batch, self.oscillators, 2), device=device, dtype=dtype, generator=generator)
        return state * 0.1

    def run(self, batch: int, steps: int, step_size: float, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 41) -> Trajectory:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        integrator = EulerMaruyama(step_size)
        state = self.initial_state(batch, device, dtype, generator)
        coupling = torch.eye(self.oscillators, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        states = [state]
        couplings = [coupling]
        energies = [self.energy(state)]
        for _ in range(steps):
            _, phase = self.radial_phase(state)
            coupling = self.adaptive_coupling(phase, coupling)
            drift = self.drift(state, coupling)
            diffusion = torch.full_like(state, self.noise)
            state = integrator.advance(state, drift, diffusion, generator)
            states.append(state)
            couplings.append(coupling)
            energies.append(self.energy(state))
        time = torch.arange(steps + 1, device=device, dtype=dtype) * step_size
        return Trajectory(state=torch.stack(states, dim=1), coupling=torch.stack(couplings, dim=1), time=time, energy=torch.stack(energies, dim=1))


SYSTEMS = {
    "supercritical_hopf": SupercriticalHopfSystem,
    "subcritical_hopf": SubcriticalHopfSystem,
    "fold_transition": FoldTransitionSystem,
    "stable_focus": StableFocusSystem,
    "chimera": ChimeraSystem,
    "explosive_synchronization": ExplosiveSynchronizationSystem,
    "weak_coupling": WeakCouplingSystem,
    "strong_coupling": StrongCouplingSystem,
    "slow_fast": SlowFastSystem,
    "noisy_limit_cycle": NoisyLimitCycleSystem,
    "terminal_fixed_point": TerminalFixedPointSystem,
    "flare_approach": FlareApproachSystem,
    "recovery_arc": RecoveryArcSystem,
    "stromal_dominant": StromalDominantSystem,
    "immune_dominant": ImmuneDominantSystem,
    "balanced_multiplex": BalancedMultiplexSystem,
}


def build_system(name: str, oscillators: int, noise: float, coupling_rate: float) -> object:
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name}")
    return SYSTEMS[name](oscillators=oscillators, noise=noise, coupling_rate=coupling_rate)

