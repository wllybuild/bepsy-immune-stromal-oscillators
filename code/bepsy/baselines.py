from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BaselineOutput:
    forecast: Tensor
    state: Tensor
    auxiliary: Tensor


class SequenceBaseline(Protocol):
    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        ...


class TimeEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.scale = nn.Parameter(torch.ones(width))
        self.shift = nn.Parameter(torch.zeros(width))

    def forward(self, times: Tensor) -> Tensor:
        phase = times.unsqueeze(-1) * self.scale + self.shift
        half = self.width // 2
        return torch.cat((torch.sin(phase[..., :half]), torch.cos(phase[..., half:half * 2])), dim=-1)


class LinearAutoregressor(nn.Module):
    def __init__(self, features: int, width: int = 32, depth: int = 2) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.05 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class ResidualAutoregressor(nn.Module):
    def __init__(self, features: int, width: int = 48, depth: int = 3) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.1 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class GatedRecurrentBaseline(nn.Module):
    def __init__(self, features: int, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.15 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class DecayRecurrentBaseline(nn.Module):
    def __init__(self, features: int, width: int = 80, depth: int = 2) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.2 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class TemporalMixerBaseline(nn.Module):
    def __init__(self, features: int, width: int = 96, depth: int = 3) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.25 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class PatchMixerBaseline(nn.Module):
    def __init__(self, features: int, width: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.3 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class InvertedAttentionBaseline(nn.Module):
    def __init__(self, features: int, width: int = 48, depth: int = 2) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.35 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class StateSpaceBaseline(nn.Module):
    def __init__(self, features: int, width: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.05 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class SelectiveStateBaseline(nn.Module):
    def __init__(self, features: int, width: int = 80, depth: int = 4) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.1 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class EchoStateBaseline(nn.Module):
    def __init__(self, features: int, width: int = 96, depth: int = 2) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.15 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class NeuralODEBaseline(nn.Module):
    def __init__(self, features: int, width: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.2 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class NeuralCDEBaseline(nn.Module):
    def __init__(self, features: int, width: int = 48, depth: int = 4) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.25 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class HamiltonianBaseline(nn.Module):
    def __init__(self, features: int, width: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.3 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class PseudoHamiltonianBaseline(nn.Module):
    def __init__(self, features: int, width: int = 80, depth: int = 3) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.35 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class LagrangianBaseline(nn.Module):
    def __init__(self, features: int, width: int = 96, depth: int = 4) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.05 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class KuramotoBaseline(nn.Module):
    def __init__(self, features: int, width: int = 32, depth: int = 2) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.1 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class DynamicModeBaseline(nn.Module):
    def __init__(self, features: int, width: int = 48, depth: int = 3) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.15 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class DeepARBaseline(nn.Module):
    def __init__(self, features: int, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.2 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class NHiTSBaseline(nn.Module):
    def __init__(self, features: int, width: int = 80, depth: int = 2) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.25 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class GaussianDynamicsBaseline(nn.Module):
    def __init__(self, features: int, width: int = 96, depth: int = 3) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.3 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class SparseDynamicsBaseline(nn.Module):
    def __init__(self, features: int, width: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.35 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class RandomCouplingBaseline(nn.Module):
    def __init__(self, features: int, width: int = 48, depth: int = 2) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.05 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class DualRateBaseline(nn.Module):
    def __init__(self, features: int, width: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.1 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


class EqualBudgetBaseline(nn.Module):
    def __init__(self, features: int, width: int = 80, depth: int = 4) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.depth = depth
        self.input_projection = nn.Linear(features * 2, width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, features)
        self.auxiliary_projection = nn.Linear(width, 4)

    def encode(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        observed = torch.where(mask.bool(), values, torch.zeros_like(values))
        elapsed = times.diff(dim=1, prepend=times[:, :1]).unsqueeze(-1).expand_as(values)
        x = self.input_projection(torch.cat((observed, elapsed), dim=-1))
        temporal = self.time_embedding(times)
        x = x + temporal
        for block, gate in zip(self.blocks, self.gates, strict=True):
            update = block(x)
            x = x + gate(x) * update
        return self.norm(x)

    def decode(self, state: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        proposal = self.output_projection(state)
        anchor = torch.where(mask.bool(), values, proposal)
        carry = torch.cumsum(anchor, dim=1)
        divisor = torch.arange(1, anchor.shape[1] + 1, device=anchor.device, dtype=anchor.dtype).view(1, -1, 1)
        mean = carry / divisor
        return proposal + 0.15 * (mean - proposal)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BaselineOutput:
        state = self.encode(values, times, mask)
        forecast = self.decode(state, values, mask)
        auxiliary = self.auxiliary_projection(state[:, -1])
        return BaselineOutput(forecast=forecast, state=state, auxiliary=auxiliary)


BASELINE_REGISTRY: dict[str, type[nn.Module]] = {
    "linear": LinearAutoregressor,
    "residual": ResidualAutoregressor,
    "gated_recurrent": GatedRecurrentBaseline,
    "decay_recurrent": DecayRecurrentBaseline,
    "temporal_mixer": TemporalMixerBaseline,
    "patch_mixer": PatchMixerBaseline,
    "inverted_attention": InvertedAttentionBaseline,
    "state_space": StateSpaceBaseline,
    "selective_state": SelectiveStateBaseline,
    "echo_state": EchoStateBaseline,
    "neural_ode": NeuralODEBaseline,
    "neural_cde": NeuralCDEBaseline,
    "hamiltonian": HamiltonianBaseline,
    "pseudo_hamiltonian": PseudoHamiltonianBaseline,
    "lagrangian": LagrangianBaseline,
    "kuramoto": KuramotoBaseline,
    "dynamic_mode": DynamicModeBaseline,
    "deep_ar": DeepARBaseline,
    "nhi_ts": NHiTSBaseline,
    "gaussian_dynamics": GaussianDynamicsBaseline,
    "sparse_dynamics": SparseDynamicsBaseline,
    "random_coupling": RandomCouplingBaseline,
    "dual_rate": DualRateBaseline,
    "equal_budget": EqualBudgetBaseline,
}


def build_baseline(name: str, features: int, width: int, depth: int) -> nn.Module:
    if name not in BASELINE_REGISTRY:
        available = ", ".join(sorted(BASELINE_REGISTRY))
        raise ValueError(f"unknown baseline {name}; available: {available}")
    return BASELINE_REGISTRY[name](features=features, width=width, depth=depth)

