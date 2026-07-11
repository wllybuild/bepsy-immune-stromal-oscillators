import math

import torch
from torch import Tensor, nn


class ModalityTokenEncoder(nn.Module):
    def __init__(self, modalities: int, width: int, heads: int, compartments: int) -> None:
        super().__init__()
        self.modalities = modalities
        self.width = width
        self.value_projection = nn.Linear(1, width)
        self.mask_embedding = nn.Embedding(2, width)
        self.modality_embedding = nn.Parameter(torch.randn(modalities, width) / math.sqrt(width))
        self.time_projection = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.normalization = nn.LayerNorm(width)
        self.compartments = nn.Parameter(torch.randn(compartments, width) / math.sqrt(width))
        self.compartment_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.state_projection = nn.Linear(width, 3)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        batch, steps, modalities = values.shape
        tokens = self.value_projection(values.unsqueeze(-1))
        tokens = tokens + self.mask_embedding(mask.long())
        tokens = tokens + self.modality_embedding[None, None]
        time_features = torch.stack((times, torch.log1p(times.clamp_min(0))), dim=-1)
        tokens = tokens + self.time_projection(time_features)[:, :, None]
        flat = tokens.reshape(batch * steps, modalities, self.width)
        attended, _ = self.attention(flat, flat, flat, need_weights=False)
        pooled = self.normalization(attended + flat).mean(dim=1).reshape(batch, steps, self.width)
        queries = self.compartments[None, None].expand(batch, steps, -1, -1)
        queries = queries.reshape(batch * steps, self.compartments.shape[0], self.width)
        context = pooled.reshape(batch * steps, 1, self.width)
        states, _ = self.compartment_attention(queries, context, context, need_weights=False)
        states = self.state_projection(states).reshape(batch, steps, -1, 3)
        radius = torch.nn.functional.softplus(states[..., 0])
        phase = math.pi * torch.tanh(states[..., 1])
        momentum = states[..., 2]
        return torch.stack((radius, phase, momentum), dim=-1)


class ObservationDecoder(nn.Module):
    def __init__(self, compartments: int, modalities: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(compartments * 3, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, modalities),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state.flatten(start_dim=-2))

