import torch
from torch import Tensor, nn


class SelectiveStateBlock(nn.Module):
    def __init__(self, inputs: int, width: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(inputs, width * 2)
        self.transition = nn.Parameter(torch.linspace(-0.1, -1.0, width))
        self.step_projection = nn.Linear(inputs, width)
        self.output_projection = nn.Linear(width, inputs)
        self.normalization = nn.LayerNorm(inputs)

    def forward(self, sequence: Tensor, deltas: Tensor) -> Tensor:
        projected, gates = self.input_projection(sequence).chunk(2, dim=-1)
        state = torch.zeros_like(projected[:, 0])
        outputs = []
        for index in range(sequence.shape[1]):
            step = torch.sigmoid(self.step_projection(sequence[:, index])) * deltas[:, index, None]
            decay = torch.exp(step * self.transition)
            state = decay * state + (1 - decay) * projected[:, index]
            outputs.append(self.output_projection(state * torch.sigmoid(gates[:, index])))
        result = torch.stack(outputs, dim=1)
        return self.normalization(result + sequence)


class DualRateForecaster(nn.Module):
    def __init__(self, compartments: int, slow_width: int, fast_width: int, gate_width: int) -> None:
        super().__init__()
        self.slow = SelectiveStateBlock(compartments, slow_width)
        self.fast = SelectiveStateBlock(compartments, fast_width)
        self.gate = nn.Sequential(
            nn.Linear(compartments * 2, gate_width),
            nn.SiLU(),
            nn.Linear(gate_width, compartments),
            nn.Sigmoid(),
        )
        self.mix = nn.Linear(compartments * 2, compartments * 2)

    def forward(self, radius: Tensor, phase: Tensor, deltas: Tensor) -> tuple[Tensor, Tensor]:
        slow = self.slow(radius, deltas / 30.0)
        fast = self.fast(torch.sin(phase), deltas * 24.0)
        gate = self.gate(torch.cat((slow, fast), dim=-1))
        mixed = self.mix(torch.cat((gate * slow, (1 - gate) * fast), dim=-1))
        radial, angular = mixed.chunk(2, dim=-1)
        return torch.nn.functional.softplus(radial), phase + angular

