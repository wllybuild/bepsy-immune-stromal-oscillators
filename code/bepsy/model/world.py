from dataclasses import dataclass

import torch
from torch import Tensor, nn

from bepsy.model.dynamics import PseudoHamiltonianGenerator, complex_order_parameter
from bepsy.model.encoder import ModalityTokenEncoder, ObservationDecoder
from bepsy.model.forecaster import DualRateForecaster
from bepsy.model.heads import BifurcationClassifier, SparseEquationReadout, TerminalArchiveDiscriminator


@dataclass(frozen=True)
class BEPSYOutput:
    reconstruction: Tensor
    forecast: Tensor
    latent: Tensor
    regime_logits: Tensor
    archive_logits: Tensor
    sparse_derivative: Tensor
    order_magnitude: Tensor
    order_phase: Tensor
    energy: Tensor
    potential: Tensor


class BEPSY(nn.Module):
    def __init__(
        self,
        modalities: int = 9,
        compartments: int = 12,
        slow_width: int = 256,
        fast_width: int = 128,
        gate_width: int = 16,
        heads: int = 8,
        classes: int = 4,
    ) -> None:
        super().__init__()
        encoder_width = max(64, heads * 16)
        self.encoder = ModalityTokenEncoder(modalities, encoder_width, heads, compartments)
        self.decoder = ObservationDecoder(compartments, modalities, encoder_width)
        self.forecaster = DualRateForecaster(compartments, slow_width, fast_width, gate_width)
        self.generator = PseudoHamiltonianGenerator(compartments, compartments * 2, encoder_width)
        self.classifier = BifurcationClassifier(compartments, encoder_width, classes)
        self.sparse = SparseEquationReadout(compartments)
        self.archive = TerminalArchiveDiscriminator(compartments * 2, 6)
        self.compartments = compartments

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> BEPSYOutput:
        encoded = self.encoder(values, times, mask)
        radius = encoded[..., 0]
        phase = encoded[..., 1]
        deltas = torch.zeros_like(times)
        deltas[:, 1:] = (times[:, 1:] - times[:, :-1]).clamp_min(0)
        forecast_radius, forecast_phase = self.forecaster(radius, phase, deltas)
        latent = torch.stack((forecast_radius, forecast_phase, encoded[..., 2]), dim=-1)
        reconstruction = self.decoder(encoded)
        forecast = self.decoder(latent)
        final_radius = forecast_radius[:, -1]
        final_phase = forecast_phase[:, -1]
        state = torch.stack((final_radius, final_phase), dim=-1).flatten(start_dim=-2)
        terms = self.generator(state, state)
        regime_logits = self.classifier(
            final_radius,
            final_phase,
            self.generator.normal_form.lyapunov,
            self.generator.normal_form.omega,
        )
        archive_logits = self.archive(state)
        sparse_derivative = self.sparse(forecast_radius, forecast_phase)
        magnitude, angle = complex_order_parameter(forecast_radius, forecast_phase)
        return BEPSYOutput(
            reconstruction,
            forecast,
            latent,
            regime_logits,
            archive_logits,
            sparse_derivative,
            magnitude,
            angle,
            terms.energy,
            terms.potential,
        )

    def forecast_horizon(self, values: Tensor, times: Tensor, mask: Tensor, horizon: int) -> Tensor:
        output = self(values, times, mask)
        current = output.latent[:, -1]
        predictions = []
        for _ in range(horizon):
            state = current[..., :2].flatten(start_dim=-2)
            advanced = self.generator.step(state, state, torch.ones(state.shape[0], device=state.device))
            paired = advanced.reshape(state.shape[0], self.compartments, 2)
            current = torch.stack((paired[..., 0], paired[..., 1], current[..., 2]), dim=-1)
            predictions.append(self.decoder(current))
        return torch.stack(predictions, dim=1)

