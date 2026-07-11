import torch

from bepsy.model.dynamics import StuartLandauField, complex_order_parameter
from bepsy.model.world import BEPSY


def test_stuart_landau_radial_field() -> None:
    field = StuartLandauField(2)
    with torch.no_grad():
        field.mu.fill_(1)
        field.lyapunov.fill_(1)
    radius = torch.tensor([[0.5, 1.0]])
    derivative = field(radius)
    assert torch.allclose(derivative[..., 0], torch.tensor([[0.375, 0.0]]))


def test_order_parameter_synchrony() -> None:
    radius = torch.ones(2, 4)
    phase = torch.zeros(2, 4)
    magnitude, angle = complex_order_parameter(radius, phase)
    assert torch.allclose(magnitude, torch.ones(2))
    assert torch.allclose(angle, torch.zeros(2))


def test_world_model_shapes() -> None:
    model = BEPSY(modalities=3, compartments=4, slow_width=8, fast_width=8, heads=2)
    values = torch.randn(2, 5, 3)
    times = torch.arange(5).repeat(2, 1).float()
    mask = torch.ones_like(values, dtype=torch.bool)
    output = model(values, times, mask)
    assert output.reconstruction.shape == values.shape
    assert output.regime_logits.shape == (2, 4)
    assert output.latent.shape == (2, 5, 4, 3)

