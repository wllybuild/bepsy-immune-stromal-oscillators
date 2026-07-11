import torch
from torch import Tensor


def nrmse(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    weight = mask.to(target.dtype)
    mse = ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1)
    selected = target[mask]
    scale = selected.max() - selected.min()
    return mse.sqrt() / scale.clamp_min(torch.finfo(target.dtype).eps)


def lag_one_autocorrelation(values: Tensor, window: int) -> Tensor:
    if window < 3 or values.shape[-1] < window:
        raise ValueError("window must be at least three and no greater than the series length")
    windows = values.unfold(-1, window, 1)
    left = windows[..., :-1]
    right = windows[..., 1:]
    left = left - left.mean(dim=-1, keepdim=True)
    right = right - right.mean(dim=-1, keepdim=True)
    numerator = (left * right).sum(dim=-1)
    denominator = left.square().sum(dim=-1).sqrt() * right.square().sum(dim=-1).sqrt()
    return numerator / denominator.clamp_min(1e-8)


def coupling_asymmetry(matrix: Tensor, fibroblast: int, macrophage: int, tcell: int) -> Tensor:
    numerator = matrix[..., fibroblast, macrophage]
    denominator = matrix[..., tcell, macrophage]
    return numerator / denominator.clamp_min(1e-8)


def confusion_matrix(prediction: Tensor, target: Tensor, classes: int) -> Tensor:
    indices = target * classes + prediction
    return torch.bincount(indices, minlength=classes * classes).reshape(classes, classes)

