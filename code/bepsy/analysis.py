from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Measure:
    value: Tensor
    count: Tensor
    lower: Tensor
    upper: Tensor


def _valid_pair(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    return prediction[valid], target[valid]


def _interval(value: Tensor, spread: Tensor, count: Tensor) -> tuple[Tensor, Tensor]:
    radius = 1.96 * spread / count.clamp_min(1).sqrt()
    return value - radius, value + radius


class NormalizedRootMeanSquareMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class RootMeanSquareMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class MeanAbsoluteMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class SymmetricPercentageMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class SpectralKolmogorovMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class PhaseLockingMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class RadialDeviationMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class FrequencyDetuningMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class CouplingAsymmetryMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class LeadTimeMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class LagOneCorrelationMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class CriticalExponentMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class BrierMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class ExpectedCalibrationMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class BinaryAreaUnderCurveMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class MulticlassAccuracyMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class BalancedAccuracyMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class CohensEffectMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class WilcoxonEffectMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class ArchiveGapMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class SubgroupGapMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class TerminalConsistencyMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class HamiltonianDriftMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class DissipationRateMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class SparsityFractionMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class RankMarginMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class ForecastCoverageMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class IntervalWidthMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class PhaseBoundaryMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(1) / scale.pow(1)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


class LyapunovEstimateMeasure:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def contributions(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction, target = _valid_pair(prediction, target)
        difference = prediction - target
        scale = target.abs().mean().clamp_min(self.epsilon)
        return difference.abs().pow(2) / scale.pow(2)

    def aggregate(self, contributions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = torch.tensor(contributions.numel(), device=contributions.device, dtype=contributions.dtype)
        value = contributions.mean() if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        spread = contributions.std(unbiased=False) if contributions.numel() else torch.full((), float("nan"), device=contributions.device)
        return value, spread, count

    def compute(self, prediction: Tensor, target: Tensor) -> Measure:
        contributions = self.contributions(prediction, target)
        value, spread, count = self.aggregate(contributions)
        lower, upper = _interval(value, spread, count)
        return Measure(value=value, count=count, lower=lower, upper=upper)

    def grouped(self, prediction: Tensor, target: Tensor, groups: Tensor) -> dict[int, Measure]:
        result: dict[int, Measure] = {}
        for group in groups.unique().tolist():
            selected = groups == group
            result[int(group)] = self.compute(prediction[selected], target[selected])
        return result

    def compare(self, first: Measure, second: Measure) -> Tensor:
        denominator = second.value.abs().clamp_min(self.epsilon)
        return (first.value - second.value) / denominator


MEASURE_REGISTRY = {
    "normalized_root_mean_square": NormalizedRootMeanSquareMeasure,
    "root_mean_square": RootMeanSquareMeasure,
    "mean_absolute": MeanAbsoluteMeasure,
    "symmetric_percentage": SymmetricPercentageMeasure,
    "spectral_kolmogorov": SpectralKolmogorovMeasure,
    "phase_locking": PhaseLockingMeasure,
    "radial_deviation": RadialDeviationMeasure,
    "frequency_detuning": FrequencyDetuningMeasure,
    "coupling_asymmetry": CouplingAsymmetryMeasure,
    "lead_time": LeadTimeMeasure,
    "lag_one_correlation": LagOneCorrelationMeasure,
    "critical_exponent": CriticalExponentMeasure,
    "brier": BrierMeasure,
    "expected_calibration": ExpectedCalibrationMeasure,
    "binary_area_under_curve": BinaryAreaUnderCurveMeasure,
    "multiclass_accuracy": MulticlassAccuracyMeasure,
    "balanced_accuracy": BalancedAccuracyMeasure,
    "cohens_effect": CohensEffectMeasure,
    "wilcoxon_effect": WilcoxonEffectMeasure,
    "archive_gap": ArchiveGapMeasure,
    "subgroup_gap": SubgroupGapMeasure,
    "terminal_consistency": TerminalConsistencyMeasure,
    "hamiltonian_drift": HamiltonianDriftMeasure,
    "dissipation_rate": DissipationRateMeasure,
    "sparsity_fraction": SparsityFractionMeasure,
    "rank_margin": RankMarginMeasure,
    "forecast_coverage": ForecastCoverageMeasure,
    "interval_width": IntervalWidthMeasure,
    "phase_boundary": PhaseBoundaryMeasure,
    "lyapunov_estimate": LyapunovEstimateMeasure,
}


def compute_measure(name: str, prediction: Tensor, target: Tensor) -> Measure:
    if name not in MEASURE_REGISTRY:
        raise ValueError(f"unknown measure {name}")
    return MEASURE_REGISTRY[name]().compute(prediction, target)

