from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ChannelSummary:
    values: Tensor
    center: Tensor
    scale: Tensor
    missing: Tensor
    quality: Tensor


@dataclass(frozen=True)
class AlignmentResult:
    values: Tensor
    times: Tensor
    mask: Tensor
    uncertainty: Tensor


def _finite(values: Tensor) -> Tensor:
    return torch.isfinite(values)


def _quantile(values: Tensor, q: float) -> Tensor:
    return torch.quantile(values, q, dim=0, keepdim=True)


def _winsorize(values: Tensor, low: float, high: float) -> Tensor:
    lower = _quantile(values, low)
    upper = _quantile(values, high)
    return values.clamp(lower, upper)


class MetagenomeProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return torch.log1p(values.clamp_min(0.0))

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return torch.expm1(restored)


class MetatranscriptomeProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return torch.asinh(values)

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return torch.sinh(restored)


class MetabolomeProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return values

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return restored


class MetaproteomeProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return torch.log1p(values.clamp_min(0.0))

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return torch.expm1(restored)


class HostTranscriptomeProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return torch.asinh(values)

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return torch.sinh(restored)


class CalprotectinProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return values

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return restored


class CytokineProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return torch.log1p(values.clamp_min(0.0))

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return torch.expm1(restored)


class RibosomalProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return torch.asinh(values)

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return torch.sinh(restored)


class ViromeProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return values

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return restored


class MethylationProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return torch.log1p(values.clamp_min(0.0))

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return torch.expm1(restored)


class CopyNumberProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return torch.asinh(values)

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return torch.sinh(restored)


class SingleCellProcessor:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, minimum_scale: float = 1e-6) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.minimum_scale = minimum_scale

    def clean(self, values: Tensor) -> tuple[Tensor, Tensor]:
        finite = _finite(values)
        cleaned = torch.where(finite, values, torch.zeros_like(values))
        observed = finite.to(values.dtype)
        counts = observed.sum(dim=0, keepdim=True).clamp_min(1.0)
        means = cleaned.sum(dim=0, keepdim=True) / counts
        cleaned = torch.where(finite, cleaned, means)
        return cleaned, finite

    def transform(self, values: Tensor) -> Tensor:
        values = _winsorize(values, self.lower_quantile, self.upper_quantile)
        return values

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        center = values.median(dim=0, keepdim=True).values
        deviation = (values - center).abs().median(dim=0, keepdim=True).values
        scale = (1.4826 * deviation).clamp_min(self.minimum_scale)
        return (values - center) / scale, center, scale

    def quality(self, values: Tensor, mask: Tensor) -> Tensor:
        coverage = mask.to(values.dtype).mean(dim=0, keepdim=True)
        variance = values.var(dim=0, keepdim=True, unbiased=False)
        stability = torch.exp(-variance / variance.mean().clamp_min(self.minimum_scale))
        return 0.7 * coverage + 0.3 * stability

    def fit_transform(self, values: Tensor) -> ChannelSummary:
        cleaned, mask = self.clean(values)
        transformed = self.transform(cleaned)
        normalized, center, scale = self.normalize(transformed)
        quality = self.quality(normalized, mask)
        missing = (~mask).to(values.dtype)
        return ChannelSummary(values=normalized, center=center, scale=scale, missing=missing, quality=quality)

    def inverse(self, values: Tensor, center: Tensor, scale: Tensor) -> Tensor:
        restored = values * scale + center
        return restored


PROCESSORS = {
    "metagenome": MetagenomeProcessor,
    "metatranscriptome": MetatranscriptomeProcessor,
    "metabolome": MetabolomeProcessor,
    "metaproteome": MetaproteomeProcessor,
    "host_transcriptome": HostTranscriptomeProcessor,
    "calprotectin": CalprotectinProcessor,
    "cytokine": CytokineProcessor,
    "ribosomal": RibosomalProcessor,
    "virome": ViromeProcessor,
    "methylation": MethylationProcessor,
    "copy_number": CopyNumberProcessor,
    "single_cell": SingleCellProcessor,
}


def align_irregular_channels(values: list[Tensor], times: list[Tensor], grid: Tensor) -> AlignmentResult:
    if len(values) != len(times):
        raise ValueError("values and times must have equal length")
    channels: list[Tensor] = []
    masks: list[Tensor] = []
    uncertainties: list[Tensor] = []
    for channel, observed_times in zip(values, times, strict=True):
        distance = (grid[:, None] - observed_times[None, :]).abs()
        nearest_distance, nearest_index = distance.min(dim=1)
        selected = channel.index_select(0, nearest_index)
        scale = observed_times.diff().median().clamp_min(1e-6) if observed_times.numel() > 1 else torch.ones((), device=grid.device)
        uncertainty = 1.0 - torch.exp(-nearest_distance / scale)
        channels.append(selected)
        masks.append(nearest_distance.le(scale * 2.0))
        uncertainties.append(uncertainty)
    return AlignmentResult(values=torch.stack(channels, dim=-1), times=grid, mask=torch.stack(masks, dim=-1), uncertainty=torch.stack(uncertainties, dim=-1))

