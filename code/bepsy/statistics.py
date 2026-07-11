from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


def bootstrap_interval(
    values: NDArray[np.float64],
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 41,
) -> Interval:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("values must be a one-dimensional array with at least two entries")
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(resamples, values.size), replace=True)
    estimates = samples.mean(axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(estimates, (alpha, 1 - alpha))
    return Interval(float(values.mean()), float(lower), float(upper))


def benjamini_hochberg(pvalues: NDArray[np.float64], alpha: float = 0.05) -> NDArray[np.bool_]:
    if np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("p-values must fall in the closed unit interval")
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    thresholds = alpha * np.arange(1, pvalues.size + 1) / pvalues.size
    accepted = ranked <= thresholds
    result = np.zeros(pvalues.size, dtype=np.bool_)
    if accepted.any():
        result[order[: np.flatnonzero(accepted)[-1] + 1]] = True
    return result


def power_law_exponent(distance: NDArray[np.float64], relaxation: NDArray[np.float64]) -> float:
    valid = (distance > 0) & (relaxation > 0) & np.isfinite(distance) & np.isfinite(relaxation)
    if valid.sum() < 3:
        raise ValueError("at least three positive finite pairs are required")
    slope, _ = np.polyfit(np.log(distance[valid]), np.log(relaxation[valid]), 1)
    return float(-slope)


def paired_effect_size(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    difference = first - second
    return float(difference.mean() / difference.std(ddof=1))

