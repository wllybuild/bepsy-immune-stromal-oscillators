import numpy as np

from bepsy.statistics import benjamini_hochberg, bootstrap_interval, power_law_exponent


def test_bootstrap_regression() -> None:
    interval = bootstrap_interval(np.array([0.09, 0.10, 0.11, 0.12]), resamples=500, seed=3)
    assert interval.lower < interval.estimate < interval.upper


def test_hopf_exponent_regression() -> None:
    distance = np.linspace(0.1, 1.0, 100)
    relaxation = distance ** -0.5
    assert abs(power_law_exponent(distance, relaxation) - 0.5) < 1e-8


def test_false_discovery_control() -> None:
    result = benjamini_hochberg(np.array([0.001, 0.01, 0.2, 0.8]))
    assert result.tolist() == [True, True, False, False]

