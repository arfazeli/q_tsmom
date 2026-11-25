import numpy as np
import pytest
from scipy import stats as sps

from overfitting import dsr, expected_max_sr, psr


def test_psr_equals_benchmark_gives_one_half():
    assert psr(0.2, 0.2, 120, -0.5, 5.0) == pytest.approx(0.5)


def test_psr_normal_returns_matches_lo_standard_error():
    # skew 0, kurtosis 3: denominator sqrt(1 + SR^2 / 2), i.e. Lo (2002)
    sr, T = 0.2, 150
    z = sr * np.sqrt(T - 1) / np.sqrt(1 + sr**2 / 2)
    assert psr(sr, 0.0, T, 0.0, 3.0) == pytest.approx(sps.norm.cdf(z))


def test_negative_skew_and_fat_tails_lower_psr():
    base = psr(0.2, 0.0, 150, 0.0, 3.0)
    assert psr(0.2, 0.0, 150, -1.0, 3.0) < base
    assert psr(0.2, 0.0, 150, 0.0, 8.0) < base


def test_expected_max_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for n in (5, 18, 100):
        v = 0.01
        sims = rng.normal(0, np.sqrt(v), (20000, n)).max(axis=1).mean()
        assert expected_max_sr(n, v) == pytest.approx(sims, rel=0.04)


def test_dsr_falls_as_trials_grow():
    vals = [dsr(0.25, 150, -0.4, 3.5, n, 0.003) for n in (2, 10, 50, 200)]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))
    assert dsr(0.25, 150, -0.4, 3.5, 2, 0.003) < psr(0.25, 0.0, 150, -0.4, 3.5)
