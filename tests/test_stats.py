import numpy as np
import pandas as pd
import pytest

from stats import (
    factor_regression,
    lo_se,
    max_drawdown,
    nw_lags,
    nw_tstat,
    perf,
    sharpe_ci,
    stationary_bootstrap_idx,
)

IDX = pd.period_range("2000-01", periods=6, freq="M")
R = pd.Series([0.02, -0.01, 0.03, -0.04, 0.01, 0.02], index=IDX)


def test_perf_by_hand():
    p = perf(R)
    mu, sd = 0.03 / 6, np.std(R.to_numpy(), ddof=1)
    assert p["ann_mean"] == pytest.approx(12 * mu)
    assert p["ann_vol"] == pytest.approx(np.sqrt(12) * sd)
    assert p["sharpe"] == pytest.approx(np.sqrt(12) * mu / sd)
    assert p["hit_rate"] == pytest.approx(4 / 6)
    assert p["sharpe_se_lo"] == pytest.approx(np.sqrt(12) * np.sqrt((1 + (mu / sd) ** 2 / 2) / 6))


def test_max_drawdown_by_hand():
    # wealth 1 -> 1.1 -> 0.99 -> 0.792 -> 0.8712: peak 1.1, trough 0.792 -> 28%
    r = pd.Series([0.10, -0.10, -0.20, 0.10])
    assert max_drawdown(r) == pytest.approx(1 - 0.792 / 1.1)
    assert max_drawdown(pd.Series([-0.5, 0.1])) == pytest.approx(0.5)  # loss from the start
    assert max_drawdown(pd.Series([0.01, 0.02])) == 0


def test_lo_se_iid_zero_sharpe():
    assert lo_se(0.0, 100) == pytest.approx(0.1)


def nw_by_hand(x, L):
    T, e = len(x), x - x.mean()
    s = e @ e / T
    for j in range(1, L + 1):
        s += 2 * (1 - j / (L + 1)) * (e[j:] @ e[:-j]) / T
    return x.mean() / np.sqrt(s / T)


def test_newey_west_by_hand():
    x = np.random.default_rng(0).normal(0.01, 0.03, 120)
    for L in (0, 1, 4):
        assert nw_tstat(pd.Series(x), lags=L) == pytest.approx(nw_by_hand(x, L), rel=1e-6)
    assert nw_lags(149) == 4


def test_regression_recovers_exact_coefficients():
    rng = np.random.default_rng(1)
    idx = pd.period_range("2000-01", periods=60, freq="M")
    f = pd.DataFrame(rng.normal(0, 0.04, (60, 2)), index=idx, columns=["mkt_rf", "smb"])
    y = 0.002 + 0.5 * f.mkt_rf - 0.3 * f.smb
    out = factor_regression(y, f, ["mkt_rf", "smb"])
    assert out["alpha_monthly"] == pytest.approx(0.002)
    assert out["beta_mkt_rf"] == pytest.approx(0.5) and out["beta_smb"] == pytest.approx(-0.3)
    assert out["r2"] == pytest.approx(1.0) and out["months"] == 60


def test_bootstrap_indices_and_block_length():
    T, L = 200, 5.0
    idx = stationary_bootstrap_idx(T, L, 500, np.random.default_rng(0))
    assert idx.shape == (500, T) and idx.min() >= 0 and idx.max() < T
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % T).mean()
    assert continues == pytest.approx(1 - 1 / L, abs=0.01)  # new block w.p. 1/L


def test_bootstrap_ci_covers_known_sharpe():
    mu, sd, T = 0.01, 0.04, 240  # true annual Sharpe = sqrt(12) * 0.25 = 0.866
    true = np.sqrt(12) * mu / sd
    rng = np.random.default_rng(42)
    hits = []
    for seed in range(100):
        r = pd.Series(rng.normal(mu, sd, T))
        ci = sharpe_ci(r, B=1000, seed=seed)
        hits.append(ci["sharpe_ci_low"] <= true <= ci["sharpe_ci_high"])
    assert 0.88 <= np.mean(hits) <= 0.99, np.mean(hits)


def test_bootstrap_width_matches_lo_on_long_iid_sample():
    r = pd.Series(np.random.default_rng(3).normal(0.01, 0.04, 3000))
    ci = sharpe_ci(r, B=2000)
    p = perf(r)
    assert ci["sharpe_ci_high"] - ci["sharpe_ci_low"] == pytest.approx(
        2 * 1.96 * p["sharpe_se_lo"], rel=0.12
    )


def test_paired_sharpe_diff():
    from stats import paired_sharpe_diff

    rng = np.random.default_rng(5)
    b = pd.Series(rng.normal(0.005, 0.03, 200))
    same = paired_sharpe_diff(b, b, B=500)
    assert same["delta_sharpe"] == 0 and same["p_one_sided"] == 1.0  # all diffs exactly 0
    better = paired_sharpe_diff(b + 0.01, b, B=2000)  # same noise, higher mean
    assert better["delta_sharpe"] > 0 and better["p_one_sided"] < 0.01
