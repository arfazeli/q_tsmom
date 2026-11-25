import numpy as np
import pandas as pd
import pytest

from volatility import ewma_vol


def noise(n=5000, sigma=0.01, seed=0):
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"X": np.random.default_rng(seed).normal(0, sigma, n)}, index=idx)


def test_recovers_known_volatility():
    v = ewma_vol(noise(), com=60, ann_days=261).X.dropna()
    assert np.median(v.iloc[200:]) == pytest.approx(0.01 * np.sqrt(261), rel=0.05)


def test_estimate_at_t_never_uses_day_t():
    r = noise(500)
    base = ewma_vol(r)
    for t in (100, 250, 499):
        bumped = r.copy()
        bumped.iloc[t] = 0.5  # huge shock on day t
        v = ewma_vol(bumped)
        pd.testing.assert_frame_equal(
            v.iloc[: t + 1], base.iloc[: t + 1], check_exact=True
        )  # dates <= t
        if t + 1 < len(r):
            assert v.X.iloc[t + 1] > base.X.iloc[t + 1]  # the shock shows up the next day


def test_warm_up_is_nan():
    v = ewma_vol(noise(300), min_periods=60).X
    assert v.iloc[:60].isna().all()  # 60 returns needed, plus the one-day lag
    assert v.iloc[60:].notna().all()


def test_matches_explicit_formula():
    r = noise(400).X
    d = 60 / 61
    t = 300  # estimate on day t uses r[0..t-1]
    x = r.iloc[:t].to_numpy()[::-1]  # most recent first
    w = d ** np.arange(len(x))
    w /= w.sum()
    mean = (w * x).sum()
    expect = np.sqrt(261 * (w * (x - mean) ** 2).sum())
    assert ewma_vol(r.to_frame()).X.iloc[t] == pytest.approx(expect, rel=1e-10)


def test_late_starting_ticker_uses_own_days():
    r = noise(300)
    r["Y"] = r.X
    r.loc[r.index[:100], "Y"] = np.nan  # Y starts at day 100
    v = ewma_vol(r)
    assert v.Y.iloc[:160].isna().all() and v.Y.iloc[160:].notna().all()
