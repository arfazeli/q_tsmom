"""No-look-ahead and synthetic-truth tests on the full pipeline:
daily prices -> returns -> EWMA vol -> signals -> positions -> backtest."""

import numpy as np
import pandas as pd
import pytest

import returns
from backtest import backtest, wide
from signals import build, monthly_excess
from volatility import ewma_vol

N_ASSETS, YEARS = 8, 12
DAYS = pd.bdate_range("2000-01-03", periods=YEARS * 261)
SIGMAS = np.linspace(0.005, 0.02, N_ASSETS)  # daily vol: bond-like to commodity-like


def run(prices: pd.DataFrame, cost_bp: float = 5.0):
    daily = returns.daily_returns(prices)
    mret = returns.monthly_returns(prices)
    rf = pd.Series(0.0, index=mret.index)
    sig = build(mret, rf, ewma_vol(daily))
    pos = wide(sig, "position").iloc[:-1]  # last formation month has no next-month return
    res = backtest(pos, monthly_excess(mret, rf), cost_bp)
    return sig, res


def to_prices(daily_ret: np.ndarray) -> pd.DataFrame:
    cols = [f"A{i}" for i in range(daily_ret.shape[1])]
    return pd.DataFrame(100 * np.exp(np.cumsum(daily_ret, axis=0)), index=DAYS, columns=cols)


def random_walk(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return to_prices(rng.normal(0.0, SIGMAS, (len(DAYS), N_ASSETS)))


def trending(seed: int, phi: float = 0.9, drift_sd: float = 0.01) -> pd.DataFrame:
    """Each asset's monthly drift follows an AR(1) with persistence phi, so past returns
    predict future returns: the setting where trend-following must work."""
    rng = np.random.default_rng(seed)
    months = DAYS.to_period("M")
    month_ids = pd.factorize(months)[0]
    n_months = month_ids.max() + 1
    drift = np.zeros((n_months, N_ASSETS))
    for t in range(1, n_months):
        drift[t] = phi * drift[t - 1] + rng.normal(0.0, drift_sd, N_ASSETS)
    daily_drift = drift[month_ids] / 21
    return to_prices(daily_drift + rng.normal(0.0, SIGMAS, (len(DAYS), N_ASSETS)))


def sharpe(r: pd.Series) -> float:
    return float(np.sqrt(12) * r.mean() / r.std())


def test_a_no_look_ahead():
    prices = random_walk(0)
    cut = pd.Timestamp("2006-06-30")
    sig, res = run(prices)
    bumped = prices.copy()
    after = bumped.index > cut
    rng = np.random.default_rng(99)
    bumped.loc[after] *= np.exp(np.cumsum(rng.normal(0.01, 0.05, (after.sum(), N_ASSETS)), axis=0))
    sig2, res2 = run(bumped)

    def formed_by(s):
        return s[s.month <= "2006-06"].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        formed_by(sig2), formed_by(sig), check_exact=True
    )  # signals, vols, positions
    pd.testing.assert_frame_equal(
        res2.loc[:"2006-06"], res.loc[:"2006-06"], check_exact=True
    )  # P&L to June
    assert not res2.loc["2006-07":].equals(res.loc["2006-07":])  # the future did change


def test_b_random_walk_sharpe_near_zero():
    s = np.array([sharpe(run(random_walk(seed), cost_bp=0)[1].net_ret) for seed in range(300)])
    assert abs(s.mean()) < 0.06, s.mean()  # 300 seeds: std error of the mean ~ 0.018
    assert 0.1 < s.std() < 0.6  # each seed is noisy, as a ~11-year Sharpe should be


def test_c_trend_persistence_gives_positive_sharpe():
    s = np.array([sharpe(run(trending(seed))[1].net_ret) for seed in range(20)])
    assert s.mean() > 0.5, s.mean()
    assert (s > 0).mean() >= 0.9, s


def test_d_zero_cost_net_equals_gross():
    _, res = run(random_walk(1), cost_bp=0)
    assert (res.net_ret == res.gross_ret).all()
    _, res5 = run(random_walk(1), cost_bp=5)
    assert (res5.gross_ret == res.gross_ret).all() and (res5.net_ret < res5.gross_ret).any()


@pytest.mark.parametrize("seed", [0])
def test_report_numbers(seed, capsys):
    """Not an assertion: prints the synthetic Sharpe levels."""
    rw = [sharpe(run(random_walk(s), 0)[1].net_ret) for s in range(300)]
    tr = [sharpe(run(trending(s))[1].net_ret) for s in range(20)]
    with capsys.disabled():
        print(
            f"\nrandom walk: mean Sharpe {np.mean(rw):+.3f} (sd {np.std(rw):.2f}); "
            f"trend: mean {np.mean(tr):.2f}, min {np.min(tr):.2f}"
        )
