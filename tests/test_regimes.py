import numpy as np
import pandas as pd
import pytest

from regimes import labels

DAYS = pd.bdate_range("2000-01-03", "2009-12-31")


def market(seed=0):
    r = np.random.default_rng(seed).normal(0.0003, 0.01, len(DAYS))
    return pd.Series(100 * np.exp(np.cumsum(r)), index=DAYS)


def test_label_for_month_m_ignores_data_from_month_m_on():
    p = market()
    base = labels(p)
    for cut in ("2004-06", "2007-01"):
        m = pd.Period(cut, "M")
        bumped = p.copy()
        after = bumped.index >= m.start_time  # change month m and everything after
        bumped[after] *= np.exp(
            np.cumsum(np.random.default_rng(1).normal(-0.01, 0.05, after.sum()))
        )
        new = labels(bumped)
        pd.testing.assert_frame_equal(
            new.loc[:m], base.loc[:m], check_exact=True
        )  # labels up to and incl. m
        assert not new.loc[m + 1 :].equals(base.loc[m + 1 :])


def test_bear_and_high_vol_by_construction():
    # flat-ish calm market, then a 30% crash with high volatility in 2008-10
    r = pd.Series(0.0002, index=DAYS) + np.random.default_rng(2).normal(0, 0.005, len(DAYS))
    oct08 = (DAYS >= "2008-10-01") & (DAYS <= "2008-10-31")
    r[oct08] = np.random.default_rng(3).normal(-0.015, 0.04, oct08.sum())
    p = pd.Series(100 * np.exp(np.cumsum(r)), index=DAYS)
    lab = labels(p)
    assert lab.loc["2008-11", "high_vol"] == True  # noqa: E712  label applies to Nov
    assert lab.loc["2008-10", "high_vol"] == False  # noqa: E712  Oct label uses Sep data
    assert lab.market_rvol.loc["2008-11"] == pytest.approx(
        p.pct_change()[oct08].std() * np.sqrt(261)
    )


def test_warm_up_is_nan():
    lab = labels(market())
    assert lab.high_vol.iloc[:35].isna().all() and lab.high_vol.iloc[35:].notna().all()
    assert lab.bear.iloc[:24].isna().all() and lab.bear.iloc[24:].notna().all()


def test_regime_table_by_hand():
    from regime_analysis import quintiles, regime_table

    idx = pd.period_range("2020-01", periods=6, freq="M")
    r = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02], index=idx)
    lab = pd.Series([True, True, False, False, False, None], index=idx, dtype="boolean")
    t, f = regime_table(r, lab, "bear")
    assert t["months"] == 2 and t["ann_mean"] == pytest.approx(12 * -0.005)
    assert f["months"] == 3 and f["hit_rate"] == pytest.approx(2 / 3)  # unknown month excluded
    q = quintiles(pd.Series(range(10), dtype=float), pd.Series(range(10), dtype=float))
    assert q.months.tolist() == [2] * 5 and q.strategy_mean.tolist() == [0.5, 2.5, 4.5, 6.5, 8.5]
