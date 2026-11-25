import numpy as np
import pandas as pd
import pytest

from futures import (
    MIN_STABILITY,
    adjust,
    bad_ticks,
    choose_variant,
    clean_returns,
    detect_rolls,
    index_from_returns,
    quality_flags,
    roll_validation,
)

GAP = 0.02  # planted roll gap: the new contract trades 2% above the old one


def stitched(years=6, seed=0):
    """A continuous front-month series built from a known contract return stream r, with a
    +2% gap planted on every roll day (3rd week of Mar/Jun/Sep/Dec) and the volume
    signature of a roll: low volume for two days, then a jump."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=years * 252)
    n = len(dates)
    r = pd.Series(rng.normal(0, 0.01, n), index=dates)
    roll_days = []
    for y in sorted(set(dates.year)):
        for m in (3, 6, 9, 12):
            d = dates[(dates.year == y) & (dates.month == m) & (dates.day >= 15)]
            if len(d):
                roll_days.append(d[0])
    roll_days = pd.DatetimeIndex(roll_days)
    factor = np.where(dates.isin(roll_days), 1 + GAP, 1.0)
    close = pd.Series(100 * np.cumprod((1 + r.to_numpy()) * factor), index=dates)
    vol = pd.Series(1e6 * rng.lognormal(0, 0.15, n), index=dates)
    for d in roll_days:
        i = dates.get_loc(d)
        vol.iloc[i - 2], vol.iloc[i - 1], vol.iloc[i] = 0.6e6, 0.2e6, 2.0e6
    return close, vol, r, roll_days


@pytest.mark.parametrize("variant", ["median", "dipjump"])
def test_planted_rolls_are_found_and_the_gap_removed(variant):
    close, vol, r, roll_days = stitched()
    found = detect_rolls(close, vol, 4, variant)
    assert found.index.equals(roll_days)
    adj = adjust(clean_returns(close), found.index)
    ordinary = adj.index.difference(roll_days)
    # on every non-roll day the adjusted return equals the true contract return exactly
    np.testing.assert_allclose(adj.loc[ordinary], r.loc[ordinary], atol=1e-12)
    assert (adj.loc[roll_days] == 0).all()
    # without the adjustment the series carries the planted gap on every roll
    raw = (1 + clean_returns(close)).prod()
    clean_total = (1 + r.iloc[1:]).prod()
    assert raw / clean_total == pytest.approx((1 + GAP) ** len(roll_days), rel=1e-9)


def test_cycle_is_recovered_and_validated():
    close, vol, _, _ = stitched()
    found = detect_rolls(close, vol, 4, "dipjump")
    v = roll_validation(found, 4)
    assert v["modal_months"] == "3 6 9 12"
    assert v["stability_exact"] == 1.0 and v["stability_jaccard"] == 1.0
    assert v["day_of_month_iqr"] <= 7  # all in the 3rd week
    variant, _, _, kept = choose_variant(close, vol, 4)
    assert kept


def test_random_volume_is_rejected():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2010-01-04", periods=8 * 252)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))), index=dates)
    vol = pd.Series(1e6 * rng.lognormal(0, 0.4, len(dates)), index=dates)
    _, _, v, kept = choose_variant(close, vol, 4)
    assert v["stability_jaccard"] < MIN_STABILITY and not kept


def test_bad_tick_is_dropped_and_path_unchanged():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2010-01-04", periods=120)
    true = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 120)), index=dates)
    glitch = true.copy()
    glitch.iloc[50] *= 0.1  # a 10x error for one day, as in yen futures
    assert bad_ticks(glitch).sum() == 1 and bad_ticks(glitch).iloc[50]
    assert not bad_ticks(true).any()
    ret = clean_returns(glitch)
    assert ret.abs().max() < 0.1
    assert (1 + ret).prod() == pytest.approx(true.iloc[-1] / true.iloc[0], rel=1e-12)
    flags = quality_flags(glitch, pd.Series(1.0, index=dates))
    assert (flags.check == "bad_tick").sum() == 1


def test_non_positive_prices_give_zero_returns_and_a_flag():
    dates = pd.bdate_range("2020-04-14", periods=8)
    close = pd.Series([25.0, 20.0, 18.0, -37.0, 10.0, 13.0, 16.0, 17.0], index=dates)
    ret = clean_returns(close)
    assert ret.iloc[2] == 0.0 and ret.iloc[3] == 0.0  # into and out of the negative price
    assert ret.iloc[4] == pytest.approx(13 / 10 - 1)  # normal again
    flags = quality_flags(close, pd.Series(1.0, index=dates))
    assert (flags.check == "non_positive").sum() == 1


def test_stale_prices_flagged_from_the_fifth_unchanged_day():
    dates = pd.bdate_range("2020-01-01", periods=9)
    close = pd.Series([10, 11, 11, 11, 11, 11, 11, 12, 13.0], index=dates)
    flags = quality_flags(close, pd.Series(1.0, index=dates))
    stale = flags[flags.check == "stale"]
    assert list(stale.date) == list(dates[5:7])  # 5th and 6th identical closes


def test_index_from_returns_round_trips():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = {"A": pd.Series([0.01, -0.02, 0.03, 0.0], index=idx), "B": pd.Series([0.1], index=idx[2:3])}
    w = index_from_returns(r)
    assert w.A.pct_change().dropna().to_numpy() == pytest.approx([-0.02, 0.03, 0.0])
    assert w.B.notna().sum() == 1 and w.A.notna().all()


def test_first_sample_month_needs_enough_contracts_and_groups():
    from futures_backtest import first_sample_month

    rows = []
    for m, tickers in {
        "2002-01": list("ABCDEFGHI"),
        "2002-02": list("ABCDEFGHIJ"),
        "2002-03": list("ABCDEFGHIJK"),
    }.items():
        rows += [{"ticker": t, "month": m} for t in tickers]
    sig = pd.DataFrame(rows)
    one_group = {t: "x" for t in "ABCDEFGHIJK"}
    few_groups = {**one_group, "A": "y"}  # 2 groups: not enough
    many = {t: "xyzw"[i % 4] for i, t in enumerate("ABCDEFGHIJK")}
    assert first_sample_month(sig, many) == pd.Period("2002-02", "M")  # 10 contracts, 4 groups
    with pytest.raises(IndexError):
        first_sample_month(sig, few_groups)
