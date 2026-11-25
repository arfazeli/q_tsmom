import numpy as np
import pandas as pd
import pytest

from db import init_db, log_run
from signals import build, momentum_signal, positions, vol_at_month_end, write_signals

M = pd.period_range("2010-01", periods=15, freq="M")


def test_signs_by_hand():
    # A: +1% every month -> long; B: -1% -> short; C: 11 x +1% then one -20% -> short
    x = pd.DataFrame({"A": 0.01, "B": -0.01, "C": 0.01}, index=M)
    x.loc[M[11], "C"] = -0.20
    s = momentum_signal(x, lookback=12)
    assert s.iloc[:11].isna().all().all()  # needs 12 months
    assert s.loc[M[11]].tolist() == [1, -1, -1]
    # C at M[12]: months 1..12 -> 10 x 1.01 * 0.8 * 1.01 < 1 -> still short
    assert s.loc[M[12], "C"] == -1


def test_zero_return_gives_zero_signal():
    x = pd.DataFrame({"A": 0.0}, index=M)
    assert momentum_signal(x).A.iloc[11] == 0


def test_cap_binds_on_low_vol_asset():
    sig = pd.DataFrame({"BOND": 1.0, "OIL": -1.0}, index=M[:1])
    vol = pd.DataFrame({"BOND": 0.05, "OIL": 0.40}, index=M[:1])
    p = positions(sig, vol, target=0.40, cap=2.0)
    assert p.BOND.iloc[0] == 2.0  # 0.40 / 0.05 = 8 -> capped
    assert p.OIL.iloc[0] == pytest.approx(-1.0)


def synthetic(n_days=900, seed=1):
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    prices = pd.DataFrame(
        {
            "X": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n_days))),
            "Y": 100 * np.exp(np.cumsum(rng.normal(-0.0004, 0.02, n_days))),
        },
        index=idx,
    )
    import returns
    from volatility import ewma_vol

    rf = pd.Series(0.001, index=pd.period_range("2009-12", "2014-12", freq="M"))
    return prices, returns.monthly_returns(prices), rf, ewma_vol(returns.daily_returns(prices))


def test_no_look_ahead():
    import returns
    from volatility import ewma_vol

    prices, mret, rf, dvol = synthetic()
    base = build(mret, rf, dvol).set_index(["ticker", "month"])
    cut = pd.Period("2011-12", "M")
    bumped = prices.copy()
    after = bumped.index > cut.end_time
    bumped.loc[after] *= np.linspace(1, 3, after.sum())[:, None]  # rewrite the future
    new = build(
        returns.monthly_returns(bumped), rf, ewma_vol(returns.daily_returns(bumped))
    ).set_index(["ticker", "month"])
    upto = lambda d: d[d.index.get_level_values("month") <= str(cut)]  # noqa: E731
    pd.testing.assert_frame_equal(upto(new), upto(base), check_exact=True)
    assert not new.equals(base)  # the perturbation did change later months


def test_build_and_write(tmp_path):
    _, mret, rf, dvol = synthetic()
    df = build(mret, rf, dvol)
    assert df.month.min() == "2011-01"  # first return Feb 2010; 12 returns -> Jan 2011
    assert set(df.signal) <= {-1, 0, 1}
    assert (df.position.abs() <= 2.0).all()
    conn = init_db(tmp_path / "t.sqlite")
    for t in ("X", "Y"):
        conn.execute("INSERT INTO assets (ticker, asset_class) VALUES (?, 'equity')", (t,))
    run_id = log_run(conn, {"test": True}, "in")
    write_signals(conn, run_id, df)
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == len(df)


def test_vol_at_month_end_takes_last_trading_day():
    idx = pd.to_datetime(["2010-01-28", "2010-01-29", "2010-02-26"])
    v = pd.DataFrame({"X": [0.1, 0.2, 0.3]}, index=idx)
    assert vol_at_month_end(v).X.tolist() == [0.2, 0.3]
