import pandas as pd
import pytest

from backtest import backtest

M = pd.period_range("2020-01", periods=3, freq="M")  # formation Jan, Feb; returns Feb, Mar


def frames():
    # The lesson's example: end of Jan SPY +0.6, TLT -1.2; end of Feb SPY +0.7, TLT -1.0
    pos = pd.DataFrame({"SPY": [0.6, 0.7], "TLT": [-1.2, -1.0]}, index=M[:2])
    x = pd.DataFrame({"SPY": [0.0, 0.02, 0.01], "TLT": [0.0, -0.01, 0.03]}, index=M)
    return pos, x


def test_hand_computed_two_etfs():
    pos, x = frames()
    r = backtest(pos, x, cost_bp=5)
    assert list(r.index.astype(str)) == ["2020-02", "2020-03"]  # indexed by return month
    # Feb: (0.6 * 2% + -1.2 * -1%) / 2 = 1.2%
    assert r.loc["2020-02", "gross_ret"] == pytest.approx(0.012)
    # Mar: (0.7 * 1% + -1.0 * 3%) / 2 = -1.15%
    assert r.loc["2020-03", "gross_ret"] == pytest.approx(-0.0115)
    # turnover: Jan = initial build |0.3| + |-0.6| = 0.9; Feb = |0.35-0.3| + |-0.5+0.6| = 0.15
    assert r.turnover.tolist() == pytest.approx([0.9, 0.15])
    assert r.loc["2020-03", "cost"] == pytest.approx(0.15 * 0.0005)
    assert (r.net_ret == r.gross_ret - r.cost).all()


def test_entering_etf_and_equal_weights():
    pos = pd.DataFrame({"A": [1.0, 1.0], "B": [None, 1.0]}, index=M[:2])
    x = pd.DataFrame({"A": [0.0, 0.01, 0.01], "B": [0.0, 0.05, 0.03]}, index=M)
    r = backtest(pos, x, cost_bp=0)
    assert r.n_assets.tolist() == [1, 2]
    assert r.loc["2020-02", "gross_ret"] == pytest.approx(0.01)  # only A held in Feb
    assert r.loc["2020-03", "gross_ret"] == pytest.approx((0.01 + 0.03) / 2)
    # Feb rebalance: A 1.0 -> 0.5, B 0 -> 0.5
    assert r.loc["2020-03", "turnover"] == pytest.approx(1.0)


def test_missing_next_month_return_raises():
    pos, x = frames()
    x.loc[M[2], "TLT"] = None
    with pytest.raises(ValueError, match="without next-month return"):
        backtest(pos, x)


def test_zero_cost_net_equals_gross():
    pos, x = frames()
    r = backtest(pos, x, cost_bp=0)
    assert (r.net_ret == r.gross_ret).all() and (r.cost == 0).all()


def test_holding_period_averages_vintages():
    m = pd.period_range("2020-01", periods=4, freq="M")
    pos = pd.DataFrame({"A": [1.0, -1.0, 1.0]}, index=m[:3])  # one ETF, flipping signal
    x = pd.DataFrame({"A": [0.0, 0.02, 0.02, 0.02]}, index=m)
    r = backtest(pos, x, cost_bp=0, holding=2)
    # weights held: Jan (1)/2 = 0.5 (only one vintage yet), Feb (1-1)/2 = 0, Mar (-1+1)/2 = 0
    assert r.gross_ret.tolist() == pytest.approx([0.5 * 0.02, 0.0, 0.0])
    assert r.turnover.tolist() == pytest.approx([0.5, 0.5, 0.0])
    one = backtest(pos, x, cost_bp=0, holding=1)
    assert one.gross_ret.tolist() == pytest.approx([0.02, -0.02, 0.02])


def test_contributions_sum_to_gross():
    from backtest import contributions

    pos, x = frames()
    for h in (1, 2):
        c = contributions(pos, x, holding=h)
        r = backtest(pos, x, cost_bp=5, holding=h)
        assert c.sum(axis=1).tolist() == pytest.approx(r.gross_ret.tolist())
    c = contributions(pos, x)
    assert c.loc["2020-02", "SPY"] == pytest.approx(0.6 / 2 * 0.02)
