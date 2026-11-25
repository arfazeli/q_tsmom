import numpy as np
import pandas as pd
import pytest

from backtest import backtest
from benchmarks import always_long, effective_n, long_only_trend, sixty_forty


def test_effective_n_extremes():
    assert effective_n(pd.DataFrame(np.eye(5))) == pytest.approx(5.0)
    assert effective_n(pd.DataFrame(np.ones((6, 6)))) == pytest.approx(1.0)
    # two blocks of perfectly correlated assets behave like two independent bets
    c = np.kron(np.eye(2), np.ones((3, 3)))
    assert effective_n(pd.DataFrame(c)) == pytest.approx(2.0)


def sig_frame():
    return pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "month": ["2020-01", "2020-02"] * 2,
            "lookback": 12,
            "signal": [1, -1, -1, 1],
            "vol_exante": 0.1,
            "position": [1.0, -0.5, -2.0, 0.7],
        }
    )


def test_always_long_and_long_only():
    s = sig_frame()
    assert always_long(s).position.tolist() == [1.0, 0.5, 2.0, 0.7]
    assert long_only_trend(s).position.tolist() == [1.0, 0.0, 0.0, 0.7]
    assert s.position.tolist() == [1.0, -0.5, -2.0, 0.7]  # input untouched


def test_sixty_forty():
    idx = pd.period_range("2020-01", periods=2, freq="M")
    r = sixty_forty(pd.Series([0.10, -0.05], index=idx), pd.Series([0.01, 0.02], index=idx))
    assert r.tolist() == pytest.approx([0.6 * 0.10 + 0.4 * 0.01, 0.6 * -0.05 + 0.4 * 0.02])


def test_class_portfolios_add_up_to_the_total():
    # 3 ETFs, classes {a1, a2} and {b1}: total = 2/3 * class A + 1/3 * class B
    m = pd.period_range("2020-01", periods=4, freq="M")
    rng = np.random.default_rng(0)
    pos = pd.DataFrame(rng.normal(0, 1, (3, 3)), index=m[:3], columns=["a1", "a2", "b1"])
    x = pd.DataFrame(rng.normal(0, 0.03, (4, 3)), index=m, columns=pos.columns)
    total = backtest(pos, x, cost_bp=0).gross_ret
    ga = backtest(pos[["a1", "a2"]], x, cost_bp=0).gross_ret
    gb = backtest(pos[["b1"]], x, cost_bp=0).gross_ret
    assert total.to_numpy() == pytest.approx((2 / 3 * ga + 1 / 3 * gb).to_numpy())
