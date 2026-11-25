import pandas as pd
import pytest

from db import init_db
from returns import compare_with_sql, daily_returns, load_prices, monthly_returns


def synthetic_db(tmp_path):
    """AAA from 2020-01-02, BBB from 2020-02-10; AAA misses 2020-02-14; RF missing on the
    last day."""
    conn = init_db(tmp_path / "t.sqlite")
    dates = pd.bdate_range("2020-01-02", "2020-04-30")
    rows = []
    for t, start, step in (("AAA", "2020-01-02", 0.003), ("BBB", "2020-02-10", -0.002)):
        conn.execute("INSERT INTO assets (ticker, asset_class) VALUES (?, 'equity')", (t,))
        for i, d in enumerate(dates[dates >= start]):
            if t == "AAA" and d == pd.Timestamp("2020-02-14"):
                continue
            rows.append((t, d.strftime("%Y-%m-%d"), 100 * (1 + step) ** i * (1 + 0.01 * (i % 3))))
    conn.executemany("INSERT INTO prices_clean (ticker, date, adj_close) VALUES (?,?,?)", rows)
    conn.executemany(
        "INSERT INTO rf_daily VALUES (?, ?)",
        [(d.strftime("%Y-%m-%d"), 0.0001) for d in dates[:-1]],
    )
    return conn


def test_sql_and_pandas_agree(tmp_path):
    d = compare_with_sql(synthetic_db(tmp_path))
    assert d["daily"] < 1e-12 and d["excess"] < 1e-12 and d["monthly"] < 1e-12
    assert d["rows_daily"] > 0 and d["rows_monthly"] == 3 + 2  # AAA Feb-Apr, BBB Mar-Apr


def test_hand_computed_month(tmp_path):
    conn = synthetic_db(tmp_path)
    p = load_prices(conn).AAA.dropna()
    jan_last, feb_last = p[p.index.month == 1].iloc[-1], p[p.index.month == 2].iloc[-1]
    m = monthly_returns(load_prices(conn))
    assert m.loc[pd.Period("2020-02", "M"), "AAA"] == pytest.approx(feb_last / jan_last - 1, 1e-15)
    sql = conn.execute(
        "SELECT ret FROM v_returns_monthly WHERE ticker='AAA' AND month='2020-02'"
    ).fetchone()[0]
    assert sql == pytest.approx(feb_last / jan_last - 1, rel=1e-15)
    # the month return equals the compounded daily returns of that month
    d = daily_returns(load_prices(conn)).AAA
    assert (1 + d[d.index.month == 2]).prod() - 1 == pytest.approx(sql, rel=1e-12)


def test_missing_day_return_spans_the_gap(tmp_path):
    conn = synthetic_db(tmp_path)
    date, ret = conn.execute(
        "SELECT date, ret FROM v_returns_daily WHERE ticker='AAA' AND date > '2020-02-13' "
        "ORDER BY date LIMIT 1"
    ).fetchone()
    p = load_prices(conn).AAA
    assert date == "2020-02-17"  # 02-14 is missing for AAA: one return over the gap
    assert ret == pytest.approx(p["2020-02-17"] / p["2020-02-13"] - 1, rel=1e-15)
