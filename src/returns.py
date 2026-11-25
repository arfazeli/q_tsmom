"""Daily and monthly returns in pandas: the cross-check of sql/views.sql.

Run `uv run python src/returns.py` to compare SQL and pandas on the real database.
"""

import sqlite3

import pandas as pd

from db import connect


def load_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    """Wide adj_close frame (index: date, columns: ticker) from prices_clean."""
    df = pd.read_sql("SELECT ticker, date, adj_close FROM prices_clean", conn, parse_dates=["date"])
    return df.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def load_rf(conn: sqlite3.Connection) -> pd.Series:
    rf = pd.read_sql("SELECT date, rf FROM rf_daily", conn, parse_dates=["date"])
    return rf.set_index("date").rf


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns P_t / P_{t-1} - 1 per ticker, over that ticker's own trading days."""
    return pd.DataFrame({t: s.dropna().pct_change() for t, s in prices.items()}).iloc[1:]


def excess(daily: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Daily excess return; NaN where the risk-free rate is missing."""
    return daily.sub(rf.reindex(daily.index), axis=0)


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Last price of each month over last price of the previous month, minus 1.
    Index is a monthly PeriodIndex."""
    out = {}
    for t, s in prices.items():
        last = s.dropna().groupby(s.dropna().index.to_period("M")).last()
        out[t] = last.pct_change().iloc[1:]
    return pd.DataFrame(out)


def _long(wide: pd.DataFrame, name: str, index=("date", "ticker")) -> pd.Series:
    s = wide.stack()
    s.index.names, s.name = list(index), name
    return s


def compare_with_sql(conn: sqlite3.Connection) -> dict[str, float]:
    """Max abs difference between SQL views and pandas, for daily, excess and monthly."""
    prices = load_prices(conn)
    d = daily_returns(prices)
    sql_d = pd.read_sql("SELECT * FROM v_returns_daily", conn, parse_dates=["date"])
    sql_d = sql_d.set_index(["date", "ticker"])
    py_d = pd.concat([_long(d, "ret"), _long(excess(d, load_rf(conn)), "excess_ret")], axis=1)
    sql_m = pd.read_sql("SELECT * FROM v_returns_monthly", conn)
    sql_m["month"] = pd.PeriodIndex(sql_m.month, freq="M")
    sql_m = sql_m.set_index(["month", "ticker"]).ret
    py_m = _long(monthly_returns(prices), "ret", ("month", "ticker"))

    def diff(a, b):
        a, b = a.align(b)  # rows present on one side only show up as NaN -> fail loudly
        if a.isna().ne(b.isna()).any():
            return float("inf")
        return float((a - b).abs().max())

    return {
        "rows_daily": len(sql_d),
        "rows_monthly": len(sql_m),
        "daily": diff(sql_d.ret, py_d.ret),
        "excess": diff(sql_d.excess_ret, py_d.excess_ret),
        "monthly": diff(sql_m, py_m),
    }


if __name__ == "__main__":
    print(compare_with_sql(connect()))
