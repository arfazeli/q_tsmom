"""Time-series momentum signals and volatility-scaled positions .

Timing: everything indexed by month t is known at the close of the last trading day of
month t. The backtest applies the position at t to the return of month t+1.

- Monthly excess return of month t: r_t - rf_t, with rf_t the French monthly risk-free rate
  (factors_monthly.rf). NaN when rf is missing.
- Signal at t: sign of the compounded excess return over months t-lookback+1 .. t,
  prod(1 + r - rf) - 1. NaN until `lookback` monthly returns exist; exactly 0 -> 0.
- Position at t: clip(signal * target / vol_t, -cap, cap), where vol_t is the EWMA vol on
  the last trading day of month t (it already excludes that day's return).

Run `uv run python src/signals.py` to write in-sample signals to the database.
"""

import sqlite3

import numpy as np
import pandas as pd

from config import STRATEGY


def monthly_excess(monthly_ret: pd.DataFrame, rf_monthly: pd.Series) -> pd.DataFrame:
    return monthly_ret.sub(rf_monthly.reindex(monthly_ret.index), axis=0)


def momentum_signal(
    monthly_excess: pd.DataFrame, lookback: int = STRATEGY["lookback_months"]
) -> pd.DataFrame:
    cum = np.log1p(monthly_excess).rolling(lookback, min_periods=lookback).sum()
    return np.sign(cum)  # sign(log(prod(1+x))) == sign(prod(1+x) - 1); NaN stays NaN


def vol_at_month_end(daily_vol: pd.DataFrame) -> pd.DataFrame:
    """Each ticker's vol on its last trading day of each month (monthly PeriodIndex)."""
    out = {}
    for t, v in daily_vol.items():
        v = v.dropna()
        out[t] = v.groupby(v.index.to_period("M")).last()
    return pd.DataFrame(out)


def positions(
    signal: pd.DataFrame,
    vol: pd.DataFrame,
    target: float = STRATEGY["vol_target"],
    cap: float = STRATEGY["leverage_cap"],
) -> pd.DataFrame:
    vol = vol.reindex(index=signal.index, columns=signal.columns)
    return (signal * target / vol).clip(-cap, cap)


def build(monthly_ret, rf_monthly, daily_vol, lookback=STRATEGY["lookback_months"]):
    """Long frame (ticker, month, lookback, signal, vol_exante, position); rows where the
    signal or position is undefined are dropped."""
    sig = momentum_signal(monthly_excess(monthly_ret, rf_monthly), lookback)
    vol = vol_at_month_end(daily_vol).reindex(index=sig.index, columns=sig.columns)
    pos = positions(sig, vol)
    df = pd.concat(
        {"signal": sig.stack(), "vol_exante": vol.stack(), "position": pos.stack()}, axis=1
    )
    df.index.names = ["month", "ticker"]
    df = df.dropna().reset_index()
    df["lookback"] = lookback
    df["signal"] = df.signal.astype(int)
    df["month"] = df.month.astype(str)
    return df[["ticker", "month", "lookback", "signal", "vol_exante", "position"]]


def write_signals(conn: sqlite3.Connection, run_id: int, df: pd.DataFrame) -> None:
    conn.executemany(
        "INSERT INTO signals (run_id, ticker, month, lookback, signal, vol_exante, position) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ((run_id, *row) for row in df.itertuples(index=False)),
    )
    conn.commit()


def load_inputs(conn: sqlite3.Connection):
    import returns
    from volatility import ewma_vol

    prices = returns.load_prices(conn)
    rf = pd.read_sql("SELECT month, rf FROM factors_monthly", conn)
    rf = rf.set_index(pd.PeriodIndex(rf.month, freq="M")).rf
    daily_vol = ewma_vol(returns.daily_returns(prices))
    return returns.monthly_returns(prices), rf, daily_vol


if __name__ == "__main__":
    from db import init_db, log_run

    conn = init_db()
    df = build(*load_inputs(conn))
    df = df[df.month <= STRATEGY["in_sample_end"][:7]]  # in-sample only
    run_id = log_run(conn, {"step": "signals", **STRATEGY}, sample="in")
    write_signals(conn, run_id, df)
    capped = df.position.abs() >= STRATEGY["leverage_cap"] - 1e-12
    print(f"run_id {run_id}: {len(df)} asset-months, {df.month.min()} .. {df.month.max()}")
    print(f"cap binds in {capped.mean():.1%} of asset-months")
    print(f"long share {(df.signal > 0).mean():.1%}")
