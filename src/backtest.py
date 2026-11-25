"""Vectorized monthly backtest.

Timing: positions formed at the close of month t (src/signals.py) earn the excess return
of month t+1. The cost of the rebalance at the end of month t is charged to month t+1.

    w_i,t      = position_i,t / N_t          (equal weight across ETFs with a position at t)
    gross_t+1  = sum_i w_i,t * x_i,t+1        (x = monthly excess return, r - rf)
    turnover_t = sum_i |w_i,t - w_i,t-1|      (w_t-1 = 0 for ETFs entering)
    net_t+1    = gross_t+1 - cost_bp / 1e4 * turnover_t

Excess returns: a long financed at the risk-free rate earns r - rf; a short earns
-(r - rf) because the sale proceeds earn rf. So the portfolio return is an excess return.

Samples are defined by return month: in = sample_start+1 .. in_sample_end,
out = out_sample_start .. sample_end. The out-of-sample run happens once and needs --confirm-oos.
Run: uv run python src/backtest.py [--sample in|out] [--confirm-oos]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from config import STRATEGY
from db import ROOT


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Long signals frame (ticker, month, ...) -> wide (monthly PeriodIndex x ticker)."""
    w = df.pivot(index="month", columns="ticker", values=col)
    w.index = pd.PeriodIndex(w.index, freq="M")
    return w.sort_index()


def weights(positions: pd.DataFrame) -> pd.DataFrame:
    return positions.div(positions.notna().sum(axis=1), axis=0)


def _held(positions: pd.DataFrame, monthly_excess: pd.DataFrame, holding: int):
    """Held weights after each formation month and the next month's excess returns.
    holding > 1: overlapping portfolios (Jegadeesh–Titman; MOP 2012): the weights held after
    month t are the average of the weights formed at t, t-1, ..., t-holding+1."""
    months = pd.period_range(positions.index.min(), positions.index.max(), freq="M")
    w_v = weights(positions.reindex(months))  # one "vintage" per formation month
    w0 = sum(w_v.shift(k).fillna(0.0) for k in range(holding)) / holding
    held = sum(w_v.shift(k).notna().astype(int) for k in range(holding)) > 0
    x_next = monthly_excess.reindex(index=months + 1, columns=w_v.columns).set_axis(months)
    missing = held & x_next.isna()
    if missing.any().any():
        bad = missing.stack()
        raise ValueError(f"positions without next-month return: {bad[bad].index[:5].tolist()}")
    return months, w0, held, x_next.fillna(0.0)


def backtest(
    positions: pd.DataFrame,
    monthly_excess: pd.DataFrame,
    cost_bp: float = STRATEGY["cost_bp"],
    holding: int = STRATEGY["holding_months"],
) -> pd.DataFrame:
    """Positions indexed by formation month -> frame indexed by return month with
    gross_ret, cost, net_ret, turnover, n_assets. The first row's turnover is the initial
    build (sum |w|). Raises if a position has no next-month return."""
    months, w0, held, x_next = _held(positions, monthly_excess, holding)
    out = pd.DataFrame(
        {
            "gross_ret": (w0 * x_next).sum(axis=1),
            "turnover": (w0 - w0.shift(1, fill_value=0.0)).abs().sum(axis=1),
            "n_assets": held.sum(axis=1),
        }
    )
    out["cost"] = cost_bp / 1e4 * out.turnover
    out["net_ret"] = out.gross_ret - out.cost
    out.index = months + 1  # return month
    out.index.name = "month"
    return out[["gross_ret", "cost", "net_ret", "turnover", "n_assets"]]


def contributions(
    positions: pd.DataFrame,
    monthly_excess: pd.DataFrame,
    cost_bp: float = 0.0,  # unused; same signature as backtest for run_sample
    holding: int = STRATEGY["holding_months"],
) -> pd.DataFrame:
    """Each ETF's share of the monthly gross return (w x next-month excess return),
    indexed by return month. Rows sum to backtest()'s gross_ret."""
    months, w0, _, x_next = _held(positions, monthly_excess, holding)
    out = w0 * x_next
    out.index = months + 1
    out.index.name = "month"
    return out


def sample_window(sample: str) -> tuple[pd.Period, pd.Period]:
    """First and last *return* month of a sample."""
    if sample == "in":
        return (
            pd.Period(STRATEGY["sample_start"], "M") + 1,
            pd.Period(STRATEGY["in_sample_end"][:7], "M"),
        )
    return (
        pd.Period(STRATEGY["out_sample_start"][:7], "M"),
        pd.Period(STRATEGY["sample_end"], "M"),
    )


def run_sample(
    sig: pd.DataFrame,
    monthly_excess: pd.DataFrame,
    sample: str,
    cost_bp: float = STRATEGY["cost_bp"],
    holding: int = STRATEGY["holding_months"],
    fn=backtest,
) -> pd.DataFrame:
    """Backtest one sample (fn=contributions gives per-ETF contributions instead).
    Out-of-sample starts from the last in-sample weights, so its
    first month's turnover is a real rebalance, not a fresh build. With holding > 1, the
    earlier vintages needed for the first return month are included, then dropped."""
    first, last = sample_window(sample)
    pos = wide(sig, "position")
    lead = (0 if sample == "in" else 1) + holding - 1
    res = fn(pos.loc[first - 1 - lead : last - 1], monthly_excess, cost_bp, holding)
    return res.loc[first:last]


def summary(res: pd.DataFrame) -> dict:
    def ann(r):
        return {
            "mean": 12 * r.mean(),
            "vol": np.sqrt(12) * r.std(),
            "sharpe": np.sqrt(12) * r.mean() / r.std(),
        }

    return {
        "months": len(res),
        "first": str(res.index[0]),
        "last": str(res.index[-1]),
        "gross": ann(res.gross_ret),
        "net": ann(res.net_ret),
        "mean_turnover": res.turnover.iloc[1:].mean(),
        "mean_n_assets": res.n_assets.mean(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["in", "out"], default="in")
    ap.add_argument(
        "--confirm-oos",
        action="store_true",
        help="required for --sample out: the single out-of-sample run ",
    )
    args = ap.parse_args()
    if args.sample == "out" and not args.confirm_oos:
        sys.exit("refusing: the out-of-sample run happens once; pass --confirm-oos")

    from db import init_db, log_run
    from signals import build, load_inputs, monthly_excess, write_signals

    conn = init_db()
    mret, rf, daily_vol = load_inputs(conn)
    sig = build(mret, rf, daily_vol)
    mex = monthly_excess(mret, rf)
    res = run_sample(sig, mex, args.sample)

    first, last = sample_window(args.sample)
    run_id = log_run(conn, {"step": "backtest", **STRATEGY}, sample=args.sample)
    formed = sig[(sig.month >= str(first - 1)) & (sig.month <= str(last - 1))]
    write_signals(conn, run_id, formed)
    conn.executemany(
        "INSERT INTO portfolio_returns (run_id, month, gross_ret, cost, net_ret, turnover) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(run_id, str(m), r.gross_ret, r.cost, r.net_ret, r.turnover) for m, r in res.iterrows()],
    )
    conn.commit()
    path = ROOT / "results" / f"portfolio_returns_{args.sample}.csv"
    res.assign(run_id=run_id).to_csv(path)
    s = summary(res)
    print(
        f"run_id {run_id} ({args.sample}): {s['months']} return months, {s['first']} .. {s['last']}"
    )
    for k in ("gross", "net"):
        a = s[k]
        print(f"  {k}: mean {a['mean']:.2%}/yr, vol {a['vol']:.2%}/yr, Sharpe {a['sharpe']:.2f}")
    print(
        f"  mean monthly turnover {s['mean_turnover']:.3f}, mean ETFs held {s['mean_n_assets']:.1f}"
    )
    print(f"  wrote {path.relative_to(ROOT)}")
