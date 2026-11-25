"""Volatility-regime conditioning (fixed in advance, not tuned).

Conditioned strategy = baseline with every position for month m multiplied by
K = 0.5 when the `high_vol` label for month m is true (known at the end of m-1).
H1: its net Sharpe beats the baseline's out of sample. Test: paired stationary block
bootstrap of the Sharpe difference; supported iff delta > 0 and one-sided p < 0.05.

Run: uv run python src/vol_conditioning.py [--sample in|out] [--confirm-oos]
"""

import argparse
import sys

import pandas as pd

from config import STRATEGY

K = 0.5  # fixed in advance, not tuned


def conditioned(sig: pd.DataFrame, high_vol: pd.Series, k: float = K) -> pd.DataFrame:
    """Scale positions formed at month t by k when the label for month t+1 is high_vol.
    Missing labels (warm-up) count as not high-vol."""
    hv = high_vol.fillna(False).astype(bool)
    applies = pd.PeriodIndex(sig.month, freq="M") + 1
    scale = pd.Series(applies).map(hv).fillna(False).map({True: k, False: 1.0}).to_numpy()
    out = sig.copy()
    out["position"] = out.position * scale
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["in", "out"], default="in")
    ap.add_argument("--confirm-oos", action="store_true")
    args = ap.parse_args()
    if args.sample == "out" and not args.confirm_oos:
        sys.exit("refusing: the out-of-sample run happens once; pass --confirm-oos")

    import returns
    from backtest import run_sample
    from db import ROOT, init_db, log_run
    from regimes import MARKET, labels
    from signals import build, load_inputs, monthly_excess
    from stats import max_drawdown, paired_sharpe_diff

    conn = init_db()
    mret, rf, daily_vol = load_inputs(conn)
    mex = monthly_excess(mret, rf)
    sig = build(mret, rf, daily_vol)
    lab = labels(returns.load_prices(conn)[MARKET])
    base = run_sample(sig, mex, args.sample)
    cond = run_sample(conditioned(sig, lab.high_vol), mex, args.sample)

    run_id = log_run(conn, {**STRATEGY, "step": "vol_conditioning", "k": K}, sample=args.sample)
    conn.executemany(
        "INSERT INTO portfolio_returns (run_id, month, gross_ret, cost, net_ret, turnover) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(run_id, str(m), r.gross_ret, r.cost, r.net_ret, r.turnover) for m, r in cond.iterrows()],
    )
    conn.commit()
    test = paired_sharpe_diff(cond.net_ret, base.net_ret)
    hv_months = int(lab.high_vol.reindex(cond.index).fillna(False).astype(bool).sum())
    out = {
        "sample": args.sample,
        "run_id": run_id,
        "k": K,
        "high_vol_months": hv_months,
        **test,
        "supported": bool(test["delta_sharpe"] > 0 and test["p_one_sided"] < 0.05),
        "maxdd_cond": max_drawdown(cond.net_ret),
        "maxdd_base": max_drawdown(base.net_ret),
        "worst_month_cond": cond.net_ret.min(),
        "worst_month_base": base.net_ret.min(),
        "ann_cost_cond": 12 * cond.cost.mean(),
        "ann_cost_base": 12 * base.cost.mean(),
    }
    pd.Series(out).to_csv(
        ROOT / "results" / f"vol_conditioning_{args.sample}.csv", header=["value"]
    )
    print(pd.Series(out).to_string())
