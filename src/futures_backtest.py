"""Futures replication, with the configuration fixed in advance: the identical ETF
configuration on the contracts that passed the roll-validity rule.

Variants: main (kept contracts, roll-adjusted); sensitivities (a) no roll adjustment,
(b) without crude oil, (c) all 29 contracts. Every variant is logged in `runs`.
Run: uv run python src/futures_backtest.py
"""

import numpy as np
import pandas as pd

import returns as etf_returns
from backtest import backtest, wide
from config import STRATEGY
from db import ROOT
from futures import index_from_returns, load_universe, process, read_snapshot
from signals import build, monthly_excess
from stats import perf, sharpe_ci
from volatility import ewma_vol

MIN_ASSETS, MIN_GROUPS = (
    10,
    4,
)  # sample starts at the first month with >= 10 contracts in >= 4 groups
END = pd.Period("2026-09", "M")
PERIODS = {
    "full": (None, None),
    "P1": (None, "2002-12"),
    "P2": ("2003-01", "2015-12"),
    "P3": ("2016-01", "2026-09"),
}


def first_sample_month(
    sig: pd.DataFrame, groups: dict, min_assets=MIN_ASSETS, min_groups=MIN_GROUPS
) -> pd.Period:
    """First formation month with enough contracts across enough groups."""
    d = (
        sig.assign(group=sig.ticker.map(groups))
        .groupby("month")
        .agg(n=("ticker", "nunique"), g=("group", "nunique"))
    )
    ok = d[(d.n >= min_assets) & (d.g >= min_groups)]
    return pd.Period(ok.index[0], "M")


def run_variant(rets: dict[str, pd.Series], groups: dict, cost_bp=STRATEGY["cost_bp"]):
    idx = index_from_returns(rets)
    daily = etf_returns.daily_returns(idx)
    vol = ewma_vol(daily)
    mret = etf_returns.monthly_returns(idx)
    zero_rf = pd.Series(0.0, index=mret.index)  # futures returns are already excess returns
    sig = build(mret, zero_rf, vol)
    start = first_sample_month(sig, groups)
    mex = monthly_excess(mret, zero_rf)
    pos = wide(sig, "position").loc[start : END - 1]
    # missing-month rule: a position whose next-month return does not exist
    # (a data gap, e.g. live cattle in January 2006) cannot be held and is dropped
    x_next = mex.reindex(index=pos.index + 1, columns=pos.columns).set_axis(pos.index)
    held_before = int(pos.notna().sum().sum())
    pos = pos.where(x_next.notna())
    dropped = held_before - int(pos.notna().sum().sum())
    res = backtest(pos, mex, cost_bp, 1)
    return res, sig, pos, mex, start, dropped


def period_slice(r: pd.Series, lo, hi) -> pd.Series:
    return r.loc[lo or r.index[0] : hi or r.index[-1]]


if __name__ == "__main__":
    from db import init_db, log_run

    uni = load_universe()
    groups = dict(zip(uni.ticker, uni.group, strict=True))
    bars = read_snapshot(uni)
    _, _, val, raw_ret, adj_ret = process(bars, uni)
    kept = list(val.ticker[val.kept])
    variants = {
        "main": {t: adj_ret[t] for t in kept},
        "no_roll_adjustment": {t: raw_ret[t] for t in kept},
        "without_crude": {t: adj_ret[t] for t in kept if t != "CL=F"},
        "all_29_contracts": dict(adj_ret),
    }
    conn = init_db()
    out = ROOT / "results"
    stat_rows, summary, by_group = [], {}, []
    etf = pd.concat(
        [
            pd.read_csv(out / f"portfolio_returns_{s}.csv", index_col=0).net_ret
            for s in ("in", "out")
        ]
    )
    etf.index = pd.PeriodIndex(etf.index, freq="M")

    for name, rets in variants.items():
        res, sig, pos, mex, start, dropped = run_variant(rets, groups)
        run_id = log_run(
            conn,
            {**STRATEGY, "step": "futures_replication", "variant": name, "contracts": sorted(rets)},
            sample="out",
        )
        conn.executemany(
            "INSERT INTO portfolio_returns (run_id, month, gross_ret, cost, net_ret, turnover) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (run_id, str(m), r.gross_ret, r.cost, r.net_ret, r.turnover)
                for m, r in res.iterrows()
            ],
        )
        conn.commit()
        for period, (lo, hi) in PERIODS.items():
            r = period_slice(res.net_ret, lo, hi)
            if len(r) < 6:
                continue
            ci = (
                sharpe_ci(r, B=5000)
                if len(r) >= 24
                else {"sharpe_ci_low": np.nan, "sharpe_ci_high": np.nan}
            )
            p = perf(r)
            stat_rows.append(
                {
                    "variant": name,
                    "period": period,
                    "run_id": run_id,
                    "contracts": len(rets),
                    **{
                        k: p[k]
                        for k in (
                            "months",
                            "first",
                            "last",
                            "ann_mean",
                            "ann_vol",
                            "sharpe",
                            "nw_tstat_mean",
                            "max_drawdown",
                            "hit_rate",
                        )
                    },
                    **ci,
                }
            )
        if name == "main":
            r = res.net_ret
            full = perf(r)
            used = sig[(sig.month >= str(start)) & (sig.month <= str(END - 1))]
            common = r.index.intersection(etf.index)
            summary = {
                "sample_first_formation_month": str(start),
                "return_months": len(r),
                "contracts_kept": len(rets),
                "groups": len(set(groups[t] for t in rets)),
                "mean_contracts_held": float(res.n_assets.mean()),
                "full_sharpe": full["sharpe"],
                "full_nw_t": full["nw_tstat_mean"],
                "H2_supported": bool(full["sharpe"] > 0 and full["nw_tstat_mean"] > 2),
                "cap_binds_share": float(
                    (used.position.abs() >= STRATEGY["leverage_cap"] - 1e-12).mean()
                ),
                "corr_with_etf_strategy": float(np.corrcoef(r.loc[common], etf.loc[common])[0, 1]),
                "common_months": len(common),
                "positions_dropped_missing_return": dropped,
                "run_id": run_id,
            }
            res.assign(run_id=run_id).to_csv(out / "futures_portfolio_returns.csv")
            for g in sorted(set(groups[t] for t in rets)):
                cols = [t for t in rets if groups[t] == g and t in pos.columns]
                rg = backtest(pos[cols], mex, STRATEGY["cost_bp"], 1).net_ret
                pg = perf(rg)
                by_group.append(
                    {
                        "group": g,
                        "contracts": len(cols),
                        "months": pg["months"],
                        "ann_mean": pg["ann_mean"],
                        "ann_vol": pg["ann_vol"],
                        "sharpe": pg["sharpe"],
                        "nw_tstat_mean": pg["nw_tstat_mean"],
                    }
                )

    stats = pd.DataFrame(stat_rows)
    stats.to_csv(out / "futures_stats.csv", index=False)
    pd.Series(summary).to_csv(out / "futures_summary.csv", header=["value"])
    pd.DataFrame(by_group).to_csv(out / "futures_by_group.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(pd.Series(summary).to_string())
    print(stats.drop(columns=["first", "last", "run_id"]).to_string(index=False))
    print(pd.DataFrame(by_group).to_string(index=False))
