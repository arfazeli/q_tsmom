"""Benchmarks, diversification, per-asset-class results and breadth (descriptive).

Computed AFTER the out-of-sample test, from the baseline configuration with no
tuning, so everything here is descriptive, not a new test. Writes results/benchmarks.csv,
benchmarks_diff.csv, combination.csv, class_standalone.csv, breadth.csv.

Run: uv run python src/benchmarks.py
"""

import numpy as np
import pandas as pd

from stats import paired_sharpe_diff, perf, sharpe_ci

W_OVERLAY = 0.25  # headline overlay weight, fixed before computing; others are descriptive
OVERLAY_GRID = (0.10, 0.25, 0.50)
STOCK_WEIGHT = 0.60  # 60/40 benchmark: 60% US stocks (Mkt-RF), 40% IEF


def effective_n(corr: pd.DataFrame) -> float:
    """Participation ratio of the correlation eigenvalues: N for uncorrelated assets,
    1 for perfectly correlated ones."""
    ev = np.linalg.eigvalsh(corr.to_numpy())
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (ev**2).sum())


def always_long(sig: pd.DataFrame) -> pd.DataFrame:
    """Same sizing, cap and equal weighting as the strategy, but always long: a passive
    risk-scaled benchmark. |position| = clip(target / vol, cap) whatever the signal."""
    out = sig.copy()
    out["position"] = out.position.abs()
    return out


def long_only_trend(sig: pd.DataFrame) -> pd.DataFrame:
    """Same as the strategy but never short: positions below zero are set to zero (cash)."""
    out = sig.copy()
    out["position"] = out.position.clip(lower=0)
    return out


def sixty_forty(mkt_rf: pd.Series, bond_excess: pd.Series) -> pd.Series:
    return STOCK_WEIGHT * mkt_rf + (1 - STOCK_WEIGHT) * bond_excess


def metrics(r: pd.Series) -> dict:
    p = perf(r)
    return {
        k: p[k]
        for k in ("months", "ann_mean", "ann_vol", "sharpe", "nw_tstat_mean", "max_drawdown")
    }


def class_standalone(sig, mex, assets, sample, run_sample):
    """Strategy run on one asset class at a time: equal weight within the class."""
    rows = []
    for cls, tickers in assets.groupby("asset_class").ticker:
        s = sig[sig.ticker.isin(tickers)]
        if s.empty:
            continue
        r = run_sample(s, mex, sample).net_ret
        r = r.loc[r.index >= s.month.min()]  # drop months before the class has any position
        ci = (
            sharpe_ci(r, B=5000)
            if len(r) > 24
            else {"sharpe_ci_low": np.nan, "sharpe_ci_high": np.nan}
        )
        m = metrics(r) if len(r) > 2 else {}
        rows.append(
            {
                "sample": sample,
                "asset_class": cls,
                "n_etfs": len(tickers),
                **m,
                "sharpe_ci_low": ci["sharpe_ci_low"],
                "sharpe_ci_high": ci["sharpe_ci_high"],
                "hit_rate": float((r > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from backtest import run_sample
    from db import ROOT, connect
    from signals import build, load_inputs, monthly_excess
    from stats import load_factors

    conn = connect()
    mret, rf, daily_vol = load_inputs(conn)
    mex = monthly_excess(mret, rf)
    sig = build(mret, rf, daily_vol)
    fac = load_factors(conn)
    assets = pd.read_sql("SELECT ticker, asset_class FROM assets", conn)
    out = ROOT / "results"

    # the strategy as run (must reproduce the stored baseline runs)
    stored = {
        s: pd.read_csv(out / f"portfolio_returns_{s}.csv", index_col=0).net_ret
        for s in ("in", "out")
    }

    bench_rows, diff_rows, comb_rows, class_rows = [], [], [], []
    series = {}
    for sample in ("in", "out"):
        strat = run_sample(sig, mex, sample).net_ret
        assert np.allclose(strat.to_numpy(), stored[sample].to_numpy(), atol=1e-12), sample
        idx = strat.index
        mkt = fac.mkt_rf.reindex(idx)
        bonds = mex["IEF"].reindex(idx)
        port = {
            "Strategy (net)": strat,
            "US stocks": mkt,
            "60/40 stocks/bonds": sixty_forty(mkt, bonds),
            "Always long, same sizing": run_sample(always_long(sig), mex, sample).net_ret,
            "Long-only trend": run_sample(long_only_trend(sig), mex, sample).net_ret,
        }
        series[sample] = port
        for name, r in port.items():
            bench_rows.append(
                {
                    "sample": sample,
                    "portfolio": name,
                    **metrics(r),
                    "corr_with_strategy": float(np.corrcoef(r, strat)[0, 1]),
                }
            )
        for name in list(port)[1:]:
            d = paired_sharpe_diff(strat, port[name])
            diff_rows.append(
                {
                    "sample": sample,
                    "strategy_minus": name,
                    **{
                        k: d[k]
                        for k in (
                            "months",
                            "delta_sharpe",
                            "delta_ci_low",
                            "delta_ci_high",
                            "p_one_sided",
                        )
                    },
                }
            )
        class_rows.append(class_standalone(sig, mex, assets, sample, run_sample))

    # overlay: 60/40 + the strategy scaled to the 60/40's in-sample volatility (constant k)
    base_in, strat_in = series["in"]["60/40 stocks/bonds"], series["in"]["Strategy (net)"]
    k = float(base_in.std() / strat_in.std())
    for sample in ("in", "out"):
        base, strat = series[sample]["60/40 stocks/bonds"], series[sample]["Strategy (net)"]
        for w in (0.0, *OVERLAY_GRID):
            comb = (1 - w) * base + w * k * strat
            d = paired_sharpe_diff(comb, base) if w > 0 else {}
            comb_rows.append(
                {
                    "sample": sample,
                    "overlay_weight": w,
                    "scale_k": k,
                    "headline": w == W_OVERLAY,
                    **metrics(comb),
                    "delta_sharpe_vs_6040": d.get("delta_sharpe", 0.0),
                    "delta_ci_low": d.get("delta_ci_low", np.nan),
                    "delta_ci_high": d.get("delta_ci_high", np.nan),
                    "p_one_sided": d.get("p_one_sided", np.nan),
                }
            )

    # breadth: effective number of independent bets among the 27 ETFs (all present)
    br = []
    for name, (a, b) in {
        "2009-2015": ("2009-01", "2015-12"),
        "2016-2026": ("2016-01", "2026-09"),
    }.items():
        d = mex.loc[a:b].dropna(axis=1)
        c = d.corr()
        off = c.to_numpy()[np.triu_indices_from(c, 1)]
        br.append(
            {
                "period": name,
                "scope": "all",
                "etfs": d.shape[1],
                "avg_pairwise_corr": float(off.mean()),
                "effective_n": effective_n(c),
            }
        )
        for cls, tk in assets.groupby("asset_class").ticker:
            tk = [t for t in tk if t in d]
            if len(tk) > 1:
                cc = d[tk].corr()
                br.append(
                    {
                        "period": name,
                        "scope": cls,
                        "etfs": len(tk),
                        "avg_pairwise_corr": float(
                            cc.to_numpy()[np.triu_indices_from(cc, 1)].mean()
                        ),
                        "effective_n": effective_n(cc),
                    }
                )

    pd.DataFrame(bench_rows).to_csv(out / "benchmarks.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(out / "benchmarks_diff.csv", index=False)
    pd.DataFrame(comb_rows).to_csv(out / "combination.csv", index=False)
    pd.concat(class_rows).to_csv(out / "class_standalone.csv", index=False)
    pd.DataFrame(br).to_csv(out / "breadth.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(pd.DataFrame(bench_rows).to_string(index=False))
    print(pd.DataFrame(diff_rows).to_string(index=False))
    print(f"overlay scale k = {k:.2f}")
    print(pd.DataFrame(comb_rows).drop(columns=["scale_k", "months"]).to_string(index=False))
    print(pd.concat(class_rows).to_string(index=False))
    print(pd.DataFrame(br).to_string(index=False))
