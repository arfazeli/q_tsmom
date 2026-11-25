"""Strategy performance by regime. Descriptive only: uses the stored baseline runs
(latest clean in-sample backtest; the single out-of-sample run) and changes nothing.

Run: uv run python src/regime_analysis.py
"""

import numpy as np
import pandas as pd

from stats import nw_tstat


def regime_table(r: pd.Series, label: pd.Series, name: str) -> list[dict]:
    """Performance of r in months where label is True / False. Newey–West t uses the
    months in date order; within a regime they are not consecutive, so treat it as rough."""
    lab = label.reindex(r.index)
    rows = []
    for value in (True, False):
        x = r[lab == value]
        rows.append(
            {
                "regime": name,
                "in_regime": value,
                "months": len(x),
                "ann_mean": 12 * x.mean() if len(x) else np.nan,
                "ann_vol": np.sqrt(12) * x.std() if len(x) > 1 else np.nan,
                "sharpe": np.sqrt(12) * x.mean() / x.std() if len(x) > 1 else np.nan,
                "nw_t": nw_tstat(x) if len(x) > 2 else np.nan,
                "hit_rate": (x > 0).mean() if len(x) else np.nan,
            }
        )
    return rows


def quintiles(r: pd.Series, mkt: pd.Series) -> pd.DataFrame:
    """Mean strategy return by quintile of the same month's market excess return."""
    df = pd.concat([r.rename("strategy"), mkt.rename("market")], axis=1, join="inner")
    df["quintile"] = pd.qcut(df.market, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return (
        df.groupby("quintile")
        .agg(
            months=("market", "size"),
            market_mean=("market", "mean"),
            strategy_mean=("strategy", "mean"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    import returns
    from backtest import contributions, run_sample
    from db import ROOT, connect
    from regimes import MARKET, labels
    from signals import build, load_inputs, monthly_excess
    from stats import load_factors, load_run

    conn = connect()
    lab = labels(returns.load_prices(conn)[MARKET])
    mkt = load_factors(conn).mkt_rf
    mret, rf, daily_vol = load_inputs(conn)
    sig, mex = build(mret, rf, daily_vol), monthly_excess(mret, rf)
    assets = pd.read_sql("SELECT ticker, asset_class FROM assets", conn).set_index("ticker")

    perf_rows, quint, worst, contrib = [], [], [], []
    for sample in ("in", "out"):
        run_id, pr = load_run(conn, sample)
        r = pr.net_ret
        for name in ("bear", "high_vol"):
            for row in regime_table(r, lab[name].astype("boolean"), name):
                perf_rows.append({"sample": sample, "run_id": run_id, **row})
        q = quintiles(r, mkt)
        q.insert(0, "sample", sample)
        quint.append(q)
        w = pd.concat([mkt.rename("market"), r.rename("strategy")], axis=1, join="inner")
        w = w.nsmallest(10, "market").reset_index().rename(columns={"index": "month"})
        w.insert(0, "sample", sample)
        worst.append(w)
        c = run_sample(sig, mex, sample, fn=contributions)
        gap = float((c.sum(axis=1) - pr.gross_ret).abs().max())
        assert gap < 1e-12, f"contributions don't match stored run {run_id}: {gap}"
        by_class = c.T.groupby(assets.asset_class).sum().T
        yearly = by_class.groupby(by_class.index.year).sum()  # additive contributions
        yearly["cost"] = -pr.cost.groupby(pr.index.year).sum()
        yearly["net_total"] = yearly.sum(axis=1)
        yearly.insert(0, "sample", sample)
        contrib.append(yearly)

    out = ROOT / "results"
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(out / "regime_perf.csv", index=False)
    pd.concat(quint).to_csv(out / "market_quintiles.csv", index=False)
    pd.concat(worst).to_csv(out / "worst_market_months.csv", index=False)
    cb = pd.concat(contrib)
    cb.index.name = "year"
    cb.to_csv(out / "class_contrib_by_year.csv")
    pd.set_option("display.width", 180)
    print(perf.round(3).to_string(index=False))
    print(pd.concat(quint).round(4).to_string(index=False))
    print(pd.concat(worst).round(4).to_string(index=False))
    print((cb.drop(columns="sample") * 100).round(2).assign(sample=cb["sample"]).to_string())
