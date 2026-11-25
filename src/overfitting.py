"""Overfitting control: in-sample sensitivity grid and the probabilistic and
deflated Sharpe ratios (Bailey and López de Prado 2012, 2014).

Run: uv run python src/overfitting.py   (in-sample only; logs one run per grid cell)
"""

import numpy as np
import pandas as pd
from scipy import stats as sps

from config import STRATEGY
from db import ROOT

EULER_GAMMA = 0.5772156649015329
LOOKBACKS = [1, 3, 6, 9, 12, 24]
HOLDINGS = [1, 3, 6]
COSTS_BP = [0, 2.5, 5, 10, 15, 20, 25]  # cost sensitivity range


def psr(sr: float, sr_star: float, T: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio: P(true SR > sr_star). sr and sr_star are per period
    (monthly here, not annualized); kurt is plain kurtosis (3 for a normal)."""
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(sps.norm.cdf((sr - sr_star) * np.sqrt(T - 1) / denom))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """Expected maximum of n_trials Sharpe estimates whose true value is 0 and whose
    variance across trials is var_sr (per period):
    sqrt(V) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))."""
    z = sps.norm.ppf
    return float(
        np.sqrt(var_sr)
        * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials) + EULER_GAMMA * z(1 - 1 / (n_trials * np.e)))
    )


def dsr(sr: float, T: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> float:
    """Deflated Sharpe ratio: PSR against the Sharpe ratio expected from the best of
    n_trials variants by luck alone."""
    return psr(sr, expected_max_sr(n_trials, var_sr), T, skew, kurt)


def monthly_sr(r: pd.Series) -> float:
    return float(r.mean() / r.std())


def grid(sig_by_lookback: dict, mex: pd.DataFrame, run_sample) -> pd.DataFrame:
    """Net and gross Sharpe for every lookback x holding cell, base cost (in-sample)."""
    rows = []
    for lb in LOOKBACKS:
        for h in HOLDINGS:
            res = run_sample(sig_by_lookback[lb], mex, "in", STRATEGY["cost_bp"], h)
            rows.append({"lookback": lb, "holding": h, "res": res})
    return rows


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from backtest import run_sample
    from db import init_db, log_run
    from signals import build, load_inputs, monthly_excess

    conn = init_db()
    mret, rf, daily_vol = load_inputs(conn)
    mex = monthly_excess(mret, rf)
    sig = {lb: build(mret, rf, daily_vol, lookback=lb) for lb in LOOKBACKS}

    def log(res, **cfg):
        run_id = log_run(conn, {**STRATEGY, **cfg, "step": "sensitivity"}, sample="in")
        conn.executemany(
            "INSERT INTO portfolio_returns (run_id, month, gross_ret, cost, net_ret, turnover) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (run_id, str(m), r.gross_ret, r.cost, r.net_ret, r.turnover)
                for m, r in res.iterrows()
            ],
        )
        return run_id

    # T13a: lookback x holding grid at the base cost
    rows, returns = [], {}
    for cell in grid(sig, mex, run_sample):
        res, lb, h = cell["res"], cell["lookback"], cell["holding"]
        run_id = log(res, lookback_months=lb, holding_months=h)
        returns[(lb, h)] = res.net_ret
        r = res.net_ret
        rows.append(
            {
                "lookback": lb,
                "holding": h,
                "run_id": run_id,
                "months": len(r),
                "net_sharpe": np.sqrt(12) * monthly_sr(r),
                "gross_sharpe": np.sqrt(12) * monthly_sr(res.gross_ret),
                "net_ann_mean": 12 * r.mean(),
                "mean_turnover": res.turnover.iloc[1:].mean(),
                "mean_n_assets": res.n_assets.mean(),
            }
        )
    g = pd.DataFrame(rows)
    g.to_csv(ROOT / "results" / "sensitivity_grid_in.csv", index=False)

    # T13b: cost sensitivity for the baseline (12-month lookback, 1-month hold)
    base_lb, base_h = STRATEGY["lookback_months"], STRATEGY["holding_months"]
    crow = []
    for c in COSTS_BP:
        res = run_sample(sig[base_lb], mex, "in", c, base_h)
        run_id = log(res, cost_bp=c)
        crow.append(
            {
                "cost_bp": c,
                "run_id": run_id,
                "net_sharpe": np.sqrt(12) * monthly_sr(res.net_ret),
                "net_ann_mean": 12 * res.net_ret.mean(),
                "ann_cost": 12 * res.cost.mean(),
            }
        )
    costs = pd.DataFrame(crow)
    costs.to_csv(ROOT / "results" / "cost_sensitivity_in.csv", index=False)

    # PSR and DSR for the baseline, deflated by the number of grid variants tried
    r = returns[(base_lb, base_h)]
    sr, T = monthly_sr(r), len(r)
    sk, ku = float(sps.skew(r)), float(sps.kurtosis(r, fisher=False))
    srs = np.array([monthly_sr(x) for x in returns.values()])
    n, v = len(srs), float(srs.var(ddof=1))
    out = {
        "lookback": base_lb,
        "holding": base_h,
        "months": T,
        "sr_monthly": sr,
        "sr_annual": np.sqrt(12) * sr,
        "skew": sk,
        "kurtosis": ku,
        "psr_vs_0": psr(sr, 0.0, T, sk, ku),
        "n_trials": n,
        "var_sr_trials_monthly": v,
        "expected_max_sr_monthly": expected_max_sr(n, v),
        "expected_max_sr_annual": np.sqrt(12) * expected_max_sr(n, v),
        "dsr": dsr(sr, T, sk, ku, n, v),
        "best_cell": str(max(returns, key=lambda k: monthly_sr(returns[k]))),
        "best_cell_sr_annual": np.sqrt(12) * srs.max(),
    }
    pd.Series(out).to_csv(ROOT / "results" / "deflated_sharpe_in.csv", header=["value"])

    # Heatmap of net Sharpe (diverging around 0, two hues + neutral midpoint)
    hm = g.pivot(index="lookback", columns="holding", values="net_sharpe")
    lim = float(np.abs(hm.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    im = ax.imshow(hm.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(HOLDINGS)), [f"{h} mo" for h in HOLDINGS])
    ax.set_yticks(range(len(LOOKBACKS)), [f"{lb} mo" for lb in LOOKBACKS])
    ax.set_xlabel("Holding period")
    ax.set_ylabel("Lookback")
    for (i, j), val in np.ndenumerate(hm.to_numpy()):
        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if abs(val) > 0.6 * lim else "black",
        )
    bi, bj = LOOKBACKS.index(base_lb), HOLDINGS.index(base_h)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, lw=2, ec="black"))
    ax.set_title("Net Sharpe ratio, in-sample 2003-08..2015-12\n(box = baseline)", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Net Sharpe (annualized)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "sensitivity_heatmap_in.png")

    pd.set_option("display.width", 160)
    print(hm.round(2).to_string())
    print(costs.round(4).to_string(index=False))
    print(pd.Series(out).to_string())
