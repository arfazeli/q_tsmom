"""Performance statistics, Newey–West t-stats, factor regressions and a stationary
block bootstrap CI for the Sharpe ratio. Inputs are monthly excess returns.

Run: uv run python src/stats.py   (uses the latest logged in-sample backtest run)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from db import ROOT


def nw_lags(T: int) -> int:
    """Newey–West (1994) rule of thumb: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def lo_se(sharpe_monthly: float, T: int) -> float:
    """Lo (2002) standard error of the monthly Sharpe ratio under i.i.d. returns:
    sqrt((1 + SR^2 / 2) / T)."""
    return float(np.sqrt((1 + sharpe_monthly**2 / 2) / T))


def max_drawdown(r: pd.Series) -> float:
    """Largest fall from a peak of the compounded series (1 + r).cumprod(), as a positive
    fraction. The series starts at 1 before the first month."""
    wealth = np.concatenate([[1.0], (1 + r.to_numpy()).cumprod()])
    return float(np.max(1 - wealth / np.maximum.accumulate(wealth)))


def nw_tstat(r: pd.Series, lags: int | None = None) -> float:
    """t-stat of the mean monthly return with Newey–West (Bartlett) standard errors."""
    lags = nw_lags(len(r)) if lags is None else lags
    fit = sm.OLS(r.to_numpy(), np.ones(len(r))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def perf(r: pd.Series) -> dict:
    T, mu, sd = len(r), r.mean(), r.std()
    sr_m = mu / sd
    return {
        "months": T,
        "first": str(r.index[0]),
        "last": str(r.index[-1]),
        "ann_mean": 12 * mu,
        "ann_vol": np.sqrt(12) * sd,
        "sharpe": np.sqrt(12) * sr_m,
        "sharpe_se_lo": np.sqrt(12) * lo_se(sr_m, T),
        "nw_tstat_mean": nw_tstat(r),
        "nw_lags": nw_lags(T),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
        "skew": float(sps.skew(r)),
        "excess_kurtosis": float(sps.kurtosis(r)),
    }


def factor_regression(
    r: pd.Series, factors: pd.DataFrame, cols: list[str], lags: int | None = None
) -> dict:
    """OLS of strategy excess returns on factors; HAC (Newey–West) t-stats."""
    df = pd.concat([r.rename("y"), factors[cols]], axis=1, join="inner").dropna()
    lags = nw_lags(len(df)) if lags is None else lags
    fit = sm.OLS(df.y, sm.add_constant(df[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {
        "months": len(df),
        "alpha_monthly": fit.params["const"],
        "alpha_ann": 12 * fit.params["const"],
        "alpha_t": fit.tvalues["const"],
        "r2": fit.rsquared,
    }
    for c in cols:
        out[f"beta_{c}"], out[f"t_{c}"] = fit.params[c], fit.tvalues[c]
    return out


def stationary_bootstrap_idx(
    T: int, mean_block: float, B: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis and Romano (1994): B resamples of length T. Each position starts a new block
    (uniform random start) with probability 1/mean_block, otherwise continues the previous
    block at the next index, wrapping around the end of the sample."""
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    new_block = rng.random((B, T)) < 1 / mean_block
    starts = rng.integers(0, T, (B, T))
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def sharpe_ci(
    r: pd.Series,
    B: int = 10_000,
    mean_block: float | None = None,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """Percentile CI for the annualized Sharpe ratio. Block-length rule: T^(1/3)."""
    x = r.to_numpy()
    mean_block = len(x) ** (1 / 3) if mean_block is None else mean_block
    samples = x[stationary_bootstrap_idx(len(x), mean_block, B, np.random.default_rng(seed))]
    sr = np.sqrt(12) * samples.mean(axis=1) / samples.std(axis=1, ddof=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(sr, [a, 1 - a])
    return {"sharpe_ci_low": lo, "sharpe_ci_high": hi, "boot_mean_block": mean_block, "boot_B": B}


def paired_sharpe_diff(
    r_a: pd.Series, r_b: pd.Series, B: int = 10_000, mean_block: float | None = None, seed: int = 0
) -> dict:
    """Annualized Sharpe(a) - Sharpe(b) on the same months, with a paired stationary block
    bootstrap (the same resampled months for both series). One-sided p-value for
    H1: diff > 0 is the share of bootstrap differences <= 0."""
    a, b = r_a.align(r_b, join="inner")
    xa, xb = a.to_numpy(), b.to_numpy()
    mean_block = len(xa) ** (1 / 3) if mean_block is None else mean_block
    idx = stationary_bootstrap_idx(len(xa), mean_block, B, np.random.default_rng(seed))

    def sr(x):
        return np.sqrt(12) * x.mean(axis=-1) / x.std(axis=-1, ddof=1)

    boot = sr(xa[idx]) - sr(xb[idx])
    return {
        "months": len(xa),
        "sharpe_a": float(sr(xa)),
        "sharpe_b": float(sr(xb)),
        "delta_sharpe": float(sr(xa) - sr(xb)),
        "p_one_sided": float((boot <= 0).mean()),
        "delta_ci_low": float(np.quantile(boot, 0.025)),
        "delta_ci_high": float(np.quantile(boot, 0.975)),
        "boot_mean_block": mean_block,
        "boot_B": B,
    }


def load_run(conn, sample: str = "in") -> tuple[int, pd.DataFrame]:
    run_id = conn.execute(
        "SELECT MAX(run_id) FROM runs WHERE sample = ? "
        "AND json_extract(config_json, '$.step') = 'backtest'",
        (sample,),
    ).fetchone()[0]
    df = pd.read_sql(
        "SELECT * FROM portfolio_returns WHERE run_id = ? ORDER BY month", conn, params=(run_id,)
    )
    df.index = pd.PeriodIndex(df.pop("month"), freq="M")
    return run_id, df


def load_factors(conn) -> pd.DataFrame:
    f = pd.read_sql("SELECT * FROM factors_monthly", conn)
    f.index = pd.PeriodIndex(f.pop("month"), freq="M")
    return f


if __name__ == "__main__":
    import argparse

    from db import connect

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["in", "out"], default="in")
    sample = ap.parse_args().sample
    conn = connect()
    run_id, pr = load_run(conn, sample)
    fac = load_factors(conn)
    series = {"gross": pr.gross_ret, "net": pr.net_ret}
    if sample == "in":  # robustness sub-period
        series["net_2009_2015"] = pr.net_ret.loc["2009-01":"2015-12"]
    rows = []
    for name, r in series.items():
        rows.append({"series": name, "run_id": run_id, **perf(r), **sharpe_ci(r)})
    st = pd.DataFrame(rows).set_index("series")
    models = {"market": ["mkt_rf"], "ff5_umd": ["mkt_rf", "smb", "hml", "rmw", "cma", "umd"]}
    reg = pd.DataFrame(
        [
            {
                "model": m,
                "series": "net",
                "run_id": run_id,
                **factor_regression(pr.net_ret, fac, cols),
            }
            for m, cols in models.items()
        ]
    ).set_index("model")
    st.to_csv(ROOT / "results" / f"stats_{sample}.csv")
    reg.to_csv(ROOT / "results" / f"regressions_{sample}.csv")
    pd.set_option("display.width", 160)
    print(st.T.to_string())
    print(reg.T.to_string())
