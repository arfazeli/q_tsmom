"""Market regime labels built from past data only.

Each label is indexed by the month it APPLIES to (the return month m) and uses only data
up to the last close of month m-1, i.e. what is known when positions for month m are
formed. Market = SPY (adjusted close) from our own data.

- bear: compounded market return over the 24 months ending at m-1 is negative
  (Daniel and Moskowitz 2016).
- high_vol: realized daily volatility of the market in month m-1 (annualized) is above the
  80th percentile of all monthly realized volatilities up to and including m-1
  (expanding window, at least 36 months of history; NaN before that).

Run: uv run python src/regimes.py   (writes in-sample labels to results/regimes_in.csv)
"""

import numpy as np
import pandas as pd

from config import STRATEGY

MARKET = "SPY"
BEAR_MONTHS = 24
VOL_PCTL, VOL_MIN_MONTHS = 0.80, 36


def labels(
    market_prices: pd.Series, ann_days: int = STRATEGY["annualization_days"]
) -> pd.DataFrame:
    p = market_prices.dropna()
    daily = p.pct_change().dropna()
    month = daily.index.to_period("M")
    rvol = daily.groupby(month).std() * np.sqrt(ann_days)  # realized vol of month t
    last = p.groupby(p.index.to_period("M")).last()
    mret = last.pct_change()
    cum24 = np.expm1(np.log1p(mret).rolling(BEAR_MONTHS, min_periods=BEAR_MONTHS).sum())
    pctl = rvol.expanding(min_periods=VOL_MIN_MONTHS).quantile(VOL_PCTL)
    known_at_t = pd.DataFrame(
        {
            "market_ret_24m": cum24,
            "market_rvol": rvol,
            "rvol_pctl80": pctl,
        }
    )
    known_at_t["bear"] = (known_at_t.market_ret_24m < 0).where(known_at_t.market_ret_24m.notna())
    known_at_t["high_vol"] = (known_at_t.market_rvol > known_at_t.rvol_pctl80).where(
        known_at_t.rvol_pctl80.notna()
    )
    out = known_at_t.copy()
    out.index = out.index + 1  # applies to the NEXT month
    out.index.name = "month"
    return out


if __name__ == "__main__":
    import returns
    from db import ROOT, connect

    prices = returns.load_prices(connect())[MARKET]
    lab = labels(prices).loc[: STRATEGY["in_sample_end"][:7]]  # in-sample only
    lab.to_csv(ROOT / "results" / "regimes_in.csv")
    ins = lab.loc[pd.Period(STRATEGY["sample_start"], "M") + 1 :]
    print(f"in-sample return months {ins.index[0]}..{ins.index[-1]}: {len(ins)}")
    print(
        f"  bear: {int(ins.bear.sum())} months, high_vol: {int(ins.high_vol.sum())} months, "
        f"both: {int((ins.bear.astype(bool) & ins.high_vol.astype(bool)).sum())}"
    )
    print(
        "  bear months:",
        ", ".join(str(m) for m in ins.index[ins.bear.astype(bool)][:3]),
        "...",
        ", ".join(str(m) for m in ins.index[ins.bear.astype(bool)][-2:]),
    )
