"""Ex-ante EWMA volatility, as in Moskowitz, Ooi and Pedersen (2012):

    sigma_t^2 = ann_days * sum_i (1 - d) d^i (r_{t-1-i} - rbar_t)^2,   d / (1 - d) = com

i.e. the exponentially weighted variance around the exponentially weighted mean
(pandas `ewm(com).var(bias=True)`), annualized, then shifted one day so the value at
day t uses returns up to t-1 only.
"""

import numpy as np
import pandas as pd

from config import STRATEGY


def ewma_vol(
    daily_ret: pd.DataFrame,
    com: float = STRATEGY["vol_com_days"],
    ann_days: int = STRATEGY["annualization_days"],
    min_periods: int = 60,
) -> pd.DataFrame:
    """Annualized volatility per ticker. NaN until `min_periods` returns exist; the value
    on day t excludes day t's return. Each ticker uses only its own trading days."""
    out = {}
    for t, r in daily_ret.items():
        r = r.dropna()
        var = r.ewm(com=com, min_periods=min_periods).var(bias=True)
        out[t] = np.sqrt(var * ann_days).shift(1)
    return pd.DataFrame(out).reindex(daily_ret.index)
