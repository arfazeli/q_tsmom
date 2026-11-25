"""Futures extension: dirty-data handling for Yahoo's continuous front-month series.

Yahoo's "=F" series stitch the nearest contract together WITHOUT adjustment: on a roll day
the series jumps from the old contract's price to the new one's, a gap that is not a return.
Steps (data only; no strategy result is computed here):

1. Download into a frozen snapshot (data/raw_snapshot/futures/) and load into SQLite.
2. Quality flags: non-positive prices (WTI, April 2020), one-day bad ticks (a 10x error in
   yen futures, December 2001), stale prices, big moves, missing volume.
3. Cleaning: bad ticks are dropped (the return spans two days); returns that involve a
   non-positive price are set to 0 and flagged (their P&L is ignored, a known limitation).
4. Roll detection. Prices barely show a roll (ES rolled with a +0.05% day), but volume does:
   the expiring contract's volume dips, then jumps when the series switches. Each year the
   K strongest months (K = rolls per year of the exchange cycle, from the universe file) are
   kept and the day with the strongest volume surge inside each is the roll day. Where volume
   is missing, a return-size score is used. Validity is checked by how stable the chosen
   months are from year to year, which random volume spikes would not be.
5. Adjustment: the return on a roll day is set to 0 (the gap is removed, along with that day's
   true return, a zero-mean error). Volume-free years use the same rule with the fallback.

Run: uv run python src/futures.py [--refresh]
"""

import argparse
import csv
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import yaml

from db import ROOT
from ingest import SNAPSHOT, _stamp

FUT_DIR = SNAPSHOT / "futures"
UNIVERSE_PATH = ROOT / "config" / "futures_universe.yaml"
BAD_TICK_MOVE = 0.25  # |return| above this and fully reversed next day = bad tick
STALE_DAYS = 5
BIG_MOVE = 0.15  # flagged for review, never removed
COLS = ["open", "high", "low", "close", "volume"]


def load_universe(path=UNIVERSE_PATH) -> pd.DataFrame:
    return pd.DataFrame(yaml.safe_load(open(path)))


# ---------------------------------------------------------------- download and storage
def download(universe: pd.DataFrame, snapshot=SNAPSHOT, refresh=False) -> list[str]:
    """One CSV per contract in snapshot/futures/. Returns tickers that failed."""
    import yfinance as yf

    out = snapshot / "futures"
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in universe.ticker if refresh or not (out / f"{t}.csv").exists()]
    if not todo:
        return []
    data = yf.download(
        todo,
        start="1999-01-01",
        end=datetime.now(UTC).strftime("%Y-%m-%d"),  # exclusive: no partial bar for today
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    failed = []
    for t in todo:
        df = data[t].dropna(how="all") if t in data.columns.get_level_values(0) else None
        if df is None or df.empty:
            failed.append(t)
            continue
        df = df.rename(columns=str.lower)[COLS]
        df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
        df.index.name = "date"
        df.to_csv(out / f"{t}.csv")
        _stamp(snapshot, f"futures/{t}.csv", "yahoo")
    return failed


def read_snapshot(universe: pd.DataFrame, snapshot=SNAPSHOT) -> dict[str, pd.DataFrame]:
    out = {}
    for t in universe.ticker:
        path = snapshot / "futures" / f"{t}.csv"
        if path.exists():
            out[t] = pd.read_csv(path, index_col="date", parse_dates=True)
    return out


def load_into_db(conn, bars: dict[str, pd.DataFrame], snapshot=SNAPSHOT) -> None:
    stamps = {r["file"]: r for r in csv.DictReader(open(snapshot / "MANIFEST.csv"))}
    for t, df in bars.items():
        meta = stamps[f"futures/{t}.csv"]
        rows = [
            (
                t,
                d.strftime("%Y-%m-%d"),
                *[None if pd.isna(x) else float(x) for x in r],
                meta["source"],
                meta["downloaded_at"],
            )
            for d, r in zip(df.index, df[COLS].to_numpy(), strict=True)
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO futures_raw (ticker, date, open, high, low, close, volume, "
            "source, downloaded_at) VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
    conn.commit()


# ---------------------------------------------------------------- quality and cleaning
def bad_ticks(close: pd.Series) -> pd.Series:
    """Days whose price is a one-day error: a move above BAD_TICK_MOVE that is fully
    reversed the next day (the round trip returns to within 20% of the first move)."""
    ret = close.pct_change()
    nxt = ret.shift(-1)
    reversed_ = ((1 + ret) * (1 + nxt) - 1).abs() < 0.2 * ret.abs()
    valid = (close > 0) & (close.shift(1) > 0)
    return (ret.abs() > BAD_TICK_MOVE) & reversed_ & valid


def quality_flags(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """One row per flagged day: (date, check, detail). Zero-volume days are only counted."""
    rows = []
    for d in close.index[~(close > 0)]:
        rows.append((d, "non_positive", f"close = {close[d]:.4f}"))
    bad = bad_ticks(close)
    for d in close.index[bad]:
        rows.append((d, "bad_tick", f"close = {close[d]:.6g} (one-day error, dropped)"))
    ret = close.pct_change()
    big = (ret.abs() > BIG_MOVE) & ~bad & (close > 0) & (close.shift(1) > 0)
    for d in close.index[big]:
        rows.append((d, "big_move", f"ret = {ret[d]:+.3f}"))
    run = (close.diff() != 0).cumsum()
    length = close.groupby(run).cumcount() + 1
    for d in close.index[length >= STALE_DAYS]:
        rows.append((d, "stale", f"unchanged for {length[d]} days"))
    return pd.DataFrame(rows, columns=["date", "check", "detail"])


def clean_returns(close: pd.Series) -> pd.Series:
    """Daily simple returns with bad ticks dropped and returns touching a non-positive
    price set to 0 (P&L ignored and flagged by quality_flags)."""
    c = close.copy()
    c[bad_ticks(c)] = np.nan
    c = c.dropna()
    ret = c.pct_change()
    undefined = (c <= 0) | (c.shift(1) <= 0)
    ret[undefined] = 0.0
    return ret.dropna()


# ---------------------------------------------------------------- roll detection
VARIANTS = ("median", "dipjump")
MIN_STABILITY = 0.70  # contracts below this Jaccard stability are excluded ("roll uncertain")
MAX_DOM_IQR = 8  # monthly contracts (k = 12): roll days must fall within 8 calendar days


def roll_scores(close: pd.Series, volume: pd.Series, variant: str = "median") -> pd.Series:
    """Roll signature in volume. Yahoo switches contracts differently across markets, so two
    scores are available and the better one per contract is chosen by validation:
    - "median": log of today's volume over the median of the previous 5 days (volume
      migrates to the next contract);
    - "dipjump": min(log(v_t / v_{t-1}), log(median_10 / v_{t-1})): yesterday unusually low
      (expiring contract) and today a big jump (next contract), so a news surge after a
      normal day scores near zero.
    Where volume is missing, a return-size score is used (|return| over its robust scale)."""
    v = volume.where(volume > 0)
    if variant == "median":
        surge = np.log(v / v.shift(1).rolling(5, min_periods=3).median())
    else:
        jump = np.log(v / v.shift(1))
        dip = np.log(v.shift(2).rolling(10, min_periods=5).median() / v.shift(1))
        surge = np.minimum(jump, dip)
    r = close.pct_change().abs()
    scale = r.shift(1).rolling(60, min_periods=20).median()
    fallback = np.log1p(r / scale.where(scale > 0))
    return surge.where(surge.notna(), fallback).replace([np.inf, -np.inf], np.nan)


def volume_years(volume: pd.Series) -> list[int]:
    """Years with (almost) complete volume data: where the volume scores are valid."""
    ok = (volume > 0).groupby(volume.index.year).mean()
    n = volume.groupby(volume.index.year).size()
    return [int(y) for y in ok.index[(ok >= 0.9) & (n >= 200)]]


def detect_rolls(
    close: pd.Series, volume: pd.Series, k: int, variant: str = "median"
) -> pd.DataFrame:
    """Roll days: in each year the k months with the strongest surge, and in each of them the
    strongest day. Years with fewer months of data get k scaled by their coverage.
    Returns a frame indexed by date with columns score, year, month."""
    s = roll_scores(close, volume, variant).dropna()
    if s.empty:
        return pd.DataFrame(columns=["score", "year", "month"])
    df = pd.DataFrame({"score": s, "year": s.index.year, "month": s.index.month})
    best = df.loc[df.groupby(["year", "month"]).score.idxmax()]
    parts = []
    for _, g in best.groupby("year"):
        kk = max(1, round(k * len(g) / 12))
        parts.append(g.nlargest(kk, "score"))
    return pd.concat(parts).sort_index()


def roll_validation(rolls: pd.DataFrame, expected: int) -> dict:
    """Stability of the chosen roll months across full years (12 months of data): share of
    years whose set equals the modal set, and mean Jaccard overlap with it. Random volume
    spikes would give unstable sets; a real exchange cycle gives stable ones."""
    sets = rolls.groupby("year").month.apply(lambda m: frozenset(m))
    counts = rolls.groupby("year").size()
    full = counts[counts >= expected].index  # years in which all k rolls were found
    sets = sets.loc[full] if len(full) else sets
    if sets.empty:
        return {"modal_months": "", "stability_exact": 0.0, "stability_jaccard": 0.0}
    modal = pd.Series(list(sets)).value_counts().index[0]
    jacc = [len(s & modal) / len(s | modal) for s in sets]
    dom = rolls.index.day  # calendar day of the roll: exchange rolls fall on similar days
    return {
        "modal_months": " ".join(str(m) for m in sorted(modal)),
        "stability_exact": float(np.mean([s == modal for s in sets])),
        "stability_jaccard": float(np.mean(jacc)),
        "day_of_month_median": float(np.median(dom)),
        "day_of_month_iqr": float(np.subtract(*np.percentile(dom, [75, 25]))),
    }


def adjust(ret: pd.Series, roll_dates) -> pd.Series:
    """Remove roll gaps: the return on a detected roll day is set to 0."""
    out = ret.copy()
    out.loc[out.index.intersection(pd.DatetimeIndex(roll_dates))] = 0.0
    return out


def index_from_returns(returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Wide price-like index (100 x cumulative product of 1 + r) per contract, NaN where a
    contract has no data, so the ETF pipeline (returns, volatility, signals) can run on it."""
    return pd.DataFrame({t: 100 * (1 + r).cumprod() for t, r in returns.items()}).sort_index()


def choose_variant(close, volume, k):
    """Run both scores, validate on volume years, pick the more stable one."""
    vy = volume_years(volume)
    best = None
    for variant in VARIANTS:
        rl = detect_rolls(close, volume, k, variant)
        v = roll_validation(rl[rl.year.isin(vy)], k) if vy else roll_validation(rl, k)
        quality = v["stability_jaccard"] if k < 12 else -v["day_of_month_iqr"]
        if best is None or quality > best[0]:
            best = (quality, variant, rl, v)
    _, variant, rl, v = best
    kept = (
        (v["stability_jaccard"] >= MIN_STABILITY)
        if k < 12
        else (v["day_of_month_iqr"] <= MAX_DOM_IQR)
    )
    return variant, rl, v, bool(kept)


def process(bars: dict[str, pd.DataFrame], universe: pd.DataFrame):
    """Quality flags, cleaned and roll-adjusted returns for every contract."""
    flags, rolls, val, raw_ret, adj_ret = [], [], [], {}, {}
    for _, u in universe.iterrows():
        t = u.ticker
        if t not in bars:
            continue
        close, volume = bars[t]["close"], bars[t]["volume"]
        k = int(u.rolls_per_year)
        f = quality_flags(close, volume)
        f.insert(0, "ticker", t)
        flags.append(f)
        r = clean_returns(close)
        variant, rl, v, kept = choose_variant(close, volume, k)
        raw_ret[t] = r
        adj_ret[t] = adjust(r, rl.index)
        rolls.append(
            pd.DataFrame(
                {
                    "ticker": t,
                    "date": rl.index,
                    "score": rl.score.to_numpy(),
                    "raw_return": r.reindex(rl.index).to_numpy(),
                }
            )
        )
        ordinary = r.drop(rl.index, errors="ignore").abs().mean()
        val.append(
            {
                "ticker": t,
                "group": u.group,
                "expected_per_year": k,
                "variant": variant,
                "found_per_year": float(rl.groupby("year").size().iloc[1:-1].mean()),
                "first_date": str(close.index[0].date()),
                "n_days": len(close),
                "zero_volume_share": float((~(volume > 0)).mean()),
                **v,
                "mean_roll_score": float(rl.score.mean()),
                "roll_day_abs_ret_over_ordinary": float(
                    r.reindex(rl.index).abs().mean() / ordinary
                ),
                "kept": kept,
            }
        )
    return (
        pd.concat(flags, ignore_index=True),
        pd.concat(rolls, ignore_index=True),
        pd.DataFrame(val),
        raw_ret,
        adj_ret,
    )


if __name__ == "__main__":
    from db import init_db

    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    uni = load_universe()
    failed = download(uni, refresh=args.refresh)
    bars = read_snapshot(uni)
    conn = init_db()
    load_into_db(conn, bars)
    flags, rolls, val, _, _ = process(bars, uni)
    conn.execute("DELETE FROM futures_rolls")
    conn.executemany(
        "INSERT INTO futures_rolls (ticker, date, score, raw_return) VALUES (?,?,?,?)",
        [
            (r.ticker, r.date.strftime("%Y-%m-%d"), r.score, r.raw_return)
            for r in rolls.itertuples()
        ],
    )
    conn.commit()
    out = ROOT / "results"
    flags["date"] = pd.to_datetime(flags["date"]).dt.strftime("%Y-%m-%d")
    flags.to_csv(out / "futures_quality_flags.csv", index=False)
    (
        flags.groupby(["ticker", "check"])
        .date.agg(count="size", first="min", last="max")
        .reset_index()
        .to_csv(out / "futures_quality.csv", index=False)
    )
    rolls.assign(date=rolls.date.dt.strftime("%Y-%m-%d")).to_csv(
        out / "futures_rolls.csv", index=False
    )
    val.to_csv(out / "futures_roll_validation.csv", index=False)
    pd.set_option("display.width", 220)
    print("failed downloads:", failed or "none")
    print(flags.groupby("check").size().to_dict())
    show = [
        "ticker",
        "group",
        "variant",
        "modal_months",
        "stability_jaccard",
        "day_of_month_median",
        "day_of_month_iqr",
        "zero_volume_share",
        "kept",
    ]
    print(val[show].round(2).to_string(index=False))
    print("kept:", int(val.kept.sum()), "of", len(val), "| excluded:", list(val.ticker[~val.kept]))
