"""Download daily ETF prices (Yahoo) and Kenneth French factors into a frozen raw
snapshot (data/raw_snapshot/), then load the snapshot into SQLite.

Download and load are separate: loading needs no network, so tests and rebuilds are
reproducible from the snapshot. Run: uv run python src/ingest.py [--refresh]
"""

import argparse
import csv
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from db import ROOT, init_db
from universe import load_universe, write_assets

SNAPSHOT = ROOT / "data" / "raw_snapshot"
PRICE_COLS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
FRENCH = {  # snapshot file -> Kenneth French dataset name (values in percent)
    "ff5_daily.csv": "F-F_Research_Data_5_Factors_2x3_daily",
    "ff5_monthly.csv": "F-F_Research_Data_5_Factors_2x3",
    "mom_monthly.csv": "F-F_Momentum_Factor",
}


def _stamp(snapshot: Path, file: str, source: str) -> None:
    """Record source and download time per file in MANIFEST.csv (mtimes don't survive git)."""
    path = snapshot / "MANIFEST.csv"
    rows = {}
    if path.exists():
        rows = {r["file"]: r for r in csv.DictReader(open(path))}
    rows[file] = {
        "file": file,
        "source": source,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, ["file", "source", "downloaded_at"])
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: r["file"]))


def download_prices(tickers, snapshot=SNAPSHOT, refresh=False) -> list[str]:
    """One CSV per ticker in snapshot/prices/. Returns tickers that failed."""
    import yfinance as yf

    out = snapshot / "prices"
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if refresh or not (out / f"{t}.csv").exists()]
    if not todo:
        return []
    data = yf.download(
        todo,
        start="1990-01-01",
        end=datetime.now(UTC).strftime("%Y-%m-%d"),  # end is exclusive: no partial bar for today
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
        df = df.rename(columns=PRICE_COLS)[list(PRICE_COLS.values())]
        df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
        df.index.name = "date"
        df.to_csv(out / f"{t}.csv")
        _stamp(snapshot, f"prices/{t}.csv", "yahoo")
    return failed


def download_factors(snapshot=SNAPSHOT, refresh=False) -> None:
    import pandas_datareader.data as web

    out = snapshot / "factors"
    out.mkdir(parents=True, exist_ok=True)
    for file, name in FRENCH.items():
        if (out / file).exists() and not refresh:
            continue
        df = web.DataReader(name, "famafrench", start="1960-01-01")[0]  # [0] = daily/monthly
        df.index = df.index.astype(str)
        df.index.name = "date" if "daily" in name else "month"
        df.columns = [c.strip() for c in df.columns]
        df.to_csv(out / file)
        _stamp(snapshot, f"factors/{file}", f"kenneth_french:{name}")


def load_snapshot(conn: sqlite3.Connection, snapshot=SNAPSHOT, tickers=None) -> None:
    """Load snapshot CSVs into prices_raw, rf_daily, factors_monthly. Idempotent.
    `tickers`: load only these price files (default: all in the snapshot)."""
    stamps = {r["file"]: r for r in csv.DictReader(open(snapshot / "MANIFEST.csv"))}
    for path in sorted((snapshot / "prices").glob("*.csv")):
        if tickers is not None and path.stem not in tickers:
            continue
        meta = stamps[f"prices/{path.name}"]
        df = pd.read_csv(path)
        df.insert(0, "ticker", path.stem)
        df["source"], df["downloaded_at"] = meta["source"], meta["downloaded_at"]
        conn.executemany(
            "INSERT OR REPLACE INTO prices_raw (ticker, date, open, high, low, close, "
            "adj_close, volume, source, downloaded_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            df.astype(object).where(df.notna(), None).itertuples(index=False),
        )
    f = snapshot / "factors"
    rf = pd.read_csv(f / "ff5_daily.csv")
    rf["date"] = pd.to_datetime(rf["date"]).dt.strftime("%Y-%m-%d")
    conn.executemany(
        "INSERT OR REPLACE INTO rf_daily (date, rf) VALUES (?, ?)",
        zip(rf["date"], rf["RF"] / 100, strict=True),
    )
    ff = pd.read_csv(f / "ff5_monthly.csv").merge(
        pd.read_csv(f / "mom_monthly.csv"), on="month", how="left"
    )
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
    ff[cols] = ff[cols] / 100
    ff = ff.astype(object).where(ff.notna(), None)
    conn.executemany(
        "INSERT OR REPLACE INTO factors_monthly (month, mkt_rf, smb, hml, rmw, cma, umd, rf) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ff[["month", *cols]].itertuples(index=False),
    )
    conn.commit()


def first_dates(conn) -> pd.DataFrame:
    """Inception date in universe.yaml vs first Yahoo price, as a cross-check."""
    return pd.read_sql(
        "SELECT a.ticker, a.inception_date, MIN(p.date) AS first_price, MAX(p.date) AS last_price, "
        "COUNT(p.date) AS n_days, julianday(MIN(p.date)) - julianday(a.inception_date) "
        "AS gap_days FROM assets a LEFT JOIN prices_raw p USING (ticker) GROUP BY a.ticker",
        conn,
    )


def counts(conn) -> dict:
    return {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("assets", "prices_raw", "rf_daily", "factors_monthly")
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download even if snapshot exists")
    args = ap.parse_args()
    conn = init_db()
    universe = load_universe()
    write_assets(conn, universe)
    failed = download_prices(universe.ticker.tolist(), refresh=args.refresh)
    download_factors(refresh=args.refresh)
    load_snapshot(conn, tickers=set(universe.ticker))
    print("failed downloads:", failed or "none")
    print(counts(conn))
