"""Load the ETF universe (config/universe.yaml) and write it to `assets`."""

import sqlite3

import pandas as pd
import yaml

from db import ROOT

UNIVERSE_PATH = ROOT / "config" / "universe.yaml"
ASSET_CLASSES = {"equity", "govt_bond", "credit", "commodity", "currency", "real_estate"}


def load_universe(path=UNIVERSE_PATH) -> pd.DataFrame:
    df = pd.DataFrame(yaml.safe_load(open(path)))
    df["inception_date"] = pd.to_datetime(df["inception_date"]).dt.strftime("%Y-%m-%d")
    if df.ticker.duplicated().any():
        raise ValueError(f"duplicate tickers: {df.ticker[df.ticker.duplicated()].tolist()}")
    if bad := set(df.asset_class) - ASSET_CLASSES:
        raise ValueError(f"unknown asset classes: {bad}")
    return df[["ticker", "name", "asset_class", "inception_date"]]


def write_assets(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO assets (ticker, name, asset_class, inception_date) "
        "VALUES (?, ?, ?, ?)",
        df.itertuples(index=False),
    )
    conn.commit()


if __name__ == "__main__":
    u = load_universe()
    print(len(u), "ETFs")
    print(u.groupby("asset_class").ticker.apply(" ".join).to_string())
