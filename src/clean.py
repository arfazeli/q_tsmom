"""Run data-quality checks (sql/quality_checks.sql), write prices_clean and the summary.

Rules: rows are flagged, not deleted, except non-positive/NULL adj_close, which the
schema forbids in prices_clean. Missing days have no row, so they are only reported.
Run: uv run python src/clean.py
"""

import re
import sqlite3

import pandas as pd

from db import ROOT, SQL_DIR, connect

RESULTS = ROOT / "results"


def checks() -> dict[str, str]:
    text = (SQL_DIR / "quality_checks.sql").read_text()
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {name: sql for name, sql in zip(parts[1::2], parts[2::2], strict=True)}


def run_checks(conn: sqlite3.Connection) -> pd.DataFrame:
    flags = pd.concat([pd.read_sql(sql, conn) for sql in checks().values()], ignore_index=True)
    return flags.sort_values(["ticker", "date", "check_name"], ignore_index=True)


def write_clean(conn: sqlite3.Connection, flags: pd.DataFrame) -> None:
    """Rebuild prices_clean from prices_raw; flag = comma-joined checks for that row."""
    conn.execute("DROP TABLE IF EXISTS temp.flags")
    conn.execute("CREATE TEMP TABLE flags (ticker TEXT, date TEXT, check_name TEXT)")
    conn.executemany(
        "INSERT INTO temp.flags VALUES (?, ?, ?)",
        flags[["ticker", "date", "check_name"]].itertuples(index=False),
    )
    conn.execute("CREATE INDEX temp.idx_flags ON flags (ticker, date)")
    conn.execute("DELETE FROM prices_clean")
    conn.execute(
        """INSERT INTO prices_clean (ticker, date, adj_close, flag)
           SELECT p.ticker, p.date, p.adj_close,
                  (SELECT group_concat(f.check_name, ',') FROM temp.flags f
                   WHERE f.ticker = p.ticker AND f.date = p.date)
           FROM prices_raw p
           WHERE p.adj_close > 0"""
    )
    conn.commit()


def summary(flags: pd.DataFrame) -> pd.DataFrame:
    return (
        flags.groupby(["ticker", "check_name"])
        .date.agg(count="size", first_date="min", last_date="max")
        .reset_index()
    )


if __name__ == "__main__":
    conn = connect()
    flags = run_checks(conn)
    write_clean(conn, flags)
    RESULTS.mkdir(exist_ok=True)
    summary(flags).to_csv(RESULTS / "data_quality.csv", index=False)
    flags.to_csv(RESULTS / "data_quality_flags.csv", index=False)
    print(flags.check_name.value_counts().to_string())
    n = conn.execute("SELECT COUNT(*), COUNT(flag) FROM prices_clean").fetchone()
    print(f"prices_clean rows: {n[0]}, flagged rows: {n[1]}")
