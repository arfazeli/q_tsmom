import sqlite3

import pytest

from db import init_db
from ingest import counts, load_snapshot


def make_snapshot(root, tickers=("AAA",)):
    (root / "prices").mkdir(parents=True)
    (root / "factors").mkdir()
    rows = ["file,source,downloaded_at"]
    for t in tickers:
        (root / "prices" / f"{t}.csv").write_text(
            "date,open,high,low,close,adj_close,volume\n"
            "2020-01-02,10,11,9,10.5,10.0,100\n"
            "2020-01-03,10.5,11,10,,,0\n"  # missing close -> NULL
        )
        rows.append(f"prices/{t}.csv,yahoo,2026-11-23T00:00:00+00:00")
    (root / "factors" / "ff5_daily.csv").write_text(
        "date,Mkt-RF,SMB,HML,RMW,CMA,RF\n2020-01-02,1.0,0,0,0,0,0.5\n2020-01-03,1.0,0,0,0,0,0.5\n"
    )
    (root / "factors" / "ff5_monthly.csv").write_text(
        "month,Mkt-RF,SMB,HML,RMW,CMA,RF\n2020-01,2.0,0.1,0.2,0.3,0.4,0.13\n"
    )
    (root / "factors" / "mom_monthly.csv").write_text("month,Mom\n2020-01,-1.5\n")
    for f in ("ff5_daily.csv", "ff5_monthly.csv", "mom_monthly.csv"):
        rows.append(f"factors/{f},kenneth_french,2026-11-23T00:00:00+00:00")
    (root / "MANIFEST.csv").write_text("\n".join(rows) + "\n")


def db_with_asset(tmp_path, ticker="AAA"):
    conn = init_db(tmp_path / "t.sqlite")
    conn.execute("INSERT INTO assets (ticker, asset_class) VALUES (?, 'equity')", (ticker,))
    return conn


def test_load_is_idempotent_and_converts_percent(tmp_path):
    snap = tmp_path / "snap"
    make_snapshot(snap)
    conn = db_with_asset(tmp_path)
    load_snapshot(conn, snap)
    first = counts(conn)
    load_snapshot(conn, snap)
    assert (
        counts(conn) == first == {"assets": 1, "prices_raw": 2, "rf_daily": 2, "factors_monthly": 1}
    )
    assert conn.execute("SELECT rf FROM rf_daily LIMIT 1").fetchone()[0] == pytest.approx(0.005)
    m = conn.execute("SELECT month, mkt_rf, umd, rf FROM factors_monthly").fetchone()
    assert m == ("2020-01", pytest.approx(0.02), pytest.approx(-0.015), pytest.approx(0.0013))
    p = conn.execute(
        "SELECT adj_close, source, downloaded_at FROM prices_raw WHERE date='2020-01-03'"
    ).fetchone()
    assert p == (None, "yahoo", "2026-11-23T00:00:00+00:00")


def test_unknown_ticker_rejected(tmp_path):
    snap = tmp_path / "snap"
    make_snapshot(snap, tickers=("ZZZ",))
    conn = db_with_asset(tmp_path)  # only AAA in assets
    with pytest.raises(sqlite3.IntegrityError):
        load_snapshot(conn, snap)


def test_load_only_listed_tickers(tmp_path):
    snap = tmp_path / "snap"
    make_snapshot(snap, tickers=("AAA", "ZZZ"))
    conn = db_with_asset(tmp_path)  # ZZZ not in assets: must be skipped, not rejected
    load_snapshot(conn, snap, tickers={"AAA"})
    assert conn.execute("SELECT DISTINCT ticker FROM prices_raw").fetchall() == [("AAA",)]
