import sqlite3

import pytest

from db import init_db

TABLES = {
    "assets",
    "prices_raw",
    "prices_clean",
    "rf_daily",
    "factors_monthly",
    "signals",
    "portfolio_returns",
    "runs",
    "futures_raw",
    "futures_rolls",
}


def tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {r[0] for r in rows} - {"sqlite_sequence"}


def test_init_builds_empty_schema_idempotently(tmp_path):
    path = tmp_path / "t.sqlite"
    init_db(path).close()
    conn = init_db(path)  # second run must not fail
    assert tables(conn) == TABLES
    assert all(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0 for t in TABLES)


def test_constraints(tmp_path):
    conn = init_db(tmp_path / "t.sqlite")
    conn.execute("INSERT INTO assets (ticker, asset_class) VALUES ('SPY', 'equity')")
    with pytest.raises(sqlite3.IntegrityError):  # foreign key
        conn.execute("INSERT INTO prices_clean VALUES ('XXX', '2020-01-02', 1.0, NULL)")
    with pytest.raises(sqlite3.IntegrityError):  # non-positive price
        conn.execute("INSERT INTO prices_clean VALUES ('SPY', '2020-01-02', 0.0, NULL)")
    with pytest.raises(sqlite3.IntegrityError):  # sample must be in/out
        conn.execute(
            "INSERT INTO runs (config_json, git_hash, created_at, sample) "
            "VALUES ('{}', 'x', 'now', 'both')"
        )


def test_window_functions_available(tmp_path):
    # returns views need LAG(), SQLite >= 3.25
    conn = init_db(tmp_path / "t.sqlite")
    assert conn.execute("SELECT LAG(1) OVER () FROM (SELECT 1)").fetchone() == (None,)
