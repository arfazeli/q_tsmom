import numpy as np
import pandas as pd

from clean import run_checks, write_clean
from db import init_db

N_TICKERS, N_DAYS = 10, 40  # 10 tickers: one missing row leaves 9/10 = 90% coverage


def synthetic_db(tmp_path):
    """Random walks (1% daily vol, no natural flags) with one injected case per check."""
    conn = init_db(tmp_path / "t.sqlite")
    rng = np.random.default_rng(0)
    dates = list(pd.bdate_range("2020-01-01", periods=N_DAYS).strftime("%Y-%m-%d"))
    rows = []
    for k in range(N_TICKERS):
        t = f"T{k}"
        conn.execute("INSERT INTO assets (ticker, asset_class) VALUES (?, 'equity')", (t,))
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, N_DAYS)))
        if t == "T2":
            px[20:] *= 1.3  # big_move: one +30% jump on day 20
        if t == "T3":
            px[5:10] = px[5]  # stale: 5 identical rows, days 5-9 -> flag on day 9
        for i, (d, p) in enumerate(zip(dates, px, strict=True)):
            if t == "T0" and i == 10:
                continue  # missing_day
            rows.append((t, d, None if (t == "T1" and i == 15) else float(p)))  # non_positive
    conn.executemany(
        "INSERT INTO prices_raw (ticker, date, adj_close, source, downloaded_at) "
        "VALUES (?, ?, ?, 'test', 'now')",
        rows,
    )
    return conn, dates, len(rows)


def test_each_check_flags_exactly_its_case(tmp_path):
    conn, dates, n_raw = synthetic_db(tmp_path)
    flags = run_checks(conn)
    got = sorted((r.check_name, r.ticker, r.date) for r in flags.itertuples())
    assert got == sorted(
        [
            ("missing_day", "T0", dates[10]),
            ("non_positive", "T1", dates[15]),
            ("big_move", "T2", dates[20]),
            ("stale", "T3", dates[9]),
        ]
    )

    write_clean(conn, flags)
    n_clean, n_flagged = conn.execute("SELECT COUNT(*), COUNT(flag) FROM prices_clean").fetchone()
    assert n_clean == n_raw - 1  # only the non_positive row is dropped
    assert n_flagged == 2  # big_move and stale rows are kept, with their flag


def test_rerun_is_idempotent(tmp_path):
    conn, _, _ = synthetic_db(tmp_path)
    write_clean(conn, run_checks(conn))
    first = conn.execute("SELECT COUNT(*), COUNT(flag) FROM prices_clean").fetchone()
    write_clean(conn, run_checks(conn))
    assert conn.execute("SELECT COUNT(*), COUNT(flag) FROM prices_clean").fetchone() == first
