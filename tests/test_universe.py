import pytest

from db import init_db
from universe import ASSET_CLASSES, load_universe, write_assets


def test_universe_file():
    u = load_universe()
    assert len(u) >= 25
    assert u.ticker.is_unique
    assert set(u.asset_class) <= ASSET_CLASSES
    assert (u.asset_class.value_counts() >= 2).all()
    assert u.inception_date.notna().all()


def test_write_assets_idempotent(tmp_path):
    conn = init_db(tmp_path / "t.sqlite")
    u = load_universe()
    write_assets(conn, u)
    write_assets(conn, u)
    assert conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == len(u)


def test_rejects_bad_class(tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text("- {ticker: X, name: x, asset_class: crypto, inception_date: 2020-01-01}\n")
    with pytest.raises(ValueError):
        load_universe(p)
