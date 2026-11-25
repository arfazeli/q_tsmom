"""SQLite connection helpers."""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "tsmom.sqlite"
SQL_DIR = ROOT / "sql"


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def run_sql_file(conn: sqlite3.Connection, name: str) -> None:
    """Execute a file from sql/ (e.g. 'schema.sql') as one script."""
    conn.executescript((SQL_DIR / name).read_text())


def init_db(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Create (or open) the database and apply the schema and views. Idempotent."""
    conn = connect(path)
    run_sql_file(conn, "schema.sql")
    run_sql_file(conn, "views.sql")
    return conn


def git_hash() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # only code and config count: output files (results/) don't change what ran
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "src",
                "sql",
                "config",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_run(conn: sqlite3.Connection, config: dict, sample: str) -> int:
    """Insert a row into `runs` (config JSON, git hash, UTC time) and return its run_id."""
    import json
    from datetime import UTC, datetime

    cur = conn.execute(
        "INSERT INTO runs (config_json, git_hash, created_at, sample) VALUES (?, ?, ?, ?)",
        (
            json.dumps(config, sort_keys=True),
            git_hash(),
            datetime.now(UTC).isoformat(timespec="seconds"),
            sample,
        ),
    )
    conn.commit()
    return cur.lastrowid
