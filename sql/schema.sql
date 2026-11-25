-- tsmom schema. Dates are ISO 'YYYY-MM-DD', months 'YYYY-MM'.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS assets (
    ticker          TEXT PRIMARY KEY,
    name            TEXT,
    asset_class     TEXT NOT NULL,
    inception_date  TEXT
);

-- As downloaded; never edited.
CREATE TABLE IF NOT EXISTS prices_raw (
    ticker         TEXT NOT NULL REFERENCES assets(ticker),
    date           TEXT NOT NULL,
    open           REAL,
    high           REAL,
    low            REAL,
    close          REAL,
    adj_close      REAL,
    volume         REAL,
    source         TEXT NOT NULL,
    downloaded_at  TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS prices_clean (
    ticker     TEXT NOT NULL REFERENCES assets(ticker),
    date       TEXT NOT NULL,
    adj_close  REAL NOT NULL CHECK (adj_close > 0),
    flag       TEXT,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS rf_daily (
    date  TEXT PRIMARY KEY,
    rf    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS factors_monthly (
    month   TEXT PRIMARY KEY,
    mkt_rf  REAL,
    smb     REAL,
    hml     REAL,
    rmw     REAL,
    cma     REAL,
    umd     REAL,
    rf      REAL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    config_json  TEXT NOT NULL,
    git_hash     TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    sample       TEXT NOT NULL CHECK (sample IN ('in', 'out'))
);

CREATE TABLE IF NOT EXISTS signals (
    run_id      INTEGER NOT NULL REFERENCES runs(run_id),
    ticker      TEXT NOT NULL REFERENCES assets(ticker),
    month       TEXT NOT NULL,
    lookback    INTEGER NOT NULL,
    signal      INTEGER NOT NULL CHECK (signal IN (-1, 0, 1)),
    vol_exante  REAL,
    position    REAL,
    PRIMARY KEY (run_id, ticker, month, lookback)
);

CREATE TABLE IF NOT EXISTS portfolio_returns (
    run_id     INTEGER NOT NULL REFERENCES runs(run_id),
    month      TEXT NOT NULL,
    gross_ret  REAL,
    cost       REAL,
    net_ret    REAL,
    turnover   REAL,
    PRIMARY KEY (run_id, month)
);


-- Futures extension: raw Yahoo continuous front-month bars and the detected roll days.
CREATE TABLE IF NOT EXISTS futures_raw (
    ticker         TEXT NOT NULL,
    date           TEXT NOT NULL,
    open           REAL,
    high           REAL,
    low            REAL,
    close          REAL,
    volume         REAL,
    source         TEXT NOT NULL,
    downloaded_at  TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS futures_rolls (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    score       REAL,
    raw_return  REAL,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_clean_date ON prices_clean(date);
CREATE INDEX IF NOT EXISTS idx_signals_month ON signals(month);
