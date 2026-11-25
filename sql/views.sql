-- Derived returns. pandas computes the same in src/returns.py; tests require agreement to 1e-12.

-- Daily simple return from adjusted closes; the first row of each ticker has no return.
-- excess_ret is NULL where the risk-free rate is missing (French data lags ~2 months).
CREATE VIEW IF NOT EXISTS v_returns_daily AS
WITH x AS (
    SELECT ticker, date,
           adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
    FROM prices_clean
)
SELECT x.ticker, x.date, x.ret, x.ret - r.rf AS excess_ret
FROM x LEFT JOIN rf_daily r ON r.date = x.date
WHERE x.ret IS NOT NULL;

-- Monthly return: last adj_close of the month over last adj_close of the previous month,
-- minus 1 (equal to compounding the daily returns). The first month of each ticker is dropped.
CREATE VIEW IF NOT EXISTS v_returns_monthly AS
WITH last_of_month AS (
    SELECT ticker, substr(date, 1, 7) AS month, adj_close,
           ROW_NUMBER() OVER (PARTITION BY ticker, substr(date, 1, 7) ORDER BY date DESC) AS rn
    FROM prices_clean
),
m AS (
    SELECT ticker, month,
           adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY month) - 1 AS ret
    FROM last_of_month
    WHERE rn = 1
)
SELECT ticker, month, ret FROM m WHERE ret IS NOT NULL;
