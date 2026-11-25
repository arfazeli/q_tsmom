-- Data-quality checks on prices_raw. Each query returns (ticker, date, check_name, detail).
-- clean.py runs the blocks one by one, split on the "-- name:" markers. Nothing is deleted here.

-- name: missing_day
-- A date on which >= 90% of the tickers already trading (between their first and last row)
-- have a row, but this ticker has none.
WITH span AS (SELECT ticker, MIN(date) AS first, MAX(date) AS last FROM prices_raw GROUP BY ticker),
dates AS (SELECT DISTINCT date FROM prices_raw),
active AS (SELECT d.date, s.ticker FROM dates d JOIN span s ON d.date BETWEEN s.first AND s.last),
coverage AS (
    SELECT a.date, COUNT(*) AS n_active, COUNT(p.ticker) AS n_have
    FROM active a LEFT JOIN prices_raw p ON p.ticker = a.ticker AND p.date = a.date
    GROUP BY a.date
)
SELECT a.ticker, a.date, 'missing_day' AS check_name,
       printf('%d of %d active tickers have a row', c.n_have, c.n_active) AS detail
FROM active a
JOIN coverage c ON c.date = a.date
LEFT JOIN prices_raw p ON p.ticker = a.ticker AND p.date = a.date
WHERE p.ticker IS NULL AND c.n_have >= 0.9 * c.n_active;

-- name: non_positive
SELECT ticker, date, 'non_positive' AS check_name,
       printf('adj_close = %s', COALESCE(CAST(adj_close AS TEXT), 'NULL')) AS detail
FROM prices_raw
WHERE adj_close IS NULL OR adj_close <= 0;

-- name: stale
-- adj_close unchanged (relative difference <= 1e-9) for >= 5 consecutive rows; flags the
-- 5th row of the run onward. Gaps-and-islands: a new group starts at every price change.
WITH x AS (
    SELECT ticker, date, adj_close,
           LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS prev
    FROM prices_raw
),
g AS (
    SELECT *, SUM(CASE WHEN prev IS NULL OR adj_close IS NULL
                         OR ABS(adj_close - prev) > 1e-9 * ABS(prev) THEN 1 ELSE 0 END)
              OVER (PARTITION BY ticker ORDER BY date) AS grp
    FROM x
),
r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker, grp ORDER BY date) AS run_len FROM g)
SELECT ticker, date, 'stale' AS check_name,
       printf('unchanged for %d rows at %.6f', run_len, adj_close) AS detail
FROM r
WHERE run_len >= 5 AND adj_close IS NOT NULL;

-- name: big_move
-- |daily simple return of adj_close| > 20%. Flagged for review, not removed.
WITH x AS (
    SELECT ticker, date, adj_close,
           LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS prev,
           LAG(date) OVER (PARTITION BY ticker ORDER BY date) AS prev_date
    FROM prices_raw
)
SELECT ticker, date, 'big_move' AS check_name,
       printf('ret = %+.4f since %s', adj_close / prev - 1, prev_date) AS detail
FROM x
WHERE prev > 0 AND adj_close > 0 AND ABS(adj_close / prev - 1) > 0.20;
