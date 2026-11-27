# tsmom

Replication and stress test of cross-asset time-series momentum (Moskowitz, Ooi and Pedersen 2012) on liquid ETFs, with a futures check. Python and SQLite.

Market data (Yahoo Finance prices, Kenneth French factors) is downloaded when you run the code and is not redistributed here. Check the providers' terms before reusing it. Because data providers revise history, your numbers may differ slightly from any published ones.

## Setup

```bash
uv sync
uv run pytest -q
```

## Run

Each module runs on its own and writes to `data/tsmom.sqlite` and `results/` (both git-ignored).

```bash
uv run python src/ingest.py            # download data, build the database
uv run python src/clean.py             # data-quality checks, cleaned prices
uv run python src/backtest.py          # in-sample backtest (logged run)
uv run python src/stats.py             # statistics and factor regressions
uv run python src/overfitting.py       # sensitivity grid, costs, PSR / DSR
uv run python src/regime_analysis.py   # regimes and asset-class contributions
uv run python src/vol_conditioning.py  # halve positions after high-volatility months
uv run python src/benchmarks.py        # benchmarks, 60/40 overlay, per-class results, breadth
uv run python src/futures.py           # futures: cleaning, roll detection, validation
uv run python src/futures_backtest.py  # futures replication
uv run python src/plots.py             # figures
```

The out-of-sample sample (2016 onward) is locked behind `--confirm-oos`: `src/backtest.py --sample out --confirm-oos`. Run it once.

## Layout

```
config/    ETF universe, strategy parameters, futures universe
sql/       schema, return views, data-quality checks
src/       one module per step
tests/     unit, synthetic-data and no-look-ahead tests
```

## Method

12-month sign signal on excess returns; positions sized to 10% volatility per ETF (EWMA, centre of mass 60 days, lagged) and capped at 2x; equal weight; monthly rebalance; positions formed at month-end earn the next month's return; 5 bp per unit of turnover. Statistics: Lo standard errors, Newey-West t-stats, Fama-French 5 + momentum regressions, stationary block bootstrap, probabilistic and deflated Sharpe ratios.

## Licence

MIT
