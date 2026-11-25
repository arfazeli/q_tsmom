"""Figures from the stored runs and results/ CSVs. Writes results/figures/*.png.

Colours follow the entity, the same in every figure: blue = baseline / in-sample,
orange = out-of-sample, aqua = volatility-conditioned strategy (validated palette, light mode).
Run: uv run python src/plots.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import STRATEGY  # noqa: E402
from db import ROOT, connect  # noqa: E402

FIG = ROOT / "results" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OOS_START = pd.Timestamp(STRATEGY["out_sample_start"])

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    }
)


def run_returns(conn, run_id: int) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT * FROM portfolio_returns WHERE run_id = ? ORDER BY month", conn, params=(run_id,)
    )
    df.index = pd.PeriodIndex(df.pop("month"), freq="M").to_timestamp(how="end").normalize()
    return df


def latest(conn, sample: str, step: str) -> int:
    return conn.execute(
        "SELECT MAX(run_id) FROM runs WHERE sample = ? AND json_extract(config_json, '$.step') = ?",
        (sample, step),
    ).fetchone()[0]


def shade_oos(ax, label=True, names=("in-sample  ", "  out-of-sample (run once)")):
    ax.axvspan(
        OOS_START,
        ax.get_xlim()[1] if False else pd.Timestamp("2026-10-01"),
        color="#f1f0ec",
        zorder=0,
        lw=0,
    )
    ax.axvline(OOS_START, color=MUTED, lw=0.8, ls="--", zorder=1)
    if label:
        y = ax.get_ylim()[1]
        ax.text(
            OOS_START - pd.Timedelta(days=60),
            y,
            names[0],
            ha="right",
            va="top",
            color=INK2,
            fontsize=8.5,
        )
        ax.text(
            OOS_START + pd.Timedelta(days=60),
            y,
            names[1],
            ha="left",
            va="top",
            color=INK2,
            fontsize=8.5,
        )


def pct(ax, axis="y", digits=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=digits)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig_cumulative(base, vol_cond):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for r, color, name in (
        (base, BLUE, "Baseline (12-month TSMOM)"),
        (vol_cond, AQUA, "Half size in high-vol months"),
    ):
        wealth = (1 + r).cumprod()
        ax.plot(wealth.index, wealth, color=color, label=name, zorder=3)
        ax.annotate(
            f"{wealth.iloc[-1]:.2f}",
            (wealth.index[-1], wealth.iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=INK2,
            fontsize=8.5,
        )
    ax.set_ylabel("Growth of $1 (net excess return)")
    ax.set_title("Cumulative net excess return, 2003-08 to 2026-09")
    ax.axhline(1, color=AXIS, lw=0.8)
    ax.set_xlim(base.index[0] - pd.Timedelta(days=60), pd.Timestamp("2027-06-01"))
    shade_oos(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.savefig(FIG / "fig1_cumulative.png")
    plt.close(fig)


def fig_drawdown(base):
    wealth = (1 + base).cumprod()
    dd = wealth / wealth.cummax() - 1
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.fill_between(dd.index, dd, 0, color=BLUE, alpha=0.25, lw=0)
    ax.plot(dd.index, dd, color=BLUE, lw=1.2)
    i = dd.idxmin()
    ax.annotate(
        f"max drawdown {dd.min():.1%} ({i:%b %Y})",
        (i, dd.min()),
        xytext=(-10, -2),
        textcoords="offset points",
        color=INK2,
        fontsize=8.5,
        va="top",
        ha="right",
    )
    pct(ax)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown from previous peak (baseline, net)")
    ax.set_xlim(base.index[0] - pd.Timedelta(days=60), pd.Timestamp("2026-12-01"))
    ax.set_ylim(dd.min() * 1.25, 0.004)
    shade_oos(ax, label=False)
    fig.savefig(FIG / "fig2_drawdown.png")
    plt.close(fig)


def fig_rolling_sharpe(base, mkt):
    roll = base.rolling(36).mean() / base.rolling(36).std() * np.sqrt(12)
    mroll = mkt.rolling(36).mean() / mkt.rolling(36).std() * np.sqrt(12)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(mroll.index, mroll, color=MUTED, lw=1.2, ls="-", label="US stock market")
    ax.plot(roll.index, roll, color=BLUE, label="Strategy (net)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Sharpe ratio (36-month window)")
    ax.set_title("Rolling 36-month Sharpe ratio")
    ax.set_xlim(base.index[0] - pd.Timedelta(days=60), pd.Timestamp("2026-12-01"))
    shade_oos(ax)
    ax.legend(loc="lower left", ncols=2)
    fig.savefig(FIG / "fig3_rolling_sharpe.png")
    plt.close(fig)


def fig_heatmap(grid):
    hm = grid.pivot(index="lookback", columns="holding", values="net_sharpe")
    lim = 1.0
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(hm.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(hm.shape[1]), [f"{h} mo" for h in hm.columns])
    ax.set_yticks(range(hm.shape[0]), [f"{lb} mo" for lb in hm.index])
    ax.set_xlabel("Holding period")
    ax.set_ylabel("Lookback")
    for (i, j), v in np.ndenumerate(hm.to_numpy()):
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if abs(v) > 0.55 else INK,
        )
    bi = list(hm.index).index(STRATEGY["lookback_months"])
    bj = list(hm.columns).index(STRATEGY["holding_months"])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, lw=2.2, ec=INK))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "Net Sharpe by lookback and holding period\nin-sample; box = baseline", fontsize=10
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    cb.set_label("Net Sharpe (annualized)", color=INK2)
    fig.savefig(FIG / "fig4_sensitivity_heatmap.png")
    plt.close(fig)


def grouped_hbar(
    ax, labels, a, b, la="In-sample 2003–2015", lb="Out-of-sample 2016–2026", fmt="{:+.1f}"
):
    y = np.arange(len(labels))
    h = 0.38
    for vals, off, color, name in ((a, -h / 2 - 0.01, BLUE, la), (b, h / 2 + 0.01, ORANGE, lb)):
        bars = ax.barh(y + off, vals, height=h, color=color, label=name, zorder=3)
        for rect, v in zip(bars, vals, strict=True):
            ax.annotate(
                fmt.format(v),
                (v, rect.get_y() + h / 2),
                xytext=(3 if v >= 0 else -3, 0),
                textcoords="offset points",
                ha="left" if v >= 0 else "right",
                va="center",
                fontsize=8,
                color=INK2,
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="y", visible=False)


def fig_classes(contrib):
    cols = ["equity", "commodity", "real_estate", "credit", "govt_bond", "currency", "cost"]
    names = ["Equity", "Commodities", "Real estate", "Credit", "Govt bonds", "Currencies", "Costs"]
    tot = contrib.groupby("sample")[cols].sum() * 100
    fig, ax = plt.subplots(figsize=(7, 3.8))
    grouped_hbar(ax, names, tot.loc["in"].to_numpy(), tot.loc["out"].to_numpy())
    ax.set_xlabel("Contribution to return (sum of monthly contributions, % points)")
    ax.set_title("Where the returns came from, by asset class")
    ax.legend(loc="lower right")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 2, hi + 3)
    fig.savefig(FIG / "fig5_asset_class_contributions.png")
    plt.close(fig)


def fig_regimes(perf):
    rows = [
        ("high_vol", True, "High-volatility months"),
        ("high_vol", False, "Calm months"),
        ("bear", True, "Bear market (in-sample only*)"),
        ("bear", False, "Not bear market"),
    ]
    a, b, labels = [], [], []
    for reg, val, name in rows:
        sel = perf[(perf.regime == reg) & (perf.in_regime == val)].set_index("sample")
        a.append(sel.loc["in", "sharpe"])
        b.append(sel.loc["out", "sharpe"])  # NaN where not estimable: no bar drawn
        n_in, n_out = int(sel.loc["in", "months"]), int(sel.loc["out", "months"])
        labels.append(f"{name}\n{n_in} in / {n_out} out months")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b = np.array(b)
    grouped_hbar(ax, labels, np.array(a), np.nan_to_num(b, nan=0.0), fmt="{:+.2f}")
    for txt in ax.texts:  # replace the label of the missing bar
        if txt.get_text() == "+0.00":
            txt.set_text("n/a: 1 month")
    ax.set_xlabel("Net Sharpe ratio (annualized)")
    ax.set_title("Performance by market regime")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    ax.text(
        0,
        -0.32,
        "* no bear market by the 24-month definition in 2016–2026",
        transform=ax.transAxes,
        fontsize=8,
        color=MUTED,
    )
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.15, hi + 0.2)
    fig.savefig(FIG / "fig6_regimes.png")
    plt.close(fig)


def fig_scatter(base, mkt):
    df = pd.concat([mkt.rename("m"), base.rename("s")], axis=1, join="inner")
    ins, oos = df[df.index < OOS_START], df[df.index >= OOS_START]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(
        ins.m,
        ins.s,
        s=22,
        color=BLUE,
        alpha=0.75,
        lw=0.6,
        ec="white",
        label="In-sample month",
        zorder=3,
    )
    ax.scatter(
        oos.m,
        oos.s,
        s=22,
        color=ORANGE,
        alpha=0.8,
        lw=0.6,
        ec="white",
        label="Out-of-sample month",
        zorder=3,
    )
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    notes = {  # month: (x offset, y offset, horizontal alignment)
        "2008-10": (6, 6, "left"),
        "2008-09": (-6, -12, "right"),
        "2020-03": (6, -10, "left"),
        "2020-02": (6, -12, "left"),
        "2018-12": (6, 4, "left"),
        "2022-09": (8, 0, "left"),
        "2022-06": (6, 8, "left"),
    }
    for m, (dx, dy, ha) in notes.items():
        t = pd.Period(m, "M").to_timestamp(how="end").normalize()
        if t in df.index:
            ax.annotate(
                pd.Period(m, "M").strftime("%b %Y"),
                (df.m[t], df.s[t]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                color=INK2,
                ha=ha,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6},
            )
    pct(ax, "x")
    pct(ax, "y", 1)
    ax.set_xlabel("US stock market excess return (month)")
    ax.set_ylabel("Strategy net return (month)")
    ax.set_title("Strategy vs stock market, every month")
    ax.legend(loc="lower left")
    fig.savefig(FIG / "fig7_vs_market.png")
    plt.close(fig)


def fig_universe(n_assets):
    fig, ax = plt.subplots(figsize=(8, 2.3))
    ax.step(n_assets.index, n_assets, where="post", color=BLUE, lw=1.6)
    ax.set_ylabel("ETFs held")
    ax.set_ylim(0, 29)
    ax.set_title("Number of ETFs in the portfolio (ETFs join once they have enough history)")
    ax.set_xlim(n_assets.index[0] - pd.Timedelta(days=60), pd.Timestamp("2026-12-01"))
    shade_oos(ax, label=False)
    fig.savefig(FIG / "fig8_universe.png")
    plt.close(fig)


def fig_benchmarks(bench):
    order = [
        "Strategy (net)",
        "Always long, same sizing",
        "Long-only trend",
        "60/40 stocks/bonds",
        "US stocks",
    ]
    b = bench.set_index(["portfolio", "sample"]).sharpe
    labels = [
        "Strategy (net)",
        "Always long, same sizing",
        "Long-only trend",
        "60/40 stocks/bonds",
        "US stocks",
    ]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    grouped_hbar(
        ax,
        labels,
        np.array([b[(o, "in")] for o in order]),
        np.array([b[(o, "out")] for o in order]),
        fmt="{:+.2f}",
    )
    ax.set_xlabel("Net Sharpe ratio (annualized)")
    ax.set_title("The strategy against passive alternatives")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.15)
    fig.savefig(FIG / "fig9_benchmarks.png")
    plt.close(fig)


def fig_futures(etf, fut):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for r, color, name in (
        (etf, BLUE, "27 ETFs (baseline)"),
        (fut, INK2, "22 futures, same rules"),
    ):
        w = (1 + r).cumprod()
        ax.plot(w.index, w, color=color, label=name, zorder=3)
        ax.annotate(
            f"{w.iloc[-1]:.2f}",
            (w.index[-1], w.iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=INK2,
            fontsize=8.5,
        )
    ax.axhline(1, color=AXIS, lw=0.8)
    ax.set_ylabel("Growth of $1 (net excess return)")
    ax.set_title("The same strategy on futures, 2001 to 2026")
    ax.set_xlim(fut.index[0] - pd.Timedelta(days=60), pd.Timestamp("2027-08-01"))
    shade_oos(ax, names=("", "  2016 onward"))
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.savefig(FIG / "fig10_futures.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    conn = connect()
    ids = {
        k: latest(conn, s, step)
        for k, (s, step) in {
            "base_in": ("in", "backtest"),
            "base_out": ("out", "backtest"),
            "vol_cond_in": ("in", "vol_conditioning"),
            "vol_cond_out": ("out", "vol_conditioning"),
        }.items()
    }
    base = pd.concat([run_returns(conn, ids["base_in"]), run_returns(conn, ids["base_out"])])
    vol_cond = pd.concat(
        [run_returns(conn, ids["vol_cond_in"]), run_returns(conn, ids["vol_cond_out"])]
    )
    mkt = pd.read_sql("SELECT month, mkt_rf FROM factors_monthly", conn)
    mkt.index = pd.PeriodIndex(mkt.pop("month"), freq="M").to_timestamp(how="end").normalize()
    mkt = mkt.mkt_rf.reindex(base.index)
    n_assets = pd.concat(
        [
            pd.read_csv(ROOT / "results" / f"portfolio_returns_{s}.csv", index_col=0)["n_assets"]
            for s in ("in", "out")
        ]
    )
    n_assets.index = pd.PeriodIndex(n_assets.index, freq="M").to_timestamp(how="end").normalize()

    fig_cumulative(base.net_ret, vol_cond.net_ret)
    fig_drawdown(base.net_ret)
    fig_rolling_sharpe(base.net_ret, mkt)
    fig_heatmap(pd.read_csv(ROOT / "results" / "sensitivity_grid_in.csv"))
    fig_classes(pd.read_csv(ROOT / "results" / "class_contrib_by_year.csv"))
    fig_regimes(pd.read_csv(ROOT / "results" / "regime_perf.csv"))
    fig_scatter(base.net_ret, mkt)
    fig_universe(n_assets)
    fig_benchmarks(pd.read_csv(ROOT / "results" / "benchmarks.csv"))
    fut = pd.read_csv(ROOT / "results" / "futures_portfolio_returns.csv", index_col=0).net_ret
    fut.index = pd.PeriodIndex(fut.index, freq="M").to_timestamp(how="end").normalize()
    fig_futures(base.net_ret, fut)
    print("runs used:", ids)
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
