"""Human-readable output for backtest results."""

from __future__ import annotations

from pathlib import Path

from .backtest import BacktestResult

_ROWS = [
    ("Period", lambda s: f"{s['start'][:10]} → {s['end'][:10]}  ({s['bars']:,} bars)"),
    ("Sharpe (net / gross)", lambda s: f"{s['sharpe']:.2f} / {s['sharpe_gross']:.2f}"),
    ("Annual return", lambda s: f"{s['annual_return']:+.2%}"),
    ("  gross | fees | funding", lambda s: f"{s['annual_gross']:+.2%} | {s['annual_fees']:+.2%}"
                                           f" | {s['annual_funding']:+.2%}"),
    ("Annual vol", lambda s: f"{s['annual_vol']:.2%}"),
    ("Max drawdown", lambda s: f"{s['max_drawdown']:.2%}"),
    ("Daily turnover", lambda s: f"{s['daily_turnover']:.1%}"),
    ("Fitness (BRAIN)", lambda s: f"{s['fitness']:.2f}"),
    ("IC mean / IR / hit", lambda s: f"{s['ic_mean']:+.4f} / {s['ic_ir']:+.3f} / "
                                     f"{s['ic_hit_rate']:.1%}"),
    ("Avg names long / short", lambda s: f"{s['avg_long_names']:.1f} / {s['avg_short_names']:.1f}"),
    ("Avg net exposure", lambda s: f"{s['avg_net_exposure']:+.3f}"),
]


def format_summary(result: BacktestResult, title: str = "") -> str:
    s = result.summary
    width = max(len(k) for k, _ in _ROWS)
    lines = [title, "-" * 56] if title else []
    lines += [f"{k:<{width}}  {fmt(s)}" for k, fmt in _ROWS]
    if not result.yearly.empty:
        lines += ["", "Year   return    sharpe   max_dd"]
        for year, row in result.yearly.iterrows():
            lines.append(f"{year}  {row['return']:+8.2%}  {row['sharpe']:7.2f}  "
                         f"{row['max_drawdown']:+8.2%}")
    return "\n".join(lines)


def save_equity_plot(result: BacktestResult, path: str | Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pnl = result.pnl.cumsum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(pnl.index, pnl["net"], label="net", linewidth=1.6)
    ax.plot(pnl.index, pnl["gross"], label="gross", linewidth=1, alpha=0.7)
    ax.plot(pnl.index, pnl["fees"], label="fees", linewidth=1, alpha=0.7)
    ax.plot(pnl.index, pnl["funding"], label="funding", linewidth=1, alpha=0.7)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("cumulative return (per unit gross)")
    ax.set_title(title or "Backtest")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
