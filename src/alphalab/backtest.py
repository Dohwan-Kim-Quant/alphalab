"""Vectorised cross-sectional backtester.

Timing (the part interviewers will poke at)
-------------------------------------------
Row t = bar t, which closes at time t + interval.

    alpha[t]        computed from data up to bar t's close
    target W[t]     = weights from alpha[t]
    position P[t]   = W[t - delay]       position held right after bar t's close
    PnL of bar t    = P[t-1] . r[t]      r[t] = close[t] / close[t-1] - 1
    turnover[t]     = sum |P[t] - P[t-1]|, traded at bar t's close, charged fee_bps
    funding of t    = -P[t-1] . f[t]     f[t] = rates settled during bar t (longs pay
                                         when the rate is positive)

With delay=0 you trade at the same close the signal was computed on, which is
optimistic (you cannot compute and execute in zero time). delay=1 (default)
trades one bar later and is the safer research setting.

All PnL is expressed per unit of gross exposure: weights are scaled so
sum |w| = 1, i.e. 0.5 long and 0.5 short when market neutral. Returns are
summed, not compounded, as in BRAIN.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MarketData
from .operators import demean, scale, ts_covariance, ts_std_dev
from .timeutil import bars_per_day, bars_per_year


@dataclass
class BacktestConfig:
    delay: int = 1
    neutralize: str = "market"        # "none" | "market" | "beta"
    fee_bps: float = 5.0              # per unit traded; Binance taker is ~4-5 bps
    max_weight: float | None = 0.10   # cap per symbol, as a share of gross
    include_funding: bool = True
    beta_lookback_days: float = 30
    ic_horizon: int = 1               # bars ahead for the IC
    min_ic_names: int = 5


@dataclass
class BacktestResult:
    config: BacktestConfig
    positions: pd.DataFrame
    pnl: pd.DataFrame                 # gross, fees, funding, net (per bar)
    turnover: pd.Series
    ic: pd.Series
    summary: dict = field(default_factory=dict)
    yearly: pd.DataFrame = field(default_factory=pd.DataFrame)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.pnl.assign(turnover=self.turnover, ic=self.ic).to_csv(out / "pnl.csv")
        self.yearly.to_csv(out / "yearly.csv")
        (out / "summary.json").write_text(json.dumps(
            {"config": asdict(self.config), "summary": self.summary}, indent=2, default=str))
        return out


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def rolling_beta(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Each symbol's beta to the equal-weighted market, from trailing data only."""
    market = returns.mean(axis=1)
    m = pd.DataFrame(np.repeat(market.to_numpy()[:, None], returns.shape[1], axis=1),
                     index=returns.index, columns=returns.columns)
    var = ts_std_dev(m, lookback) ** 2
    return ts_covariance(returns, m, lookback) / var.where(var > 0)


def beta_neutralize(a: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    """Per row, remove the projection of `a` on [1, beta]: the result has
    sum(w) = 0 and sum(w * beta) = 0, so it is both dollar- and beta-neutral."""
    beta = beta.reindex_like(a)
    active = a.notna() & beta.notna()
    A = a.where(active, 0.0)
    B = beta.where(active, 0.0)
    n = active.sum(axis=1).astype("float64")
    sb, sbb = B.sum(axis=1), (B * B).sum(axis=1)
    sw, sbw = A.sum(axis=1), (A * B).sum(axis=1)
    det = n * sbb - sb * sb
    ok = det.abs() > 1e-12
    c0 = ((sbb * sw - sb * sbw) / det).where(ok)
    c1 = ((n * sbw - sb * sw) / det).where(ok)
    resid = a.sub(c0, axis=0) - beta.mul(c1, axis=0)
    # Degenerate rows (e.g. all betas equal) fall back to plain demeaning.
    ok_panel = pd.DataFrame(np.broadcast_to(ok.to_numpy()[:, None], a.shape),
                            index=a.index, columns=a.columns)
    resid = resid.where(ok_panel, demean(a))
    return resid.where(active)


def alpha_to_weights(alpha: pd.DataFrame, universe: pd.DataFrame, neutralize: str = "market",
                     beta: pd.DataFrame | None = None,
                     max_weight: float | None = None) -> pd.DataFrame:
    a = alpha.reindex_like(universe).where(universe)
    a = a.replace([np.inf, -np.inf], np.nan)

    def neutral(x: pd.DataFrame) -> pd.DataFrame:
        if neutralize == "none":
            return x
        if neutralize == "market":
            return demean(x)
        if neutralize == "beta":
            if beta is None:
                raise ValueError("beta neutralisation needs a beta panel")
            return beta_neutralize(x, beta)
        raise ValueError(f"unknown neutralize={neutralize!r}")

    w = scale(neutral(a)).fillna(0.0)
    if max_weight:
        w = cap_weights(w, max_weight, balance=neutralize != "none")
    return w


def cap_weights(w: pd.DataFrame, cap: float, balance: bool = True,
                max_iter: int = 100) -> pd.DataFrame:
    """Cap |w| per symbol by water-filling each side separately: names over the
    cap are pinned at it and the excess is spread pro rata over the rest of
    that side, so each side keeps its gross.

    A side with too few names may be unable to keep its gross under the cap
    (n_names * cap < gross). With balance=True (neutral books) the other side
    is shrunk to match, keeping long gross == short gross at the cost of a
    smaller book; with balance=False each side just does what it can.
    Beta neutrality is only approximately preserved by capping."""
    vals = w.to_numpy(dtype="float64")
    sides = {s: np.where(s * vals > 0, np.abs(vals), 0.0) for s in (1.0, -1.0)}
    targets = {s: np.minimum(v.sum(axis=1, keepdims=True),
                             (v > 0).sum(axis=1, keepdims=True) * cap)
               for s, v in sides.items()}
    if balance:
        both = np.minimum(targets[1.0], targets[-1.0])
        targets = {1.0: both, -1.0: both}

    out = np.zeros(w.shape)
    for sgn, side in sides.items():
        target = targets[sgn]
        pinned = np.zeros(side.shape, dtype=bool)
        for _ in range(max_iter):
            pinned |= side > cap * (1 + 1e-12)
            free = np.where(pinned, 0.0, side)
            free_sum = free.sum(axis=1, keepdims=True)
            room = target - pinned.sum(axis=1, keepdims=True) * cap
            with np.errstate(invalid="ignore", divide="ignore"):
                k = np.where(free_sum > 0, np.maximum(room, 0) / free_sum, 0.0)
            side = np.where(pinned, cap, free * k)
            if not (side > cap * (1 + 1e-12)).any():
                break
        out += sgn * side
    return pd.DataFrame(out, index=w.index, columns=w.columns)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def cross_sectional_ic(alpha: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
                       min_names: int = 5) -> pd.Series:
    """Per-bar Spearman correlation between alpha and forward returns."""
    valid = mask & alpha.notna() & fwd.notna()
    ra = alpha.where(valid).rank(axis=1)
    rf = fwd.where(valid).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rf = rf.sub(rf.mean(axis=1), axis=0)
    num = (ra * rf).sum(axis=1)
    den = np.sqrt((ra * ra).sum(axis=1) * (rf * rf).sum(axis=1))
    ic = num / den.where(den > 0)
    return ic.where(valid.sum(axis=1) >= min_names)


def _sharpe(x: pd.Series, ppy: float) -> float:
    sd = x.std()
    return float(x.mean() / sd * np.sqrt(ppy)) if sd and sd > 0 else float("nan")


def summarize(pnl: pd.DataFrame, turnover: pd.Series, ic: pd.Series, positions: pd.DataFrame,
              interval: str) -> tuple[dict, pd.DataFrame]:
    ppy, bpd = bars_per_year(interval), bars_per_day(interval)
    net = pnl["net"]
    equity = net.cumsum()
    drawdown = equity - equity.cummax()
    ann_ret = float(net.mean() * ppy)
    daily_turnover = float(turnover.mean() * bpd)
    sharpe = _sharpe(net, ppy)
    ic_clean = ic.dropna()
    summary = {
        "start": str(net.index.min()), "end": str(net.index.max()), "bars": int(len(net)),
        "sharpe": sharpe,
        "sharpe_gross": _sharpe(pnl["gross"], ppy),
        "annual_return": ann_ret,
        "annual_vol": float(net.std() * np.sqrt(ppy)),
        "max_drawdown": float(drawdown.min()),
        "daily_turnover": daily_turnover,
        # BRAIN's fitness: rewards Sharpe and returns, penalises turnover.
        "fitness": float(sharpe * np.sqrt(abs(ann_ret) / max(daily_turnover, 0.125)))
        if np.isfinite(sharpe) else float("nan"),
        "annual_gross": float(pnl["gross"].mean() * ppy),
        "annual_fees": float(pnl["fees"].mean() * ppy),
        "annual_funding": float(pnl["funding"].mean() * ppy),
        "ic_mean": float(ic_clean.mean()) if len(ic_clean) else float("nan"),
        "ic_ir": float(ic_clean.mean() / ic_clean.std())
        if len(ic_clean) > 1 and ic_clean.std() > 0 else float("nan"),
        "ic_hit_rate": float((ic_clean > 0).mean()) if len(ic_clean) else float("nan"),
        "avg_long_names": float((positions > 0).sum(axis=1).mean()),
        "avg_short_names": float((positions < 0).sum(axis=1).mean()),
        "avg_net_exposure": float(positions.sum(axis=1).mean()),
    }
    by_year = net.groupby(net.index.year)
    yearly = pd.DataFrame({
        "return": by_year.sum(),
        "sharpe": by_year.apply(lambda s: _sharpe(s, ppy)),
        "max_drawdown": by_year.apply(lambda s: float((s.cumsum() - s.cumsum().cummax()).min())),
    })
    yearly.index.name = "year"
    return summary, yearly


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run_backtest(alpha: pd.DataFrame, data: MarketData, universe: pd.DataFrame,
                 cfg: BacktestConfig | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    if cfg.delay < 0:
        raise ValueError("delay must be >= 0")
    universe = universe.reindex_like(data.close).fillna(False).astype(bool)
    returns = data.returns

    beta = None
    if cfg.neutralize == "beta":
        lookback = int(cfg.beta_lookback_days * bars_per_day(data.interval))
        beta = rolling_beta(returns, lookback)

    weights = alpha_to_weights(alpha, universe, cfg.neutralize, beta, cfg.max_weight)
    positions = weights.shift(cfg.delay).fillna(0.0)
    held = positions.shift(1).fillna(0.0)

    # A held symbol with a missing return (data gap / delisting) contributes 0.
    gross = (held * returns.fillna(0.0)).sum(axis=1)
    turnover = (positions - held).abs().sum(axis=1)
    fees = turnover * cfg.fee_bps / 1e4
    funding = (-(held * data.funding.reindex_like(held).fillna(0.0)).sum(axis=1)
               if cfg.include_funding else gross * 0.0)
    pnl = pd.DataFrame({"gross": gross, "fees": -fees, "funding": funding})
    pnl["net"] = pnl.sum(axis=1)

    h = cfg.ic_horizon
    fwd = data.close.shift(-(cfg.delay + h)) / data.close.shift(-cfg.delay) - 1
    ic = cross_sectional_ic(alpha, fwd, universe, cfg.min_ic_names)

    # Evaluate from the bar where the first position is opened, so the cost of
    # building the initial book is included.
    active = positions.abs().sum(axis=1) > 0
    if not active.any():
        raise ValueError("alpha never produced a position; check universe and warm-up")
    first = active.idxmax()
    pnl, turnover, ic = pnl.loc[first:], turnover.loc[first:], ic.loc[first:]
    positions = positions.loc[first:]

    summary, yearly = summarize(pnl, turnover, ic, positions, data.interval)
    return BacktestResult(cfg, positions, pnl, turnover, ic, summary, yearly)
