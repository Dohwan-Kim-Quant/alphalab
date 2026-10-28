"""WorldQuant BRAIN-style operators on wide panels.

Every panel is a DataFrame with rows = bar open time and columns = symbols.
Names follow BRAIN's Fast Expression operators so an expression like

    rank(-ts_delta(close, 5))

ports almost verbatim:

    rank(-ts_delta(d.close, 24 * 5))     # note: windows are in *bars*, not days

Conventions
-----------
* Time-series operators need a full window (min_periods = d): a value is NaN
  until d bars of history exist, and NaN if the window contains a NaN. This is
  stricter than some libraries but never silently uses short windows.
* Every operator only looks backwards. No operator here can introduce lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

Panel = pd.DataFrame
_CHUNK_ROWS = 2048


def _window(d) -> int:
    if int(d) != d or d < 1:
        raise ValueError(f"window must be a positive integer, got {d!r}")
    return int(d)


def _like(x: Panel, values: np.ndarray) -> Panel:
    return pd.DataFrame(values, index=x.index, columns=x.columns)


def _rolling_window_apply(x: Panel, d: int, fn) -> Panel:
    """Apply fn to (rows, symbols, d) windows, chunked over time to bound memory.
    Windows containing NaN produce NaN."""
    arr = x.to_numpy(dtype="float64")
    out = np.full(arr.shape, np.nan)
    n = arr.shape[0]
    for start in range(0, n, _CHUNK_ROWS):
        end = min(start + _CHUNK_ROWS, n)
        block_start = max(0, start - d + 1)
        block = arr[block_start:end]
        if block.shape[0] < d:
            continue
        win = sliding_window_view(block, d, axis=0)          # (rows, symbols, d)
        res = fn(win).astype("float64")
        res[np.isnan(win).any(axis=-1)] = np.nan
        out[block_start + d - 1:end] = res
    return _like(x, out)


# ---------------------------------------------------------------------------
# Element-wise
# ---------------------------------------------------------------------------
def abs_(x: Panel) -> Panel:
    return x.abs()


def sign(x: Panel) -> Panel:
    return np.sign(x)


def log(x: Panel) -> Panel:
    return np.log(x.where(x > 0))


def signed_power(x: Panel, e: float) -> Panel:
    return np.sign(x) * x.abs() ** e


def s_log_1p(x: Panel) -> Panel:
    return np.sign(x) * np.log1p(x.abs())


def inverse(x: Panel) -> Panel:
    return 1.0 / x.where(x != 0)


def where(cond: Panel, a, b) -> Panel:
    """BRAIN if_else(cond, a, b)."""
    a_ = a if isinstance(a, pd.DataFrame) else pd.DataFrame(a, index=cond.index, columns=cond.columns)
    return a_.where(cond, b)


# ---------------------------------------------------------------------------
# Cross-sectional (per row)
# ---------------------------------------------------------------------------
def rank(x: Panel) -> Panel:
    """Cross-sectional rank scaled to [0, 1] (lowest -> 0, highest -> 1)."""
    r = x.rank(axis=1, method="average").to_numpy()
    n = x.notna().sum(axis=1).to_numpy(dtype="float64")[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 1, (r - 1) / (n - 1), 0.5)
    out[np.isnan(r)] = np.nan
    return _like(x, out)


def zscore(x: Panel) -> Panel:
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return x.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)


def demean(x: Panel) -> Panel:
    """Market neutralisation: subtract the cross-sectional mean."""
    return x.sub(x.mean(axis=1), axis=0)


def scale(x: Panel, a: float = 1.0) -> Panel:
    """Rescale each row so sum(|x|) == a."""
    s = x.abs().sum(axis=1)
    return x.div(s.where(s > 0), axis=0) * a


def winsorize(x: Panel, std: float = 4.0) -> Panel:
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return x.clip(lower=mu - std * sd, upper=mu + std * sd, axis=0)


def group_neutralize(x: Panel, groups: dict[str, str] | pd.Series) -> Panel:
    """Subtract the group mean within each row. `groups` maps symbol -> group."""
    g = pd.Series(groups).reindex(x.columns).fillna("_other")
    return x - x.T.groupby(g).transform("mean").T


def group_rank(x: Panel, groups: dict[str, str] | pd.Series) -> Panel:
    g = pd.Series(groups).reindex(x.columns).fillna("_other")
    parts = [rank(x.loc[:, g[g == k].index]) for k in g.unique()]
    return pd.concat(parts, axis=1)[x.columns]


# ---------------------------------------------------------------------------
# Time series (per column)
# ---------------------------------------------------------------------------
def ts_delay(x: Panel, d: int) -> Panel:
    return x.shift(_window(d))


def ts_delta(x: Panel, d: int) -> Panel:
    return x - x.shift(_window(d))


def ts_returns(x: Panel, d: int = 1) -> Panel:
    return x / x.shift(_window(d)) - 1


def ts_mean(x: Panel, d: int) -> Panel:
    d = _window(d)
    return x.rolling(d, min_periods=d).mean()


def ts_sum(x: Panel, d: int) -> Panel:
    d = _window(d)
    return x.rolling(d, min_periods=d).sum()


def ts_std_dev(x: Panel, d: int) -> Panel:
    d = _window(d)
    return x.rolling(d, min_periods=d).std()


def ts_min(x: Panel, d: int) -> Panel:
    d = _window(d)
    return x.rolling(d, min_periods=d).min()


def ts_max(x: Panel, d: int) -> Panel:
    d = _window(d)
    return x.rolling(d, min_periods=d).max()


def ts_zscore(x: Panel, d: int) -> Panel:
    sd = ts_std_dev(x, d)
    return (x - ts_mean(x, d)) / sd.where(sd > 0)


def ts_scale(x: Panel, d: int) -> Panel:
    lo, hi = ts_min(x, d), ts_max(x, d)
    rng = hi - lo
    return (x - lo) / rng.where(rng > 0)


def ts_corr(x: Panel, y: Panel, d: int) -> Panel:
    d = _window(d)
    return x.rolling(d, min_periods=d).corr(y)


def ts_covariance(x: Panel, y: Panel, d: int) -> Panel:
    d = _window(d)
    return x.rolling(d, min_periods=d).cov(y)


def ts_decay_linear(x: Panel, d: int) -> Panel:
    """Linearly weighted mean, newest bar weight d, oldest weight 1."""
    d = _window(d)
    total = None
    for lag in range(d):
        term = x.shift(lag) * (d - lag)
        total = term if total is None else total + term
    return total / (d * (d + 1) / 2)


def ts_rank(x: Panel, d: int) -> Panel:
    """Rank of today's value within the last d bars, scaled to [0, 1]."""
    d = _window(d)
    if d < 2:
        raise ValueError("ts_rank needs d >= 2")

    def fn(win):
        last = win[..., -1:]
        less = (win < last).sum(axis=-1)
        equal = (win == last).sum(axis=-1)
        return (less + 0.5 * (equal - 1)) / (d - 1)

    return _rolling_window_apply(x, d, fn)


def ts_arg_max(x: Panel, d: int) -> Panel:
    """Bars since the window maximum (0 = the current bar is the max)."""
    d = _window(d)
    return _rolling_window_apply(x, d, lambda w: np.argmax(w[..., ::-1], axis=-1))


def ts_arg_min(x: Panel, d: int) -> Panel:
    d = _window(d)
    return _rolling_window_apply(x, d, lambda w: np.argmin(w[..., ::-1], axis=-1))


def ts_backfill(x: Panel, d: int) -> Panel:
    """Carry the last valid value forward for at most d bars."""
    return x.ffill(limit=_window(d))


def ts_regression_residual(y: Panel, x: Panel, d: int) -> Panel:
    """Residual of a rolling per-symbol regression y ~ a + b x."""
    beta = ts_covariance(y, x, d) / ts_std_dev(x, d) ** 2
    alpha = ts_mean(y, d) - beta * ts_mean(x, d)
    return y - (alpha + beta * x)


# ---------------------------------------------------------------------------
# Event-driven
# ---------------------------------------------------------------------------
def trade_when(trigger, alpha: Panel, exit_) -> Panel:
    """BRAIN trade_when: take alpha when trigger; go flat (NaN) on exit; else hold
    the previous value. Trigger and exit may be panels or scalars."""
    t = _bool_panel(trigger, alpha)
    e = _bool_panel(exit_, alpha) & ~t
    a = alpha.to_numpy(dtype="float64")
    out = np.full(a.shape, np.nan)
    prev = np.full(a.shape[1], np.nan)
    for i in range(a.shape[0]):
        prev = np.where(t[i], a[i], np.where(e[i], np.nan, prev))
        out[i] = prev
    return _like(alpha, out)


def _bool_panel(v, like: Panel) -> np.ndarray:
    if isinstance(v, pd.DataFrame):
        return v.reindex_like(like).fillna(False).to_numpy(dtype=bool)
    return np.full(like.shape, bool(v))
