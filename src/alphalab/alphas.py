"""Example alphas, written the way you'd write them on BRAIN.

Each docstring gives the BRAIN-style expression. Windows are given in days and
converted to bars, so every alpha works for any bar interval.

These are *starting points to test*, not claims that they work on crypto. The
point of the project is to find out which ideas survive the move from equities.

Add your own: decorate a function taking MarketData and returning a panel, or
point the CLI at any file with `--alpha path/to/my_alphas.py:my_func`.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .data import MarketData
from .operators import (
    rank, ts_corr, ts_delay, ts_mean, ts_returns, ts_std_dev, ts_zscore,
)
from .timeutil import bars_per_day

AlphaFn = Callable[[MarketData], pd.DataFrame]
REGISTRY: dict[str, AlphaFn] = {}


def register(name: str):
    def deco(fn: AlphaFn) -> AlphaFn:
        REGISTRY[name] = fn
        return fn
    return deco


def bars(d: MarketData, days: float) -> int:
    return max(1, int(round(days * bars_per_day(d.interval))))


@register("reversal_1d")
def reversal_1d(d: MarketData) -> pd.DataFrame:
    """rank(-ts_returns(close, 1d)) — short-term reversal: fade yesterday's movers."""
    return rank(-ts_returns(d.close, bars(d, 1)))


@register("momentum_30d_skip_1d")
def momentum_30d_skip_1d(d: MarketData) -> pd.DataFrame:
    """rank(ts_delay(close, 1d) / ts_delay(close, 30d) - 1) — momentum, skipping the
    last day so it doesn't collide with short-term reversal."""
    return rank(ts_delay(d.close, bars(d, 1)) / ts_delay(d.close, bars(d, 30)) - 1)


@register("open_volume_corr")
def open_volume_corr(d: MarketData) -> pd.DataFrame:
    """-ts_corr(open, volume, 10d) — Alpha#6 from '101 Formulaic Alphas'
    (Kakushadze, 2016), a classic price-volume signal from equities."""
    return -ts_corr(d.open, d.volume, bars(d, 10))


@register("low_volatility")
def low_volatility(d: MarketData) -> pd.DataFrame:
    """rank(-ts_std_dev(returns, 14d)) — low-volatility anomaly."""
    return rank(-ts_std_dev(d.returns, bars(d, 14)))


@register("vwap_deviation")
def vwap_deviation(d: MarketData) -> pd.DataFrame:
    """rank(-ts_mean((close - vwap) / vwap, 1d)) — fade closes far above the
    volume-weighted price."""
    dev = (d.close - d.vwap) / d.vwap
    return rank(-ts_mean(dev, bars(d, 1)))


@register("taker_flow")
def taker_flow(d: MarketData) -> pd.DataFrame:
    """rank(ts_zscore(taker_buy_ratio, 7d)) — crypto-only: unusually aggressive
    buying. No equity analogue on BRAIN; test both signs."""
    return rank(ts_zscore(ts_mean(d.taker_buy_ratio, bars(d, 1)), bars(d, 7)))


@register("funding_carry")
def funding_carry(d: MarketData) -> pd.DataFrame:
    """rank(-funding_rate) — crypto-only carry: short contracts where longs pay
    the most (crowded longs), long those where shorts pay."""
    return rank(-d.funding_rate)


def load_alpha(spec: str) -> AlphaFn:
    """Resolve 'name' from the registry or 'path/to/file.py:function'."""
    if spec in REGISTRY:
        return REGISTRY[spec]
    if ":" in spec:
        import importlib.util
        from pathlib import Path

        path, func = spec.rsplit(":", 1)
        mod_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
        if mod_spec is None or mod_spec.loader is None:
            raise ValueError(f"cannot load {path}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        return getattr(module, func)
    raise KeyError(f"unknown alpha {spec!r}; available: {', '.join(sorted(REGISTRY))}")
