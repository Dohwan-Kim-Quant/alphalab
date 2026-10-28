"""Tradable universe selection (point-in-time, no lookahead)."""

from __future__ import annotations

import pandas as pd

from .data import MarketData
from .timeutil import bars_per_day


def top_liquidity(data: MarketData, top_n: int = 50, lookback_days: float = 30,
                  min_history_days: float = 30) -> pd.DataFrame:
    """Boolean panel: the top_n symbols by trailing average quote volume.

    Only data up to and including bar t is used at row t. A symbol also needs
    `min_history_days` of prior bars, which keeps fresh listings (often wild
    and illiquid in their first weeks) out of the book.
    """
    bpd = bars_per_day(data.interval)
    lookback = max(1, int(lookback_days * bpd))
    min_hist = int(min_history_days * bpd)
    adv = data.quote_volume.rolling(lookback, min_periods=max(1, lookback // 2)).mean()
    history = data.close.notna().cumsum()
    eligible = data.close.notna() & (history >= min_hist) & adv.notna()
    order = adv.where(eligible).rank(axis=1, ascending=False, method="first")
    return order <= top_n
