"""Synthetic market with a known, planted signal.

Used by the tests and by `alphalab demo` to check the whole research stack
end to end without network access: if the backtester can't find a signal we
planted on purpose, it can't be trusted on real data either.

Model: r[i,t] = beta_i * m[t] + e[i,t] - strength * mean(e[i, t-L .. t-1])
i.e. part of each coin's idiosyncratic move over the last L bars reverses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MarketData
from .timeutil import interval_to_timedelta


def make_synthetic_market(n_symbols: int = 60, n_bars: int = 24 * 365, interval: str = "1h",
                          reversal_window: int = 24, strength: float = 0.1,
                          seed: int = 7) -> MarketData:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_bars, freq=interval_to_timedelta(interval),
                          tz="UTC").as_unit("ms")
    symbols = [f"C{i:02d}USDT" for i in range(n_symbols)]

    market = rng.standard_t(4, n_bars) * 0.006
    beta = rng.uniform(0.6, 1.6, n_symbols)
    idio = rng.standard_t(4, (n_bars, n_symbols)) * 0.008
    past = pd.DataFrame(idio).rolling(reversal_window).mean().shift(1).fillna(0).to_numpy()
    rets = market[:, None] * beta + idio - strength * past

    close = 100 * np.exp(np.cumsum(np.log1p(rets), axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    spread = np.abs(rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    size = rng.lognormal(0, 1, n_symbols)
    volume = rng.lognormal(10, 0.5, close.shape) * size
    quote_volume = volume * (open_ + close) / 2
    taker_share = np.clip(0.5 + 0.1 * rng.standard_normal(close.shape), 0.05, 0.95)

    funding = np.zeros(close.shape)
    settle = (index.hour % 8 == 7)          # 1h bars 07:00/15:00/23:00 close on settlements
    funding[settle] = rng.normal(0.0001, 0.0002, (settle.sum(), n_symbols))
    funding_rate = pd.DataFrame(np.where(settle[:, None], funding, np.nan)).ffill().to_numpy()

    def panel(a):
        return pd.DataFrame(a, index=index, columns=symbols)

    return MarketData(
        interval=interval, open=panel(open_), high=panel(high), low=panel(low),
        close=panel(close), volume=panel(volume), quote_volume=panel(quote_volume),
        trades=panel(np.round(volume / 10)), taker_buy_volume=panel(volume * taker_share),
        taker_buy_quote_volume=panel(quote_volume * taker_share),
        funding=panel(funding), funding_rate=panel(funding_rate),
    )
