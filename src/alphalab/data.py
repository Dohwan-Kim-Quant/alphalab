"""Load stored data into aligned wide panels for research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .store import KLINE_COLUMNS, ParquetStore
from .timeutil import interval_to_timedelta, to_utc_timestamp


@dataclass
class MarketData:
    """Aligned panels (rows = bar open time on a regular grid, columns = symbols).

    Data in row t describes bar t, which closes at t + interval. A signal
    computed from row t is therefore only actionable after that close.
    """

    interval: str
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    quote_volume: pd.DataFrame
    trades: pd.DataFrame
    taker_buy_volume: pd.DataFrame
    taker_buy_quote_volume: pd.DataFrame
    funding: pd.DataFrame        # rate settled at the close of bar t (0 if none)
    funding_rate: pd.DataFrame   # last settled rate known at the close of bar t

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def returns(self) -> pd.DataFrame:
        """Close-to-close return of bar t (realised at bar t's close)."""
        return self.close / self.close.shift(1) - 1

    @property
    def vwap(self) -> pd.DataFrame:
        return self.quote_volume / self.volume.where(self.volume > 0)

    @property
    def taker_buy_ratio(self) -> pd.DataFrame:
        """Share of quote volume initiated by aggressive buyers (0.5 = balanced)."""
        return self.taker_buy_quote_volume / self.quote_volume.where(self.quote_volume > 0)

    def slice(self, start=None, end=None) -> "MarketData":
        kw = {f: getattr(self, f).loc[start:end] for f in _PANEL_FIELDS}
        return MarketData(interval=self.interval, **kw)


_PANEL_FIELDS = KLINE_COLUMNS + ["funding", "funding_rate"]


def bar_of_event(event_times: pd.DatetimeIndex, interval: str) -> pd.DatetimeIndex:
    """Map event timestamps to the bar whose close they fall on or inside:
    open < T <= open + interval. A funding settlement at exactly 08:00 belongs
    to the 07:00 bar of 1h data, i.e. it is paid by whoever held through 08:00."""
    return (event_times - pd.Timedelta(1, "ms")).floor(interval_to_timedelta(interval))


def align_funding(events: dict[str, pd.Series], index: pd.DatetimeIndex,
                  interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    charged, known = {}, {}
    for sym, s in events.items():
        s = s.dropna()
        if s.empty:
            continue
        bars = bar_of_event(s.index, interval)
        charged[sym] = s.groupby(bars).sum().reindex(index, fill_value=0.0)
        known[sym] = s.groupby(bars).last().reindex(index).ffill()
    return (pd.DataFrame(charged, index=index), pd.DataFrame(known, index=index))


def build_market_data(klines: dict[str, pd.DataFrame], funding: dict[str, pd.Series],
                      interval: str, start=None, end=None) -> MarketData:
    klines = {s: df for s, df in klines.items() if not df.empty}
    if not klines:
        raise ValueError("no kline data loaded")
    lo = min(df.index.min() for df in klines.values())
    hi = max(df.index.max() for df in klines.values())
    if start is not None:
        lo = max(lo, to_utc_timestamp(start))
    if end is not None:
        hi = min(hi, to_utc_timestamp(end))
    # A regular grid makes "d bars ago" mean the same wall-clock time for every symbol.
    index = pd.date_range(lo, hi, freq=interval_to_timedelta(interval), tz="UTC").as_unit("ms")
    symbols = sorted(klines)
    panels = {
        f: pd.DataFrame({s: klines[s][f] for s in symbols}).reindex(index)
        for f in KLINE_COLUMNS
    }
    charged, known = align_funding({s: funding[s] for s in symbols if s in funding},
                                   index, interval)
    panels["funding"] = charged.reindex(columns=symbols).fillna(0.0)
    panels["funding_rate"] = known.reindex(columns=symbols)
    return MarketData(interval=interval, **panels)


def load_market_data(store: ParquetStore, interval: str, symbols: list[str] | None = None,
                     start=None, end=None) -> MarketData:
    symbols = symbols or store.kline_symbols(interval)
    klines = {s: store.read(store.kline_path(s, interval)) for s in symbols}
    funding = {}
    for s in symbols:
        df = store.read(store.funding_path(s))
        if not df.empty:
            funding[s] = df["funding_rate"]
    return build_market_data(klines, funding, interval, start, end)
