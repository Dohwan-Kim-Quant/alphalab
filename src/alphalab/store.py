"""On-disk storage: one Parquet file per (dataset, symbol).

Layout under the data root:

    klines/<interval>/<SYMBOL>.parquet   index: bar open time (UTC)
    funding/<SYMBOL>.parquet             index: funding settlement time (UTC)
    _manifest/*.txt                      source files already ingested

Why one file per symbol: research reads whole histories per symbol, a few
hundred files load quickly, and each symbol has exactly one writer at a time.

Writes are atomic (write to a temp file, then os.replace), so a crash never
leaves a half-written file. Two *processes* writing the same symbol at once can
still lose an update (last writer wins), so don't run the historical
downloader and the realtime collector on the same interval simultaneously.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

KLINE_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_volume",
    "trades", "taker_buy_volume", "taker_buy_quote_volume",
]
FUNDING_COLUMNS = ["funding_rate"]


def merge_frames(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Union two time-indexed frames; on duplicate timestamps the new row wins."""
    frames = [f for f in (old, new) if f is not None and not f.empty]
    if not frames:
        return new.iloc[0:0]
    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


class ParquetStore:
    suffix = ".parquet"

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    # ---- paths -----------------------------------------------------------
    def kline_path(self, symbol: str, interval: str) -> Path:
        return self.root / "klines" / interval / f"{symbol}{self.suffix}"

    def funding_path(self, symbol: str) -> Path:
        return self.root / "funding" / f"{symbol}{self.suffix}"

    def manifest_path(self, name: str) -> Path:
        return self.root / "_manifest" / f"{name}.txt"

    # ---- format hooks (overridden in tests) -----------------------------
    def _read_df(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _write_df(self, df: pd.DataFrame, path: Path) -> None:
        df.to_parquet(path)

    # ---- API -------------------------------------------------------------
    def read(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return self._read_df(path)

    def upsert(self, path: Path, new: pd.DataFrame) -> int:
        """Merge `new` into the file at `path`. Returns the resulting row count."""
        if new.empty:
            return 0
        old = self._read_df(path) if path.exists() else None
        merged = merge_frames(old, new.sort_index())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        self._write_df(merged, tmp)
        os.replace(tmp, path)
        return len(merged)

    def last_timestamp(self, path: Path) -> pd.Timestamp | None:
        df = self.read(path)
        return None if df.empty else df.index.max()

    def kline_symbols(self, interval: str) -> list[str]:
        d = self.root / "klines" / interval
        return sorted(p.stem for p in d.glob(f"*{self.suffix}")) if d.exists() else []

    def funding_symbols(self) -> list[str]:
        d = self.root / "funding"
        return sorted(p.stem for p in d.glob(f"*{self.suffix}")) if d.exists() else []


class Manifest:
    """Append-only list of ingested source keys, so re-runs skip finished files."""

    def __init__(self, path: Path):
        self.path = path
        self._keys: set[str] = set()
        if path.exists():
            self._keys = {line.strip() for line in path.read_text().splitlines() if line.strip()}

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def add_many(self, keys) -> None:
        new = [k for k in keys if k not in self._keys]
        if not new:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write("".join(k + "\n" for k in new))
        self._keys.update(new)
