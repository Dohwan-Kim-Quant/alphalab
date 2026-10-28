"""Test helpers: an in-format-agnostic store and fake Binance archive files."""

from __future__ import annotations

import hashlib
import io
import pickle
import zipfile

import numpy as np
import pandas as pd

from alphalab.store import ParquetStore


class PickleStore(ParquetStore):
    """Same layout and merge logic as ParquetStore, without needing pyarrow."""

    suffix = ".pkl"

    def _read_df(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _write_df(self, df, path):
        with open(path, "wb") as f:
            pickle.dump(df, f)


def kline_csv(start: str, n: int, interval_ms: int = 3_600_000, header: bool = True,
              unit: str = "ms", seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    t0 = int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    lines = []
    if header:
        lines.append("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                     "taker_buy_volume,taker_buy_quote_volume,ignore")
    mult = 1000 if unit == "us" else 1
    for i in range(n):
        ot = (t0 + i * interval_ms) * mult
        ct = (t0 + (i + 1) * interval_ms - 1) * mult
        c = 100 + rng.normal()
        lines.append(f"{ot},{c},{c + 1},{c - 1},{c},{10 + i},{ct},{1000 + i},{5 + i},4,400,0")
    return ("\n".join(lines) + "\n").encode()


def funding_csv(times: list[str], rates: list[float]) -> bytes:
    lines = ["calc_time,funding_interval_hours,last_funding_rate"]
    for t, r in zip(times, rates):
        lines.append(f"{int(pd.Timestamp(t, tz='UTC').value // 1_000_000)},8,{r}")
    return ("\n".join(lines) + "\n").encode()


def zip_bytes(name: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, payload)
    return buf.getvalue()


def checksum_for(data: bytes, name: str) -> bytes:
    return f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode()
