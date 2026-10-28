"""Historical USDⓈ-M futures data from Binance's public archive (data.binance.vision).

Design notes
------------
* Symbol discovery uses the archive's S3 listing, which still contains symbols
  that were delisted. Downloading those avoids survivorship bias: a backtest
  run only on today's listed coins silently drops every coin that died.
* Monthly files cover closed months; daily files fill the current month.
  When a monthly file later appears, its rows simply overwrite the daily ones
  (upsert dedupes by timestamp).
* Every zip is verified against its published SHA-256 .CHECKSUM file.
* A manifest records ingested files so re-runs only fetch what's new.
* Network I/O is injected (`fetch_text`, `fetch_bytes`), so the logic is
  testable offline and the aiohttp dependency stays at the edge.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable

import pandas as pd

from .store import FUNDING_COLUMNS, KLINE_COLUMNS, Manifest, ParquetStore
from .timeutil import to_utc_index, to_utc_timestamp

log = logging.getLogger(__name__)

LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILE_BASE = "https://data.binance.vision/"
UM_PREFIX = "data/futures/um"

FetchText = Callable[[str, dict | None], Awaitable[str]]
FetchBytes = Callable[[str], Awaitable[bytes]]


class NotFoundError(Exception):
    pass


class ChecksumError(Exception):
    pass


# ---------------------------------------------------------------------------
# S3 listing
# ---------------------------------------------------------------------------
@dataclass
class ListingPage:
    prefixes: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    next_marker: str | None = None


def parse_listing(xml_text: str) -> ListingPage:
    """Parse one page of an S3 ListObjects (v1) response."""
    root = ET.fromstring(xml_text)
    ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    prefixes = [cp.findtext(f"{ns}Prefix") for cp in root.findall(f"{ns}CommonPrefixes")]
    keys = [c.findtext(f"{ns}Key") for c in root.findall(f"{ns}Contents")]
    prefixes = [p for p in prefixes if p]
    keys = [k for k in keys if k]
    truncated = (root.findtext(f"{ns}IsTruncated") or "").lower() == "true"
    marker = None
    if truncated:
        # NextMarker is only sent when a delimiter is used; otherwise use the last entry.
        marker = root.findtext(f"{ns}NextMarker") or max(prefixes + keys, default=None)
    return ListingPage(prefixes, keys, marker)


_DATE_RE = re.compile(r"-(\d{4}-\d{2}(?:-\d{2})?)\.zip$")


def key_period(key: str) -> str | None:
    """'.../BTCUSDT-1h-2024-01.zip' -> '2024-01'; daily keys -> '2024-01-15'."""
    m = _DATE_RE.search(key)
    return m.group(1) if m else None


def select_keys(
    monthly_keys: Iterable[str],
    daily_keys: Iterable[str],
    start: pd.Timestamp | None,
    done: Manifest | set[str],
) -> list[str]:
    """Pick the archive files to fetch: all monthly files, plus daily files only
    for months that have no monthly file yet; drop files before `start` and
    files already ingested."""
    monthly = [k for k in monthly_keys if k.endswith(".zip")]
    covered = {key_period(k) for k in monthly}
    daily = [
        k for k in daily_keys
        if k.endswith(".zip") and (p := key_period(k)) and p[:7] not in covered
    ]

    def ends_before_start(k: str) -> bool:
        if start is None:
            return False
        period = key_period(k)
        if period is None:
            return False
        first = pd.Timestamp(period + ("-01" if len(period) == 7 else ""), tz="UTC")
        last = first + (pd.offsets.MonthEnd(0) if len(period) == 7 else pd.Timedelta(0))
        return last.normalize() + pd.Timedelta(days=1) <= start

    return sorted(k for k in monthly + daily if k not in done and not ends_before_start(k))


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------
_KLINE_RAW = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def unzip_single_csv(data: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError("zip contains no csv")
        return zf.read(names[0])


def _has_header(raw: bytes) -> bool:
    first = raw.lstrip()[:1]
    return bool(first) and not (first.isdigit() or first == b"-")


def parse_kline_csv(raw: bytes) -> pd.DataFrame:
    """Parse a kline CSV (with or without header) into the store schema."""
    if not raw.strip():
        return pd.DataFrame(columns=KLINE_COLUMNS)
    df = pd.read_csv(
        io.BytesIO(raw), header=None, skiprows=1 if _has_header(raw) else 0,
        names=_KLINE_RAW,
    )
    out = df[KLINE_COLUMNS].astype("float64")
    out["trades"] = df["trades"].astype("int64")
    out.index = to_utc_index(df["open_time"])
    out.index.name = "open_time"
    return out.sort_index()


def parse_funding_csv(raw: bytes) -> pd.DataFrame:
    """Parse a fundingRate CSV. Current files: calc_time, funding_interval_hours,
    last_funding_rate. A two-column variant (time, rate) is also accepted."""
    if not raw.strip():
        return pd.DataFrame(columns=FUNDING_COLUMNS)
    df = pd.read_csv(io.BytesIO(raw), header=None, skiprows=1 if _has_header(raw) else 0)
    rate_col = df.columns[-1]
    out = pd.DataFrame({"funding_rate": df[rate_col].astype("float64").to_numpy()})
    out.index = to_utc_index(df[df.columns[0]])
    out.index.name = "funding_time"
    return out.sort_index()


def verify_checksum(data: bytes, checksum_text: str) -> None:
    expected = checksum_text.strip().split()[0].lower()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ChecksumError(f"sha256 mismatch: expected {expected}, got {actual}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class BinanceVisionClient:
    def __init__(self, fetch_text: FetchText, fetch_bytes: FetchBytes,
                 concurrency: int = 8, verify: bool = True):
        self._fetch_text = fetch_text
        self._fetch_bytes = fetch_bytes
        self._sem = asyncio.Semaphore(concurrency)
        self.verify = verify

    async def list_all(self, prefix: str) -> ListingPage:
        """List one 'directory', following pagination."""
        result = ListingPage()
        marker: str | None = None
        while True:
            params = {"delimiter": "/", "prefix": prefix}
            if marker:
                params["marker"] = marker
            async with self._sem:
                page = parse_listing(await self._fetch_text(LIST_URL, params))
            result.prefixes += page.prefixes
            result.keys += page.keys
            if not page.next_marker or page.next_marker == marker:
                return result
            marker = page.next_marker

    async def list_symbols(self, quote: str = "USDT") -> list[str]:
        """All perpetual symbols ever archived (live and delisted) for a quote asset."""
        found: set[str] = set()
        for freq in ("monthly", "daily"):
            page = await self.list_all(f"{UM_PREFIX}/{freq}/klines/")
            for p in page.prefixes:
                sym = p.rstrip("/").rsplit("/", 1)[-1]
                # Dated delivery contracts look like BTCUSDT_250328; skip them.
                if sym.endswith(quote) and "_" not in sym:
                    found.add(sym)
        return sorted(found)

    async def download(self, key: str) -> bytes:
        async with self._sem:
            if not self.verify:
                return await self._fetch_bytes(FILE_BASE + key)
            data, checksum = await asyncio.gather(
                self._fetch_bytes(FILE_BASE + key),
                self._fetch_bytes(FILE_BASE + key + ".CHECKSUM"),
            )
        verify_checksum(data, checksum.decode())
        return data


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class DownloadConfig:
    interval: str = "1h"
    start: pd.Timestamp | None = None
    symbols: list[str] | None = None
    quote: str = "USDT"
    include_funding: bool = True
    symbol_concurrency: int = 4


async def _ingest(client: BinanceVisionClient, store: ParquetStore, manifest: Manifest,
                  keys: list[str], parser, path) -> int:
    if not keys:
        return 0
    results = await asyncio.gather(*(client.download(k) for k in keys), return_exceptions=True)
    frames, ok_keys = [], []
    for key, res in zip(keys, results):
        if isinstance(res, NotFoundError):
            log.warning("missing in archive, skipped: %s", key)
            continue
        if isinstance(res, BaseException):
            log.error("failed %s: %r", key, res)
            continue
        frames.append(parser(unzip_single_csv(res)))
        ok_keys.append(key)
    if frames:
        await asyncio.to_thread(store.upsert, path, pd.concat(frames))
        manifest.add_many(ok_keys)  # only after the data is safely on disk
    return len(ok_keys)


async def sync_symbol(client: BinanceVisionClient, store: ParquetStore, cfg: DownloadConfig,
                      symbol: str, kline_manifest: Manifest, funding_manifest: Manifest) -> dict:
    iv = cfg.interval
    monthly, daily = await asyncio.gather(
        client.list_all(f"{UM_PREFIX}/monthly/klines/{symbol}/{iv}/"),
        client.list_all(f"{UM_PREFIX}/daily/klines/{symbol}/{iv}/"),
    )
    keys = select_keys(monthly.keys, daily.keys, cfg.start, kline_manifest)
    n_k = await _ingest(client, store, kline_manifest, keys, parse_kline_csv,
                        store.kline_path(symbol, iv))
    n_f = 0
    if cfg.include_funding:
        fm = await client.list_all(f"{UM_PREFIX}/monthly/fundingRate/{symbol}/")
        fkeys = select_keys(fm.keys, [], cfg.start, funding_manifest)
        n_f = await _ingest(client, store, funding_manifest, fkeys, parse_funding_csv,
                            store.funding_path(symbol))
    if n_k or n_f:
        log.info("%s: +%d kline files, +%d funding files", symbol, n_k, n_f)
    return {"symbol": symbol, "kline_files": n_k, "funding_files": n_f}


async def download_history(client: BinanceVisionClient, store: ParquetStore,
                           cfg: DownloadConfig) -> list[dict]:
    if cfg.start is not None:
        cfg.start = to_utc_timestamp(cfg.start)
    symbols = cfg.symbols or await client.list_symbols(cfg.quote)
    log.info("syncing %d symbols (%s)", len(symbols), cfg.interval)
    kline_manifest = Manifest(store.manifest_path(f"klines_{cfg.interval}"))
    funding_manifest = Manifest(store.manifest_path("funding"))
    sem = asyncio.Semaphore(cfg.symbol_concurrency)

    async def one(sym: str) -> dict:
        async with sem:
            try:
                return await sync_symbol(client, store, cfg, sym, kline_manifest, funding_manifest)
            except Exception as e:  # keep going; report at the end
                log.error("%s failed: %r", sym, e)
                return {"symbol": sym, "error": repr(e)}

    return await asyncio.gather(*(one(s) for s in symbols))


# ---------------------------------------------------------------------------
# aiohttp edge
# ---------------------------------------------------------------------------
def make_aiohttp_fetchers(session, retries: int = 4) -> tuple[FetchText, FetchBytes]:
    """Build fetch functions on an aiohttp.ClientSession with retry + backoff."""
    import aiohttp

    async def _get(url: str, params: dict | None) -> bytes:
        delay = 1.0
        for attempt in range(retries + 1):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 404:
                        raise NotFoundError(url)
                    if resp.status in (418, 429) or resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status)
                    resp.raise_for_status()
                    return await resp.read()
            except NotFoundError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == retries:
                    raise
                log.debug("retry %s after %r", url, e)
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    async def fetch_text(url: str, params: dict | None = None) -> str:
        return (await _get(url, params)).decode()

    async def fetch_bytes(url: str) -> bytes:
        return await _get(url, None)

    return fetch_text, fetch_bytes


async def run_download(store: ParquetStore, cfg: DownloadConfig,
                       concurrency: int = 8, verify: bool = True) -> list[dict]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        fetch_text, fetch_bytes = make_aiohttp_fetchers(session)
        client = BinanceVisionClient(fetch_text, fetch_bytes, concurrency, verify)
        return await download_history(client, store, cfg)
