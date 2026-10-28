"""Realtime kline + funding collector for Binance USDⓈ-M futures (asyncio).

How it stays gap-free
---------------------
1. Each websocket connection subscribes to up to MAX_STREAMS_PER_CONN kline
   streams and keeps only *closed* bars (k.x == true).
2. Right after every (re)connect, the collector backfills over REST from the
   last stored bar up to now. Any bars missed while disconnected are filled,
   and duplicates with the websocket are harmless (upsert dedupes).
3. Binance drops connections after 24h, so connections are recycled proactively
   before that; the backfill in step 2 covers the switch-over.
4. Bars are buffered in memory and flushed to Parquet every few seconds in a
   worker thread, so disk I/O never blocks the event loop.
5. Funding settlements are polled over REST periodically (the public archive
   only publishes them monthly).

Binance has been reorganising its websocket endpoints, so the base URL is
configurable; check the current USDⓈ-M docs if connecting fails.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from typing import Awaitable, Callable

import pandas as pd

from .store import KLINE_COLUMNS, ParquetStore
from .timeutil import interval_to_ms, to_utc_index

log = logging.getLogger(__name__)

WS_BASE = "wss://fstream.binance.com/stream"
REST_BASE = "https://fapi.binance.com"
MAX_STREAMS_PER_CONN = 200
KLINE_LIMIT = 1500
FUNDING_LIMIT = 1000

RestGet = Callable[[str, dict], Awaitable[object]]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def parse_ws_kline(msg: dict) -> tuple[str, dict] | None:
    """Return (symbol, row) for a *closed* kline event, else None."""
    data = msg.get("data", msg)
    if data.get("e") != "kline":
        return None
    k = data["k"]
    if not k.get("x"):
        return None
    row = {
        "open_time": int(k["t"]),
        "open": float(k["o"]), "high": float(k["h"]), "low": float(k["l"]),
        "close": float(k["c"]), "volume": float(k["v"]), "quote_volume": float(k["q"]),
        "trades": int(k["n"]), "taker_buy_volume": float(k["V"]),
        "taker_buy_quote_volume": float(k["Q"]),
    }
    return k["s"], row


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    idx = to_utc_index(df.pop("open_time"))
    df.index = idx
    df.index.name = "open_time"
    return df[KLINE_COLUMNS]


def parse_rest_klines(rows: list[list]) -> pd.DataFrame:
    """REST /fapi/v1/klines arrays -> store schema."""
    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    return rows_to_frame([
        {
            "open_time": int(r[0]), "open": float(r[1]), "high": float(r[2]),
            "low": float(r[3]), "close": float(r[4]), "volume": float(r[5]),
            "quote_volume": float(r[7]), "trades": int(r[8]),
            "taker_buy_volume": float(r[9]), "taker_buy_quote_volume": float(r[10]),
        }
        for r in rows
    ])


def parse_rest_funding(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame({"funding_rate": [float(r["fundingRate"]) for r in rows]})
    df.index = to_utc_index([int(r["fundingTime"]) for r in rows])
    df.index.name = "funding_time"
    return df


def chunked(items: list, n: int) -> list[list]:
    return [items[i:i + n] for i in range(0, len(items), n)]


def stream_url(base: str, symbols: list[str], interval: str) -> str:
    streams = "/".join(f"{s.lower()}@kline_{interval}" for s in symbols)
    return f"{base}?streams={streams}"


def closed_only(df: pd.DataFrame, interval: str, now_ms: int) -> pd.DataFrame:
    """Drop the still-forming bar: a bar is closed once open_time + interval <= now."""
    if df.empty:
        return df
    open_ms = df.index.as_unit("ms").asi8
    return df[open_ms + interval_to_ms(interval) <= now_ms]


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------
async def fetch_klines_since(rest_get: RestGet, symbol: str, interval: str,
                             since_ms: int, now_ms: int) -> pd.DataFrame:
    step = interval_to_ms(interval)
    rows: list[list] = []
    start = since_ms
    while start < now_ms:
        batch = await rest_get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval, "startTime": start, "limit": KLINE_LIMIT,
        })
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < KLINE_LIMIT:
            break
        start = int(batch[-1][0]) + step
    return closed_only(parse_rest_klines(rows), interval, now_ms)


async def fetch_funding_since(rest_get: RestGet, symbol: str, since_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    start = since_ms
    while True:
        batch = await rest_get("/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": start, "limit": FUNDING_LIMIT,
        })
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < FUNDING_LIMIT:
            break
        start = int(batch[-1]["fundingTime"]) + 1
    return parse_rest_funding(rows)


async def list_perpetuals(rest_get: RestGet, quote: str = "USDT") -> list[str]:
    info = await rest_get("/fapi/v1/exchangeInfo", {})
    return sorted(
        s["symbol"] for s in info["symbols"]
        if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == quote
        and s.get("status") == "TRADING"
    )


async def top_by_volume(rest_get: RestGet, n: int, quote: str = "USDT") -> list[str]:
    tradable = set(await list_perpetuals(rest_get, quote))
    tickers = await rest_get("/fapi/v1/ticker/24hr", {})
    ranked = sorted(
        (t for t in tickers if t["symbol"] in tradable),
        key=lambda t: float(t["quoteVolume"]), reverse=True,
    )
    return [t["symbol"] for t in ranked[:n]]


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class KlineCollector:
    def __init__(self, store: ParquetStore, symbols: list[str], interval: str,
                 rest_get: RestGet, ws_connect, *, ws_base: str = WS_BASE,
                 flush_seconds: float = 5.0, funding_poll_seconds: float = 900.0,
                 initial_lookback: pd.Timedelta = pd.Timedelta(days=7),
                 max_conn_age: float = 23 * 3600, receive_timeout: float = 60.0):
        self.store = store
        self.symbols = symbols
        self.interval = interval
        self.rest_get = rest_get
        self.ws_connect = ws_connect          # async ctx manager factory: url -> ws
        self.ws_base = ws_base
        self.flush_seconds = flush_seconds
        self.funding_poll_seconds = funding_poll_seconds
        self.initial_lookback = initial_lookback
        self.max_conn_age = max_conn_age
        self.receive_timeout = receive_timeout
        self._buffer: dict[str, list[dict]] = defaultdict(list)
        self._backfilled: dict[str, list[pd.DataFrame]] = defaultdict(list)
        self._rest_sem = asyncio.Semaphore(5)
        self._stop = asyncio.Event()
        self._bg_tasks: set[asyncio.Task] = set()
        self.bars_written = 0

    def stop(self) -> None:
        self._stop.set()

    # -- storage ------------------------------------------------------------
    def _last_open_ms(self, symbol: str) -> int | None:
        ts = self.store.last_timestamp(self.store.kline_path(symbol, self.interval))
        return None if ts is None else int(ts.value // 1_000_000)

    async def flush(self) -> None:
        buffer, self._buffer = self._buffer, defaultdict(list)
        backfilled, self._backfilled = self._backfilled, defaultdict(list)
        for sym in set(buffer) | set(backfilled):
            parts = list(backfilled.get(sym, []))
            if buffer.get(sym):
                parts.append(rows_to_frame(buffer[sym]))
            df = pd.concat(parts)
            try:
                await asyncio.to_thread(self.store.upsert,
                                        self.store.kline_path(sym, self.interval), df)
                self.bars_written += len(df)
            except Exception as e:
                log.error("flush failed for %s, will retry: %r", sym, e)
                self._backfilled[sym].append(df)

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), self.flush_seconds)
            except asyncio.TimeoutError:
                pass
            await self.flush()

    # -- REST backfill -------------------------------------------------------
    async def _rest(self, path: str, params: dict):
        async with self._rest_sem:
            res = await self.rest_get(path, params)
            await asyncio.sleep(0.1)  # stay well under the request-weight limit
            return res

    async def backfill(self, symbols: list[str]) -> None:
        now_ms = int(time.time() * 1000)
        step = interval_to_ms(self.interval)
        default_since = now_ms - int(self.initial_lookback.total_seconds() * 1000)

        async def one(sym: str) -> None:
            last = await asyncio.to_thread(self._last_open_ms, sym)
            since = default_since if last is None else last + step
            try:
                df = await fetch_klines_since(self._rest, sym, self.interval, since, now_ms)
            except Exception as e:
                log.warning("backfill %s failed: %r", sym, e)
                return
            if not df.empty:
                self._backfilled[sym].append(df)
                log.info("backfilled %s: %d bars", sym, len(df))

        await asyncio.gather(*(one(s) for s in symbols))

    async def _funding_loop(self) -> None:
        while not self._stop.is_set():
            for sym in self.symbols:
                if self._stop.is_set():
                    return
                path = self.store.funding_path(sym)
                last = await asyncio.to_thread(self.store.last_timestamp, path)
                since = 0 if last is None else int(last.value // 1_000_000) + 1
                try:
                    df = await fetch_funding_since(self._rest, sym, since)
                    if not df.empty:
                        await asyncio.to_thread(self.store.upsert, path, df)
                except Exception as e:
                    log.warning("funding poll %s failed: %r", sym, e)
            try:
                await asyncio.wait_for(self._stop.wait(), self.funding_poll_seconds)
            except asyncio.TimeoutError:
                pass

    # -- websocket -----------------------------------------------------------
    async def _connection(self, symbols: list[str]) -> None:
        url = stream_url(self.ws_base, symbols, self.interval)
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with self.ws_connect(url, self.receive_timeout) as ws:
                    log.info("connected: %d streams", len(symbols))
                    backoff = 1.0
                    # Subscribe first, then backfill: nothing falls between the two.
                    task = asyncio.create_task(self.backfill(symbols))
                    self._bg_tasks.add(task)  # keep a reference so it isn't GC'd
                    task.add_done_callback(self._bg_tasks.discard)
                    opened = time.monotonic()
                    async for msg in ws:
                        parsed = parse_ws_kline(msg)
                        if parsed:
                            sym, row = parsed
                            self._buffer[sym].append(row)
                        if self._stop.is_set():
                            return
                        if time.monotonic() - opened > self.max_conn_age:
                            log.info("recycling connection before 24h limit")
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("connection error: %r; reconnecting in %.0fs", e, backoff)
                try:
                    await asyncio.wait_for(self._stop.wait(), backoff + random.random())
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60.0)

    async def run(self) -> None:
        tasks = [asyncio.create_task(self._connection(chunk))
                 for chunk in chunked(self.symbols, MAX_STREAMS_PER_CONN)]
        tasks.append(asyncio.create_task(self._funding_loop()))
        flusher = asyncio.create_task(self._flush_loop())
        try:
            await self._stop.wait()
        finally:
            self._stop.set()  # also reached on cancellation (e.g. Ctrl+C on Windows)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await flusher          # final flush happens after stop is set
            await self.flush()


# ---------------------------------------------------------------------------
# aiohttp edge
# ---------------------------------------------------------------------------
def make_aiohttp_rest(session, base: str = REST_BASE) -> RestGet:
    import aiohttp

    async def rest_get(path: str, params: dict):
        delay = 1.0
        for attempt in range(5):
            try:
                async with session.get(base + path, params=params) as resp:
                    if resp.status in (418, 429):
                        wait = float(resp.headers.get("Retry-After", 30))
                        log.warning("rate limited, sleeping %.0fs", wait)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == 4:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError(f"REST {path} failed after retries")

    return rest_get


def make_aiohttp_ws(session):
    """Return url -> async context manager yielding decoded JSON messages."""
    import contextlib
    import json

    import aiohttp

    @contextlib.asynccontextmanager
    async def connect(url: str, receive_timeout: float):
        # autoping answers Binance's server pings; receive_timeout catches dead links.
        async with session.ws_connect(url, autoping=True, receive_timeout=receive_timeout,
                                      max_msg_size=0) as ws:
            async def messages():
                async for m in ws:
                    if m.type == aiohttp.WSMsgType.TEXT:
                        yield json.loads(m.data)
                    elif m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        raise ConnectionError(f"websocket closed: {m.type}")
                raise ConnectionError("websocket stream ended")
            yield messages()

    return connect


async def run_collector(store: ParquetStore, interval: str, symbols: list[str] | None,
                        top: int | None, ws_base: str = WS_BASE, rest_base: str = REST_BASE,
                        flush_seconds: float = 5.0) -> None:
    import signal

    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        rest_get = make_aiohttp_rest(session, rest_base)
        if not symbols:
            symbols = (await top_by_volume(rest_get, top) if top
                       else await list_perpetuals(rest_get))
        log.info("collecting %d symbols at %s", len(symbols), interval)
        collector = KlineCollector(store, symbols, interval, rest_get,
                                   make_aiohttp_ws(session), ws_base=ws_base,
                                   flush_seconds=flush_seconds)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, collector.stop)
            except NotImplementedError:  # Windows: Ctrl+C raises KeyboardInterrupt instead
                pass
        try:
            await collector.run()
        finally:
            log.info("stopped; %d bars written", collector.bars_written)
