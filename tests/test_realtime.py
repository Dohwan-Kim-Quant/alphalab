import asyncio
import contextlib
import time

import pandas as pd

from alphalab.realtime import (
    KlineCollector, chunked, closed_only, fetch_klines_since, parse_rest_klines, parse_ws_kline,
    stream_url,
)
from helpers import PickleStore

H = 3_600_000


def ws_msg(open_ms, closed=True, sym="BTCUSDT", close=100.0):
    return {"stream": f"{sym.lower()}@kline_1h", "data": {
        "e": "kline", "s": sym, "k": {
            "t": open_ms, "T": open_ms + H - 1, "s": sym, "i": "1h", "o": "99", "c": str(close),
            "h": "101", "l": "98", "v": "10", "n": 7, "x": closed, "q": "1000",
            "V": "4", "Q": "400"}}}


def rest_row(open_ms, close=100.0):
    return [open_ms, "99", "101", "98", str(close), "10", open_ms + H - 1, "1000", 7, "4", "400", "0"]


def test_parse_ws_kline_only_closed():
    sym, row = parse_ws_kline(ws_msg(0))
    assert sym == "BTCUSDT" and row["close"] == 100.0 and row["trades"] == 7
    assert parse_ws_kline(ws_msg(0, closed=False)) is None
    assert parse_ws_kline({"data": {"e": "markPriceUpdate"}}) is None


def test_ws_and_rest_rows_have_identical_schema():
    from alphalab.realtime import rows_to_frame
    a = rows_to_frame([parse_ws_kline(ws_msg(H * 10))[1]])
    b = parse_rest_klines([rest_row(H * 10)])
    pd.testing.assert_frame_equal(a, b)


def test_closed_only_drops_forming_bar():
    df = parse_rest_klines([rest_row(0), rest_row(H), rest_row(2 * H)])
    assert len(closed_only(df, "1h", now_ms=2 * H + 5)) == 2


def test_stream_url_and_chunking():
    assert stream_url("wss://x/stream", ["BTCUSDT", "ETHUSDT"], "1h") == \
        "wss://x/stream?streams=btcusdt@kline_1h/ethusdt@kline_1h"
    assert [len(c) for c in chunked(list(range(450)), 200)] == [200, 200, 50]


def test_fetch_klines_since_paginates():
    calls = []

    async def rest_get(path, params):
        calls.append(params["startTime"])
        start = params["startTime"]
        n = min(params["limit"], max(0, (10_000 * H - start) // H))
        return [rest_row(start + i * H) for i in range(n)]

    df = asyncio.run(fetch_klines_since(rest_get, "BTCUSDT", "1h", 0, now_ms=10_000 * H))
    assert len(df) == 10_000 and df.index.is_unique
    assert calls[:2] == [0, 1500 * H]


def test_collector_merges_backfill_and_websocket(tmp_path):
    """Backfill fills the gap up to now, the websocket adds new closed bars, and
    overlap between the two is deduplicated on disk."""
    store = PickleStore(tmp_path)
    now = int(time.time() * 1000) // H * H              # current (still forming) bar open
    # Already stored: bars up to 5 hours ago.
    store.upsert(store.kline_path("BTCUSDT", "1h"),
                 parse_rest_klines([rest_row(now - H * k) for k in range(10, 4, -1)]))

    async def rest_get(path, params):
        if path == "/fapi/v1/fundingRate":
            return []
        start = params["startTime"]
        return [rest_row(t) for t in range(start, now + H, H)]  # includes the forming bar

    @contextlib.asynccontextmanager
    async def ws_connect(url, timeout):
        async def gen():
            yield ws_msg(now - H)                 # duplicate of a backfilled bar
            yield ws_msg(now, closed=False)       # still forming: ignored
            await asyncio.sleep(3600)
            yield None
        yield gen()

    async def main():
        c = KlineCollector(store, ["BTCUSDT"], "1h", rest_get, ws_connect, flush_seconds=0.05)
        task = asyncio.create_task(c.run())
        await asyncio.sleep(0.5)
        c.stop()
        await asyncio.wait_for(task, 5)

    asyncio.run(main())
    df = store.read(store.kline_path("BTCUSDT", "1h"))
    expected = pd.DatetimeIndex([pd.Timestamp(now - H * k, unit="ms", tz="UTC")
                                 for k in range(10, 0, -1)]).as_unit("ms")
    assert list(df.index) == list(expected)          # contiguous, no forming bar, no dupes
