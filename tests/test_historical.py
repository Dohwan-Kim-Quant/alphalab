import asyncio

import pandas as pd

from alphalab.historical import (
    FILE_BASE, BinanceVisionClient, ChecksumError, DownloadConfig, NotFoundError,
    download_history, parse_funding_csv, parse_kline_csv, parse_listing, select_keys,
    verify_checksum,
)
from helpers import PickleStore, checksum_for, funding_csv, kline_csv, zip_bytes

NS = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'


def listing_xml(prefixes=(), keys=(), truncated=False, next_marker=None):
    cps = "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes)
    cs = "".join(f"<Contents><Key>{k}</Key><Size>1</Size></Contents>" for k in keys)
    nm = f"<NextMarker>{next_marker}</NextMarker>" if next_marker else ""
    return (f'<?xml version="1.0"?><ListBucketResult {NS}><Name>b</Name>'
            f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>{nm}{cps}{cs}"
            f"</ListBucketResult>")


def test_parse_listing_with_namespace_and_marker():
    page = parse_listing(listing_xml(["a/", "b/"], ["a/x.zip"], truncated=True, next_marker="b/"))
    assert page.prefixes == ["a/", "b/"]
    assert page.keys == ["a/x.zip"]
    assert page.next_marker == "b/"


def test_parse_listing_truncated_without_next_marker_uses_last_entry():
    page = parse_listing(listing_xml([], ["k1", "k2"], truncated=True))
    assert page.next_marker == "k2"
    assert parse_listing(listing_xml([], ["k1"])).next_marker is None


def test_select_keys_prefers_monthly_and_respects_start_and_manifest():
    p = "data/futures/um/{}/klines/BTCUSDT/1h/BTCUSDT-1h-{}.zip"
    monthly = [p.format("monthly", "2024-01"), p.format("monthly", "2024-02"),
               p.format("monthly", "2024-02") + ".CHECKSUM"]
    daily = [p.format("daily", "2024-02-10"), p.format("daily", "2024-03-01"),
             p.format("daily", "2024-03-02")]
    done = {p.format("daily", "2024-03-01")}
    got = select_keys(monthly, daily, pd.Timestamp("2024-02-15", tz="UTC"), done)
    assert got == sorted([p.format("monthly", "2024-02"), p.format("daily", "2024-03-02")])


def test_parse_kline_csv_header_and_microseconds_agree():
    a = parse_kline_csv(kline_csv("2024-01-01", 5, header=True, unit="ms"))
    b = parse_kline_csv(kline_csv("2024-01-01", 5, header=False, unit="us"))
    pd.testing.assert_frame_equal(a, b)
    assert a.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert a.index.freq is None and len(a) == 5
    assert list(a.columns)[:5] == ["open", "high", "low", "close", "volume"]
    assert a["trades"].dtype == "int64"


def test_parse_funding_csv():
    df = parse_funding_csv(funding_csv(["2024-01-01 00:00", "2024-01-01 08:00"], [0.0001, -0.0002]))
    assert df["funding_rate"].tolist() == [0.0001, -0.0002]
    assert df.index[1] == pd.Timestamp("2024-01-01 08:00", tz="UTC")


def test_verify_checksum():
    verify_checksum(b"abc", checksum_for(b"abc", "f.zip").decode())
    try:
        verify_checksum(b"abd", checksum_for(b"abc", "f.zip").decode())
    except ChecksumError:
        return
    raise AssertionError("expected ChecksumError")


class FakeArchive:
    """Serves a tiny fake data.binance.vision: listings + zips + checksums."""

    def __init__(self):
        um = "data/futures/um"
        self.files = {}
        jan = kline_csv("2024-01-01", 24 * 31)
        feb_day = kline_csv("2024-02-01", 24, seed=1)
        self._add(f"{um}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip", jan)
        self._add(f"{um}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-02-01.zip", feb_day)
        self._add(f"{um}/monthly/klines/LUNAUSDT/1h/LUNAUSDT-1h-2024-01.zip",
                  kline_csv("2024-01-01", 48))
        self._add(f"{um}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip",
                  funding_csv(["2024-01-01 08:00", "2024-01-01 16:00"], [0.0001, 0.0002]))
        self.dirs = {
            f"{um}/monthly/klines/": ["BTCUSDT", "LUNAUSDT", "BTCUSDT_250328", "ETHBTC"],
            f"{um}/daily/klines/": ["BTCUSDT"],
        }
        self.requests = 0

    def _add(self, key, csv):
        name = key.rsplit("/", 1)[-1]
        z = zip_bytes(name.replace(".zip", ".csv"), csv)
        self.files[key] = z
        self.files[key + ".CHECKSUM"] = checksum_for(z, name)

    async def fetch_text(self, url, params=None):
        self.requests += 1
        prefix = params["prefix"]
        if prefix in self.dirs:
            names = self.dirs[prefix]
            # Paginate: one prefix per page, to exercise the marker loop.
            marker = params.get("marker")
            start = 0 if marker is None else [prefix + n + "/" for n in names].index(marker) + 1
            chunk = names[start:start + 1]
            more = start + 1 < len(names)
            return listing_xml([prefix + n + "/" for n in chunk], [], truncated=more,
                               next_marker=(prefix + chunk[-1] + "/") if more else None)
        keys = [k for k in self.files if k.startswith(prefix) and "/" not in k[len(prefix):]]
        return listing_xml([], sorted(keys))

    async def fetch_bytes(self, url):
        key = url[len(FILE_BASE):]
        if key not in self.files:
            raise NotFoundError(url)
        return self.files[key]


def test_list_symbols_includes_delisted_and_skips_dated_and_other_quotes():
    arc = FakeArchive()
    client = BinanceVisionClient(arc.fetch_text, arc.fetch_bytes)
    assert asyncio.run(client.list_symbols("USDT")) == ["BTCUSDT", "LUNAUSDT"]


def test_download_history_end_to_end_and_incremental(tmp_path):
    arc = FakeArchive()
    store = PickleStore(tmp_path)
    client = BinanceVisionClient(arc.fetch_text, arc.fetch_bytes)
    res = asyncio.run(download_history(client, store, DownloadConfig(interval="1h")))
    by_sym = {r["symbol"]: r for r in res}
    assert by_sym["BTCUSDT"]["kline_files"] == 2
    assert by_sym["BTCUSDT"]["funding_files"] == 1
    assert by_sym["LUNAUSDT"]["kline_files"] == 1

    btc = store.read(store.kline_path("BTCUSDT", "1h"))
    assert len(btc) == 24 * 31 + 24
    assert btc.index.is_monotonic_increasing and not btc.index.duplicated().any()
    assert len(store.read(store.funding_path("BTCUSDT"))) == 2

    # Second run: the manifest means nothing new is fetched.
    res2 = asyncio.run(download_history(BinanceVisionClient(arc.fetch_text, arc.fetch_bytes),
                                        store, DownloadConfig(interval="1h")))
    assert sum(r["kline_files"] + r["funding_files"] for r in res2) == 0


def test_corrupt_file_is_not_ingested(tmp_path):
    arc = FakeArchive()
    key = "data/futures/um/monthly/klines/LUNAUSDT/1h/LUNAUSDT-1h-2024-01.zip"
    arc.files[key] = arc.files[key][:-5] + b"xxxxx"   # checksum no longer matches
    store = PickleStore(tmp_path)
    client = BinanceVisionClient(arc.fetch_text, arc.fetch_bytes)
    asyncio.run(download_history(client, store,
                                 DownloadConfig(interval="1h", symbols=["LUNAUSDT"])))
    assert store.read(store.kline_path("LUNAUSDT", "1h")).empty
