"""Command line entry point: `alphalab <command>` or `python -m alphalab <command>`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

DEFAULT_DATA = "data"


def _cmd_download(args) -> int:
    from .historical import DownloadConfig, run_download
    from .store import ParquetStore

    cfg = DownloadConfig(
        interval=args.interval, start=args.start,
        symbols=[s.upper() for s in args.symbols] if args.symbols else None,
        quote=args.quote, include_funding=not args.no_funding,
    )
    results = asyncio.run(run_download(ParquetStore(args.data_dir), cfg,
                                       concurrency=args.concurrency, verify=not args.no_verify))
    errors = [r for r in results if "error" in r]
    files = sum(r.get("kline_files", 0) + r.get("funding_files", 0) for r in results)
    print(f"done: {len(results)} symbols, {files} new files, {len(errors)} errors")
    for r in errors:
        print(f"  {r['symbol']}: {r['error']}")
    return 1 if errors else 0


def _cmd_realtime(args) -> int:
    from .realtime import WS_BASE, run_collector
    from .store import ParquetStore

    symbols = [s.upper() for s in args.symbols] if args.symbols else None
    try:
        asyncio.run(run_collector(ParquetStore(args.data_dir), args.interval, symbols, args.top,
                                  ws_base=args.ws_base or WS_BASE,
                                  flush_seconds=args.flush_seconds))
    except KeyboardInterrupt:
        pass
    return 0


def _run_and_report(alpha_name, data, args) -> int:
    from .alphas import load_alpha
    from .backtest import BacktestConfig, run_backtest
    from .report import format_summary, save_equity_plot
    from .universe import top_liquidity

    cfg = BacktestConfig(delay=args.delay, neutralize=args.neutralize, fee_bps=args.fee_bps,
                         max_weight=args.max_weight or None,
                         include_funding=not args.no_funding)
    universe = top_liquidity(data, top_n=args.top_n)
    alpha = load_alpha(alpha_name)(data)
    result = run_backtest(alpha, data, universe, cfg)
    title = (f"{alpha_name}  |  {data.interval}, top {args.top_n}, delay {cfg.delay}, "
             f"{cfg.neutralize}-neutral, fee {cfg.fee_bps}bp")
    print(format_summary(result, title))
    out = result.save(Path(args.out) / Path(alpha_name.replace(":", "_")).name)
    save_equity_plot(result, out / "equity.png", title)
    print(f"\nsaved to {out}/ (pnl.csv, yearly.csv, summary.json, equity.png)")
    return 0


def _cmd_backtest(args) -> int:
    from .data import load_market_data
    from .store import ParquetStore

    data = load_market_data(ParquetStore(args.data_dir), args.interval,
                            [s.upper() for s in args.symbols] if args.symbols else None,
                            args.start, args.end)
    return _run_and_report(args.alpha, data, args)


def _cmd_demo(args) -> int:
    from .synthetic import make_synthetic_market

    print("Synthetic market with a planted 1-day reversal; reversal_1d should find it.\n")
    data = make_synthetic_market()
    return _run_and_report(args.alpha, data, args)


def _cmd_list(args) -> int:
    from .alphas import REGISTRY

    for name, fn in sorted(REGISTRY.items()):
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"{name:<22} {doc}")
    return 0


def _add_backtest_opts(p: argparse.ArgumentParser, default_alpha: str | None = None) -> None:
    p.add_argument("--alpha", default=default_alpha, required=default_alpha is None,
                   help="registered name (see list-alphas) or path/to/file.py:function")
    p.add_argument("--top-n", type=int, default=50, help="universe size by liquidity")
    p.add_argument("--delay", type=int, default=1, help="bars between signal and trade")
    p.add_argument("--neutralize", choices=["none", "market", "beta"], default="market")
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--max-weight", type=float, default=0.10, help="0 disables the cap")
    p.add_argument("--no-funding", action="store_true", help="ignore funding payments")
    p.add_argument("--out", default="results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphalab", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch history from data.binance.vision")
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--interval", default="1h")
    p.add_argument("--start", default=None, help="e.g. 2021-01-01 (default: everything)")
    p.add_argument("--symbols", nargs="*", help="default: every USDT perpetual, incl. delisted")
    p.add_argument("--quote", default="USDT")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--no-funding", action="store_true")
    p.add_argument("--no-verify", action="store_true", help="skip SHA-256 checks")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("realtime", help="stream closed bars over websocket")
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--interval", default="1h")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--symbols", nargs="*")
    g.add_argument("--top", type=int, default=None, help="top N by 24h quote volume")
    p.add_argument("--flush-seconds", type=float, default=5.0)
    p.add_argument("--ws-base", default=None, help="override websocket base URL")
    p.set_defaults(func=_cmd_realtime)

    p = sub.add_parser("backtest", help="run an alpha on stored data")
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--interval", default="1h")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    _add_backtest_opts(p)
    p.set_defaults(func=_cmd_backtest)

    p = sub.add_parser("demo", help="run the stack on synthetic data (no network)")
    _add_backtest_opts(p, default_alpha="reversal_1d")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("list-alphas", help="show registered alphas")
    p.set_defaults(func=_cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
