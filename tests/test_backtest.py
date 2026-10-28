import numpy as np
import pandas as pd

from alphalab.backtest import (
    BacktestConfig, alpha_to_weights, beta_neutralize, run_backtest,
)
from alphalab.data import MarketData, bar_of_event, build_market_data
from alphalab.synthetic import make_synthetic_market
from alphalab.universe import top_liquidity


def market_from_close(close: pd.DataFrame, funding: pd.DataFrame | None = None) -> MarketData:
    ones = close * 0 + 1.0
    f = funding if funding is not None else close * 0.0
    return MarketData(interval="1h", open=close, high=close, low=close, close=close,
                      volume=ones, quote_volume=ones, trades=ones, taker_buy_volume=ones * 0.5,
                      taker_buy_quote_volume=ones * 0.5, funding=f, funding_rate=f)


def two_asset_close():
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    ra = [0.0, 0.01, -0.02, 0.03, 0.01, -0.01]
    rb = [0.0, -0.01, 0.01, 0.00, 0.02, 0.01]
    close = pd.DataFrame({"A": 100 * np.cumprod(1 + np.array(ra)),
                          "B": 50 * np.cumprod(1 + np.array(rb))}, index=idx)
    return close, ra, rb


def test_pnl_timing_is_exact():
    close, ra, rb = two_asset_close()
    data = market_from_close(close)
    alpha = pd.DataFrame({"A": 1.0, "B": -1.0}, index=close.index)
    uni = alpha.notna()
    cfg = BacktestConfig(delay=0, fee_bps=0, max_weight=None, include_funding=False)
    res = run_backtest(alpha, data, uni, cfg)
    # Position opened at bar 0's close (pnl 0 that bar) earns from bar 1 on: +0.5 A, -0.5 B.
    expected = [0.0] + [0.5 * a - 0.5 * b for a, b in zip(ra[1:], rb[1:])]
    np.testing.assert_allclose(res.pnl["gross"].to_numpy(), expected)
    assert res.pnl.index[0] == close.index[0]

    # delay=1 shifts everything by exactly one bar.
    res1 = run_backtest(alpha, data, uni, BacktestConfig(delay=1, fee_bps=0, max_weight=None,
                                                         include_funding=False))
    np.testing.assert_allclose(res1.pnl["gross"].to_numpy(), [0.0] + expected[2:])
    assert res1.pnl.index[0] == close.index[1]


def test_fees_charged_on_turnover():
    close, _, _ = two_asset_close()
    data = market_from_close(close)
    alpha = pd.DataFrame({"A": 1.0, "B": -1.0}, index=close.index)
    res = run_backtest(alpha, data, alpha.notna(),
                       BacktestConfig(delay=0, fee_bps=10, max_weight=None, include_funding=False))
    # Opening the book trades 1.0 of gross once, then the position never changes.
    assert np.isclose(-res.pnl["fees"].sum(), 1.0 * 10 / 1e4)


def test_future_peeking_alpha_only_works_without_delay():
    """Sanity check of alignment: an alpha that knows the next bar's return should
    print money at delay=0 and be useless at delay=1. If the timing were off by one
    bar in either direction this test would fail."""
    d = make_synthetic_market(n_symbols=20, n_bars=2000, strength=0.0)
    cheat = d.returns.shift(-1)
    uni = d.close.notna()
    cfg0 = BacktestConfig(delay=0, fee_bps=0, max_weight=None)
    cfg1 = BacktestConfig(delay=1, fee_bps=0, max_weight=None)
    assert run_backtest(cheat, d, uni, cfg0).summary["sharpe_gross"] > 20
    assert abs(run_backtest(cheat, d, uni, cfg1).summary["sharpe_gross"]) < 3


def test_funding_sign_long_pays_positive_rate():
    close, _, _ = two_asset_close()
    funding = close * 0.0
    funding.iloc[3, 0] = 0.001  # A's rate settles at the close of bar 3
    data = market_from_close(close, funding)
    alpha = pd.DataFrame({"A": 1.0, "B": -1.0}, index=close.index)
    res = run_backtest(alpha, data, alpha.notna(),
                       BacktestConfig(delay=0, fee_bps=0, max_weight=None))
    assert np.isclose(res.pnl["funding"].sum(), -0.5 * 0.001)


def test_funding_event_maps_to_bar_that_closes_on_it():
    t = pd.DatetimeIndex(["2024-01-01 08:00", "2024-01-01 08:00:00.005"], tz="UTC")
    bars = bar_of_event(t, "1h")
    assert bars[0] == pd.Timestamp("2024-01-01 07:00", tz="UTC")
    assert bars[1] == pd.Timestamp("2024-01-01 08:00", tz="UTC")


def test_build_market_data_regular_grid_and_funding():
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    k = pd.DataFrame({c: 1.0 for c in ["open", "high", "low", "close", "volume", "quote_volume",
                                       "trades", "taker_buy_volume", "taker_buy_quote_volume"]},
                     index=idx)
    gap = k.drop(idx[4])                               # a missing bar
    f = pd.Series([0.0001], index=pd.DatetimeIndex(["2024-01-01 08:00"], tz="UTC"))
    md = build_market_data({"A": k, "B": gap}, {"A": f}, "1h")
    assert len(md.index) == 10 and np.isnan(md.close.loc[idx[4], "B"])
    assert md.funding.loc[idx[7], "A"] == 0.0001 and md.funding["A"].sum() == 0.0001
    assert np.isnan(md.funding_rate.loc[idx[6], "A"]) and md.funding_rate.loc[idx[9], "A"] == 0.0001
    assert (md.funding["B"] == 0).all()


def test_beta_neutralize_constraints():
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.normal(size=(50, 12)))
    beta = pd.DataFrame(rng.uniform(0.5, 2, size=(50, 12)))
    w = beta_neutralize(a, beta)
    assert np.allclose(w.sum(axis=1), 0, atol=1e-10)
    assert np.allclose((w * beta).sum(axis=1), 0, atol=1e-10)


def test_weights_respect_cap_and_neutrality():
    rng = np.random.default_rng(1)
    alpha = pd.DataFrame(rng.lognormal(size=(30, 25)) ** 3)  # very skewed
    uni = alpha.notna()
    w = alpha_to_weights(alpha, uni, "market", max_weight=0.06)
    assert w.abs().max().max() <= 0.06 + 1e-12
    assert np.allclose(w.sum(axis=1), 0, atol=1e-12)          # neutral even when squeezed
    assert (w.abs().sum(axis=1) <= 1 + 1e-12).all()

    normal = pd.DataFrame(rng.normal(size=(30, 25)))
    w2 = alpha_to_weights(normal, normal.notna(), "market", max_weight=0.1)
    assert w2.abs().max().max() <= 0.1 + 1e-12
    assert np.allclose(w2.sum(axis=1), 0, atol=1e-12)
    assert np.allclose(w2.abs().sum(axis=1), 1)                # feasible: full book kept


def test_cap_with_too_few_names_shrinks_that_side():
    w = pd.DataFrame([[0.5, -0.25, -0.25]])
    capped = alpha_to_weights(w, w.notna(), "none", max_weight=0.3)
    assert np.allclose(capped.to_numpy(), [[0.3, -0.25, -0.25]])


def test_universe_has_no_lookahead():
    d = make_synthetic_market(n_symbols=30, n_bars=24 * 60)
    u1 = top_liquidity(d, top_n=10)
    d.quote_volume.iloc[24 * 45:] *= np.linspace(0.01, 100, 30)  # reshuffle the future
    u2 = top_liquidity(d, top_n=10)
    pd.testing.assert_frame_equal(u1.iloc[:24 * 45], u2.iloc[:24 * 45])
    assert (u1.sum(axis=1).iloc[-1] == 10)


def test_planted_signal_is_found_and_costs_matter():
    d = make_synthetic_market()
    uni = top_liquidity(d, top_n=40)
    from alphalab.alphas import reversal_1d
    res = run_backtest(reversal_1d(d), d, uni)
    assert res.summary["ic_mean"] > 0.01
    assert res.summary["sharpe_gross"] > res.summary["sharpe"] > 1
