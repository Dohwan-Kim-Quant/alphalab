import numpy as np
import pandas as pd

from alphalab import operators as op


def panel(values, cols=("A", "B", "C")):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame(values, index=idx, columns=list(cols), dtype="float64")


def random_panel(rows=300, cols=6, seed=0, nan_frac=0.05):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(rows, cols))
    a[rng.random(a.shape) < nan_frac] = np.nan
    return panel(a, cols=[f"S{i}" for i in range(cols)])


def test_rank_is_zero_to_one_and_keeps_nan():
    x = panel([[1, 2, 3], [5, np.nan, 1], [np.nan, 7, np.nan]])
    r = op.rank(x)
    assert r.iloc[0].tolist() == [0.0, 0.5, 1.0]
    assert r.iloc[1, 0] == 1.0 and np.isnan(r.iloc[1, 1]) and r.iloc[1, 2] == 0.0
    assert r.iloc[2, 1] == 0.5  # a lone value sits in the middle


def test_scale_and_demean():
    x = panel([[1, 2, 3], [-1, 0, 4]])
    s = op.scale(op.demean(x))
    assert np.allclose(s.sum(axis=1), 0)
    assert np.allclose(s.abs().sum(axis=1), 1)


def test_ts_rank_matches_brute_force_across_chunk_boundaries():
    x = random_panel(rows=257)
    d = 10
    old = op._CHUNK_ROWS
    op._CHUNK_ROWS = 37  # force many chunk boundaries
    try:
        got = op.ts_rank(x, d)
    finally:
        op._CHUNK_ROWS = old
    exp = np.full(x.shape, np.nan)
    a = x.to_numpy()
    for t in range(d - 1, len(a)):
        for j in range(a.shape[1]):
            w = a[t - d + 1:t + 1, j]
            if np.isnan(w).any():
                continue
            exp[t, j] = ((w < w[-1]).sum() + 0.5 * ((w == w[-1]).sum() - 1)) / (d - 1)
    np.testing.assert_allclose(got.to_numpy(), exp, equal_nan=True)


def test_ts_decay_linear():
    x = panel([[1], [2], [3], [4]], cols=["A"])
    got = op.ts_decay_linear(x, 3).iloc[:, 0].tolist()
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert np.isclose(got[2], (3 * 3 + 2 * 2 + 1 * 1) / 6)
    assert np.isclose(got[3], (3 * 4 + 2 * 3 + 1 * 2) / 6)


def test_ts_arg_max_counts_bars_since_max():
    x = panel([[1], [3], [2], [2]], cols=["A"])
    got = op.ts_arg_max(x, 3).iloc[:, 0].tolist()
    assert got[2] == 1 and got[3] == 2


def test_trade_when_holds_until_exit():
    alpha = panel([[1], [2], [3], [4], [5]], cols=["A"])
    trig = panel([[1], [0], [0], [1], [0]], cols=["A"]).astype(bool)
    exit_ = panel([[0], [0], [1], [0], [0]], cols=["A"]).astype(bool)
    got = op.trade_when(trig, alpha, exit_).iloc[:, 0].tolist()
    assert got[0] == 1 and got[1] == 1 and np.isnan(got[2]) and got[3] == 4 and got[4] == 4


def test_no_operator_looks_ahead():
    """Changing the future must not change any past output."""
    x = random_panel(rows=200, seed=1)
    y = random_panel(rows=200, seed=2)
    cut = 120
    x2, y2 = x.copy(), y.copy()
    x2.iloc[cut:] = x2.iloc[cut:] * 7 + 3
    y2.iloc[cut:] = -y2.iloc[cut:]
    ops = {
        "ts_mean": lambda a, b: op.ts_mean(a, 10),
        "ts_std_dev": lambda a, b: op.ts_std_dev(a, 10),
        "ts_zscore": lambda a, b: op.ts_zscore(a, 10),
        "ts_rank": lambda a, b: op.ts_rank(a, 10),
        "ts_arg_max": lambda a, b: op.ts_arg_max(a, 10),
        "ts_decay_linear": lambda a, b: op.ts_decay_linear(a, 10),
        "ts_corr": lambda a, b: op.ts_corr(a, b, 10),
        "ts_regression_residual": lambda a, b: op.ts_regression_residual(a, b, 10),
        "ts_delta": lambda a, b: op.ts_delta(a, 3),
        "ts_backfill": lambda a, b: op.ts_backfill(a, 3),
        "rank": lambda a, b: op.rank(a),
        "zscore": lambda a, b: op.zscore(a),
    }
    for name, f in ops.items():
        before = f(x, y).iloc[:cut]
        after = f(x2, y2).iloc[:cut]
        pd.testing.assert_frame_equal(before, after, obj=name)


def test_group_neutralize():
    x = panel([[1, 3, 10]])
    g = {"A": "l1", "B": "l1", "C": "meme"}
    got = op.group_neutralize(x, g).iloc[0].tolist()
    assert got == [-1.0, 1.0, 0.0]
