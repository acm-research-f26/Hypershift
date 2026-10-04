"""Phase 1.5b data gate (DATA_COMPATIBILITY.md): synthetic TDD for the post-2017 panel builder."""
import numpy as np
import pandas as pd
import pytest

from hypershift.data import post2017 as p
from hypershift.data.rsr import FILL, parse_eod


def _prices(n, seed=0, start=50.0):
    rng = np.random.default_rng(seed)
    return start * np.cumprod(1 + rng.normal(0, 0.01, n))


def _returns(c):
    r = np.full(len(c), np.nan)
    r[1:] = c[1:] / c[:-1] - 1
    return r


def test_identity_accepts_same_security():
    c = _prices(800)
    ok, share, n = p.identity_test(_returns(c), _returns(c))
    assert ok and share == 1.0 and n == 799


def test_ticker_reuse_rejected_by_overlap_failure():
    a, b = _prices(800, seed=1), _prices(800, seed=2)
    ok, share, _ = p.identity_test(_returns(b), _returns(a))          # same symbol, different company
    assert not ok and share < 0.05


def test_splice_rejected():
    a, b = _prices(800, seed=1), _prices(800, seed=2)
    spliced = np.concatenate([a[:400], b[400:]])
    ok, share, _ = p.identity_test(_returns(spliced), _returns(a))
    assert not ok and 0.4 < share < 0.6


def test_identity_needs_min_matched_days():
    c = _prices(200)
    ok, share, n = p.identity_test(_returns(c), _returns(c))
    assert not ok and share == 1.0 and n < 250


def test_undo_splits_recovers_raw():
    raw = np.array([100.0, 102.0, 51.0, 52.0, 26.5, 27.0])             # 2:1 effective at idx 2 and at idx 4
    ratios = np.array([0, 0, 2.0, 0, 2.0, 0])
    split_adj = np.array([25.0, 25.5, 25.5, 26.0, 26.5, 27.0])         # Yahoo-style: earlier prices divided by later ratios
    np.testing.assert_allclose(p.undo_splits(split_adj, ratios), raw)


def test_returns_on_calendar_masks_missing_days():
    cal = pd.bdate_range("2020-01-01", periods=6)
    s = pd.Series([10, 11, 13, 14, 15], index=cal[[0, 1, 3, 4, 5]], dtype=float)   # cal[2] missing
    r = p.returns_on_calendar(s, cal)
    assert np.isnan(r[0]) and np.isnan(r[2]) and np.isnan(r[3])        # no return across the gap, none invented
    assert r[1] == pytest.approx(0.1)
    assert r[4] == pytest.approx(14 / 13 - 1) and r[5] == pytest.approx(15 / 14 - 1)


def test_returns_on_calendar_values():
    cal = pd.bdate_range("2020-01-01", periods=5)
    s = pd.Series([10.0, 11.0, 12.1, 12.1, 13.31], index=cal)
    np.testing.assert_allclose(p.returns_on_calendar(s, cal)[1:], [0.1, 0.1, 0.0, 0.1], atol=1e-12)


def test_match_share():
    a = np.array([0.01, 0.02, np.nan, 0.03])
    b = np.array([0.01, 0.02015, 0.5, 0.0300])
    assert p.match_share(a, b, tol=1e-4) == (2, 3)                     # 1.5e-4 apart: not a match at 1e-4
    assert p.match_share(a, b, tol=2e-4) == (3, 3)


def test_ma_matches_rsr_on_synthetic():
    """build_features on a gap-free series reproduces the RSR file layout (MA5/10/20/30/close) after parse_eod."""
    c = _prices(120, seed=3)
    scale = c.max()
    raw = np.zeros((len(c), 6))
    raw[:, 0] = np.arange(len(c))
    for k, w in enumerate((5, 10, 20, 30)):
        raw[:, 1 + k] = pd.Series(c).rolling(w, min_periods=w).mean().to_numpy() / scale
    raw[:, 5] = c / scale
    feats, mask, gt, base = parse_eod(raw[29:], drop_last=False)       # rows with a full MA30 window
    out = p.build_features(c, train_end=len(c) - 1)
    np.testing.assert_allclose(out.features[29:], feats, atol=1e-6)
    np.testing.assert_allclose(out.base_price[29:], base, atol=1e-6)
    np.testing.assert_allclose(out.gt[30:], gt[1:], atol=1e-6)         # gt[29] has no previous RSR row in the cut file
    assert out.mask[29:].all() and not out.mask[:29].any()             # MA30 not available earlier -> masked


def test_missing_day_masked_and_filled():
    c = _prices(80, seed=4)
    c[50] = np.nan
    out = p.build_features(c, train_end=60)
    assert out.mask[50] == 0 and (out.features[50] == FILL).all() and out.base_price[50] == FILL
    assert out.mask[49] == 1 and out.mask[51] == 1
    assert out.gt[50] == 0 and out.gt[51] == 0                         # no return across a missing day
    ff = pd.Series(c).ffill().to_numpy()                               # documented INFERRED rule: forward-filled window
    assert out.features[52, 0] == pytest.approx(ff[48:53].mean() / np.nanmax(c[:61]), rel=1e-5)


def test_delisting_masks_never_zero_fills():
    c = _prices(100, seed=5)
    c[70:] = np.nan                                                    # delisted after index 69
    out = p.build_features(c, train_end=60)
    assert (out.mask[70:] == 0).all() and (out.features[70:] == FILL).all()
    assert (out.gt[70:] == 0).all()                                    # placeholder only; mask is 0
    assert out.mask[69] == 1


def test_future_price_cannot_change_earlier_feature():
    c = _prices(150, seed=6)
    base = p.build_features(c, train_end=60)
    c2 = c.copy()
    c2[120:] *= 3.0
    alt = p.build_features(c2, train_end=60)
    for name in ("features", "mask", "gt", "base_price"):
        np.testing.assert_array_equal(getattr(base, name)[:120], getattr(alt, name)[:120])
    c3 = c.copy()
    c3[61:] *= 100.0                                                   # post-train change never moves the scale
    np.testing.assert_allclose(p.build_features(c3, train_end=60).features[:61], base.features[:61])


def test_truncate_gaps_ticker_reuse_signature():
    obs = np.ones(300, bool)
    obs[100:170] = False                                               # 70-day gap, then it reappears
    out = p.truncate_gaps(obs, gap=60)
    assert out[:100].all() and not out[100:].any()
    short = np.ones(300, bool)
    short[100:130] = False                                             # 30-day gap is kept
    assert (p.truncate_gaps(short, gap=60) == short).all()
    never = np.ones(300, bool)
    never[200:] = False                                                # plain delisting: unchanged
    assert (p.truncate_gaps(never, gap=60) == never).all()


def test_panel_exact_ticker_order_and_missing_rows():
    cal = pd.bdate_range("2019-01-01", periods=80)
    order = ["ZZZ", "AAA", "MMM", "BRK.B"]
    series = {"MMM": pd.Series(_prices(80, 1), index=cal), "AAA": pd.Series(_prices(80, 2), index=cal),
              "BRK.B": pd.Series(_prices(80, 3), index=cal)}           # ZZZ has no data at all
    panel = p.build_panel(series, order, cal, train_end=40)
    assert panel.tickers == order
    assert panel.features.shape == (4, 80, 5)
    assert panel.mask[0].sum() == 0 and (panel.features[0] == FILL).all()   # unavailable ticker keeps its slot
    np.testing.assert_allclose(panel.base_price[1, 79], series["AAA"].iloc[79] / series["AAA"].iloc[:41].max(), rtol=1e-5)
    order2 = list(reversed(order))
    assert p.build_panel(dict(reversed(list(series.items()))), order2, cal, train_end=40).tickers == order2


def test_availability_mask_matches_build_features_mask():
    c = _prices(120, seed=7)
    c[:10] = np.nan
    c[60] = np.nan
    c[100:] = np.nan
    obs = np.isfinite(c)
    np.testing.assert_array_equal(p.availability_mask(obs), p.build_features(c, train_end=50).mask > 0.5)
    assert not p.availability_mask(np.zeros(5, bool)).any()
