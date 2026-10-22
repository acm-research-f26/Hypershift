import numpy as np
import pytest

from hypershift.featbench.data import (FB, FEATURES, compute_features, load_fb, neutral_rankic, target,
                                       transform)
from hypershift.screening.data import split_signals


def synthetic(n=40, t=400, seed=0, zero_volume=False):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, .02, (n, t))
    close = 50 * np.exp(np.cumsum(r, 1))
    op = close * (1 + rng.normal(0, .003, (n, t)))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, .005, (n, t))))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, .005, (n, t))))
    vol = rng.lognormal(13, .5, (n, t))
    if zero_volume:
        vol[:, ::7] = 0
    ohlcv = np.stack([op, hi, lo, close, vol], -1).astype(np.float32)
    observed = np.ones((n, t), bool)
    beta = np.ones((n, t), np.float32)
    sectors = np.arange(n) % 4
    eligible = np.zeros((n, t), bool)
    eligible[:, 63:] = True
    return ohlcv, observed, beta, sectors, eligible


def synthetic_fb(n=40, t=400, seed=0):
    ohlcv, observed, beta, sectors, eligible = synthetic(n, t, seed)
    X = compute_features(ohlcv, observed, beta, sectors, eligible)
    rng = np.random.default_rng(seed + 1)
    y = rng.normal(0, .03, (n, t)).astype(np.float32)
    y_ok = np.ones((n, t), bool)
    beta = rng.uniform(.5, 1.5, (n, t)).astype(np.float32)
    dates = np.arange('2015-01-01', t, dtype='datetime64[D]')
    sig = np.arange(64, t - 5)
    return FB(X=X, y=y, y_ok=y_ok, eligible=eligible, beta=beta, sectors=sectors, dates=dates,
              price=ohlcv[..., 3], observed=observed, neutral=y.copy(), train=sig[:200], valid=sig[200:260],
              evaluate=sig[260:], horizon=5, year=2019, mu=np.zeros(len(FEATURES)), sd=np.ones(len(FEATURES)))


def test_feature_list_has_twenty_names():
    assert len(FEATURES) == 20 and len(set(FEATURES)) == 20


def test_features_causal():
    ohlcv, observed, beta, sectors, eligible = synthetic()
    a = compute_features(ohlcv, observed, beta, sectors, eligible)
    mutated = ohlcv.copy()
    mutated[:, 301:] *= np.random.default_rng(9).uniform(.5, 2, mutated[:, 301:].shape).astype(np.float32)
    b = compute_features(mutated, observed, beta, sectors, eligible)
    np.testing.assert_array_equal(a[:, :301], b[:, :301])
    assert not np.array_equal(a[:, 301:], b[:, 301:])


def test_features_finite_with_zero_volume():
    ohlcv, observed, beta, sectors, eligible = synthetic(zero_volume=True)
    x = compute_features(ohlcv, observed, beta, sectors, eligible)
    assert x.shape == (40, 400, 20)
    assert np.isfinite(x[eligible]).all()


def test_split_purged_for_horizon():
    dates = np.arange('2015-01-01', '2021-01-01', dtype='datetime64[D]')
    dates = dates[np.is_busday(dates)]
    train, valid, _ = split_signals(dates, 2020, horizon=20, train_stride=1)
    assert (train + 20).max() < valid.min()


def test_year_guard():
    with pytest.raises(ValueError):
        load_fb(2024)


def test_neutral_rankic_constant_prediction():
    rng = np.random.default_rng(0)
    pred = np.ones((3, 30))
    neutral = rng.normal(size=(3, 30))
    mask = np.ones((3, 30), bool)
    np.testing.assert_array_equal(neutral_rankic(pred, neutral, mask), np.zeros(3))


def test_residual_target_orthogonal():
    fb = synthetic_fb()
    y, mask = target(fb, fb.evaluate, 'residual')
    j = FEATURES.index
    for d, s in enumerate(fb.evaluate[:20]):
        m = mask[d]
        for exposure in (fb.beta[:, s], fb.X[:, s, j('ret_60')], fb.X[:, s, j('vol_60')]):
            assert abs(np.dot(y[d, m], exposure[m] - exposure[m].mean())) < 1e-6
        for g in np.unique(fb.sectors):
            assert abs(y[d, m & (fb.sectors == g)].sum()) < 1e-6


def test_sector_target_demeaned():
    fb = synthetic_fb()
    y, mask = target(fb, fb.evaluate, 'sector')
    for g in np.unique(fb.sectors):
        ix = mask & (fb.sectors == g)[None]
        assert np.abs((y * ix).sum(1)).max() < 1e-6


def test_transform_cs_z_moments():
    fb = synthetic_fb()
    s = fb.evaluate[:10]
    xs = fb.X[:, s].transpose(1, 0, 2)
    m = fb.eligible[:, s].T
    z = transform(xs, m, 'cs_z', fb)
    for d in range(10):
        assert np.allclose(z[d, m[d]].mean(0), 0, atol=1e-5)
        sd = z[d, m[d]].std(0)
        assert np.all((np.abs(sd - 1) < 1e-3) | (sd < 1e-6))


def test_transform_kinds_shapes():
    fb = synthetic_fb()
    s = fb.evaluate[:5]
    xs, m = fb.X[:, s].transpose(1, 0, 2), fb.eligible[:, s].T
    for kind in ('raw', 'cs_rank', 'cs_z', 'sector_z', 'volscaled_z'):
        z = transform(xs, m, kind, fb)
        assert z.shape == xs.shape and np.isfinite(z).all()
        assert (z[~m] == 0).all()
