import numpy as np
import pytest
from pathlib import Path
from scipy.stats import spearmanr

from hypershift.fbaudit import stats, diag

FB_ROOT = Path('results/FEATBENCH_SP500_20261022')


def test_spearman_daily_matches_scipy():
    rng = np.random.default_rng(0)
    p, y = rng.normal(size=(6, 50)), rng.normal(size=(6, 50))
    m = rng.random((6, 50)) > .2
    out = stats.spearman_daily(p, y, m)
    for d in range(6):
        assert abs(out[d] - spearmanr(p[d, m[d]], y[d, m[d]])[0]) < 1e-12


def test_bootstraps_long_block():
    x = np.random.default_rng(1).normal(size=30)
    for f in (stats.stationary_means, stats.mbb_means):
        b = f(x, len(x) + 5, 200, 0)
        assert b.shape == (200,) and np.isfinite(b).all()


def test_nw_se_iid_close_to_naive():
    x = np.random.default_rng(2).normal(size=5000)
    r = stats.nw_se(x, 5) / (x.std(ddof=1) / np.sqrt(len(x)))
    assert .9 <= r <= 1.1


def test_neutralise_pred_removes_exposure():
    rng = np.random.default_rng(3)
    S, N, K = 4, 60, 5
    expo = rng.normal(size=(S, N, K))
    pred = rng.normal(size=(S, N)) + expo[..., 0]
    m = np.ones((S, N), bool); m[:, :5] = False
    r = diag.neutralise_pred(pred, expo, m)
    for d in range(S):
        A = np.column_stack([np.ones(m[d].sum()), expo[d, m[d]]])
        assert np.abs(A.T @ r[d, m[d]]).max() < 1e-8


def test_neutralise_pred_thin_date():
    rng = np.random.default_rng(4)
    expo = rng.normal(size=(1, 10, 5))
    pred = rng.normal(size=(1, 10))
    m = np.zeros((1, 10), bool); m[0, :3] = True
    assert np.array_equal(diag.neutralise_pred(pred, expo, m), pred)


@pytest.mark.data
def test_recompute_matches_saved():
    p = FB_ROOT / '2021' / 'lgbm__rich__residual' / 'seed_7'
    if not p.exists():
        pytest.skip('FB results missing')
    z = stats.load_pred(FB_ROOT, 2021, 'lgbm:rich:residual')
    mine = stats.spearman_daily(z['prediction'], z['neutral'], z['mask'])
    saved, _ = stats.load_daily(FB_ROOT, 2021, 'lgbm:rich:residual')
    assert np.abs(mine - saved).max() < 1e-6


def _toy_fb(T=20, N=6, seed=5):
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    price = np.cumprod(1 + .01 * rng.normal(size=(N, T)), 1)
    return SimpleNamespace(price=price, observed=np.ones((N, T), bool), beta=np.ones((N, T)),
                           sectors=np.array([0, 0, 0, 1, 1, 1]), eligible=np.ones((N, T), bool),
                           evaluate=np.arange(10, 15), horizon=5)


def test_shifted_labels_drop_tail():
    fb = _toy_fb()
    neu, raw, ok, keep = diag.shifted_neutral(fb, fb.evaluate, 'close1')
    assert keep.tolist() == [True, True, True, True, False]
    assert neu.shape == (5, 6) and np.isfinite(neu).all()


def test_shifted_close1_equals_manual():
    from hypershift.featbench.data import _neutralise
    fb = _toy_fb()
    neu, raw, ok, keep = diag.shifted_neutral(fb, fb.evaluate, 'close1')
    t = 11
    assert abs(raw[1, 2] - (fb.price[2, t + 6] / fb.price[2, t + 1] - 1)) < 1e-12
    assert np.allclose(neu[keep], _neutralise(raw[keep], ok[keep], fb.beta[:, fb.evaluate[keep]].T, fb.sectors))
