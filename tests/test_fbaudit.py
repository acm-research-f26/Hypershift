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


def _toy_target_fb(S_=8, N=40, seed=6):
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    T = S_ + 5
    return SimpleNamespace(y=rng.normal(size=(N, T)), y_ok=np.ones((N, T), bool), eligible=np.ones((N, T), bool),
                           sectors=np.repeat(np.arange(4), N // 4), beta=rng.normal(1, .3, size=(N, T)),
                           neutral=rng.normal(size=(N, T)).astype(np.float32),
                           X=rng.normal(size=(N, T, 20)).astype(np.float32))


def test_refit_target_kinds_shapes():
    from hypershift.fbaudit import refit
    fb = _toy_target_fb()
    sig = np.arange(8)
    for kind in ('raw', 'sector', 'residual', 'demean', 'sector_beta', 'neutral'):
        y, m = refit.target_values(fb, sig, kind)
        assert y.shape == (8, 40) and m.shape == (8, 40) and np.isfinite(y).all()
    y, m = refit.target_values(fb, sig, 'demean')
    assert np.abs(y.mean(1)).max() < 1e-12
    y, m = refit.target_values(fb, sig, 'sector_beta')
    for d in range(8):
        A = np.column_stack([(fb.sectors[:, None] == np.arange(4)).astype(float), fb.beta[:, d]])
        assert np.abs(A.T @ y[d]).max() < 1e-8


@pytest.mark.data
def test_run_arm_matches_saved():
    if not (FB_ROOT / '2021' / 'lgbm__rich__residual' / 'seed_7').exists():
        pytest.skip('FB results missing')
    from hypershift.fbaudit import refit
    from hypershift.featbench.data import load_fb
    out = refit.run_arm(load_fb(2021, 5), 'lgbm', 'rich', 'residual', seed=7)
    saved, _ = stats.load_daily(FB_ROOT, 2021, 'lgbm:rich:residual')
    assert np.abs(out['daily'] - saved).max() < 1e-4


def test_freeze_hashes_unchanged():
    import hashlib
    import json
    man = json.load(open('docs/featbench_20261022/FREEZE_MANIFEST.json'))
    for path, h in man['source_sha256'].items():
        got = hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        assert got == h, path


def _toy_style_fb(T=90, N=40, seed=8):
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    price = np.cumprod(1 + .01 * rng.normal(size=(N, T)), 1)
    X = rng.normal(size=(N, T, 20)).astype(np.float32)
    X[..., 9] = np.abs(X[..., 9]) * .02 + .005          # vol_60 positive
    return SimpleNamespace(price=price, observed=np.ones((N, T), bool), eligible=np.ones((N, T), bool),
                           beta=rng.normal(1, .3, size=(N, T)), sectors=np.repeat(np.arange(4), N // 4), X=X)


def test_rr_sign_and_neutral():
    from hypershift.fbaudit import diag2
    from hypershift.featbench.data import FEATURES
    fb = _toy_style_fb()
    sig = np.arange(30, 60)
    rr = diag2.residual_reversal(fb, sig, k=21)
    n21 = -rr * np.maximum(fb.X[:, sig, FEATURES.index('vol_60')].T, 1e-4)
    for g in range(4):
        assert np.abs(n21[:, fb.sectors == g].mean(1)).max() < 1e-10
    r = (fb.price[:, sig] / fb.price[:, sig - 21] - 1).T
    assert np.corrcoef(r[0], n21[0])[0, 1] > .5


def test_quintile_dummies_ties():
    from hypershift.fbaudit import diag2
    fb = _toy_style_fb()
    fb.beta[:] = 1.0
    Q = diag2.quintile_dummies(fb, np.arange(30, 35))
    ns = len(np.unique(fb.sectors))
    assert Q.shape == (5, 40, ns + 16)
    assert np.all(Q[..., ns:ns + 4] == 0)          # constant beta -> no beta dummies
    assert np.isfinite(Q).all()


def test_nl_resid_orthogonal():
    from hypershift.fbaudit import diag2
    fb = _toy_style_fb()
    sig = np.arange(30, 35)
    m = np.ones((5, 40), bool)
    y = np.random.default_rng(9).normal(size=(5, 40))
    r = diag2.nl_resid(y, fb, sig, m)
    Q = diag2.quintile_dummies(fb, sig)
    for d in range(5):
        assert np.abs(np.column_stack([np.ones(40), Q[d]]).T @ r[d]).max() < 1e-8


def test_book_gross_and_beta():
    from hypershift.fbaudit import book
    rng = np.random.default_rng(10)
    N = 45
    sectors = np.r_[np.repeat(np.arange(4), 10), np.full(5, 4)]      # sector 4 has 5 stocks -> skipped
    pre, hedged = book.book_targets(rng.normal(size=N), np.ones(N, bool), sectors, rng.normal(1, .3, N))
    assert abs(np.abs(pre).sum() - 1) < 1e-9 and abs(pre.sum()) < 1e-9
    assert np.all(pre[sectors == 4] == 0)
    beta = rng.normal(1, .3, N)
    pre, hedged = book.book_targets(rng.normal(size=N), np.ones(N, bool), sectors, beta)
    assert abs(hedged @ beta) < 1e-9


def _toy_book_fb(T=40, N=40, seed=11, flat=False):
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    price = np.ones((N, T)) if flat else np.cumprod(1 + .01 * rng.normal(size=(N, T)), 1)
    return SimpleNamespace(price=price, observed=np.ones((N, T), bool), eligible=np.ones((N, T), bool),
                           beta=np.ones((N, T)), sectors=np.repeat(np.arange(4), N // 4))


def test_book_drift_buy_and_hold():
    from hypershift.fbaudit import book
    fb = _toy_book_fb(flat=True)
    sig = np.arange(5, 30)
    pred = np.tile(np.random.default_rng(12).normal(size=40), (len(sig), 1))
    out = book.simulate_book(pred, fb, sig, 'weekly', 'close1')
    assert out['traded'][0] > 0 and np.all(out['traded'][1:] < 1e-12)
    assert np.all(np.abs(out['gross']) < 1e-12)


def test_book_missing_price_no_nan():
    from hypershift.fbaudit import book
    fb = _toy_book_fb()
    fb.observed[3, 10:15] = False
    fb.price[3, 10:15] = np.nan
    sig = np.arange(5, 30)
    pred = np.random.default_rng(13).normal(size=(len(sig), 40))
    for ex in ('close1', 'open1'):
        ohlcv = np.stack([fb.price] * 5, -1)
        out = book.simulate_book(pred, fb, sig, 'weekly_buffer', ex, ohlcv=ohlcv)
        assert np.isfinite(out['gross']).all() and np.isfinite(out['traded']).all()


def test_lf_corrected_columns_used():
    from types import SimpleNamespace
    from hypershift.fbaudit import prospective_lf as LF
    from hypershift.featbench.data import FEATURES, _roll_mean
    rng = np.random.default_rng(14)
    N, T = 6, 80
    cl = np.cumprod(1 + .01 * rng.normal(size=(N, T)), 1) * 50
    vo = rng.integers(1000, 5000, size=(N, T)).astype(float)
    ohlcv = np.stack([cl, cl * 1.01, cl * .99, cl, vo], -1)
    factor = np.exp(-np.linspace(.3, 0, T))[None] * np.ones((N, 1))   # future-dividend factor: adj = raw * factor
    raw_close = cl / factor
    fb = SimpleNamespace(X=np.zeros((N, T, 20), np.float32), observed=np.ones((N, T), bool), eligible=np.ones((N, T), bool))
    LF.make_leakfree(fb, ohlcv, raw_close)
    want = np.log1p(_roll_mean(raw_close * vo, 20))[:, 50]
    assert np.allclose(fb.X[:, 50, FEATURES.index('log_dvol_20')], want, rtol=1e-5)


def test_lf_guard():
    from types import SimpleNamespace
    from hypershift.fbaudit import prospective_lf as LF
    fb = SimpleNamespace(dates=np.array(['2026-10-20', '2026-10-21', '2026-10-22'], dtype='datetime64[D]'),
                         evaluate=np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        LF.score_lf(fb, {'arms': {}})


def test_audit_scripts_parse():
    """Regression: a sed edit once put a comment inside a call in fbaudit_dividends.py (SyntaxError)."""
    import ast
    for p in sorted(Path('scripts').glob('fbaudit_*.py')) + sorted(Path('scripts').glob('bcreview_*.py')):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))


def test_v1lf_hashes_match_addendum():
    import hashlib
    import re
    text = Path('docs/audit_20261022/PROSPECTIVE_ADDENDUM.md').read_text(encoding='utf-8')
    for path in ('src/hypershift/fbaudit/prospective_lf.py', 'src/hypershift/fbaudit/leakfree.py'):
        want = re.search(re.escape(f'`{path}`') + r'.*?`([0-9a-f]{64})`', text).group(1)
        got = hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        assert got == want, path
