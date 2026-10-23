import numpy as np
import pytest
from types import SimpleNamespace

from hypershift.strategy import signals as SG, engine as EN, factors as FA


def _data(N=20, T=320, seed=0, flat=False):
    rng = np.random.default_rng(seed)
    close = np.ones((N, T)) * 100 if flat else 100 * np.cumprod(1 + .01 * rng.normal(size=(N, T)), 1)
    open_ = close.copy() if flat else np.concatenate([close[:, :1], close[:, :-1] * (1 + .002 * rng.normal(size=(N, T - 1)))], 1)
    dates = np.arange('2020-01-01', T, dtype='datetime64[D]')
    return SimpleNamespace(close=close, open=open_, observed=np.ones((N, T), bool), eligible=np.ones((N, T), bool),
                           beta=np.ones((N, T)), sectors=np.repeat(np.arange(4), N // 4), dates=dates)


def test_momentum_lowvol_manual():
    d = _data()
    t = 300
    m = SG.momentum(d.close, d.observed, t)
    assert np.allclose(m, d.close[:, t - 21] / d.close[:, t - 252] - 1)
    r = d.close[:, t - 251:t + 1] / d.close[:, t - 252:t] - 1
    assert np.allclose(SG.lowvol(d.close, d.observed, t), -r.std(1, ddof=1))


def test_composite_clip_mean():
    e = np.ones(6, bool)
    a = np.array([0., 1, 2, 3, 4, 100.])          # outlier clipped at +3 sd
    b = np.array([5., 4, 3, 2, 1, 0.])
    c = SG.composite([a, b, np.full(6, np.nan)], e)
    za = np.clip((a - a.mean()) / a.std(), -3, 3)
    zb = np.clip((b - b.mean()) / b.std(), -3, 3)
    assert np.allclose(c, (za + zb) / 2)
    assert np.isnan(SG.composite([a, np.full(6, np.nan), np.full(6, np.nan)], e)).all()   # < 2 available


def test_month_starts():
    dates = np.array(['2020-01-30', '2020-01-31', '2020-02-03', '2020-02-04', '2020-03-02'], dtype='datetime64[D]')
    assert EN.month_starts(dates).tolist() == [2, 4]


def test_open_execution_no_lookahead():
    d = _data(N=4, T=40, flat=True)
    d.sectors = np.zeros(4, int)
    t = 20
    d.close[0, t - 1:] = 110.0      # jump at the close of the signal day (t-1)
    d.open[0, t:] = 110.0
    scores = {t: np.array([10., 0, 0, 0])}
    out = EN.run(scores, d, mode='LO', top=.25, band=.5, execution='open', cost_bp=0)
    assert abs(out['gross'].sum()) < 1e-12     # bought at the post-jump open: the jump is not captured


def test_benchmark_costed():
    d = _data(N=8, T=60, flat=True)
    sc = {20: np.arange(8.), 41: np.arange(8.)[::-1]}
    b = EN.benchmark_ew(sc, d, execution='open', cost_bp=10)
    assert b['traded'][0] > 0 and np.isclose(b['net'][0], -10e-4 * b['traded'][0])


def test_engine_missing_open():
    d = _data(N=8, T=60)
    d.observed[3, 20] = False
    d.open[3, 20] = np.nan
    d.close[3, 20] = np.nan
    out = EN.run({20: np.arange(8.)}, d, mode='LO', top=.5, band=.75, execution='open', cost_bp=10)
    assert np.isfinite(out['net']).all() and np.isfinite(out['traded']).all()


def test_borrow_accrual():
    d = _data(N=40, T=60, flat=True)
    d.sectors = np.repeat(np.arange(2), 20)
    out = EN.run({20: np.arange(40.)}, d, mode='LS', top=.2, band=.4, execution='open', cost_bp=0, borrow_bp=50)
    short_gross = .5 + 0          # flat prices, beta 1 for all -> hedge exactly offsets; short leg 0.5 of NAV
    assert np.allclose(out['borrow'][1:], 50e-4 / 252 * short_gross, rtol=1e-6)


def test_engine_flat_prices_cost_only():
    d = _data(N=10, T=60, flat=True)
    out = EN.run({20: np.arange(10.)}, d, mode='LO', top=.2, band=.4, execution='open', cost_bp=10)
    assert np.allclose(out['gross'], 0) and np.isclose(out['traded'][0], 1.0)
    assert np.isclose(out['net'].sum(), -10e-4)


def test_ff_loader_units(tmp_path):
    (tmp_path / 'F-F_Research_Data_5_Factors_2x3_daily.csv').write_text(
        'header\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20200102,  1.00, 0.10, 0.20, 0.30, 0.40, 0.01\n20200103, -0.50, 0, 0, 0, 0, 0.01\n\nCopyright\n')
    (tmp_path / 'F-F_Momentum_Factor_daily.csv').write_text('x\n\n,Mom\n20200102, 0.50\n20200103, 0.25\n\nCopyright\n')
    (tmp_path / 'F-F_ST_Reversal_Factor_daily.csv').write_text('x\n\n,ST_Rev\n20200103, 0.30\n\nCopyright\n')
    ff = FA.load_ff(tmp_path)
    assert ff['dates'].tolist() == [np.datetime64('2020-01-03')]       # only dates present in all files
    assert np.isclose(ff['Mkt-RF'][0], -.005) and np.isclose(ff['STREV'][0], .003) and np.isclose(ff['MOM'][0], .0025)


def test_attribute_recovers_alpha():
    rng = np.random.default_rng(1)
    n = 3000
    dates = np.arange('2010-01-01', n, dtype='datetime64[D]')
    ff = {'dates': dates, **{k: rng.normal(0, .01, n) for k in FA.FACTORS}, 'RF': np.zeros(n)}
    ret = .0004 + .5 * ff['Mkt-RF'] - .3 * ff['MOM'] + rng.normal(0, .002, n)
    a = FA.attribute(ret, dates, ff)
    assert abs(a['alpha_ann'] - .0004 * 252) < .02 and abs(a['betas']['Mkt-RF'] - .5) < .02 and a['t'] > 3


def test_trace_contributions_sum_to_gross_close_exec():
    d = _data(N=40, T=80, seed=3)
    d.sectors = np.repeat(np.arange(2), 20)
    out = EN.run({20: np.arange(40.), 45: np.arange(40.)[::-1]}, d, mode='LS', top=.2, band=.4, execution='close',
                 cost_bp=0, borrow_bp=0, trace=True)
    contrib = (out['w_start'] * out['r']).sum(1)
    assert np.allclose(contrib, out['gross'], atol=1e-12)
    assert out['w_start'].shape == (len(out['gross']), 40)


def test_ls_weights_beta_neutral():
    rng = np.random.default_rng(5)
    N = 60
    sectors = np.repeat(np.arange(3), 20)
    beta = rng.normal(1, .3, N)
    tradable = np.ones(N, bool)
    w, lo, sh = EN.ls_weights(rng.normal(size=N), tradable, sectors, beta, np.zeros(N, bool), np.zeros(N, bool), .1, .3)
    assert abs(w @ beta) < 1e-10
    assert len(lo) == len(sh) == 6


def _toy_fb(N=40, T=200, seed=7):
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    return SimpleNamespace(X=rng.normal(size=(N, T, 20)).astype(np.float32), y=rng.normal(0, .02, (N, T)),
                           y_ok=np.ones((N, T), bool), eligible=np.ones((N, T), bool), beta=rng.normal(1, .2, (N, T)),
                           sectors=np.repeat(np.arange(4), N // 4), horizon=5)


def test_style_predictions_guard_and_shape():
    from hypershift.strategy import paper
    fb = _toy_fb()
    with pytest.raises(ValueError):
        paper.style_scores(fb, np.arange(70, 190), asof=190, seeds=(7,))      # label of t=189 matures after 190
    s = paper.style_scores(fb, np.arange(70, 185), asof=190, seeds=(7, 19))
    assert s.shape == (40,) and np.isfinite(s).all()


def test_paper_trading_manifest_hashes_unchanged():
    import hashlib
    import json
    from pathlib import Path
    m = json.load(open('docs/strategy_20261022/STRATEGY_MANIFEST.json'))
    for f, h in m['source_sha256'].items():
        assert hashlib.sha256(Path(f).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == h, f
