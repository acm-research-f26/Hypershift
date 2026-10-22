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


# ---------------------------------------------------------------- Task 3: models and CLI
import csv  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from hypershift.featbench.models import evaluate_arm, fit_predict  # noqa: E402


def planted():
    fb = synthetic_fb(n=60, t=420, seed=3)
    j = FEATURES.index('ret_5')
    z = fb.X[..., j] / max(fb.X[..., j][fb.eligible].std(), 1e-9)
    fb.y = (0.02 * z + np.random.default_rng(4).normal(0, .02, z.shape)).astype(np.float32)
    fb.neutral = fb.y.copy()
    return fb


def split_arrays(fb, s):
    return fb.X[:, s].transpose(1, 0, 2), fb.y[:, s].T, fb.eligible[:, s].T


def test_fit_predict_planted_signal():
    fb = planted()
    Xtr, ytr, mtr = split_arrays(fb, fb.train)
    Xva, nva, mva = split_arrays(fb, fb.valid)
    Xev, _, _ = split_arrays(fb, fb.evaluate)
    for model in ('ridge', 'lgbm'):
        pred, info = fit_predict(model, Xtr, ytr, mtr, Xva, nva, mva, Xev, seed=7, threads=2)
        assert pred.shape == (len(fb.evaluate), 60)
        assert info['val_neutral_RankIC'] > 0.3, (model, info)


@pytest.mark.slow
def test_reversal16_matches_v3():
    rows = list(csv.DictReader(open('docs/screening_20261019/SCREENING_RESULTS.csv')))
    for year in (2022, 2023):
        m = evaluate_arm('block:reversal16', year, 7)
        ref = [r for r in rows if r['arm'] == 'reversal' and r['fold'] == str(year)][0]
        assert abs(m['RankIC'] - float(ref['RankIC'])) < 1e-6, (year, m['RankIC'], ref['RankIC'])
        assert abs(m['neutral_RankIC'] - float(ref['neutral_RankIC'])) < 1e-6, (year, m['neutral_RankIC'])


def test_cli_smoke_writes_run_folder(tmp_path):
    root = tmp_path / 'smoke_root'
    cmd = [sys.executable, 'scripts/featbench.py', '--years', '2019', '--arms', 'ridge:rich:raw:5:cs_z',
           '--max-days', '30', '--root', str(root)]
    subprocess.run(cmd, check=True, env={**__import__('os').environ, 'CUDA_VISIBLE_DEVICES': '-1'})
    run = root / '2019' / 'ridge:rich:raw:5:cs_z'.replace(':', '__') / 'seed_7'
    for name in ('config.json', 'metrics.json', 'predictions.npz', 'daily.npz'):
        assert (run / name).exists(), name


def test_cli_refuses_smoke_without_smoke_root(tmp_path):
    cmd = [sys.executable, 'scripts/featbench.py', '--years', '2019', '--arms', 'ridge:rich',
           '--max-days', '30', '--root', str(tmp_path / 'real')]
    assert subprocess.run(cmd).returncode != 0


# ---------------------------------------------------------------- Task 4: analysis
def _fake_run(root, year, arm, seed, values, dates):
    p = Path(root) / str(year) / arm.replace(':', '__') / f'seed_{seed}'
    p.mkdir(parents=True)
    np.savez(p / 'daily.npz', neutral_RankIC=np.asarray(values, float), RankIC=np.asarray(values, float),
             dates=np.asarray(dates))


def test_paired_matches_manual(tmp_path):
    sys.path.insert(0, 'scripts')
    from featbench_analysis import paired, verdict
    d1, d2 = [f'2019-01-{i:02d}' for i in range(1, 21)], [f'2020-01-{i:02d}' for i in range(1, 21)]
    rng = np.random.default_rng(0)
    a1, b1, a2, b2 = (rng.normal(size=20) for _ in range(4))
    _fake_run(tmp_path, 2019, 'x:a', 7, a1, d1); _fake_run(tmp_path, 2019, 'x:b', 7, b1, d1)
    _fake_run(tmp_path, 2020, 'x:a', 7, a2, d2); _fake_run(tmp_path, 2020, 'x:b', 7, b2[:19], d2[:19])
    r = paired(tmp_path, [2019, 2020], 'x:a', 'x:b')
    manual = np.mean([(a1 - b1).mean(), (a2[:19] - b2[:19]).mean()])
    assert abs(r['pooled'] - manual) < 1e-12
    assert r['ci'][0] <= r['pooled'] <= r['ci'][1]
    assert verdict(0.01, (0.001, 0.02), [0.01, 0.02, 0.01, 0.005, -0.001]) == 'SUPPORTED'
    assert verdict(0.01, (0.001, 0.02), [0.01, 0.02, -0.01, 0.005, -0.001]) == 'NO EVIDENCE'
    assert verdict(0.10, (0.001, 0.02), [0.01] * 5) == 'NO EVIDENCE'


# ---------------------------------------------------------------- Task 7: portfolios and subgroups
from hypershift.featbench.portfolio import simulate, subgroup_rankic, weights_rule  # noqa: E402


def test_simulate_constant_weights_turnover_zero():
    fb = synthetic_fb()
    s = fb.evaluate
    w = np.zeros((len(s), 40)); w[:, :10] = .1
    r = simulate(w, fb, s, hold=5, cost=.001)
    assert r['turnover'] < 1e-12 + .1 * 5 / len(s) * 2  # only the initial ramp-in trades
    r2 = simulate(w, fb, s, hold=5, cost=0.)
    assert r2['net_Sharpe'] >= r['net_Sharpe']


def test_beta_ls_zero_beta():
    fb = synthetic_fb()
    s = fb.evaluate[:10]
    pred = np.random.default_rng(1).normal(size=(10, 40))
    w = weights_rule(pred, fb, s, 'D_beta_ls')
    for d, t in enumerate(s):
        assert abs(w[d] @ fb.beta[:, t]) < 1e-8
        assert abs(w[d][w[d] > 0].sum() - 1) < 1e-12


def test_rules_respect_eligibility_and_gross():
    fb = synthetic_fb()
    s = fb.evaluate[:10]
    fb.eligible[:5, s] = False
    pred = np.random.default_rng(2).normal(size=(10, 40)); pred[:, :5] = 99
    for rule in ('A_top10', 'B_sector_ls', 'C_volscaled', 'D_beta_ls', 'E_buffer'):
        w = weights_rule(pred, fb, s, rule)
        assert np.abs(w[:, :5]).max() == 0, rule
        assert np.allclose(w.clip(0).sum(1), 1), rule


def test_subgroup_terciles_cover_eligible():
    fb = synthetic_fb()
    s = fb.evaluate[:30]
    pred = fb.neutral[:, s].T + np.random.default_rng(3).normal(0, .01, (30, 40))
    out = subgroup_rankic(pred, fb, s, 'vol_60')
    assert set(out) == {'low', 'mid', 'high'}
    assert all(v.mean() > .5 for v in out.values())


# ---------------------------------------------------------------- Task 9: freeze and prospective scorer
from hypershift.featbench.prospective import FREEZE, check_scoring_dates, fit_fixed, prospective_signals  # noqa: E402


def test_prospective_guard():
    with pytest.raises(ValueError):
        check_scoring_dates(np.array(['2026-10-22', '2026-10-23'], dtype='datetime64[D]'))
    check_scoring_dates(np.array(['2026-10-23', '2026-11-02'], dtype='datetime64[D]'))


def test_prospective_signals_split_at_freeze():
    dates = np.arange('2026-01-01', '2027-03-01', dtype='datetime64[D]')
    dates = dates[np.is_busday(dates)]
    train, score = prospective_signals(dates, horizon=5, history=64)
    assert (dates[train + 5] <= FREEZE).all() and (dates[score] > FREEZE).all()
    assert (score + 5 < len(dates)).all()
    assert len(np.intersect1d(train, score)) == 0


def test_fit_fixed_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    Xtr, ytr = rng.normal(size=(30, 20, 4)), rng.normal(size=(30, 20))
    mtr = rng.random((30, 20)) > .2
    Xev = rng.normal(size=(5, 20, 4))
    pred = fit_fixed('ridge', Xtr, ytr, mtr, Xev, {'alpha': 1000})
    ys = ytr[mtr] / ytr[mtr].std()
    ref = Ridge(alpha=1000).fit(Xtr[mtr], ys).predict(Xev.reshape(-1, 4)).reshape(5, 20)
    np.testing.assert_allclose(pred, ref, rtol=1e-10)


def test_score_runs_on_synthetic_post_freeze():
    from hypershift.featbench.prospective import score
    fb = synthetic_fb(n=40, t=400)
    fb.dates = np.arange('2025-12-01', 400, dtype='datetime64[D]')  # spans the freeze
    fb.train = np.arange(64, 300)
    fb.evaluate = np.arange(np.searchsorted(fb.dates, FREEZE) + 1, 390)
    manifest = {'arms': {'frozen': {'model': 'lgbm', 'features': 'rich', 'target': 'residual', 'repr': 'cs_z',
                                    'params': {'n_estimators': 20}},
                         'ridge': {'model': 'ridge', 'features': 'rich', 'target': 'raw', 'repr': 'cs_z',
                                   'params': {'alpha': 1000}},
                         'reversal16': {'model': 'block', 'features': 'reversal16'},
                         'random': {'model': 'random', 'seed': 7}}}
    out = score(fb, manifest, threads=2)
    assert set(out) == set(manifest['arms'])
    assert all(len(v['neutral_RankIC']) == len(fb.evaluate) for v in out.values())
