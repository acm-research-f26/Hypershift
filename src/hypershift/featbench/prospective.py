"""Study FB prospective test: the frozen specification scored only on signal dates after the 2026-10-22 freeze.

Hyperparameters are fixed from the development folds (FREEZE_MANIFEST.json); the retrain has no validation step,
so no FB metric is ever computed on 2024-01-01..2026-10-22 (2024+ was spent for Study J arms only).
"""
import numpy as np

from hypershift.featbench.data import FB, compute_features, load_panel, neutral_rankic, target, transform, _neutralise
from hypershift.featbench.models import LGBM_PARAMS, FEATURE_SETS, lagblock
from hypershift.featbench.data import FEATURES
from hypershift.screening.data import forward_labels
from hypershift.screening.metrics import rank_metrics

FREEZE = np.datetime64('2026-10-22')


def check_scoring_dates(dates):
    dates = np.asarray(dates).astype('datetime64[D]')
    if (dates <= FREEZE).any():
        raise ValueError(f'prospective scoring refuses signal dates on or before the {FREEZE} freeze')


def prospective_signals(dates, horizon=5, history=64):
    t = np.arange(history - 1, len(dates) - horizon)
    train = t[dates[t + horizon] <= FREEZE]
    score = t[dates[t] > FREEZE]
    return train, score


def fit_fixed(model, Xtr, ytr, mtr, Xev, params, seed=7, threads=4):
    """Fit with frozen hyperparameters (no validation); returns [S_ev, N] predictions."""
    xtr, yt = Xtr[mtr], ytr[mtr]
    yt = yt / max(float(yt.std()), 1e-8)
    xev = Xev.reshape(-1, Xev.shape[-1])
    if model == 'ridge':
        from sklearn.linear_model import Ridge
        pred = Ridge(alpha=params['alpha']).fit(xtr, yt).predict(xev)
    elif model == 'lgbm':
        import lightgbm as lgb
        p = {**LGBM_PARAMS, 'n_estimators': int(params['n_estimators'])}
        pred = lgb.LGBMRegressor(**p, random_state=seed, n_jobs=threads).fit(xtr, yt).predict(xev)
    else:
        raise ValueError(model)
    return pred.reshape(Xev.shape[:2])


def load_range(panel_path, ohlcv_path, horizon=5):
    """All dates of an extended panel (same format as the development panel)."""
    z = np.load(panel_path)
    last_year = int(str(z['dates'].astype('datetime64[D]')[-1])[:4])
    panel = load_panel(panel_path, last_year, train_stride=1)
    o = np.load(ohlcv_path)
    ident = z['identity_ok'].astype(bool)
    assert (o['tickers'][ident] == panel.tickers).all()
    X = compute_features(o['ohlcv'][ident][:, :len(panel.dates)], panel.observed, panel.beta, panel.sectors,
                         panel.eligible)
    y, ok = forward_labels(panel.price, panel.observed, horizon)
    neutral = _neutralise(y.T, (ok & panel.eligible).T, panel.beta.T, panel.sectors).T
    tr, sc = prospective_signals(panel.dates, horizon)
    tr = tr[(panel.eligible & ok)[:, tr].sum(0) >= 40]
    sc = sc[(panel.eligible & ok)[:, sc].sum(0) >= 40]
    xs = X[:, tr][panel.eligible[:, tr]]
    return FB(X=X, y=y, y_ok=ok, eligible=panel.eligible, beta=panel.beta, sectors=panel.sectors, dates=panel.dates,
              price=panel.price, observed=panel.observed, neutral=neutral.astype(np.float32), train=tr, valid=tr[:0],
              evaluate=sc, horizon=horizon, year=last_year, mu=xs.mean(0), sd=np.maximum(xs.std(0), 1e-8), panel=panel)


def score(fb, manifest, threads=8):
    """Daily neutral RankIC of the frozen spec and comparators on post-freeze signals."""
    check_scoring_dates(fb.dates[fb.evaluate])
    ev, out = fb.evaluate, {}
    for name, spec in manifest['arms'].items():
        if spec['model'] == 'block':
            pred = lagblock(fb, ev, spec['features'])
        elif spec['model'] == 'random':
            pred = np.random.default_rng(spec.get('seed', 7)).normal(size=(len(ev), fb.X.shape[0]))
        else:
            idx = [FEATURES.index(f) for f in FEATURE_SETS[spec['features']]]
            sub = lambda s: transform(fb.X[:, s][..., idx].transpose(1, 0, 2), fb.eligible[:, s].T, spec['repr'], fb)
            ytr, mtr = target(fb, fb.train, spec['target'])
            pred = fit_fixed(spec['model'], sub(fb.train), ytr, mtr & fb.eligible[:, fb.train].T, sub(ev),
                             spec['params'], threads=threads)
        mask = (fb.y_ok[:, ev] & fb.eligible[:, ev]).T
        out[name] = {'neutral_RankIC': neutral_rankic(pred, fb.neutral[:, ev].T, mask),
                     'raw': rank_metrics(pred, np.where(mask, fb.y[:, ev].T, 0.), mask)[0]['RankIC']}
    return out
