"""Study FB models (Ridge, LightGBM, MLP), lag-block scorers and single-arm evaluation."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from hypershift.featbench.data import FEATURES, load_fb, neutral_rankic, target, transform
from hypershift.screening.metrics import rank_metrics, trading

RIDGE_ALPHAS = (0.1, 1, 10, 100, 1000)
LGBM_PARAMS = dict(objective='regression', num_leaves=15, max_depth=4, learning_rate=0.05, n_estimators=1000,
                   min_child_samples=1000, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                   deterministic=True, force_row_wise=True, verbose=-1)
FEATURE_SETS = {
    'rich': FEATURES,
    'price': [f for f in FEATURES if f not in ('log_dvol_20', 'relvol_5_60', 'volchg_5_20', 'amihud_20', 'hl_range_20')],
}
BLOCKS = ('l1', 'l2_5', 'l6_20', 'l21_60', 'reversal16')
LAG_SET = 'lags'
DEFAULTS = ['?', '?', 'raw', '5', 'cs_z']


def parse_arm(arm):
    parts = arm.split(':')
    parts += DEFAULTS[len(parts):]
    model, feats, tgt, h, rep = parts[:5]
    return {'model': model, 'features': feats, 'target': tgt, 'horizon': int(h), 'repr': rep}


def folder_name(arm):
    return arm.replace(':', '__')


def _rows(X, y, m):
    return X[m], (y[m] if y is not None else None)


def _scatter(pred_rows, m):
    out = np.zeros(m.shape, np.float64)
    out[m] = pred_rows
    return out


def _val_score(pred_rows, nva, mva):
    return float(neutral_rankic(_scatter(pred_rows, mva), nva, mva).mean())


def fit_predict(model, Xtr, ytr, mtr, Xva, nva, mva, Xev, seed=7, threads=4):
    """Arrays are [S, N, F] / [S, N]; selection on validation mean neutral RankIC. Returns ([S_ev, N], info)."""
    xtr, yt = _rows(Xtr, ytr, mtr)
    scale = max(float(yt.std()), 1e-8)
    yt = yt / scale
    xva = Xva[mva]
    mev = np.ones(Xev.shape[:2], bool)
    xev = Xev.reshape(-1, Xev.shape[-1])
    info = {}
    if model == 'ridge':
        from sklearn.linear_model import Ridge
        best = None
        for a in RIDGE_ALPHAS:
            r = Ridge(alpha=a).fit(xtr, yt)
            s = _val_score(r.predict(xva), nva, mva)
            info[f'val_alpha_{a}'] = s
            if best is None or s > best[0]:
                best = (s, a, r)
        info.update(alpha=best[1], val_neutral_RankIC=best[0])
        pred = best[2].predict(xev)
    elif model == 'lgbm':
        import lightgbm as lgb
        state = {'best': -np.inf, 'iter': 0}

        def monitor(env):
            it = env.iteration + 1
            if it % 10:
                return
            s = _val_score(env.model.predict(xva, num_iteration=it), nva, mva)
            if s > state['best']:
                state.update(best=s, iter=it)
            elif it - state['iter'] >= 50:
                raise lgb.callback.EarlyStopException(state['iter'], [])
        reg = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=seed, n_jobs=threads)
        reg.fit(xtr, yt, callbacks=[monitor])
        it = max(state['iter'], 10)
        info.update(best_iteration=it, val_neutral_RankIC=state['best'])
        pred = reg.booster_.predict(xev, num_iteration=it)
    elif model == 'mlp':
        from sklearn.neural_network import MLPRegressor
        preds, vals = [], []
        for s in (7, 19, 42):
            mlp = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-4, learning_rate_init=1e-3, max_iter=50,
                               early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, random_state=s)
            mlp.fit(xtr, yt)
            preds.append(mlp.predict(xev)); vals.append(mlp.predict(xva))
        pred = np.mean(preds, 0)
        info.update(val_neutral_RankIC=_val_score(np.mean(vals, 0), nva, mva), seeds=[7, 19, 42])
    else:
        raise ValueError(model)
    return pred.reshape(Xev.shape[:2]), info


def _lagret(price, signals, a, b):
    """close_{t-a} / close_{t-b} - 1 at each signal t, [S, N]."""
    return (price[:, signals - a] / np.maximum(price[:, signals - b], 1e-12) - 1).T


def lagblock(fb, signals, block):
    if block == 'reversal16':
        return -_lagret(fb.price, signals, 0, 16)
    a, b = {'l1': (0, 1), 'l2_5': (1, 5), 'l6_20': (5, 20), 'l21_60': (20, 60)}[block]
    return _lagret(fb.price, signals, a, b)


def _block_sign(fb, block):
    if block == 'reversal16':
        return 1.
    tr = fb.train
    m = (fb.y_ok[:, tr] & fb.eligible[:, tr]).T
    s = neutral_rankic(lagblock(fb, tr, block), fb.neutral[:, tr].T, m).mean()
    return 1. if s >= 0 else -1.


def _features(fb, signals, feats, rep):
    if feats == LAG_SET:
        raw = np.stack([lagblock(fb, signals, b) for b in BLOCKS[:4]], -1)
    else:
        idx = [FEATURES.index(f) for f in FEATURE_SETS[feats]]
        raw = fb.X[:, signals][..., idx].transpose(1, 0, 2)
        if rep == 'raw':
            sub = type('Sub', (), {})()
            sub.mu, sub.sd, sub.sectors = fb.mu[idx], fb.sd[idx], fb.sectors
            return transform(raw, fb.eligible[:, signals].T, 'raw', sub)
    if rep == 'volscaled_z' and feats != 'rich':
        raise ValueError('volscaled_z needs the rich feature set (uses vol_60)')
    return transform(raw, fb.eligible[:, signals].T, rep, fb)


_FB_CACHE = {}


def _get_fb(year, horizon):
    key = (year, horizon)
    if key not in _FB_CACHE:
        _FB_CACHE.clear()
        _FB_CACHE[key] = load_fb(year, horizon)
    return _FB_CACHE[key]


def _truncate(fb, max_days):
    if max_days:
        fb.train, fb.valid, fb.evaluate = fb.train[-max_days:], fb.valid[-max_days:], fb.evaluate[:max_days]
    return fb


def evaluate_arm(arm, year, seed, threads=4, max_days=None, return_arrays=False):
    a = parse_arm(arm)
    fb = _get_fb(year, a['horizon'])
    if max_days:
        import copy
        fb = _truncate(copy.copy(fb), max_days)
    ev = fb.evaluate
    info = {}
    if a['model'] == 'block':
        pred = _block_sign(fb, a['features']) * lagblock(fb, ev, a['features'])
        info['sign'] = _block_sign(fb, a['features'])
    else:
        Xtr = _features(fb, fb.train, a['features'], a['repr'])
        ytr, mtr = target(fb, fb.train, a['target'])
        mtr = mtr & fb.eligible[:, fb.train].T
        Xva = _features(fb, fb.valid, a['features'], a['repr'])
        mva = (fb.y_ok[:, fb.valid] & fb.eligible[:, fb.valid]).T
        nva = fb.neutral[:, fb.valid].T
        Xev = _features(fb, ev, a['features'], a['repr'])
        pred, info = fit_predict(a['model'], Xtr, ytr, mtr, Xva, nva, mva, Xev, seed=seed, threads=threads)
    raw = fb.y[:, ev].T
    mask = (fb.y_ok[:, ev] & fb.eligible[:, ev]).T
    pred = np.where(mask | fb.eligible[:, ev].T, pred, 0.)
    metrics, daily = rank_metrics(pred, np.where(mask, raw, 0.), mask)
    neutral = fb.neutral[:, ev].T
    nm, nd = rank_metrics(pred, np.where(mask, neutral, 0.), mask)
    style, _ = target(fb, ev, 'residual')
    sm, sd = rank_metrics(pred, style, mask)
    metrics.update(neutral_RankIC=nm['RankIC'], neutral_spread10=nm['spread10'], style_neutral_RankIC=sm['RankIC'],
                   NDCG10_excess=metrics['NDCG10'] - metrics['random_NDCG10'], days=int(len(ev)), info=info)
    if a['horizon'] == 5:
        tr, pnl = trading(pred, ev, fb.panel)
        metrics.update(tr)
    else:
        pnl = np.zeros(0)
    daily_out = dict(RankIC=daily['RankIC'], neutral_RankIC=nd['RankIC'], style_neutral_RankIC=sd['RankIC'],
                     dates=fb.dates[ev].astype(str), portfolio=pnl)
    if return_arrays:
        return metrics, daily_out, dict(prediction=pred.astype(np.float32), target=raw.astype(np.float32),
                                        neutral=neutral.astype(np.float32), mask=mask, signals=ev,
                                        dates=fb.dates[ev].astype(str), tickers=fb.panel.tickers)
    return metrics


def source_hashes():
    out = {}
    for p in ('src/hypershift/featbench/data.py', 'src/hypershift/featbench/models.py', 'scripts/featbench.py'):
        if Path(p).exists():
            out[p] = hashlib.sha256(Path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return out


def run_one(root, arm, year, seed, threads=4, max_days=None):
    path = Path(root) / str(year) / folder_name(arm) / f'seed_{seed}'
    if (path / 'metrics.json').exists():
        return 'skip'
    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    path.mkdir(parents=True, exist_ok=True)
    metrics, daily, arrays = evaluate_arm(arm, year, seed, threads, max_days, return_arrays=True)
    cfg = {'arm': arm, **parse_arm(arm), 'year': year, 'seed': seed, 'protocol': 'featbench-fb-20261022',
           'max_days': max_days, 'lgbm_params': LGBM_PARAMS, 'ridge_alphas': RIDGE_ALPHAS, 'source': source_hashes()}
    (path / 'config.json').write_text(json.dumps(cfg, indent=1, default=str))
    np.savez_compressed(path / 'predictions.npz', **arrays)
    np.savez_compressed(path / 'daily.npz', **daily)
    (path / 'metrics.json').write_text(json.dumps(metrics, indent=1, default=float))
    return f"{year} {arm} seed {seed}: RankIC {metrics['RankIC']:+.4f} neutral {metrics['neutral_RankIC']:+.4f}"
