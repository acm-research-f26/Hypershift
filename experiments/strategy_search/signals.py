"""Ten prespecified families. Every fitted quantity uses the first 60 sessions."""
import numpy as np

from .data import SCOPES
from .features import window_sum, moments, normalize, peers, event_targets
from .storage import arrays, atomic, read, sha


def key(config):
    p = config['parameters']
    return '_'.join(map(str, (config['family'], p.get('window', p.get('opening_window', p.get('early_window', 15))),
                             p.get('factors', 0), config['model'], p.get('forecast_persistence', 0), p.get('max_hold', 0))))


def raw_feature(study, config, scope, fitted):
    m = study.market(scope)
    p, family = config['parameters'], config['family']
    w = p.get('window', p.get('opening_window', p.get('early_window', 15)))
    if family == 'S02':
        return window_sum(m['spread'], m['spread_valid'], m['sessions'], w)
    if family == 'S05':
        filled = np.where(m['valid'], m['raw'] - fitted['pca_mean'], 0)
        basis = fitted['pca_basis'][:, :p['factors']]
        residual = filled - (filled @ basis) @ basis.T
        return window_sum(residual, m['valid'], m['sessions'], w)
    if family == 'S07':
        volume, typical = study.volume(scope)
        good = np.isfinite(typical) & (typical > 0) & np.isfinite(volume) & (volume >= 0) & (m['close'] > 0)
        weighted, ok = window_sum(volume * typical, good, m['sessions'], w)
        total, _ = window_sum(volume, good, m['sessions'], w)
        vwap = np.divide(weighted, total, out=np.ones_like(weighted), where=total > 0)
        ok &= total > 0
        deviation = np.log(np.divide(m['close'], vwap, out=np.ones_like(vwap), where=ok))
        return deviation, ok
    if family == 'S09':
        sector, ok1 = window_sum(m['sector_peers'], m['sector_ok'], m['sessions'], w)
        industry, ok2 = window_sum(m['industry_peers'], m['industry_ok'], m['sessions'], w)
        return .5 * sector + .5 * industry, ok1 & ok2
    if family == 'S08':
        return window_sum(m['raw'], m['valid'], m['sessions'], w)
    return window_sum(m['residual'], m['residual_valid'], m['sessions'], w)


def forecast_mean(study, config, scope):
    forecast = study.forecast(config['model'], scope)
    m = study.market(scope)
    w = config['parameters']['forecast_persistence']
    total, ok = window_sum(forecast, m['eligible_full'], m['sessions'], w)
    positive, _ = window_sum((forecast > 0).astype(float), m['eligible_full'], m['sessions'], w)
    negative, _ = window_sum((forecast < 0).astype(float), m['eligible_full'], m['sessions'], w)
    return total / w, ok, positive == w, negative == w


def fit(study, configs):
    path = study.output / 'fit.npz'
    receipt = study.output / 'fit.json'
    if receipt.exists():
        saved = read(receipt)
        assert sha(path) == saved['sha256']
        with np.load(path) as z:
            return {name: z[name] for name in z.files}
    m = study.market(SCOPES[0])
    train_ids = m['meta']['session_ids'][:60]
    train = np.isin(m['sessions'], train_ids)
    count = m['valid'][train].sum(0)
    mean = np.divide(np.where(m['valid'][train], m['raw'][train], 0).sum(0), count,
                     out=np.zeros(len(count)), where=count > 0)
    x = np.where(m['valid'][train], m['raw'][train] - mean, 0)
    _, vectors = np.linalg.eigh(x.T @ x / max(1, len(x) - 1))
    fitted = dict(pca_mean=mean, pca_basis=vectors[:, ::-1][:, :8],
                  basket_members=m['valid'][train].mean(0) >= .9)
    volume, _ = study.volume(SCOPES[0])
    baseline = np.zeros((390, len(count)))
    for clock in range(390):
        rows = train & (m['clock'] == clock)
        good = np.isfinite(volume[rows]) & (volume[rows] > 0)
        counts = good.sum(0)
        baseline[clock] = np.divide(np.where(good, volume[rows], 0).sum(0), counts,
                                    out=np.full(len(count), np.nan), where=counts >= 20)
    fitted['volume_baseline'] = baseline
    done = set()
    calibration = []
    for config in configs:
        k = key(config)
        if k in done:
            continue
        done.add(k)
        values, ok = raw_feature(study, config, SCOPES[0], fitted)
        center, sd = moments(values, ok, train)
        fitted[k + '_mean'], fitted[k + '_sd'] = center, sd
        if config['family'] == 'S03':
            prediction, pred_ok, _, _ = forecast_mean(study, config, SCOPES[0])
            h = config['parameters']['max_hold']
            forward, forward_ok = window_sum(m['residual'], m['residual_valid'], m['sessions'], h)
            y = np.zeros_like(forward)
            good = np.zeros_like(forward_ok)
            y[:-h] = forward[h:]
            good[:-h] = forward_ok[h:] & train[h:, None] & (m['sessions'][:-h] == m['sessions'][h:])[:, None]
            good &= train[:, None] & pred_ok & m['eligible_full']
            a, b = prediction[good], y[good]
            xm, ym = a.mean(), b.mean()
            beta = float(np.sum((a - xm) * (b - ym)) / max(np.sum((a - xm) ** 2), 1e-20))
            alpha = float(ym - beta * xm)
            fitted[k + '_calibration'] = np.array([alpha, beta])
            calibration.append(dict(key=k, horizon=h, observations=int(good.sum()), alpha=alpha, beta=beta))
    arrays(path, **fitted)
    atomic(receipt, dict(sha256=sha(path), sessions=train_ids,
                        last_completed_ns=int(m['completed'][np.flatnonzero(train)[-1]]),
                        fitting_only=True, training_performance_excluded_from_selection=True,
                        calibration=calibration, pca_imputation='training-mean residual zero for missing returns'))
    return fitted


def build(study, config, scope, fitted):
    m = study.market(scope)
    p, family = config['parameters'], config['family']
    k = key(config)
    raw, available = raw_feature(study, config, scope, fitted)
    z, available = normalize(raw, available, fitted[k + '_mean'], fitted[k + '_sd'])
    eligible = m['eligible_full'] & available
    score = -z
    threshold = p.get('entry_z', 0.)
    score = np.where(np.abs(z) >= threshold, score, 0)
    exit_z = p.get('exit_z', .5)
    exit_long = available & (z >= -exit_z)
    exit_short = available & (z <= exit_z)
    if family == 'S02':
        eligible &= fitted['basket_members'][None, :]
    elif family == 'S03':
        prediction, ok, positive, negative = forecast_mean(study, config, scope)
        alpha, beta = fitted[k + '_calibration']
        edge = alpha + beta * prediction
        score = np.where(((score > 0) & positive & (edge > 0)) | ((score < 0) & negative & (edge < 0)), score, 0)
        eligible &= ok
    elif family == 'S06':
        volume, _ = study.volume(scope)
        baseline = fitted['volume_baseline'][np.minimum(m['clock'], 389)]
        relative = np.divide(volume, baseline, out=np.zeros_like(volume), where=np.isfinite(baseline) & (baseline > 0))
        score = np.where((relative >= p['relative_volume_min']) & (score * m['residual'] > 0), score, 0)
    elif family == 'S07':
        signed, ok = window_sum(m['raw'], m['valid'], m['sessions'], p['efficiency_window'])
        total, _ = window_sum(np.abs(m['raw']), m['valid'], m['sessions'], p['efficiency_window'])
        efficiency = np.divide(np.abs(signed), total, out=np.ones_like(total), where=total > 0)
        eligible &= ok & (efficiency <= p['efficiency_max'])
        score = np.where(np.abs(z) < p['stop_z'], score, 0)
        exit_long = available & ((z >= 0) | (z <= -p['stop_z']))
        exit_short = available & ((z <= 0) | (z >= p['stop_z']))
    elif family == 'S08':
        score = np.zeros_like(z)
        exit_long = np.zeros_like(eligible)
        exit_short = np.zeros_like(eligible)
        w = p['opening_window']
        relative, relative_ok = window_sum(m['residual'], m['residual_valid'], m['sessions'], w)
        for sid in m['meta']['session_ids']:
            ix = np.flatnonzero(m['sessions'] == sid)
            if len(ix) <= w:
                continue
            good = (np.isfinite(m['close'][ix[:w]]) & (m['close'][ix[:w]] > 0)).all(0)
            high = np.max(m['close'][ix[:w]], axis=0)
            low = np.min(m['close'][ix[:w]], axis=0)
            rows = ix[w:]
            up = np.log(np.divide(m['close'][rows], high, out=np.ones_like(z[rows]), where=good[None, :] & (m['close'][rows] > 0)))
            down = np.log(np.divide(m['close'][rows], low, out=np.ones_like(z[rows]), where=good[None, :] & (m['close'][rows] > 0)))
            hurdle = p['breakout_training_SD'] * fitted[k + '_sd']
            score[rows] = np.where((up > hurdle) & (relative[rows] > 0), up / fitted[k + '_sd'],
                                   np.where((down < -hurdle) & (relative[rows] < 0), down / fitted[k + '_sd'], 0))
            exit_long[rows] = m['close'][rows] <= high
            exit_short[rows] = m['close'][rows] >= low
            eligible[ix[:w]] = False
            eligible[rows] &= good[None, :] & relative_ok[rows]
    elif family == 'S09':
        w = p['window']
        own, own_ok = window_sum(m['raw'], m['valid'], m['sessions'], w)
        positive, _ = peers((own > 0).astype(float), own_ok, m['sector'])
        negative, _ = peers((own < 0).astype(float), own_ok, m['sector'])
        score = np.where((z >= 2) & (positive >= p['breadth_min']) & (own <= 0), z,
                         np.where((z <= -2) & (negative >= p['breadth_min']) & (own >= 0), z, 0))
        eligible &= own_ok
        exit_long = available & own_ok & ((own >= .5 * raw) | (raw <= 0))
        exit_short = available & own_ok & ((own <= .5 * raw) | (raw >= 0))
    elif family == 'S10':
        score = np.zeros_like(z)
        exit_long = np.zeros_like(eligible)
        exit_short = np.zeros_like(eligible)
        eligible = eligible.copy()
        for sid in m['meta']['session_ids']:
            ix = np.flatnonzero(m['sessions'] == sid)
            w = p['early_window']
            start = max(w, len(ix) - p['entry_minutes_before_close'])
            early_ok = (m['open'][ix[0]] > 0) & (m['close'][ix[w - 1]] > 0)
            early = np.log(np.divide(m['close'][ix[w - 1]], m['open'][ix[0]], out=np.ones(z.shape[1]), where=early_ok))
            current_ok = (m['close'][ix] > 0) & (m['open'][ix[0]] > 0)
            current = np.log(np.divide(m['close'][ix], m['open'][ix[0]], out=np.ones_like(z[ix]), where=current_ok))
            for sector in np.unique(m['sector']):
                stocks = np.flatnonzero((m['sector'] == sector) & early_ok)
                if len(stocks) < 3:
                    continue
                er = early[stocks] - early[stocks].mean()
                cr = current[:, stocks] - np.mean(current[:, stocks], axis=1)[:, None]
                score[np.ix_(ix[start:], stocks)] = np.where(cr[start:] * er[None, :] > 0,
                                                           er[None, :] / fitted[k + '_sd'][stocks], 0)
            eligible[ix[:start]] = False
            eligible[ix] &= current_ok & early_ok[None, :]
    score = np.where(eligible & np.isfinite(score), score, 0)
    base, count = event_targets(np.asarray(score, float), np.asarray(eligible, bool),
                                np.asarray(exit_long, bool), np.asarray(exit_short, bool), np.asarray(m['vol']),
                                m['sessions'], m['sector'], m['industry'], config['style'] == 'long_short',
                                p.get('max_hold', 390), p.get('max_entries_stock_day', 100000),
                                p.get('max_events_sector_day', 0), family == 'S04', p.get('rebalance', 1),
                                family == 'S02', fitted['basket_members'])
    return base, dict(events=count, family=family, model=config['model'], score_positive=int((score > 0).sum()),
                      score_negative=int((score < 0).sum()), eligible_observations=int(eligible.sum()))
