"""Independent frozen-checkpoint, ranking, and continuous-account verification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_historical_ndx import replay, independent_metrics
from audit_historical_ndx_ranking import average_ranks, diagnostics
from ndx_alpaca_models import make, inputs, scale
from pit_research_engine import load_data, fold, historical_groups
from prepare_ndx_alpaca import RAW, CALENDAR, IDENTITY
from report_ndx_alpaca import check_complete
from run_ndx_alpaca_training import RUN, FEATURE_ROOT, feature_data, sha, write
from run_robust_research import predict, model_data
from training_device import configure_device


def independent_ensemble(predictions, mask):
    if len(predictions)==1:
        return predictions[0]
    ranked = []
    for prediction in predictions:
        scores = np.zeros_like(prediction, dtype=float)
        for j, eligible in enumerate(mask):
            scores[j, eligible] = average_ranks(prediction[j, eligible])/eligible.sum()
        ranked.append(scores)
    return np.mean(ranked, axis=0)


def sample_intraday_review():
    review = json.loads((FEATURE_ROOT/'REVIEW.json').read_text())
    if sha(Path(__file__).with_name('prepare_ndx_alpaca.py'))!=review['code_sha256']:
        raise ValueError('Feature-builder source changed')
    if sha(RAW/'manifest.json')!=review['raw_manifest_sha256']:
        raise ValueError('Raw source manifest changed')
    for relative, expected in review['source_hashes'].items():
        if sha(RAW/relative) != expected:
            raise ValueError('Raw source bytes changed')
    if sha(CALENDAR) != review['calendar_sha256']:
        raise ValueError('Calendar bytes changed')
    if sha(IDENTITY) != review['identity_rules_sha256']:
        raise ValueError('Reviewed symbol identity rules changed')
    summary = pd.read_csv(FEATURE_ROOT/'session_summaries.csv.gz')
    for symbol, rule in review['identity_rules']['symbols'].items():
        rows = summary[summary.source_symbol==symbol]
        if not rows.security_id.eq(rule['security_id']).all():
            raise ValueError('Canonical identity mismatch')
        if 'first_valid_date' in rule and (rows.date<rule['first_valid_date']).any():
            raise ValueError('Source session precedes issuer validity')
        if 'end_valid_date' in rule and (rows.date>=rule['end_valid_date']).any():
            raise ValueError('Source session follows issuer validity')
    if summary.duplicated(['security_id','date']).any():
        raise ValueError('Overlapping accepted canonical sessions')
    rejected = pd.read_csv(FEATURE_ROOT/'rejected_symbol_sessions.csv')
    if len(rejected)!=review['rejected_symbol_sessions']:
        raise ValueError('Rejected-session inventory mismatch')
    complete = summary[summary.complete.astype(bool)].sample(40, random_state=20261019)
    records = json.loads((RAW/'manifest.json').read_text())['chunks']
    calendar = {r['date']: r for r in json.loads(CALENDAR.read_text())}
    checks = []
    cache = {}
    for row in complete.itertuples(index=False):
        record = next(r for r in records if r['query']['symbol']==row.source_symbol and
                      r['query']['start'][:10] <= row.date < r['query']['end_exclusive'][:10])
        filename = record['file']
        if filename not in cache:
            frame = pd.read_csv(RAW/filename)
            frame['ts'] = pd.to_datetime(frame.timestamp_utc, utc=True, format='ISO8601')
            cache[filename] = frame
        frame = cache[filename]
        session = calendar[row.date]
        start = pd.Timestamp(row.date+' '+session['open'], tz='America/New_York').tz_convert('UTC')
        stop = pd.Timestamp(row.date+' '+session['close'], tz='America/New_York').tz_convert('UTC')
        part = frame[(frame.ts>=start)&(frame.ts+pd.Timedelta(minutes=15)<=stop)].sort_values('ts')
        expected = int((stop-start).total_seconds()/900)
        if len(part)!=expected or part.ts.duplicated().any():
            raise ValueError('Sample session completeness mismatch')
        o, h, l, c, v, vw = [part[name].to_numpy(float) for name in ['open','high','low','close','volume','vwap']]
        calculated = dict(rth_return=c[-1]/o[0]-1, rth_range=h.max()/l.min()-1,
            within_bar_realized_vol=np.linalg.norm(np.log(c/o)), first_hour_return=c[3]/o[0]-1,
            last_hour_return=c[-1]/o[-4]-1, late_volume_share=v[-4:].sum()/v.sum(),
            close_to_vwap=c[-1]/np.dot(v, vw)*v.sum()-1, intraday_coverage=1.)
        for name, value in calculated.items():
            np.testing.assert_allclose(value, getattr(row, name), rtol=1e-8, atol=1e-10)
        checks.append(dict(symbol=row.source_symbol, date=row.date, expected_bars=expected, passed=True))
    pd.DataFrame(checks).to_csv(RUN/'independent_intraday_samples.csv', index=False)
    return len(checks), len(review['source_hashes'])


def main():
    audit_sources = [Path(__file__).resolve(), Path(__file__).with_name('audit_historical_ndx.py'),
                     Path(__file__).with_name('audit_historical_ndx_ranking.py')]
    audit_lock = {str(path):sha(path) for path in audit_sources}
    plan = check_complete()
    if not (RUN/'completion.json').exists():
        raise ValueError('Evaluation must complete before audit')
    for name, expected in json.loads((RUN/'EVALUATION_CODE_LOCK.json').read_text()).items():
        if sha(name)!=expected:
            raise ValueError('Evaluation source changed')
    data = load_data()
    feature, _, _ = feature_data()
    ids = np.concatenate([fold(data, year)[2] for year in plan['test_years']])
    forecasts, reproduced = {}, 0
    device = configure_device('cuda')
    for view in plan['views']:
        raw = inputs(data, view, feature if view=='intraday_augmented' else None)
        by_kind = {kind: [[] for _ in plan['seeds']] for kind in plan['models']}
        ridge_parts = []
        for year in plan['test_years']:
            train, val, test = fold(data, year)
            x, mean, std = scale(raw, data['mask'], train)
            with np.load(RUN/view/str(year)/'preprocessing.npz') as saved:
                for name, value in [('mean', mean), ('std', std), ('train', train), ('val', val), ('test', test)]:
                    np.testing.assert_array_equal(saved[name], value)
            # Perturb future inputs and verify fitted normalization never changes.
            changed = raw.copy()
            changed[np.max(train)+1:] = 1e10
            _, other_mean, other_std = scale(changed, data['mask'], train)
            np.testing.assert_array_equal(mean, other_mean)
            np.testing.assert_array_equal(std, other_std)
            groups = historical_groups(data, train)
            for kind in plan['models']:
                for s, seed in enumerate(plan['seeds']):
                    folder = RUN/view/str(year)
                    checkpoint = torch.load(folder/f'{kind}_{seed}.pt', map_location='cpu', weights_only=False)
                    if checkpoint['groups'] != [sorted(g) for g in groups] or checkpoint['features'] != 13:
                        raise ValueError('Checkpoint architecture metadata mismatch')
                    model = make(kind, len(data['panel'].tickers), groups).to(device)
                    model.load_state_dict(checkpoint['state_dict'])
                    vp = predict(model, x, data['mask'], val, plan['batch'])
                    tp = predict(model, x, data['mask'], test, plan['batch'])
                    with np.load(folder/f'{kind}_{seed}_predictions.npz') as saved:
                        np.testing.assert_allclose(vp, saved['validation'], rtol=0, atol=1e-6)
                        np.testing.assert_allclose(tp, saved['test'], rtol=0, atol=1e-6)
                    history = json.loads((folder/f'{kind}_{seed}_history.json').read_text())
                    best, chosen = np.inf, 0
                    for record in history:
                        if record['validation_mse'] < best-1e-7:
                            best, chosen = record['validation_mse'], record['epoch']
                    if checkpoint['epoch'] != chosen:
                        raise ValueError('Selected checkpoint epoch mismatch')
                    mse = ((vp-data['y'][val]*100)**2)[data['labelmask'][val]].mean()
                    np.testing.assert_allclose(mse, best, rtol=1e-6, atol=1e-6)
                    by_kind[kind][s].append(tp)
                    reproduced += 1
            design = x.transpose(0, 2, 1, 3).reshape(len(x), len(data['panel'].tickers), -1)
            design = np.concatenate([design, np.ones((*design.shape[:2],1))], axis=-1)
            a = design[train][data['labelmask'][train]]
            y = (data['y'][train]*100)[data['labelmask'][train]]
            penalty = np.eye(a.shape[-1])*10.
            penalty[-1,-1] = 0.
            coefficients = np.linalg.solve(a.T@a+penalty,a.T@y)
            with np.load(RUN/'ridge'/view/f'{year}.npz') as saved:
                np.testing.assert_allclose(coefficients, saved['coefficients'], rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(design[test]@coefficients, saved['test'], rtol=1e-9, atol=1e-9)
            ridge_parts.append(design[test]@coefficients)
            print(f'Audited {view} {year}: {reproduced}/280 checkpoints', flush=True)
        for kind, seeds in by_kind.items():
            forecasts[view, kind] = [np.concatenate(parts) for parts in seeds]
        forecasts[view, 'ridge'] = [np.concatenate(ridge_parts)]
    forecasts['control', 'momentum126'] = [data['momentum'][ids]]
    forecasts['control', 'reversal5'] = [data['reversal'][ids]]
    independent_scores = {}
    ranking_differences = []
    ranking = pd.read_csv(RUN/'ranking.csv').set_index(['view','model'])
    with np.load(RUN/'portfolio_scores.npz') as saved:
        np.testing.assert_array_equal(saved['ids'], ids)
        for key, available in forecasts.items():
            adjusted = model_data(data, key[1])
            ensemble = independent_ensemble(available, adjusted['mask'][ids])
            independent_scores[key] = ensemble
            np.testing.assert_allclose(ensemble, saved['__'.join(key)], rtol=0, atol=1e-12)
            fields = []
            for number, i in enumerate(ids):
                observed = adjusted['labelmask'][i]
                if observed.sum()>=2:
                    fields.append(diagnostics(ensemble[number,observed], adjusted['y'][i,observed], (5,10,20)))
            frame = pd.DataFrame(fields)
            for name in ['IC','RankIC','NDCG@5','NDCG@10','NDCG@20']:
                difference = float(abs(frame[name].mean()-ranking.loc[key,name]))
                ranking_differences.append(dict(view=key[0],model=key[1],metric=name,absolute_difference=difference))
                # Production Pearson IC centers float32 targets; NumPy's independent
                # correlation promotes to float64. Preserve tight rank/NDCG checks.
                np.testing.assert_allclose(frame[name].mean(), ranking.loc[key,name],
                                           atol=1e-8 if name=='IC' else 1e-10, rtol=1e-9)
    pd.DataFrame(ranking_differences).to_csv(RUN/'independent_ranking_precision.csv',index=False)
    curves = pd.read_csv(RUN/'curves.csv', dtype={'seed':str})
    metrics = pd.read_csv(RUN/'metrics.csv', dtype={'seed':str})
    keys = ['view','model','seed','k','cost_multiplier']
    indexed = metrics.set_index(keys)
    largest, checked = 0., 0
    for key, part in curves.groupby(keys, sort=False):
        view, kind, seed, k, mult = key
        if len(part)!=1003 or not part.date.is_unique or not part.date.is_monotonic_increasing:
            raise ValueError('Invalid continuous return calendar')
        observed = independent_metrics(part.net)
        for name, value in observed.items():
            np.testing.assert_allclose(value, indexed.loc[key,name], rtol=1e-10, atol=1e-10)
        if kind in ['SPY','QQQ']:
            root = Path(data['manifest']['root'])
            prices = pd.read_csv(root/'etf_adj_close.csv', index_col=0)[kind].to_numpy()
            volumes = pd.read_csv(root/'etf_dollar_volume.csv', index_col=0)[kind].to_numpy()
            first = data['panel'].dates.get_loc(pd.Timestamp('2022-01-03'))
            rate = mult*(.0007+.0003*np.sqrt(1e6/volumes[first-20:first].mean()/.01))
            units = 1e6/(1+rate)/prices[first]
            nav = units*prices[first:]
            rate = mult*(.0007+.0003*np.sqrt(nav[-1]/volumes[-21:-1].mean()/.01))
            nav[-1] *= 1-rate
        else:
            adjusted = model_data(data, kind)
            if kind=='monthly_basket':
                score = np.zeros_like(data['y'][ids])
            elif seed=='ensemble':
                score = independent_scores[view, kind]
            else:
                score = forecasts[view, kind][plan['seeds'].index(int(seed))]
            nav = replay(adjusted, ids, score, k, mult, basket=kind=='monthly_basket')
        published = 1e6*np.cumprod(1+part.net.to_numpy())
        largest = max(largest, float(np.max(np.abs(nav-published))))
        np.testing.assert_allclose(nav, published, rtol=1e-10, atol=1e-5)
        checked += 1
        if checked%30==0:
            print(f'{checked}/241 accounts independently replayed', flush=True)
    samples, chunks = sample_intraday_review()
    ledger = pd.read_csv(RUN/'primary_account_ledger.csv',dtype={'seed':str})
    np.testing.assert_allclose(ledger.nav,ledger.cash+ledger.holdings,rtol=1e-12,atol=1e-7)
    if ledger.cash.min() < -1e-5:
        raise ValueError('Negative cash in exported primary ledger')
    for (view,kind), account in ledger.groupby(['view','model'],sort=False):
        path = curves[(curves.view==view)&(curves.model==kind)&(curves.seed=='ensemble')&
                      (curves.k==10)&(curves.cost_multiplier==1)]
        np.testing.assert_array_equal(account.date,path.date)
        np.testing.assert_allclose(account.nav,1e6*np.cumprod(1+path.net),rtol=1e-10,atol=1e-5)
    if reproduced!=280 or checked!=241:
        raise ValueError('Incomplete independent audit')
    if any(sha(name)!=expected for name,expected in audit_lock.items()):
        raise ValueError('Independent audit source changed during verification')
    result = dict(status='passed', checkpoints_reproduced=reproduced, independent_accounts=checked,
        ridge_models_reproduced=8, largest_nav_difference_dollars=largest,
        raw_chunk_hashes_verified=chunks, independent_intraday_session_samples=samples,
        symbol_identity_windows_verified=True,
        rank_ensembles_and_metrics_recomputed=len(independent_scores), annual_resets=0,
        maximum_ranking_metric_difference=max(row['absolute_difference'] for row in ranking_differences),
        pearson_ic_absolute_tolerance=1e-8,
        pearson_precision_note='Production centers float32 targets; independent NumPy correlation uses float64. Rank/NDCG tolerance remains 1e-10.',
        limitation='Exploratory inspected years; conditional inference; raw provider identities not independently certified for every row.')
    write(RUN/'AUDIT.json', result)
    write(RUN/'AUDIT_CODE_LOCK.json', audit_lock)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
