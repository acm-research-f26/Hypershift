"""Independently check frozen ranking summaries; publish dated diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data

RUN = Path('runs/nasdaq100_historical_monthly_20261011')
OUT = Path('docs/nasdaq_monthly_pit')
FIELDS = ['IC', 'RankIC', 'NDCG@5', 'NDCG@10', 'NDCG@20']


def average_ranks(values):
    """Pairwise definition, independent of production pandas rank calls."""
    v = np.asarray(values)
    return (v[:, None] > v).sum(axis=1) + ((v[:, None] == v).sum(axis=1) + 1) / 2


def diagnostics(scores, returns, ks=(5, 10, 20)):
    p, y = np.asarray(scores), np.asarray(returns)
    if len(y) < 2:
        return {field: np.nan for field in FIELDS + [f'Random NDCG@{k}' for k in ks]}
    ranked_p, ranked_y = average_ranks(p), average_ranks(y)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else np.nan
    result = dict(IC=corr(p, y), RankIC=corr(ranked_p, ranked_y))
    gains = (ranked_y - 1) / (len(y) - 1)
    # Each tied prediction occupies each position in its tied group uniformly.
    starts = (p[:, None] < p).sum(axis=1)
    sizes = (p[:, None] == p).sum(axis=1)
    for k in ks:
        discounts = np.zeros(len(y))
        discounts[:min(k, len(y))] = 1 / np.log2(np.arange(min(k, len(y))) + 2)
        expected_discount = np.array([discounts[a:a+b].sum() / b for a, b in zip(starts, sizes)])
        ideal = np.sort(gains)[::-1] @ discounts
        result[f'NDCG@{k}'] = float(gains @ expected_discount / ideal) if ideal else np.nan
        result[f'Random NDCG@{k}'] = float(gains.mean() * discounts.sum() / ideal) if ideal else np.nan
    return result


def main():
    data = load_data()
    plan = json.loads((RUN/'PROTOCOL.json').read_text())
    inputs = json.loads((RUN/'INPUT_CODE_LOCK.json').read_text())
    inputs.update(json.loads((RUN/'DEPENDENCY_CODE_LOCK.json').read_text()))
    for path, expected in inputs.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    frozen = pd.read_csv(RUN/'ranking.csv').set_index('model')
    records = []
    hashes = {}
    for kind in frozen.index:
        for year in plan['test_years']:
            folder = RUN/str(year)
            with np.load(folder/'preprocessing.npz') as prep:
                ids, mean, std = prep['test'], prep['mean'], prep['std']
            eligible = data['mask'][ids].copy()
            observed = data['labelmask'][ids].copy()
            if kind in plan['models']:
                seed_ranks = []
                for seed in plan['training_seeds']:
                    path = folder/f'{kind}_{seed}_predictions.npz'
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path) as saved:
                        pred = saved['test']
                    rank = np.zeros_like(pred, dtype=float)
                    for j, mask in enumerate(eligible):
                        rank[j, mask] = average_ranks(pred[j, mask]) / mask.sum()
                    seed_ranks.append(rank)
                score = np.mean(seed_ranks, axis=0)
            elif kind == 'ridge':
                x = (np.clip((data['x'][ids] - mean) / std, -8, 8) * .2).astype('float32')
                design = x.transpose(0, 2, 1, 3).reshape(len(x), len(data['panel'].tickers), -1)
                design = np.concatenate([design, np.ones((*design.shape[:2], 1))], axis=-1)
                path = folder/'ridge_coef.npy'
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                score = design @ np.load(path)
            else:
                score = data['momentum' if kind == 'momentum126' else 'reversal'][ids]
                available = np.isfinite(score)
                eligible &= available
                observed &= available
            for j, i in enumerate(ids):
                mask = observed[j]
                assert not (mask & ~eligible[j]).any()
                records.append(dict(model=kind, year=year,
                    signal_date=str(data['panel'].dates[data['origin'][i]].date()),
                    entry_date=str(data['panel'].dates[data['entry'][i]].date()),
                    exit_date=str(data['panel'].dates[data['exit'][i]].date()),
                    eligible=int(eligible[j].sum()), observed=int(mask.sum()),
                    unobserved=int((eligible[j] & ~mask).sum()),
                    **diagnostics(score[j, mask], data['y'][i, mask])))
    dated = pd.DataFrame(records)
    for kind, sub in dated.groupby('model'):
        for field in FIELDS:
            np.testing.assert_allclose(sub[field].mean(), frozen.loc[kind, field], atol=1e-10)
        for field in ['IC', 'RankIC']:
            np.testing.assert_allclose(sub[field].mean()/sub[field].std(), frozen.loc[kind, field+'IR'], atol=1e-10)
            assert sub[field].count() == frozen.loc[kind, field+'_periods']
        assert sub.unobserved.sum() == frozen.loc[kind, 'unobserved_labels']
    OUT.mkdir(exist_ok=True)
    dated.to_csv(OUT/'ranking_periods.csv', index=False)
    dated.groupby(['model', 'year'])[FIELDS + ['Random NDCG@10']].mean().to_csv(OUT/'ranking_by_year.csv')
    report = dict(status='passed', models=len(frozen), full_forecast_periods=int(dated.IC.count()/len(frozen)),
                  fields_checked=FIELDS+['ICIR', 'RankICIR', 'IC_periods', 'RankIC_periods', 'unobserved_labels'],
                  method='Independent pairwise ranks, NumPy correlation, tie-averaged discount weights; no production ranking metric call',
                  random_baseline='Exact expected NDCG for a uniformly random ordering of the same observed stocks',
                  prediction_and_ridge_hashes=hashes,
                  source_ranking_sha256=hashlib.sha256((RUN/'ranking.csv').read_bytes()).hexdigest())
    (OUT/'RANKING_AUDIT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Independently verified ranking metrics for', len(frozen), 'models')


if __name__ == '__main__':
    main()
