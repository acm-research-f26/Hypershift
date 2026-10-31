"""Descriptive forecast/issuer-target agreement; never used to select a seed."""
import json
import numpy as np
import pandas as pd
import run_ndx_consistency_retraining as study
from ndx_drawdown_engine import ISSUERS
from ndx_rank_retraining import percentile_predictions


def main():
    plan = study.verify()
    lock = json.loads((study.RUN/'ACCOUNTS_COMPLETE.json').read_text())
    for name in ('METRICS.csv','TARGETS.npz'):
        assert study.sha(study.RUN/name) == lock['output_sha256'][name]
    data = study.parent.prior.parent.parent.load_data()
    forecast_rows = []
    for family in plan['families']:
        for year in plan['years']:
            test = study.parent.prior.parent.parent.fold(data, year)[2]
            eligible = data['mask'][test]
            for method in plan['methods']:
                values = []
                for seed in plan['seeds']:
                    if method == 'original':
                        path = study.source_folder(family, year, seed)/'loss_predictions.npz'
                    else:
                        arm, kind = method.rsplit('_', 1)
                        path = study.RUN/'fits'/arm/family/str(year)/str(seed)/f'{kind}_predictions.npz'
                    with np.load(path, allow_pickle=False) as saved:
                        values.append(percentile_predictions(saved['test'], eligible))
                values = np.asarray(values)
                disagreement = np.std(values, axis=0, ddof=0)[eligible].mean()
                fractions = []
                for prediction in values:
                    counts = eligible.sum(0); supported = counts > 0
                    means = np.divide((prediction*eligible).sum(0), counts,
                                      out=np.zeros_like(counts, dtype=float), where=supported)
                    grand = prediction[eligible].mean()
                    total = np.mean((prediction[eligible]-grand)**2)
                    between = np.sum(counts*(means-grand)**2)/eligible.sum()
                    fractions.append(float(between/total) if total > 1e-15 else 0.)
                forecast_rows.append(dict(family=family, year=year, method=method,
                    mean_cross_seed_percentile_sd=float(disagreement),
                    mean_static_stock_variance_fraction=float(np.mean(fractions))))
    pd.DataFrame(forecast_rows).to_csv(study.RUN/'FORECAST_STABILITY.csv', index=False)
    names = data['panel'].tickers
    issuers = sorted({ISSUERS.get(t, t) for t in names})
    mapping = [[j for j, t in enumerate(names) if ISSUERS.get(t,t) == issuer] for issuer in issuers]
    target_rows = []
    with np.load(study.RUN/'TARGETS.npz', allow_pickle=False) as saved:
        for mixture in plan['mixtures']:
            for method in plan['methods']:
                selected = []
                for seed in plan['seeds']:
                    key = f'{mixture}__{method}__seed_{seed}__c1__b03'
                    w = saved[key][:, :-1]
                    selected.append(np.column_stack([(w[:, js] > 0).any(1) for js in mapping]))
                selected = np.asarray(selected).transpose(1,0,2).astype(float)
                intersection = selected @ selected.transpose(0,2,1)
                counts = selected.sum(-1)
                union = counts[:,:,None]+counts[:,None,:]-intersection
                jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union>0)
                a, b = np.triu_indices(20, k=1)
                target_rows.append(dict(mixture=mixture, method=method,
                    mean_pairwise_issuer_target_jaccard=float(jaccard[:,a,b].mean()),
                    seed_pairs=len(a), decision_rows=selected.shape[0]))
    pd.DataFrame(target_rows).to_csv(study.RUN/'TARGET_STABILITY.csv', index=False)
    intervals = []
    with np.load(study.RUN/'PATHS.npz', allow_pickle=False) as saved:
        n = len(saved['dates']); rng = np.random.default_rng(20261031)
        indices = ((rng.integers(0,n,(2000,(n+19)//20,1))+np.arange(20))%n).reshape(2000,-1)[:,:n]
        rf = 1.03**(1/252)-1
        contrasts = [(m,'original') for m in plan['methods'] if m != 'original']
        contrasts += [('consistency_raw','dropout_control_raw'),('consistency_ema','dropout_control_ema'),
                      ('dropout_control_ema','dropout_control_raw')]
        for group in plan['groups']:
            for left, right in contrasts:
                a = saved[f'both__{left}__{group}__c1__b03']
                b = saved[f'both__{right}__{group}__c1__b03']
                aa, bb = a[indices], b[indices]
                differences = np.sqrt(252)*((aa.mean(1)-rf)/aa.std(1,ddof=1)-(bb.mean(1)-rf)/bb.std(1,ddof=1))
                intervals.append(dict(group=group,left=left,right=right,
                    delta=study.performance(a)['sharpe']-study.performance(b)['sharpe'],
                    lower95=float(np.quantile(differences,.025)),upper95=float(np.quantile(differences,.975))))
    pd.DataFrame(intervals).to_csv(study.RUN/'PAIRED_INTERVALS.csv', index=False)
    study.DOC.mkdir(parents=True, exist_ok=True)
    for name in ('FORECAST_STABILITY.csv','TARGET_STABILITY.csv','PAIRED_INTERVALS.csv'):
        study.shutil.copyfile(study.RUN/name, study.DOC/name)
    lines = ['# Seed agreement diagnostics', '',
        'These diagnostics describe all twenty seeds; none selects a seed or a financial strategy. '
        'Agreement can increase while Sharpe falls. The financial improvement gate remains unchanged.', '',
        '| Mixture | Method | Mean pairwise issuer-target Jaccard |', '|---|---|---:|']
    for r in target_rows:
        lines.append(f"| {r['mixture']} | {r['method']} | {r['mean_pairwise_issuer_target_jaccard']:.3f} |")
    lines += ['', 'Jaccard is the number of common selected issuers divided by the union, averaged over '
        'all 190 seed pairs and all 201 decisions. Share classes are consolidated. These are individual-seed '
        'decision targets before cohort averaging, rather than actual drifted holdings or independent market trials.', '',
        '[Forecast diagnostics](FORECAST_STABILITY.csv) give per-family/year disagreement in eligible-stock '
        'percentile ranks and the fraction of forecast variance explained by stock-specific time means. '
        'The latter is a descriptive decomposition of predictions within each test year; no future return labels '
        'are used and no decomposition is fed into training or allocation. A larger static fraction can reflect '
        'less time-varying signal, so reduced disagreement alone is not a success criterion.', '',
        '[Paired Sharpe intervals](PAIRED_INTERVALS.csv) retain all four primary candidate-versus-original '
        'contrasts and the three matched consistency/EMA contrasts, for every seven-group account. '
        'They use 2,000 common circular 20-session block draws with seed 20261031. These are conditional, '
        'pointwise 95% intervals on already fitted/selected paths, with no correction for this or earlier '
        'development searches. Neither overlapping seed ensembles nor these intervals establish unseen-market generalization.', '']
    (study.DOC/'STABILITY.md').write_text('\n'.join(lines), encoding='utf8')
    study.write(study.RUN/'STABILITY_DIAGNOSTIC.json', dict(
        source_sha256=study.sha(study.ROOT/'research/reporting/analyze_ndx_consistency_stability.py'),
        scope='Descriptive diagnostics only; no candidate selection or changed accounts',
        output_sha256={name:study.sha(study.RUN/name) for name in ('FORECAST_STABILITY.csv','TARGET_STABILITY.csv','PAIRED_INTERVALS.csv')}))
    print(pd.DataFrame(target_rows).to_string(index=False), flush=True)


if __name__ == '__main__': main()
