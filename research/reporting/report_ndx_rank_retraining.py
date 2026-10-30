"""Evaluate and publish the fixed retraining study after every fit is locked."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from run_ndx_rank_retraining import ROOT, DEFAULT, verify
from run_ndx_retention_replication import sha, write
import run_ndx_drawdown_research as parent
from ndx_rank_retraining import percentile_predictions
from ndx_retention import retention_targets, hedged_targets
from ndx_drawdown_engine import allocations, daily_allocation, risk_inputs, exposure_series, market_inputs
from ndx_matched_controls import etf_data, momentum_scores
from ndx_weak_market_short import append_qqq, simulate_signed
from audit_ndx_weak_market_short import value_replay
from research_engine import performance


def check_fits(run):
    plan = verify(run)
    lock = json.loads((run/'FITS_COMPLETE.json').read_text())
    assert lock['fits'] == plan['fits'] == 120
    for name, expected in lock['hashes'].items(): assert sha(run/name) == expected, name
    return plan


def scores_for(run, data, ids, plan):
    output = {}; validation = []; epochs = []
    seeds = plan['groups']['fifteen_seed']
    for method in plan['checkpoint_methods']:
        families = {}
        for family in plan['families']:
            predictions = []
            for seed in seeds:
                annual = []
                for year in plan['years']:
                    _, va, te = parent.fold(data, year)
                    folder = run/'fits'/family/str(year)/str(seed)
                    with np.load(folder/(method+'_predictions.npz'), allow_pickle=False) as saved:
                        assert saved['validation'].shape == data['y'][va].shape
                        assert saved['test'].shape == data['y'][te].shape
                        annual.append(saved['test'].copy())
                    record = json.loads((folder/'RECORD.json').read_text())
                    detail = next(row for row in record['diagnostics'] if row['method'] == method)
                    validation.append(dict(method=method, family=family, year=year, seed=seed,
                        **{'selection_'+key:value for key,value in detail['selection'].items()},
                        **{'audit_'+key:value for key,value in detail['audit'].items()}))
                    epochs.append(dict(method=method, family=family, year=year, seed=seed,
                                       selected_epochs=','.join(map(str, detail['epochs']))))
                predictions.append(percentile_predictions(np.concatenate(annual), data['mask'][ids]))
            families[family] = np.asarray(predictions)
        for group, members in plan['groups'].items():
            subset = [seeds.index(seed) for seed in members]
            output[method, group, 'ensemble'] = .5*families['think'][subset].mean(0)+.5*families['dynamic'][subset].mean(0)
        for j, seed in enumerate(seeds):
            output[method, f'seed_{seed}', 'single_seed'] = .5*families['think'][j]+.5*families['dynamic'][j]
    pd.DataFrame(validation).to_csv(run/'VALIDATION.csv', index=False)
    pd.DataFrame(epochs).to_csv(run/'SELECTED_EPOCHS.csv', index=False)
    # Fixed causal controls; neither is duplicated across fictitious seed groups.
    mom = momentum_scores(data, ids)
    output['momentum_12minus1', 'seed_free', 'control'] = mom
    identity = []
    for year in plan['years']:
        tr, _, te = parent.fold(data, year); observed = data['labelmask'][tr]
        rank = percentile_predictions(data['y'][tr], observed)
        count = observed.sum(0)
        # Prior of20 observations with neutral rank0.5; no test-label use.
        average = (rank.sum(0)+10)/(count+20)
        identity.append(np.broadcast_to(average, data['y'][te].shape).copy())
    output['training_mean_rank', 'seed_free', 'control'] = np.concatenate(identity)
    return output


def summarize(frame, plan):
    rows = []
    ensemble = frame[frame.kind == 'ensemble']
    groups = list(plan['groups'])
    for (method, allocation, cap, cost, borrow), block in ensemble.groupby(
            ['method','allocation','cap','cost','borrow'], sort=False):
        block = block.set_index('group'); row = dict(method=method, allocation=allocation,
                                                      cap=cap, cost=cost, borrow=borrow)
        for group in groups:
            for field in ('sharpe','annualized_return','maximum_drawdown'):
                row[group+'_'+field] = float(block.loc[group, field])
        disjoint = [row[group+'_sharpe'] for group in ('original_five','new_five','fresh_five')]
        row.update(min_disjoint_sharpe=min(disjoint), disjoint_gap=max(disjoint)-min(disjoint),
                   goal_passed=method != 'initial' and all(row[group+'_sharpe'] > 1.5 for group in groups))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['cost','borrow','min_disjoint_sharpe'], ascending=[True,True,False])


def evaluate(run):
    plan = check_fits(run)
    if (run/'accounts').exists(): raise FileExistsError('Account attempt already exists; preserve it')
    dest = run/'accounts'; dest.mkdir(); (dest/'ledgers').mkdir()
    data = parent.load_data(); ids = np.concatenate([parent.fold(data,y)[2] for y in plan['years']])
    assert len(ids) == 201 and data['panel'].dates[-1].year == 2025
    dates = data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    market = market_inputs(data); combined = append_qqq(data, etf_data(data, 'QQQ'))
    scores = scores_for(run, data, ids, plan)
    metrics = []; checks = []; annual = []; paths = {}; targets = {}
    for (method, group, kind), score in scores.items():
        eligible = data['mask'][ids].copy()
        if method == 'momentum_12minus1': eligible &= np.isfinite(score)
        assert np.all(eligible.sum(1) >= 10)
        for allocation in (['retain5'] if kind == 'single_seed' else plan['allocations']):
            if allocation == 'issuer5':
                # Match causal masks for models; momentum's unavailable histories
                # get an ineligible low score before issuer selection.
                clean = np.where(eligible, score, -1e9)
                w = allocations(data, ids, clean, 'issuer5')
                assert np.all(w[~eligible] == 0)
            else:
                w = retention_targets(score, eligible, 5 if allocation == 'retain5' else 10)
            risk = risk_inputs(data, daily_allocation(data, ids, w))
            long = exposure_series(dict(vol='basket',window=20,target=.20), market, risk)[data['origin'][ids]]
            for cap in ([.75] if kind == 'single_seed' else plan['hedge_caps']):
                desired = hedged_targets(w, long, market['qqqvol20'][data['origin'][ids]], cap)
                assert np.max(np.abs(desired).sum(1)) <= 1+1e-12
                for setting in plan['cost_regimes']:
                    cost, borrow = setting['cost'], setting['borrow']
                    key = f'{method}__{group}__{allocation}__cap{int(cap*100)}__c{cost}__b{int(borrow*100):02d}'
                    measured, path, ledger = simulate_signed(combined, ids, desired, borrow, cost, detailed=True)
                    independent = value_replay(combined, ids, desired, borrow, cost)
                    error = float(np.max(np.abs(path-independent))); assert error < 1e-12
                    for field, value in performance(path).items():
                        np.testing.assert_allclose(value, measured[field], rtol=0, atol=1e-12)
                    assert ledger.date.tolist() == dates.tolist() and ledger.nav.min() > 0
                    assert ledger['long'].iloc[-1] == 0 and ledger['short'].iloc[-1] == 0
                    ledger.to_csv(dest/'ledgers'/(key+'.csv'), index=False)
                    metrics.append(dict(id=key, method=method, group=group, kind=kind,
                        allocation=allocation, cap=cap, cost=cost, borrow=borrow, **measured))
                    checks.append(dict(id=key, max_independent_daily_error=error))
                    paths[key] = path; targets[key] = desired
                    for year in plan['years']:
                        r = path[pd.DatetimeIndex(dates).year == year]
                        annual.append(dict(id=key, year=year, calendar_return=float(np.prod(1+r)-1)))
        print(f'Accounts {method}/{group}: {len(metrics)}/{plan["accounts"]}', flush=True)
    assert len(metrics) == plan['accounts'] == 576
    frame = pd.DataFrame(metrics); frame.to_csv(run/'METRICS.csv', index=False)
    summary = summarize(frame, plan); summary.to_csv(run/'CONSISTENCY.csv', index=False)
    pd.DataFrame(annual).to_csv(run/'CALENDAR.csv', index=False)
    pd.DataFrame(checks).to_csv(run/'ACCOUNT_CHECKS.csv', index=False)
    np.savez_compressed(dest/'PATHS.npz', dates=dates, **paths)
    np.savez_compressed(dest/'TARGETS.npz', ids=ids,
        security_ids=np.asarray(combined['panel'].tickers,dtype=str), **targets)
    base = summary[(summary.cost == 1)&(summary.borrow == .03)]
    write(run/'AUDIT.json', dict(status='passed', fits=120, accounts=576,
        neural_ensemble_accounts=int((frame.kind == 'ensemble').sum()),
        individual_seed_accounts=int((frame.kind == 'single_seed').sum()),
        deterministic_control_accounts=int((frame.kind == 'control').sum()),
        passing_trained_recipes=int(base.goal_passed.sum()),
        max_independent_daily_error=max(r['max_independent_daily_error'] for r in checks),
        selection='First-half purged validation only; second half diagnostic, no test/seed selection',
        limits='Bundled initialization/objective adaptation; same spent test history; no confirmation of future alpha'))
    verify(run)
    write(run/'COMPLETE.json', dict(status='complete', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        output_sha256={p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob('*')) if p.is_file()}))
    print(base[['method','allocation','cap','original_five_sharpe','new_five_sharpe',
        'fresh_five_sharpe','ten_seed_sharpe','fifteen_seed_sharpe','goal_passed']].to_string(index=False), flush=True)


def report(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = json.loads((run/'COMPLETE.json').read_text())
    for name, expected in complete['output_sha256'].items(): assert sha(run/name) == expected
    plan = json.loads((run/'PROTOCOL.json').read_text())['config']
    doc = ROOT/'docs'/run.name; doc.mkdir(parents=True, exist_ok=True)
    for path in run.glob('*.csv'): shutil.copyfile(path, doc/path.name)
    for name in ('PROTOCOL.json','FREEZE.json','AUDIT.json','ENVIRONMENT.json'):
        shutil.copyfile(run/name, doc/name)
    summary = pd.read_csv(run/'CONSISTENCY.csv')
    base = summary[(summary.cost == 1)&(summary.borrow == .03)]
    trained = base[base.method != 'initial'].sort_values('min_disjoint_sharpe', ascending=False)
    best = trained.iloc[0]; audit = json.loads((run/'AUDIT.json').read_text())
    metrics = pd.read_csv(run/'METRICS.csv')
    selected = metrics[(metrics.method == best.method)&(metrics.allocation == best.allocation)&
        (metrics.cap == best.cap)&(metrics.cost == 1)&(metrics.borrow == .03)&(metrics.kind == 'ensemble')].set_index('group')
    groups = list(plan['groups']); fig, axes = plt.subplots(1,2,figsize=(14,5.5))
    axes[0].bar(np.arange(5), [selected.loc[g,'sharpe'] for g in groups],color=['#2857b8']*3+['#159777']*2)
    axes[0].axhline(1.5,color='#b45309',linestyle='--',label='Declared SR >1.5 goal')
    axes[0].set_xticks(np.arange(5),[g.replace('_',' ') for g in groups],rotation=15)
    axes[0].set_title(f'Highest weaker-group trained recipe\n{best.method}/{best.allocation}/cap{int(best.cap*100)}')
    axes[0].set_ylabel('Net Sharpe'); axes[0].legend(fontsize=8); axes[0].grid(axis='y',alpha=.2)
    for method,color in [('initial','#7c8796'),('loss','#df851b'),('rank','#2857b8'),('snapshots','#159777')]:
        b=base[base.method==method]
        axes[1].scatter(b.original_five_sharpe,b.new_five_sharpe,label=method,color=color,s=55)
    axes[1].axhline(1.5,color='#b45309',linestyle=':'); axes[1].axvline(1.5,color='#b45309',linestyle=':')
    axes[1].set(xlabel='Original-five SR',ylabel='New-five SR',title='All fixed neural recipes (base costs)')
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.suptitle('Neutral stock-ID initialization + ranking-aligned retraining\n2022–2025 spent development history; all15predefinedseeds retained')
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(doc/'retraining_consistency.png',dpi=160);plt.close(fig)
    lines=['# Stock-ID bias attribution and ranking retraining — October29,2026','',
        f'Completed **120 fixed fits,12epochs each,15predefinedseeds,576 financial accounts**. '
        f'**{audit["passing_trained_recipes"]} trained recipes achieved SR>1.5 in every one of the five declared ensembles** at base costs. '
        'The history is repeatedly inspected development data; these findings are not untouched-period confirmation.','',
        '![Retraining consistency](retraining_consistency.png)','','## What explained the historical discrepancy','',
        'Independent asset attribution reproduces all three old retention paths to6.67e-16 daily-return error. '
        'In2023, original-five THINK mean stock forecasts had about0.995 rank correlation with the saved final stock-specific mixer biases. '
        'Those biases retain about0.999 rank correlation with their reconstructed initial values across all ten seeds. '
        'This is strong descriptive evidence of persistent random stock-ID preferences, not proof that every part of the network is untrained. '
        'The original account also carried bothGOOG andGOOGL: together they averaged about32.6% of account NAV at the beginning of trading days in2023. '
        'Concentration and favorable realized stocks contribute to the headline; buying those retrospectively identified winners would introduce lookahead. '
        '[Asset attribution and initialization checks](../ndx_seed_attribution_20261029/ASSET_CONTRIBUTIONS.csv) accompany the separate diagnostic study.','',
        '## Fixed base-cost comparisons','',
        '| Method | Allocation | Hedge cap | Original5 SR | New5 SR | Fresh5 SR | Ten SR | Fifteen SR | Goal |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for row in base.sort_values('min_disjoint_sharpe',ascending=False).itertuples():
        lines.append(f'| {row.method} | {row.allocation} | {row.cap:.0%} | {row.original_five_sharpe:.3f} | {row.new_five_sharpe:.3f} | {row.fresh_five_sharpe:.3f} | {row.ten_seed_sharpe:.3f} | {row.fifteen_seed_sharpe:.3f} | {row.goal_passed} |')
    lines += ['','## Design and limits','',
        'The two original model families keep their train-only25-channel inputs and parameter inventories. '
        'The THINK stock-specific mixer output matrix/bias start at zero; dynamic stock filters start at one. '
        'All parameters remain trainable. Both families optimize a date-balanced, centered/normalized Huber plus pairwise-ranking objective. '
        'This bundles objective and initialization changes, so it is not an isolated causal estimate of either intervention. '
        'The first half of prior-year validation selects epochs, with crossing labels purged; the second half is diagnostic and never controls selection. '
        'Loss, robustRankIC and three-snapshot checkpoint rules share the same fitted paths. Every epoch state and validation forecast is preserved. '
        'Epoch0 is an untrained neutral-initialization control, not a qualifying learned strategy.','',
        'All three disjoint five-seed groups were fixed before fitting. The additional five seeds are the next five from the October20predeclared list. '
        'Ten/fifteen ensembles overlap those groups and all use the same market history. All15single-seed pairs are reported for the fixed retention-five/cap75 policy. '
        'Deterministic momentum and training-mean-rank controls appear once each; they are not repeated as artificial seed ensembles. '
        'Allocations test five-security retention, ten-security retention and five issuers; fees, borrow, timing and risk controls remain inherited.','',
        'Cherry-picking seeds by realized Sharpe would bias paper claims. Report the best case alongside the full distribution and search history; '
        'even passing this historical target would not establish future SR>1.5. More seeds do not create independent market periods. '
        '[Bailey and Lopez de Prado,Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) describes selection-bias concerns.','',
        '## Verification and reproduction','',
        f'All576accounts match independent value-position accounting (maximum daily discrepancy {audit["max_independent_daily_error"]:.3g}); '
        'all target budgets, masks, chronology and terminal liquidation are checked. Code/data hashes are frozen, and every fit is locked before account scoring. '
        'These are saved-export/accounting checks; historical vendor vintage and live broker execution remain assumptions.','',
        '[Every account](METRICS.csv) · [Every ensemble and cost regime](CONSISTENCY.csv) · [Selected epochs](SELECTED_EPOCHS.csv) · '
        '[Selection/audit metrics](VALIDATION.csv) · [Protocol](PROTOCOL.json) · [Audit](AUDIT.json)','','```powershell',
        f'.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_rank_retraining --stage freeze --name {run.name}_fresh',
        f'.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_rank_retraining --stage train --name {run.name}_fresh',
        f'.\\.venv\\Scripts\\python.exe -m hypershift run report_ndx_rank_retraining --stage evaluate --name {run.name}_fresh',
        f'.\\.venv\\Scripts\\python.exe -m hypershift run report_ndx_rank_retraining --stage report --name {run.name}_fresh',
        '```','',f'Full epoch/checkpoint/prediction/account artifacts are retained in `runs/{run.name}/`. No2026market returns,live trades or prospective activation were used.','']
    (doc/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print('Published '+str(doc.relative_to(ROOT)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['evaluate','report'],required=True)
    parser.add_argument('--name',default=DEFAULT);args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.name):parser.error('Invalid run name')
    run=ROOT/'runs'/args.name
    evaluate(run) if args.stage=='evaluate' else report(run)


if __name__=='__main__':main()
