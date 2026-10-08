"""Read-only evaluation of a predeclared batch. Never retrains or retunes models."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from research_engine import load_panel, prepare_panel, fold_indices, performance
from robust_engine import backtest
from run_architecture_research import robust_score, write_json
from report_architecture_research import LABELS as ORIGINAL_LABELS, table

LABELS = dict(ORIGINAL_LABELS, market_proxy='SPY',
              think_huber='THINK / Huber', think_mix_huber='THINK+mixer / Huber',
              stockmixer_huber='StockMixer / Huber', think_mix_shrink='THINK+mixer x0.1 / MSE',
              think_mix_shrink_huber='THINK+mixer x0.1 / Huber',
              multiscale_mse='Multi-scale mixer / MSE', multiscale_huber='Multi-scale mixer / Huber')
PAIRS = [('think', 'think_huber'), ('think_mix', 'think_mix_huber'),
         ('stockmixer', 'stockmixer_huber'), ('think_mix_shrink', 'think_mix_shrink_huber'),
         ('multiscale_mse', 'multiscale_huber'), ('think_mix', 'think_mix_shrink'),
         ('think_mix_huber', 'think_mix_shrink_huber'), ('stockmixer', 'multiscale_mse')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stratified_interval(a, b, repeats=1000, block=20):
    """Paired circular blocks, sampled separately inside each annual fold."""
    rng = np.random.default_rng(20261005)
    samples = []
    for _ in range(repeats):
        left, right = [], []
        for x, y in zip(a, b):
            assert len(x) == len(y)
            n = len(x)
            starts = rng.integers(0, n, int(np.ceil(n/block)))
            ids = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
            left.extend(x[ids]); right.extend(y[ids])
        samples.append(performance(left)['sharpe']-performance(right)['sharpe'])
    return dict(low=float(np.quantile(samples, .025)), high=float(np.quantile(samples, .975)),
                bootstrap_fraction_positive=float(np.mean(np.asarray(samples)>0)))


def draw_heatmap(ax, values, row_labels, columns, title):
    extent = max(np.nanmax(np.abs(values)), .01)
    ax.imshow(values, aspect='auto', cmap='RdBu', norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent))
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=9)
    ax.set_title(title, fontsize=14, pad=15)
    for i, j in np.ndindex(values.shape):
        v = values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(v)>extent*.55 else '#222222')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', default=[Path('runs/robust_2022_2025'), Path('runs/daily_2022_2025')])
    parser.add_argument('--out', type=Path, default=Path('runs/robust_comparison'))
    args = parser.parse_args()
    out = args.out
    out.mkdir(exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    # Combine choices using ONLY recorded validation scores, before loading metrics.
    combined, run_info = {}, []
    for root in args.runs:
        protocol = json.loads((root/'PROTOCOL.json').read_text())
        assert (root/'completion.json').is_file(), f'Incomplete run: {root}'
        for source, expected in protocol['code_hashes'].items():
            assert digest(source) == expected, source
            assert digest(root/'source_snapshot'/Path(source).name) == expected
        assert digest('config/robust_experiment_plan.json') == protocol['experiment_plan_sha256']
        for source, expected in json.loads((root/'DATA_INPUT_LOCK.json').read_text()).items():
            assert digest(source) == expected
        policies = json.loads((root/'validation_search.json').read_text())
        lock = json.loads((root/'SELECTION_LOCK.json').read_text())
        for row in policies:
            assert row['selected'] == max(row['candidates'], key=lambda r:r['score'])
        for dataset in lock:
            for year, selection in lock[dataset]['by_year'].items():
                expected = {kind:robust_score([r['selected']['score'] for r in policies
                            if r['dataset']==dataset and r['model']==kind and r['year']<=int(year)])
                            for kind in selection['validation_scores']}
                assert expected == selection['validation_scores']
                assert selection['winner'] == max(expected, key=expected.get)
                key = f'{dataset}_{year}'
                for kind, score in expected.items():
                    combined.setdefault(key, []).append(dict(run=str(root), horizon=protocol['horizon'], model=kind, score=score))
        run_info.append((root, protocol, lock))
    meta = {key:max(candidates, key=lambda r:r['score']) for key,candidates in combined.items()}
    write_json(out/'COMBINED_SELECTION_LOCK.json', dict(selected=meta, candidates=combined,
               rule='Maximum prior/current-year validation aggregate, no test scores used. Historical retrospective rule, not pristine forward evidence.'))
    manifest = json.loads(Path('data/recent_cohorts/manifest.json').read_text())
    for source, expected in manifest['hashes'].items():
        assert digest(Path('data/recent_cohorts')/source) == expected, source
    frames, annual, audit, bootstrap, pooled, ablations, policy_effects = [], [], [], [], [], [], []
    paths, primaries, random_rows, seed_rows, rank_frames = {}, {}, [], [], {}
    for root, protocol, lock in run_info:
        horizon = protocol['horizon']
        frame = pd.read_csv(root/'all_metrics.csv'); frame['horizon'] = horizon
        frames.append(frame)
        primary = frame[(frame.seed=='ensemble') & frame.selected_policy & (frame.cost_multiplier==1)].copy()
        assert not primary.duplicated(['dataset','year','model']).any()
        primaries[horizon] = primary
        for dataset in primary.dataset.unique():
            data = prepare_panel(load_panel(dataset), horizon=horizon)
            years = sorted(primary[primary.dataset==dataset].year.unique())
            for year in years:
                folder = root/f'{dataset}_{year}'
                train, val, test = fold_indices(data, int(year))
                recorded = np.load(folder/'preprocessing.npz')
                for field, ids in [('train',train), ('validation',val), ('test',test)]:
                    np.testing.assert_array_equal(recorded[field], ids)
                assert data['exit'][train[-1]] < data['origin'][val[0]]
                assert data['exit'][val[-1]] < data['origin'][test[0]]
                spy = pd.read_csv(folder/'market_proxy_daily.csv')
                subset = primary[(primary.dataset==dataset)&(primary.year==year)]
                for _, row in subset.iterrows():
                    path = pd.read_csv(folder/f'{row.model}_daily.csv')
                    assert path.date.equals(spy.date)
                    assert len(path) == len(test)*horizon
                    assert pd.to_datetime(path.date).is_monotonic_increasing and path.date.is_unique
                    for prefix in ['gross','net']:
                        calculated = performance(path[prefix])
                        for field in ['sharpe','annualized_return','cumulative_return','annualized_volatility','maximum_drawdown']:
                            np.testing.assert_allclose(row[f'{prefix}_{field}'], calculated[field], atol=1e-9)
                    key = (horizon,dataset,int(year),row.model)
                    paths[key] = path
                    spy_sr = subset[subset.model=='market_proxy'].net_sharpe.iloc[0]
                    active = path.net.to_numpy()-spy.net.to_numpy()
                    tracking = active.std(ddof=1)*np.sqrt(252)
                    beta, intercept = np.linalg.lstsq(np.c_[spy.net-((1.03)**(1/252)-1), np.ones(len(spy))],
                                                     path.net-((1.03)**(1/252)-1),rcond=None)[0]
                    annual.append(dict(row, label=LABELS[row.model], spy_sharpe=spy_sr,
                                       delta_spy=row.net_sharpe-spy_sr, beats_spy=bool(row.net_sharpe>spy_sr),
                                       beta_to_spy=beta, alpha_annualized=intercept*252,
                                       information_ratio=active.mean()*252/tracking if tracking>1e-12 else None,
                                       selected_architecture=lock[dataset]['by_year'][str(year)]['winner']==row.model,
                                       eligible_min=int(data['mask'][test].sum(1).min()),
                                       eligible_max=int(data['mask'][test].sum(1).max()),
                                       start=path.date.iloc[0], end=path.date.iloc[-1]))
                    audit.append(dict(horizon=horizon,dataset=dataset,year=year,model=row.model,status='verified'))
                    ranking_path = folder/f'{row.model}_ranking.csv'
                    if ranking_path.exists():
                        rank_frames[key] = pd.read_csv(ranking_path)
                # Fixed random controls are diagnostics, not selectable strategies.
                for seed in range(50):
                    scores = np.random.default_rng(89000+int(year)*100+seed).standard_normal(data['y'][test].shape)
                    metric, _, _ = backtest(data,test,scores,10,'equal')
                    random_rows.append(dict(horizon=horizon,dataset=dataset,year=year,seed=seed,k=10,**metric))
                # Fixed-policy paired ablations do not let portfolio re-selection explain the difference.
                for before, after in PAIRS:
                    if after not in set(subset.model) or before not in set(subset.model): continue
                    a = frame[(frame.dataset==dataset)&(frame.year==year)&(frame.model==before)&
                              (frame.seed=='ensemble')&(frame.cost_multiplier==1)&(frame.k==10)&(frame.method=='equal')].iloc[0]
                    b = frame[(frame.dataset==dataset)&(frame.year==year)&(frame.model==after)&
                              (frame.seed=='ensemble')&(frame.cost_multiplier==1)&(frame.k==10)&(frame.method=='equal')].iloc[0]
                    ablations.append(dict(horizon=horizon,dataset=dataset,year=year,before=before,after=after,
                                           delta_sharpe=b.net_sharpe-a.net_sharpe,delta_IC=b.IC-a.IC,
                                           delta_RankIC=b.RankIC-a.RankIC,delta_NDCG10=b['NDCG@10']-a['NDCG@10']))
                all_equal = frame[(frame.dataset==dataset)&(frame.year==year)&(frame.seed=='ensemble')&
                                  (frame.cost_multiplier==1)&(frame.k==10)]
                for kind in all_equal.model.unique():
                    group = all_equal[all_equal.model==kind].set_index('method')
                    for method in ['inverse_vol','buffer_equal','buffer_inverse_vol']:
                        a,b = group.loc['equal'],group.loc[method]
                        policy_effects.append(dict(horizon=horizon,dataset=dataset,year=year,model=kind,method=method,
                                                   delta_sharpe=b.net_sharpe-a.net_sharpe,
                                                   delta_turnover=b.turnover_annual-a.turnover_annual))
                print(f'Audited horizon={horizon} {dataset} {year}; 50 random K10 controls', flush=True)
            kinds = primary[primary.dataset==dataset].model.unique()
            for kind in kinds:
                annual_paths = [paths[(horizon,dataset,int(year),kind)] for year in years]
                spy_paths = [paths[(horizon,dataset,int(year),'market_proxy')] for year in years]
                net = np.concatenate([p.net for p in annual_paths])
                gross = np.concatenate([p.gross for p in annual_paths])
                spy_net = np.concatenate([p.net for p in spy_paths])
                rows = primary[(primary.dataset==dataset)&(primary.model==kind)]
                combined_rank = [rank_frames[(horizon,dataset,int(year),kind)] for year in years
                                 if (horizon,dataset,int(year),kind) in rank_frames]
                ranking = pd.concat(combined_rank,ignore_index=True) if combined_rank else None
                rank_summary = {} if ranking is None else {key:float(ranking[key].mean()) for key in ['IC','RankIC','NDCG@5','NDCG@10','NDCG@20']}
                if ranking is not None:
                    for metric in ['IC','RankIC']:
                        rank_summary[metric+'IR'] = ranking[metric].mean()/ranking[metric].std()
                metrics = {f'net_{key}':val for key,val in performance(net).items()}
                metrics.update({f'gross_{key}':val for key,val in performance(gross).items()})
                spy_metrics = performance(spy_net)
                deltas = rows.set_index('year').net_sharpe-primary[(primary.dataset==dataset)&(primary.model=='market_proxy')].set_index('year').net_sharpe
                interval = stratified_interval([p.net.to_numpy() for p in annual_paths], [p.net.to_numpy() for p in spy_paths])
                bootstrap.append(dict(horizon=horizon,dataset=dataset,model=kind,**interval))
                pooled.append(dict(horizon=horizon,dataset=dataset,model=kind,label=LABELS[kind],**metrics,**rank_summary,
                                   spy_net_sharpe=spy_metrics['sharpe'],delta_spy=metrics['net_sharpe']-spy_metrics['sharpe'],
                                   years_beating_spy=int((deltas>0).sum()),median_yearly_sharpe=rows.net_sharpe.median(),
                                   worst_yearly_sharpe=rows.net_sharpe.min(),std_yearly_sharpe=rows.net_sharpe.std(),
                                   turnover_annual=np.average(rows.turnover_annual,weights=rows.net_days),
                                   profitable_periods=np.average(rows.profitable_periods,weights=rows.net_days),**interval))
        for (dataset,year,kind), group in frame[(frame.seed!='ensemble')&(frame.cost_multiplier==1)].groupby(['dataset','year','model']):
            seed_rows.append(dict(horizon=horizon,dataset=dataset,year=year,model=kind,
                                  seed_sharpe_mean=group.net_sharpe.mean(),seed_sharpe_std=group.net_sharpe.std(),
                                  seed_sharpe_min=group.net_sharpe.min(),seed_sharpe_max=group.net_sharpe.max()))
    full = pd.concat(frames,ignore_index=True)
    annual = pd.DataFrame(annual); pooled = pd.DataFrame(pooled)
    ablations = pd.DataFrame(ablations); policy_effects = pd.DataFrame(policy_effects)
    random = pd.DataFrame(random_rows)
    for filename, frame in [('all_metrics',full),('annual_comparison',annual),('pooled_comparison',pooled),
                            ('fixed_k10_ablations',ablations),('fixed_k10_portfolio_effects',policy_effects),
                            ('random_top10_controls',random),('seed_stability',pd.DataFrame(seed_rows)),
                            ('accounting_audit',pd.DataFrame(audit))]:
        frame.to_csv(out/f'{filename}.csv',index=False)
    write_json(out/'paired_bootstrap.json',bootstrap)
    selected = annual[annual.selected_architecture].copy()
    selected.to_csv(out/'validation_selected_by_year.csv',index=False)
    meta_rows = []
    for key, choice in meta.items():
        dataset, year = key.rsplit('_',1)
        row = annual[(annual.horizon==choice['horizon'])&(annual.dataset==dataset)&
                     (annual.year==int(year))&(annual.model==choice['model'])].iloc[0]
        meta_rows.append(dict(row))
    pd.DataFrame(meta_rows).to_csv(out/'combined_validation_selected.csv',index=False)
    # Horizon effects: same model/policy on each horizon's own matched-SPY calendar.
    fixed = full[(full.seed=='ensemble')&(full.cost_multiplier==1)&(full.method=='equal')&(full.k==10)]
    daily = fixed[fixed.horizon==1]; weekly = fixed[fixed.horizon==5]
    horizon_pairs = daily.merge(weekly,on=['dataset','year','model'],suffixes=('_daily','_five'))
    horizon_pairs['delta_sharpe_daily_minus_five'] = horizon_pairs.net_sharpe_daily-horizon_pairs.net_sharpe_five
    horizon_pairs['delta_turnover_daily_minus_five'] = horizon_pairs.turnover_annual_daily-horizon_pairs.turnover_annual_five
    horizon_pairs.to_csv(out/'horizon_comparison.csv',index=False)
    # Exact shared-calendar diagnostic; existing accounts kept intact, no re-selection.
    matched = []
    for _, row in horizon_pairs.iterrows():
        a = paths[(1,row.dataset,int(row.year),row.model)]
        b = paths[(5,row.dataset,int(row.year),row.model)]
        dates = sorted(set(a.date)&set(b.date))
        # Read the fixed-policy accounts rather than the selected-policy paths above.
        roots = {p['horizon']:r for r,p,_ in run_info}
        a = pd.read_csv(roots[1]/f'{row.dataset}_{row.year}'/f'{row.model}_equal_k10_daily.csv').set_index('date')
        b = pd.read_csv(roots[5]/f'{row.dataset}_{row.year}'/f'{row.model}_equal_k10_daily.csv').set_index('date')
        sa,sb = performance(a.loc[dates,'net'])['sharpe'],performance(b.loc[dates,'net'])['sharpe']
        matched.append(dict(dataset=row.dataset,year=row.year,model=row.model,shared_days=len(dates),
                            daily_sharpe=sa,five_sharpe=sb,delta_sharpe=sa-sb))
    pd.DataFrame(matched).to_csv(out/'horizon_common_calendar.csv',index=False)
    render(out,run_info,annual,pooled,full,selected,ablations,policy_effects,horizon_pairs,paths,random,pd.DataFrame(meta_rows))
    snapshots = out/'source_snapshot'; snapshots.mkdir(exist_ok=True)
    source_names = ['report_robust_research.py','test_robust_research.py','config/robust_experiment_plan.json','STOCKMIXER_PAPER_REVIEW.md']
    for name in source_names: (snapshots/Path(name).name).write_bytes(Path(name).read_bytes())
    write_json(out/'report_sources.json',{name:digest(name) for name in source_names})
    # Hash all inputs used for the comparison, including frozen predictions/policies.
    hashes = {}
    for root,_,_ in run_info:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ['.csv','.json','.npz']:
                hashes[str(path)] = digest(path)
    write_json(out/'comparison_input_hashes.json',hashes)
    write_json(out/'completion.json',dict(status='complete',audited_primary_paths=len(audit),
               random_controls=len(random),run_directories=[str(r) for r in args.runs]))
    print(f'COMPLETE: {len(audit)} primary paths verified; report in {out}',flush=True)


def render(out,run_info,annual,pooled,full,selected,ablations,policy_effects,horizon_pairs,paths,random,meta_rows):
    """Descriptive tables retain all tested kinds; never choose a deployment from tests."""
    charts = ['# Daily versus five-session experiments: complete graph collection','',
              'All curves include modeled costs. Every panel uses matched-date SPY. Policies were selected using prior validation data. '
              '2024-2025 have been inspected before; all results are exploratory and public-data survivorship bias remains.','',
              'Annual accounts start from cash. Pooled curves compound annual-fold returns and omit untested boundary sessions; '
              'they are not a continuous live-account backtest. Daily and five-session pipelines also use different training batch sizes.','',
              '[Full report](REPORT.md) | [Earlier 2017/2024/2025 charts](../recent_2024_2025/CHARTS.md)','']
    palette = list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('Dark2').colors)
    colors = {kind:palette[i] for i,kind in enumerate(LABELS)}
    colors.update(market_proxy='#111111',buy_hold='#777777',equal_universe='#217a45')
    inventory = []
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    for root,protocol,_ in run_info:
        h = protocol['horizon']; primary = annual[annual.horizon==h]
        kinds = protocol['arguments']['variants']+['ridge','momentum126','reversal5','buy_hold','equal_universe','market_proxy']
        columns = [(d,int(y)) for d in primary.dataset.unique() for y in sorted(primary.year.unique())]
        values = primary.pivot(index='model',columns=['dataset','year'],values='net_sharpe').reindex(index=kinds,columns=columns).to_numpy()
        fig, ax = plt.subplots(figsize=(16, max(8,len(kinds)*.42)))
        draw_heatmap(ax,values,[LABELS[k] for k in kinds],[f'{d.replace("_recent", "")}\n{y}' for d,y in columns],f'{h}-session prediction and rebalance: net Sharpe')
        fig.tight_layout(); filename=f'sharpe_h{h}.png'; fig.savefig(out/'figures'/filename,dpi=150); plt.close(fig)
        charts.extend([f'## {h}-session results: all models','',f'![All model Sharpes, horizon {h}](figures/{filename})',''])
        for dataset in primary.dataset.unique():
            years=sorted(primary.year.unique())
            fig,axes=plt.subplots(1,len(years),figsize=(22,8),sharey=True)
            for ax,year in zip(axes,years):
                for kind in kinds:
                    p=paths[(h,dataset,int(year),kind)]
                    ax.plot(pd.to_datetime(p.date),np.cumprod(1+p.net),label=LABELS[kind],color=colors[kind],
                            linestyle='--' if 'huber' in kind else '-',linewidth=2.5 if kind=='market_proxy' else 1.1)
                    inventory.append(dict(horizon=h,dataset=dataset,year=int(year),model=kind,source=str(root/f'{dataset}_{year}'/f'{kind}_daily.csv')))
                assert len(ax.lines)==len(kinds)
                ax.set_title(str(year));ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=35,labelsize=8)
            axes[0].set_ylabel('Net wealth per $1 initial capital')
            fig.suptitle(f'{dataset.replace("_recent", " cohort")} | {h}-session forecasts | all {len(kinds)} tested kinds',fontsize=18,y=.98)
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=9,bbox_to_anchor=(.5,.035),frameon=False)
            fig.subplots_adjust(left=.05,right=.99,top=.89,bottom=.30,wspace=.10)
            fig.text(.02,.015,'Separate annual accounts; validation-selected K/weights. Fixed historical cohort with incomplete public histories. Black: matched SPY. Dashed: Huber.',fontsize=9)
            filename=f'all_models_h{h}_{dataset}.png';fig.savefig(out/'figures'/filename,dpi=160);plt.close(fig)
            charts.extend([f'![All {len(kinds)} model curves: {dataset}, horizon {h}](figures/{filename})',''])
            # Individual panels retain all lines and a readable external legend.
            for year in years:
                fig,ax=plt.subplots(figsize=(15,7))
                for kind in kinds:
                    p=paths[(h,dataset,int(year),kind)]
                    ax.plot(pd.to_datetime(p.date),np.cumprod(1+p.net),label=LABELS[kind],color=colors[kind],
                            linestyle='--' if 'huber' in kind else '-',linewidth=2.5 if kind=='market_proxy' else 1.2)
                ax.set_title(f'{dataset.replace("_recent", "")} {year} | {h}-session | all {len(kinds)} kinds')
                ax.set_ylabel('Net wealth per $1');ax.grid(alpha=.2)
                ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=9,frameon=False)
                fig.text(.02,.02,'Validation-selected portfolios; modeled costs; black SPY; public-data coverage bias remains.',fontsize=9)
                fig.tight_layout(rect=(0,.05,1,1));filename=f'panel_h{h}_{dataset}_{year}.png'
                fig.savefig(out/'figures'/filename,dpi=150);plt.close(fig)
                charts.extend([f'![Full size: {dataset} {year}, horizon {h}](figures/{filename})',''])
        # Fixed equal-K10 loss ablations, showing the entire four-year sequence.
        pairs=[pair for pair in PAIRS[:5] if pair[1] in kinds]
        fig,axes=plt.subplots(len(pairs),2,figsize=(16,3.7*len(pairs)),squeeze=False)
        for i,(before,after) in enumerate(pairs):
            for j,dataset in enumerate(primary.dataset.unique()):
                ax=axes[i,j]
                for kind,color,style in [(before,'#246ba0','-'),(after,'#dc6b25','-'),('market_proxy','#111111','--')]:
                    segments=[]
                    for year in sorted(primary.year.unique()):
                        suffix=f'{kind}_daily.csv' if kind=='market_proxy' else f'{kind}_equal_k10_daily.csv'
                        segments.append(pd.read_csv(root/f'{dataset}_{year}'/suffix))
                    p=pd.concat(segments,ignore_index=True)
                    ax.plot(pd.to_datetime(p.date),np.cumprod(1+p.net),color=color,linestyle=style,label=LABELS[kind])
                ax.set_title(f'{dataset.replace("_recent", "")} | {LABELS[before]} vs Huber',fontsize=11)
                ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_ylabel('Net fold-compounded wealth')
        fig.suptitle(f'{h}-session: MSE versus Huber at fixed equal-weight top 10',fontsize=17)
        fig.text(.02,.01,'Annual cash resets and boundary gaps retained. These are descriptive pooled folds, not a continuously held portfolio.',fontsize=9)
        fig.tight_layout(rect=(0,.03,1,.97));filename=f'huber_fixed_k10_h{h}.png';fig.savefig(out/'figures'/filename,dpi=145);plt.close(fig)
        charts.extend([f'## {h}-session: paired Huber comparisons','',f'![Fixed top-10 Huber comparisons](figures/{filename})',''])
    # Selected procedures, not ex-post best-model curves.
    fig,axes=plt.subplots(2,2,figsize=(16,11),squeeze=False)
    for i,h in enumerate([5,1]):
        for j,dataset in enumerate(annual.dataset.unique()):
            ax=axes[i,j]
            choices=selected[(selected.horizon==h)&(selected.dataset==dataset)].sort_values('year')
            for label,kind,color in [('Validation-selected procedure',None,'#256ea0'),('SPY','market_proxy','#111111'),('Equal-weight buy-and-hold','buy_hold','#777777')]:
                p=pd.concat([paths[(h,dataset,int(row.year),kind or row.model)] for _,row in choices.iterrows()],ignore_index=True)
                ax.plot(pd.to_datetime(p.date),np.cumprod(1+p.net),label=label,color=color)
            ax.set_title(f'{dataset.replace("_recent", "")} | {h}-session');ax.legend(fontsize=8);ax.grid(alpha=.2)
            ax.set_ylabel('Net fold-compounded wealth')
    fig.suptitle('Actual validation selection procedures versus SPY, 2022-2025',fontsize=17)
    fig.text(.02,.015,'Not hindsight winners. Annual accounts reset; gaps omitted. Costs included; fixed survivor cohorts.',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(out/'figures'/'selected_procedures.png',dpi=160);plt.close(fig)
    charts[8:8]=['## Validation-selected procedures','', '![Validation-selected procedures](figures/selected_procedures.png)','']
    (out/'CHARTS.md').write_text('\n'.join(charts),encoding='utf-8')
    write_json(out/'line_chart_inventory.json',inventory)
    assert len(inventory)==len(annual)
    report=['# Robust-loss and daily-session experiments, 2022-2025','',
            '[All graphs](CHARTS.md) | [Paper review](../../STOCKMIXER_PAPER_REVIEW.md) | [Frozen plan](../../config/robust_experiment_plan.json)','',
            '## What was tested','',
            'Two fixed public-data cohorts, four test years and three initialization seeds. Five-session forecasts: 16 neural variants; '
            'daily forecasts: eight variants covering THINK, THINK+mixer, StockMixer-inspired and multi-scale mixer, each paired across MSE/Huber. '
            'Ridge, momentum, reversal, equal-weight buy-and-hold, rebalanced universe and SPY are retained. '
            'Daily training includes all daily origins; its batch size is 40 versus 8 for five-session training. '
            'The frequency comparison therefore changes prediction horizon, rebalance interval and batch size.','',
            'Huber delta is fixed at one percentage point, with a factor of two matching MSE curvature near zero. '
            'Early stopping uses validation MSE in all variants. Mixer shrinkage fixes the residual multiplier at 0.1. '
            'The new multi-scale mixer uses patch scales 1/2/4, causal temporal layers and an eight-state stock-market bottleneck. '
            'These remain local adaptations, not official paper replications.','',
            'Portfolio search: K=5/10/20; equal, confidence, inverse-volatility, target-retention buffer, or buffer plus inverse-volatility. '
            'Inverse volatility uses 60 past sessions, a 10% annualized volatility floor and 2/K maximum weight. '
            'Buffer retains previous target names while inside the top 2K; targets may include unfilled orders. '
            'Long-short is diagnostic only because historical borrow availability is absent.','',
            '## Validation-selected results','',
            'Architecture and portfolio choices are fixed from available prior validation data. The table below follows that selection '
            'procedure instead of replacing a disappointing selection with a test-period winner.','',
            table(selected,[('horizon','Horizon'),('dataset','Cohort'),('year','Year'),('label','Selected model'),('method','Weights'),('k','K'),
                            ('net_sharpe','Net SR'),('spy_sharpe','SPY SR'),('delta_spy','Difference'),('net_cumulative_return','Net return')],['net_cumulative_return']),'',
            '## Selection across both frequencies','',
            'This additional rule compares the same prior validation scores across both runs; it also never uses test results to select.','',
            table(meta_rows,[('dataset','Cohort'),('year','Year'),('horizon','Horizon'),('label','Selected model'),('net_sharpe','Net SR'),('spy_sharpe','Matched SPY SR')]),'',
            '## Complete model comparison','',
            'Each pooled Sharpe is recomputed from daily returns across four separate annual folds, not an average of annual Sharpes. '
            'Annual cash resets, fixed $1m sizing and excluded boundary sessions mean the pooled series is not a continuous live account. '
            'The confidence interval is an exploratory paired 20-session block bootstrap, stratified by year, with 1,000 repeats. '
            'It is not adjusted for this search or earlier searches.','']
    for h in [5,1]:
        for dataset in annual.dataset.unique():
            sub=pooled[(pooled.horizon==h)&(pooled.dataset==dataset)]
            report.extend([f'### {dataset}: {h}-session forecasts','',
                table(sub,[('label','Model'),('gross_sharpe','Gross SR'),('net_sharpe','Net SR'),('delta_spy','SR-SPY'),
                           ('years_beating_spy','Years > SPY'),('net_annualized_return','Ann. return'),('net_cumulative_return','Cum. return'),
                           ('net_annualized_volatility','Ann. vol'),('net_maximum_drawdown','Max DD')],
                          ['net_annualized_return','net_cumulative_return','net_annualized_volatility','net_maximum_drawdown']),'',
                table(sub,[('label','Model'),('IC','IC'),('RankIC','RankIC'),('ICIR','ICIR'),('RankICIR','RankICIR'),
                           ('NDCG@5','NDCG@5'),('NDCG@10','NDCG@10'),('NDCG@20','NDCG@20'),('turnover_annual','Turnover/year'),
                           ('profitable_periods','Win rate')],['profitable_periods']),''])
    grouped=ablations.groupby(['horizon','before','after']).agg(mean_delta_sharpe=('delta_sharpe','mean'),median_delta_sharpe=('delta_sharpe','median'),
                     positive_windows=('delta_sharpe',lambda x:int((x>0).sum())),windows=('delta_sharpe','size'),
                     mean_delta_IC=('delta_IC','mean'),mean_delta_RankIC=('delta_RankIC','mean')).reset_index()
    report.extend(['## What helped in paired ablations?','',
                   'All comparisons here use fixed equal-weight K=10, independent of the validation-selected portfolio. '
                   'A positive difference favors the second variant. Cohort/year windows overlap in market exposure and are not independent trials.','',
                   table(grouped,[('horizon','Horizon'),('before','Before'),('after','After'),('median_delta_sharpe','Median SR change'),
                                  ('positive_windows','Positive windows'),('windows','Windows'),('mean_delta_IC','Mean IC change'),('mean_delta_RankIC','Mean RankIC change')]),''])
    policy=policy_effects.groupby(['horizon','method']).agg(median_delta_sharpe=('delta_sharpe','median'),median_delta_turnover=('delta_turnover','median'),
                         positive_windows=('delta_sharpe',lambda x:int((x>0).sum())),windows=('delta_sharpe','size')).reset_index()
    report.extend(['## Portfolio-rule effects at fixed K=10','',
                   'Each rule is compared with equal weights using exactly the same scores. Lower turnover need not imply better Sharpe.','',
                   table(policy,[('horizon','Horizon'),('method','Rule'),('median_delta_sharpe','Median SR change'),('median_delta_turnover','Median turnover change'),
                                 ('positive_windows','Positive cases'),('windows','Cases')]),'',
                   '## Daily versus five-session forecasts','',
                   'Same model name and equal-weight K=10; each uses its own matched-date SPY and forecast schedule. '
                   'The common-calendar CSV additionally compares only shared dates without resetting existing positions. '
                   'This is a pipeline comparison, not an isolated rebalance-only experiment.',''])
    horizon_summary=horizon_pairs.groupby('model').agg(median_delta_sharpe=('delta_sharpe_daily_minus_five','median'),
                            median_delta_turnover=('delta_turnover_daily_minus_five','median'),
                            positive_windows=('delta_sharpe_daily_minus_five',lambda x:int((x>0).sum())),windows=('year','size')).reset_index()
    report.extend([table(horizon_summary,[('model','Model'),('median_delta_sharpe','Daily minus five SR'),('median_delta_turnover','Turnover change'),
                                         ('positive_windows','Daily better windows'),('windows','Windows')]),'',
        '## Trading, data and statistical limits','',
        '- Annual train/validation/test folds are chronological and purge labels that cross boundaries. Training uses available history within the preceding four training years; 2022 has less history because the source begins in 2018 and needs a feature warmup. Exact dates are in each fold.json.',
        '- Scores use information through close t; orders execute at close t+1 plus modeled costs. Daily targets run from t+1 to t+2; five-session targets from t+1 to t+6. No same-signal-close fill is assumed.',
        '- $1m initial NAV; minimum lagged dollar ADV $1m; trades capped at 1% signal-time ADV. Costs per side: 2bp commission, 5bp spread/slippage, and 3bp times sqrt(participation/1%). Stress tests at 0x/2x/4x costs are retained. Cash earns zero; Sharpe uses a fixed 3% hurdle.',
        '- Adjusted-price returns approximate total returns and auction fills. Terminal liquidation is charged but can exceed the participation cap; stale marks and execution constraints are recorded. Missing future labels are reported rather than used to remove stocks at signal time.',
        '- The universes are fixed survivor cohorts, not all NYSE/NASDAQ stocks or point-in-time S&P 500 members. Public-history coverage and delisting gaps materially limit conclusions. SPY has a different investment universe.',
        '- 2024-2025 were inspected before this batch. Extending to 2022-2023 broadens regimes, but choosing that extension retrospectively does not make it untouched confirmation data. No additional candidates were introduced after this batch\'s outcomes.',
        '- IC/RankIC use the full eligible cross-section with observed labels. NDCG uses realized-return percentile relevance, including negative-return periods. ICIR/RankICIR are unannualized period-mean divided by period standard deviation. They should not be compared across horizons as if their sampling frequency were identical.',
        '- Seed stability and K sensitivity are diagnostics. Confidence intervals are not corrected for multiple comparisons; neither an isolated high Sharpe nor its unadjusted interval establishes a repeatable edge.',
        '', '## Reproduction and detailed artifacts','',
        'Training commands (use fresh output directories):','',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe -m unittest -v test_research test_robust_research',
        '.\\.venv\\Scripts\\python.exe run_robust_research.py --workers 3 --out runs/robust_new',
        '.\\.venv\\Scripts\\python.exe run_robust_research.py --workers 3 --horizon 1 --batch 40 --variants think think_huber think_mix think_mix_huber stockmixer stockmixer_huber multiscale_mse multiscale_huber --out runs/daily_new',
        '.\\.venv\\Scripts\\python.exe report_robust_research.py --runs runs/robust_new runs/daily_new --out runs/comparison_new',
        '```','',
        'The actual five-session run reused verified old 2024-2025 baseline fits; exact predictions, preprocessing, fold definitions, training settings and model/source hashes were checked. FIT_REUSE_AUDIT.json records them. Fresh reproduction retrains all fits.','',
        '- [All metrics, policies, K, seeds and cost stresses](all_metrics.csv)',
        '- [Annual comparison and SPY-relative metrics](annual_comparison.csv)',
        '- [Pooled comparison including intervals](pooled_comparison.csv)',
        '- [Fixed-K10 architecture/loss ablations](fixed_k10_ablations.csv)',
        '- [Fixed-K10 portfolio-rule effects](fixed_k10_portfolio_effects.csv)',
        '- [Daily/five-session comparison](horizon_comparison.csv) and [shared-date comparison](horizon_common_calendar.csv)',
        '- [Seed stability](seed_stability.csv) and [50 random top-10 controls per window](random_top10_controls.csv)',
        '- [Verified path accounting](accounting_audit.csv), [combined validation lock](COMBINED_SELECTION_LOCK.json), and [input hashes](comparison_input_hashes.json)',
        '', 'The complete model specifications, training dates, eligible universe, chosen policies, checkpoints, predictions and histories remain in the two source run directories.',''])
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')


if __name__=='__main__':
    main()
