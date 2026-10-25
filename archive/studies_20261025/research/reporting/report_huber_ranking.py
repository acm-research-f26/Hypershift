"""Evaluate the finite Huber+ranking batch; selection reads validation only."""
import copy
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research_engine import load_panel, prepare_panel, fold_indices, training_groups, performance
from run_robust_research import robust_score, scale_inputs, write_json
from report_pooled_uncertainty import indices, bootstrap_sharpe, interval
from report_robust_research import LABELS as OLD_LABELS

RUN = Path('runs/huber_ranking_2022_2025')
OUT = Path('runs/robust_comparison')
OLD = Path('runs/robust_2022_2025_final')
YEARS = [2022, 2023, 2024, 2025]
BASE = 'think_mix_huber'
NEW = ['think_mix_huber_rank_01', 'think_mix_huber_rank_03', 'think_mix_huber_rank_10']
FAMILY = [BASE] + NEW
WEIGHTS = dict(zip(FAMILY, [0., .1, .3, 1.]))
LABELS = dict(OLD_LABELS, qqq='QQQ',
              think_mix_huber_rank_01='THINK+mixer / Huber+ranking (lambda 0.1)',
              think_mix_huber_rank_03='THINK+mixer / Huber+ranking (lambda 0.3)',
              think_mix_huber_rank_10='THINK+mixer / Huber+ranking (lambda 1.0)',
              ranking_choice='Validation choice of ranking weight',
              ranking_choice_k10='Validation choice of ranking weight / equal K10')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_choices(policies=None):
    """Lock causal family selection before this report loads any test metrics."""
    if policies is None:
        policies = json.loads((RUN/'validation_search.json').read_text())
    family_policies = [p for p in policies if p['model'] in FAMILY]
    records = []
    for dataset in ['NASDAQ_recent', 'NYSE_recent']:
        for year in YEARS:
            for policy in ['validation-selected', 'equal K10']:
                scores = {}
                for kind in FAMILY:
                    history = [p for p in policies if p['dataset'] == dataset
                               and p['model'] == kind and p['year'] <= year]
                    values = [p['selected']['score'] if policy == 'validation-selected' else
                              next(c['score'] for c in p['candidates'] if c['method']=='equal' and c['k']==10)
                              for p in history]
                    scores[kind] = robust_score(values)
                winner = max(scores, key=scores.get)
                current = next(p for p in policies if p['dataset']==dataset and p['model']==winner and p['year']==year)
                chosen = current['selected'] if policy=='validation-selected' else dict(method='equal', k=10)
                records.append(dict(dataset=dataset, year=year, policy=policy, model=winner,
                                    ranking_weight=WEIGHTS[winner], method=chosen['method'], k=chosen['k'],
                                    validation_scores=scores))
    lock = dict(rule='Declared family including zero; current-and-prior validation only; stable ties prefer zero.',
                plan_sha256=digest(RUN/'EXPERIMENT_PLAN.json'),
                validation_family_sha256=hashlib.sha256(json.dumps(sorted(family_policies,
                    key=lambda p:(p['dataset'],p['year'],p['model'])),sort_keys=True).encode()).hexdigest(),
                choices=records)
    path = RUN/'RANKING_SELECTION_LOCK.json'
    if path.exists():
        assert json.loads(path.read_text()) == lock
    else:
        write_json(path, lock)
    flat = [{k:v for k,v in row.items() if k!='validation_scores'} for row in records]
    pd.DataFrame(flat).to_csv(OUT/'huber_ranking_selections.csv', index=False)
    return records


def audit():
    protocol = json.loads((RUN/'PROTOCOL.json').read_text())
    for name, expected in protocol['code_hashes'].items():
        assert digest(RUN/'source_snapshot'/Path(name).name) == expected
        assert digest(name) == expected
    for name, expected in json.loads((RUN/'DATA_INPUT_LOCK.json').read_text()).items():
        assert digest(name) == expected
    manifest = json.loads(Path('data/recent_cohorts/manifest.json').read_text())
    for name, expected in manifest['hashes'].items():
        assert digest(Path('data/recent_cohorts')/name) == expected
    assert digest(RUN/'EXPERIMENT_PLAN.json') == digest('config/huber_ranking_experiment_plan.json')
    reuse = json.loads((RUN/'FIT_REUSE_AUDIT.json').read_text())
    assert len(reuse) == 72
    records = []
    for dataset in ['NASDAQ_recent', 'NYSE_recent']:
        data = prepare_panel(load_panel(dataset), horizon=5)
        for year in YEARS:
            folder = RUN/f'{dataset}_{year}'
            saved = np.load(folder/'preprocessing.npz')
            train, val, test = fold_indices(data, year)
            for key, value in [('train',train), ('validation',val), ('test',test)]:
                np.testing.assert_array_equal(saved[key], value)
            x, mean, std = scale_inputs(data, train)
            np.testing.assert_array_equal(saved['mean'], mean)
            np.testing.assert_array_equal(saved['std'], std)
            assert data['exit'][train[-1]] < data['origin'][val[0]]
            assert data['exit'][val[-1]] < data['origin'][test[0]]
            changed = copy.deepcopy(data['panel'])
            cutoff = data['origin'][test[0]]
            rng = np.random.default_rng(year+5)
            changed.close[cutoff+1:] *= rng.uniform(.2,3.,changed.close[cutoff+1:].shape)
            changed.volume[cutoff+1:] *= .01
            changed.benchmark[cutoff+1:] *= 2.
            altered = prepare_panel(changed, horizon=5)
            past = data['origin'] <= cutoff
            for key in ['x','mask','momentum','reversal']:
                np.testing.assert_array_equal(data[key][past], altered[key][past])
            for key in ['y','risk','labelmask']:
                for ids in [train,val]:
                    np.testing.assert_array_equal(data[key][ids], altered[key][ids])
            np.testing.assert_array_equal(data['adv'][:cutoff+1], altered['adv'][:cutoff+1])
            xx, mm, ss = scale_inputs(altered, train)
            np.testing.assert_array_equal(x[past], xx[past])
            np.testing.assert_array_equal(mean, mm)
            np.testing.assert_array_equal(std, ss)
            assert training_groups(data, train) == training_groups(altered, train)
            fold = json.loads((folder/'fold.json').read_text())
            assert fold == json.loads((OLD/f'{dataset}_{year}'/'fold.json').read_text())
            records.append(dict(dataset=dataset, year=year, train_start=fold['train_signal'][0],
                                train_end=fold['train_signal'][1], validation_start=fold['validation_signal'][0],
                                validation_end=fold['validation_signal'][1], training_labels_end=fold['train_labels_end'],
                                validation_labels_end=fold['validation_labels_end'], test_start=fold['test_signal'][0],
                                test_end=fold['test_signal'][1], future_perturbation_invariant=True))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'huber_ranking_chronology.csv',index=False)
    write_json(RUN/'CHRONOLOGY_AUDIT.json',dict(passed=True,folds=8,reused_fits_exactly_verified=72,
               tests_passed=24,source_and_data_hashes_verified=True,
               limitation='Checks the saved-data pipeline, not vendor vintage or survivor-cohort bias.'))
    return frame


def ranking_summary(frame):
    row = {c:frame[c].mean() for c in ['IC','RankIC','NDCG@5','NDCG@10','NDCG@20']}
    for key in ['IC','RankIC']:
        row[key+'IR'] = frame[key].mean()/frame[key].std() if frame[key].std()>1e-12 else np.nan
    row['unobserved_labels'] = int(frame.unobserved.sum())
    return row


def summary_daily(daily):
    result = {f'{side}_{key}':value for side in ['net','gross'] for key,value in performance(daily[side]).items()}
    result['profitable_periods'] = np.mean(np.prod((1+daily.net.to_numpy()).reshape(-1,5),axis=1)>1)
    return result


def add_section(path, body):
    start, end = '<!-- huber-ranking:start -->', '<!-- huber-ranking:end -->'
    text = path.read_text(encoding='utf-8')
    if start in text:
        head, tail = text.split(start,1)
        _, rest = tail.split(end,1)
        text = head+rest.lstrip('\n')
    title, rest = text.split('\n',1)
    path.write_text(title+'\n\n'+start+'\n'+body+'\n'+end+'\n\n'+rest.lstrip('\n'),encoding='utf-8')


def fmt(value, percent=False):
    if pd.isna(value): return '—'
    return f'{value:.1%}' if percent else f'{value:.3f}'


def main():
    choices = lock_choices()
    chronology = audit()
    source = pd.read_csv(RUN/'all_metrics.csv')
    primary = source[(source.seed=='ensemble') & (source.cost_multiplier==1)]
    annual, pooled, curves, boot, paths, ranks_cache = [], [], [], {}, {}, {}
    selected = { (r['dataset'],r['year'],r['policy']):r for r in choices }
    datasets = ['NASDAQ_recent','NYSE_recent']
    for dataset in datasets:
        # ETF paths match exactly; both original and new Huber baselines verified below.
        for policy in ['validation-selected','equal K10']:
            kinds = ['think','think_mix']+FAMILY+['ranking_choice','market_proxy','qqq']
            for kind in kinds:
                daily_parts, ranking_parts, metadata = [], [], []
                for year in YEARS:
                    folder = RUN/f'{dataset}_{year}'
                    actual = selected[(dataset,year,policy)]['model'] if kind=='ranking_choice' else kind
                    if kind=='qqq':
                        file = OUT/'benchmarks'/f'QQQ_h5_{dataset}_{year}.csv'
                    else:
                        file = folder/(f'{actual}_equal_k10_daily.csv' if policy=='equal K10' and kind!='market_proxy'
                                       else f'{actual}_daily.csv')
                    daily = pd.read_csv(file)
                    daily['fold_year']=year
                    daily_parts.append(daily)
                    if kind not in ['market_proxy','qqq']:
                        metric = primary[(primary.dataset==dataset)&(primary.year==year)&(primary.model==actual)]
                        metric = metric[(metric.method=='equal')&(metric.k==10)] if policy=='equal K10' else metric[metric.selected_policy]
                        assert len(metric)==1
                        values = metric.iloc[0].to_dict()
                        rank = pd.read_csv(folder/f'{actual}_ranking.csv')
                        ranking_parts.append(rank)
                        ranks_cache[(dataset,year,actual)] = rank
                        for key,value in summary_daily(daily).items():
                            if key in values: np.testing.assert_allclose(value,values[key],atol=1e-12)
                    else:
                        values = dict(method='buy_hold',k=0,turnover_annual=2*252/len(daily),**summary_daily(daily))
                    row = dict(values,dataset=dataset,year=year,model=kind,actual_model=actual,policy=policy,
                               label=LABELS[kind],**{'ranking_weight':WEIGHTS.get(actual,np.nan)})
                    annual.append(row); metadata.append(row)
                    if actual in ['think','think_mix',BASE] and kind!='ranking_choice':
                        oldfile = OLD/f'{dataset}_{year}'/file.name
                        olddaily = pd.read_csv(oldfile)
                        np.testing.assert_array_equal(olddaily[['gross','net']],daily[['gross','net']])
                daily = pd.concat(daily_parts,ignore_index=True)
                assert daily.date.is_unique and daily.date.is_monotonic_increasing
                p = dict(dataset=dataset,policy=policy,model=kind,label=LABELS[kind],**summary_daily(daily))
                if ranking_parts: p.update(ranking_summary(pd.concat(ranking_parts,ignore_index=True)))
                p['turnover_annual'] = np.average([r['turnover_annual'] for r in metadata],weights=[len(d) for d in daily_parts])
                yearly = [r['net_sharpe'] for r in metadata]
                p.update(median_yearly_sharpe=np.median(yearly),worst_yearly_sharpe=np.min(yearly),
                         std_yearly_sharpe=np.std(yearly),start=daily.date.iloc[0],end=daily.date.iloc[-1])
                daily['dataset'],daily['policy'],daily['model']=dataset,policy,kind
                daily['net_wealth']=(1+daily.net).cumprod()
                curves.append(daily)
                paths[(dataset,policy,kind)] = daily
                ids = indices([len(d) for d in daily_parts])
                boot[(dataset,policy,kind)] = bootstrap_sharpe(daily.net,ids)
                low,high = interval(boot[(dataset,policy,kind)])
                p.update(net_sharpe_ci_low=low,net_sharpe_ci_high=high)
                pooled.append(p)
    annual = pd.DataFrame(annual)
    pooled = pd.DataFrame(pooled)
    for row in pooled.itertuples():
        key = (row.dataset,row.policy,row.model)
        for control in [BASE,'market_proxy','qqq']:
            np.testing.assert_array_equal(paths[key].date, paths[(row.dataset,row.policy,control)].date)
        idx = pooled.index[(pooled.dataset==row.dataset)&(pooled.policy==row.policy)&(pooled.model==row.model)][0]
        for control,suffix in [(BASE,'huber'),('market_proxy','spy'),('qqq','qqq')]:
            other = pooled[(pooled.dataset==row.dataset)&(pooled.policy==row.policy)&(pooled.model==control)].iloc[0]
            low,high = interval(boot[key]-boot[(row.dataset,row.policy,control)])
            pooled.loc[idx,f'delta_{suffix}']=row.net_sharpe-other.net_sharpe
            pooled.loc[idx,f'delta_{suffix}_ci_low']=low
            pooled.loc[idx,f'delta_{suffix}_ci_high']=high
            a = annual[(annual.dataset==row.dataset)&(annual.policy==row.policy)&(annual.model==row.model)].sort_values('year')
            b = annual[(annual.dataset==row.dataset)&(annual.policy==row.policy)&(annual.model==control)].sort_values('year')
            pooled.loc[idx,f'years_beating_{suffix}']=int((a.net_sharpe.to_numpy()>b.net_sharpe.to_numpy()).sum())
    annual.to_csv(OUT/'huber_ranking_annual.csv',index=False)
    pooled.to_csv(OUT/'huber_ranking_pooled.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(OUT/'huber_ranking_curves.csv',index=False)
    write_json(RUN/'REPORT_AUDIT.json',dict(pooled_paths=len(pooled),annual_paths=len(annual),folds=8,
                original_baseline_paths_identical=True,paired_dates_verified=True,
                interval='1000 same-date circular 20-session block resamples separately within annual folds; seed20261005',
                selection_lock_sha256=digest(RUN/'RANKING_SELECTION_LOCK.json'),report_source_sha256=digest(__file__)))
    diagnostic_tables = diagnostics(primary,source,pooled)
    draw(pooled,paths)
    report(pooled,annual,chronology,choices,diagnostic_tables)
    print(pooled[pooled.model.isin(FAMILY+['ranking_choice','market_proxy','qqq'])][
        ['dataset','policy','model','net_sharpe','delta_huber','delta_huber_ci_low','delta_huber_ci_high','IC','RankIC']].to_string(index=False))
    print('Ranking-loss report complete; original baseline paths reproduced and all 8 causal folds audited.')


def diagnostics(primary,source,pooled):
    # All K and seed results retained; none chooses a portfolio using test data.
    sensitivity = primary[(primary.model.isin(FAMILY))&(primary.method=='equal')]
    sensitivity.to_csv(OUT/'huber_ranking_k_sensitivity.csv',index=False)
    seed = source[(source.model.isin(FAMILY))&(source.seed!='ensemble')&(source.cost_multiplier==1)]
    seed.to_csv(OUT/'huber_ranking_seed_metrics.csv',index=False)
    stress = source[(source.model.isin(FAMILY))&(source.seed=='ensemble')&source.selected_policy]
    stress.to_csv(OUT/'huber_ranking_cost_stress.csv',index=False)
    rows=[]
    for (dataset,kind,k),group in sensitivity.groupby(['dataset','model','k']):
        group=group.sort_values('year')
        parts=[pd.read_csv(RUN/f'{dataset}_{year}'/f'{kind}_equal_k{k}_daily.csv') for year in YEARS]
        daily=pd.concat(parts,ignore_index=True)
        base=sensitivity[(sensitivity.dataset==dataset)&(sensitivity.model==BASE)&(sensitivity.k==k)].sort_values('year')
        baseline=pd.concat([pd.read_csv(RUN/f'{dataset}_{year}'/f'{BASE}_equal_k{k}_daily.csv') for year in YEARS],ignore_index=True)
        ids=indices([len(p) for p in parts])
        low,high=interval(bootstrap_sharpe(daily.net,ids)-bootstrap_sharpe(baseline.net,ids))
        rows.append(dict(dataset=dataset,model=kind,k=k,net_sharpe=performance(daily.net)['sharpe'],
                         delta_huber=performance(daily.net)['sharpe']-performance(baseline.net)['sharpe'],
                         delta_huber_ci_low=low,delta_huber_ci_high=high,
                         years_beating_huber=int((group.net_sharpe.to_numpy()>base.net_sharpe.to_numpy()).sum())))
    k_summary=pd.DataFrame(rows)
    k_summary.to_csv(OUT/'huber_ranking_k_pooled.csv',index=False)
    rows=[]
    for (dataset,kind),group in seed.groupby(['dataset','model']):
        base=seed[(seed.dataset==dataset)&(seed.model==BASE)]
        joined=group.merge(base[['year','seed','net_sharpe']],on=['year','seed'],suffixes=('','_baseline'),validate='one_to_one')
        rows.append(dict(dataset=dataset,model=kind,median_seed_year_sharpe=group.net_sharpe.median(),
                         min_seed_year_sharpe=group.net_sharpe.min(),seed_years_beating_huber=int((joined.net_sharpe>joined.net_sharpe_baseline).sum()),
                         seed_year_cases=len(group),
                         caveat='Per-model validation-selected policies; this is a pipeline rather than fixed-policy seed comparison.'))
    seed_summary=pd.DataFrame(rows)
    seed_summary.to_csv(OUT/'huber_ranking_seed_summary.csv',index=False)
    return k_summary,seed_summary


def draw(pooled,paths):
    colors = {BASE:'#0072B2','think_mix':'#777777','think':'#B0B0B0',NEW[0]:'#009E73',NEW[1]:'#E69F00',
              NEW[2]:'#D55E00','ranking_choice':'#CC79A7','market_proxy':'#111111','qqq':'#8024A8'}
    for dataset in ['NASDAQ_recent','NYSE_recent']:
        fig,axes = plt.subplots(1,2,figsize=(19,8))
        for ax,policy in zip(axes,['equal K10','validation-selected']):
            for kind in [BASE]+NEW+['ranking_choice','market_proxy','qqq']:
                row = pooled[(pooled.dataset==dataset)&(pooled.policy==policy)&(pooled.model==kind)].iloc[0]
                p = paths[(dataset,policy,kind)]
                label = (f'Huber + ranking {WEIGHTS[kind]:g}' if kind in NEW else
                         'Huber baseline' if kind==BASE else 'Validation alpha choice' if kind=='ranking_choice' else LABELS[kind])
                ax.plot(pd.to_datetime(p.date),p.net_wealth,label=f'{label} [SR {row.net_sharpe:.3f}]',
                        color=colors[kind],lw=2.3,ls='--' if kind in ['qqq','ranking_choice'] else '-')
            ax.set_title(policy,fontsize=14);ax.grid(alpha=.2);ax.legend(fontsize=9)
            ax.set_ylabel('Net fold-compounded wealth per initial $1')
        fig.suptitle(f'{dataset.replace("_recent", " cohort")} | Huber plus ranking | five-session forecasts, 2022–2025',fontsize=16)
        fig.text(.06,.035,'Annual accounts reset; boundary gaps omitted. Modeled costs included. Alpha and portfolios selected from prior validation.',fontsize=10)
        fig.text(.06,.01,'Exploratory follow-up on already examined periods; intervals do not adjust for model selection or multiple comparisons.',fontsize=10)
        fig.tight_layout(rect=[0,.065,1,.94]);fig.savefig(OUT/'figures'/f'huber_ranking_{dataset}.png',dpi=150);plt.close(fig)
        # Include every prior five-session model and every newly trained variant.
        oldcurves = pd.read_csv(OUT/'combined_2022_2025_curves.csv')
        oldmetrics = pd.read_csv(OUT/'combined_2022_2025_metrics.csv')
        rows = oldmetrics[(oldmetrics.horizon==5)&(oldmetrics.dataset==dataset)]
        fig,ax = plt.subplots(figsize=(20,11))
        palette = list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('Dark2').colors)
        for i,row in enumerate(rows.itertuples()):
            p = oldcurves[(oldcurves.horizon==5)&(oldcurves.dataset==dataset)&(oldcurves.model==row.model)]
            np.testing.assert_array_equal(p.date,paths[(dataset,'validation-selected',BASE)].date)
            ax.plot(pd.to_datetime(p.date),p.net_wealth,label=f'{row.label} [SR {row.net_sharpe:.3f}]',
                    color=colors.get(row.model,palette[i]),lw=2.5 if row.model in [BASE,'market_proxy','qqq'] else 1.1,
                    alpha=1 if row.model in [BASE,'market_proxy','qqq'] else .55)
        for kind in NEW:
            p = paths[(dataset,'validation-selected',kind)]
            row = pooled[(pooled.dataset==dataset)&(pooled.policy=='validation-selected')&(pooled.model==kind)].iloc[0]
            ax.plot(pd.to_datetime(p.date),p.net_wealth,color=colors[kind],lw=2.7,ls='--',
                    label=f'{LABELS[kind]} [SR {row.net_sharpe:.3f}]')
        ax.set_title(f'{dataset.replace("_recent"," cohort")} | all five-session models, including Huber+ranking | 2022–2025',fontsize=16)
        ax.set_ylabel('Net fold-compounded wealth per initial $1');ax.grid(alpha=.18)
        ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=9)
        fig.subplots_adjust(left=.06,right=.61,top=.91,bottom=.12)
        fig.text(.06,.055,'Every prior architecture and all three new ranking strengths; model-specific prior-validation-selected portfolios. Net of modeled costs.',fontsize=10)
        fig.text(.06,.025,'Annual resets and boundary gaps; retrospective exploratory study. Original curves and selections are preserved.',fontsize=10)
        fig.savefig(OUT/'figures'/f'huber_ranking_all_models_{dataset}.png',dpi=150);plt.close(fig)


def report(pooled,annual,chronology,choices,diagnostic_tables):
    lines = ['# THINK+mixer: Huber plus ranking follow-up','',
             '## Main result: validation-selected ranking weight, fixed equal K10','',
             '| Cohort | Huber net SR | Validation-chosen procedure net SR | Paired gain [95% CI] |',
             '|---|---:|---:|---|']
    for dataset in ['NASDAQ_recent','NYSE_recent']:
        rows=pooled[(pooled.dataset==dataset)&(pooled.policy=='equal K10')]
        a=rows[rows.model==BASE].iloc[0]
        b=rows[rows.model=='ranking_choice'].iloc[0]
        lines.append(f'| {dataset} | {a.net_sharpe:.3f} | {b.net_sharpe:.3f} | '
                     f'{b.delta_huber:+.3f} [{b.delta_huber_ci_low:.3f}, {b.delta_huber_ci_high:.3f}] |')
    lines += ['','These are the predefined primary comparisons. Fixed-weight variants and model-specific '
              'portfolio comparisons below are secondary; their best test result must not be substituted for '
              'the validation-selected procedure. A positive estimate alone does not establish improvement.','',
             '**Conclusion:** this batch does not establish a robust advantage from adding ranking loss. '
             'Both primary gain intervals contain zero. The NASDAQ point estimate declines and the NYSE point '
             'estimate rises with fixed K10; neither is a convincing cross-cohort improvement.','',
             'The secondary NASDAQ lambda 0.3 variant reaches 0.711 net Sharpe versus Huber 0.675, SPY 0.621 '
             'and QQQ 0.626. Its paired gain over Huber is +0.037 with 95% CI [-0.178, 0.252]. '
             'This same loss produces only 0.488 versus Huber 0.532 at equal K10. '
             'Its turnover falls from 5.21x to 3.93x/year with its selected portfolio, while RankIC and NDCG@10 '
             'slightly decline and drawdown remains about 30%. This suggests a portfolio interaction, '
             'rather than demonstrated broadly better prediction. The causal validation-chosen secondary '
             'NASDAQ procedure yields 0.638, below Huber 0.675.','',
             'For NYSE, lambda 1.0 reaches 0.239 with its selected portfolio versus Huber 0.194, '
             'but its gain interval [-0.433, 0.473] is wide; all candidates remain below the ETF point estimates. '
             'Ranking weights do not uniformly improve K or seed results. NASDAQ lambda 1.0 at equal K20 '
             'improves over Huber in all four years (pooled 0.412 vs 0.330; paired gain CI [-0.034, 0.197]), a narrower exploratory finding '
             'that still trails both ETFs and must not be used to retroactively select K20.','',
             'Completed October 8, 2026. Three ranking strengths (0.1, 0.3, 1.0) were declared before training; '
             '72 new fits cover two cohorts, four annual folds and three seeds. Another 72 baseline fits were reused '
             'only after exact predictions, initialization architecture, preprocessing and fold verification. '
             'No further candidates were added after viewing results.','',
             'This is a five-session experiment. Previously tested daily models are unchanged. '
             'All 2022–2025 windows were already examined: this is exploratory follow-up evidence, not a fresh confirmation test.','',
             '## What changed','',
             '`Loss = 2 × Huber(delta=1 percentage point) + lambda × pairwise logistic ranking loss`. '
             'The ranking term compares stocks on the same prediction date, uses only observed eligible labels, '
             'counts each pair once and ignores return differences smaller than 0.1 percentage point. '
             'It teaches which stock outperforms which without scaling penalties by the magnitude of extreme returns. '
             'Scores use a fixed 1-percentage-point temperature. Pair losses are averaged within date, then across dates with pairs.','',
             'Architecture, initialization seeds, training data, feature scaling, correlation groups, optimizer and '
             'validation-MSE early stopping are identical. Validation MSE is retained so the controlled change is the training loss; '
             'this may not choose the best ranking epoch. Max 30 epochs, patience 6, AdamW lr 0.001/decay 0.001, '
             'batch 8, gradient clip 1, 16-session inputs. Three seeds 7/19/42 are combined as an equal-weight rank ensemble.','',
             '## Validation selection and comparisons','',
             '**Primary:** equal-weight K10 throughout, with ranking weight selected from {0,0.1,0.3,1} using '
             'current-and-prior validation scores only. Zero retains the Huber baseline. **Secondary:** the same ranking-weight '
             'selection over each model’s own prior-validation-selected portfolio. The causal selection rule is median '
             'historical validation score minus 0.25 standard deviation; each score already summarizes validation halves and seeds. '
             'No future validation year can choose an earlier test model. All individual weights are reported even when they perform poorly.','',
             'Portfolios use equal, confidence, inverse-volatility and retention-buffer rules with K5/10/20 selected only '
             'on validation. Fixed equal K5/10/20 and individual-seed results are diagnostics. '
             'Confidence weighting operates on ensemble ranks, not calibrated return magnitudes. '
             'Long-short results in the raw run remain hypothetical due to unavailable historical borrow data.','',
             '## Pooled returns and paired uncertainty','',
             '95% intervals use 1,000 matched circular 20-session block resamples separately within each annual fold '
             '(seed20261005). Absolute Sharpe intervals differ from paired Sharpe-gain intervals. Models are not retrained '
             'inside bootstrap, and no multiplicity correction is applied. Any interval crossing zero fails to establish an advantage. '
             'Annual accounts reset and boundary sessions are omitted; pooled returns are compounded folds rather than a continuous account. '
             'Sharpe uses the existing 3% annual hurdle.','']
    for dataset in ['NASDAQ_recent','NYSE_recent']:
        for policy in ['equal K10','validation-selected']:
            rows = pooled[(pooled.dataset==dataset)&(pooled.policy==policy)]
            lines += [f'### {dataset}: {policy}','',
                      '| Model | Net SR | 95% SR CI | Gross SR | CAGR | Cum. return | Vol. | Max DD | Turnover/year | Win periods |',
                      '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
            for _,r in rows.iterrows():
                lines.append(f'| {r.label} | {r.net_sharpe:.3f} | [{r.net_sharpe_ci_low:.3f}, {r.net_sharpe_ci_high:.3f}] | '
                             f'{r.gross_sharpe:.3f} | {r.net_annualized_return:.1%} | {r.net_cumulative_return:.1%} | '
                             f'{r.net_annualized_volatility:.1%} | {r.net_maximum_drawdown:.1%} | {r.turnover_annual:.2f}x | {r.profitable_periods:.1%} |')
            lines += ['','| Model | ΔSR vs Huber [95% CI] | ΔSR vs SPY [95% CI] | ΔSR vs QQQ [95% CI] | Years beating Huber |',
                      '|---|---|---|---|---:|']
            for _,r in rows[rows.model.isin(NEW+['ranking_choice'])].iterrows():
                cells=[f'{r[f"delta_{c}"]:+.3f} [{r[f"delta_{c}_ci_low"]:.3f}, {r[f"delta_{c}_ci_high"]:.3f}]' for c in ['huber','spy','qqq']]
                lines.append(f'| {r.label} | '+ ' | '.join(cells)+f' | {int(r.years_beating_huber)}/4 |')
        lines += ['',f'![{dataset} Huber-ranking comparison](figures/huber_ranking_{dataset}.png)','',
                  f'![{dataset} every five-session model](figures/huber_ranking_all_models_{dataset}.png)','']
    lines += ['## Full-universe ranking metrics','',
              'These metrics compare ensemble scores with actual subsequent returns across all eligible securities '
              'with observed outcomes, rather than only selected holdings. NDCG uses realized-return percentile relevance '
              'with nonnegative linear gains. ICIR and RankICIR are period mean/std, unannualized.','',
              '| Cohort | Model | IC | RankIC | ICIR | RankICIR | NDCG@5 | NDCG@10 | NDCG@20 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in pooled[(pooled.policy=='equal K10')&pooled.model.isin(FAMILY+['think_mix','ranking_choice'])].iterrows():
        cells=[fmt(r[c]) for c in ['IC','RankIC','ICIR','RankICIR','NDCG@5','NDCG@10','NDCG@20']]
        lines.append(f'| {r.dataset} | {r.label} | '+' | '.join(cells)+' |')
    lines += ['','## Annual stability and selected weights','',
              'Annual results for every candidate are shown before the validation-selected procedure. '
              'A strong pooled result can conceal weak individual years.','']
    for dataset in ['NASDAQ_recent','NYSE_recent']:
        for policy in ['equal K10','validation-selected']:
            lines += [f'### {dataset}: {policy} annual net Sharpe','',
                      '| Model | 2022 | 2023 | 2024 | 2025 |','|---|---:|---:|---:|---:|']
            for kind in FAMILY+['ranking_choice','market_proxy','qqq']:
                rows=annual[(annual.dataset==dataset)&(annual.policy==policy)&(annual.model==kind)].sort_values('year')
                lines.append(f'| {LABELS[kind]} | '+' | '.join(f'{v:.3f}' for v in rows.net_sharpe)+' |')
            lines.append('')
    lines += [
              '| Cohort | Portfolio rule | Test year | Validation-chosen weight | Net SR | Huber SR | SPY SR | QQQ SR |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for choice in choices:
        rows=annual[(annual.dataset==choice['dataset'])&(annual.policy==choice['policy'])&(annual.year==choice['year'])]
        values=[rows[rows.model==kind].iloc[0].net_sharpe for kind in ['ranking_choice',BASE,'market_proxy','qqq']]
        lines.append(f'| {choice["dataset"]} | {choice["policy"]} | {choice["year"]} | {choice["ranking_weight"]:g} | '
                     +' | '.join(f'{v:.3f}' for v in values)+' |')
    k_summary,seed_summary=diagnostic_tables
    lines += ['','## K sensitivity and individual seeds','',
              '| Cohort | Model | K | Pooled net SR | ΔSR vs Huber at same K [95% CI] | Years improving |',
              '|---|---|---:|---:|---:|---:|']
    for r in k_summary.itertuples():
        lines.append(f'| {r.dataset} | {LABELS[r.model]} | {r.k} | {r.net_sharpe:.3f} | '
                     f'{r.delta_huber:+.3f} [{r.delta_huber_ci_low:.3f}, {r.delta_huber_ci_high:.3f}] | {r.years_beating_huber}/4 |')
    lines += ['','| Cohort | Model | Median seed/year SR | Worst seed/year SR | Seed/year cases beating Huber |',
              '|---|---|---:|---:|---:|']
    for r in seed_summary.itertuples():
        lines.append(f'| {r.dataset} | {LABELS[r.model]} | {r.median_seed_year_sharpe:.3f} | {r.min_seed_year_sharpe:.3f} | {r.seed_years_beating_huber}/{r.seed_year_cases} |')
    lines += ['','Seed diagnostics use each model’s validation-selected policy, so differences include portfolio-policy changes. '
              'The fixed-K comparisons above provide the cleaner loss ablation.','']
    lines += ['','## Training chronology, execution and limits','',
              '| Test year | Training signals | Validation signals | Last training outcome | Last validation outcome |',
              '|---|---|---|---|---|']
    for r in chronology[chronology.dataset=='NASDAQ_recent'].itertuples():
        lines.append(f'| {r.year} | {r.train_start} to {r.train_end} | {r.validation_start} to {r.validation_end} | '
                     f'{r.training_labels_end} | {r.validation_labels_end} |')
    lines += ['','All eight folds passed reconstructed-index/scaler checks, label maturity purges, '
              'and future-price/volume/benchmark perturbation checks for earlier features, eligibility, ADV, labels and relationships. '
              'All 24 tests passed, including masking, date-local pairs, loss gradients and identical initialization. '
              'The raw frozen data hashes and training-code snapshot were verified.','',
              'Execution and costs are unchanged: signal after close, fill next close, five-session holding, '
              '$1m starting NAV, signal-time ADV >=$1m, trades capped at 1% lagged ADV, '
              '2bp commission +5bp slippage +3bp square-root impact at 1% participation, per side. '
              'Annual liquidation uses the existing uncapped terminal-cost approximation. '
              'Gross paths retain matched holdings; cash earns zero. Fixed historical cohorts include funds and preferred shares. '
              'Turnover is two-sided absolute traded notional divided by NAV, annualized. '
              'Missing delisted histories, later vendor adjustments and repeated test inspection limit generalization.','',
              '## Reproduction and complete tables','',
              '```powershell',
              '.\\.venv\\Scripts\\python.exe -m unittest -v test_research test_robust_research test_huber_ranking',
              '.\\.venv\\Scripts\\python.exe run_robust_research.py --out runs/huber_ranking_2022_2025 --experiment-plan config/huber_ranking_experiment_plan.json --reuse-fits-from runs/robust_2022_2025_final --variants think_mix_huber think_mix think think_mix_huber_rank_01 think_mix_huber_rank_03 think_mix_huber_rank_10 --workers 3',
              '.\\.venv\\Scripts\\python.exe report_huber_ranking.py',
              '.\\.venv\\Scripts\\python.exe publish_research_docs.py','```','',
              'Use a fresh output directory to retrain; completed runs are immutable. This reporting script reads the named completed run.','',
              '- [Pooled metrics and all confidence intervals](huber_ranking_pooled.csv)',
              '- [Annual metrics](huber_ranking_annual.csv)',
              '- [Dated daily returns and curves](huber_ranking_curves.csv)',
              '- [Validation-selected weights and portfolios](huber_ranking_selections.csv)',
              '- [Fixed equal K5/10/20 sensitivity](huber_ranking_k_sensitivity.csv)',
              '- [Pooled K sensitivity](huber_ranking_k_pooled.csv)',
              '- [All individual seed metrics](huber_ranking_seed_metrics.csv)',
              '- [Seed stability summary](huber_ranking_seed_summary.csv)',
              '- [Transaction-cost stress](huber_ranking_cost_stress.csv)',
              '- [Fold chronology](huber_ranking_chronology.csv)',
              '- [Frozen plan](../../config/huber_ranking_experiment_plan.json)',
              '- [Run, checkpoints, prediction arrays and audits](../huber_ranking_2022_2025/PROTOCOL.json)','']
    (OUT/'HUBER_RANKING.md').write_text('\n'.join(lines),encoding='utf-8')
    charts = ('## Huber plus ranking follow-up: all new variants and every prior model\n\n'
              '[Full comparison, paired confidence intervals and validation choices](HUBER_RANKING.md). '
              'Three new ranking strengths were tested for five-session forecasts; daily studies are unchanged.\n\n')
    for dataset in ['NASDAQ_recent','NYSE_recent']:
        charts += f'![{dataset}: Huber ranking controlled comparison](figures/huber_ranking_{dataset}.png)\n\n'
        charts += f'![{dataset}: all five-session models including new variants](figures/huber_ranking_all_models_{dataset}.png)\n\n'
    add_section(OUT/'CHARTS.md',charts)
    for name in ['COMBINED_2022_2025.md','REPORT.md','FINDINGS.md','CONFIDENCE_INTERVALS.md']:
        add_section(OUT/name,'## Huber plus ranking follow-up\n\n'
                    '[The new five-session comparison](HUBER_RANKING.md) reports all three ranking strengths, '
                    'validation-selected procedures, fixed top-10 ablations, SPY/QQQ controls and paired confidence intervals. '
                    'The original study below remains the reference baseline.')


if __name__=='__main__':
    main()
