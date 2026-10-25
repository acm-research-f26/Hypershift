"""Report every frozen draw and seed, without choosing a new winner."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_engine import load_panel,prepare_panel,fold_indices,performance
from run_robust_research import ensemble_predictions,model_data
from audit_backtests import share_ledger
from report_pooled_uncertainty import indices,bootstrap_sharpe,interval
from refresh_audited_charts import draw,BENCHMARK_NOTE,LABEL,COLORS

ROOT=Path('runs/random_trials_20261010');OUT=Path('docs/random_trials_20261010')
MODELS=['think','think_mix','think_huber','think_mix_huber','ridge','momentum126','reversal5']

def main():
    assert (ROOT/'completion.json').exists(),'Wait for all locked training and evaluation to complete'
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    plan=json.loads(Path('config/random_trials_20261010.json').read_text())
    manifest=json.loads(Path('data/random_trials_20261010/manifest.json').read_text())
    rawcoverage=pd.read_csv('data/random_trials_20261010/coverage.csv').set_index('ticker')
    annual=[];pooled=[];seeds=[];curves=[];coverage=[];paths={};independent_paths=0
    for dataset,names in manifest['selection'].items():
        data=prepare_panel(load_panel(dataset),horizon=5)
        coverage.append(dict(dataset=dataset,chosen=100,recovered=int((rawcoverage.loc[names].observations>0).sum()),
                             eligible_min=int(data['mask'].sum(1).min()),eligible_max=int(data['mask'].sum(1).max())))
        pieces={};seedpieces={}
        for year in plan['test_years']:
            folder=ROOT/f'{dataset}_{year}';_,_,test=fold_indices(data,year)
            for kind in MODELS:
                cache=np.load(folder/f'{kind}_predictions.npz');dd=model_data(data,kind)
                score=ensemble_predictions(cache['test'],dd['mask'][test]) if len(cache['test'])>1 else cache['test'][0]
                chosen=json.loads((folder/f'{kind}_policy.json').read_text())['selected']
                for policy,k,method,suffix in [('fixed_k10',10,'equal','equal_k10_daily'),('selected',chosen['k'],chosen['method'],'daily')]:
                    p=pd.read_csv(folder/f'{kind}_{suffix}.csv')
                    independent,_,_=share_ledger(dd,test,score,k,method)
                    np.testing.assert_allclose(p.net,independent,atol=2e-12,rtol=1e-10);independent_paths+=1
                    p=p.assign(model=kind,dataset=dataset,fold_year=year,policy=policy,horizon=5)
                    pieces.setdefault((kind,policy),[]).append(p);curves.append(p)
                    annual.append(dict(dataset=dataset,year=year,model=kind,policy=policy,k=k,method=method,**performance(p.net)))
                for seed,pred in zip(plan['training_seeds'] if len(cache['test'])>1 else [0],cache['test']):
                    from robust_engine import backtest
                    _,p,_=backtest(dd,test,pred,10,'equal');seedpieces.setdefault((kind,seed),[]).append(p)
            for kind in ['equal_universe','buy_hold','market_proxy']+(['qqq'] if dataset.startswith('NASDAQ') else []):
                f=folder/f'{kind}_daily.csv' if kind!='qqq' else Path(f'runs/robust_comparison/benchmarks/QQQ_h5_NASDAQ_recent_{year}.csv')
                p=pd.read_csv(f)
                for policy in ['fixed_k10','selected']:
                    piece=p.assign(model=kind,dataset=dataset,fold_year=year,policy=policy,horizon=5)
                    pieces.setdefault((kind,policy),[]).append(piece);curves.append(piece)
        for (kind,policy),values in pieces.items():
            p=pd.concat(values,ignore_index=True);assert p.date.is_unique
            paths[(dataset,kind,policy)]=p
            pooled.append(dict(dataset=dataset,market=dataset.split('_')[0],sampling_seed=int(dataset.split('draw')[1]),model=kind,policy=policy,**performance(p.net)))
        for (kind,seed),values in seedpieces.items():
            seeds.append(dict(dataset=dataset,model=kind,training_seed=seed,**performance(pd.concat(values).net)))
        print('Reported and independently audited',dataset,flush=True)
    metrics=pd.DataFrame(pooled);allcurves=pd.concat(curves,ignore_index=True)
    mean=metrics.groupby(['market','model','policy'],sort=False).agg(mean_sharpe=('sharpe','mean'),sd_sharpe=('sharpe','std'),
          min_sharpe=('sharpe','min'),max_sharpe=('sharpe','max'),mean_annualized_return=('annualized_return','mean'),mean_drawdown=('maximum_drawdown','mean')).reset_index()
    paired=[]
    for market in ['NYSE','NASDAQ']:
      for policy in ['fixed_k10','selected']:
        for a,b in [('think_mix','think'),('think_mix_huber','think_huber'),('momentum126','ridge')]:
            distributions=[];observed=[];wins=0
            for seed in plan['sampling_seeds']:
                pa,pb=paths[(f'{market}_draw{seed}',a,policy)],paths[(f'{market}_draw{seed}',b,policy)]
                np.testing.assert_array_equal(pa.date,pb.date)
                ids=indices(pa.groupby('fold_year',sort=True).size().tolist())
                distributions.append(bootstrap_sharpe(pa.net,ids)-bootstrap_sharpe(pb.net,ids))
                difference=performance(pa.net)['sharpe']-performance(pb.net)['sharpe'];observed.append(difference);wins+=difference>0
            low,high=interval(np.mean(distributions,axis=0))
            paired.append(dict(market=market,policy=policy,model_a=a,model_b=b,mean_sharpe_difference=np.mean(observed),
                               low=low,high=high,draws_a_beats_b=int(wins),draw_count=3,
                               interpretation='Descriptive exploratory paired interval with same date/block indices across dependent draws; no multiple-testing correction'))
    paired=pd.DataFrame(paired)
    for name,value in [('annual.csv',pd.DataFrame(annual)),('pooled.csv',metrics),('seed_fixed_k10.csv',pd.DataFrame(seeds)),
                       ('mean_across_draws.csv',mean),('paired_differences.csv',paired),('coverage.csv',pd.DataFrame(coverage)),('curves.csv',allcurves)]:
        value.to_csv(OUT/name,index=False)
    sections=['# Three fresh random samples per exchange','',BENCHMARK_NOTE,'',
              'Three 100-name draws per exchange were fixed with seeds 101, 202 and 303 before fetching histories. '
              'Each neural model uses training seeds 7, 19 and 42, and annual 2022–2025 tests. All 288 neural fits completed. '
              'The four neural variants isolate THINK, THINK+mixer, and their Huber counterparts. '
              'Ridge, momentum, reversal and passive controls accompany each draw. No missing issuer was replaced. '
              'Fresh trials use five-session forecasts. The original and top-100 daily paths were audited and retained, but daily models were not retrained on these new draws.','',
              'The primary comparison fixes equal top-10 allocation. Secondary comparisons preserve the original prior-validation-only '
              'K/allocation search. All draws and training seeds remain visible. Dates are identical across draws, and source names can overlap; '
              'these are dependent sensitivity trials, not independent new market histories. Averaging reduces sensitivity to a particular ticker draw '
              'but cannot remove missing delistings, universe bias or hindsight from previously inspected test years.','',
              '[Mean and spread across draws](mean_across_draws.csv) · [Every draw](pooled.csv) · [Every training seed](seed_fixed_k10.csv) · '
              '[Paired differences](paired_differences.csv) · [Coverage](coverage.csv) · [Dated paths](curves.csv)','',
              '126-session momentum ranks past six-month adjusted returns. Ridge fits regularized linear forecasts on training-only lagged features. '
              'Neither optimizes test returns. Strong results on the current top-100 cohort do not imply the same results on these historical samples.','']
    original=pd.read_csv('docs/audit/backtests_20261010/pooled.csv')
    for market in ['NYSE','NASDAQ']:
        group=metrics[(metrics.market==market)&(metrics.policy=='fixed_k10')]
        labels=MODELS+['equal_universe','market_proxy']+(['qqq'] if market=='NASDAQ' else [])
        fig,ax=plt.subplots(figsize=(15,7))
        for seed,offset,color in [(101,-.22,'#0072b2'),(202,0,'#d55e00'),(303,.22,'#009e73')]:
            sub=group[group.sampling_seed==seed].set_index('model');ax.scatter(np.arange(len(labels))+offset,[sub.loc[m,'sharpe'] for m in labels],s=65,label=f'Draw {seed}',color=color)
        for study,marker,color in [('original','x','#555555'),('top100','D','#8844aa')]:
            sub=original[(original.study==study)&original.dataset.str.startswith(market)&(original.horizon==5)&(original.policy=='fixed_k10')].set_index('model')
            available=[(j,m) for j,m in enumerate(labels) if m in sub.index]
            ax.scatter([j for j,m in available],[sub.loc[m,'sharpe'] for j,m in available],marker=marker,s=50,color=color,label=study+' cohort')
        ax.axhline(0,color='#777777',lw=.7);ax.set_xticks(np.arange(len(labels)),[LABEL.get(m,m).replace('NYSE cohort basket (simulated)','Cohort basket') for m in labels],rotation=25,ha='right')
        ax.set(ylabel='Net Sharpe, fixed equal top 10',title=f'{market}: sensitivity to ticker draw, 2022–2025');ax.grid(axis='y',alpha=.2);ax.legend()
        fig.tight_layout();f=OUT/'figures'/f'{market}_draw_sensitivity.png';fig.savefig(f,dpi=140);plt.close(fig)
        sections += [f'## {market}: all draws','',f'![Draw sensitivity](figures/{f.name})','',
                     '| Model | Draw 101 SR | Draw 202 SR | Draw 303 SR | Mean SR | SD across draws | Mean annualized return |',
                     '|---|---:|---:|---:|---:|---:|---:|']
        for kind in labels:
            values=group[group.model==kind].sort_values('sampling_seed');r=mean[(mean.market==market)&(mean.model==kind)&(mean.policy=='fixed_k10')].iloc[0]
            sections.append(f'| {LABEL.get(kind,kind).replace("NYSE cohort basket (simulated)","Cohort basket")} | '+ ' | '.join(f'{s:.3f}' for s in values.sharpe)+f' | {r.mean_sharpe:.3f} | {r.sd_sharpe:.3f} | {r.mean_annualized_return:.1%} |')
        sections += ['', '| Paired comparison | Mean ΔSharpe | Exploratory 95% interval | Draws with advantage |','|---|---:|---:|---:|']
        for r in paired[(paired.market==market)&(paired.policy=='fixed_k10')].itertuples():
            sections.append(f'| {LABEL.get(r.model_a,r.model_a)} minus {LABEL.get(r.model_b,r.model_b)} | {r.mean_sharpe_difference:.3f} | [{r.low:.3f}, {r.high:.3f}] | {r.draws_a_beats_b}/3 |')
        for seed in plan['sampling_seeds']:
            dataset=f'{market}_draw{seed}';cov=next(r for r in coverage if r['dataset']==dataset)
            sections += ['',f'### Draw {seed}: {cov["recovered"]}/100 histories recovered; eligibility {cov["eligible_min"]}–{cov["eligible_max"]}','']
            for policy in ['fixed_k10','selected']:
                piece=allcurves[(allcurves.dataset==dataset)&(allcurves.policy==policy)]
                f=OUT/'figures'/f'{dataset}_{policy}.png';draw(piece,f,f'{dataset} | '+('fixed equal top 10' if policy=='fixed_k10' else 'prior-validation-selected allocation'))
                sections += [f'![{dataset} {policy}](figures/{f.name})','']
    sections += ['## How to interpret the averages','',
                 'Mean Sharpe is the arithmetic mean of the three separately computed draw Sharpes. It is a descriptive statistic, '
                 'not the Sharpe of a tradable pooled portfolio. The same-date block resamples are shared across draws when computing '
                 'paired intervals, preserving their common market shocks. Three draws give a first sensitivity check; more draws would '
                 'improve the sampling estimate but still use the same observed years. A convincing next test needs a frozen rule on '
                 'unseen dates and a point-in-time universe with delisted returns.','',
                 'The old cohorts and current top-100 cohort also differ in training backend (CPU versus GPU). These fresh neural fits '
                 'run on CPU with PyTorch 2.14.0+cu130; the original CPU fits used 2.14.0+cpu. The device class matches, '
                 'while the wheel build differs. Architecture claims should rely on comparisons within the same draw and backend.','']
    (OUT/'CHARTS.md').write_text('\n'.join(sections),encoding='utf-8')
    (OUT/'audit.json').write_text(json.dumps(dict(status='passed',neural_fits=288,independent_share_ledger_paths=independent_paths,
        selection_locked_before_test_metrics=True,draw_count_per_exchange=3,known_limitations=plan['limitations']),indent=2))
    print('Published complete random-trial charts and descriptive averages',flush=True)

if __name__=='__main__':main()
