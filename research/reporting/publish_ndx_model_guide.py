"""Plain-language shortlist and matched SPY/QQQ comparisons from saved accounts."""
from datetime import datetime,timezone
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter,StrMethodFormatter
import numpy as np
import pandas as pd

import publish_ndx_seed_robustness as overview
from pit_research_engine import load_data,etf_backtest
from research_engine import performance
from run_ndx_retention_replication import sha,write

ROOT=overview.ROOT
DOC=ROOT/'docs/ndx_model_guide_20261030'
LABELS={
    'prior':('Prior mixture / twenty seeds','#245fc4','-'),
    'baseline':('Previous neural issuer model','#168878','--'),
    'simple':('Historical prior only / no training seeds','#a4770a','-.'),
    'legacy':('Old selected retention recipe','#747b84',':'),
    'SPY':('SPY buy and hold','#bd4e3d','-'),
    'QQQ':('QQQ buy and hold','#8254ac','-')}


def checked_inputs():
    dates,metrics,paths,hashes=overview.comparison()
    controls=overview.prior.RUN/'CONTROL_METRICS.csv'
    published=json.loads((ROOT/'docs/ndx_prior_robustness_20261030/MANIFEST.json').read_text())
    assert sha(controls)==published['output_sha256']['CONTROL_METRICS.csv']
    control_frame=pd.read_csv(controls).set_index('id')
    simple={}
    for allocation in ('retain5','issuer5'):
        key=f'prior13__{allocation}__seed_free__c1__b03'
        ledger=overview.prior.RUN/'controls'/(key+'.csv');f=pd.read_csv(ledger)
        assert f.date.tolist()==dates.strftime('%Y-%m-%d').tolist()
        nav=np.r_[1e6,f.nav.to_numpy()];daily=nav[1:]/nav[:-1]-1
        measured=performance(daily)
        for field,value in measured.items():np.testing.assert_allclose(value,control_frame.loc[key,field],rtol=1e-10,atol=1e-12)
        simple[allocation]=(daily,measured)
        hashes[ledger.relative_to(ROOT).as_posix()]=sha(ledger)
    hashes[controls.relative_to(ROOT).as_posix()]=sha(controls)
    old=ROOT/'runs/ndx_retention_replication_20261029'
    lock=json.loads((old/'COMPLETE.json').read_text())['output_sha256']
    for name in ('PATHS.npz','METRICS.csv'):
        assert sha(old/name)==lock[name];hashes[(old/name).relative_to(ROOT).as_posix()]=sha(old/name)
    legacy={}
    fm=pd.read_csv(old/'METRICS.csv').set_index('id')
    with np.load(old/'PATHS.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['dates'],dates.strftime('%Y-%m-%d'))
        for group in ('original_five','new_five','ten_seed'):
            key=f'{group}__retention__cap75__b03__c1';daily=saved[key].copy()
            measured=performance(daily)
            for field,value in measured.items():np.testing.assert_allclose(value,fm.loc[key,field],rtol=1e-10,atol=1e-12)
            legacy[group]=(daily,measured)
    source=ROOT/'outputs/model_hedge_20261027/before_paths.npz'
    source_metrics=ROOT/'outputs/model_hedge_20261027/before_all_strategies.csv'
    with np.load(source,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['dates'],dates.strftime('%Y-%m-%d'))
        oldpaths=dict(zip(saved['ids'].astype(str),saved['returns'].T))
    oldmetrics=pd.read_csv(source_metrics).set_index('id')
    benchmarks={};replays=[];data=load_data()
    for ticker in ('SPY','QQQ'):
        measured,ledger=etf_backtest(data,ticker,cost_multiplier=1.)
        daily=ledger.net.to_numpy();key='original/'+ticker+'/10'
        error=float(np.max(np.abs(daily-oldpaths[key])))
        assert error<1e-12
        for field,oldfield in [('sharpe','sharpe'),('annualized_return','cagr'),('maximum_drawdown','max_drawdown')]:
            np.testing.assert_allclose(measured[field],oldmetrics.loc[key,oldfield],rtol=0,atol=1e-12)
        benchmarks[ticker]=(daily,performance(daily))
        replays.append(dict(benchmark=ticker,max_daily_error_vs_published=error,**performance(daily)))
    for p in (source,source_metrics,ROOT/'research/core/pit_research_engine.py'):
        hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    return dates,metrics,paths,simple,legacy,benchmarks,replays,hashes


def format_axes(axes,x):
    for col in range(axes.shape[1]):
        axes[0,col].yaxis.set_major_formatter(StrMethodFormatter('${x:.0f}'))
        axes[1,col].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        for ax in axes[:,col]:
            ax.grid(axis='y',alpha=.23);ax.spines[['top','right']].set_visible(False)
            ax.set_xlim(x[0],x[-1]);ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));ax.tick_params(labelsize=9)
    axes[0,0].set_ylabel('Net wealth from $100');axes[1,0].set_ylabel('Drawdown from prior peak')


def plot(dates,paths,simple,legacy,benchmarks,groups,filename,title,headline=False):
    n=len(groups);fig,axes=plt.subplots(2,n,figsize=(14,9.5) if headline else (19 if n==4 else 17,9),
        sharex=True,sharey='row',squeeze=False,gridspec_kw={'height_ratios':[2,1]})
    x=dates.insert(0,dates[0]-pd.offsets.BDay(1));handles={};labels={}
    for col,group in enumerate(groups):
        series=[('prior',paths[group,'prior',1,.03]),('baseline',paths[group,'baseline',1,.03])]
        if headline:series.append(('simple',simple['retain5'][0]))
        elif group in legacy:series.append(('legacy',legacy[group][0]))
        series += [(ticker,benchmarks[ticker][0]) for ticker in ('SPY','QQQ')]
        for name,daily in series:
            label,color,style=LABELS[name]
            if name=='prior' and not headline:label='Prior-regularized issuer ensemble'
            eq=np.r_[1.,np.cumprod(1+daily)];dd=eq/np.maximum.accumulate(eq)-1
            handles[name],=axes[0,col].plot(x,eq*100,color=color,ls=style,lw=2.1 if name=='prior' else 1.45)
            axes[1,col].plot(x,dd,color=color,ls=style,lw=1.25)
            if headline:
                m=performance(daily);label+=f' | SR {m["sharpe"]:.3f} | CAGR {m["annualized_return"]:.1%} | DD {m["maximum_drawdown"]:.1%}'
            labels[name]=label
        axes[0,col].set_title(overview.LABELS[group]+(' — candidate fixed before scoring' if group=='unused_five' else ''),
                              loc='left',fontsize=11 if n>1 else 12,weight='bold')
    format_axes(axes,x)
    fig.suptitle(title,x=.06,y=.98,ha='left',fontsize=18,weight='bold')
    fig.text(.06,.925,'2022–2025 • same 1,003 sessions • fees included • Sharpe above a 3% annual hurdle\nModels use stock risk sizing and a partial QQQ short; SPY/QQQ are fully invested buy-and-hold references.',
             va='top',fontsize=10,color='#536174')
    order=[name for name in ('prior','baseline','simple','legacy','SPY','QQQ') if name in handles]
    fig.legend([handles[k] for k in order],[labels[k] for k in order],loc='lower left',
        bbox_to_anchor=(.055,.048),frameon=False,ncol=1 if headline else 3,fontsize=9.5,handlelength=3)
    fig.text(.06,.02,'Benchmarks have no training seeds; repeated curves are the same reference. Historical results do not establish future performance.',fontsize=9,color='#536174')
    fig.subplots_adjust(left=.06,right=.985,top=.83,bottom=.24 if headline else .17,hspace=.13,wspace=.12)
    fig.savefig(DOC/filename,dpi=160,facecolor='white');plt.close(fig)


def main():
    DOC.mkdir(exist_ok=True)
    dates,metrics,paths,simple,legacy,benchmarks,replays,hashes=checked_inputs()
    plot(dates,paths,simple,legacy,benchmarks,['expanded_twenty'],'current_models_vs_benchmarks.png',
         'Current shortlist versus SPY and QQQ',True)
    plot(dates,paths,simple,legacy,benchmarks,overview.GROUPS[:4],'separate_cohorts_vs_benchmarks.png',
         'Separate seed cohorts: current model, old peak and benchmarks')
    plot(dates,paths,simple,legacy,benchmarks,overview.GROUPS[4:],'pooled_models_vs_benchmarks.png',
         'Ten, fifteen and twenty seeds versus the same benchmarks')
    base=metrics[(metrics.cost==1)&(metrics.borrow==.03)].set_index(['group','model'])
    headline=[]
    for group,model,label in [('expanded_twenty','prior','Prior mixture / twenty seeds'),
                              ('expanded_twenty','baseline','Previous neural issuer / twenty seeds'),
                              ('expanded_twenty','cohort','Neural retention / cohort targets / twenty seeds')]:
        row=base.loc[(group,model),:]
        headline.append(dict(label=label,**{f:float(row[f]) for f in performance(paths[group,model,1,.03])}))
    for allocation,label in [('retain5','Historical prior alone / five securities'),('issuer5','Historical prior alone / five issuers')]:
        headline.append(dict(label=label,**simple[allocation][1]))
    for group,label in [('original_five','Old selected retention / original five'),('new_five','Old selected retention / second five')]:
        headline.append(dict(label=label,**legacy[group][1]))
    for ticker in ('SPY','QQQ'):headline.append(dict(label=ticker+' buy and hold',**benchmarks[ticker][1]))
    pd.DataFrame(headline).to_csv(DOC/'HEADLINE_METRICS.csv',index=False)
    rows=[]
    for group in overview.GROUPS:
        row=dict(group=group,prior=float(base.loc[(group,'prior'),'sharpe']),
                 baseline=float(base.loc[(group,'baseline'),'sharpe']),
                 legacy=legacy[group][1]['sharpe'] if group in legacy else np.nan,
                 SPY=benchmarks['SPY'][1]['sharpe'],QQQ=benchmarks['QQQ'][1]['sharpe'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(DOC/'SEED_COMPARISON.csv',index=False)
    pd.DataFrame(replays).to_csv(DOC/'BENCHMARK_REPLAY.csv',index=False)
    p=base.loc[('expanded_twenty','prior'),:];s=simple['retain5'][1];si=simple['issuer5'][1]
    table=['| Strategy / reference | Sharpe | CAGR | Annual volatility | Worst drawdown |',
           '|---|---:|---:|---:|---:|']
    for row in headline:
        table.append(f'| {row["label"]} | {row["sharpe"]:.3f} | {row["annualized_return"]:.2%} | {row["annualized_volatility"]:.2%} | {row["maximum_drawdown"]:.2%} |')
    seed_table=['| Ensemble | New prior mixture SR | Previous neural SR | Old selected recipe SR | SPY SR | QQQ SR |',
                '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        old='—' if np.isnan(r['legacy']) else f'{r["legacy"]:.3f}'
        seed_table.append(f'| {overview.LABELS[r["group"]]} | {r["prior"]:.3f} | {r["baseline"]:.3f} | {old} | {r["SPY"]:.3f} | {r["QQQ"]:.3f} |')
    text=['# Best current results, with SPY and QQQ — October 30, 2026','',
        f'- **Best-supported current neural strategy: the prior mixture.** Twenty-seed SR **{p.sharpe:.3f}**, CAGR **{p.annualized_return:.2%}**, worst drawdown **{p.maximum_drawdown:.2%}**. Four separate five-seed groups score **1.272 / 1.184 / 1.409 / 1.322**. The last group was scored after the candidate was fixed.',
        f'- **Strong simple alternative: the historical-prior model alone.** SR **{s["sharpe"]:.3f}** with five-security retention, or **{si["sharpe"]:.3f}** with five issuers. It has no neural training randomness. This is one deterministic control, not repeated independent seed evidence.',
        '- **Highest selected historical peak: the old retention recipe.** SR **1.731** on the original five, **0.648** on the second five and **1.381** across ten. Its peak is higher; its initialization consistency is much weaker. The current mixture does not beat that old recipe in every available group.','',
        '**No evaluated strategy has achieved SR>1.5 in every predefined ensemble.** The current mixture has stronger cross-seed support than the old peak, not a demonstrated 1.7 Sharpe that generalizes.','',
        '![Current shortlist versus passive benchmarks](current_models_vs_benchmarks.png)','',*table,'',
        'Sharpe measures daily return above a 3% annual hurdle relative to how bumpy the path is. CAGR is compound growth per year; worst drawdown is the largest fall from an earlier account peak. The simple prior earns slightly higher CAGR than the twenty-seed mixture but has a bumpier path, so its Sharpe is lower. Different portfolio rules are present: the simple retention control and old strategy select securities, while the preferred mixture counts companies/issuers.','',
        '## How the preferred model works, in everyday terms','',
        '1. **Make a historical report card.** For each stock, use only the fold\'s training period and ask how well its returns ranked against other stocks on average. Recent training observations get more weight: an observation 26 decision-weeks older has half the weight. Limited histories are pulled toward a neutral score. The report card is fixed for the annual fold; it does not use that evaluation year\'s outcomes. For the 2025 fold, this prior ends in 2023, before 2024 validation.',
        '2. **Ask two neural models for their stock orderings.** THINK and the dynamic-hypergraph model use daily, after-hours and premarket inputs to rank stocks. Each seed starts another version of each network. A five-seed group therefore contains five versions of each neural model, not five stocks.',
        '3. **Combine the votes.** The common historical report card receives 50% of the ranking score; mean THINK rank gets 25% and mean dynamic-model rank 25%. For example, report card 85, THINK 90 and dynamic 70 give an overall 82.5. This is an ordering score, not an 82.5% return forecast.',
        '4. **Build and risk-size a portfolio.** About every five trading sessions, each five-seed committee selects five companies. Counting issuers stops GOOG and GOOGL from using two company slots. When the chosen basket has been more volatile, the stock allocation is reduced. A partial QQQ short uses the permitted remaining gross budget. It tends to gain when QQQ falls and lose when QQQ rises. If 70% of the target budget is long stocks, at most 30% can be short QQQ under this rule. The 75% short limit is a ceiling, not a permanent position. The 20% stock-volatility target guides sizing; actual account volatility can differ.',
        '5. **Combine portfolios for larger ensembles.** Twenty seeds form four five-seed committees. Each builds its own five-issuer portfolio; the account averages those signed targets and trades the net order. It can hold more than five issuers. This preserves different reasonable buy lists instead of forcing every model into one top-five vote. All predefined seeds have equal weight.','',
        'The simple historical-prior control skips the neural votes. Its stock preference list is mostly fixed within each annual fold; eligibility,retention and risk sizing can still change positions. Its strong result means a simple training-history strategy remains an important competitor. These results do not establish a statistically significant advantage for the more complex ensemble.','',
        '## Comparison across every seed ensemble','',*seed_table,'',
        'SPY and QQQ have no training seeds. Their repeated columns/curves are the same two buy-and-hold references, not extra experiments. Dashes mean the old recipe was not fitted for that group; no missing old result is extrapolated. The same cost formula, price convention, 1,003 sessions and 3% Sharpe hurdle apply. Benchmarks buy at the first common fill close and liquidate at the last; the active models trade repeatedly and additionally pay short borrowing.','',
        '![Four separate seed cohorts and benchmarks](separate_cohorts_vs_benchmarks.png)','',
        '![Pooled ensembles and benchmarks](pooled_models_vs_benchmarks.png)','',
        '**These are strategy comparisons, not a pure test of neural prediction skill.** SPY/QQQ stay fully invested; the active models select stocks, adjust exposure and partially short QQQ. Their market exposure and volatility differ. Higher historical Sharpe does not establish future superiority.','',
        '## Why this is more consistent than the old 1.7 result','',
        'The old retention rule gave last week\'s selected names a small score bonus before choosing five securities. That encourages keeping a near-tied name. It does not correct arbitrary model rankings, and it can count two share classes as separate slots.','',
        'The old result had a specific initialization problem. In the inspected 2023 fold, THINK\'s stock ordering correlated about 0.995 with its stock-specific output biases; those biases retained about 0.999 rank correlation with their random starting values. In plain language, some stocks began with arbitrary head starts that training barely changed. The lucky original group also held a large combined GOOG/GOOGL position. That is evidence for one initialization mechanism, not proof that every year or every part of the network was untrained.','',
        'The newer training starts those stock-specific output weights/biases neutrally, gives dynamic stock filters a common start, and trains on relative stock rankings rather than mainly fitting the common market-return level. Validation is separated chronologically for checkpoint selection and diagnostics. These training changes were bundled, so their individual causal contributions are not isolated.','',
        'The current winning addition is an anchor built from training data: half of the final score comes from the same historical report card in every seed group. Random starts still affect the neural half. Portfolio averaging further limits the effect of borderline top-five changes.','',
        'The reserved cohort is the clearest comparison: the newer neural issuer baseline scores 0.877 and the fixed prior mixture 1.322. Because this group contains only one five-seed committee, portfolio averaging cannot explain that gain; the ranking prior changes the stock choices. Across all four disjoint groups, baseline SR spread falls 0.323 → 0.225. The older 1.731 → 0.648 gap was 1.084 under a different recipe/training generation.','',
        'Consistency improved at a cost to the old selected peak. We retained all seeds and exposed the weak cohort instead of choosing the original five. Removing the stock-specific mixer entirely,using longer shared-feature forests,selecting different family weights or applying more beta hedging did not solve the problem; those failed curves and full grids remain below in the master gallery.','',
        '## What remains unproven','',
        'All cohorts share the repeatedly inspected 2022–2025 market history, and pooled groups overlap their constituent models. The preferred mixture was selected on development results, then frozen before the reserved cohort\'s financial outcomes. Its secondary seed-confirmation test passes; the original stricter 0.15 spread gate and the all-ensemble 1.5 target remain failed. A deterministic control has no initialization lottery but can still overfit the market history used to choose it. None of this is an untouched-market validation.','',
        '[Full experiment history and verification](../ndx_seed_robustness_20261030/REPORT.md) · [Headline metrics](HEADLINE_METRICS.csv) · [Seed comparison](SEED_COMPARISON.csv) · [Benchmark replay](BENCHMARK_REPLAY.csv) · [Provenance](MANIFEST.json)','']
    (DOC/'GUIDE.md').write_text('\n'.join(text),encoding='utf-8')
    (DOC/'HEADLINE_TABLE.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    (DOC/'SEED_TABLE.md').write_text('\n'.join(seed_table)+'\n',encoding='utf-8')
    write(DOC/'MANIFEST.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(ROOT/'research/reporting/publish_ndx_model_guide.py'),source_sha256=hashes,
        benchmark_replay_max_daily_error=max(r['max_daily_error_vs_published'] for r in replays),
        new_training=False,new_strategy_selection=False,
        output_sha256={p.name:sha(p) for p in DOC.iterdir() if p.is_file() and p.name!='MANIFEST.json'}))
    print('Published plain-language benchmark guide:',DOC,flush=True)


if __name__=='__main__':main()
