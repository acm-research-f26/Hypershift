"""Publish the complete audited risk-control grid and focused comparisons."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

import run_ndx_drawdown_research as study
from research_engine import performance

ROOT, RUN, DOC = study.ROOT, study.RUN, study.DOC
BASE = 'equal5/full'
RISK = 'issuer5/basketvol20_target20_weekly'
BALANCED = 'issuer5/ma200off50_basketvol20_target25_weekly'
LOWER = 'equal5/ma200off50_basketvol20_target20_weekly'
LABEL = {BASE:'Original blend: full exposure', RISK:'Five issuers + 20% volatility target',
         BALANCED:'Five issuers + 25% volatility target + trend cap',
         LOWER:'Original selections + 20% volatility target + trend cap'}


def table(frame, columns):
    lines=['| '+' | '.join(title for _,title,_ in columns)+' |',
           '| '+' | '.join('---' for _ in columns)+' |']
    for row in frame.to_dict('records'):
        lines.append('| '+' | '.join(formatter(row[key]) for key,_,formatter in columns)+' |')
    return '\n'.join(lines)


def metrics_table(frame):
    return table(frame,[('label','Variant',str),('group','Seed group',str),
                        ('sharpe','Sharpe',lambda x:f'{x:.3f}'),
                        ('annualized_return','Net CAGR',lambda x:f'{x:.2%}'),
                        ('maximum_drawdown','Max drawdown',lambda x:f'{x:.2%}')])


def plots(metrics,paths,dates):
    figures=DOC/'figures';figures.mkdir(parents=True,exist_ok=True)
    colors={BASE:'#6a7380',RISK:'#087d85',BALANCED:'#b36a16'}
    original=pd.read_csv(study.PARENT/'curves.csv')
    benchmarks={ticker:original[original.id==f'{ticker}/benchmark/ensemble/0/1'].net.to_numpy() for ticker in ['SPY','QQQ']}
    fig,axes=plt.subplots(2,2,figsize=(16,10),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    legend=[]
    for col,group in enumerate(['original_five','ten_seed']):
        for recipe in [BASE,BALANCED,RISK]:
            a=paths[recipe+'/'+group];wealth=np.r_[1,np.cumprod(1+a)]
            x=dates.insert(0,pd.Timestamp('2021-12-31'))
            row=metrics[(metrics.recipe==recipe)&(metrics.group==group)].iloc[0]
            line,=axes[0,col].plot(x,wealth,lw=2,color=colors[recipe],
                                  label=f'{LABEL[recipe]}')
            axes[1,col].plot(x,wealth/np.maximum.accumulate(wealth)-1,lw=1.7,color=colors[recipe])
            if col==0:legend.append(line)
        for ticker,color in [('SPY','#bb4c4c'),('QQQ','#8368a9')]:
            wealth=np.r_[1,np.cumprod(1+benchmarks[ticker])]
            line,=axes[0,col].plot(x,wealth,lw=1,ls='--',alpha=.8,color=color,label=ticker)
            axes[1,col].plot(x,wealth/np.maximum.accumulate(wealth)-1,lw=.8,ls='--',color=color,alpha=.8)
            if col==0:legend.append(line)
        group_label='Original five seeds' if col==0 else 'All ten seeds'
        axes[0,col].set_title(group_label,fontsize=13,weight='bold')
        chosen=metrics[(metrics.recipe==BALANCED)&(metrics.group==group)].iloc[0]
        axes[0,col].text(.03,.96,f'Return-preserving candidate\nSR {chosen.sharpe:.3f} | CAGR {chosen.annualized_return:.2%} | DD {chosen.maximum_drawdown:.2%}',
                         transform=axes[0,col].transAxes,va='top',fontsize=10,color=colors[BALANCED])
        axes[1,col].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        for ax in axes[:,col]:
            ax.grid(alpha=.18)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_xlim(x[0],x[-1])
    axes[0,0].set_ylabel('Net wealth per initial $1')
    axes[1,0].set_ylabel('Drawdown from prior peak')
    fig.suptitle('Lower historical drawdowns using the same saved model forecasts',fontsize=18,weight='bold',y=.98)
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,fontsize=10,frameon=False)
    fig.text(.5,.015,'2022-2025 exploratory development | 305 recipes searched | No leverage | Cash earns zero | Historical selection; future performance unconfirmed',
             ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.13,1,.94));fig.savefig(figures/'risk_control_comparison.png',dpi=160);plt.close(fig)
    # Show the entire search, including inferior recipes.
    ten=metrics[metrics.group=='ten_seed']
    fig,ax=plt.subplots(figsize=(12,7))
    scatter=ax.scatter(-ten.maximum_drawdown*100,ten.annualized_return*100,c=ten.sharpe,cmap='viridis',s=28,alpha=.75)
    scatter_labels={BASE:'Full-exposure baseline', RISK:'Five issuers + 20% volatility target',
                    BALANCED:'Five issuers + 25% volatility target + trend cap'}
    for recipe,offset in [(BASE,(-160,16)),(RISK,(-185,-32)),(BALANCED,(18,12))]:
        row=ten[ten.recipe==recipe].iloc[0]
        ax.scatter(-row.maximum_drawdown*100,row.annualized_return*100,s=100,marker='D',facecolors='none',edgecolors='black',lw=1.6)
        ax.annotate(scatter_labels[recipe]+f'\nSR {row.sharpe:.3f}',(-row.maximum_drawdown*100,row.annualized_return*100),
                    xytext=offset,textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='-',lw=.7,color='#555555'))
    ax.axvline(25,color='#777777',lw=1,ls='--');ax.grid(alpha=.18)
    ax.set_xlabel('Maximum drawdown magnitude (%) — lower is better')
    ax.set_ylabel('Net CAGR (%) — higher is better')
    ax.set_title('All 305 risk-control variants: ten-seed return / drawdown tradeoff',fontsize=14)
    fig.colorbar(scatter,ax=ax,label='Sharpe (fixed 3% annual hurdle)')
    fig.tight_layout();fig.savefig(figures/'all_variants_tradeoff.png',dpi=160);plt.close(fig)


def main():
    audit=json.loads((RUN/'AUDIT.json').read_text())
    if audit['status']!='passed':raise ValueError('Independent audit not passed')
    complete=json.loads((RUN/'COMPLETE.json').read_text())
    for name,h in complete['output_sha256'].items():
        if study.sha(RUN/name)!=h:raise ValueError('Completed output changed: '+name)
    DOC.mkdir(parents=True,exist_ok=True)
    exports=['PROTOCOL.json','FREEZE.json','AUDIT.json','COMPLETE.json','metrics.csv','summary.csv',
             'cost_stress.csv','annual.csv','uncertainty.csv']
    for name in exports:shutil.copyfile(RUN/name,DOC/name)
    metrics=pd.read_csv(RUN/'metrics.csv');summary=pd.read_csv(RUN/'summary.csv')
    stress=pd.read_csv(RUN/'cost_stress.csv');annual=pd.read_csv(RUN/'annual.csv')
    uncertainty=pd.read_csv(RUN/'uncertainty.csv')
    curves=pd.read_csv(RUN/'selected_curves.csv')
    focus=curves[curves.recipe.isin([BASE,RISK,BALANCED,LOWER])]
    focus.to_csv(DOC/'comparison_curves.csv',index=False)
    with np.load(RUN/'paths.npz') as f:
        paths=dict(zip(f['ids'],f['returns'].T));dates=pd.DatetimeIndex(f['dates'])
    plots(metrics,paths,dates)
    focus_recipes=[BASE,BALANCED,RISK,LOWER]
    overview=pd.concat([metrics[(metrics.recipe==recipe)&metrics.group.isin(['original_five','ten_seed'])] for recipe in focus_recipes]).copy()
    overview['label']=overview.recipe.map(LABEL)
    seed_table=metrics[metrics.recipe.isin([BASE,BALANCED,RISK])].copy();seed_table['label']=seed_table.recipe.map(LABEL)
    stress_table=stress[stress.recipe.isin([BASE,BALANCED,RISK])&(stress.group=='ten_seed')].copy();stress_table['label']=stress_table.recipe.map(LABEL)
    ablations=['equal5/full','equal5/constant_75','issuer5/full','equal5/ma200_off50_weekly',
               'equal5/basketvol20_target20_weekly',RISK,BALANCED]
    ablation=metrics[(metrics.group=='ten_seed')&metrics.recipe.isin(ablations)].copy()
    annual_table=annual[(annual.group=='ten_seed')&annual.recipe.isin([BASE,BALANCED,RISK])].copy()
    sensitivity=[]
    for recipe in [BASE,BALANCED,RISK]:
        a=paths[recipe+'/ten_seed']
        for label,mask in [('2022-2023',dates.year<=2023),('2024-2025',dates.year>=2024)]:
            sensitivity.append(dict(recipe=recipe,period=label,**performance(a[mask])))
        for year in [2022,2023,2024,2025]:
            # This is a descriptive subperiod metric; deleting years is not a tradable path.
            sensitivity.append(dict(recipe=recipe,period='omit '+str(year),**performance(a[dates.year!=year])))
    pd.DataFrame(sensitivity).to_csv(DOC/'subperiod_sensitivity.csv',index=False)
    rules='''1. Keep the original 50:50 THINK+mixer AH+PM / dynamic family percentile-rank blend and its five-session ranking schedule.
2. Rank issuers by their best share-class score, select five issuers, and use equal weights within the equity sleeve. The best class represents each issuer; tied best classes share its budget. GOOG/GOOGL, FOX/FOXA and LBTYA/LBTYK are treated as paired classes.
3. At each rebalance, apply those proposed stock weights to the last 20 daily constituent economic returns. Compute their sample standard deviation and annualize with sqrt(252). This incorporates the basket's past correlations; it does not use subsequent returns.
4. For the return-preserving candidate, stock exposure is min(1, 0.25 / estimated basket volatility, trend cap). The trend cap is 1 above QQQ's trailing 200-session average and 0.5 below it. For the higher-Sharpe candidate, exposure is min(1, 0.20 / estimated basket volatility), with no trend rule.
5. Keep remaining capital in zero-interest cash, matching the existing accounting. No borrowing, options, shorts, or leverage. Weekly decisions use the prior close's risk data and the current 09:15-cutoff AH+PM forecast; orders freeze at 09:20 and fill at the regular close.
6. Retain parent share sizing, modeled costs, 1% ADV caps, cancellations, corporate actions, cash scaling, and terminal liquidation. Targets are not hard limits on realized volatility or drawdown; weights drift between fills.'''
    ci=uncertainty[uncertainty.recipe==BALANCED].iloc[0]
    sections=[
        '# Drawdown-control research — October 26, 2026','',
        '**Six of 305 frozen recipes met the historical return/risk screen.** The return-preserving candidate combines five-issuer selection, a 25% basket-volatility target and a half-exposure QQQ/MA200 cap. It retains 31.20% CAGR in the original seed ensemble while improving Sharpe to 1.169 and reducing maximum drawdown to 21.68%. Across all ten seeds it returns 28.70% CAGR / 1.085 Sharpe / 22.98% maximum drawdown.','',
        '**These are selected 2022-2025 development results, not untouched confirmation.** No model was retrained, no 2026 returns were opened, and no orders or automations were created. The existing workbook and frozen studies are unchanged.','',
        '## Matched historical comparison','',metrics_table(overview),'',
        '![Matched wealth and drawdown comparisons](figures/risk_control_comparison.png)','',
        'Original-five and ten-seed accounts are separate rank ensembles on the same history. Compare each new account with its matching full-exposure baseline. The higher-return candidate above has the highest ten-seed CAGR among the six screen passes; the 20% target candidate has the highest ten-seed Sharpe across the grid. These reporting preferences were selected retrospectively.','',
        '## What the two leading rules do','',rules,'',
        'Example: if the selected basket has 40% estimated annualized volatility, the 25% target permits 62.5% stock exposure in a healthy trend. Below QQQ MA200, the exposure cap lowers that to 50%. A 20% target without a trend filter permits 50%. These are example arithmetic, not additional fitted rules.','',
        'The return-preserving account averages approximately 74.8% stock exposure across the ten-seed path; the 20% target account averages 66.6%. Thus the improvement includes materially less time-weighted market exposure, with lower upside in strong markets.','',
        '## What was actually tested','',
        'The protocol enumerated **305 recipes and 915 base-cost accounts before new financial metrics**. Four allocation families crossed 76 risk rules, plus a top-ten control. The search included unchanged top five, five issuers, issuer inverse-volatility weights, soft-weighted top eight issuers, fixed cash fractions, QQQ moving averages of 100/150/200/250 sessions, multi-trend scaling, 20/60-session basket and QQQ volatility targets, QQQ drawdown thresholds, and trend/volatility combinations. Weekly and daily risk schedules were both tested. Daily rules used a 10-percentage-point target-exposure band.','',
        'All original five, new five and ten-seed groups were evaluated. The screen required original-five and ten-seed Sharpe at least 0.9194653, new-five Sharpe at least its 0.6330 baseline, drawdown at most 25% in every group, and ten-seed CAGR at least 23.9%. Daily overlays did not outperform the selected weekly rules. No second batch was needed after the initial grid met the screen.','',
        'The nine-account-recipe shortlist, including baseline, was also evaluated across all three groups at 2x and 4x costs: **54 additional unique higher-cost accounts**, beyond the 915 base accounts. Every failed variant remains in [metrics.csv](metrics.csv) and [summary.csv](summary.csv).','',
        '![Every risk-control recipe](figures/all_variants_tradeoff.png)','',
        '## Seed sensitivity','',metrics_table(seed_table),'',
        '**The new-five half remains much weaker:** the two leaders have approximately 0.643 and 0.677 Sharpe, respectively. They improve base-cost Sharpe and drawdown relative to that half\'s baseline, but reduce its CAGR from 20.92% to approximately 16.32% and 15.77%. This is not uniform high-Sharpe replication across seeds.','',
        '## Component controls: ten seeds','',
        table(ablation,[('recipe','Recipe',str),('sharpe','Sharpe',lambda x:f'{x:.3f}'),('annualized_return','CAGR',lambda x:f'{x:.2%}'),('maximum_drawdown','Max drawdown',lambda x:f'{x:.2%}')]),'',
        'Issuer selection alone does not reduce the 2022 maximum drawdown. Constant 75% exposure reduces drawdown but lowers Sharpe. Basket-volatility scaling supplies most of the historical drawdown improvement. Adding the trend cap to a 20% target has little additional ten-seed Sharpe benefit, while a 25% target plus trend cap retains more return. These component controls are exploratory associations on one history, not causal proof or future guarantees.','',
        '## Cost stress: ten seeds','',
        table(stress_table,[('label','Variant',str),('cost_multiplier','Cost multiplier',str),('sharpe','Sharpe',lambda x:f'{x:.3f}'),('annualized_return','CAGR',lambda x:f'{x:.2%}'),('maximum_drawdown','Max drawdown',lambda x:f'{x:.2%}')]),'',
        'The two leaders remain ahead of the matched ten-seed baseline in Sharpe and drawdown at 2x and 4x costs, although their CAGR is lower. At 4x costs their Sharpe falls below the original 0.919 headline. The new-five half does not outperform its matched baseline in every cost-stress comparison. Higher fees are not a full market-impact, tax, or execution-capacity test.','',
        '## Annual net returns: ten seeds','',
        table(annual_table,[('recipe','Recipe',str),('year','Year',str),('cumulative_return','Calendar-year return',lambda x:f'{x:.2%}'),('maximum_drawdown','Within-year drawdown',lambda x:f'{x:.2%}')]),'',
        'Both leading controls reduce the 2022 loss, but also surrender part of the gains in 2023-2025. The original maximum drawdown and much of the opportunity to improve it are concentrated in one bear-market episode. [Subperiod and leave-one-year-out descriptive sensitivity](subperiod_sensitivity.csv) does not turn the already inspected later years into untouched data.','',
        '## Uncertainty and what cannot be concluded','',
        f'For the return-preserving ten-seed candidate, the pointwise 95% Sharpe interval is [{ci.sr_low:.3f}, {ci.sr_high:.3f}]. Its paired Sharpe-difference interval versus its baseline is [{ci.paired_sr_difference_low:.3f}, {ci.paired_sr_difference_high:.3f}], which includes zero. These use 10,000 circular-block draws of 20 sessions.','',
        'The uncertainty conditions on fitted paths and excludes selection across these 305 recipes, the earlier 346-strategy search, retraining uncertainty and vendor revisions. No claim of statistically established Sharpe superiority follows from the grid. Search-adjusted evidence is weaker than the displayed pointwise intervals. A 25% historical screen is not a guaranteed future loss limit. The 16.5% drawdown variants at a 15% volatility target retain only about 21% ten-seed CAGR and miss the return/new-seed screen.','',
        'The universe remains historical NASDAQ-100 membership and the inherited economic-price, sparse AH/PM, publication-latency, corporate-action, cold-start, fill and capacity limitations remain. Idle cash earns zero; neither Treasury yields nor taxes were added. The confirmation gate remains unactivated. Freeze a complete candidate before collecting any prospective confirmation path.','',
        '## Verification and reproducibility','',
        f'Five meaningful unit tests passed: issuer/tie invariance, capped allocation, no future returns in volatility, parity with parent settlement/cash/cost accounting, and next-close daily-risk execution. All six real-data full-exposure controls matched the saved parent daily paths. The independent auditor recalculated all 915 account metrics, independently rebuilt all 15 allocation/seed-group combinations, and replayed {audit["independent_replays"]} accounts, including all shortlist seed/cost accounts and one daily-risk account. Maximum NAV difference was ${audit["maximum_nav_difference_dollars"]:.10f}.','',
        'Run from the repository root with local market inputs and saved checkpoints available:','',
        '```powershell\npython -m hypershift test test_ndx_drawdown_engine -v\npython -m hypershift run run_ndx_drawdown_research\npython -m hypershift run audit_ndx_drawdown_research\npython -m hypershift run publish_ndx_drawdown_research\n```','',
        'The runner refuses changed frozen recipes or inputs. Use a new declared batch for further experiments. Full paths and selected ledgers remain under `runs/ndx_drawdown_20261026/batch1/`; this directory supplies portable summaries and focused curves.','',
        '[Protocol](PROTOCOL.json) · [Freeze](FREEZE.json) · [Independent audit](AUDIT.json) · [Complete batch](COMPLETE.json) · [All metrics](metrics.csv) · [All recipe summaries](summary.csv) · [Cost stress](cost_stress.csv) · [Annual results](annual.csv) · [Conditional uncertainty](uncertainty.csv) · [Focused daily curves](comparison_curves.csv)','',
        'Research motivation: [Moreira and Muir, Volatility-Managed Portfolios](https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf). Their broader results motivate a test; they do not validate this selected model or its targets.',''
    ]
    (DOC/'REPORT.md').write_text('\n'.join(sections),encoding='utf8',newline='\n')
    sources=[ROOT/'research/core/ndx_drawdown_engine.py',ROOT/'research/analysis/run_ndx_drawdown_research.py',
             ROOT/'research/evaluation/audit_ndx_drawdown_research.py',ROOT/'research/reporting/publish_ndx_drawdown_research.py',
             ROOT/'tests/test_ndx_drawdown_engine.py']
    publication=dict(status='audited_exploratory_publication',date='2026-10-26',
                     risk_control_variants=305,base_accounts=915,additional_cost_accounts=54,
                     target_screen_passes=int(summary.target_met.sum()),
                     source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):study.sha(p) for p in sources},
                     artifact_sha256={str(p.relative_to(DOC)).replace('\\','/'):study.sha(p) for p in DOC.rglob('*') if p.is_file() and p.name!='PUBLICATION.json'})
    study.write(DOC/'PUBLICATION.json',publication)
    print('Published audited drawdown-control comparison:',DOC.relative_to(ROOT))


if __name__=='__main__':main()
