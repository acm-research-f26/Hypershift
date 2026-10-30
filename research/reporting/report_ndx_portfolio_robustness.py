"""Publish the complete fixed portfolio/shared-head experiment, without tuning."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

import run_ndx_portfolio_robustness_v2 as study
from run_ndx_retention_replication import sha, write

ROOT=study.ROOT
DOC=ROOT/'docs'/study.NAME
LABELS={'original_five':'Original five','new_five':'Second five','fresh_five':'Third five',
        'ten_seed':'Combined ten','fifteen_seed':'Combined fifteen'}


def charts(plan,selection):
    original=np.load(study.SOURCE/'accounts/PATHS.npz',allow_pickle=False)
    new={arm:np.load(study.RUN/arm/'PATHS.npz',allow_pickle=False) for arm in plan['architectures']}
    dates=pd.DatetimeIndex(original['dates']);x=dates.insert(0,dates[0]-pd.offsets.BDay(1))
    for filename,groups in [('disjoint_comparison.png',plan['groups'][:3]),
                            ('pooled_comparison.png',plan['groups'][3:])]:
        fig,axes=plt.subplots(2,len(groups),figsize=(17,8.5) if len(groups)==3 else (14,8.5),
                              squeeze=False,sharex=True,sharey='row',
                              gridspec_kw={'height_ratios':[2,1]})
        handles={}
        for col,group in enumerate(groups):
            base=f'loss__{group}__issuer5__cap75__c1__b03'
            shared=study.key_for('shared','issuer5','rank_pool',group,0.,1,.03)
            chosen=study.key_for(selection['arm'],selection['allocation'],selection['aggregation'],group,
                                 float(selection['band']),1,.03)
            paths=[('Prior seed-stable issuer baseline',original[base],'#168878','--'),
                   ('Shared THINK + dynamic / issuer baseline',new['shared'][shared],'#9560b4',':'),
                   ('Selected portfolio candidate',new[selection['arm']][chosen],'#265bc1','-')]
            for label,path,color,style in paths:
                equity=np.r_[1.,np.cumprod(1+path)];dd=equity/np.maximum.accumulate(equity)-1
                handles[label],=axes[0,col].plot(x,100*equity,color=color,ls=style,lw=1.8)
                axes[1,col].plot(x,dd,color=color,ls=style,lw=1.25)
            axes[0,col].set_title(LABELS[group],loc='left',weight='bold')
            axes[0,col].yaxis.set_major_formatter(StrMethodFormatter('${x:.0f}'))
            axes[1,col].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            for ax in axes[:,col]:
                ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False)
                ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.set_xlim(x[0],dates[-1])
        axes[0,0].set_ylabel('Net wealth from $100');axes[1,0].set_ylabel('Drawdown')
        name=f'{selection["arm"]} / {selection["allocation"]} / {selection["aggregation"]} / band {selection["band"]:.1%}'
        fig.suptitle('Portfolio aggregation, trading bands and shared stock scoring',x=.07,y=.98,ha='left',fontsize=16,weight='bold')
        fig.text(.07,.925,name+'\n2022–2025 development history; base modeled fees and 3% short borrowing included',fontsize=10,color='#536174',va='top')
        fig.legend(handles.values(),handles.keys(),loc='lower left',bbox_to_anchor=(.065,.055),frameon=False,ncol=3,fontsize=10)
        fig.text(.07,.025,'Outcome-selected candidate; every recipe is retained. Seed consistency does not establish unseen-market generalization.',fontsize=9,color='#536174')
        fig.subplots_adjust(left=.07,right=.98,top=.83,bottom=.16,hspace=.12,wspace=.12)
        fig.savefig(DOC/filename,dpi=160,facecolor='white');plt.close(fig)
    original.close()
    for value in new.values():value.close()


def main():
    plan=study.verify();selection=json.loads((study.RUN/'SELECTION.json').read_text())
    DOC.mkdir(parents=True,exist_ok=True)
    for name in ('METRICS.csv','CONSISTENCY.csv','SELECTION.json','PROTOCOL.json','FREEZE.json','ENVIRONMENT.json'):
        shutil.copyfile(study.RUN/name,DOC/name)
    for name in ('EXECUTION_AMENDMENT.json','RESUME_CHECK.json'):
        if (study.RUN/name).exists():shutil.copyfile(study.RUN/name,DOC/name)
    metrics=pd.read_csv(study.RUN/'METRICS.csv');summary=pd.read_csv(study.RUN/'CONSISTENCY.csv')
    base=summary[(summary.cost==1)&(summary.borrow==.03)]
    choice=selection['selected']
    if choice is None:choice=base.sort_values('min_all_sharpe',ascending=False).iloc[0].to_dict()
    charts(plan,choice)
    confirmation=study.RUN/'confirmation'
    result=None
    if (confirmation/'RESULT.json').exists():
        result=json.loads((confirmation/'RESULT.json').read_text())
        for name in ('RESULT.json','PROTOCOL.json','FREEZE.json','METRICS.csv','ACCOUNT_CHECKS.csv'):
            shutil.copyfile(confirmation/name,DOC/('CONFIRMATION_'+name))
        if (confirmation/'POOLING_CONFIRMATION.json').exists():
            shutil.copyfile(confirmation/'POOLING_CONFIRMATION.json',DOC/'POOLING_CONFIRMATION.json')
    old=pd.read_csv(study.SOURCE/'METRICS.csv')
    rows=['# Portfolio robustness and shared THINK — October 30, 2026','',
        f'Completed **60 shared-THINK fits and 540 fixed accounts**. **{selection["primary_passes"]}** recipes pass the predefined historical improvement/stability goal; **{selection["all_groups_above_1_5"]}** exceed SR>1.5 in all five ensembles.','',
        'This is development on repeatedly inspected 2022–2025 data. A new seed group can confirm initialization robustness, not out-of-time market performance.','',
        f'Displayed candidate: **{choice["arm"]} / {choice["allocation"]} / {choice["aggregation"]} / {choice["band"]:.1%} NAV band**. '+
        ('It passes the historical goal and was frozen before reserved-seed confirmation.' if selection['selected'] is not None else 'No candidate passed; this is only the highest observed minimum-SR row, not a success claim.'),'',
        '![Separate cohorts](disjoint_comparison.png)','','![Combined ensembles](pooled_comparison.png)','',
        '| Ensemble | Prior stable issuer SR | Shared THINK + dynamic SR | Candidate SR | Candidate CAGR | Candidate max DD | Candidate vol |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group in plan['groups']:
        prior=old[(old.method=='loss')&(old.allocation=='issuer5')&(old.cap==.75)&(old.cost==1)&(old.borrow==.03)&(old.group==group)].iloc[0]
        shared=metrics[metrics.id==study.key_for('shared','issuer5','rank_pool',group,0.,1,.03)].iloc[0]
        candidate=metrics[metrics.id==study.key_for(choice['arm'],choice['allocation'],choice['aggregation'],group,float(choice['band']),1,.03)].iloc[0]
        rows.append(f'| {LABELS[group]} | {prior.sharpe:.3f} | {shared.sharpe:.3f} | {candidate.sharpe:.3f} | {candidate.annualized_return:.2%} | {candidate.maximum_drawdown:.2%} | {candidate.annualized_volatility:.2%} |')
    rows += ['','## Reserved-seed confirmation','']
    if result is None:rows.append('No completed reserved-seed confirmation is available. Historical selection alone is not confirmation.')
    else:
        rows += [f'Predefined confirmation passed: **{result["passed"]}**. Four disjoint cohort SR spread: **{result["disjoint_gap"]:.3f}**.',
                 '','[Exact frozen recipe and seeds](CONFIRMATION_PROTOCOL.json) · [All confirmation accounts](CONFIRMATION_METRICS.csv) · [Result](CONFIRMATION_RESULT.json).',
                 '','This separate follow-up froze the original no-band retention/cohort-target proposal before new-seed fitting; it is not the displayed1%band development representative. The expanded twenty-seed check uses every seed without filtering. No choice changes after these outcomes. The pooling-specific confirmation also fails: reserved-fiveSR0.854 and four-cohortspread0.491. [Pooling result](POOLING_CONFIRMATION.json). The later prior-regularized candidate has its own separately frozen confirmation in the [current overview](../ndx_seed_robustness_20261030/REPORT.md).']
    rows += ['','## What changed and what is controlled','',
        '- **Portfolio targets:** compare rank pooling with averaged individually risk-sized seed portfolios and averaged consecutive five-seed-cohort portfolios. The final combined account nets orders and pays its own fees. Return-series averaging is never substituted for execution. No leverage is added after averaging; exposure, volatility and concentration can differ, and are reported.',
        '- **Actual-holdings bands:** fixed 0%, 1%, 2.5% NAV per-component thresholds use preceding-close holdings/NAV and marks. Ineligible exits and gross budget take priority. Kept proposals exceeding gross1 are scaled. ADV/funding limits, fill prices, costs, events and terminal liquidation remain inherited.',
        '- **Shared scorer:** remove THINK\'s stock-ID dense residual mixer, retaining its existing shared THINK stock output and AH/PM branch. Construct the matched model first so every retained initialized tensor exactly matches the corresponding old seed. Same optimizer, loss, dates, seed list, 12epochs and first-half validation loss selection. Reuse the original dynamic family unchanged. This is a controlled architecture ablation; it is not a new paper replication.',
        '- **Every case retained:** two architectures × two allocations × three aggregation rules × three bands × five groups × three cost regimes =540 accounts. Cohort averaging equals rank pooling for a single five-seed cohort, so some paths duplicate; they are not independent evidence.',
        '- **Improvement gate:** every base-cost group must beat the prior loss/issuer5/cap75 SR by>=0.01; minimum across all5>1.10; disjoint-five spread<=0.15; stressed-cost worst-group SR must not fall below that prior baseline. Separately report the harder all-group1.5 target.',
        '', '## Verification and limitations','',
        'Source/input hashes are frozen before scoring; all zero-band accounts match the original signed engine. Original rank-pool/no-band targets and daily paths also reproduce the completed source accounts. Every account passes independent dollar-position replay using the actual executed decision targets. Eligibility, gross budgets and final liquidation are checked.',
        '', 'A first attempt failed its baseline-parity gate before accepting an account: averaging per-seed family blends first changed floating-point ties versus the original family-first averaging. The failed frozen source/run is preserved; v2 restores exact family-first association. No result-driven rule change followed from that failure.',
        '', 'A transient Windows progress-file replacement error interrupted one worker after its epoch8 checkpoint. The execution amendment retries that atomic file replacement and preserves all scientific source and complete-prefix hashes. An actual replay from epoch7 to8 matches model parameters and validation forecasts exactly (zero maximum absolute error). Every seed remains included.',
        '', 'The same market years have been repeatedly inspected. Historical recipe selection remains exposed to multiple testing; a seed reserve only checks initialization. No untouched-period or live-execution evidence is supplied. Borrow, impact, vendor vintages and broker margin assumptions remain inherited. Actual-holdings bands can leave small positions in eligible stocks outside the new top five.',
        '', '[All540accounts](METRICS.csv) · [All108recipe/cost summaries](CONSISTENCY.csv) · [Frozen protocol](PROTOCOL.json) · [Selection](SELECTION.json)',
        '', '## Reproduction','', '```powershell',
        '.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_portfolio_robustness_v2 --stage freeze',
        '.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_portfolio_robustness_v2 --stage accounts --arm original',
        '.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_portfolio_robustness_v2 --stage train',
        '.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_portfolio_robustness_v2 --stage accounts --arm shared',
        '.\\.venv\\Scripts\\python.exe -m hypershift run run_ndx_portfolio_robustness_v2 --stage summarize',
        '# If a candidate qualifies: confirmation freeze, train, evaluate, then publish.',
        '.\\.venv\\Scripts\\python.exe -m hypershift run report_ndx_portfolio_robustness',
        '```','', 'Existing runs are preserved; these commands expect an absent run for the freeze stage.','']
    (DOC/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    write(DOC/'MANIFEST.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(ROOT/'research/reporting/report_ndx_portfolio_robustness.py'),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
            [study.RUN/'FITS_COMPLETE.json',study.RUN/'original/COMPLETE.json',study.RUN/'shared/COMPLETE.json',
             study.RUN/'SELECTION.json',study.RUN/'METRICS.csv',study.RUN/'CONSISTENCY.csv']},
        output_sha256={p.name:sha(p) for p in sorted(DOC.iterdir()) if p.is_file() and p.name!='MANIFEST.json'}))
    print('Published',DOC,flush=True)


if __name__=='__main__':main()
