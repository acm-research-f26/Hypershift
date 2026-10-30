"""Publish all family-isolation results and matched saved portfolio curves."""
from datetime import datetime,timezone
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter,StrMethodFormatter
import numpy as np
import pandas as pd

import run_ndx_family_robustness as study
from run_ndx_retention_replication import sha,write


def main():
    plan=study.verify();run=study.RUN;doc=study.ROOT/'docs'/study.NAME;doc.mkdir(exist_ok=True)
    for name in ('PROTOCOL.json','FREEZE.json','METRICS.csv','CONSISTENCY.csv','SELECTION.json','FAMILY_WEIGHTS.json','VALIDATION.csv'):
        shutil.copyfile(run/name,doc/name)
    metrics=pd.read_csv(run/'METRICS.csv');selection=json.loads((run/'SELECTION.json').read_text())
    baseline=pd.read_csv(study.study.RUN/'original/METRICS.csv')
    labels={'original_think':'THINK + stock-ID mixer','dynamic':'Dynamic only','shared_think':'Shared THINK only',
            'validation_original':'Validation-weighted original blend','validation_shared':'Validation-weighted shared blend'}
    sources={case:np.load(run/case/'PATHS.npz',allow_pickle=False) for case in study.CASES}
    prior=np.load(study.study.RUN/'original/PATHS.npz',allow_pickle=False)
    dates=pd.DatetimeIndex(prior['dates']);x=dates.insert(0,dates[0]-pd.offsets.BDay(1))
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True,sharey='row',gridspec_kw={'height_ratios':[2,1]})
    handles={}
    for col,group in enumerate(('ten_seed','fifteen_seed')):
        paths=[('Fixed 50:50 original blend',prior[study.study.key_for('original','retain5','cohort_targets',group,0.,1,.03)],'#168878','-')]
        for case,color in zip(study.CASES,['#3264c8','#a46e06','#b25498','#b64a2d','#7b7d82']):
            key=study.study.key_for(case,'retain5','cohort_targets',group,0.,1,.03)
            paths.append((labels[case],sources[case][key],color,'--' if case.startswith('validation') else '-'))
        for label,path,color,style in paths:
            eq=np.r_[1.,np.cumprod(1+path)];dd=eq/np.maximum.accumulate(eq)-1
            handles[label],=axes[0,col].plot(x,100*eq,color=color,ls=style,lw=1.6)
            axes[1,col].plot(x,dd,color=color,ls=style,lw=1.1)
        axes[0,col].set_title('Combined '+('ten' if col==0 else 'fifteen')+' seeds',loc='left',weight='bold')
        axes[0,col].yaxis.set_major_formatter(StrMethodFormatter('${x:.0f}'))
        axes[1,col].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        for ax in axes[:,col]:
            ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False)
            ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_xlim(x[0],dates[-1])
    axes[0,0].set_ylabel('Net wealth from $100');axes[1,0].set_ylabel('Drawdown')
    fig.suptitle('Family isolation: the fixed blend remains stronger',x=.07,y=.98,ha='left',weight='bold',fontsize=17)
    fig.text(.07,.925,'Retention five; cohort target averaging; no band; base fees and 3% short borrowing\n2022–2025 development history; validation weights use only the purged prior-year selection half',va='top',fontsize=10,color='#536174')
    fig.legend(handles.values(),handles.keys(),loc='lower left',bbox_to_anchor=(.065,.045),ncol=3,frameon=False,fontsize=9)
    fig.text(.07,.02,'Every family, validation blend and cost case is reported; no untouched-market confirmation is claimed.',fontsize=9,color='#536174')
    fig.subplots_adjust(left=.07,right=.98,top=.83,bottom=.18,hspace=.13,wspace=.12)
    fig.savefig(doc/'family_comparison.png',dpi=160,facecolor='white');plt.close(fig)
    rows=['# Model-family isolation — October 30, 2026','',
        f'The fixed **300-account** follow-up has **{selection["primary_passes"]}** passing improvement recipes and **{selection["all_groups_above_1_5"]}** with SR>1.5 in every group. No new models were fitted in this arm.','',
        '![Family comparison](family_comparison.png)','',
        'All rows below use retention five, cohort-target averaging, no band, base fees and3%borrow.','',
        '| Predictor | Original five SR | Second five SR | Third five SR | Ten SR | Fifteen SR |',
        '|---|---:|---:|---:|---:|---:|']
    values=baseline[(baseline.allocation=='retain5')&(baseline.aggregation=='cohort_targets')&(baseline.band==0)&(baseline.cost==1)&(baseline.borrow==.03)]
    rows.append('| Fixed 50:50 original blend | '+' | '.join(f'{values[values.group==g].iloc[0].sharpe:.3f}' for g in plan['groups'])+' |')
    for case in study.CASES:
        values=metrics[(metrics.arm==case)&(metrics.allocation=='retain5')&(metrics.band==0)&(metrics.cost==1)&(metrics.borrow==.03)]
        rows.append('| '+labels[case]+' | '+' | '.join(f'{values[values.group==g].iloc[0].sharpe:.3f}' for g in plan['groups'])+' |')
    rows+=['','Both original families alone underperform their fixed50:50blend in this portfolio comparison. Validation-only weight selection also fails to improve it consistently. This does not show one family is universally inferior: rank combination and top-k portfolio decisions are nonlinear. The shared-THINK failure is preserved rather than excluded.','',
        'Validation weights are selected independently each year from[0,.25,.5,.75,1] using first-half prior-year meanRankIC−0.5quarterSD across all15seeds. Exact ties prefer0.5thenlowerweight. The diagnostic half is never used to choose weights. Every validation-grid result and every allocation/band/cost case is retained. Group results share model-selection weights and market outcomes; they are not independent market trials.','',
        'The follow-up was frozen after the initial540-account study failed its strict goal. This is additional disclosed development search on spent history, not a fresh market test. All300accounts match independent dollar-position replay.','',
        '[All accounts](METRICS.csv) · [All recipe/cost summaries](CONSISTENCY.csv) · [Validation grid](VALIDATION.csv) · [Selected family weights](FAMILY_WEIGHTS.json) · [Protocol](PROTOCOL.json)','']
    (doc/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    for source in sources.values():source.close()
    prior.close()
    write(doc/'MANIFEST.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(study.ROOT/'research/reporting/report_ndx_family_robustness.py'),
        source_sha256={str((run/case/'COMPLETE.json').relative_to(study.ROOT)).replace('\\','/'):sha(run/case/'COMPLETE.json') for case in study.CASES},
        output_sha256={p.name:sha(p) for p in doc.iterdir() if p.is_file() and p.name!='MANIFEST.json'}))
    print('Published',doc,flush=True)


if __name__=='__main__':main()
