"""Publish complete feature-forest and training-prior experiments."""
import argparse
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

import run_ndx_tabular_alpha as tree
import run_ndx_prior_robustness as prior
from research_engine import performance
from run_ndx_retention_replication import sha,write


def publish(which):
    study=tree if which=='tree' else prior
    plan=study.verify();run=study.RUN;root=study.ROOT;doc=root/'docs'/study.NAME;doc.mkdir(exist_ok=True)
    cases=plan['architectures'];frame=pd.read_csv(run/'METRICS.csv');summary=pd.read_csv(run/'CONSISTENCY.csv')
    selection=json.loads((run/'SELECTION.json').read_text())
    source_hashes={}
    for case in cases:
        complete=run/case/'COMPLETE.json';lock=json.loads(complete.read_text())
        for name,expected in lock['output_sha256'].items():assert sha(run/case/name)==expected,name
        source_hashes[complete.relative_to(root).as_posix()]=sha(complete)
    for name in ('PROTOCOL.json','FREEZE.json','ENVIRONMENT.json','METRICS.csv','CONSISTENCY.csv','SELECTION.json',
                 'ALL_METRICS.csv','ACCOUNT_AUDIT.json','CONTROL_METRICS.csv','TRAINING_PRIORS.csv'):
        if (run/name).exists():shutil.copyfile(run/name,doc/name)
    if which=='tree':
        labels={'tree':'Shared ExtraTrees','tree_neural50':'50:50 tree / neural rank blend'}
        controls=pd.read_csv(run/'ALL_METRICS.csv');controls=controls[controls.kind=='control']
        count=282;title='Long-history feature models: all seeds retained'
    else:
        labels={'prior13':'13-week training prior + neural','prior26':'26-week training prior + neural',
                'prior52':'52-week training prior + neural','prior_full':'Full-history training prior + neural'}
        controls=pd.read_csv(run/'CONTROL_METRICS.csv');count=264;title='Training-return priors: fixed shrinkage experiment'
    controls.to_csv(doc/'CONTROLS.csv',index=False)
    base_run=tree.parent.RUN/'original';baseline=np.load(base_run/'PATHS.npz',allow_pickle=False)
    saved={case:np.load(run/case/'PATHS.npz',allow_pickle=False) for case in cases}
    dates=pd.DatetimeIndex(baseline['dates']);x=dates.insert(0,dates[0]-pd.offsets.BDay(1))
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True,sharey='row',gridspec_kw={'height_ratios':[2,1]})
    handles={};colors=['#315fc3','#a46805','#ad4c85','#7957b5']
    for col,group in enumerate(('ten_seed','fifteen_seed')):
        key=tree.parent.key_for('original','retain5','cohort_targets',group,0.,1,.03)
        paths=[('Prior neural blend / averaged cohort targets',baseline[key],'#168878','--')]
        for case,color in zip(cases,colors):
            key=tree.parent.key_for(case,'retain5','cohort_targets',group,0.,1,.03)
            path=saved[case][key];row=frame[frame.id==key].iloc[0]
            for field,value in performance(path).items():np.testing.assert_allclose(value,row[field],rtol=1e-10,atol=1e-12)
            paths.append((labels[case],path,color,'-'))
        for label,path,color,style in paths:
            equity=np.r_[1.,np.cumprod(1+path)];dd=equity/np.maximum.accumulate(equity)-1
            handles[label],=axes[0,col].plot(x,100*equity,color=color,ls=style,lw=1.7)
            axes[1,col].plot(x,dd,color=color,ls=style,lw=1.15)
        axes[0,col].set_title('Combined '+('ten' if col==0 else 'fifteen')+' seeds',loc='left',weight='bold')
        axes[0,col].yaxis.set_major_formatter(StrMethodFormatter('${x:.0f}'))
        axes[1,col].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        for ax in axes[:,col]:
            ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False)
            ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_xlim(x[0],dates[-1])
    axes[0,0].set_ylabel('Net wealth from $100');axes[1,0].set_ylabel('Drawdown')
    fig.suptitle(title,x=.07,y=.98,ha='left',weight='bold',fontsize=17)
    fig.text(.07,.925,'Retention five; cohort-target averaging; no trading band; base fees and 3% short borrowing\n2022–2025 repeatedly inspected development history; no market holdout',va='top',fontsize=10,color='#536174')
    fig.legend(handles.values(),handles.keys(),loc='lower left',bbox_to_anchor=(.065,.05),ncol=2,frameon=False,fontsize=9)
    fig.text(.07,.02,'Fixed representative curves; every allocation, aggregation and cost variant remains in the accompanying tables.',fontsize=9,color='#536174')
    fig.subplots_adjust(left=.07,right=.98,top=.83,bottom=.18,hspace=.13,wspace=.12)
    fig.savefig(doc/'model_comparison.png',dpi=160,facecolor='white');plt.close(fig)
    rows=[f'# {title} — October 30, 2026','',
        f'The fixed **{count}-account** arm has **{selection["primary_passes"]}** passing improvement recipes and **{selection["all_groups_above_1_5"]}** with SR>1.5 in all five predefined ensembles. '+
        ('All **60 forest fits and four deterministic Ridge fits** locked before financial evaluation.' if which=='tree' else 'No new neural fits; every training prior uses only its fold\'s observed training labels.'),'',
        '![Model comparison](model_comparison.png)','',
        'Fixed chart/table representatives: retention five, averaged cohort targets, no band, base fees and3%borrow. These are not a substitute for the full grid.','',
        '| Model | Original five SR | Second five SR | Third five SR | Ten SR | Fifteen SR |',
        '|---|---:|---:|---:|---:|---:|']
    for case in cases:
        block=frame[(frame.arm==case)&(frame.allocation=='retain5')&(frame.aggregation=='cohort_targets')&(frame.band==0)&(frame.cost==1)&(frame.borrow==.03)]
        rows.append('| '+labels[case]+' | '+' | '.join(f'{block[block.group==g].iloc[0].sharpe:.3f}' for g in plan['groups'])+' |')
    rows+=['','## Seed-free controls','', '| Control | Allocation | Band | Sharpe | CAGR | Max DD |','|---|---|---:|---:|---:|---:|']
    for r in controls[(controls.cost==1)&(controls.borrow==.03)].itertuples():
        name='Ridge' if which=='tree' else r.case
        band=r.band if hasattr(r,'band') else 0.
        rows.append(f'| {name} | {r.allocation} | {band:.1%} | {r.sharpe:.3f} | {r.annualized_return:.2%} | {r.maximum_drawdown:.2%} |')
    rows+=['','Deterministic controls are reported once, not copied into artificial seed ensembles. Neural blended groups overlap and share the same historical outcomes.','', '## Design and interpretation','']
    if which=='tree':
        rows+=['A shared ExtraTrees regressor uses512trees,depth6,minimumleaf64,75%feature sampling and80%row bootstrap per tree. All15seed values are fixed from the earlier protocol. The130features combine26causal price/volatility/liquidity/beta styles, eligible cross-sectional feature ranks, existing train-only daily/AH/PM summaries and QQQ context. There are no stock-ID or ticker features. Labels are within-date training-return zscores clipped[-3,3], with equal date weights. Training-only imputation/scaling precedes both forest and Ridge(alpha100). No validation or financial result tunes these parameters. Predictions accumulate trees serially for reproducibility.','',
            'A second fixed model blends50%forest ranks with25%THINK and25%dynamic ranks. The existing neural checkpoints remain untouched. The report includes all240ensemble accounts,30individual-seed base accounts and12Ridge control cases. Stronger seed averaging cannot create a missing predictive signal; financial results must be evaluated separately from validation RankIC.']
    else:
        rows+=['The four priors use training-return percentile ranks with13/26/52decision-week exponential half-lives or full training history, plus20neutral0.5pseudo-observations per stock. Every annual prior ends before its prior-year validation window; no validation/test outcome estimates the stock prior. Unseen/unsupported stocks are neutral. All four priors are tried without handpicking names.','',
            'Neural blends use50%eligible percentile rank of the training prior plus25%meanTHINKseed ranks and25%mean dynamicseed ranks. The24prior-only controls retain the raw mean-rank score scale used in the older deterministic control. Retention is sensitive to this scale; issuer selection is order-based. All six full-history control cases reproduce the previous completed source metrics.','',
            'The arm was frozen while five unused neural seeds trained, before any unused-seed account was scored. Any source-selected candidate must be frozen before their financial results are examined. This deliberate stock-specific historical preference is not a model that generalizes to unseen stock identities.']
    rows+=['','All account paths match independent dollar-position accounting. Frozen source/input hashes, chronology, gross budgets, eligibility and terminal liquidation remain inherited and checked.','',
        'These are disclosed additional development searches after the earlier failed shared/family experiments. No unseen market evaluation or multiple-search-adjusted significance is claimed. Seed consistency alone cannot establish future Sharpe.','',
        '[Every ensemble account](METRICS.csv) · [Every recipe/cost summary](CONSISTENCY.csv) · [Seed-free controls](CONTROLS.csv) · [Frozen protocol](PROTOCOL.json) · [Selection](SELECTION.json)','',
        'The [current overview](../ndx_seed_robustness_20261030/REPORT.md) adds the separately frozen reserved-seed confirmation, full search history and final model comparison.','']
    (doc/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    for source in saved.values():source.close()
    baseline.close()
    write(doc/'MANIFEST.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(root/'research/reporting/report_ndx_feature_prior_studies.py'),source_sha256=source_hashes,
        output_sha256={p.name:sha(p) for p in doc.iterdir() if p.is_file() and p.name!='MANIFEST.json'}))
    print('Published',doc,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',choices=['tree','prior'],required=True)
    publish(p.parse_args().study)
