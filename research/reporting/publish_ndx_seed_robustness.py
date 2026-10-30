"""Audited final overview of the October30 seed-robustness experiments."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter,StrMethodFormatter
import numpy as np
import pandas as pd

import run_ndx_portfolio_robustness_v2 as portfolio
import run_ndx_family_robustness as family
import run_ndx_tabular_alpha as tree
import run_ndx_prior_robustness as prior
import run_ndx_beta_robustness as beta
import confirm_ndx_portfolio_robustness as reserve
import confirm_ndx_rank_prior as confirmation
from research_engine import performance
from run_ndx_retention_replication import sha,write

ROOT=portfolio.ROOT
DOC=ROOT/'docs/ndx_seed_robustness_20261030'
GROUPS=['original_five','new_five','fresh_five','unused_five','ten_seed','fifteen_seed','expanded_twenty']
NAMES=['Original five','Second five','Third five','Reserved five','Combined ten','Combined fifteen','Combined twenty']
LABELS=dict(zip(GROUPS,NAMES))
MODELS={'baseline':('Previous stable issuer model','#168878','--'),
        'cohort':('Neural retention / cohort targets','#b27d16',':'),
        'prior':('Prior-regularized issuer ensemble','#285fc8','-')}


def load(run,sub,source_hashes):
    complete=run/'COMPLETE.json';lock=json.loads(complete.read_text())['output_sha256']
    arrays=run/sub/'PATHS.npz';metrics=run/'METRICS.csv'
    for path in (arrays,metrics):
        assert sha(path)==lock[path.relative_to(run).as_posix()],path
        source_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    source_hashes[complete.relative_to(ROOT).as_posix()]=sha(complete)
    with np.load(arrays,allow_pickle=False) as saved:paths={k:saved[k].copy() for k in saved.files}
    return paths,pd.read_csv(metrics).set_index('id')


def comparison():
    source_hashes={}
    sources={
        'base':load(portfolio.SOURCE,'accounts',source_hashes),
        'cohort':load(portfolio.RUN/'original','',source_hashes),
        'prior':load(prior.RUN/'prior26','',source_hashes),
        'reserve':load(reserve.CONF,'',source_hashes),
        'confirmed_prior':load(confirmation.RUN,'',source_hashes)}
    dates=sources['base'][0]['dates']
    assert len(dates)==1003
    for paths,_ in sources.values():np.testing.assert_array_equal(paths['dates'],dates)
    rows=[];output={}
    for group in GROUPS:
        for model in MODELS:
            for cost,borrow in ((1,.03),(2,.03),(2,.06)):
                if group in ('unused_five','expanded_twenty'):
                    source='confirmed_prior' if model=='prior' else 'reserve'
                    strategy={'baseline':'baseline_issuer','cohort':'candidate','prior':'prior_candidate'}[model]
                    key=f'{group}__{strategy}__c{cost}__b{int(borrow*100):02d}'
                elif model=='baseline':source='base';key=f'loss__{group}__issuer5__cap75__c{cost}__b{int(borrow*100):02d}'
                else:
                    source='cohort' if model=='cohort' else 'prior'
                    key=portfolio.key_for('original' if model=='cohort' else 'prior26',
                        'retain5' if model=='cohort' else 'issuer5','cohort_targets',group,0.,cost,borrow)
                paths,metrics=sources[source];daily=paths[key];saved=metrics.loc[key]
                measured=performance(daily)
                for field,value in measured.items():np.testing.assert_allclose(value,saved[field],rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(1e6*np.prod(1+daily),saved.terminal_nav,rtol=1e-11,atol=1e-6)
                output[group,model,cost,borrow]=daily
                rows.append(dict(group=group,model=model,cost=cost,borrow=borrow,source=source,id=key,
                    **measured,cost_dollars=float(saved.cost_dollars),borrow_dollars=float(saved.borrow_dollars),
                    turnover_annual=float(saved.turnover_annual),mean_short=float(saved.mean_short)))
    frame=pd.DataFrame(rows);assert len(frame)==63
    return pd.DatetimeIndex(dates),frame,output,source_hashes


def figures(dates,paths):
    x=dates.insert(0,dates[0]-pd.offsets.BDay(1))
    for filename,groups in [('disjoint_ensembles.png',GROUPS[:4]),('pooled_ensembles.png',GROUPS[4:])]:
        fig,axes=plt.subplots(2,len(groups),figsize=(19,9) if len(groups)==4 else (17,9),
            squeeze=False,sharex=True,sharey='row',gridspec_kw={'height_ratios':[2,1]})
        handles={}
        for col,group in enumerate(groups):
            for model,(label,color,style) in MODELS.items():
                daily=paths[group,model,1,.03];equity=np.r_[1.,np.cumprod(1+daily)]
                dd=equity/np.maximum.accumulate(equity)-1
                handles[model],=axes[0,col].plot(x,100*equity,color=color,ls=style,lw=1.9)
                axes[1,col].plot(x,dd,color=color,ls=style,lw=1.25)
            axes[0,col].set_title(LABELS[group]+(' *' if group=='unused_five' else ''),loc='left',weight='bold')
            axes[0,col].yaxis.set_major_formatter(StrMethodFormatter('${x:.0f}'))
            axes[1,col].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            for ax in axes[:,col]:
                ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False)
                ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.set_xlim(x[0],dates[-1]);ax.tick_params(labelsize=9)
        axes[0,0].set_ylabel('Net wealth from $100');axes[1,0].set_ylabel('Drawdown from prior peak')
        fig.suptitle('Higher Sharpe with a fixed training-history prior',x=.06,y=.98,ha='left',fontsize=18,weight='bold')
        fig.text(.06,.925,'2022–2025 • 1,003 sessions • same modeled fees, 3% borrowing and 20% stock-volatility target\n* Prior mixture fixed before scoring the reserved five; all cohorts share the same market history.',fontsize=10,va='top',color='#536174')
        fig.legend([handles[k] for k in MODELS],[MODELS[k][0] for k in MODELS],loc='lower left',
                   bbox_to_anchor=(.055,.055),frameon=False,ncol=3,fontsize=10)
        fig.text(.06,.025,'Initialization confirmation only. Every model/portfolio search and failed variant is retained; SR>1.5 across ensembles remains unmet.',fontsize=9,color='#536174')
        fig.subplots_adjust(left=.06,right=.985,top=.83,bottom=.16,hspace=.13,wspace=.12)
        fig.savefig(DOC/filename,dpi=160,facecolor='white');plt.close(fig)


def audit():
    completed=[portfolio.RUN/a/'COMPLETE.json' for a in ('original','shared')]
    completed += [family.RUN/c/'COMPLETE.json' for c in family.CASES]
    completed += [tree.RUN/c/'COMPLETE.json' for c in ('tree','tree_neural50')]
    completed += [prior.RUN/c/'COMPLETE.json' for c in prior.CASES]
    completed += [reserve.CONF/'COMPLETE.json',confirmation.RUN/'COMPLETE.json',beta.RUN/'COMPLETE.json']
    output_files=0
    for lock_path in completed:
        for name,expected in json.loads(lock_path.read_text())['output_sha256'].items():
            assert sha(lock_path.parent/name)==expected,(lock_path,name)
            output_files+=1
    frames=[]
    def add(path,name):
        f=pd.read_csv(path)[['id','max_independent_daily_error']].copy();f.insert(0,'study',name);frames.append(f)
    for arm in ('original','shared'):add(portfolio.RUN/arm/'ACCOUNT_CHECKS.csv','portfolio_'+arm)
    for case in family.CASES:add(family.RUN/case/'ACCOUNT_CHECKS.csv','family_'+case)
    add(tree.RUN/'ALL_METRICS.csv','tabular')
    add(prior.RUN/'METRICS.csv','prior_blends');add(prior.RUN/'CONTROL_METRICS.csv','prior_controls')
    add(reserve.CONF/'ACCOUNT_CHECKS.csv','reserved_pooling');add(confirmation.RUN/'METRICS.csv','reserved_prior')
    add(beta.RUN/'ACCOUNT_CHECKS.csv','risk')
    checks=pd.concat(frames,ignore_index=True)
    assert len(checks)==1578 and not checks.duplicated(['study','id']).any()
    assert checks.max_independent_daily_error.max()<1e-12
    checks.to_csv(DOC/'ALL_ACCOUNT_CHECKS.csv',index=False)
    fits=[]
    for run in (portfolio.RUN,tree.RUN,reserve.CONF):
        lock=json.loads((run/'FITS_COMPLETE.json').read_text())
        for name,expected in lock['hashes'].items():assert sha(run/name)==expected,name
        fits.append(dict(run=run.relative_to(ROOT).as_posix(),fits=lock['fits'],files_checked=len(lock['hashes']),
                         lock_sha256=sha(run/'FITS_COMPLETE.json')))
    assert sum(r['fits'] for r in fits)==164
    result=dict(status='passed',fits=164,fit_locks=fits,accounts=1578,
        completed_output_locks=len(completed),output_files_checked=output_files,
        max_independent_daily_error=float(checks.max_independent_daily_error.max()),
        displayed_accounts=63,new_core_unit_tests_passed=14,
        limits='Accounting and saved-fit integrity; no independent vendor, live broker or untouched-market validation')
    write(DOC/'AUDIT.json',result)
    return result


def main():
    portfolio.verify();family.verify();tree.verify();prior.verify();confirmation.verify();beta.verify()
    DOC.mkdir(parents=True,exist_ok=True)
    dates,frame,paths,source_hashes=comparison()
    frame.to_csv(DOC/'COMPARISON_METRICS.csv',index=False);figures(dates,paths);checked=audit()
    result=json.loads((confirmation.RUN/'RESULT.json').read_text())
    pooling=json.loads((reserve.CONF/'POOLING_CONFIRMATION.json').read_text())
    risk_selection=json.loads((beta.RUN/'SELECTION.json').read_text())
    assert result['passed'] and not result['all_above_1_5'] and risk_selection['passes']==0
    for path,name in ((confirmation.RUN/'PROTOCOL.json','PRIOR_CONFIRMATION_PROTOCOL.json'),
                      (confirmation.RUN/'FREEZE.json','PRIOR_CONFIRMATION_FREEZE.json'),
                      (confirmation.RUN/'RESULT.json','PRIOR_CONFIRMATION_RESULT.json'),
                      (reserve.CONF/'POOLING_CONFIRMATION.json','POOLING_CONFIRMATION_RESULT.json'),
                      (beta.RUN/'CONSISTENCY.csv','RISK_CONSISTENCY.csv'),
                      (beta.RUN/'METRICS.csv','RISK_METRICS.csv'),
                      (beta.RUN/'PROTOCOL.json','RISK_PROTOCOL.json'),
                      (beta.RUN/'FREEZE.json','RISK_FREEZE.json')):
        shutil.copyfile(path,DOC/name);source_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    base=frame[(frame.cost==1)&(frame.borrow==.03)].set_index(['group','model'])
    stress=frame[(frame.cost==2)&(frame.borrow==.06)].set_index(['group','model'])
    old=[float(base.loc[(g,'baseline'),:].sharpe) for g in GROUPS[:4]]
    new=[float(base.loc[(g,'prior'),:].sharpe) for g in GROUPS[:4]]
    table=['| Ensemble | Previous stable issuer SR | Prior mixture SR | Change | Prior CAGR | Prior max DD |',
           '|---|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        a,b=base.loc[(g,'baseline'),:],base.loc[(g,'prior'),:]
        table.append(f'| {LABELS[g]} | {a.sharpe:.3f} | {b.sharpe:.3f} | {b.sharpe-a.sharpe:+.3f} | {b.annualized_return:.2%} | {b.maximum_drawdown:.2%} |')
    rows=['# Seed-robust Sharpe improvement — October 30, 2026','',
        '**The prior-regularized ensemble improves Sharpe in all seven evaluated groups, including a reserved five-seed cohort. Consistent SR>1.5 was not achieved.**',
        '',f'Completed **164 additional fits and 1,578 recorded accounts**, retaining all failed candidates, deterministic controls and cost stresses. Maximum independent daily-accounting discrepancy: **{checked["max_independent_daily_error"]:.3g}**. Counts include overlapping groups and duplicate paths, not1,578independent discoveries.','',
        '![Separate initialization cohorts](disjoint_ensembles.png)','','![Pooled models](pooled_ensembles.png)','',
        *table,'',
        'All rows use the same1,003market sessions,20%stock-basket volatility target,continuousQQQshort ceiling75%,gross target budget100%,base modeled fees and3%short borrowing. All models retain the predefined seeds equally; none is filtered by its financial result. The baseline is the October29loss-selected five-issuer model, not the older selected1.731retention best case.','',
        '## What changed','',
        'The candidate combines50%eligible-stock percentile ranks of a training-history stock prior with25%meanTHINKseed ranks and25%mean dynamic-hypergraph seed ranks. The prior averages observed within-date training-return ranks with a26decision-week exponential half-life and20neutral0.5pseudo-observations per stock. It uses only the training portion of each annual fold, ending before its validation year. It is not a current-six-month momentum signal, and no retrospectively identified winners are named or favored. Unsupported stock histories get neutral priors.','',
        'Each consecutive five-seed cohort selects five issuers and gets its own inherited risk sizing. The final account trades the equally averaged signed targets of those cohorts, nets trades and pays actual modeled execution/borrowing fees. The combined account may hold more than five issuers. It is not an arithmetic mean of saved account-return series.','',
        'On the reserved group there is only one five-seed cohort, so cohort averaging cannot explain the0.877→1.322gain. That comparison isolates the change to ranking through the training prior within the same issuer/risk/execution framework. The twenty-seed comparison additionally changes aggregation. The improvement is in the ranking/portfolio layer; the neural architectures and learned source checkpoints remain unchanged.','',
        '## What the confirmation establishes','',
        'Four priors and their fixed blends were explored on the first15seeds. The original strict source gate (including a0.15cohort-spread limit) failed. A clearly labeled secondary candidate was selected from configurations improving every source group, then frozen **before** any reserved-seed account was scored. Its source spread0.225exceeded the original0.15limit; the confirmation does not retroactively change that failure.','',
        f'The reserved five improve **{base.loc[("unused_five","baseline"),:].sharpe:.3f}→{base.loc[("unused_five","prior"),:].sharpe:.3f}**; expanded twenty improve **{base.loc[("expanded_twenty","baseline"),:].sharpe:.3f}→{base.loc[("expanded_twenty","prior"),:].sharpe:.3f}**. Across all four disjoint groups, the SR spread falls from **{max(old)-min(old):.3f}** to **{max(new)-min(new):.3f}**. The original three groups alone had understated the baseline\'s weakness.','',
        f'At2×fees and6%borrowing, reserved-five SR improves **{stress.loc[("unused_five","baseline"),:].sharpe:.3f}→{stress.loc[("unused_five","prior"),:].sharpe:.3f}** and expanded twenty **{stress.loc[("expanded_twenty","baseline"),:].sharpe:.3f}→{stress.loc[("expanded_twenty","prior"),:].sharpe:.3f}**. The secondary reserved-seed criterion passes; every stress case is retained in the comparison CSV.','',
        '[Exact pre-score candidate and secondary goal](PRIOR_CONFIRMATION_PROTOCOL.json) · [Freeze](PRIOR_CONFIRMATION_FREEZE.json) · [Confirmation result](PRIOR_CONFIRMATION_RESULT.json).','',
        '## Failed proposals remain visible','',
        '- Portfolio target averaging improves pooled retention SR, but the separately frozen no-band averaging proposal fails its new-seed consistency goal: the reserved cohort scores0.854. It still improves expanded-twenty retention SR1.015→1.237. [Full pooling confirmation](POOLING_CONFIRMATION_RESULT.json).',
        '- The fixed60-fit/540-account shared-THINK/band study fails its stricter improvement gate. Removing the stock-specific mixer greatly reduces financial performance. [All results](../ndx_portfolio_robustness_20261030_v2/REPORT.md).',
        '- Standalone families and validation-only blend weights fail in300additional accounts. [Family comparisons](../ndx_family_robustness_20261030/REPORT.md).',
        '- The64-fit/282-account long-history ExtraTrees/Ridge arm fails; more randomized trees did not create useful financial signal. [Feature models](../ndx_tabular_alpha_20261030/REPORT.md).',
        '- All264prior/control cases are retained, including the source third-five1.652case whose other cohorts are weaker. That high single-cohort result was not selected as representative. [Prior study](../ndx_prior_robustness_20261030/REPORT.md).',
        '- The final168-account risk arm rejects more beta hedging and a lower stock-vol target. All42original-risk reference accounts reproduce. Trailing beta60/120rules sharply reduce SR; no risk variant reaches1.5across allseven groups. Existing risk settings remain in the confirmed candidate. [All risk accounts](RISK_METRICS.csv) · [Risk summaries](RISK_CONSISTENCY.csv) · [Frozen risk protocol](RISK_PROTOCOL.json).','',
        '## Limits for the paper','',
        '**This is initialization robustness on spent market history, not unseen-market generalization.** All cohorts share2022–2025outcomes; pooled groups overlap their constituents. The reserved seed values were fixed in the October20list and unused in this retraining protocol, not newly chosen by return. The candidate was fixed before their financial outcomes, while later rejected risk diagnostics used them as development. Further choices cannot reuse this cohort as untouched confirmation.','',
        'Many strategy/model variants have been inspected. Seed confirmation does not undo selection on market history or provide a multiple-search-adjusted significance claim. [Bailey and López de Prado,Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) discusses bias from searching and selecting high historical Sharpe ratios. No livebroker,independentvendor or newmarket evaluation is supplied. The1.5goal remains unmet;1.731remains a selected historical best case with its weaker seed cohort disclosed.','',
        '## Verification and reproduction','',
        'All164new-fit locks and every referenced file hash verify. All1,578accounts pass independent dollar-position accounting; all63displayed account metrics reconstruct from saved daily returns. Fourteen new core tests passed, covering actual-holdings causality/budgets,zero-band replay,matched shared-head initialization,causal features,training-only scaling/labels,forest reproducibility,prior labels/decay and beta-risk causality/caps. The interrupted shared fit resumes exactly (zero parameter/prediction discrepancy); its file-retry amendment changes logging only.','',
        '[Every account check](ALL_ACCOUNT_CHECKS.csv) · [Fit/account audit](AUDIT.json) · [All comparison metrics/costs](COMPARISON_METRICS.csv) · [Provenance](MANIFEST.json)','',
        'The source studies retain code/input freezes and complete checkpoint/model/forecast/ledger artifacts under their dated runs. This overview only republishes verified daily paths:','',
        '```powershell','.\\.venv\\Scripts\\python.exe -m hypershift run publish_ndx_seed_robustness','```','']
    (DOC/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    (DOC/'CHART_TABLE.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    write(DOC/'MANIFEST.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(ROOT/'research/reporting/publish_ndx_seed_robustness.py'),source_sha256=source_hashes,
        output_sha256={p.name:sha(p) for p in DOC.iterdir() if p.is_file() and p.name!='MANIFEST.json'}))
    print('Published confirmed seed improvement:',DOC,flush=True)


if __name__=='__main__':main()
