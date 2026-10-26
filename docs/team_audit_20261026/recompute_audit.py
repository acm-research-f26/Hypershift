"""Read committed development artifacts only; never load protected market data."""
from pathlib import Path
import csv
import hashlib
import io
import json
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REF = '0b06f26e81eb198fed222bac172fcbc60e7daeee'
PREFIX = 'outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/'
sources = {}

def blob(ref, path):
    raw = subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=ROOT)
    sources[f'{ref}:{path}'] = hashlib.sha256(raw).hexdigest()
    return raw

def rows(ref, path):
    return list(csv.DictReader(io.StringIO(blob(ref, path).decode('utf-8-sig'))))

def main():
    allrows = rows(REF, PREFIX+'all_strategies.csv')
    # The new 346-path matrix is ignored/unpublished. Replay only a tracked
    # older curve set; never equate a publication manifest with our replay.
    curves=rows(REF,'docs/nasdaq_monthly_pit/topk_pseudohuber/curves.csv')
    metrics=rows(REF,'docs/nasdaq_monthly_pit/topk_pseudohuber/metrics.csv')
    keys=('model','seed','k','cost_multiplier')
    grouped={}
    for row in curves:
        key=tuple(row[k] for k in keys)
        grouped.setdefault(key,[]).append(float(row['net']))
    errors = {}
    for row in metrics:
        r=np.array(grouped[tuple(row[k] for k in keys)])
        assert len(r)==1003 and np.isfinite(r).all() and (r>-1).all()
        wealth=np.r_[1.,np.cumprod(1+r)]
        calc={'annualized_return':wealth[-1]**(252/len(r))-1,
              'sharpe':(r.mean()-(1.03**(1/252)-1))/r.std(ddof=1)*np.sqrt(252),
              'maximum_drawdown':(wealth/np.maximum.accumulate(wealth)-1).min()}
        for col,value in calc.items():
            errors[col]=max(errors.get(col,0),abs(value-float(row[col])))
            np.testing.assert_allclose(value,float(row[col]),atol=1e-10,rtol=1e-10)
    seeds = rows('unify-20261024','docs/unify_20261024/tables/seed20_seeds.csv')
    controls = json.loads(blob('unify-20261024','docs/unify_20261024/tables/seed20_pit.json'))['controls']
    reversal = controls['reversal5']['RankIC']['mean']
    # Benchmark is gross EW as specified in the original report, not a net comparator.
    ew = controls.get('equal_weight', controls.get('ew', {}))
    seedstats = {}
    for model in sorted(set(x['model'] for x in seeds)):
        values = [x for x in seeds if x['model']==model]
        seedstats[model] = dict(n=len(values), above_reversal_rankic=sum(float(x['RankIC'])>reversal for x in values),
            above_ew_cagr_rounded_10_8pct=sum(float(x['cagr10_10'])>.108 for x in values),
            median_cagr=float(np.median([float(x['cagr10_10']) for x in values])))
    report = dict(rishi_ref=REF, verified_older_paths=len(metrics), published_current_rows=len(allrows), sessions=1003,
        unavailable='Current analysis_paths.npz, reference-network daily ledgers and per-stock predictions are not in Git',
        maximum_metric_errors=errors, highest_published_sharpe_id=max(allrows,key=lambda x:float(x['sharpe']))['id'],
        highest_published_cagr_id=max(allrows,key=lambda x:float(x['cagr']))['id'], seed20_reversal_rankic=reversal, seed20=seedstats,
        controls_keys=list(controls), source_sha256=sources)
    # Fixed selected paths; replay of published numbers, not a strategy search.
    selected = [x for x in allrows if 'dynamic_restricted_blend' in x['id'] or x['id'] in ['original/SPY/10','original/QQQ/10']]
    report['selected_published_rows']=selected
    selected_metrics = rows(REF,'latest/results/reference_networks/metrics.csv')
    report['reference_metrics']=[x for x in selected_metrics if x['seed']=='ensemble' and x['cost_multiplier'] in ['1','2','4']]
    report['paired_intervals']=[x for x in rows(REF,'latest/results/reference_networks/paired_intervals.csv') if x['block_sessions']=='20' and x['measure']=='sharpe_difference' and 'dynamic_restricted_blend' in x['left_id']]
    report['ranking']=rows(REF,'latest/results/reference_networks/ranking.csv')
    report['annual_blend']=[x for x in rows(REF,'latest/results/reference_networks/annual.csv') if x.get('id','').startswith('dynamic_restricted_blend/') and x.get('id','').endswith('/1')]
    report['source_sha256']=sources
    (OUT/'RECOMPUTED_EVIDENCE.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','selected_published_rows','reference_metrics','paired_intervals','ranking','annual_blend']},indent=2))
    inventory()
    figures(report)


def inventory():
    refs=subprocess.check_output(['git','for-each-ref','--format=%(refname:short)','refs/heads','refs/remotes/origin'],cwd=ROOT,text=True).splitlines()
    records=[]
    for ref in refs:
        if ref=='origin/HEAD':continue
        details=subprocess.check_output(['git','log','-1','--format=%H%n%aI%n%s',ref],cwd=ROOT,text=True).splitlines()
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only',ref],cwd=ROOT,text=True).splitlines()
        research=[p for p in paths if p.endswith(('.md','.csv','.json','.yaml','.yml')) and any(k in p.lower() for k in ['report','audit','result','progress','handoff','claude','readme','chart','protocol','prereg','metric'])]
        changed=subprocess.check_output(['git','log',ref,'--since=2026-10-23T00:00:00-05:00','--until=2026-10-27T00:00:00-05:00','--format=%H %aI %s','--name-only'],cwd=ROOT,text=True).splitlines()
        records.append(dict(ref=ref,sha=details[0],author_time=details[1],subject=details[2],tracked_files=len(paths),research_artifact_paths=research,oct23_26_changes=changed))
    worktrees=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True)
    obj=dict(audit_date='2026-10-26',published_branches=['ethan','feat-rs','kavin','main','team-report-20261023','tom-shlom'],
             branch_discovery='GitHub connector pagination exhausted; git fetch then local ref enumeration',
             coverage='Inventory is exhaustive for available refs; paths listed are not a claim each file was independently replayed',
             refs=records,worktrees_porcelain=worktrees)
    (OUT/'BRANCH_INVENTORY.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')
    print('Inventory:',len(records),'refs')


def figures(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    selected=[r for r in report['paired_intervals'] if r['group']=='ten_seed']
    fig,ax=plt.subplots(figsize=(10,5.2))
    for i,row in enumerate(selected):
        point=float(row['estimate']);lo=float(row['paired_low']);hi=float(row['paired_high'])
        slo=float(row['simultaneous_low']);shi=float(row['simultaneous_high'])
        ax.plot([slo,shi],[i+.08,i+.08],color='#b3bec8',lw=5,solid_capstyle='round')
        ax.plot([lo,hi],[i-.08,i-.08],color='#176794',lw=2)
        ax.scatter([point],[i-.08],c='#176794',s=30,zorder=3)
    labels=['Blend minus '+r['contrast'].split(' minus ')[-1].replace('baseline','parent').replace('standalone','dynamic alone') for r in selected]
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.axvline(0,color='#933b34',lw=1,ls='--')
    ax.set_xlabel('Paired annualized net Sharpe difference (3% annual hurdle)')
    ax.set_title('Primary ten-seed, top-ten portfolio: the advantage remains uncertain',loc='left',fontweight='bold',pad=18)
    ax.plot([],[],color='#176794',lw=2,label='Pointwise 95% interval')
    ax.plot([],[],color='#b3bec8',lw=5,label='Within-study simultaneous 95% (23 contrasts)')
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.08),ncol=2,frameon=False,fontsize=9)
    ax.grid(axis='x',alpha=.2)
    fig.text(.02,.025,'NDX 2022-2025, 1,003 sessions. Published paired bootstrap, block 20. Fitted paths and full historical search remain limitations.',fontsize=8)
    fig.tight_layout(rect=(0,.18,1,1));fig.savefig(OUT/'FIGURE_1_PAIRED_EVIDENCE.png',dpi=160);plt.close(fig)
    metrics=report['reference_metrics']
    fig,axes=plt.subplots(1,2,figsize=(11,5.2),sharey=True)
    groups=['original_five','new_five','ten_seed']
    colors={'baseline':'#717d8b','dynamic_restricted':'#d18a2d','dynamic_restricted_blend':'#176794'}
    names={'baseline':'Parent','dynamic_restricted':'Dynamic alone','dynamic_restricted_blend':'Fixed blend'}
    for ax,k in zip(axes,[10,5]):
        for method,color in colors.items():
            vals=[float(next(r['sharpe'] for r in metrics if r['method']==method and r['group']==g and int(r['k'])==k and r['cost_multiplier']=='1')) for g in groups]
            ax.plot(range(3),vals,marker='o',color=color,label=names[method],lw=1.8)
        ax.set_xticks(range(3),['Original five','New five','All ten']);ax.set_title(f'Top {k}'+(' (primary)' if k==10 else ' (secondary)'))
        ax.axhline(.4978751005186223,color='#a1423b',ls='--',lw=1,label='SPY base cost')
        ax.grid(axis='y',alpha=.2);ax.set_ylim(0,1.05)
    axes[0].set_ylabel('Net annualized Sharpe, 3% hurdle');axes[1].legend(frameon=False,loc='lower right',fontsize=9)
    fig.suptitle('Seed-group robustness is distinct from independent market evidence',x=.06,ha='left',fontweight='bold',fontsize=13)
    fig.text(.02,.02,'Same NDX 2022-2025 history, modeled base costs. Lines connect overlapping estimator groups; they do not show a time series or new replications.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/'FIGURE_2_SEED_GROUPS.png',dpi=160);plt.close(fig)
    print('Wrote two figures from committed summary tables; no new strategy scores.')

if __name__=='__main__':
    main()
