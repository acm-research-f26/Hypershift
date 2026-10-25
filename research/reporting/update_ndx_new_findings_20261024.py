"""Publish the sealed loss/optimizer and reference-network findings to the catalog."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import textwrap
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020'
DOC=ROOT/'docs/strategy_comparison_20261024'
BEFORE=ROOT/'.cache/strategy_new_findings_20261024_previous'
BOOK=ROOT/'docs/NASDAQ100_strategy_comparison.xlsx'
STUDIES={'loss_optimizer':'ndx_loss_optimizer_20261024','reference':'ndx_reference_networks_20261024'}
GROUPS={'ten_seed':'ten seeds','original_five':'original five seeds','new_five':'new five seeds'}
LABELS={
 'fixed_adamw_mse':'THINK+mixer AH+PM / MSE / fixed AdamW',
 'fixed_adamw_huber':'THINK+mixer AH+PM / Huber / fixed AdamW',
 'fixed_adamw_pseudohuber':'THINK+mixer AH+PM / pseudo-Huber / fixed AdamW',
 'tuned_adamw_mse':'THINK+mixer AH+PM / MSE / tuned AdamW',
 'tuned_radam_mse':'THINK+mixer AH+PM / MSE / tuned RAdam',
 'tuned_sgd_mse':'THINK+mixer AH+PM / MSE / tuned momentum SGD',
 'baseline':'THINK+mixer AH+PM / MSE / fixed AdamW',
 'master':'MASTER transfer', 'dynamic_restricted':'Dynamic hypergraph (restricted)',
 'prism':'PRISM-VQ transfer',
 'master_blend':'THINK+mixer AH+PM + MASTER / 50:50 ranks',
 'dynamic_restricted_blend':'THINK+mixer AH+PM + dynamic HG / 50:50 ranks',
 'prism_blend':'THINK+mixer AH+PM + PRISM-VQ / 50:50 ranks',
 'SPY':'SPY','QQQ':'QQQ'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
def read(p):return pd.read_csv(p,dtype={'seed':str})


def snapshot():
    if (BEFORE/'MANIFEST.json').exists():return
    BEFORE.mkdir(parents=True,exist_ok=True);files=[BOOK,ROOT/'charts.md',ROOT/'README.md',ROOT/'docs/README.md']
    files += [OUT/n for n in ['all_strategies.csv','workbook_data.json','sharpe_block_sensitivity.csv','analysis_paths.npz','ANALYSIS.json','AUDIT.json','COMPARISON.md','build_workbook.mjs']]
    files += [DOC/n for n in ['chart_catalog.csv','AUDIT.json','FIGURE_MANIFEST.json','figures/historical_leaders.png']]
    manifest={}
    for p in files:
        target=BEFORE/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        manifest[str(p.relative_to(ROOT))]=sha(p)
    dump(BEFORE/'MANIFEST.json',manifest)


def old(p):return BEFORE/Path(p).relative_to(ROOT)


def verify_studies():
    for name in STUDIES.values():
        folder=ROOT/'runs'/name;record=json.loads((folder/'FINAL_PUBLICATION.json').read_text())
        assert json.loads((folder/'AUDIT.json').read_text())['status']=='passed'
        for relative,h in record['artifact_hashes'].items():assert sha(folder/relative)==h,(name,relative)


def control_id(method,group,seed,k,cost):
    if method in ['SPY','QQQ']:return f'original/{method}/10' if cost==1 else f'extended_detail/{method}/ten_seed/ensemble/{cost}'
    if group=='individual':return f'extended_detail/after_hours_premarket/individual/{seed}/1'
    if k==10:return f'extended/after_hours_premarket/{group}' if cost==1 else f'extended_detail/after_hours_premarket/{group}/ensemble/{cost}'
    return f'holdings5/after_hours_premarket/{group}' if cost==1 else f'top5_detail/after_hours_premarket/{group}/ensemble/5/{cost}'


def assemble():
    verify_studies();prior=read(old(OUT/'all_strategies.csv'));catalog=read(old(DOC/'chart_catalog.csv'))
    additions=[];details=[];new_panels=[];bounds=[];paths={};summaries=[]
    existing=set(catalog.id);seen=set(existing)
    with np.load(old(OUT/'analysis_paths.npz')) as f:
        dates=f['dates'].copy();old_ids=f['ids'].copy();old_returns=f['returns'].copy()
    for study,name in STUDIES.items():
        folder=ROOT/'runs'/name;m=read(folder/'metrics.csv');ci=read(folder/'sharpe_intervals.csv');ranking=read(folder/'ranking.csv')
        quality=read(folder/'prediction_quality.csv');curve=read(folder/'curves.csv')
        ps={key:p for key,p in curve.groupby('id',sort=False)}
        ic={(r.method,r.group):float(r.rankic) for r in ranking.itertuples()}
        individual={(r.method,str(r.seed)):float(r.rankic) for r in quality.itertuples()}
        title='AH+PM losses and optimizers' if study=='loss_optimizer' else 'Author network transfers'
        for row in m.to_dict('records'):
            method,group,seed,k,cost=[row[n] for n in ['method','group','seed','k','cost_multiplier']]
            control=method in ['fixed_adamw_mse','baseline','SPY','QQQ']
            is_main=not control and seed=='ensemble' and cost==1
            canonical=control_id(method,group,seed,k,cost) if control else f'{study}/{method}/{group}/top{k}' if is_main else f'new_detail/{study}/{row["id"]}'
            c=ci[(ci.id==row['id'])&(ci.block_sessions==20)].iloc[0]
            rank=ic.get((method,group),individual.get((method,seed)))
            label=LABELS[method]
            strategy=label if k==0 else f'{label} / {GROUPS.get(group,"seed "+seed)} / top {k}'
            role='Benchmark' if k==0 else 'Fee stress' if cost!=1 else 'Individual seed' if group=='individual' else 'Primary ten-seed top ten' if group=='ten_seed' and k==10 else 'Secondary ten-seed top five' if group=='ten_seed' else 'Seed-half diagnostic'
            item=dict(id=canonical,source_id=row['id'],source_study=study,method=method,study=title,strategy=strategy,group=group,seed=seed,k=k,cost_multiplier=cost,
                sharpe=float(row['sharpe']),cagr=float(row['annualized_return']),rankic=rank,max_drawdown=float(row['maximum_drawdown']),
                sr_low=float(c.sr_low),sr_high=float(c.sr_high),role=role)
            details.append(item)
            p=ps[row['id']];np.testing.assert_array_equal(p.date,dates)
            if canonical in paths:np.testing.assert_allclose(paths[canonical],p.net.to_numpy(),atol=1e-14,rtol=0)
            paths[canonical]=p.net.to_numpy()
            if control:assert canonical in existing,canonical
            if is_main:
                additions.append({n:item[n] for n in ['id','study','strategy','k','sharpe','cagr','rankic','max_drawdown','sr_low','sr_high']}|dict(policy='full',section='Models' if k==10 else 'Portfolio variants'))
                for b in ci[ci.id==row['id']].itertuples():bounds.append(dict(id=canonical,block_sessions=b.block_sessions,sr_low=b.sr_low,sr_high=b.sr_high))
            if canonical not in seen:new_panels.append(item);seen.add(canonical)
            if (not control or method=='fixed_adamw_mse') and group=='ten_seed' and cost==1 and seed=='ensemble':
                halves={g:m[(m.method==method)&(m.group==g)&(m.k==k)&(m.cost_multiplier==1)].iloc[0] for g in ['original_five','new_five']}
                summaries.append(item|dict(strategy=label,original_sr=float(halves['original_five'].sharpe),new_sr=float(halves['new_five'].sharpe),
                    original_cagr=float(halves['original_five'].annualized_return),new_cagr=float(halves['new_five'].annualized_return)))
    assert len(additions)==66 and len(details)==346 and len(new_panels)==278 and len(summaries)==24
    combined=pd.concat([prior,pd.DataFrame(additions)],ignore_index=True);assert len(combined)==346 and combined.id.is_unique
    for ticker in ['SPY','QQQ']:
        item=next(r for r in details if r['method']==ticker and r['cost_multiplier']==1)
        summaries.append(item|dict(original_sr=None,new_sr=None,original_cagr=None,new_cagr=None))
    summaries.sort(key=lambda r:-r['cagr'])
    return dict(combined=combined,additions=additions,details=details,new_panels=new_panels,summaries=summaries,paths=paths,dates=dates,
                catalog=catalog,bounds=bounds,old_ids=old_ids,old_returns=old_returns)


NOTES=[
 ['New experiments','Both completed studies are independently audited. All 2022–2025 results remain exploratory development evidence. Models adds 33 top-ten rows; Portfolio variants adds 33 top-five rows.'],
 ['Still uses THINK','The AH+PM dynamic rank blend retains the original THINK+mixer core and its AH+PM branch. A separately trained restricted dynamic hypergraph network supplies complementary rankings.'],
 ['Fixed rank blend','Each network averages eligible cross-sectional percentile ranks across the same fixed seeds. Final score is 0.5 × THINK ensemble rank + 0.5 × dynamic ensemble rank. Select one top-five or top-ten equal-weight portfolio. No blend weights were fitted to test returns.'],
 ['Trading clock','AH/PM features end at 09:15 ET, decision at 09:20, fill at that session’s regular close. Rebalance every five exchange sessions. These are regular-close trades using extended-hours information.'],
 ['Seed comparison','Ten seeds are primary. Original-five and new-five halves diagnose initialization sensitivity on the same market history. They are not independent market periods. New findings places both halves beside each ten-seed result.'],
 ['Reference adaptations','Author neural code was transferred to common NASDAQ features and declared training limits. Restricted dynamic uses one train-only correlation graph rather than the full sector/lead-lag pipeline. PRISM uses lagged JKP themes with unresolved historical publication-vintage limitations.'],
 ['Loss and optimizer scope','Huber and pseudo-Huber use delta=1 percentage point. Optimizer rates were selected only from prior-year validation MSE. SGD keeps MSE and momentum 0.9. Shared MSE stopping means this is a finite recipe comparison, not proof of a universally best loss or optimizer.'],
 ['Uncertainty and confirmation','Absolute SR intervals use a 3% annual hurdle and 10,000 circular-block draws, 20 sessions primary. Paired benchmark and within-batch simultaneous intervals are in the source studies. Conditional path intervals omit the full prior search and retraining uncertainty. No untouched-period superiority is established.'],
 ['New result sources','Sealed local studies: runs/ndx_loss_optimizer_20261024 and runs/ndx_reference_networks_20261024. Metrics, ranking, curves, SR intervals, paired intervals and audit records supply the new rows. New accounts retains all 346 declared accounts, including repeated controls.'],
]


def secondary_intervals():
    """Additional exploratory sensitivity, not prespecified primary inference."""
    verify_studies();rows=[]
    for study,name in STUDIES.items():
        folder=ROOT/'runs'/name;m=read(folder/'metrics.csv').set_index('id');curves=read(folder/'curves.csv')
        primary=read(folder/'paired_intervals.csv')
        base=primary[(primary.group=='ten_seed')&(primary.block_sessions==20)&(primary.measure=='sharpe_difference')]
        pairs=[]
        for k in [5,10]:
            for r in base.itertuples():pairs.append((k,r.contrast,r.left_id.replace('/ensemble/10/1',f'/ensemble/{k}/1'),r.right_id.replace('/ensemble/10/1',f'/ensemble/{k}/1')))
        keys=list(dict.fromkeys(v for _,_,a,b in pairs for v in [a,b]));matrix=np.column_stack([curves[curves.id==key].net.to_numpy() for key in keys])
        n=len(matrix);rng=np.random.default_rng(20261040);sr=[];growth=[]
        for _ in range(100):
            starts=rng.integers(0,n,(100,math.ceil(n/20)));index=((starts[:,:,None]+np.arange(20))%n).reshape(100,-1)[:,:n];sample=matrix[index]
            sr.append(np.sqrt(252)*(sample.mean(1)-(1.03**(1/252)-1))/sample.std(1,ddof=1));growth.append(np.expm1(np.log1p(sample).sum(1)*252/n))
        for measure,draws,field in [('sharpe_difference',np.concatenate(sr),'sharpe'),('cagr_difference',np.concatenate(growth),'annualized_return')]:
            values=np.column_stack([draws[:,keys.index(a)]-draws[:,keys.index(b)] for _,_,a,b in pairs]);point=np.array([m.loc[a,field]-m.loc[b,field] for _,_,a,b in pairs])
            error=values.std(0,ddof=1);critical=np.quantile(np.max(np.abs((values-point)/np.maximum(error,1e-12)),1),.95)
            for j,(k,label,left,right) in enumerate(pairs):
                low,high=np.quantile(values[:,j],[.025,.975])
                if k==10:
                    check=primary[(primary.group=='ten_seed')&(primary.block_sessions==20)&(primary.measure==measure)&(primary.contrast==label)].iloc[0]
                    np.testing.assert_allclose([point[j],low,high],[check.estimate,check.paired_low,check.paired_high],atol=2e-9,rtol=1e-9)
                rows.append(dict(study=study,k=k,group='ten_seed',contrast=label,measure=measure,left_id=left,right_id=right,estimate=point[j],paired_low=low,paired_high=high,
                    simultaneous_low=point[j]-critical*error[j],simultaneous_high=point[j]+critical*error[j],family_comparisons=len(pairs),block_sessions=20,replicates=10000,
                    interpretation='Additional exploratory sensitivity over both holding counts, after primary results were inspected; excludes full prior search and retraining uncertainty'))
    assert len(rows)==172;pd.DataFrame(rows).to_csv(OUT/'new_holdings_uncertainty.csv',index=False)
    print('Secondary both-holding-count sensitivity computed; primary paired intervals independently reproduced',flush=True)


def prepare():
    snapshot();a=assemble();a['combined'].to_csv(OUT/'all_strategies.csv',index=False)
    pd.DataFrame(a['details']).to_csv(OUT/'new_experiment_all_accounts.csv',index=False)
    pd.DataFrame(a['summaries']).to_csv(OUT/'new_findings_seed_summary.csv',index=False)
    payload=json.loads(old(OUT/'workbook_data.json').read_text())
    for row in a['additions']:payload['sheets'][0 if row['section']=='Models' else 2]['rows'].append(row)
    payload['methodology'].extend(NOTES);payload['new_experiment_accounts']=a['details'];payload['new_findings']=a['summaries']
    dump(OUT/'workbook_data.json',payload)
    dump(OUT/'new_findings_workbook_data.json',dict(input_workbook=str(old(BOOK)),input_sha256=sha(old(BOOK)),
        output_workbook=str(BEFORE/'updated_staged.xlsx'),existing_counts={s['name']:len(s['rows']) for s in json.loads(old(OUT/'workbook_data.json').read_text())['sheets']},
        additions=a['additions'],details=a['details'],summary=a['summaries'],notes=NOTES))
    pd.concat([read(old(OUT/'sharpe_block_sensitivity.csv')),pd.DataFrame(a['bounds'])],ignore_index=True).to_csv(OUT/'sharpe_block_sensitivity.csv',index=False)
    ids=np.r_[a['old_ids'],np.array([r['id'] for r in a['additions']])]
    returns=np.column_stack([a['old_returns'],*[a['paths'][r['id']] for r in a['additions']]])
    assert returns.shape==(1003,346)
    np.savez_compressed(OUT/'analysis_paths.npz',dates=a['dates'],ids=ids,returns=returns)
    print('Prepared 66 new main rows, 346 account details and 26 seed-comparison rows',flush=True)


def md_table(rows,criteria=False):
    lines=['| '+('Historical distinction | ' if criteria else '')+'Strategy | Sharpe | CAGR | RankIC | Max drawdown | 95% SR interval |',
           '| '+('--- | ' if criteria else '')+'--- | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        ic='n.a.' if r['rankic'] is None or pd.isna(r['rankic']) else f"{r['rankic']:.4f}"
        lines.append('| '+(r['criterion']+' | ' if criteria else '')+f"{r['strategy']} | {r['sharpe']:.3f} | {r['cagr']:.2%} | {ic} | {r['max_drawdown']:.2%} | [{r['sr_low']:.3f}, {r['sr_high']:.3f}] |")
    return lines


def charts():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    a=assemble();figures=DOC/'figures';x=pd.to_datetime(a['dates']);main=a['combined'].set_index('id')
    with np.load(OUT/'analysis_paths.npz') as f:all_paths={key:f['returns'][:,j] for j,key in enumerate(f['ids'])}
    benchmark={t:all_paths[f'original/{t}/10'] for t in ['SPY','QQQ']}
    def nav(r):return np.cumprod(1+r)
    def dd(r):w=np.r_[1.,nav(r)];return (w/np.maximum.accumulate(w)-1)[1:]*100
    def finish(fig,name):fig.tight_layout(rect=(0,.025,1,.965));fig.savefig(figures/name,dpi=140);plt.close(fig)
    def get(method,k,group='ten_seed'):
        return next(r for r in a['details'] if r['source_study']=='reference' and r['method']==method and r['group']==group and r['k']==k and r['cost_multiplier']==1)
    def sgd(k,group='ten_seed'):
        return next(r for r in a['details'] if r['method']=='tuned_sgd_mse' and r['group']==group and r['k']==k and r['cost_multiplier']==1)
    fig,axes=plt.subplots(2,2,figsize=(15,10),sharex=True)
    for col,k in enumerate([5,10]):
        selected=[(get('baseline',k),'THINK+mixer AH+PM','#164a70'),(get('dynamic_restricted_blend',k),'THINK + dynamic rank blend','#15907c'),(sgd(k),'THINK+mixer AH+PM, SGD','#d18a32')]
        for row,label,color in selected:
            r=a['paths'][row['id']];axes[0,col].plot(x,nav(r),label=f'{label} ({row["cagr"]:.1%})',color=color,lw=1.6);axes[1,col].plot(x,dd(r),color=color,lw=1.2)
        for ticker,color in [('SPY','#cc5555'),('QQQ','#8873aa')]:
            axes[0,col].plot(x,nav(benchmark[ticker]),label=ticker,color=color,lw=1);axes[1,col].plot(x,dd(benchmark[ticker]),color=color,lw=.9)
        axes[0,col].set_title(f'{k} holdings — '+('secondary' if k==5 else 'primary'));axes[0,col].legend(fontsize=8,loc='upper left')
        axes[0,col].set_ylabel('Net wealth per initial $1');axes[1,col].set_ylabel('Drawdown (%)')
    for ax in axes.flat:ax.grid(alpha=.18);ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.suptitle('New findings: fixed ten-seed ensembles, 2022–2025 base costs',fontsize=14);finish(fig,'new_findings_overview.png')
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for col,k in enumerate([5,10]):
        positions=np.arange(3)
        for j,(group,label,color) in enumerate([('original_five','Original five','#587e9a'),('new_five','New five','#d18a32'),('ten_seed','All ten','#15907c')]):
            rs=[get('baseline',k,group),get('dynamic_restricted_blend',k,group),sgd(k,group)]
            axes[0,col].bar(positions+(j-1)*.24,[r['cagr']*100 for r in rs],width=.24,label=label,color=color)
            axes[1,col].bar(positions+(j-1)*.24,[r['sharpe'] for r in rs],width=.24,color=color)
        for t,color in [('SPY','#cc5555'),('QQQ','#8873aa')]:
            base=main.loc[f'original/{t}/10'];axes[0,col].axhline(base.cagr*100,color=color,ls='--',lw=1,label=t);axes[1,col].axhline(base.sharpe,color=color,ls='--',lw=1)
        axes[0,col].set(title=f'{k} holdings',ylabel='CAGR (%)');axes[1,col].set_ylabel('Sharpe (3% hurdle)');axes[0,col].legend(fontsize=8,ncol=3)
        for ax in axes[:,col]:ax.set_xticks(positions,['AH+PM baseline','AH+PM + dynamic','AH+PM SGD']);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Initialization sensitivity: seed halves share the same market history',fontsize=14);finish(fig,'new_findings_seeds.png')
    primary=[r for r in a['summaries'] if r['k'] in [0,10]]
    fig,axes=plt.subplots(1,2,figsize=(18,11))
    for j,r in enumerate(primary):axes[0].plot([r['sr_low'],r['sr_high']],[j,j],color='#587e9a',lw=2);axes[0].scatter(r['sharpe'],j,color='#164a70',s=23)
    axes[0].set_yticks(range(len(primary)),[textwrap.fill(r['strategy'],43) for r in primary]);axes[0].invert_yaxis();axes[0].axvline(0,color='#666666',lw=.8);axes[0].set_title('Absolute 95% SR intervals')
    paired=[]
    for study,name in STUDIES.items():
        p=read(ROOT/'runs'/name/'paired_intervals.csv')
        mask=(p.group=='ten_seed')&(p.block_sessions==20)&(p.measure=='sharpe_difference')
        for r in p[mask].to_dict('records'):
            if r['contrast'].endswith(' minus baseline') or r['contrast'].endswith(' minus fixed MSE'):
                method=r['left_id'].split('/')[0];paired.append(r|dict(label=LABELS[method]))
    for j,r in enumerate(paired):
        axes[1].plot([r['simultaneous_low'],r['simultaneous_high']],[j,j],color='#b5c1ca',lw=5,label='Within-batch simultaneous 95%' if j==0 else None)
        axes[1].plot([r['paired_low'],r['paired_high']],[j,j],color='#164a70',lw=2,label='Paired 95%' if j==0 else None);axes[1].scatter(r['estimate'],j,color='#164a70',s=20)
    axes[1].set_yticks(range(len(paired)),[textwrap.fill(r['label'],43) for r in paired]);axes[1].invert_yaxis();axes[1].axvline(0,color='#666666',lw=.8);axes[1].set_title('SR difference versus matched AH+PM baseline');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='x',alpha=.18)
    fig.suptitle('Primary top-ten / ten-seed uncertainty — 10,000 paired 20-session block draws',fontsize=13)
    fig.text(.5,.008,'Within-batch families contain 20 or 23 contrasts. Full prior research and retraining uncertainty are not included.',ha='center',fontsize=9)
    finish(fig,'new_findings_intervals.png')
    # Refresh the shared historical leaderboard without altering prior gallery pages.
    best=a['combined'];leaders=[]
    criteria=[('Highest observed CAGR',best.loc[best.cagr.idxmax()]['id']),('Highest observed Sharpe',best.loc[best.sharpe.idxmax()]['id']),
      ('Highest top-ten CAGR',best[best.section=='Models'].sort_values('cagr',ascending=False).iloc[0]['id']),
      ('Ten-seed blend, secondary five holdings',get('dynamic_restricted_blend',5)['id']),('Ten-seed SGD, five holdings',sgd(5)['id']),
      ('Original daily reference','original/think_mix/10')]
    for criterion,key in criteria:leaders.append(dict(main.loc[key],id=key,criterion=criterion))
    fig,axes=plt.subplots(2,1,figsize=(14,10),sharex=True,gridspec_kw={'height_ratios':[2,1]});seen=set()
    for r in leaders+[dict(main.loc[f'original/{t}/10'],id=f'original/{t}/10') for t in ['SPY','QQQ']]:
        if r['id'] in seen:continue
        seen.add(r['id']);line,=axes[0].plot(x,nav(all_paths[r['id']]),label=f'{r["strategy"]} ({r["cagr"]:.1%})',lw=1.25);axes[1].plot(x,dd(all_paths[r['id']]),color=line.get_color(),lw=1)
    axes[0].set_title('Observed historical leaders and benchmarks — seed groups are separate diagnostics');axes[0].set_ylabel('Net wealth per initial $1');axes[0].legend(fontsize=7,loc='upper left');axes[1].set_ylabel('Drawdown (%)')
    for ax in axes:ax.grid(alpha=.18)
    finish(fig,'historical_leaders.png')
    entries=[];pages=[]
    for start in range(0,len(a['new_panels']),9):
        items=a['new_panels'][start:start+9];nr=math.ceil(len(items)/3);fig,axes=plt.subplots(nr,3,figsize=(18,4.7*nr+.4),sharex=True,squeeze=False)
        filename=f'new_experiments_{start//9+1:02d}.png'
        for ax,r in zip(axes.flat,items):
            ax.plot(x,nav(a['paths'][r['id']]),color='#164a70',lw=1.5)
            for ticker,color in [('SPY','#cc5555'),('QQQ','#8873aa')]:ax.plot(x,nav(benchmark[ticker]),color=color,lw=.9,alpha=.8)
            ax.set_title(textwrap.fill(r['strategy']+f' / {r["cost_multiplier"]}x fees',54)+f'\nCAGR {r["cagr"]:.2%}   SR {r["sharpe"]:.3f}   DD {r["max_drawdown"]:.1%}',fontsize=9)
            ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));ax.tick_params(labelsize=8);ax.grid(alpha=.18)
            entries.append(dict(id=r['id'],strategy=r['strategy'],figure=filename,section=r['study']+' / '+r['role'],figure_path='docs/strategy_comparison_20261024/figures/'+filename))
        for ax in list(axes.flat)[len(items):]:ax.set_visible(False)
        fig.suptitle('New experiment accounts — blue: strategy; red: SPY; purple: QQQ at base costs',fontsize=13);finish(fig,filename);pages.append(filename)
    catalog=pd.concat([a['catalog'],pd.DataFrame(entries)],ignore_index=True)
    assert len(catalog)==634 and catalog.id.is_unique;catalog.to_csv(DOC/'chart_catalog.csv',index=False)
    # Keep the existing complete gallery, replacing its superseded headline section.
    markdown=old(ROOT/'charts.md').read_text(encoding='utf8');tail=markdown[markdown.index('## Files and scope'):]
    tail=tail.replace('280 distinct base-cost strategies; 72 Models, 31 Defensive and 177 Portfolio variants.','346 distinct base-cost strategies; 105 Models, 31 Defensive and 210 Portfolio variants.')
    tail=tail.replace('356 chart-panel entries covering 280 main strategies and 76 additional diagnostic accounts.','634 chart-panel entries covering 346 main strategies and 288 additional diagnostic accounts.')
    tail=tail.replace('including individual seeds and 2x/4x fees, also in the new workbook tab.','including individual seeds and 2x/4x fees, also in Extended-hours.')
    tail=tail.replace('## Source reports and uncertainty','- [All 346 new experiment accounts](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/new_experiment_all_accounts.csv). New findings compares seed halves; New accounts includes individual seeds and fee stresses.\n\n## Source reports and uncertainty',1)
    tail=tail.replace('## Complete model and strategy gallery','- [Sealed loss/optimizer metrics](runs/ndx_loss_optimizer_20261024/metrics.csv) and [audit](runs/ndx_loss_optimizer_20261024/AUDIT.json)\n- [Sealed reference-network metrics](runs/ndx_reference_networks_20261024/metrics.csv) and [audit](runs/ndx_reference_networks_20261024/AUDIT.json)\n\n## Complete model and strategy gallery',1)
    intro=['# NASDAQ-100 master charts','','Updated October 24, 2026. Historical monthly NASDAQ-100 membership, January 3, 2022–December 31, 2025. All paths use the same 1,003 sessions.','',
      '## New findings: losses, optimizers and complementary networks','',
      'The fixed THINK+mixer AH+PM + restricted dynamic-hypergraph rank blend is the strongest new historical candidate. Its ten-seed portfolio returned **31.87% CAGR / 0.896 Sharpe at five holdings**, or **23.92% / 0.755 at ten**. Five-holding momentum SGD returned **25.05% / 0.719**, with 0.759 and 0.799 Sharpe in its two fixed seed halves. Top ten was primary; top five was a predeclared secondary portfolio test. No genuinely untouched-period result exists yet.','',
      '| Strategy | Holdings | Ten-seed CAGR | Ten-seed SR | Original-five SR | New-five SR | Max drawdown | 95% SR interval |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for method,k in [('baseline',10),('dynamic_restricted_blend',10),('tuned_sgd_mse',10),('baseline',5),('dynamic_restricted_blend',5),('tuned_sgd_mse',5)]:
        summary_method='fixed_adamw_mse' if method=='baseline' else method
        r=next(r for r in a['summaries'] if r['method']==summary_method and r['k']==k)
        intro.append(f'| {r["strategy"]} | {k} | {r["cagr"]:.2%} | {r["sharpe"]:.3f} | {r["original_sr"]:.3f} | {r["new_sr"]:.3f} | {r["max_drawdown"]:.2%} | [{r["sr_low"]:.3f}, {r["sr_high"]:.3f}] |')
    intro += ['', 'SPY: 10.85% CAGR / 0.498 Sharpe. QQQ: 11.93% / 0.469. Huber and pseudo-Huber weakened this AH+PM recipe. Standalone MASTER and PRISM-VQ transfers also underperformed. The restricted dynamic network alone returned 15.32% / 0.530, showing why a complementary blend can be more useful than choosing the strongest standalone CAGR. All negative cells remain in the workbook and gallery.','',
      '![New findings overview](docs/strategy_comparison_20261024/figures/new_findings_overview.png)','',
      '## Does the dynamic blend still use THINK?','',
      '**Yes.** One branch is the original THINK+mixer daily core plus its bounded AH+PM correction, trained with MSE and AdamW. The other is a separately trained dynamic hypergraph network using the same 25 stock-feature channels. Within each branch, fixed-seed forecasts are converted to eligible-stock percentile ranks and averaged. The final score is the fixed average of those two ensemble rank scores. One top-five or top-ten equal-weight portfolio is selected from it. This is not a 50/50 capital allocation between two portfolios.','',
      '```mermaid','flowchart LR','  A[Daily + AH + PM features] --> B[THINK + mixer AH+PM]','  A --> C[Restricted dynamic hypergraph]','  B --> D[Mean percentile ranks across seeds]','  C --> E[Mean percentile ranks across seeds]','  D --> F[50/50 rank average]','  E --> F','  F --> G[Select top 5 or top 10]','  G --> H[Equal weights at regular close]','```','',
      'Features end at 09:15 ET, the decision is at 09:20, and execution occurs at that session’s regular close. Portfolios rebalance every five exchange sessions. No after-hours trades or test-fitted blend weights are used.','',
      '## Seed robustness and uncertainty','',
      'The ten-holding blend’s original-five/new-five Sharpe split is 0.848 versus 0.467; its new-five result does not improve the matched baseline’s 0.480. At five holdings the blend improves on the matched baseline in both halves, while SGD is more consistent across those halves. These are initialization diagnostics on one shared history, not independent market replications. SGD’s low turnover appears partly related to stable stock-specific mixer preferences, which also need testing on new dates.','',
      'For the primary ten-holding blend, the paired 95% SR difference versus SPY is approximately [-0.161, 0.649]. Its improvement versus the baseline is not conclusive after the within-batch comparison adjustment. An additional secondary five-holding sensitivity gives a paired SR difference versus SPY of [-0.077, 0.849]; accounting for both holding counts and tested methods also leaves superiority over QQQ inconclusive. These secondary diagnostics do not turn the selected historical leader into a prespecified confirmation result. [Additional secondary sensitivity](outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/new_holdings_uncertainty.csv).','',
      'Absolute SR intervals use a fixed 3% annual hurdle and 10,000 circular-block draws, with 20 sessions primary and 5/60-session sensitivity. They condition on fitted paths, omit the full prior research search and retraining uncertainty, and do not give the probability of beating a benchmark.','',
      '![Seed robustness](docs/strategy_comparison_20261024/figures/new_findings_seeds.png)','',
      '![Primary uncertainty](docs/strategy_comparison_20261024/figures/new_findings_intervals.png)','',
      'The author networks are NASDAQ transfers with declared feature and training adaptations. The restricted dynamic model replaces the full sector/lead-lag graph inputs with one train-only correlation snapshot. PRISM’s current factor vintage does not establish historical publication availability. These are not reproductions of published paper performance.','',
      '## Best observed historical results','',
      'These are retrospective rankings across all 346 base-cost strategies, including diagnostic seed halves and portfolio variants. The ten-seed comparison above remains separate.','']
    intro+=md_table(leaders,True)+['','![Historical leaders](docs/strategy_comparison_20261024/figures/historical_leaders.png)','',
      '## Earlier unblended AH+PM holding-count test','',
      'Before the new optimizer/network tests, the original-five unblended AH+PM ensemble returned 14.51% CAGR / 0.485 SR at five holdings versus 22.39% / 0.700 at ten. Those saved-forecast results are preserved in AH PM holding counts and [the audited report](docs/ndx_extended_hours_top5_20261024/REPORT.md). They are a different strategy from the new dynamic rank blend.','']
    markdown='\n'.join(intro)+'\n'+tail+'\n### All new loss, optimizer, reference and rank-blend accounts\n\n'
    markdown+='\n\n'.join(f'![New experiment accounts {j+1}](docs/strategy_comparison_20261024/figures/{name})' for j,name in enumerate(pages))+'\n'
    (ROOT/'charts.md').write_text(markdown,encoding='utf8')
    comparison=old(OUT/'COMPARISON.md').read_text(encoding='utf8')
    title,rest=comparison.split('\n',1)
    comparison=title+'\n\nCurrent consolidation: **346 distinct base-cost strategies**. The refreshed [master charts](../../../charts.md) give the current leaderboard, seed comparisons and new model definitions. Earlier study-specific headline sections below record the preceding research snapshots.\n'+rest
    comparison+='\n## New loss, optimizer and reference-network findings\n\nSixty-six new base-cost ensembles are appended. See [master charts](../../../charts.md) for seed splits, uncertainty, model definitions and the complete new gallery. New findings in the workbook compares all ten-seed recipes and both seed halves. New accounts retains all 346 declared new-batch accounts.\n\n'
    comparison+='\n'.join(md_table(a['additions']))+'\n';(OUT/'COMPARISON.md').write_text(comparison,encoding='utf8')
    dump(DOC/'FIGURE_MANIFEST.json',{p.name:sha(p) for p in figures.glob('*.png')})
    print('Created 31 gallery pages and three overview figures; catalog now covers 634 unique accounts',flush=True)


def native_state(path):
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        styles=ET.fromstring(z.read('xl/styles.xml'))
        parts={key:list(styles.find('s:'+key,ns)) for key in ['fonts','fills','borders','cellXfs']}
        fmts={r.attrib['numFmtId']:r.attrib['formatCode'] for r in styles.findall('s:numFmts/s:numFmt',ns)}
        def canonical(n):return (n.tag,tuple(sorted(n.attrib.items())),n.text if n.text and n.text.strip() else '',tuple(canonical(c) for c in n))
        def signature(index):
            xf=parts['cellXfs'][index];attrs={k:v for k,v in xf.attrib.items() if k not in ['fontId','fillId','borderId','numFmtId','xfId'] and not k.startswith('apply')}
            return (tuple(sorted(attrs.items())),fmts.get(xf.attrib.get('numFmtId','0'),xf.attrib.get('numFmtId','0')),
              *[canonical(parts[key][int(xf.attrib.get(attr,'0'))]) for key,attr in [('fonts','fontId'),('fills','fillId'),('borders','borderId')]],tuple(canonical(c) for c in xf))
        signatures=[signature(i) for i in range(len(parts['cellXfs']))]
        names=[r.attrib['name'] for r in ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets',ns)]
        result={}
        for i,name in enumerate(names,1):
            sheet=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
            result[name]=dict(styles={c.attrib['r']:signatures[int(c.attrib.get('s','0'))] for c in sheet.findall('s:sheetData/s:row/s:c',ns)},
              rows={r.attrib['r']:dict(r.attrib) for r in sheet.findall('s:sheetData/s:row',ns)},
              cols=[dict(c.attrib) for c in sheet.findall('s:cols/s:col',ns)],
              panes=[dict(c.attrib) for c in sheet.findall('s:sheetViews/s:sheetView/s:pane',ns)],
              features={tag:[canonical(c) for c in sheet.findall('s:'+tag,ns)] for tag in ['dataValidations','conditionalFormatting','mergeCells','hyperlinks']})
        tables={}
        for name in z.namelist():
            if name.startswith('xl/tables/table') and name.endswith('.xml'):
                t=ET.fromstring(z.read(name));style=t.find('s:tableStyleInfo',ns)
                tables[t.attrib['name']]=dict(ref=t.attrib['ref'],style=canonical(style) if style is not None else None,
                    columns=[dict(c.attrib) for c in t.findall('s:tableColumns/s:tableColumn',ns)],
                    filters=[canonical(c) for c in t.findall('s:autoFilter/s:filterColumn',ns)])
        return result,tables


def audit(workbook_path=None,write_outputs=True):
    a=assemble()
    from update_ndx_top5_comparison_20261024 import cells
    before=cells(old(BOOK));after=cells(workbook_path or BOOK)
    for name,values in before.items():
        for address,value in values.items():assert after[name].get(address)==value,(name,address)
    native_before,tables_before=native_state(old(BOOK));native_after,tables_after=native_state(workbook_path or BOOK)
    for name,state in native_before.items():
        for address,style in state['styles'].items():assert native_after[name]['styles'][address]==style,('prior style changed',name,address)
        for row,attrs in state['rows'].items():assert native_after[name]['rows'][row]==attrs,('prior row layout changed',name,row)
        for field in ['cols','panes','features']:assert native_after[name][field]==state[field],('prior feature changed',name,field)
    for name,table in tables_before.items():
        new=tables_after[name]
        for field in ['style','columns','filters']:assert new[field]==table[field],('native table changed',name,field)
        if name not in ['Comparison1','Comparison3']:assert new['ref']==table['ref']
    assert tables_after['Comparison1']['ref']=='A5:H110' and tables_after['Comparison3']['ref']=='A5:H217'
    for name in ['New findings','New accounts']:
        pane=native_after[name]['panes'][0];assert float(pane['xSplit'])==1 and float(pane['ySplit'])==5
    for section in ['Models','Portfolio variants']:
        existing=max(int(re.sub('[A-Z]','',key)) for key in before[section])
        selected=[r for r in a['additions'] if r['section']==section]
        for i,r in enumerate(selected,existing+1):
            expected=[r[n] for n in ['strategy','study','sharpe','cagr','rankic','max_drawdown','sr_low','sr_high']]
            for col,value in zip('ABCDEFGH',expected):
                actual=after[section][f'{col}{i}']
                if isinstance(value,str):assert actual==value
                else:np.testing.assert_allclose(actual,value,atol=1e-12,rtol=1e-12)
    for i,r in enumerate(a['summaries'],6):
        expected=[r['strategy'],r['k'],r['sharpe'],r['cagr'],r['rankic'] if r['rankic'] is not None else 'n.a.',r['max_drawdown'],r['sr_low'],r['sr_high'],r['original_sr'] if r['original_sr'] is not None else 'n.a.',r['new_sr'] if r['new_sr'] is not None else 'n.a.',r['original_cagr'] if r['original_cagr'] is not None else 'n.a.',r['new_cagr'] if r['new_cagr'] is not None else 'n.a.']
        for col,value in zip('ABCDEFGHIJKL',expected):
            actual=after['New findings'][f'{col}{i}']
            if isinstance(value,str):assert actual==value
            else:np.testing.assert_allclose(actual,value,atol=1e-12,rtol=1e-12)
    for i,r in enumerate(a['details'],6):
        expected=[r['strategy'],r['study'],r['group'],r['k'],r['cost_multiplier'],r['seed'],r['sharpe'],r['cagr'],r['rankic'] if r['rankic'] is not None else 'n.a.',r['max_drawdown'],r['sr_low'],r['sr_high'],r['role']]
        for col,value in zip('ABCDEFGHIJKLM',expected):
            actual=after['New accounts'][f'{col}{i}']
            if isinstance(value,str):assert actual==value,(i,col)
            else:np.testing.assert_allclose(actual,value,atol=1e-12,rtol=1e-12)
    for i,row in enumerate(NOTES,34):
        assert after['Method'][f'A{i}']==row[0] and after['Method'][f'B{i}']==row[1]
    assert len(after)==8
    for p in re.findall(r'\]\(([^)]+)\)',(ROOT/'charts.md').read_text()):assert (ROOT/p).exists(),p
    assert len(read(DOC/'chart_catalog.csv'))==634 and len(read(OUT/'all_strategies.csv'))==346
    assert len(read(OUT/'new_holdings_uncertainty.csv'))==172
    old_figures=json.loads(old(DOC/'FIGURE_MANIFEST.json').read_text())
    for name,h in old_figures.items():
        if name!='historical_leaders.png':assert sha(DOC/'figures'/name)==h,('prior gallery changed',name)
    with np.load(OUT/'analysis_paths.npz') as f:
        assert f['returns'].shape==(1003,346)
        np.testing.assert_array_equal(f['ids'][:280],a['old_ids']);np.testing.assert_array_equal(f['returns'][:,:280],a['old_returns'])
    proof=dict(status='passed',main_strategies=346,sections={'Models':105,'Defensive':31,'Portfolio variants':210},
      added_main_rows=66,new_account_rows=346,seed_comparison_rows=26,workbook_sheets=8,chart_catalog_entries=634,
      additional_diagnostic_accounts=288,prior_workbook_cells_preserved=True,prior_280_return_paths_preserved=True,
      prior_native_styles_layout_tables_and_panes_preserved=True,all_346_new_account_rows_verified=True,
      source_audits={key:json.loads((ROOT/'runs'/name/'AUDIT.json').read_text()) for key,name in STUDIES.items()},
      output_sha256={str(p.relative_to(ROOT)):sha(p) for p in [BOOK,ROOT/'charts.md',OUT/'all_strategies.csv',DOC/'chart_catalog.csv',
        OUT/'new_experiment_all_accounts.csv',OUT/'new_findings_seed_summary.csv',OUT/'new_holdings_uncertainty.csv',OUT/'sharpe_block_sensitivity.csv',DOC/'FIGURE_MANIFEST.json']},
      analysis_source_sha256=sha(__file__),workbook_builder_sha256=sha(OUT/'build_workbook.mjs'))
    if not write_outputs:
        proof['status']='passed_staged_workbook';proof.pop('output_sha256')
        print(json.dumps(proof,indent=2));return
    dump(OUT/'AUDIT.json',proof);dump(DOC/'AUDIT.json',proof)
    record=json.loads(old(OUT/'ANALYSIS.json').read_text());record.update(status='completed_audited',strategies=346,workbook_sheets=8,
      sections=proof['sections'],chart_panels=634,additional_diagnostic_accounts=288,new_experiment_accounts=346,new_findings_source=Path(__file__).name,
      workbook_comparison_rows=850,gallery_pages=int(read(DOC/'chart_catalog.csv').figure.nunique()),figures=len(list((DOC/'figures').glob('*.png'))),
      analysis_source_sha256=sha(__file__),source_policy='Preserved 280 prior main rows and paths; appended 66 base-cost ensembles, 346 declared batch accounts and 278 new unique chart accounts. No new models fit for consolidation.')
    record.setdefault('source_hashes',{}).update({str((ROOT/'runs'/name/'FINAL_PUBLICATION.json').relative_to(ROOT)):sha(ROOT/'runs'/name/'FINAL_PUBLICATION.json') for name in STUDIES.values()})
    dump(OUT/'ANALYSIS.json',record)
    print(json.dumps({k:v for k,v in proof.items() if k not in ['source_audits','output_sha256']},indent=2))


def publish():
    if sha(BOOK)!=sha(old(BOOK)):raise ValueError('Workbook changed since the snapshot; preserve and reconcile the new user edits before publishing')
    staged=BEFORE/'updated_staged.xlsx';audit(staged,False)
    shutil.copyfile(staged,BOOK)
    audit()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','charts','audit','check-staged','secondary_intervals','publish']);stage=parser.parse_args().stage
    if stage=='check-staged':audit(BEFORE/'updated_staged.xlsx',False)
    else:globals()[stage]()
