"""Integrate qualified drawdown controls into the existing master artifacts.

This is a publication update. Completed study protocols and metrics stay intact.
Prepare a snapshot/payload, use the bundled JS workbook builder, audit, publish.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'hypershift').is_dir())
WORK=ROOT/'outputs/01a178f0-fcc4-7a02-96bf-702b11c930d6/master_update_20261026'
BEFORE=WORK/'before'
STUDY=ROOT/'docs/ndx_drawdown_20261026'
DOC=STUDY/'master_update'
OLD=ROOT/'outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020'
GALLERY=ROOT/'docs/strategy_comparison_20261024'
BOOK=ROOT/'docs/NASDAQ100_strategy_comparison.xlsx'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
BASE='equal5/full'
BEST='equal5/basketvol20_target20_weekly'
ISSUER='issuer5/basketvol20_target20_weekly'
RETURN='issuer5/ma200off50_basketvol20_target25_weekly'
LOW='equal5/ma200off50_basketvol20_target20_weekly'
NAMES={BASE:'Full-exposure blend',BEST:'20% basket volatility target',
       'equal5/ma200off50_basketvol20_target25_weekly':'25% volatility target + trend cap',
       LOW:'20% volatility target + trend cap',ISSUER:'Five issuers + 20% volatility target',
       'issuer5/ma200off50_basketvol20_target20_weekly':'Five issuers + 20% volatility target + trend cap',
       RETURN:'Five issuers + 25% volatility target + trend cap'}
GROUPS={'original_five':'Original five','new_five':'New five','ten_seed':'Ten seeds'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def snapshot():
    if (BEFORE/'MANIFEST.json').exists():return
    paths=[BOOK,ROOT/'charts.md',OLD/'all_strategies.csv',OLD/'analysis_paths.npz',GALLERY/'chart_catalog.csv']
    record={}
    for path in paths:
        target=BEFORE/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);record[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    write(BEFORE/'MANIFEST.json',record)


def old(path):return BEFORE/path.relative_to(ROOT)


def inputs():
    publication=json.loads((STUDY/'PUBLICATION.json').read_text())
    for path,h in publication['artifact_sha256'].items():
        if sha(STUDY/path)!=h:raise ValueError('Study publication changed: '+path)
    if json.loads((STUDY/'AUDIT.json').read_text())['status']!='passed':raise ValueError('Audit not passed')
    metrics=pd.read_csv(STUDY/'metrics.csv');summary=pd.read_csv(STUDY/'summary.csv')
    with np.load(ROOT/'runs/ndx_drawdown_20261026/batch1/paths.npz') as saved:
        dates=pd.DatetimeIndex(saved['dates']);paths=dict(zip(saved['ids'],saved['returns'].T))
    with np.load(old(OLD/'analysis_paths.npz')) as saved:
        np.testing.assert_array_equal(saved['dates'],dates.strftime('%Y-%m-%d').to_numpy())
        prior_paths=dict(zip(saved['ids'],saved['returns'].T))
    return metrics,summary,dates,paths,prior_paths


def intervals(paths,keys):
    matrix=np.column_stack([paths[key] for key in keys]);n=len(matrix)
    rng=np.random.default_rng(20261026);draws=[]
    for _ in range(100):
        starts=rng.integers(0,n,(100,int(np.ceil(n/20))))
        index=((starts[:,:,None]+np.arange(20))%n).reshape(100,-1)[:,:n]
        sample=matrix[index]
        draws.append(np.sqrt(252)*(sample.mean(1)-(1.03**(1/252)-1))/sample.std(1,ddof=1))
    bounds=np.quantile(np.concatenate(draws),[.025,.975],axis=0)
    return {key:(float(bounds[0,j]),float(bounds[1,j])) for j,key in enumerate(keys)}


def prepare():
    snapshot();DOC.mkdir(parents=True,exist_ok=True)
    metrics,summary,dates,paths,prior_paths=inputs()
    qualified=summary[summary.target_met].sort_values('ten_seed_sharpe',ascending=False).recipe.tolist()
    selected=metrics[metrics.recipe.isin(qualified)].copy()
    bounds=intervals(paths,selected.id.tolist())
    selected['sr_low']=[bounds[key][0] for key in selected.id]
    selected['sr_high']=[bounds[key][1] for key in selected.id]
    selected.to_csv(DOC/'leader_metrics.csv',index=False,lineterminator='\n')
    prior=pd.read_csv(old(OLD/'all_strategies.csv'))
    additions=[]
    for row in selected.to_dict('records'):
        additions.append(dict(id='drawdown/'+row['id'],study='Drawdown controls',
                              strategy=NAMES[row['recipe']]+' / '+GROUPS[row['group']],policy=row['recipe'],k=5,
                              section='Defensive',rankic=None,sharpe=row['sharpe'],cagr=row['annualized_return'],
                              max_drawdown=row['maximum_drawdown'],sr_low=row['sr_low'],sr_high=row['sr_high']))
    combined=pd.concat([prior,pd.DataFrame(additions)],ignore_index=True)
    if len(prior)!=346 or len(combined)!=364 or combined.id.duplicated().any():raise ValueError('Master coverage mismatch')
    combined.to_csv(WORK/'all_strategies.csv',index=False,lineterminator='\n')
    all_paths=prior_paths|{'drawdown/'+key:paths[key] for key in selected.id}
    np.savez_compressed(WORK/'analysis_paths.npz',ids=combined.id.to_numpy(dtype=str),dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str),returns=np.column_stack([all_paths[key] for key in combined.id]))
    focus_order=[(BEST,'original_five','Highest observed Sharpe'),
                 ('equal5/ma200off50_basketvol20_target25_weekly','original_five','Higher-return original-five alternative'),
                 (LOW,'original_five','Lower-drawdown original-five alternative'),
                 (RETURN,'original_five','Issuer-aware return-preserving alternative'),
                 (ISSUER,'original_five','Issuer-aware 20% target'),
                 (ISSUER,'ten_seed','Highest ten-seed Sharpe'),
                 (RETURN,'ten_seed','Return-preserving ten-seed alternative'),
                 (BASE,'original_five','Prior highest-Sharpe reference'),
                 (BASE,'ten_seed','Matched ten-seed reference')]
    headline=[]
    for recipe,group,role in focus_order:
        row=metrics[(metrics.recipe==recipe)&(metrics.group==group)].iloc[0]
        key='drawdown/'+row.id if recipe!=BASE else 'reference/dynamic_restricted_blend/'+group+'/top5'
        main=combined[combined.id==key].iloc[0]
        headline.append(dict(recipe=recipe,group=GROUPS[group],label=NAMES[recipe],role=role,
                             sharpe=float(row.sharpe),cagr=float(row.annualized_return),dd=float(row.maximum_drawdown),
                             low=float(main.sr_low),high=float(main.sr_high)))
    for ticker in ['SPY','QQQ']:
        row=prior[prior.id=='original/'+ticker+'/10'].iloc[0]
        headline.append(dict(recipe=ticker,group='Benchmark',label=ticker,role='Buy and hold',sharpe=float(row.sharpe),cagr=float(row.cagr),dd=float(row.max_drawdown),low=float(row.sr_low),high=float(row.sr_high)))
    controls=metrics[metrics.recipe==BASE].copy()
    stress=pd.read_csv(STUDY/'cost_stress.csv');extra=stress[stress.cost_multiplier!=1]
    accounts=pd.concat([metrics,extra],ignore_index=True).sort_values(['recipe','group','cost_multiplier'])
    if len(accounts)!=969:raise ValueError('Account coverage mismatch')
    flags=summary.set_index('recipe').target_met.to_dict()
    rows=[]
    for row in accounts.to_dict('records'):
        rows.append([NAMES.get(row['recipe'],row['recipe']),GROUPS[row['group']],int(row['cost_multiplier']),
                     row['sharpe'],row['annualized_return'],row['maximum_drawdown'],row['annualized_volatility'],
                     row['turnover_annual'],row['cost_dollars'],row['terminal_nav'],bool(flags[row['recipe']]),row['recipe']])
    # Frozen comparison paths drive wealth and drawdown charts for each ensemble.
    series=[]
    for group in ['original_five','ten_seed']:
        for recipe in [BEST,RETURN,ISSUER,LOW,BASE]:
            series.append((recipe+'/'+group,paths[recipe+'/'+group]))
    series += [('SPY',prior_paths['original/SPY/10']),('QQQ',prior_paths['original/QQQ/10'])]
    labels=['20% vol target','Issuer + 25% + trend','Issuer + 20% target','20% + trend','Full exposure','SPY','QQQ']
    chart_series=[]
    for group in ['original_five','ten_seed']:
        chart_series.append([next(j for j,(key,_) in enumerate(series) if key==recipe+'/'+group) for recipe in [BEST,RETURN,ISSUER,LOW,BASE]]+[10,11])
    chart_legends=[]
    for group in ['original_five','ten_seed']:
        legend=[]
        for label,recipe in zip(labels,[BEST,RETURN,ISSUER,LOW,BASE,'SPY','QQQ']):
            if recipe in ['SPY','QQQ']:
                row=prior[prior.id=='original/'+recipe+'/10'].iloc[0]
                sr,cagr=row.sharpe,row.cagr
            else:
                row=metrics[(metrics.recipe==recipe)&(metrics.group==group)].iloc[0]
                sr,cagr=row.sharpe,row.annualized_return
            legend.append(f'{label} | SR {sr:.3f} | CAGR {cagr:.2%}')
        chart_legends.append(legend)
    # Use seven actual year labels as categories; full daily source observations remain.
    payload=dict(input_workbook=str(old(BOOK)),input_sha256=sha(old(BOOK)),output_workbook=str(WORK/'master_staged.xlsx'),
                 headline=headline,additions=additions,accounts=rows,
                 summary=[[NAMES[recipe],5,*[float(summary.set_index('recipe').loc[recipe,name]) for name in
                          ['ten_seed_sharpe','ten_seed_annualized_return','ten_seed_maximum_drawdown','original_five_sharpe','original_five_annualized_return','original_five_maximum_drawdown','new_five_sharpe','new_five_annualized_return','new_five_maximum_drawdown']],True] for recipe in qualified],
                 dates=['2021-12-31']+dates.strftime('%Y-%m-%d').tolist(),
                 returns=[[0.]*len(series)]+np.column_stack([values for _,values in series]).tolist(),
                 source_headers=[key for key,_ in series],chart_indices=chart_series,chart_labels=labels,chart_legends=chart_legends,
                 chart_colors=['#086A9F','#B36A16','#087D85','#639C2F','#69727D','#C44832','#7544A0'],
                 method_notes=[['October 26 drawdown-control integration','Six qualifying recipes, 18 base-cost seed-group accounts added to Defensive. New findings compares their leaders and preserves October 24 findings below.'],
                               ['Drawdown search and timing','305 frozen recipes, 915 base-cost accounts and 54 additional higher-cost accounts. Weekly and daily risk rules use prior-close inputs with subsequent regular-close fills. No 2026 performance.'],
                               ['Risk-control recipe','Basket volatility uses proposed weights and prior 20-session constituent returns. Exposure is capped at 100%. Cash earns zero. Issuer rules pair GOOG/GOOGL, FOX/FOXA and LBTYA/LBTYK.'],
                               ['Interpretation of new intervals','10,000 paired-date circular-block draws, 20 sessions. Absolute 95% SR intervals condition on fitted paths and omit full search selection. Paired SR improvement remains inconclusive.'],
                               ['Drawdown source','docs/ndx_drawdown_20261026/REPORT.md and master_update/leader_metrics.csv. Full failed variants and higher costs are retained in Drawdown accounts. No new model fitting.']])
    with ZipFile(old(BOOK)) as z:
        old_sheet=ET.fromstring(z.read('xl/worksheets/sheet7.xml'))
        payload['prior_findings_row_heights']={int(r.attrib['r']):float(r.attrib['ht']) for r in old_sheet.findall('s:sheetData/s:row',NS) if 'ht' in r.attrib}
    write(WORK/'payload.json',payload)
    write(DOC/'PREPARED.json',dict(status='prepared',prior_snapshot_sha256=sha(BEFORE/'MANIFEST.json'),new_main_accounts=18,master_accounts=364,all_drawdown_accounts=969))
    plots(metrics,selected,qualified,dates,paths,prior_paths,combined,all_paths)
    markdown(payload)
    print('Prepared workbook payload, focused comparison, expanded master graph and 18 gallery panels')


def plots(metrics,selected,qualified,dates,paths,prior_paths,combined,all_paths):
    x=dates.insert(0,pd.Timestamp('2021-12-31'))
    colors=['#086A9F','#B36A16','#087D85','#639C2F','#69727D','#C44832','#7544A0']
    short=['20% volatility target','Issuer limit + 25% target + trend','Issuer limit + 20% target','20% target + trend','Prior full-exposure blend','SPY','QQQ']
    fig=plt.figure(figsize=(18,11))
    grid=fig.add_gridspec(3,2,height_ratios=[3.2,1.75,1.65],hspace=.20,wspace=.13)
    axes=np.array([[fig.add_subplot(grid[0,c]) for c in range(2)],
                   [fig.add_subplot(grid[1,c]) for c in range(2)]])
    metric_index=metrics.set_index(['recipe','group'])
    wealth_bounds=[]
    for col,group in enumerate(['original_five','ten_seed']):
        recipes=[BEST,RETURN,ISSUER,LOW,BASE,'SPY','QQQ']
        for j,recipe in enumerate(recipes):
            a=paths[recipe+'/'+group] if recipe not in ['SPY','QQQ'] else prior_paths['original/'+recipe+'/10']
            wealth=100*np.r_[1,np.cumprod(1+a)]
            wealth_bounds.extend([wealth.min(),wealth.max()])
            if recipe in ['SPY','QQQ']:
                row=combined[combined.id=='original/'+recipe+'/10'].iloc[0];sr,cagr=row.sharpe,row.cagr
            else:
                row=metric_index.loc[(recipe,group)];sr,cagr=row.sharpe,row.annualized_return
            label=f'{short[j]}\nSR {sr:.3f}  |  CAGR {cagr:.2%}'
            style=dict(color=colors[j],lw=2.2 if j!=4 else 1.3,ls='--' if j in [5,6] else '-')
            axes[0,col].plot(x,wealth,label=label,**style)
            axes[1,col].plot(x,wealth/np.maximum.accumulate(wealth)-1,**style)
        axes[0,col].set_title('Original five seeds' if col==0 else 'All ten seeds',fontsize=13,weight='bold')
        axes[1,col].set_ylim(-.45,.01)
        axes[1,col].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
        for ax in axes[:,col]:
            ax.grid(alpha=.18);ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));ax.set_xlim(x[0],x[-1])
        axes[0,col].tick_params(labelbottom=False)
        legend_ax=fig.add_subplot(grid[2,col]);legend_ax.axis('off')
        handles,labels=axes[0,col].get_legend_handles_labels()
        legend=legend_ax.legend(handles,labels,loc='upper left',ncol=2,fontsize=9.6,frameon=False,handlelength=2.4,labelspacing=1.0,columnspacing=1.8)
        for i,text in enumerate(legend.get_texts()):
            if i in [0,2,5,6]:text.set_weight('bold')
    axes[0,0].set_ylabel('Growth of $100 after modeled costs')
    axes[1,0].set_ylabel('Drawdown from previous peak')
    for ax in axes[0]:ax.set_ylim(min(wealth_bounds)*.95,max(wealth_bounds)*1.05)
    fig.suptitle('Highest-Sharpe models compared with SPY and QQQ',fontsize=19,weight='bold',y=.97)
    fig.text(.5,.018,'2022-2025 exploratory development. SR and CAGR use the complete net return path. Drawdown = wealth / previous peak - 1.',ha='center',fontsize=10,color='#555555')
    fig.subplots_adjust(left=.065,right=.985,top=.91,bottom=.065)
    fig.savefig(DOC/'high_sharpe_comparison.png',dpi=160);plt.close(fig)
    fig,(ax,ddax)=plt.subplots(2,1,figsize=(16,10),sharex=True,gridspec_kw={'height_ratios':[3,1.5],'hspace':.10})
    highlighted=['drawdown/'+BEST+'/original_five','drawdown/'+ISSUER+'/ten_seed','original/SPY/10','original/QQQ/10']
    for key in combined.id:
        if key not in highlighted:
            wealth=100*np.r_[1,np.cumprod(1+all_paths[key])]
            ax.plot(x,wealth,color='#939da8',lw=.6,alpha=.20)
            ddax.plot(x,wealth/np.maximum.accumulate(wealth)-1,color='#939da8',lw=.6,alpha=.12)
    for key,color,label in zip(highlighted,[colors[0],colors[2],colors[5],colors[6]],['Highest SR: original five','Highest SR: ten seeds','SPY','QQQ']):
        row=combined[combined.id==key].iloc[0]
        wealth=100*np.r_[1,np.cumprod(1+all_paths[key])]
        ax.plot(x,wealth,lw=2.7,color=color,label=f'{label} (SR {row.sharpe:.3f}, CAGR {row.cagr:.2%})')
        ddax.plot(x,wealth/np.maximum.accumulate(wealth)-1,lw=2.2,color=color)
    ax.set_title('All 364 main strategies, with the new Sharpe leaders and benchmarks emphasized',fontsize=15,weight='bold')
    ax.set_ylabel('Growth of $100 after modeled costs');ax.grid(alpha=.18);ax.set_xlim(x[0],x[-1]);ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    leg=ax.legend(loc='upper left',fontsize=10,frameon=False)
    for t in leg.get_texts():t.set_weight('bold')
    ddax.set_ylabel('Drawdown from previous peak');ddax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0));ddax.grid(alpha=.18)
    fig.text(.5,.02,'346 prior strategies + 18 qualifying risk-control accounts. All 915 drawdown-study base accounts remain available separately. Historical selection; no untouched confirmation.',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.065,right=.985,top=.94,bottom=.08);fig.savefig(DOC/'master_all_strategies.png',dpi=160);plt.close(fig)
    catalog=pd.read_csv(old(GALLERY/'chart_catalog.csv'));additions=[]
    for page in range(2):
        fig,axes=plt.subplots(3,3,figsize=(17,12),sharex=True)
        for row,recipe in enumerate(qualified[page*3:page*3+3]):
            for col,group in enumerate(['original_five','new_five','ten_seed']):
                ax=axes[row,col];a=paths[recipe+'/'+group]
                m=selected[(selected.recipe==recipe)&(selected.group==group)].iloc[0]
                ax.plot(x,np.r_[1,np.cumprod(1+a)],lw=1.7,color='#086A9F')
                for ticker,color in [('SPY',colors[5]),('QQQ',colors[6])]:ax.plot(x,np.r_[1,np.cumprod(1+prior_paths['original/'+ticker+'/10'])],color=color,lw=.9)
                label=NAMES[recipe]+' / '+GROUPS[group]
                import textwrap
                ax.set_title(textwrap.fill(label,43)+f'\nSR {m.sharpe:.3f} | CAGR {m.annualized_return:.2%} | DD {m.maximum_drawdown:.2%}',fontsize=9)
                ax.grid(alpha=.18);ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                additions.append(dict(id='drawdown/'+recipe+'/'+group,strategy=label,figure=f'drawdown_leaders_{page+1:02d}.png',section='Drawdown-control qualifiers',figure_path='docs/ndx_drawdown_20261026/master_update/'+f'drawdown_leaders_{page+1:02d}.png'))
        fig.suptitle('Drawdown-control qualifiers at base costs. Blue: strategy; red: SPY; purple: QQQ',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(DOC/f'drawdown_leaders_{page+1:02d}.png',dpi=140);plt.close(fig)
    pd.concat([catalog,pd.DataFrame(additions)],ignore_index=True).to_csv(WORK/'chart_catalog.csv',index=False,lineterminator='\n')


def markdown(payload):
    prior=old(ROOT/'charts.md').read_text(encoding='utf8')
    body=prior[prior.index('## What the dynamic hypergraph does, in plain language'):]
    body=body.replace('346 distinct base-cost strategies; 105 Models, 31 Defensive and 210 Portfolio variants.','364 distinct base-cost strategies; 105 Models, 49 Defensive and 210 Portfolio variants.')
    body=body.replace('634 chart-panel entries covering 346 main strategies and 288 additional diagnostic accounts.','652 chart-panel entries covering 364 main strategies and 288 additional diagnostic accounts.')
    # The old snapshot remains dated, rather than claiming the current SR lead.
    prior_head=prior[prior.index('## Best model first:'):prior.index('## What the dynamic hypergraph does, in plain language')]
    prior_head=prior_head.replace('## Best model first:','## October 24 full-exposure snapshot:').replace('Our best observed historical model is','The highest-Sharpe full-exposure model in the October 24 comparison was')
    prior_head=prior_head.replace('## Master graph: every strategy, with the leader and SPY/QQQ emphasized','## Earlier master graph: October 24 models')
    head=['# NASDAQ-100 master charts','','Updated October 26, 2026. Historical monthly NASDAQ-100 membership, January 3, 2022–December 31, 2025. Every path uses the same 1,003 sessions.','',
          '## Highest-Sharpe models versus SPY and QQQ','',
          '![New high-Sharpe models compared with SPY and QQQ, separated by seed ensemble](docs/ndx_drawdown_20261026/master_update/high_sharpe_comparison.png)','',
          '**The highest observed Sharpe is now 1.199** for the original-five blend with a 20% basket-volatility target: **28.96% CAGR and -17.78% drawdown**. The **highest ten-seed Sharpe is 1.112**, using five-issuer selection plus a 20% volatility target: **26.81% CAGR and -21.69% drawdown**. The 25% target plus issuer/trend control retains more ten-seed return at **28.70% CAGR / 1.085 Sharpe / -22.98% drawdown**.','',
          '| Strategy | Seed ensemble | Sharpe | CAGR | Max drawdown | 95% SR interval |','| --- | --- | ---: | ---: | ---: | --- |']
    for row in payload['headline']:
        bold=row['role'] in ['Highest observed Sharpe','Highest ten-seed Sharpe'] or row['recipe'] in ['SPY','QQQ']
        values=[row['label'],row['group'],f'{row["sharpe"]:.3f}',f'{row["cagr"]:.2%}',f'{row["dd"]:.2%}',f'[{row["low"]:.3f}, {row["high"]:.3f}]']
        if bold:values=['**'+v+'**' for v in values]
        head.append('| '+' | '.join(values)+' |')
    head+=['','The figure compares four new risk-control recipes, the corresponding full-exposure blend, and SPY/QQQ. Left: original five seeds; right: all ten seeds. Each panel has its own legend with full-period SR and CAGR for every line. Drawdown panels below the wealth charts show the percentage decline from each account\'s previous peak; 0% means a new or matched high. The original-five leader is a diagnostic ensemble; the ten-seed comparison remains the broader seed check.','',
           '**Historical development findings:** six of 305 frozen recipes passed the screen. All 915 base-cost accounts and 54 additional cost stresses are retained in the workbook\'s Drawdown accounts sheet and the study report. The new-five half remains weaker, and paired SR-improvement intervals include zero. These results are selected from repeatedly inspected 2022-2025 history.','',
           'The new rules keep the THINK/dynamic forecasts and 50:50 rank blend. They change issuer selection and/or stock exposure using prior-close basket volatility and an optional QQQ/200-session-average cap. Five-session decisions remain at 09:20 ET with regular-close fills. No leverage; idle cash earns zero. A volatility target is not a guaranteed drawdown limit. [Complete drawdown study and audit](docs/ndx_drawdown_20261026/REPORT.md).','',
           '## Master graph: all 364 main strategies','',
           '![All 364 main strategies with the newest Sharpe leaders, SPY and QQQ emphasized](docs/ndx_drawdown_20261026/master_update/master_all_strategies.png)','',
           'The thick blue and teal lines identify the original-five and ten-seed Sharpe leaders. SPY and QQQ are emphasized in red and purple. The other 360 paths are thin gray lines. This combines the previous 346 main strategies with 18 seed-group accounts from the six qualifying risk-control recipes. The complete 305-recipe search remains separately disclosed, including failures.','',
           '## New drawdown-control gallery','',
           '![Drawdown-control qualifiers, first three recipes](docs/ndx_drawdown_20261026/master_update/drawdown_leaders_01.png)','',
           '![Drawdown-control qualifiers, next three recipes](docs/ndx_drawdown_20261026/master_update/drawdown_leaders_02.png)','',
           prior_head,body]
    (WORK/'charts.md').write_text('\n'.join(head),encoding='utf8',newline='\n')


def cell_state(path):
    """Read semantic cell values, formulas and resolved styles without authoring."""
    with ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            shared=[''.join(item.itertext()) for item in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        styles=ET.fromstring(z.read('xl/styles.xml'))
        def canonical(node):
            if node is None:return None
            return (node.tag,tuple(sorted(node.attrib.items())),node.text.strip() if node.text and node.text.strip() else '',
                    tuple(sorted((canonical(child) for child in node),key=repr)))
        fonts=list(styles.find('s:fonts',NS));fills=list(styles.find('s:fills',NS));borders=list(styles.find('s:borders',NS))
        formats={item.attrib['numFmtId']:item.attrib['formatCode'] for item in styles.findall('s:numFmts/s:numFmt',NS)}
        signatures=[]
        for xf in styles.find('s:cellXfs',NS):
            attributes={k:v for k,v in xf.attrib.items() if k not in ['fontId','fillId','borderId','numFmtId','xfId'] and not k.startswith('apply')}
            signatures.append((tuple(sorted(attributes.items())),formats.get(xf.attrib.get('numFmtId','0'),xf.attrib.get('numFmtId','0')),
                               canonical(fonts[int(xf.attrib.get('fontId','0'))]),canonical(fills[int(xf.attrib.get('fillId','0'))]),
                               canonical(borders[int(xf.attrib.get('borderId','0'))]),tuple(canonical(child) for child in xf)))
        workbook=ET.fromstring(z.read('xl/workbook.xml'))
        rels={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        output={}
        for node in workbook.find('s:sheets',NS):
            target=rels[node.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            sheet=ET.fromstring(z.read(target));cells={}
            for cell in sheet.findall('s:sheetData/s:row/s:c',NS):
                v=cell.find('s:v',NS);formula=cell.find('s:f',NS);inline=cell.find('s:is',NS)
                value=v.text if v is not None else None
                kind=cell.attrib.get('t')
                if kind=='s' and value is not None:value=shared[int(value)]
                elif kind=='inlineStr' and inline is not None:value=''.join(inline.itertext())
                elif kind=='b' and value is not None:value=value=='1'
                elif kind in [None,'n'] and value is not None:value=float(value)
                if kind=='e':raise ValueError('Spreadsheet error: '+node.attrib['name']+'!'+cell.attrib['r']+' '+str(value))
                if value is not None or formula is not None:
                    cells[cell.attrib['r']]=dict(value=value,formula=formula.text if formula is not None else None,
                                                style=signatures[int(cell.attrib.get('s','0'))])
            output[node.attrib['name']]=dict(cells=cells,
                columns=[dict(c.attrib) for c in sheet.findall('s:cols/s:col',NS)],
                panes=[dict(c.attrib) for c in sheet.findall('s:sheetViews/s:sheetView/s:pane',NS)])
        tables={}
        for name in z.namelist():
            if name.startswith('xl/tables/table') and name.endswith('.xml'):
                table=ET.fromstring(z.read(name));tables[table.attrib['name']]=dict(ref=table.attrib['ref'],style=canonical(table.find('s:tableStyleInfo',NS)),
                    columns=[{k:v for k,v in c.attrib.items() if not k.endswith('}uid')} for c in table.findall('s:tableColumns/s:tableColumn',NS)])
        charts=[]
        cns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        for name in z.namelist():
            if '/charts/chart' in name and name.endswith('.xml'):
                chart=ET.fromstring(z.read(name));series=chart.findall('.//c:ser',cns)
                charts.append(dict(part=name,series=len(series),formulas=[e.text for e in chart.findall('.//c:f',cns)],
                                   colors=[e.attrib['val'] for e in chart.findall('.//c:ser/c:spPr/a:ln/a:solidFill/a:srgbClr',cns)]))
    return output,tables,charts


def audit():
    opening_view()
    payload=json.loads((WORK/'payload.json').read_text())
    before,old_tables,_=cell_state(old(BOOK));after,new_tables,charts=cell_state(WORK/'master_staged.xlsx')
    if list(after)[:8]!=list(before) or set(after)-set(before)!={'Drawdown accounts','Leader paths'}:raise ValueError('Unexpected tab change')
    total=0;style_changes=[]
    for name,sheet in before.items():
        for address,cell in sheet['cells'].items():
            mapped=address
            if name=='New findings':
                column=''.join(c for c in address if c.isalpha());row=int(''.join(c for c in address if c.isdigit()))
                mapped=column+str(row+80)
            found=after[name]['cells'].get(mapped)
            if found is None:raise ValueError('Prior cell missing: '+name+'!'+address)
            if cell['formula']!=found['formula']:raise ValueError('Prior formula changed: '+name+'!'+address)
            a,b=cell['value'],found['value']
            if isinstance(a,(float,int)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
            elif a!=b:raise ValueError('Prior value changed: '+name+'!'+address)
            if cell['style']!=found['style']:style_changes.append(name+'!'+address)
            total+=1
        if name!='New findings' and sheet['panes']!=after[name]['panes']:raise ValueError('Prior panes changed: '+name)
        if sheet['columns']!=after[name]['columns']:raise ValueError('Prior column layout changed: '+name)
    if style_changes:raise ValueError('Prior cell styles changed: '+str(style_changes[:12]))
    for name,table in old_tables.items():
        current=new_tables[name]
        expected='A5:H56' if name=='Comparison2' else 'A85:L111' if name=='NewFindings' else table['ref']
        if current['ref']!=expected or current['style']!=table['style'] or current['columns']!=table['columns']:
            raise ValueError('Prior native table changed unexpectedly: '+name)
    for i,row in enumerate(payload['accounts'],6):
        for col,value in enumerate(row):
            actual=after['Drawdown accounts']['cells'][chr(65+col)+str(i)]['value']
            if isinstance(value,(float,int)) and not isinstance(value,bool):np.testing.assert_allclose(actual,value,atol=1e-10,rtol=1e-12)
            elif actual!=value:raise ValueError('Raw account mismatch')
    for i,row in enumerate(payload['headline'],32):
        for col,field in [('A','label'),('B','group'),('C','sharpe'),('D','cagr'),('E','dd'),('F','low'),('G','high')]:
            actual=after['New findings']['cells'][col+str(i)]['value'];expected=row[field]
            if isinstance(expected,(float,int)):np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
            elif actual!=expected:raise ValueError('Headline mismatch')
    # Verify every saved formula-backed chart point, rather than only terminal wealth.
    p=after['Leader paths']['cells'];returns=np.asarray(payload['returns']);n=len(returns)
    for group,start in [(0,15),(1,23)]:
        for j,source in enumerate(payload['chart_indices'][group]):
            col=start+j+1;letters=''
            while col:col,rem=divmod(col-1,26);letters=chr(65+rem)+letters
            expected=100*np.cumprod(1+returns[:,source])
            actual=np.array([p[letters+str(row)]['value'] for row in range(6,6+n)])
            np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-11)
    labels=[p['B'+str(row)]['value'] for row in range(6,6+n)]
    if len(set(labels))!=n:raise ValueError('Native chart categories are not unique daily dates')
    for group,start in [(0,31),(1,39)]:
        for j,source in enumerate(payload['chart_indices'][group]):
            col=start+j+1;letters=''
            while col:col,rem=divmod(col-1,26);letters=chr(65+rem)+letters
            wealth=100*np.cumprod(1+returns[:,source]);expected=wealth/np.maximum.accumulate(wealth)-1
            actual=np.array([p[letters+str(row)]['value'] for row in range(6,6+n)])
            np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-10)
    for group,start in [(0,15),(1,23),(0,31),(1,39)]:
        for j,label in enumerate(payload['chart_legends'][group]):
            col=start+j+1;letters=''
            while col:col,rem=divmod(col-1,26);letters=chr(65+rem)+letters
            if p[letters+'5']['value']!=label:raise ValueError('Chart legend metrics mismatch')
    if len(charts)!=4 or any(c['series']!=7 for c in charts):raise ValueError('Wrong native chart coverage')
    for chart in charts:
        if chart['colors']!=[c.lstrip('#') for c in payload['chart_colors']]:raise ValueError('Chart colors/strokes missing in export')
        if not all('Leader paths' in f for f in chart['formulas']):raise ValueError('Chart not bound to source sheet')
    old_rows=pd.read_csv(old(OLD/'all_strategies.csv'));new_rows=pd.read_csv(WORK/'all_strategies.csv')
    pd.testing.assert_frame_equal(new_rows.iloc[:len(old_rows)].reset_index(drop=True),old_rows,check_exact=False,rtol=1e-12,atol=1e-12)
    with np.load(old(OLD/'analysis_paths.npz')) as a,np.load(WORK/'analysis_paths.npz') as b:
        np.testing.assert_array_equal(a['returns'],b['returns'][:,:346])
        np.testing.assert_array_equal(a['ids'],b['ids'][:346])
        if b['returns'].shape!=(1003,364):raise ValueError('Master path coverage mismatch')
    catalog=pd.read_csv(WORK/'chart_catalog.csv')
    if len(catalog)!=652 or not catalog.id.is_unique:raise ValueError('Chart catalog coverage mismatch')
    check=json.loads((WORK/'workbook_inspection.json').read_text())
    if not check['formulaMutationVerified'] or not check.get('drawdownMutationVerified') or 'matched 0 entries' not in check['errors']:raise ValueError('Artifact formula checks failed')
    # A chart-only follow-up must preserve all previously published account cells.
    current_snapshot=WORK/'legend_before/master.xlsx'
    current_cells_preserved=0
    if current_snapshot.exists():
        current,_,_=cell_state(current_snapshot)
        for name,sheet in current.items():
            for address,cell in sheet['cells'].items():
                if name=='Leader paths' and re.fullmatch(r'(P|Q|R|S|T|U|V|X|Y|Z|AA|AB|AC|AD)5',address):continue
                updated=after[name]['cells'][address]
                if cell['formula']!=updated['formula'] or cell['value']!=updated['value']:
                    a,b=cell['value'],updated['value']
                    if cell['formula']!=updated['formula'] or not isinstance(a,(float,int)) or isinstance(a,bool):raise ValueError('Chart follow-up changed prior data: '+name+'!'+address)
                    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-12)
                current_cells_preserved+=1
    report=dict(status='passed_staged_master',prior_cells_preserved=total,prior_styles_preserved=True,
                prior_tables_and_panes_preserved=True,prior_findings_shift_rows=80,prior_return_paths_preserved=346,
                main_strategies=364,qualified_accounts_added=18,full_drawdown_accounts=969,
                chart_catalog_entries=652,native_charts=4,native_chart_series_each=7,
                formula_mutation_verified=True,all_native_chart_points_verified=28*n,opening_sheet='New findings',
                chart_legends_include_sr_cagr=True,drawdown_formula='wealth / running maximum wealth - 1',
                prior_published_cells_preserved=current_cells_preserved,
                input_workbook_sha256=sha(old(BOOK)),staged_workbook_sha256=sha(WORK/'master_staged.xlsx'),
                scope='Presentation integration and preservation audit. Study evidence remains exploratory.')
    write(DOC/'AUDIT.json',report);print(json.dumps(report,indent=2))


def opening_view():
    """Set only Excel opening-view metadata; Artifact Tool has no documented setter.

    The workbook's cells, charts, formulas and styling are authored by Artifact
    Tool. Preserve its XML bytes apart from these view attributes and tab order.
    """
    target=WORK/'master_staged.xlsx';temporary=WORK/'opening_view_staged.xlsx'
    with ZipFile(target) as source,ZipFile(temporary,'w') as output:
        for info in source.infolist():
            data=source.read(info.filename)
            if info.filename=='xl/workbook.xml':
                def activate(match):
                    tag=match.group(0)
                    if b'activeTab=' in tag:return re.sub(rb'activeTab="[^"]*"',b'activeTab="6"',tag)
                    return tag[:-2]+b' activeTab="6"/>' if tag.endswith(b'/>') else tag[:-1]+b' activeTab="6">'
                if not re.search(rb'<(?:\w+:)?workbookView\b',data):
                    data=data.replace(b'<x:sheets>',b'<x:bookViews><x:workbookView activeTab="6"/></x:bookViews><x:sheets>',1)
                else:data=re.sub(rb'<(?:\w+:)?workbookView\b[^>]*>',activate,data,count=1)
            elif re.fullmatch(r'xl/worksheets/sheet\d+\.xml',info.filename):
                selected=info.filename=='xl/worksheets/sheet7.xml'
                def selection(match):
                    tag=match.group(0);value=b'1' if selected else b'0'
                    if b'tabSelected=' in tag:return re.sub(rb'tabSelected="[^"]*"',b'tabSelected="'+value+b'"',tag)
                    return tag[:-2]+b' tabSelected="'+value+b'"/>' if tag.endswith(b'/>') else tag[:-1]+b' tabSelected="'+value+b'">'
                data=re.sub(rb'<(?:\w+:)?sheetView\b[^>]*>',selection,data,count=1)
            output.writestr(info,data)
    temporary.replace(target)
    with ZipFile(target) as z:
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        if wb.find('s:bookViews/s:workbookView',NS).attrib.get('activeTab')!='6':raise ValueError('Opening worksheet not selected')


def publish():
    result=json.loads((DOC/'AUDIT.json').read_text())
    if result['status']!='passed_staged_master' or sha(WORK/'master_staged.xlsx')!=result['staged_workbook_sha256']:
        raise ValueError('Staged audit missing or stale')
    for source,target in [(WORK/'master_staged.xlsx',BOOK),(WORK/'charts.md',ROOT/'charts.md'),
                          (WORK/'all_strategies.csv',OLD/'all_strategies.csv'),
                          (WORK/'analysis_paths.npz',OLD/'analysis_paths.npz'),(WORK/'chart_catalog.csv',GALLERY/'chart_catalog.csv')]:
        shutil.copyfile(source,target)
    record=dict(status='published_audited_master_update',timestamp_utc=datetime.now(timezone.utc).isoformat(),
                historical_period='2022-2025',main_strategies=364,new_qualified_accounts=18,
                study_variants=305,study_base_accounts=915,higher_cost_accounts=54,
                source_sha256=sha(ROOT/'research/reporting/update_ndx_drawdown_master.py'),
                builder_sha256=sha(WORK/'update_master.mjs'),
                prior_snapshot_sha256=sha(BEFORE/'MANIFEST.json'),
                artifact_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                    [BOOK,ROOT/'charts.md',OLD/'all_strategies.csv',OLD/'analysis_paths.npz',GALLERY/'chart_catalog.csv']},
                overview_sha256={p.name:sha(p) for p in DOC.glob('*.png')},
                note='Supersedes prior master-workbook/charts snapshots; original study publications remain unchanged.')
    write(DOC/'PUBLICATION.json',record)
    print('Published master workbook, chart-first narrative, 364 strategies and 652 gallery entries')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','audit','publish'],default='prepare');args=parser.parse_args()
    globals()[args.stage]()
