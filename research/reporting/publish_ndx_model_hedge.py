"""Prepare and publish audited own-model hedge findings and workbook payload."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import run_ndx_model_hedge as study
from update_ndx_drawdown_master import cell_state

ROOT=study.ROOT;DOC=study.DOC;WORK=ROOT/'outputs/model_hedge_20261027'
BOOK=ROOT/'docs/NASDAQ100_strategy_comparison.xlsx'
OLD=ROOT/'outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020'
GROUPS={'original_five':'Original five','ten_seed':'Ten seeds','new_five':'New five'}
LABELS={'raw_mean':'Raw market mean','raw_basket':'Raw selected-basket return','raw_breadth60':'Raw negative breadth60%',
        'raw_agreement':'Raw family agreement','cal_mean':'Calibrated market mean','cal_basket':'Calibrated basket return',
        'cal_breadth60':'Calibrated negative breadth60%','cal_agreement':'Calibrated family agreement',
        'qqq_ma200':'QQQ MA200','always':'Continuous','none':'Unhedged'}
MODES={'hedge25':'hedge up to 25%','hedge50':'hedge up to 50%','switch50':'switch to 50% short','cash':'exit to cash','baseline':'20% volatility target'}


def label(recipe):
    if recipe=='baseline':return 'Five issuers + 20% volatility target'
    signal,mode=recipe.split('/')
    return LABELS[signal]+' / '+MODES[mode]


def ci(paths):
    names=list(paths);matrix=np.column_stack(list(paths.values()));n=len(matrix)
    rng=np.random.default_rng(20261027);draws=[]
    for _ in range(100):
        starts=rng.integers(0,n,(100,int(np.ceil(n/20))))
        indices=((starts[:,:,None]+np.arange(20))%n).reshape(100,-1)[:,:n]
        sample=matrix[indices]
        draws.append(np.sqrt(252)*(sample.mean(1)-(1.03**(1/252)-1))/sample.std(1,ddof=1))
    bounds=np.quantile(np.concatenate(draws),[.025,.975],axis=0)
    return {name:(float(bounds[0,j]),float(bounds[1,j])) for j,name in enumerate(names)}


def metrics_table(rows):
    text=['|Strategy|Seeds|Sharpe|CAGR|Max drawdown|95% SR interval|',
          '|---|---|---:|---:|---:|---|']
    for r in rows:
        text.append(f"|{r['label']}|{r['group']}|{r['sharpe']:.3f}|{r['cagr']:.2%}|{r['dd']:.2%}|[{r['low']:.3f}, {r['high']:.3f}]|")
    return '\n'.join(text)


def prepare():
    study.verify()
    audit=json.loads((DOC/'AUDIT.json').read_text());assert audit['status']=='passed'
    WORK.mkdir(parents=True,exist_ok=True)
    for source,name in [(BOOK,'before.xlsx'),(ROOT/'charts.md','before_charts.md'),
                        (OLD/'all_strategies.csv','before_all_strategies.csv'),(OLD/'analysis_paths.npz','before_paths.npz')]:
        target=WORK/name
        if not target.exists():shutil.copyfile(source,target)
    m=pd.read_csv(DOC/'metrics.csv');base=m[(m.cost_multiplier==1)&(m.borrow_rate==.03)].copy()
    with np.load(study.RUN/'paths.npz') as s:
        dates=pd.to_datetime(s['dates']);paths=dict(zip(s['ids'].astype(str),s['returns'].T))
    prior=pd.read_csv(WORK/'before_all_strategies.csv');assert len(prior)==364
    with np.load(WORK/'before_paths.npz') as s:
        np.testing.assert_array_equal(s['dates'],dates.strftime('%Y-%m-%d'))
        oldpaths=dict(zip(s['ids'].astype(str),s['returns'].T))
    basepaths={row.id:paths[row.id] for row in base.itertuples()}
    from finalize_ndx_model_hedge import inference_with_noops
    control_contrasts=[]
    for group in study.parent.GROUPS:
        control_contrasts.extend([(f'always/hedge50/{group}/c1/b0.03',f'{other}/{group}/c1/b0.03') for other in ['baseline','qqq_ma200/hedge50']])
    control_uncertainty=inference_with_noops(basepaths,control_contrasts,20)
    control_uncertainty.to_csv(DOC/'control_uncertainty.csv',index=False)
    intervals=ci(basepaths)
    base['sr_low']=[intervals[k][0] for k in base.id];base['sr_high']=[intervals[k][1] for k in base.id]
    base.to_csv(DOC/'base_metrics_with_intervals.csv',index=False)
    bestactive={};bestown={}
    for group in study.parent.GROUPS:
        model=base[(base.group==group)&(base.kind=='model')]
        bestown[group]=model.sort_values(['sharpe','recipe'],ascending=[False,True]).iloc[0].recipe
        bestactive[group]=model[model.signal.str.startswith('cal')].sort_values(['sharpe','recipe'],ascending=[False,True]).iloc[0].recipe
    additions=[]
    for row in base[base.recipe!='baseline'].itertuples():
        inherited=prior[prior.id==f'reference/dynamic_restricted_blend/{row.group}/top5'].iloc[0]
        additions.append(dict(id='model_hedge/'+row.id,study='Model hedge signals',strategy=label(row.recipe)+' / '+GROUPS[row.group],
            policy=row.recipe,k=5,section='Defensive',rankic=float(inherited.rankic),sharpe=row.sharpe,
            cagr=row.annualized_return,max_drawdown=row.maximum_drawdown,sr_low=row.sr_low,sr_high=row.sr_high))
    combined=pd.concat([prior,pd.DataFrame(additions)],ignore_index=True);assert len(combined)==472
    combined.to_csv(WORK/'all_strategies.csv',index=False,lineterminator='\n')
    allpaths=oldpaths|{'model_hedge/'+key:value for key,value in basepaths.items() if not key.startswith('baseline/')}
    np.savez_compressed(WORK/'analysis_paths.npz',ids=combined.id.to_numpy(dtype=str),dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str),
                        returns=np.column_stack([allpaths[k] for k in combined.id]))
    headline=[];focused=[];chart_indices=[];chart_legends=[];chart_keys=[]
    for group in ['original_five','ten_seed']:
        selected=['always/hedge50','qqq_ma200/hedge50',bestactive[group],'baseline']
        indices=[];legends=[]
        for recipe in selected:
            row=base[(base.recipe==recipe)&(base.group==group)].iloc[0]
            low,high=intervals[row.id]
            headline.append(dict(id=row.id,label=label(recipe),group=GROUPS[group],sharpe=float(row.sharpe),cagr=float(row.annualized_return),
                                dd=float(row.maximum_drawdown),low=low,high=high,
                                role='Continuous-hedge control' if recipe.startswith('always') else 'QQQ timing control' if recipe.startswith('qqq') else 'Unhedged / raw signals never trigger' if recipe=='baseline' else 'Best active model signal; selected'))
            focused.append(row.id);indices.append(len(chart_keys));chart_keys.append(row.id)
            short={'always/hedge50':'Continuous hedge 50%','qqq_ma200/hedge50':'QQQ MA200 hedge 50%','baseline':'Unhedged baseline'}.get(recipe,'Best active model signal')
            legends.append(f'{short} | SR {row.sharpe:.3f} | CAGR {row.annualized_return:.2%}')
        for ticker in ['SPY','QQQ']:
            key='original/'+ticker+'/10';row=prior[prior.id==key].iloc[0]
            if group=='original_five':
                headline.append(dict(label=ticker,group='Benchmark',sharpe=float(row.sharpe),cagr=float(row.cagr),dd=float(row.max_drawdown),
                                     low=float(row.sr_low),high=float(row.sr_high),role='Buy and hold'))
            indices.append(len(chart_keys));chart_keys.append(key)
            legends.append(f'{ticker} | SR {row.sharpe:.3f} | CAGR {row.cagr:.2%}')
        chart_indices.append(indices);chart_legends.append(legends)
    chartpaths={key:(oldpaths[key] if key.startswith('original/') else paths[key]) for key in chart_keys}
    colors=['#1764A2','#15928B','#B96D21','#7C8794','#C5433C','#7546A4']
    fig=plt.figure(figsize=(18,10));grid=fig.add_gridspec(3,2,height_ratios=[3,1.5,1],hspace=.18,wspace=.13)
    x=dates.insert(0,pd.Timestamp('2021-12-31'))
    for g,group in enumerate(['original_five','ten_seed']):
        ax=fig.add_subplot(grid[0,g]);ddax=fig.add_subplot(grid[1,g]);legendax=fig.add_subplot(grid[2,g]);legendax.axis('off')
        for j,index in enumerate(chart_indices[g]):
            key=chart_keys[index];equity=100*np.r_[1.,np.cumprod(1+chartpaths[key])];dd=equity/np.maximum.accumulate(equity)-1
            style=dict(color=colors[j],lw=2.1,ls='--' if j>=4 else '-')
            ax.plot(x,equity,label=chart_legends[g][j].replace(' | ','\n',1),**style);ddax.plot(x,dd,**style)
        ax.set_title(GROUPS[group],fontweight='bold');ax.set_ylim(50,340);ax.tick_params(labelbottom=False)
        ddax.set_ylim(-.4,.01);ddax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        for a in [ax,ddax]:a.grid(alpha=.18);a.set_xlim(x[0],x[-1]);a.xaxis.set_major_locator(mdates.YearLocator());a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_ylabel('Growth of $100, after costs');ddax.set_ylabel('Drawdown')
        legendax.legend(*ax.get_legend_handles_labels(),loc='upper left',ncol=2,fontsize=9.5,frameon=False)
    fig.suptitle('Model hedge signals: continuous hedging has the highest observed Sharpe',fontweight='bold',fontsize=16)
    fig.text(.08,.025,'2022–2025 exploratory history. Model-signal leaders selected retrospectively. Assumed 3% annual borrow. Five stock issuers in every neural strategy.',fontsize=10)
    fig.savefig(DOC/'high_sharpe_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(15,7))
    for key in combined.id:ax.plot(x,100*np.r_[1.,np.cumprod(1+allpaths[key])],color='#AAB3BD',alpha=.14,lw=.6)
    for key,color in [('model_hedge/always/hedge50/original_five/c1/b0.03',colors[0]),('model_hedge/always/hedge50/ten_seed/c1/b0.03',colors[1]),('original/SPY/10',colors[4]),('original/QQQ/10',colors[5])]:
        row=combined[combined.id==key].iloc[0];name=GROUPS['original_five']+' continuous hedge' if 'original_five' in key else GROUPS['ten_seed']+' continuous hedge' if 'ten_seed' in key else 'SPY' if '/SPY/' in key else 'QQQ'
        ax.plot(x,100*np.r_[1.,np.cumprod(1+allpaths[key])],color=color,lw=2.5,label=f'{name}: SR {row.sharpe:.3f}, CAGR {row.cagr:.2%}')
    ax.set_title('All 472 main strategy accounts, including 108 new hedge accounts',fontweight='bold');ax.set_ylabel('Growth of $100, after costs');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(DOC/'master_all_strategies.png',dpi=160);plt.close(fig)
    fields=['Strategy','Study','Sharpe','CAGR','RankIC','Max drawdown','95% SR low','95% SR high']
    defensive_rows=[[r['strategy'],r['study'],r['sharpe'],r['cagr'],r['rankic'],r['max_drawdown'],r['sr_low'],r['sr_high']] for r in additions]
    account_headers=['Strategy','Seed ensemble','Signal','Action','Cost multiple','Borrow rate','Sharpe','CAGR','Max drawdown','Mean short exposure','Trading cost USD','Borrow cost USD','Terminal NAV USD','Account ID','95% SR low','95% SR high']
    accounts=[[label(r.recipe),GROUPS[r.group],LABELS[r.signal],MODES[r.mode],int(r.cost_multiplier),float(r.borrow_rate),r.sharpe,r.annualized_return,r.maximum_drawdown,r.mean_short,r.cost_dollars,r.borrow_dollars,r.terminal_nav,r.id,*(intervals[r.id] if r.id in intervals else [None,None])] for r in m.itertuples()]
    signals=pd.read_csv(DOC/'signals.csv');signal_rows=[]
    for (group,year),block in signals.groupby(['group','year'],sort=False):
        signal_rows.append([GROUPS[group],int(year),len(block),*[int(block[name].sum()) for name in study.SIGNALS]])
    payload=dict(input_sha256=study.parent.sha(WORK/'before.xlsx'),headline=headline,accounts=accounts,account_headers=account_headers,
        defensive_rows=defensive_rows,defensive_start_row=57,chart_keys=chart_keys,chart_indices=chart_indices,chart_legends=chart_legends,colors=colors,
        dates=['2021-12-31']+dates.strftime('%Y-%m-%d').tolist(),returns=[[0.]*len(chart_keys)]+np.column_stack([chartpaths[k] for k in chart_keys]).tolist(),
        signal_rows=signal_rows,signal_headers=['Seeds','Year','Rebalances',*study.SIGNALS],
        best_active=bestactive,best_own=bestown)
    study.parent.write(WORK/'payload.json',payload)
    summary=metrics_table(headline)
    top=f'''# NASDAQ-100 master charts

Updated October27,2026. Historical monthly NASDAQ-100 membership, January3,2022–December31,2025. Every path uses the same1,003sessions.

## Highest observed Sharpe: model hedge experiment

![Original-five and ten-seed model hedge comparisons with SPY and QQQ](docs/ndx_model_hedge_20261027/high_sharpe_comparison.png)

**The new highest observed Sharpe is1.454 for the original-five stock blend with a continuous partial QQQ hedge:27.79%CAGR /−14.56%maximum drawdown. The ten-seed version reaches1.325Sharpe:24.83%CAGR /−14.78%drawdown.** This is an always-on hedge control. Its timing does not come from the trained models.

**Our own models did not supply a useful hedge trigger in this test.** All four raw-forecast signals stayed off at all201rebalances in every seed group. Their account paths exactly match the unhedged baseline. All four bias-corrected signals fired at the same50rebalances in2023 and stayed off in2022/2024/2025. They missed2022's weakness and hedged during2023's rebound. The best active model-signal results were1.111Sharpe for original-five and1.045for ten seeds, below their unhedged1.178/1.112baselines. Equal-Sharpe calibrated variants are ties, not distinct replications.

{summary}

All neural rows select **five stock issuers**, regardless of seed count. Hedge rows add a short QQQ position capped at25%or50%NAV, available cash budget and a20%QQQ volatility target. The highest-SR control keeps the50%cap on continuously; the model rules use only their predictions to turn hedging on. QQQ's trailing volatility still sizes the short. Gross target exposure is capped at100%; positions can drift between regular-close fills. Cash earns zero. Base modeled trading costs and assumed3%annual borrow are included.

**495accounts /37recipes:** eight own-model signals crossed with cash exit,25%hedge,50%hedge and50%short switch, plus unhedged,QQQ-MA200 and continuous-hedge controls, across original-five/new-five/ten-seed groups and cost/borrow assumptions. The strongest Sharpe control is selected from inspected historical results. Confidence intervals condition on fitted paths and do not adjust prior model searches. The newer five-seed half remains materially weaker. No claim of prospective superiority follows.

[Complete signal definitions, seed/cost results and audit](docs/ndx_model_hedge_20261027/REPORT.md) · [All495accounts](docs/ndx_model_hedge_20261027/metrics.csv) · [All signal decisions](docs/ndx_model_hedge_20261027/signals.csv) · [Comparison workbook](docs/NASDAQ100_strategy_comparison.xlsx)

## Updated master graph: all472main strategy accounts

![All472main strategy accounts with continuous-hedge leaders and benchmarks emphasized](docs/ndx_model_hedge_20261027/master_all_strategies.png)

The previous364accounts are preserved, with108new nonbaseline hedge-study accounts added. Different rules can produce identical paths;472counts declared accounts, not472independent strategies. Full cost/borrow cases remain in the workbook's Hedge accounts sheet. Open Model hedges for the new comparison charts.

## Previous published comparison: October26snapshot

The sections below preserve the preceding dated comparison. The October27results above supersede its headline leader claims.

'''
    # Normalize prose spacing without altering identifiers or numeric precision.
    replacements={'October27':'October 27','October26':'October 26','January3':'January 3','December31':'December 31',
        'same1,003sessions':'same 1,003 sessions','is1.454':'is 1.454','blend with a continuous':'blend with a continuous',
        ':27.79%CAGR':': 27.79% CAGR','version reaches1.325Sharpe:24.83%CAGR':'version reaches 1.325 Sharpe: 24.83% CAGR',
        '−14.56%maximum':'−14.56% maximum','−14.78%drawdown':'−14.78% drawdown',
        'all201rebalances':'all 201 rebalances','same50rebalances in2023':'same 50 rebalances in 2023',
        'in2022/2024/2025':'in 2022/2024/2025','missed2022':'missed 2022','during2023':'during 2023',
        'were1.111Sharpe':'were 1.111 Sharpe','and1.045for':'and 1.045 for','unhedged1.178/1.112baselines':'unhedged 1.178/1.112 baselines',
        'at25%or50%NAV':'at 25% or 50% NAV','a20%QQQ':'a 20% QQQ','the50%cap':'the 50% cap','at100%':'at 100%',
        'assumed3%annual':'assumed 3% annual','**495accounts /37recipes:**':'**495 accounts / 37 recipes:**',
        '25%hedge':'25% hedge','50%hedge':'50% hedge','50%short':'50% short','unhedged,QQQ':'unhedged, QQQ',
        'All495accounts':'All 495 accounts','all472main':'all 472 main','All472main':'All 472 main',
        'previous364accounts':'previous 364 accounts','with108new':'with 108 new','paths;472counts':'paths; 472 counts',
        'not472independent':'not 472 independent','October27results':'October 27 results','20%QQQ':'20% QQQ'}
    for a,b in replacements.items():top=top.replace(a,b)
    for a,b in {',2026':', 2026',',2022':', 2022',',2025':', 2025','/−':'/ −',
                'exit,25%':'exit, 25%','hedge,50%':'hedge, 50%','and50%':'and 50%',
                'October 26snapshot':'October 26 snapshot','October 27results':'October 27 results'}.items():top=top.replace(a,b)
    old=(WORK/'before_charts.md').read_text(encoding='utf8').split('\n',1)[1].lstrip()
    (WORK/'charts.md').write_text(top+old,encoding='utf8',newline='\n')
    u=pd.read_csv(DOC/'uncertainty.csv');u20=u[u.block_sessions==20]
    active=base[base.signal.str.startswith('cal')]
    bestrows=active.sort_values('sharpe',ascending=False).groupby('group').head(1)
    details='\n'.join(f"- {GROUPS[r.group]}: `{r.recipe}`, SR{r.sharpe:.3f}, CAGR{r.annualized_return:.2%}, drawdown{r.maximum_drawdown:.2%}." for r in bestrows.itertuples())
    stress=base[base.recipe.isin(['baseline','always/hedge50','qqq_ma200/hedge50',*bestactive.values()])]
    stress.to_csv(DOC/'focused_base_metrics.csv',index=False)
    report=f'''# Own-model hedge signals — October27,2026

The own-model signals did not improve Sharpe. Raw forecasts never triggered. Prior-year bias correction triggered hedging throughout2023, reducing gains in that rebound. The strongest observed Sharpe came from the continuous partial-hedge control: **1.454 original-five /1.325 ten-seed**. This is a control outcome, not successful model timing.

{summary}

![Comparison](high_sharpe_comparison.png)

## Signal design and timing

The fixed grid uses eligible-market mean<0, proposed five-issuer basket return<0, at least60%negative forecasts, or both model-family means<0. Each family averages its raw percentage-point predictions across its fixed seeds; the two family predictions receive equal50:50weight. These raw averages are separate from the percentile-rank blend used to select holdings. Four additional rules subtract each family/group/year's mean prior-year validation forecast error. Errors average equally by date. No slope or threshold was optimized on test returns.

Calibration uses saved validation predictions from the same annual fit. Those labels precede the first test signal. Checkpoints were selected on that validation fold, so it is not an independent calibration sample. No new neural-network fit was made. Features end09:15ET, decisions form09:20ET, shares use preceding-close NAV/marks/ADV, and fills occur at that day's regular close. Rebalances remain every five exchange sessions. Every neural account holds five issuers, apart from boundary ties/trading effects, and optionally shorts QQQ.

Raw signals stayed off on all201test dates across all three groups. The calibrated gates all produced the same50weak dates, all in2023. Those four nominal model signals therefore give identical paths within each action and seed group. The annual intercept absorbed the prior-year return regime and did not provide useful within-year market discrimination here. This finding concerns these fixed tests; it does not rule out every possible market-timing model.

## Best model signals that actually triggered

{details}

The highest own-model Sharpe if no-op rules are included is just the unhedged baseline:1.178original-five /1.112ten-seed /0.677new-five. The new-five continuous hedge gives0.811Sharpe, materially below the original-five leader. Different seed groups share the same market and the ten-seed ensemble includes both halves.

## Hedge mechanics and stress

Cash exits remove longs when weak. Hedge25/50keeps the baseline20%basket-volatility-target stock sleeve and shorts QQQ within min(25/50%NAV,unused long budget,20%/QQQ20-session annualized volatility). Switch50exits stocks and uses a similarly volatility-capped50%short. The always-on control uses hedge50at every rebalance. Model triggers do not use QQQ price trend; QQQ trailing volatility sizes every short for comparability. All targets have gross exposure<=100%,with possible realized drift. The signed engine retains1%ADVcaps, cancellations, adjusted economic-price dividend/settlement approximations, calendar-day borrowing and the fee reserve.

37recipes x3seed groups x3trading costs=333accounts.27actual short recipes x3groups x2extra borrow assumptions=162more, **495total**. Base borrow is assumed3%annual; stress rates1%/10%at base trading costs. Trading costs are also tested separately at2x/4x. These are assumptions, not historical broker quotes. Idle cash and short-proceeds rebate are zero. Actual locates, recalls, margin enforcement, taxes and full market impact are not modeled. All [accounts](metrics.csv), [annual results](annual.csv), [targets](targets.csv), [calibration estimates](calibration.csv) and [signal decisions](signals.csv) remain available.

## Statistical interpretation

There are120fixed contrasts:32model rules versus matched baseline plus8switches versus matched cash, each in three seed groups. Exactly60contrasts have identical paths. The inherited bootstrap rejected them as zero variance after every account had been saved. A timestamped [statistical amendment](STATISTICAL_AMENDMENT.json) retains the frozen source/grid/accounts, assigns deterministic zero differences and zero-width intervals to identical paths, and bootstraps the other60jointly after deduplicating identical columns. No epsilon was introduced and no trading outcome changed.

10,000joint circular-block draws use5/20/60sessions. Simultaneous annualized arithmetic active-mean intervals cover the120-comparison family. At primary20-session blocks, {int((u20.simultaneous_active_low>0).sum())}have a positive lower bound. SR/CAGR-difference intervals and absoluteSRintervals are pointwise and conditional on fits. The new continuous/QQQ controls are descriptive controls outside that own-model120-contrast family. Displayed leader intervals do not adjust for their retrospective selection or earlier model/risk research. All2022–2025data are repeatedly inspected development history.

## Verification

Eleven meaningful unit tests passed across model signals, deterministic-noop inference and signed accounting. The independent auditor reconstructed all{audit['signal_flags_rebuilt']}signal flags,24calibration estimates and111target schedules and replayed{audit['replayed_accounts']}accounts using dollar-position accounting. Maximum NAV discrepancy was${audit['max_nav_error']:.9f}. Every account metric was remeasured and shared QQQ/baseline paths reproduce the earlier study. This verifies the declared calculations, not vendor or broker execution accuracy.

No2026data, neural retraining, trades, automations or prospective-policy changes were made. The current prospective long/cash candidate remains unchanged. The workbook retains its prior ten sheets and adds Model hedges, Hedge accounts and Hedge paths. The defensive table adds108base-cost nonbaseline accounts. Charts at the top of the master file explicitly separate successful continuous hedging from unsuccessful own-model timing.

Reproduce: `python -m hypershift run run_ndx_model_hedge --stage freeze`, then `--stage run`. The preserved original runner stops on deterministic no-op uncertainty; `python -m hypershift run finalize_ndx_model_hedge` completes the amended inference from its saved account paths. Audit with `python -m hypershift run audit_ndx_model_hedge`. Completed runs refuse overwrite.

[Protocol](PROTOCOL.json) · [Freeze](FREEZE.json) · [Amendment](STATISTICAL_AMENDMENT.json) · [Independent audit](AUDIT.json) · [All495accounts](metrics.csv) · [All120contrasts](uncertainty.csv)
'''
    # Retain compact identifiers in raw data, but use normal spacing in prose.
    import re
    parts=re.split(r'(`[^`]*`|\[[^\]]*\]\([^)]*\))',report)
    for i in range(0,len(parts),2):
        parts[i]=re.sub(r'(?<=[A-Za-z%])(?=\d)', ' ',parts[i])
        parts[i]=re.sub(r'(?<=\d)(?=[A-Za-z])', ' ',parts[i])
    report=''.join(parts)
    for a,b in {',2026':', 2026','/1.325':'/ 1.325','%negative':'% negative','%basket':'% basket',
        '%NAV':'% NAV','%short':'% short','%ADVcaps':'% ADV caps','%annual':'% annual','%at':'% at',
        '%,with':'%, with','accounts.27':'accounts. 27','all 4824':'all 4,824',
        'flags,24':'flags, 24','drawdown-':'drawdown -','baseline:1.178':'baseline: 1.178',
        '/1.112':'/ 1.112','/0.677':'/ 0.677','was$':'was $','absoluteSRintervals':'absolute SR intervals',
        'contrasts:32':'contrasts: 32','at 2 x/4 x':'at 2x/4x'}.items():report=report.replace(a,b)
    report += '\nSecondary descriptive paired comparisons of the continuous hedge versus the unhedged and QQQ-timed controls are in [control_uncertainty.csv](control_uncertainty.csv). These six post hoc contrasts have their own simultaneous intervals and do not extend the frozen own-model family.\n'
    (DOC/'REPORT.md').write_text(report,encoding='utf8',newline='\n')
    study.parent.write(WORK/'PREPARED.json',dict(status='prepared',input_workbook_sha256=payload['input_sha256'],
                      prior_main_accounts=364,added_main_accounts=108,total_main_accounts=472))
    print('Prepared report,master charts,472strategy paths and workbook payload',flush=True)


def publish():
    payload=json.loads((WORK/'payload.json').read_text());staged=WORK/'comparison_staged.xlsx'
    before,bt,bc=cell_state(WORK/'before.xlsx');after,at,ac=cell_state(staged)
    count=0
    for name,sheet in before.items():
        for address,expected in sheet['cells'].items():
            if name=='New findings' and address=='A1':continue
            actual=after[name]['cells'].get(address)
            assert actual is not None,(name,address)
            assert expected['formula']==actual['formula'],(name,address,'formula')
            a,b=expected['value'],actual['value']
            if isinstance(a,(int,float)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-10)
            else:assert a==b,(name,address,a,b)
            assert expected['style']==actual['style'],(name,address,'style')
            count+=1
        assert sheet['columns']==after[name]['columns'],name+' columns'
        assert sheet['panes']==after[name]['panes'],name+' panes'
    for name,t in bt.items():
        assert t['style']==at[name]['style'] and t['columns']==at[name]['columns'],name
        expected='A5:H164' if t['ref']=='A5:H56' else t['ref']
        assert at[name]['ref']==expected,(name,at[name]['ref'],expected)
    assert len(ac)==len(bc)+4
    for old,new in zip(bc,ac[:len(bc)]):
        assert old['series']==new['series'] and old['formulas']==new['formulas'] and old['colors']==new['colors']
    for i,row in enumerate(payload['defensive_rows'],57):
        for j,value in enumerate(row):
            actual=after['Defensive']['cells'][chr(65+j)+str(i)]['value']
            if isinstance(value,(int,float)):np.testing.assert_allclose(actual,value,rtol=1e-11,atol=1e-10)
            else:assert actual==value
    # Verify all native chart helper values and bindings, not only endpoints.
    n=len(payload['dates']);p=after['Hedge paths']['cells'];returns=np.asarray(payload['returns'])
    def col(index):
        text='';index+=1
        while index:index,r=divmod(index-1,26);text=chr(65+r)+text
        return text
    for j in range(len(payload['chart_keys'])):
        wealth=100*np.cumprod(1+returns[:,j]);dd=wealth/np.maximum.accumulate(wealth)-1
        for start,expected in [(13,wealth),(25,dd)]:
            actual=np.array([p[col(start+j)+str(i)]['value'] for i in range(6,6+n)])
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-9)
    inspection=json.loads((WORK/'workbook_inspection.json').read_text())
    assert inspection['formulaMutationVerified'] and 'matched 0 entries' in inspection['errors']
    allowed={payload['input_sha256']}
    previous_publication=DOC/'MASTER_UPDATE_AUDIT.json'
    if previous_publication.exists():allowed.add(json.loads(previous_publication.read_text())['workbook_sha256'])
    assert study.parent.sha(BOOK) in allowed,'Workbook changed while preparing'
    # Only view metadata needs a package edit; workbook cells/charts are Artifact-authored.
    temporary=WORK/'opening_view.xlsx'
    with ZipFile(staged) as source,ZipFile(temporary,'w') as output:
        for info in source.infolist():
            b=source.read(info.filename)
            if info.filename=='xl/workbook.xml':
                root=ET.fromstring(b);ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                view=root.find('s:bookViews/s:workbookView',ns)
                if view is None:
                    views=ET.Element('{'+ns['s']+'}bookViews');root.insert(0,views)
                    view=ET.SubElement(views,'{'+ns['s']+'}workbookView')
                view.set('activeTab','10')
                b=ET.tostring(root,encoding='utf-8',xml_declaration=True)
            elif info.filename.startswith('xl/worksheets/sheet') and info.filename.endswith('.xml'):
                import re
                selected=info.filename=='xl/worksheets/sheet11.xml'
                def set_selected(match):
                    tag=match.group(0);value=b'1' if selected else b'0'
                    if b'tabSelected=' in tag:return re.sub(rb'tabSelected="[^"]*"',b'tabSelected="'+value+b'"',tag)
                    end=b'/>' if tag.endswith(b'/>') else b'>'
                    return tag[:-len(end)]+b' tabSelected="'+value+b'"'+end
                b=re.sub(rb'<(?:\w+:)?sheetView\b[^>]*>',set_selected,b,count=1)
            output.writestr(info,b)
    temporary.replace(staged)
    study.verify()
    for source,target in [(staged,BOOK),(WORK/'charts.md',ROOT/'charts.md'),(WORK/'all_strategies.csv',OLD/'all_strategies.csv'),(WORK/'analysis_paths.npz',OLD/'analysis_paths.npz')]:shutil.copyfile(source,target)
    result=dict(status='published',timestamp_utc=datetime.now(timezone.utc).isoformat(),prior_cells_preserved=count,
        prior_tables_styles_panes_preserved=True,prior_native_charts=len(bc),native_charts=len(ac),
        new_account_rows=495,new_main_rows=108,main_accounts=472,all_new_chart_points_verified=24*n,
        formula_mutation_verified=True,workbook_sha256=study.parent.sha(BOOK),charts_sha256=study.parent.sha(ROOT/'charts.md'))
    study.parent.write(DOC/'MASTER_UPDATE_AUDIT.json',result)
    study.parent.write(DOC/'PUBLICATION.json',dict(status='complete',
        authoring_source_sha256={'research/reporting/publish_ndx_model_hedge.py':study.parent.sha(__file__),
                                'outputs/model_hedge_20261027/update_comparison.mjs':study.parent.sha(WORK/'update_comparison.mjs')},
        output_sha256={p.name:study.parent.sha(p) for p in DOC.iterdir() if p.is_file() and p.name!='PUBLICATION.json'}))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','publish'],required=True)
    prepare() if p.parse_args().stage=='prepare' else publish()
