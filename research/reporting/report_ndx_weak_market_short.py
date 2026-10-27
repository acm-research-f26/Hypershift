"""Readable, reproducible report of every frozen weak-market short result."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_ndx_weak_market_short as study


def table(frame,fields):
    labels={'recipe':'Rule','group':'Seeds','cost_multiplier':'Trading cost','borrow_rate':'Borrow rate',
            'annualized_return':'Net CAGR','sharpe':'Sharpe','maximum_drawdown':'Max drawdown',
            'mean_short':'Average short exposure','year':'Year','annual_return':'Year return',
            'signal':'Signal','weak_rebalances':'Weak signals','all_rebalances':'Total signals',
            'weak_fraction':'Fraction weak','qqq_rise_fraction':'QQQ rose after weak signal',
            'mean_subsequent_qqq_return':'Mean next holding-period QQQ return'}
    percent={'annualized_return','maximum_drawdown','borrow_rate','mean_short','annual_return',
             'weak_fraction','qqq_rise_fraction','mean_subsequent_qqq_return'}
    lines=['| '+' | '.join(labels.get(k,k) for k in fields)+' |','| '+' | '.join('---' for _ in fields)+' |']
    for _,row in frame.iterrows():
        cells=[]
        for k in fields:
            v=row[k]
            cells.append(f'{v:.2%}' if k in percent else f'{v:.3f}' if k=='sharpe' else str(v))
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def report():
    study.verify()
    audit=json.loads((study.RUN/'INDEPENDENT_AUDIT.json').read_text())
    assert audit['status']=='passed'
    frame=pd.read_csv(study.RUN/'metrics.csv')
    primary=frame[(frame.cost_multiplier==1)&(frame.borrow_rate==.03)]
    ten=primary[primary.group=='ten_seed']
    shorts=ten[ten['mode'].str.startswith(('switch','hedge'))].sort_values('sharpe',ascending=False)
    best=shorts.iloc[0].recipe
    best_growth=shorts.sort_values('annualized_return',ascending=False).iloc[0].recipe
    chosen=list(dict.fromkeys(['full/baseline','vol20/baseline','vol25trend/baseline',
           'vol20/ma200/cash','vol20/ma200/switch25','vol20/ma200/switch50','vol20/ma200/switch100',
           'vol20/ma200/hedge25','vol20/ma200/hedge50',best,best_growth]))
    display=ten.set_index('recipe').loc[chosen].reset_index()
    comparisons=[]
    for row in primary.itertuples():
        if row.mode=='baseline':continue
        base=primary[(primary.recipe==row.base+'/baseline')&(primary.group==row.group)].iloc[0]
        comparisons.append(dict(recipe=row.recipe,group=row.group,
            cagr_delta=row.annualized_return-base.annualized_return,sharpe_delta=row.sharpe-base.sharpe,
            drawdown_delta=row.maximum_drawdown-base.maximum_drawdown))
    compare=pd.DataFrame(comparisons)
    compare.to_csv(study.DOC/'paired_point_estimates.csv',index=False)
    short_compare=compare[compare.recipe.str.contains('switch|hedge')]
    allgroups=short_compare.groupby('recipe').apply(lambda b: bool((b.sharpe_delta>0).all() and (b.drawdown_delta>0).all()),include_groups=False)
    all_three=short_compare.groupby('recipe').apply(lambda b: bool((b.sharpe_delta>0).all() and (b.drawdown_delta>0).all() and (b.cagr_delta>0).all()),include_groups=False)
    regimes=pd.read_csv(study.RUN/'regime_diagnostics.csv')
    rows=[]
    for gate,block in regimes.groupby('gate',sort=False):
        weak=block[block.weak]
        rows.append(dict(signal=gate,weak_rebalances=len(weak),all_rebalances=len(block),
            weak_fraction=len(weak)/len(block),qqq_rise_fraction=float((weak.subsequent_qqq_return>0).mean()),
            mean_subsequent_qqq_return=float(weak.subsequent_qqq_return.mean())))
    reg=pd.DataFrame(rows);reg.to_csv(study.DOC/'signal_summary.csv',index=False)
    inference=pd.read_csv(study.RUN/'uncertainty.csv')
    infer=inference[inference.block_sessions==20]
    supported=infer[infer.simultaneous_active_low>0]
    seedview=primary[primary.recipe.isin(['vol20/baseline','vol20/ma200/switch50','vol20/ma200/hedge25',best])]
    seedview=seedview.sort_values(['recipe','group'])
    stress=frame[(frame.group=='ten_seed')&frame.recipe.isin(list(dict.fromkeys(['vol20/baseline','vol20/ma200/switch50','vol20/ma200/hedge25',best])))]
    annual=pd.read_csv(study.RUN/'annual.csv')
    annual=annual.merge(ten[['id','recipe']],on='id')
    annual['annual_return']=annual.cumulative_return
    annualview=annual[annual.recipe.isin(list(dict.fromkeys(['vol20/baseline','vol20/ma200/cash','vol20/ma200/switch50','vol20/ma200/hedge25',best])))]
    selected=list(dict.fromkeys(['vol20/baseline','vol20/ma200/cash','vol20/ma200/switch50','vol20/ma200/hedge25',best]))
    labels={'vol20/baseline':'Current 20% target','vol20/ma200/cash':'MA200: move to cash',
            'vol20/ma200/switch50':'MA200: switch to 50% short',
            'vol20/ma200/hedge25':'MA200: hedge up to 25%',
            'vol20/ma200/hedge50':'MA200: hedge up to 50% (selected)'}
    labels.setdefault(best,'Selected best short: '+best)
    with np.load(study.RUN/'paths.npz') as saved:
        dates=pd.to_datetime(saved['dates']);paths=dict(zip(saved['ids'].astype(str),saved['returns'].T))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True,gridspec_kw={'height_ratios':[1.2,1]})
    exported=[]
    for name in selected:
        ret=paths[name+'/ten_seed/c1/b0.03'];equity=np.r_[1.,np.cumprod(1+ret)]
        dd=equity/np.maximum.accumulate(equity)-1
        axes[0].plot(dates,equity[1:],label=labels[name],linewidth=1.8)
        axes[1].plot(dates,dd[1:]*100,linewidth=1.5)
        exported.extend(dict(date=str(d.date()),recipe=name,net=float(r),wealth=float(v),drawdown=float(x)) for d,r,v,x in zip(dates,ret,equity[1:],dd[1:]))
    axes[0].set_ylabel('Growth of $1, net');axes[1].set_ylabel('Drawdown (%)')
    axes[0].legend(loc='upper left',fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Weak-market shorting: ten-seed ensemble, 2022–2025',fontweight='bold')
    fig.text(.08,.015,'Exploratory, repeatedly inspected history. Costs included; assumed 3% annual borrow. Best rule selected retrospectively.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(study.DOC/'comparison.png',dpi=180);plt.close(fig)
    pd.DataFrame(exported).to_csv(study.DOC/'comparison_curves.csv',index=False)
    bestrow=shorts.iloc[0];baserow=ten[ten.recipe==bestrow.base+'/baseline'].iloc[0]
    report=f'''# Weak-market shorting experiment — October 27, 2026

The highest observed short-rule Sharpe was **{bestrow.sharpe:.3f}**, with **{bestrow.annualized_return:.2%} net CAGR** and **{bestrow.maximum_drawdown:.2%} maximum drawdown**, for `{best}`. Its matching long/cash baseline was {baserow.sharpe:.3f} Sharpe / {baserow.annualized_return:.2%} CAGR / {baserow.maximum_drawdown:.2%} drawdown. This is a retrospective selection within 30 short rules, not a confirmed improvement.

**{int(allgroups.sum())} of 30 short rules improved Sharpe and drawdown in all three seed ensembles; {int(all_three.sum())} improved CAGR as well in all three.** The groups share the same market history and overlap in seeds; they are sensitivity checks, not independent market replications.

## Main matched results

All figures are after modeled trading costs and an assumed 3% annual borrow fee, on a continuous $1m reference account. Sharpe uses the inherited 3% annual hurdle. The primary strategy remains five issuers, a 20% basket volatility target and the fixed 50:50 THINK+mixer AH+PM / restricted dynamic-hypergraph rank blend.

{table(display,['recipe','annualized_return','sharpe','maximum_drawdown','mean_short'])}

![Matched wealth and drawdown](comparison.png)

`switch25/50/100` exits the stock basket and shorts QQQ while weak, at the smaller of the named NAV cap and a 20% QQQ volatility target. `hedge25/50` keeps the baseline stock sleeve and shorts only within its unused capital budget, also respecting the named cap and QQQ volatility target. `cash` exits the basket with no short. `vol25trend` is the pre-existing secondary 25% target with half exposure below MA200. `full` is unchanged issuer selection at full exposure.

## What “weak” means

We cannot know future weakness. The rules use QQQ's preceding close relative to its trailing 100- or 200-session average; the third requires MA200 weakness plus a negative prior 63-session return. Targets update only on the inherited five-session rebalance calendar. Prices through signal close feed the rule, shares freeze using that close's NAV/marks/ADV, and the orders fill at the following regular close. AH+PM rankings retain the 09:15 cutoff/09:20 decision before that fill. No hindsight bear-market labels or same-close fills are used.

{table(reg,list(reg.columns))}

The next holding-period QQQ returns above are descriptive outcomes measured after the rule was fixed. Positive QQQ periods lose money for a short before fees. The final holding period is partial. A weak trend signal does not imply the following week will decline.

## Seed sensitivity

{table(seedview,['recipe','group','annualized_return','sharpe','maximum_drawdown'])}

## Trading-cost and borrow sensitivity

Each of 39 recipes was run for original-five, new-five and ten-seed ensembles at 1x/2x/4x trading costs: 351 accounts. Each of 30 actual short rules also ran at assumed 1% and 10% annual borrow, with base trading costs: 180 more accounts, **531 total**. The borrow and transaction stresses are separate, not a full joint stress grid. Cash interest and short-proceeds rebates remain zero, matching the parent controls. All accounts are disclosed in [metrics.csv](metrics.csv).

{table(stress,['recipe','cost_multiplier','borrow_rate','annualized_return','sharpe','maximum_drawdown'])}

## Annual attribution

{table(annualview,['recipe','year','annual_return'])}

These annual breakdowns show whether protection in the 2022 decline was offset by foregone stock gains or losing shorts during later rebounds. These calendar years have already been inspected repeatedly; none is a new holdout.

## Uncertainty

The 54 fixed ten-seed contrasts comprise all 36 conditional cash/short rules versus the matching pre-existing baseline, plus all 18 stock-to-short switches versus their identical signal's cash exit. Joint circular-block bootstrap uses 10,000 draws at block lengths 5/20/60 sessions; 20 is primary. Simultaneous intervals on annualized arithmetic active means cover this 54-comparison family. Paired Sharpe and CAGR intervals are pointwise. **{len(supported)} of 54 primary simultaneous intervals have a positive lower bound.** Neither calculation adjusts the earlier model/risk-rule searches, data revisions, fitted-model uncertainty, or the fact that 2022–2025 was already used for development. See every result in [uncertainty.csv](uncertainty.csv).

## Accounting, verification and limits

Gross target exposure never exceeds 100% of NAV, and short proceeds never enlarge the target budget. Realized positions drift; the maximum observed gross exposure over every cost/seed/borrow case was {frame.max_gross.max():.3f}x. This is not a hard intraday leverage or drawdown limit. Actual margin calls, collateral rules, locates, recalls and short execution restrictions are not modeled. Borrow rates are assumptions, not historical QQQ fee quotes.

Short positions use the parent's adjusted economic-price units. Their signed price changes embed the total-return/dividend obligation approximation; adding another dividend debit would double-count it. This is an economic-price simulation, not a broker execution reconstruction. Short sales can incur borrowing fees and dividend obligations; see the [SEC investor bulletin](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-51).

Six synthetic tests passed, covering no future information, long-only parity, short gains/losses and calendar borrow, next-close fixed quantities, settlements/missing prices, and cash-funded residual shorts. All nine base-cost baseline paths reproduce parent daily paths; the two risk baselines also reproduce the parent cost-stress endpoints. Every one of 531 paths was remeasured and daily NAV/P&L reconciled. A separate dollar-position implementation rebuilt all 117 target schedules and replayed **{audit['replayed_accounts']} accounts**, including cash, shorts, hedges, seed groups and stress cases; maximum NAV discrepancy was **${audit['max_nav_error']:.9f}**. See [independent audit](INDEPENDENT_AUDIT.json).

The initial attempt stopped at account 80 because cash-scaled covering left a tiny short whose subsequent $0.118 borrow fee exceeded its remaining cash by $0.064. That attempt's protocol and source snapshots are preserved under `runs/ndx_weak_market_short_20261027/`. Corrected batch2 holds a reserve equal to 28 calendar days of borrow on the short value before positive trades. This provides a buffer for the five-session rebalance interval and gaps; it is a simulation funding convention, not a guaranteed bound under extreme moves. The 39 recipes and forecasts did not change. Some partial account results were visible during debugging; the repair was accounting-driven, with no return-based grid changes.

The prospective long/cash policy remains unchanged and unactivated. No model was retrained, no 2026 observations used, no trades placed and no recurring jobs created. Existing master workbook and prior studies remain unchanged.

## Reproduce and inspect

Run `.\\.venv\\Scripts\\python.exe -m hypershift test test_ndx_weak_market_short -v`. The new study runner supports `--stage freeze` then `--stage run` and refuses changed locks or completed-run overwrites. Run `audit_ndx_weak_market_short` for the separate replay and `report_ndx_weak_market_short` for this report. Use the repository's `python -m hypershift run` launcher. Existing data and parent saved forecast artifacts are required.

[Frozen protocol](PROTOCOL.json) · [Freeze](FREEZE.json) · [All recipe summaries](summary.csv) · [All 531 accounts](metrics.csv) · [Paired point estimates](paired_point_estimates.csv) · [All annual results](annual.csv) · [Target exposures](targets.csv) · [Uncertainty](uncertainty.csv) · [Signal diagnostics](regime_diagnostics.csv) · [Independent replay](INDEPENDENT_AUDIT.json) · [Focused daily curves](comparison_curves.csv)
'''
    (study.DOC/'REPORT.md').write_text(report,encoding='utf8')
    study.parent.write(study.DOC/'PUBLICATION.json',dict(status='complete',
        reporter_sha256=study.parent.sha(__file__),
        output_sha256={p.name:study.parent.sha(p) for p in study.DOC.iterdir() if p.is_file() and p.name!='PUBLICATION.json'}))
    print('Best observed short rule:',best,'; best short CAGR:',best_growth,flush=True)
    print('All-three-group SR/DD improvements:',int(allgroups.sum()),'; also CAGR:',int(all_three.sum()),flush=True)
    print('Simultaneous positive contrasts:',len(supported),'/54',flush=True)
    print(table(display,['recipe','annualized_return','sharpe','maximum_drawdown']),flush=True)


if __name__=='__main__':report()
