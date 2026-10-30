"""Fixed factor-control arm, separate from the120-fit/576-account experiment."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

import run_ndx_drawdown_research as parent
from run_ndx_retention_replication import ROOT,sha,write
from run_ndx_rank_retraining import verify as verify_training
from report_ndx_rank_retraining import check_fits
from ndx_rank_retraining import percentile_predictions
from ndx_causal_factors import factor_scores,factor_blend
from ndx_retention import retention_targets,hedged_targets
from ndx_drawdown_engine import allocations,daily_allocation,risk_inputs,exposure_series,market_inputs
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq,simulate_signed
from audit_ndx_weak_market_short import value_replay

SOURCE=ROOT/'runs/ndx_rank_retraining_20261029'
RUN=ROOT/'runs/ndx_factor_controls_20261029'
CONFIG=ROOT/'config/ndx_factor_controls_20261029.json'


def freeze():
    verify_training(SOURCE)
    assert not (SOURCE/'METRICS.csv').exists(),'Arm must freeze before new account metrics'
    RUN.mkdir(parents=True,exist_ok=False)
    files=[CONFIG,SOURCE/'PROTOCOL.json',ROOT/'research/core/ndx_causal_factors.py',
           Path(__file__),ROOT/'tests/test_ndx_causal_factors.py']
    # __file__ is the launcher root copy: hash its canonical counterpart for portability.
    files[3]=ROOT/'research/analysis/run_ndx_factor_controls.py'
    write(RUN/'PROTOCOL.json',dict(config=json.loads(CONFIG.read_text()),
        hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(RUN/'PROTOCOL.json'),new_training_account_metrics_absent=True))
    print('Frozen198factor-control accounts before any new financial results',flush=True)


def verify():
    protocol=json.loads((RUN/'PROTOCOL.json').read_text())
    assert sha(RUN/'PROTOCOL.json')==json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    for name,expected in protocol['hashes'].items():assert sha(ROOT/name)==expected,name
    verify_training(SOURCE)
    return protocol['config']


def evaluate():
    cfg=verify();plan=check_fits(SOURCE)
    if (RUN/'ledgers').exists():raise FileExistsError('Preserve previous account attempt')
    (RUN/'ledgers').mkdir();data=parent.load_data()
    ids=np.concatenate([parent.fold(data,y)[2] for y in plan['years']]);mask=data['mask'][ids]
    dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    factor=factor_scores(data,ids);families={};seeds=plan['groups']['fifteen_seed']
    for family in plan['families']:
        individual=[]
        for seed in seeds:
            parts=[]
            for year in plan['years']:
                with np.load(SOURCE/'fits'/family/str(year)/str(seed)/'rank_predictions.npz') as p:parts.append(p['test'])
            individual.append(percentile_predictions(np.concatenate(parts),mask))
        families[family]=np.asarray(individual)
    neural={}
    for group,members in plan['groups'].items():
        index=[seeds.index(seed) for seed in members]
        neural[group]=.5*families['think'][index].mean(0)+.5*families['dynamic'][index].mean(0)
    market=market_inputs(data);combined=append_qqq(data,etf_data(data,'QQQ'));rows=[];paths={};targets={};checks=[]
    for style in cfg['factors']:
        for weight in cfg['neural_weights']:
            for group in (['seed_free'] if weight==0 else list(plan['groups'])):
                score=factor[style] if weight==0 else factor_blend(factor[style],neural[group],weight)
                for allocation in cfg['allocations']:
                    w=retention_targets(score,mask,5) if allocation=='retain5' else allocations(data,ids,score,'issuer5')
                    risk=risk_inputs(data,daily_allocation(data,ids,w))
                    long=exposure_series(dict(vol='basket',window=20,target=.20),market,risk)[data['origin'][ids]]
                    desired=hedged_targets(w,long,market['qqqvol20'][data['origin'][ids]],.75)
                    for setting in cfg['cost_regimes']:
                        cost,borrow=setting['cost'],setting['borrow'];key=f'{style}__nn{int(weight*100)}__{group}__{allocation}__c{cost}__b{int(borrow*100):02d}'
                        m,path,ledger=simulate_signed(combined,ids,desired,borrow,cost,detailed=True)
                        error=float(np.max(abs(path-value_replay(combined,ids,desired,borrow,cost))));assert error<1e-12
                        rows.append(dict(id=key,style=style,neural_weight=weight,group=group,allocation=allocation,cost=cost,borrow=borrow,**m))
                        paths[key]=path;targets[key]=desired;checks.append(dict(id=key,max_daily_error=error));ledger.to_csv(RUN/'ledgers'/(key+'.csv'),index=False)
        print('Completed factor arm',style,len(rows),'/198',flush=True)
    assert len(rows)==cfg['accounts']==198
    frame=pd.DataFrame(rows);frame.to_csv(RUN/'METRICS.csv',index=False);summaries=[]
    for (style,weight,allocation,cost,borrow),block in frame[frame.neural_weight>0].groupby(['style','neural_weight','allocation','cost','borrow']):
        row=dict(style=style,neural_weight=weight,allocation=allocation,cost=cost,borrow=borrow)
        pure=frame[(frame.style==style)&(frame.neural_weight==0)&(frame.allocation==allocation)&(frame.cost==cost)&(frame.borrow==borrow)].iloc[0]
        values=[]
        for group in plan['groups']:
            value=float(block[block.group==group].iloc[0].sharpe);row[group+'_sharpe']=value;values.append(value)
        row.update(pure_factor_sharpe=float(pure.sharpe),minimum_group_sharpe=min(values),
            minimum_increment_over_pure=min(values)-float(pure.sharpe),goal_passed=min(values)>1.5)
        summaries.append(row)
    summary=pd.DataFrame(summaries).sort_values('minimum_group_sharpe',ascending=False);summary.to_csv(RUN/'CONSISTENCY.csv',index=False)
    pd.DataFrame(checks).to_csv(RUN/'ACCOUNT_CHECKS.csv',index=False)
    np.savez_compressed(RUN/'PATHS.npz',dates=dates,**paths);np.savez_compressed(RUN/'TARGETS.npz',ids=ids,**targets)
    base=summary[(summary.cost==1)&(summary.borrow==.03)]
    write(RUN/'AUDIT.json',dict(status='passed',accounts=198,max_independent_daily_error=max(c['max_daily_error'] for c in checks),
        passing_hybrid_recipes=int(base.goal_passed.sum()),source_fit_lock_sha256=sha(SOURCE/'FITS_COMPLETE.json'),
        warning='Pure factor controls are seed-free and hybrid results share a deterministic factor; neither automatically proves neural incremental value or future performance'))
    verify();write(RUN/'COMPLETE.json',dict(output_sha256={p.relative_to(RUN).as_posix():sha(p) for p in RUN.rglob('*') if p.is_file()}))
    print(base.to_string(index=False),flush=True)


def report():
    lock=json.loads((RUN/'COMPLETE.json').read_text())
    for name,expected in lock['output_sha256'].items():assert sha(RUN/name)==expected
    doc=ROOT/'docs'/RUN.name;doc.mkdir(parents=True,exist_ok=True)
    for p in RUN.glob('*.csv'):shutil.copyfile(p,doc/p.name)
    for name in ['PROTOCOL.json','FREEZE.json','AUDIT.json']:shutil.copyfile(RUN/name,doc/name)
    table=pd.read_csv(RUN/'CONSISTENCY.csv');base=table[(table.cost==1)&(table.borrow==.03)]
    audit=json.loads((RUN/'AUDIT.json').read_text());lines=['# Causal factor controls and neural blends — October29,2026','',
        f'Completed198fixedaccounts; **{audit["passing_hybrid_recipes"]} hybrid recipes have SR>1.5 in all five groups** at base costs. '
        'This arm was specified during fitting, before any new account metrics. It does not alter the separate120-fit/576-account study. '
        'These remain selected development results on spent2022–2025history.','',
        '| Style | Neural weight | Allocation | Original SR | New SR | Fresh SR | Ten SR | Fifteen SR | Pure-factor SR | Minimum increment over pure | Goal |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in base.itertuples():lines.append(f'| {r.style} | {r.neural_weight:.0%} | {r.allocation} | {r.original_five_sharpe:.3f} | {r.new_five_sharpe:.3f} | {r.fresh_five_sharpe:.3f} | {r.ten_seed_sharpe:.3f} | {r.fifteen_seed_sharpe:.3f} | {r.pure_factor_sharpe:.3f} | {r.minimum_increment_over_pure:.3f} | {r.goal_passed} |')
    lines+=['','## Interpretation','',
        'Momentum63/20-dayvol and12-minus-1momentum/60-dayvol use trailing marks through each decision origin. '
        'A third score averages their percentiles with liquidity(logADV). Missing factor history is neutral0.5; current stock eligibility is unchanged. '
        'All hybrid scores use the ranking-selected neural forecasts only, with fixed25%/50% neural weight and fixed75%/50% factor weight. '
        'Pure factors are evaluated once each, not duplicated across seed groups. '
        'All accounts keep the20%basket-volatility target and existing continuousQQQ cap75 execution/costs. '
        'Higher hybrid seed consistency can come from the shared deterministic component. Compare each row to its pure-factor control before crediting neural training; '
        'a negative minimum increment means at least one neural ensemble underperformed the matching pure factor.','',
        'All198paths pass independent dollar-position replay; all results/cost stresses are retained. '
        'No outperformance significance or untouched future evidence is claimed. '
        'The controls are motivated by [Jegadeesh/Titman](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x) and '
        '[Moreira/Muir](https://www.nber.org/papers/w22208); the specific features/holding rules here do not replicate those papers.','',
        '[Allmetrics](METRICS.csv) · [Allcostregimes](CONSISTENCY.csv) · [Audit](AUDIT.json) · [Protocol](PROTOCOL.json)','']
    (doc/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');print('Published',doc.relative_to(ROOT),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['freeze','evaluate','report'],required=True);a=p.parse_args()
    dict(freeze=freeze,evaluate=evaluate,report=report)[a.stage]()
