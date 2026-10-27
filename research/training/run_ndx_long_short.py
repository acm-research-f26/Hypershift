"""Predeclare and evaluate signed portfolios using already audited forecasts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_ndx_topk_pseudohuber import load_forecasts, RUN as SOURCE
from run_robust_research import ensemble_predictions, model_data, bootstrap_difference
from long_short_pit import signed_backtest
from research_engine import performance

RUN=Path('runs/nasdaq_long_short_20261011')
CONFIG=Path('config/nasdaq_long_short_20261011.json')
LEDGER=['nav','cash','long_dollars','short_dollars','gross_exposure','net_exposure',
        'fee','borrow_fee','traded_dollars','settled_dollars']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def verify_lock():
    for path,expected in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        assert digest(path)==expected,path


def freeze():
    assert json.loads((SOURCE/'AUDIT.json').read_text())['status']=='passed'
    lock=json.loads((SOURCE/'INPUT_CODE_LOCK.json').read_text())
    lock.update(json.loads((SOURCE/'SOURCE_ARTIFACT_LOCK.json').read_text()))
    for relative,expected in json.loads((SOURCE/'FITS_COMPLETE_LOCK.json').read_text())['hashes'].items():
        lock[str(SOURCE/relative)]=expected
    for name in ['metrics.csv','curves.csv','AUDIT.json','ranking.csv','PROTOCOL.json']:
        lock[str(SOURCE/name)]=digest(SOURCE/name)
    for path in [CONFIG,Path(__file__).name,'long_short_pit.py','audit_ndx_long_short.py','test_long_short_pit.py']:
        lock[str(path)]=digest(path)
    RUN.mkdir(exist_ok=True)
    if (RUN/'INPUT_CODE_LOCK.json').exists():
        assert json.loads((RUN/'INPUT_CODE_LOCK.json').read_text())==lock,'Frozen short experiment changed'
    else:
        write(RUN/'INPUT_CODE_LOCK.json',lock)
        (RUN/'PROTOCOL.json').write_bytes(CONFIG.read_bytes())
    verify_lock()


def prepared_forecasts():
    data,ids,available=load_forecasts()
    scores={}
    for kind,individual in available.items():
        d=model_data(data,kind)
        scores[kind,'ensemble']=ensemble_predictions(individual,d['mask'][ids]) if len(individual)>1 else individual[0]
        if len(individual)>1:
            for seed,pred in zip([7,19,42,61,97],individual):
                scores[kind,str(seed)]=pred
    return data,ids,scores


def main():
    freeze()
    if (RUN/'completion.json').exists():
        print('Completed experiment exists; not recomputing or changing allocations.',flush=True)
        return
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    data,ids,scores=prepared_forecasts()
    previous=pd.read_csv(SOURCE/'metrics.csv',dtype={'seed':str})
    old_curves=pd.read_csv(SOURCE/'curves.csv',dtype={'seed':str})
    qqq=old_curves[(old_curves.model=='QQQ')&(old_curves.seed=='ensemble')&
        (old_curves.k==10)&(old_curves.cost_multiplier==1)].net.to_numpy()
    metrics=[]; curves=[]; ledgers={}; path_id=0
    models=list(dict.fromkeys(key[0] for key in scores))

    def record(model,seed,k,policy,multiplier=1.,borrow=.03):
        nonlocal path_id
        long,short=plan['policies'][policy]; borrow=borrow if short else 0.
        tags=dict(path_id=path_id,model=model,seed=seed,k=k,policy=policy,
            long_budget=long,short_budget=short,cost_multiplier=multiplier,borrow_rate=borrow)
        reused=policy=='long100'
        if reused:
            row=previous[(previous.model==model)&(previous.seed==seed)&(previous.k==k)&
                (previous.cost_multiplier==multiplier)]
            assert len(row)==1
            result=row.iloc[0].to_dict()
            daily=old_curves[(old_curves.model==model)&(old_curves.seed==seed)&(old_curves.k==k)&
                (old_curves.cost_multiplier==multiplier)][['date','net','gross']].copy()
            result.update(borrow_dollars=0.)
        else:
            result,daily,account=signed_backtest(model_data(data,model),ids,scores[model,seed],k,
                long,short,borrow,multiplier,plan['aum'])
            ledgers[str(path_id)]=account[LEDGER].to_numpy()
        assert len(daily)==len(qqq)==1003
        result.update(beta_to_QQQ=float(np.cov(daily.net,qqq,ddof=1)[0,1]/np.var(qqq,ddof=1)),
            correlation_to_QQQ=float(np.corrcoef(daily.net,qqq)[0,1]))
        metrics.append(dict(**result,**{key:value for key,value in tags.items() if key not in result},parent_reused=reused))
        curves.append(daily.assign(path_id=path_id)[['path_id','date','net','gross']])
        path_id+=1

    for model in models:
        for k in plan['portfolio_sizes']:
            for policy in plan['policies']:
                record(model,'ensemble',k,policy)
            for policy in ['long75_short25','long50_short50']:
                for mult in plan['cost_stress_multipliers']:
                    record(model,'ensemble',k,policy,mult)
                for borrow in plan['borrow_stress_rates']:
                    record(model,'ensemble',k,policy,borrow=borrow)
                for seed in map(str,plan['training_seeds']):
                    if (model,seed) in scores:
                        record(model,seed,k,policy)
        print(f'All fixed allocations, sizes, seeds and cost stresses evaluated: {model}',flush=True)
    frame=pd.concat(curves,ignore_index=True); rows=pd.DataFrame(metrics)
    assert len(rows)==600 and len(ledgers)==570
    np.savez_compressed(RUN/'ledgers.npz',**ledgers)
    frame.to_csv(RUN/'curves.csv',index=False); rows.to_csv(RUN/'metrics.csv',index=False)
    annual=[]; monthly=[]
    for path,sub in frame.groupby('path_id',sort=False):
        dated=pd.to_datetime(sub.date)
        for year,part in sub.groupby(dated.dt.year):
            annual.append(dict(path_id=path,year=int(year),**performance(part.net)))
        for month,part in sub.groupby(dated.dt.to_period('M').astype(str)):
            monthly.append(dict(path_id=path,month=month,net_return=float((1+part.net).prod()-1)))
    pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False)
    pd.DataFrame(monthly).to_csv(RUN/'monthly.csv',index=False)
    # All declared ensemble short allocations versus same-long cash, full long, QQQ.
    primary=rows[(rows.seed=='ensemble')&(rows.cost_multiplier==1)&
        (np.isclose(rows.borrow_rate,.03)|np.isclose(rows.borrow_rate,0))].set_index(['model','k','policy'])
    paths={path:sub.net.to_numpy() for path,sub in frame.groupby('path_id',sort=False)}
    paired=[]
    for model in models:
        for k in plan['portfolio_sizes']:
            for policy,matched in [('long75_short25','long75_cash25'),('long50_short50','long50_cash50')]:
                a=paths[int(primary.loc[(model,k,policy),'path_id'])]
                for comparison in [matched,'long100','QQQ']:
                    b=qqq if comparison=='QQQ' else paths[int(primary.loc[(model,k,comparison),'path_id'])]
                    paired.append(dict(model=model,k=k,policy=policy,comparison=comparison,**bootstrap_difference(a,b)))
    pd.DataFrame(paired).to_csv(RUN/'paired_bootstrap.csv',index=False)
    primary.reset_index().to_csv(RUN/'primary_metrics.csv',index=False)
    frame[frame.path_id.isin(primary.path_id)].to_csv(RUN/'primary_curves.csv',index=False)
    # Individual seeds assess sensitivity; do not select the best seed.
    seed_summary=[]
    for (model,k,policy),sub in rows[(rows.seed!='ensemble')&(rows.cost_multiplier==1)].groupby(['model','k','policy']):
        seed_summary.append(dict(model=model,k=k,policy=policy,seeds=len(sub),
            median_CAGR=sub.annualized_return.median(),min_CAGR=sub.annualized_return.min(),
            max_CAGR=sub.annualized_return.max(),positive_seeds=int((sub.annualized_return>0).sum()),
            seeds_beating_QQQ=int((sub.annualized_return>performance(qqq)['annualized_return']).sum())))
    pd.DataFrame(seed_summary).to_csv(RUN/'seed_summary.csv',index=False)
    write(RUN/'completion.json',dict(status='completed_exploratory_signed_extension',new_model_fits=0,
        total_portfolio_paths=600,new_signed_account_paths=570,parent_long_only_paths_reused=30,
        primary_ensemble_paths=150,annual_resets=0,portfolio_sizes=plan['portfolio_sizes'],
        policies=list(plan['policies']),short_availability_assumed=True,
        dates=['2022-01-03','2025-12-31']))
    print('Completed 600 disclosed paths; independent audit required before publication.',flush=True)


if __name__=='__main__':
    main()
