"""Frozen historical monthly Nasdaq-100 batch; all fits precede test metrics."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pit_research_engine import load_data,fold,historical_groups,continuous_backtest,etf_backtest
from run_robust_research import scale_inputs,fit_seed_job,ensemble_predictions,model_data,bootstrap_difference
from research_engine import performance,ranking_metrics


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/nasdaq100_historical_monthly_20261011'))
    parser.add_argument('--fit-limit',type=int,default=None,help='Stop after this many new fits for a throughput check; never evaluates partial batches.')
    args=parser.parse_args()
    protocol_path=Path('config/nasdaq100_historical_monthly.json');plan=json.loads(protocol_path.read_text())
    root=Path('data/nasdaq_monthly_pit/assembled');data=load_data(root)
    files=['run_historical_ndx.py','pit_research_engine.py','research_models.py','think_model.py',
           'robust_models.py','run_robust_research.py','training_device.py',str(protocol_path),str(root/'manifest.json')]
    lock={path:digest(path) for path in files}
    lock.update({str(root/name):expected for name,expected in data['manifest']['hashes'].items()})
    args.out.mkdir(parents=True,exist_ok=True)
    lock_path=args.out/'INPUT_CODE_LOCK.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text())!=lock:raise ValueError('Inputs/code changed; choose a fresh output directory')
    else:lock_path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    (args.out/'PROTOCOL.json').write_bytes(protocol_path.read_bytes())
    fit_args=SimpleNamespace(device='cuda',causal_membership=True,epochs=plan['epochs'],
                             patience=plan['patience'],lr=plan['lr'],batch=8)
    records=[];new_fits=0;stored={};all_test=[]
    for year in plan['test_years']:
        train,val,test=fold(data,year);all_test.extend(test)
        folder=args.out/str(year);folder.mkdir(exist_ok=True)
        x,mean,std=scale_inputs(data,train);groups=historical_groups(data,train)
        np.savez_compressed(folder/'preprocessing.npz',train=train,val=val,test=test,mean=mean,std=std)
        for kind in plan['models']:
            predictions=[]
            for seed in plan['training_seeds']:
                path=folder/f'{kind}_{seed}_predictions.npz';record_path=folder/f'{kind}_{seed}_record.json'
                if path.exists() and record_path.exists():
                    prediction=np.load(path);record=json.loads(record_path.read_text())
                    assert digest(folder/f'{kind}_{seed}.pt')==record['checkpoint_sha256']
                    tp=prediction['test'];vp=prediction['validation']
                else:
                    print(f'Fitting {year} {kind} seed {seed}',flush=True)
                    vp,tp,chosen,completed,parameters=fit_seed_job(data,x,train,val,test,kind,seed,groups,fit_args,folder)
                    if not np.isfinite(tp[data['mask'][test]]).all():raise ValueError('Nonfinite active prediction')
                    np.savez_compressed(path,validation=vp,test=tp)
                    record=dict(year=year,model=kind,seed=seed,chosen_epoch=chosen,epochs_run=completed,
                                parameters=parameters,checkpoint_sha256=digest(folder/f'{kind}_{seed}.pt'))
                    record_path.write_text(json.dumps(record,indent=2),encoding='utf-8');new_fits+=1
                records.append(record);predictions.append(tp)
                if args.fit_limit is not None and new_fits>=args.fit_limit:
                    print('Throughput check complete; no test performance computed.',flush=True);return
            stored[(year,kind)]=predictions
        design=x.transpose(0,2,1,3).reshape(len(x),len(data['panel'].tickers),-1)
        design=np.concatenate([design,np.ones((*design.shape[:2],1))],axis=-1)
        a=design[train][data['labelmask'][train]];y=(data['y'][train]*100)[data['labelmask'][train]]
        penalty=np.eye(a.shape[-1])*10.;penalty[-1,-1]=0.
        coef=np.linalg.solve(a.T@a+penalty,a.T@y);np.save(folder/'ridge_coef.npy',coef)
        stored[(year,'ridge')]=[design[test]@coef]
        stored[(year,'momentum126')]=[data['momentum'][test]]
        stored[(year,'reversal5')]=[data['reversal'][test]]
    if len(records)!=plan['planned_neural_fits']:raise ValueError('Incomplete neural batch')
    pd.DataFrame(records).to_csv(args.out/'training_records.csv',index=False)
    (args.out/'FITS_COMPLETE_LOCK.json').write_text(json.dumps(dict(neural_fits=len(records),
        checkpoint_hashes={str(p.relative_to(args.out)):digest(p) for p in args.out.rglob('*.pt')},
        evaluation='All declared fits complete before any replacement test metric'),indent=2),encoding='utf-8')
    print('All fits locked. Beginning continuous held-out evaluation.',flush=True)
    ids=np.asarray(all_test);curves=[];metrics=[];accounts=[];primary={};rank_rows=[]
    def record(kind,seed,k,mult,metric,daily,account=None):
        metrics.append(dict(model=kind,seed=seed,k=k,cost_multiplier=mult,**metric))
        curves.append(daily.assign(model=kind,seed=str(seed),k=k,cost_multiplier=mult))
        if account is not None and k==10 and mult==1:accounts.append(account.assign(model=kind,seed=str(seed)))
        if seed=='ensemble' and k==10 and mult==1:primary[kind]=daily
    for kind in plan['models']+['ridge','momentum126','reversal5']:
        adjusted=model_data(data,kind)
        seed_paths=[np.concatenate([stored[(year,kind)][s] for year in plan['test_years']])
                    for s in range(len(stored[(2022,kind)]))]
        ensemble=ensemble_predictions(seed_paths,adjusted['mask'][ids]) if len(seed_paths)>1 else seed_paths[0]
        ranking,_=ranking_metrics(ensemble,adjusted['y'][ids],adjusted['mask'][ids],adjusted['labelmask'][ids],[5,10,20])
        rank_rows.append(dict(model=kind,**ranking))
        for k in [5,10,20]:
            for mult in [1.,2.,4.]:
                metric,daily,account=continuous_backtest(adjusted,ids,ensemble,k,mult)
                record(kind,'ensemble',k,mult,metric,daily,account)
        if kind in plan['models']:
            for seed,pred in zip(plan['training_seeds'],seed_paths):
                metric,daily,account=continuous_backtest(adjusted,ids,pred,10,1.)
                record(kind,seed,10,1.,metric,daily,account)
    for mult in [1.,2.,4.]:
        metric,daily,account=continuous_backtest(data,ids,np.zeros_like(data['y'][ids]),10,mult,basket=True)
        record('monthly_basket','ensemble',10,mult,metric,daily,account)
        for ticker in ['SPY','QQQ']:
            metric,daily=etf_backtest(data,ticker,mult);record(ticker,'ensemble',10,mult,metric,daily)
    frame=pd.concat(curves,ignore_index=True);frame.to_csv(args.out/'curves.csv',index=False)
    pd.DataFrame(metrics).to_csv(args.out/'metrics.csv',index=False)
    pd.concat(accounts,ignore_index=True).to_csv(args.out/'account_ledger.csv',index=False)
    pd.DataFrame(rank_rows).to_csv(args.out/'ranking.csv',index=False)
    monthly=[];annual=[]
    for key,piece in frame.groupby(['model','seed','k','cost_multiplier'],sort=False):
        values=piece.copy();values['month']=pd.to_datetime(values.date).dt.to_period('M').astype(str)
        values['year']=pd.to_datetime(values.date).dt.year
        for month,sub in values.groupby('month'):
            monthly.append(dict(model=key[0],seed=key[1],k=key[2],cost_multiplier=key[3],month=month,net_return=float((1+sub.net).prod()-1)))
        for year,sub in values.groupby('year'):
            annual.append(dict(model=key[0],seed=key[1],k=key[2],cost_multiplier=key[3],year=year,**performance(sub.net)))
    pd.DataFrame(monthly).to_csv(args.out/'monthly.csv',index=False);pd.DataFrame(annual).to_csv(args.out/'annual.csv',index=False)
    paired=[]
    for kind in plan['models']+['ridge','momentum126','reversal5','monthly_basket']:
        for ticker in ['SPY','QQQ']:
            paired.append(dict(model=kind,comparison=ticker,**bootstrap_difference(primary[kind].net.to_numpy(),primary[ticker].net.to_numpy())))
    for base,mixed in [('think','think_mix'),('think_huber','think_mix_huber')]:
        paired.append(dict(model=mixed,comparison=base,**bootstrap_difference(primary[mixed].net.to_numpy(),primary[base].net.to_numpy())))
    pd.DataFrame(paired).to_csv(args.out/'paired_bootstrap.csv',index=False)
    (args.out/'completion.json').write_text(json.dumps(dict(status='completed_exploratory_research',
         neural_fits=len(records),continuous_portfolio_paths=len(metrics),
         first_date=primary['SPY'].date.iloc[0],last_date=primary['SPY'].date.iloc[-1],
         initial_nav=1_000_000,annual_account_resets=0),indent=2),encoding='utf-8')
    print('Historical monthly NASDAQ-100 batch complete.',flush=True)


if __name__=='__main__':main()
