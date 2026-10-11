"""Freeze, fit and evaluate the user-requested extension of the audited study."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data, fold, historical_groups, continuous_backtest
from run_robust_research import scale_inputs, ensemble_predictions, model_data, bootstrap_difference
from research_engine import ranking_metrics, performance
from pseudohuber_think_mixer import KIND, fit_seed

PARENT = Path('runs/nasdaq100_historical_monthly_20261011')
RUN = Path('runs/nasdaq_topk_pseudohuber_20261011')
CONFIG = Path('config/nasdaq_topk_pseudohuber_20261011.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def verify_lock():
    for source, expected in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        assert digest(source) == expected, source
    for source, expected in json.loads((RUN/'SOURCE_ARTIFACT_LOCK.json').read_text()).items():
        assert digest(source) == expected, source


def freeze():
    assert json.loads((PARENT/'AUDIT.json').read_text())['status'] == 'passed'
    lock = json.loads((PARENT/'INPUT_CODE_LOCK.json').read_text())
    lock.update(json.loads((PARENT/'DEPENDENCY_CODE_LOCK.json').read_text()))
    for source, expected in lock.items():
        assert digest(source) == expected, source
    for source in [CONFIG, Path(__file__).name, 'pseudohuber_think_mixer.py']:
        lock[str(source)] = digest(source)
    source_files = list(PARENT.rglob('*.npz')) + list(PARENT.rglob('*.npy'))
    source_files += [PARENT/name for name in ['AUDIT.json','FITS_COMPLETE_LOCK.json','metrics.csv',
        'curves.csv','annual.csv','monthly.csv','ranking.csv','training_records.csv','PROTOCOL.json']]
    source_lock = {str(source):digest(source) for source in source_files}
    RUN.mkdir(exist_ok=True)
    for name, value in [('INPUT_CODE_LOCK.json',lock), ('SOURCE_ARTIFACT_LOCK.json',source_lock)]:
        path = RUN/name
        if path.exists():
            assert json.loads(path.read_text()) == value, 'Extension inputs changed; do not overwrite a frozen run'
        else:
            write(path, value)
    (RUN/'PROTOCOL.json').write_bytes(CONFIG.read_bytes())


def fit_year(year):
    verify_lock()
    plan = json.loads((RUN/'PROTOCOL.json').read_text())
    data = load_data();train,val,test = fold(data,year)
    x,mean,std = scale_inputs(data,train);groups = historical_groups(data,train)
    folder = RUN/str(year);folder.mkdir(exist_ok=True)
    with np.load(PARENT/str(year)/'preprocessing.npz') as previous:
        for name, value in [('train',train),('val',val),('test',test),('mean',mean),('std',std)]:
            np.testing.assert_array_equal(previous[name],value)
    np.savez_compressed(folder/'preprocessing.npz',train=train,val=val,test=test,mean=mean,std=std)
    records = []
    for seed in plan['training_seeds']:
        path = folder/f'{KIND}_{seed}_predictions.npz';record_path=folder/f'{KIND}_{seed}_record.json'
        if path.exists() and record_path.exists():
            record=json.loads(record_path.read_text());assert digest(folder/f'{KIND}_{seed}.pt')==record['checkpoint_sha256']
            assert digest(path)==record['predictions_sha256']
        else:
            print(f'Fitting pseudo-Huber: {year}, seed {seed}',flush=True)
            vp,tp,chosen,completed,parameters=fit_seed(data,x,train,val,test,seed,groups,plan,folder)
            assert np.isfinite(tp[data['mask'][test]]).all()
            np.savez_compressed(path,validation=vp,test=tp)
            record=dict(year=year,model=KIND,seed=seed,chosen_epoch=chosen,epochs_run=completed,
                parameters=parameters,checkpoint_sha256=digest(folder/f'{KIND}_{seed}.pt'),predictions_sha256=digest(path))
            write(record_path,record)
        records.append(record)
        print(f'Completed pseudo-Huber: {year}, seed {seed}, chosen epoch {record["chosen_epoch"]}',flush=True)
    return records


def train():
    freeze();plan=json.loads((RUN/'PROTOCOL.json').read_text())
    if (RUN/'FITS_COMPLETE_LOCK.json').exists():
        print('All 20 fits already locked; not refitting.',flush=True);return
    records=[]
    with ProcessPoolExecutor(max_workers=plan['fitting_workers']) as executor:
        jobs=[executor.submit(fit_year,year) for year in plan['test_years']]
        for job in as_completed(jobs):
            records.extend(job.result())
            print(f'{len(records)}/20 fits complete; no financial metrics evaluated',flush=True)
    assert len(records)==plan['planned_new_fits']==20
    pd.DataFrame(records).sort_values(['year','seed']).to_csv(RUN/'training_records.csv',index=False)
    write(RUN/'FITS_COMPLETE_LOCK.json',dict(new_neural_fits=20,
        hashes={str(p.relative_to(RUN)):digest(p) for p in sorted(RUN.rglob('*'))
                if p.suffix in ['.pt','.npz'] or p.name.endswith(('_history.json','_record.json'))},
        rule='All 20 new fits completed before extension financial test metrics'))
    print('20 fits locked. Ready for held-out evaluation.',flush=True)


def load_forecasts():
    verify_lock();plan=json.loads((RUN/'PROTOCOL.json').read_text())
    fitlock=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())
    assert fitlock['new_neural_fits']==20
    for source,expected in fitlock['hashes'].items():
        assert digest(RUN/source)==expected,source
    data=load_data();parent_plan=json.loads((PARENT/'PROTOCOL.json').read_text())
    kinds=parent_plan['models']+['ridge','momentum126','reversal5',KIND]
    forecasts={kind:[] for kind in kinds};all_ids=[]
    for year in plan['test_years']:
        train,val,test=fold(data,year);all_ids.extend(test);x,_,_=scale_inputs(data,train)
        for kind in kinds:
            if kind in parent_plan['models'] or kind==KIND:
                folder=(RUN if kind==KIND else PARENT)/str(year)
                scores=[]
                for seed in plan['training_seeds']:
                    with np.load(folder/f'{kind}_{seed}_predictions.npz') as saved:
                        scores.append(saved['test'].copy())
            elif kind=='ridge':
                design=x[test].transpose(0,2,1,3).reshape(len(test),len(data['panel'].tickers),-1)
                design=np.concatenate([design,np.ones((*design.shape[:2],1))],axis=-1)
                scores=[design@np.load(PARENT/str(year)/'ridge_coef.npy')]
            else:
                scores=[data['momentum' if kind=='momentum126' else 'reversal'][test]]
            forecasts[kind].append(scores)
    paths={kind:[np.concatenate([year[j] for year in yearly]) for j in range(len(yearly[0]))]
           for kind,yearly in forecasts.items()}
    return data,np.asarray(all_ids),paths


def evaluate():
    data,ids,forecasts=load_forecasts();plan=json.loads((RUN/'PROTOCOL.json').read_text())
    old_metrics=pd.read_csv(PARENT/'metrics.csv',dtype={'seed':str})
    old_curves=pd.read_csv(PARENT/'curves.csv',dtype={'seed':str})
    metrics=[];curves=[];accounts=[];rank=[];primary={};reused=0
    def record(kind,seed,k,mult):
        nonlocal reused
        prior=old_metrics[(old_metrics.model==kind)&(old_metrics.seed==str(seed))&
            (old_metrics.k==k)&(old_metrics.cost_multiplier==mult)]
        if len(prior):
            assert len(prior)==1
            metric=prior.iloc[0].to_dict()
            daily=old_curves[(old_curves.model==kind)&(old_curves.seed==str(seed))&
                (old_curves.k==k)&(old_curves.cost_multiplier==mult)].copy()
            assert len(daily)==1003
            reused+=1
        else:
            d=model_data(data,kind);available=forecasts[kind]
            score=(ensemble_predictions(available,d['mask'][ids]) if len(available)>1 else available[0]) if seed=='ensemble' else available[plan['training_seeds'].index(int(seed))]
            value,daily,account=continuous_backtest(d,ids,score,k,mult)
            metric=dict(model=kind,seed=str(seed),k=k,cost_multiplier=mult,**value)
            daily=daily.assign(model=kind,seed=str(seed),k=k,cost_multiplier=mult)
            accounts.append(account.assign(model=kind,seed=str(seed),k=k,cost_multiplier=mult))
        metrics.append(metric);curves.append(daily)
        if seed=='ensemble' and mult==1:
            primary[kind,k]=daily
    for kind,available in forecasts.items():
        d=model_data(data,kind)
        score=ensemble_predictions(available,d['mask'][ids]) if len(available)>1 else available[0]
        summary,_=ranking_metrics(score,d['y'][ids],d['mask'][ids],d['labelmask'][ids],[5,10,20])
        rank.append(dict(model=kind,**summary))
        for k in plan['portfolio_sizes']:
            for mult in plan['cost_multipliers']:
                record(kind,'ensemble',k,mult)
            if len(available)>1:
                for seed in plan['training_seeds']:
                    record(kind,seed,k,1)
        print(f'Evaluated all declared sizes and seeds: {kind}',flush=True)
    for kind in ['monthly_basket','SPY','QQQ']:
        for mult in plan['cost_multipliers']:
            record(kind,'ensemble',10,mult)
    frame=pd.concat(curves,ignore_index=True);frame.to_csv(RUN/'curves.csv',index=False)
    pd.DataFrame(metrics).to_csv(RUN/'metrics.csv',index=False)
    pd.concat(accounts,ignore_index=True).to_csv(RUN/'new_account_ledger.csv',index=False)
    pd.DataFrame(rank).to_csv(RUN/'ranking.csv',index=False)
    annual=[];monthly=[]
    for key,piece in frame.groupby(['model','seed','k','cost_multiplier'],sort=False):
        tags=dict(zip(['model','seed','k','cost_multiplier'],key))
        piece=piece.assign(year=pd.to_datetime(piece.date).dt.year,month=pd.to_datetime(piece.date).dt.to_period('M').astype(str))
        for year,sub in piece.groupby('year'):
            annual.append(dict(**tags,year=int(year),**performance(sub.net)))
        for month,sub in piece.groupby('month'):
            monthly.append(dict(**tags,month=month,net_return=float((1+sub.net).prod()-1)))
    pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False)
    pd.DataFrame(monthly).to_csv(RUN/'monthly.csv',index=False)
    paired=[]
    for k in plan['portfolio_sizes']:
        for comparison in ['think_mix','think_mix_huber','QQQ']:
            other=primary[comparison,k if comparison!='QQQ' else 10]
            paired.append(dict(model=KIND,comparison=comparison,k=k,
                **bootstrap_difference(primary[KIND,k].net.to_numpy(),other.net.to_numpy())))
    pd.DataFrame(paired).to_csv(RUN/'paired_bootstrap.csv',index=False)
    assert len(metrics)==204 and reused==120
    write(RUN/'completion.json',dict(status='completed_exploratory_extension',new_neural_fits=20,
        combined_portfolio_paths=len(metrics),parent_paths_reused=reused,new_portfolio_paths=len(metrics)-reused,
        portfolio_sizes=plan['portfolio_sizes'],first_date='2022-01-03',last_date='2025-12-31',annual_resets=0))
    print('Extension evaluated: 204 paths, including 120 unchanged parent paths.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['fit','evaluate'],required=True)
    args=parser.parse_args()
    train() if args.phase=='fit' else evaluate()
