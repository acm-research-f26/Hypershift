"""Fixed shared-feature ExtraTrees/Ridge research; all seeds and failures retained."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

import run_ndx_portfolio_robustness_v2 as parent
import run_ndx_family_robustness as families
from ndx_loss_optimizer import prepared_fold
from ndx_tabular_alpha import feature_cube,standardize_train_only,normalized_labels
from ndx_rank_retraining import split_validation,rank_statistics,percentile_predictions
from ndx_portfolio_robustness import average_targets,simulate_banded
from ndx_drawdown_engine import market_inputs
from ndx_weak_market_short import append_qqq
from ndx_matched_controls import etf_data
from audit_ndx_weak_market_short import value_replay
from run_ndx_retention_replication import sha,write

ROOT=parent.ROOT
NAME='ndx_tabular_alpha_20261030'
RUN=ROOT/'runs'/NAME
CONFIG=ROOT/'config'/f'{NAME}.json'


def freeze():
    old=parent.verify();family=families.verify()
    selection=json.loads((families.RUN/'SELECTION.json').read_text())
    assert selection['primary_passes']==0
    p=json.loads((families.RUN/'PROTOCOL.json').read_text())
    code=dict(p['source_sha256']);inputs=dict(p['input_sha256'])
    for name in ('research/training/run_ndx_tabular_alpha.py','research/core/ndx_tabular_alpha.py','tests/test_ndx_tabular_alpha.py'):
        code[name]=sha(ROOT/name)
    for path in (CONFIG,families.RUN/'SELECTION.json',families.RUN/'CONSISTENCY.csv'):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    plan=json.loads(CONFIG.read_text())
    plan.update(architectures=plan['models'],groups=old['groups'],seed_groups=old['seed_groups'],accounts_per_arm=120)
    RUN.mkdir(exist_ok=False)
    write(RUN/'PROTOCOL.json',dict(config=plan,source_sha256=code,input_sha256=inputs))
    write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(RUN/'PROTOCOL.json'),fits_absent=True,accounts_absent=True))
    write(RUN/'ENVIRONMENT.json',dict(numpy=np.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,
        device='CPU',fit_jobs=2,prediction_jobs=1,workers=2,blas_threads=1))
    for name in code:
        dest=RUN/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    print('Frozen64fits/282accounts before any new model outcomes',flush=True)


def verify():
    p=json.loads((RUN/'PROTOCOL.json').read_text())
    assert sha(RUN/'PROTOCOL.json')==json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    for section in ('source_sha256','input_sha256'):
        for name,expected in p[section].items():assert sha(ROOT/name)==expected,name
    return p['config']


def fit_year(year):
    plan=verify();data,x,extra,tr,va,te,_=prepared_fold(year)
    selection,audit=split_validation(data,va)
    assert data['exit'][tr].max()<data['origin'][selection].min()
    assert data['exit'][audit].max()<data['origin'][te].min()
    market=market_inputs(data);raw,names=feature_cube(data,x,extra,market['qqq'])
    features,mean,sd=standardize_train_only(raw,tr,data['labelmask'])
    observed=data['labelmask'][tr];target=normalized_labels(data['y'][tr]*100,observed)
    values=features[tr][observed];labels=target[observed]
    counts=observed.sum(1);weights=np.broadcast_to((counts.mean()/np.maximum(counts,1))[:,None],observed.shape)[observed]
    fold=RUN/'folds'/str(year);fold.mkdir(parents=True,exist_ok=True)
    preprocessing=fold/'PREPROCESSING.npz'
    if not preprocessing.exists():
        np.savez_compressed(preprocessing,features=features,mean=mean,sd=sd,feature_names=np.array(names,dtype=str),
            train=tr,validation=va,selection=selection,audit=audit,test=te)
    else:
        with np.load(preprocessing,allow_pickle=False) as saved:np.testing.assert_array_equal(features,saved['features'])
    si=np.searchsorted(va,selection);ai=np.searchsorted(va,audit)
    cases=[('tree',seed) for seed in plan['seed_groups']['fifteen_seed']]+[('ridge','seed_free')]
    with threadpool_limits(limits=1):
        for kind,seed in cases:
            folder=RUN/'fits'/kind/str(year)/str(seed);record=folder/'RECORD.json'
            if record.exists():
                for name,expected in json.loads(record.read_text())['hashes'].items():assert sha(folder/name)==expected
                continue
            folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
            model=(ExtraTreesRegressor(**plan['forest'],random_state=int(seed)) if kind=='tree' else Ridge(alpha=100.))
            model.fit(values,labels,sample_weight=weights)
            if kind=='tree':model.n_jobs=1
            vp=model.predict(features[va].reshape(-1,features.shape[-1])).reshape(len(va),features.shape[1])
            tp=model.predict(features[te].reshape(-1,features.shape[-1])).reshape(len(te),features.shape[1])
            assert np.isfinite(vp).all() and np.isfinite(tp).all()
            joblib.dump(model,folder/'MODEL.joblib',compress=3)
            np.savez_compressed(folder/'PREDICTIONS.npz',validation=vp,test=tp)
            importance=model.feature_importances_ if kind=='tree' else model.coef_
            pd.DataFrame(dict(feature=names,value=importance)).to_csv(folder/'FEATURE_WEIGHTS.csv',index=False)
            write(record,dict(kind=kind,year=year,seed=seed,fit_seconds=time.monotonic()-start,
                selection=rank_statistics(vp[si],data['y'][selection]*100,data['labelmask'][selection]),
                audit=rank_statistics(vp[ai],data['y'][audit]*100,data['labelmask'][audit]),
                selection_used_for_training_or_tuning=False,
                hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='RECORD.json'}))
            print(f'Fit {kind}/{year}/{seed}: {time.monotonic()-start:.1f}s',flush=True)
    return year


def train():
    plan=verify()
    if (RUN/'FITS_COMPLETE.json').exists():raise FileExistsError('Fits already complete')
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(fit_year,y) for y in reversed(plan['years'])]
        for future in as_completed(jobs):print('Tabular year done',future.result(),flush=True)
    records=list((RUN/'fits').rglob('RECORD.json'));assert len(records)==64
    hashes={}
    for record in records:
        for name,expected in json.loads(record.read_text())['hashes'].items():
            path=record.parent/name;assert sha(path)==expected;hashes[path.relative_to(RUN).as_posix()]=expected
        hashes[record.relative_to(RUN).as_posix()]=sha(record)
    for path in (RUN/'folds').rglob('*.npz'):hashes[path.relative_to(RUN).as_posix()]=sha(path)
    verify();write(RUN/'FITS_COMPLETE.json',dict(fits=64,hashes=hashes,financial_accounts_absent=True))
    print('All64fits locked before account evaluation',flush=True)


def predictions(plan,data,ids):
    output={}
    for kind,seeds in [('tree',plan['seed_groups']['fifteen_seed']),('ridge',['seed_free'])]:
        values=[]
        for seed in seeds:
            annual=[]
            for year in plan['years']:
                with np.load(RUN/'fits'/kind/str(year)/str(seed)/'PREDICTIONS.npz',allow_pickle=False) as p:annual.append(p['test'].copy())
            values.append(percentile_predictions(np.concatenate(annual),data['mask'][ids]))
        output[kind]=np.asarray(values)
    return output


def evaluate():
    plan=verify();lock=json.loads((RUN/'FITS_COMPLETE.json').read_text())
    for name,expected in lock['hashes'].items():assert sha(RUN/name)==expected,name
    data=parent.parent.load_data();ids=np.concatenate([parent.parent.fold(data,y)[2] for y in plan['years']])
    p=predictions(plan,data,ids);neural=parent.load_scores(plan,data,ids,'original')
    market=market_inputs(data);combined=append_qqq(data,etf_data(data,'QQQ'))
    eligible=np.column_stack((data['mask'][ids],np.ones(len(ids),bool)))
    dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    all_rows=[];aux=RUN/'diagnostics';aux.mkdir(exist_ok=False);(aux/'ledgers').mkdir()
    def account(dest,key,target,band,cost,borrow,**metadata):
        metrics,path,ledger,effective,_=simulate_banded(combined,ids,target,eligible,band,borrow,cost)
        error=float(np.max(np.abs(path-value_replay(combined,ids,effective,borrow,cost))))
        assert error<1e-12
        ledger.to_csv(dest/'ledgers'/(key+'.csv'),index=False)
        return dict(id=key,band=band,cost=cost,borrow=borrow,max_independent_daily_error=error,**metadata,**metrics),path,effective
    for method in plan['models']:
        dest=RUN/method;dest.mkdir(exist_ok=False);(dest/'ledgers').mkdir();rows=[];paths={};targets={}
        def pool(index):
            tree=p['tree'][index].mean(0)
            return tree if method=='tree' else .5*tree+.25*neural[0,index].mean(0)+.25*neural[1,index].mean(0)
        for allocation in plan['allocations']:
            for group,members in plan['seed_groups'].items():
                index=[plan['seed_groups']['fifteen_seed'].index(s) for s in members]
                for aggregation in plan['aggregations']:
                    w=(parent.signed_targets(data,ids,pool(index),allocation,market) if aggregation=='rank_pool' else
                       average_targets([parent.signed_targets(data,ids,pool(index[k:k+5]),allocation,market) for k in range(0,len(index),5)]))
                    for band in plan['bands']:
                        for setting in plan['cost_regimes']:
                            cost,borrow=setting['cost'],setting['borrow'];key=parent.key_for(method,allocation,aggregation,group,band,cost,borrow)
                            row,path,effective=account(dest,key,w,band,cost,borrow,arm=method,allocation=allocation,aggregation=aggregation,group=group,kind='ensemble')
                            rows.append(row);paths[key]=path;targets[key]=effective
                print(f'Tabular accounts {method}/{allocation}/{group}: {len(rows)}/120',flush=True)
        for j,seed in enumerate(plan['seed_groups']['fifteen_seed']):
            target=parent.signed_targets(data,ids,pool([j]),'retain5',market)
            key=f'{method}__seed_{seed}';row,path,_=account(aux,key,target,0.,1,.03,
                arm=method,allocation='retain5',aggregation='single_seed',group=f'seed_{seed}',kind='individual')
            all_rows.append(row)
        assert len(rows)==120
        all_rows.extend(rows);pd.DataFrame(rows).to_csv(dest/'METRICS.csv',index=False)
        np.savez_compressed(dest/'PATHS.npz',dates=dates,**paths)
        np.savez_compressed(dest/'TARGETS.npz',ids=ids,**targets)
        write(dest/'COMPLETE.json',dict(accounts=120,output_sha256={f.relative_to(dest).as_posix():sha(f) for f in sorted(dest.rglob('*')) if f.is_file()}))
    for allocation in plan['allocations']:
        w=parent.signed_targets(data,ids,p['ridge'][0],allocation,market)
        for band in plan['bands']:
            for setting in plan['cost_regimes']:
                cost,borrow=setting['cost'],setting['borrow'];key=parent.key_for('ridge',allocation,'seed_free','seed_free',band,cost,borrow)
                row,_,_=account(aux,key,w,band,cost,borrow,arm='ridge',allocation=allocation,aggregation='seed_free',group='seed_free',kind='control')
                all_rows.append(row)
    assert len(all_rows)==282
    pd.DataFrame(all_rows).to_csv(RUN/'ALL_METRICS.csv',index=False)
    verify();write(RUN/'ACCOUNT_AUDIT.json',dict(accounts=282,status='passed',
        max_independent_daily_error=max(r['max_independent_daily_error'] for r in all_rows)))


def summarize():
    verify();previous=parent.RUN
    try:
        parent.RUN=RUN;parent.summarize()
    finally:parent.RUN=previous


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['freeze','train','evaluate','summarize'],required=True)
    args=p.parse_args();{'freeze':freeze,'train':train,'evaluate':evaluate,'summarize':summarize}[args.stage]()


if __name__=='__main__':main()
