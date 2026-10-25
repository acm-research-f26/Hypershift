"""Freeze and fit the broader causal-feature research batch; never tune on test returns."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ndx_broad_models import prepare,splits,normalize,SharedNetwork,predict_network,stability
from run_ndx_alpaca_training import ROOT,sha,write
from run_ndx_ranking import RUN as PREVIOUS
from run_robust_research import ensemble_predictions
from training_device import configure_device,device_metadata

RUN = ROOT/'runs/ndx_broad_20261020'
CONFIG = ROOT/'config/ndx_broad_research_20261020.json'
_DATA = None


def data_cache():
    global _DATA
    if _DATA is None:
        _DATA=prepare()
    return _DATA


def verify():
    for name,value in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        if sha(name)!=value:
            raise ValueError('Broad experiment input/source changed: '+name)


def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=json.loads((PREVIOUS/'INPUT_CODE_LOCK.json').read_text())
    for name in ['ndx_broad_models.py','ndx_policy_engine.py','run_ndx_broad.py','test_ndx_broad.py',
                 'requirements-research-20261020.txt','config/ndx_broad_research_20261020.json']:
        lock[str(ROOT/name)]=sha(ROOT/name)
    tests=json.loads((RUN/'UNIT_TESTS.json').read_text())
    if tests['status']!='passed':
        raise ValueError('Broad research tests have not passed')
    for name,value in tests['source_hashes'].items():
        lock[str(ROOT/name)]=value
    path=RUN/'INPUT_CODE_LOCK.json'
    if path.exists() and json.loads(path.read_text())!=lock:
        raise ValueError('Cannot change frozen broad training protocol')
    write(path,lock); verify()
    if not (RUN/'PROTOCOL.json').exists():
        (RUN/'PROTOCOL.json').write_bytes(CONFIG.read_bytes())
        snapshot=RUN/'source_snapshot'; snapshot.mkdir(exist_ok=True)
        for name in lock:
            p=Path(name)
            if p.suffix=='.py':
                (snapshot/p.name).write_bytes(p.read_bytes())
        write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(CONFIG),before_all_new_fits=not any(RUN.glob('*/*/*_record.json')),
            history='exploratory; earlier financial outcomes already viewed'))
        ledger=ROOT/'docs/audit/research_ledger.jsonl'
        with ledger.open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(dict(run=str(RUN.relative_to(ROOT)),protocol_sha256=sha(CONFIG),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),selected_fold_models=176,candidate_fits=208,
                prior_research='Full prior search is broader than this ledger; prior published runs remain preserved.'))+'\n')
    import sklearn,scipy
    environment=dict(numpy=np.__version__,pandas=pd.__version__,torch=str(torch.__version__),sklearn=sklearn.__version__,scipy=scipy.__version__)
    envpath=RUN/'ENVIRONMENT.json'
    if envpath.exists() and json.loads(envpath.read_text())!=environment:
        raise ValueError('Fitting environment changed')
    write(envpath,environment)
    return json.loads(CONFIG.read_text())


def tree_prediction(model,x,iterations):
    for number,value in enumerate(model.staged_predict(x),1):
        if number==iterations:
            return value
    raise ValueError('Missing selected tree iteration')


def fit_network(data,x,target,tr,va,te,seed,plan):
    torch.manual_seed(seed); device=configure_device('cuda')
    cfg=plan['network_parameters']; model=SharedNetwork(x.shape[-1],cfg['hidden']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    y=data['targets'][target].astype('float32'); months=data['dates'][data['origin'][va]].to_period('M').astype(str)
    best,state,stale,chosen,history=-np.inf,None,0,0,[]
    for epoch in range(1,cfg['epochs']+1):
        model.train(); losses=[]
        for start in range(0,len(tr),cfg['batch_dates']):
            ids=tr[start:start+cfg['batch_dates']]; optimizer.zero_grad(set_to_none=True)
            score=model(torch.from_numpy(x[ids]).to(device),torch.from_numpy(data['mask'][ids]).to(device))
            observed=torch.from_numpy(data['labelmask'][ids]).to(device)
            error=torch.where(observed,score-torch.from_numpy(y[ids]).to(device),torch.zeros_like(score))
            loss=(error.square().sum(1)/observed.sum(1).clamp_min(1)).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite shared-network loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_norm_cap']); optimizer.step()
            losses.append(float(loss.detach()))
        vp=predict_network(model,x,data['mask'],va,device,cfg['batch_dates'])
        score=stability(vp,data['targets']['residual'][va],data['labelmask'][va],months)
        history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_stability=score))
        if score>best+1e-7:
            best,chosen,stale=score,epoch,0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
        if stale>=cfg['patience']:
            break
    if state is None:
        raise ValueError('No finite shared-network checkpoint')
    model.load_state_dict(state)
    vp,tp=[predict_network(model,x,data['mask'],ids,device,cfg['batch_dates']) for ids in [va,te]]
    np.testing.assert_allclose(stability(vp,data['targets']['residual'][va],data['labelmask'][va],months),best,atol=1e-12,rtol=0)
    checkpoint=dict(state_dict=state,features=x.shape[-1],hidden=cfg['hidden'],epoch=chosen,seed=seed,device=device_metadata(device))
    return checkpoint,vp,tp,dict(selected_epoch=chosen,validation_stability=best,history=history)


def fit_group(view,year,target,family):
    verify(); data=data_cache(); plan=json.loads((RUN/'PROTOCOL.json').read_text())
    tr,va,te=splits(data,year); x,mean,std=normalize(data['views'][view],data['mask'],tr)
    folder=RUN/view/str(year); prefix=f'{family}_{target}'
    with np.load(folder/'preprocessing.npz') as saved:
        for name,value in dict(train=tr,validation=va,test=te,mean=mean,std=std).items():
            np.testing.assert_array_equal(saved[name],value)
    months=data['dates'][data['origin'][va]].to_period('M').astype(str)
    observed=data['labelmask'][tr]; y=data['targets'][target][tr][observed]
    if observed.sum(1).min()<2:
        raise ValueError('Insufficient observed training labels on an origin')
    weights=np.broadcast_to((100/observed.sum(1))[:,None],observed.shape)[observed]
    flat=x[tr][observed]; validation_x=x[va].reshape(-1,x.shape[-1]); test_x=x[te].reshape(-1,x.shape[-1])
    seeds=[plan['seeds'][0]] if family=='ridge' else plan['seeds']
    for seed in seeds:
        base=folder/f'{prefix}_{seed}'; recordpath=base.with_name(base.name+'_record.json')
        if recordpath.exists():
            record=json.loads(recordpath.read_text())
            for name,value in record['artifact_hashes'].items():
                if sha(folder/name)!=value:
                    raise ValueError('Broad fit artifact changed')
            continue
        if family=='ridge':
            a=np.column_stack([flat,np.ones(len(flat))]); a=a*np.sqrt(weights[:,None]); b=y*np.sqrt(weights)
            candidates=[]; coefficients=[]; best=-np.inf
            va_design=np.concatenate([x[va],np.ones((*x[va].shape[:2],1))],axis=-1)
            te_design=np.concatenate([x[te],np.ones((*x[te].shape[:2],1))],axis=-1)
            for penalty in plan['ridge_penalties']:
                matrix=np.eye(a.shape[-1])*penalty; matrix[-1,-1]=0.
                coefficient=np.linalg.solve(a.T@a+matrix,a.T@b); coefficients.append(coefficient)
                prediction=va_design@coefficient
                score=stability(prediction,data['targets']['residual'][va],data['labelmask'][va],months)
                candidates.append(dict(penalty=penalty,validation_stability=score))
                if score>best+1e-7:
                    best,selected,chosen=score,penalty,coefficient
            modelpath=base.with_suffix('.npz'); np.savez_compressed(modelpath,coefficients=chosen,all_coefficients=coefficients)
            vp,tp=va_design@chosen,te_design@chosen
            metadata=dict(selected_penalty=selected,validation_stability=best,candidates=candidates)
        elif family=='boosted_trees':
            cfg={k:v for k,v in plan['tree_parameters'].items() if k!='candidate_iterations'}
            model=HistGradientBoostingRegressor(**cfg,random_state=seed)
            with threadpool_limits(limits=2):
                model.fit(flat,y,sample_weight=weights)
                best,metadata=-np.inf,dict(candidates=[])
                for iteration,prediction in enumerate(model.staged_predict(validation_x),1):
                    if iteration not in plan['tree_parameters']['candidate_iterations']:
                        continue
                    prediction=prediction.reshape(len(va),-1)
                    score=stability(prediction,data['targets']['residual'][va],data['labelmask'][va],months)
                    metadata['candidates'].append(dict(iteration=iteration,validation_stability=score))
                    if score>best+1e-7:
                        best,selected,vp=score,iteration,prediction.copy()
                tp=tree_prediction(model,test_x,selected).reshape(len(te),-1)
            metadata.update(selected_iteration=selected,validation_stability=best)
            modelpath=base.with_suffix('.joblib'); joblib.dump(model,modelpath,compress=3)
        else:
            checkpoint,vp,tp,metadata=fit_network(data,x,target,tr,va,te,seed,plan)
            modelpath=base.with_suffix('.pt'); torch.save(checkpoint,modelpath)
        if not np.isfinite(tp[data['mask'][te]]).all():
            raise ValueError('Nonfinite eligible test forecast')
        predictionpath=base.with_name(base.name+'_predictions.npz')
        metapath=base.with_name(base.name+'_history.json')
        np.savez_compressed(predictionpath,validation=vp,test=tp); write(metapath,metadata)
        record=dict(view=view,year=year,target=target,family=family,seed=seed,
                    validation_stability=metadata['validation_stability'],
                    artifact_hashes={p.name:sha(p) for p in [modelpath,predictionpath,metapath]})
        write(recordpath,record)
        print(f'Fit {view} {year} {target} {family} seed {seed}',flush=True)
    return len(seeds)


def lock_selection(data,plan):
    choices=[]; hashes={}
    for year in plan['test_years']:
        _,va,_=splits(data,year); months=data['dates'][data['origin'][va]].to_period('M').astype(str)
        cells=[]
        for view in plan['views']:
            for target in plan['targets']:
                for family in plan['families']:
                    seeds=[plan['seeds'][0]] if family=='ridge' else plan['seeds']
                    available=[]
                    for seed in seeds:
                        path=RUN/view/str(year)/f'{family}_{target}_{seed}_predictions.npz'
                        hashes[str(path.relative_to(RUN))]=sha(path)
                        with np.load(path) as saved:
                            available.append(saved['validation'].copy())
                    score=ensemble_predictions(available,data['mask'][va])
                    value=stability(score,data['targets']['residual'][va],data['labelmask'][va],months)
                    cells.append(dict(view=view,target=target,family=family,validation_stability=value,protocol_order=len(cells)))
        ordered=[]
        while cells:
            best=max(c['validation_stability'] for c in cells)
            choice=min((c for c in cells if c['validation_stability']>=best-1e-7),key=lambda c:c['protocol_order'])
            ordered.append(choice); cells.remove(choice)
        choices.append(dict(year=year,top_three=ordered[:3],single_best=ordered[0],all_cells_ordered=ordered))
    value=dict(created_before_new_financial_metrics=not (RUN/'metrics.csv').exists(),choices=choices,validation_artifact_hashes=hashes)
    path=RUN/'SELECTION_LOCK.json'
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('Annual validation selections changed')
    write(path,value)


def train():
    plan=freeze(); data=data_cache()
    write(RUN/'FEATURES.json',dict(names=data['feature_names'],features=len(data['feature_names']),
          daily_training_origins=len(data['origin']),historical_security_ids=len(data['base']['panel'].tickers)))
    for view in plan['views']:
        for year in plan['test_years']:
            tr,va,te=splits(data,year); _,mean,std=normalize(data['views'][view],data['mask'],tr)
            folder=RUN/view/str(year); folder.mkdir(parents=True,exist_ok=True)
            values=dict(train=tr,validation=va,test=te,mean=mean,std=std)
            path=folder/'preprocessing.npz'
            if path.exists():
                with np.load(path) as saved:
                    for name,value in values.items():
                        np.testing.assert_array_equal(saved[name],value)
            else:
                np.savez_compressed(path,**values)
    jobs=[(v,y,t,f) for y in plan['test_years'] for v in plan['views'] for t in plan['targets'] for f in plan['families']]
    completed=0
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(fit_group,*job) for job in jobs]):
            completed+=future.result()
            write(RUN/'PROGRESS.json',dict(stage='training',selected_fold_models=completed,planned=176))
            print(f'{completed}/176 selected fold models; no financial test metrics',flush=True)
    records=list(RUN.glob('*/*/*_record.json'))
    if len(records)!=176:
        raise ValueError('Incomplete broad model family')
    lock_selection(data,plan)
    hashes={str(p.relative_to(RUN)):sha(p) for v in plan['views'] for p in (RUN/v).rglob('*')
            if p.suffix in ['.npz','.pt','.joblib','.json']}
    hashes['SELECTION_LOCK.json']=sha(RUN/'SELECTION_LOCK.json')
    write(RUN/'FITS_COMPLETE_LOCK.json',dict(selected_fold_models=176,candidate_fits=208,hashes=hashes))
    print('All broad model fits and validation-only annual selections are locked.',flush=True)


def complete():
    verify(); plan=json.loads((RUN/'PROTOCOL.json').read_text())
    lock=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())
    if lock['selected_fold_models']!=176:
        raise ValueError('All fits must finish before financial evaluation')
    for name,value in lock['hashes'].items():
        if sha(RUN/name)!=value:
            raise ValueError('Broad fit/selection artifact changed')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['train','evaluate','audit','publish','all'],default='all')
    args=parser.parse_args()
    if args.stage in ['train','all'] and not (RUN/'FITS_COMPLETE_LOCK.json').exists():
        train()
    for stage in ['evaluate','audit','publish'] if args.stage=='all' else [args.stage]:
        if stage=='train':
            continue
        marker=dict(evaluate='EVALUATED.json',audit='AUDIT.json',publish='FINAL_PUBLICATION.json')[stage]
        if (RUN/marker).exists():
            continue
        subprocess.run([sys.executable,str(ROOT/'report_ndx_broad.py'),stage],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
