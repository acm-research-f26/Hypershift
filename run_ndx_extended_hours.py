"""Freeze and run the 120-fit extended-hours information experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd
import torch

from ndx_extended_hours import ExtendedModel,prepared,raw_auxiliary,scale_auxiliary,predict,timing
from pit_research_engine import fold,historical_groups
from prepare_ndx_extended_hours import OUT as FEATURES
from prepare_ndx_alpaca import CALENDAR,calendar_frame
from run_ndx_alpaca_training import ROOT,sha,write
from run_robust_research import scale_inputs
from training_device import configure_device,device_metadata,tensor_on

RUN=ROOT/'runs/ndx_extended_hours_20261024'
CONFIG=ROOT/'config/ndx_extended_hours_20261024.json'
ORIGINAL=ROOT/'runs/nasdaq100_historical_monthly_20261011'
SEEDS=ROOT/'runs/ndx_seed_expansion_20261020'
SOURCE_NAMES=['prepare_ndx_extended_hours.py','ndx_extended_hours.py','run_ndx_extended_hours.py','test_ndx_extended_hours.py']


def immutable(path,value):
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Refusing to change frozen artifact: '+str(path))
    write(path,value)


def verify():
    for name,value in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        if sha(name)!=value:raise ValueError('Frozen input/source changed: '+name)


def preflight():
    RUN.mkdir(parents=True,exist_ok=True)
    hashes={n:sha(ROOT/n) for n in SOURCE_NAMES}
    receipt=RUN/'UNIT_TESTS.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('source_hashes')==hashes:
        if json.loads(receipt.read_text())['status']!='passed':raise ValueError('Preflight failed')
        return
    if (RUN/'INPUT_CODE_LOCK.json').exists():raise ValueError('Frozen training tests changed')
    suite=unittest.defaultTestLoader.loadTestsFromName('test_ndx_extended_hours')
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    write(receipt,dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,source_hashes=hashes))
    if not result.wasSuccessful():raise ValueError('Extended-hours tests failed')


def freeze():
    plan=json.loads(CONFIG.read_text());review=json.loads((FEATURES/'REVIEW.json').read_text())
    if review['code_sha256']!=sha(ROOT/'prepare_ndx_extended_hours.py'):raise ValueError('Feature builder changed after review')
    lock=json.loads((ROOT/'runs/nasdaq100_alpaca_20261019/BASELINE_DATA_CODE_LOCK.json').read_text())
    for p in [CONFIG,FEATURES/'REVIEW.json',FEATURES/'features.npz',CALENDAR]+[ROOT/n for n in SOURCE_NAMES]:lock[str(p)]=sha(p)
    for name,expected in review['artifact_hashes'].items():
        if sha(FEATURES/name)!=expected:raise ValueError('Reviewed feature artifact changed')
    immutable(RUN/'INPUT_CODE_LOCK.json',lock);immutable(RUN/'PROTOCOL.json',plan);verify()
    if not (RUN/'FREEZE.json').exists():
        snapshot=RUN/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for name in lock:
            p=Path(name)
            if p.suffix=='.py':(snapshot/p.name).write_bytes(p.read_bytes())
        frozen=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(CONFIG),
                    planned_fits=120,fits_absent=not any(RUN.glob('*/*/*_record.json')),all_test_years_exploratory=True)
        write(RUN/'FREEZE.json',frozen)
        write(RUN/'ENVIRONMENT.json',device_metadata(configure_device('cuda')))
        with (ROOT/'docs/audit/research_ledger.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(experiment=RUN.name,protocol_sha256=sha(CONFIG),planned_fits=120,
                         timestamp_utc=frozen['timestamp_utc'],primary='ten-seed after-hours minus daily',test_history='previously inspected'))+'\n')
    return plan


def old_paths(year,seed):
    original=seed in [7,19,42,61,97]
    folder=(ORIGINAL if original else SEEDS)/str(year)
    prediction=folder/(f'think_mix_{seed}_predictions.npz' if original else f'{seed}_predictions.npz')
    return prediction,folder/f'think_mix_{seed}.pt'


def fit_group(view,year):
    verify();plan=json.loads((RUN/'PROTOCOL.json').read_text());data,cubes=prepared();train,val,test=fold(data,year)
    x,mean,std=scale_inputs(data,train);extra,amean,astd=scale_auxiliary(raw_auxiliary(data,cubes,view),data['mask'],train)
    folder=RUN/view/str(year);groups=historical_groups(data,train);device=configure_device('cuda')
    with np.load(folder/'preprocessing.npz') as saved:
        for name,value in dict(train=train,val=val,test=test,mean=mean,std=std,aux_mean=amean,aux_std=astd).items():np.testing.assert_array_equal(saved[name],value)
    target=data['y']*100
    for seed in plan['seeds']:
        record=folder/f'{seed}_record.json'
        if record.exists():
            for name,value in json.loads(record.read_text())['hashes'].items():
                if sha(folder/name)!=value:raise ValueError('Fit cache changed')
            continue
        torch.manual_seed(seed);model=ExtendedModel(len(data['panel'].tickers),groups,view).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
        best=np.inf;state=None;stale=0;chosen=0;history=[]
        for epoch in range(1,plan['epochs']+1):
            model.train();total=0.;count=0
            for start in range(0,len(train),plan['batch']):
                sub=train[start:start+plan['batch']];optimizer.zero_grad(set_to_none=True)
                score,_=model(tensor_on(x[sub],device),tensor_on(extra[sub],device),tensor_on(data['mask'][sub].astype('float32'),device))
                observed=tensor_on(data['labelmask'][sub].astype('float32'),device)
                loss=((score-tensor_on(target[sub],device)).square()*observed).sum()/observed.sum().clamp_min(1)
                if not torch.isfinite(loss):raise ValueError('Nonfinite extended-hours loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_norm_cap']);optimizer.step()
                total+=float(loss.detach())*len(sub);count+=len(sub)
            vp=predict(model,x,extra,data['mask'],val,device,plan['batch'])
            mse=float(((vp-target[val])**2)[data['labelmask'][val]].mean())
            history.append(dict(epoch=epoch,training_loss=total/count,validation_mse_pp2=mse))
            if mse<best-1e-7:
                best=mse;chosen=epoch;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if stale>=plan['patience']:break
        model.load_state_dict(state);vp,tp=[predict(model,x,extra,data['mask'],ids,device,plan['batch']) for ids in [val,test]]
        if not np.isfinite(tp[data['mask'][test]]).all():raise ValueError('Nonfinite eligible forecast')
        parent_files=[]
        if view=='daily':
            previous,cp=old_paths(year,seed);parent=torch.load(cp,map_location='cpu',weights_only=False)
            if chosen!=parent['epoch']:raise ValueError('Daily control checkpoint differs from original')
            with np.load(previous) as saved:
                np.testing.assert_allclose(vp,saved['validation'],atol=1e-6,rtol=0);np.testing.assert_allclose(tp,saved['test'],atol=1e-6,rtol=0)
            parent_files=[previous,cp]
        checkpoint=folder/f'{seed}.pt';prediction=folder/f'{seed}_predictions.npz';hp=folder/f'{seed}_history.json'
        torch.save(dict(state_dict=state,view=view,seed=seed,year=year,groups=[sorted(g) for g in groups],epoch=chosen,
                        parameters=sum(p.numel() for p in model.parameters()),device=device_metadata(device)),checkpoint)
        np.savez_compressed(prediction,validation=vp,test=tp);write(hp,history)
        write(record,dict(view=view,year=year,seed=seed,chosen_epoch=chosen,validation_mse=best,
              parent_hashes={str(p):sha(p) for p in parent_files},hashes={p.name:sha(p) for p in [checkpoint,prediction,hp]}))
        print(f'Fit {view} {year} seed {seed}: epoch {chosen}',flush=True)
    return len(plan['seeds'])


def train():
    preflight()
    if not (FEATURES/'REVIEW.json').exists():subprocess.run([sys.executable,str(ROOT/'prepare_ndx_extended_hours.py')],cwd=ROOT,check=True)
    plan=freeze();data,cubes=prepared();cal=calendar_frame(json.loads(CALENDAR.read_text()))
    time=timing(data,cal);tp=RUN/'TIMING.csv'
    if tp.exists():
        pd.testing.assert_frame_equal(pd.read_csv(tp),time)
    else:time.to_csv(tp,index=False)
    for view in plan['views']:
        raw=raw_auxiliary(data,cubes,view)
        for year in plan['test_years']:
            tr,va,te=fold(data,year);_,mean,std=scale_inputs(data,tr);_,amean,astd=scale_auxiliary(raw,data['mask'],tr)
            folder=RUN/view/str(year);folder.mkdir(parents=True,exist_ok=True);path=folder/'preprocessing.npz'
            values=dict(train=tr,val=va,test=te,mean=mean,std=std,aux_mean=amean,aux_std=astd)
            if path.exists():
                with np.load(path) as saved:
                    for name,value in values.items():np.testing.assert_array_equal(saved[name],value)
            else:np.savez_compressed(path,**values)
    count=0;jobs=[(v,y) for y in plan['test_years'] for v in plan['views']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(fit_group,*job) for job in jobs]):
            count+=future.result();write(RUN/'PROGRESS.json',dict(stage='training',fits_complete=count,planned=120))
            print(f'{count}/120 extended-hours fits locked; no new test metrics',flush=True)
    records=list(RUN.glob('*/*/*_record.json'))
    if len(records)!=120:raise ValueError('Incomplete extended-hours fit inventory')
    files=[p for view in plan['views'] for p in (RUN/view).rglob('*') if p.is_file()]+[tp]
    immutable(RUN/'FITS_COMPLETE_LOCK.json',dict(fits=120,hashes={str(p.relative_to(RUN)):sha(p) for p in files}))
    write(RUN/'PROGRESS.json',dict(stage='fits_complete',fits_complete=120))


def complete():
    verify();locked=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())
    if locked['fits']!=120:raise ValueError('Incomplete fits')
    for name,value in locked['hashes'].items():
        if sha(RUN/name)!=value:raise ValueError('Frozen fit artifact changed')
    return json.loads((RUN/'PROTOCOL.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['train','evaluate','audit','publish'],default='train');args=parser.parse_args()
    if args.stage=='train':train()
    else:subprocess.run([sys.executable,str(ROOT/'report_ndx_extended_hours.py'),args.stage],cwd=ROOT,check=True)
