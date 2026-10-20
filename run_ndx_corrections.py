"""Train 20 cores and 80 bounded adapters before any new test performance."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import platform
from types import SimpleNamespace

import numpy as np
import torch

from ndx_corrections import prepared,partitions,normalized,Correction,correction_loss,proxy,choose,accepted,ensemble
from pit_research_engine import historical_groups
from run_ndx_alpaca_training import ROOT,sha,write
from run_robust_research import fit_model,predict,scale_inputs
from training_device import configure_device,tensor_on

RUN=ROOT/'runs/ndx_corrections_20261020'
CONFIG=ROOT/'config/ndx_corrections_20261020.json'
CELLS=[f'{view}__{loss}' for view in ['intraday_window','rich_features'] for loss in ['top_rank','net_utility']]


def immutable(path,value):
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('Refusing to change locked artifact: '+str(path))
    write(path,value)


def verify():
    for path,value in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        if sha(path)!=value:
            raise ValueError('Frozen correction input/source changed: '+path)


def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'runs/ndx_broad_20261020/INPUT_CODE_LOCK.json'
    lock=json.loads(parent.read_text())
    files=[CONFIG,ROOT/'ndx_corrections.py',ROOT/'run_ndx_corrections.py',ROOT/'test_ndx_corrections.py']
    lock.update({str(p):sha(p) for p in files})
    receipt=json.loads((RUN/'UNIT_TESTS.json').read_text())
    if receipt['status']!='passed' or receipt['hashes']!={p.name:sha(p) for p in files if p.suffix=='.py'}:
        raise ValueError('Correction tests missing/stale')
    immutable(RUN/'INPUT_CODE_LOCK.json',lock);verify()
    plan=json.loads(CONFIG.read_text());immutable(RUN/'PROTOCOL.json',plan)
    if not (RUN/'FREEZE.json').exists():
        snapshot=RUN/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for name in lock:
            path=Path(name)
            if path.suffix=='.py':(snapshot/path.name).write_bytes(path.read_bytes())
        write(RUN/'ENVIRONMENT.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
            torch=torch.__version__,cuda=torch.version.cuda,workers=2,precision='FP32; deterministic; no AMP/TF32'))
        write(RUN/'FREEZE.json',dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(CONFIG),
              new_fits_absent=not any(RUN.glob('*/*/*_record.json')),planned_fits=100))
        ledger=ROOT/'docs/audit/research_ledger.jsonl'
        with ledger.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(experiment=RUN.name,protocol_sha256=sha(CONFIG),
                         frozen_at_utc=json.loads((RUN/'FREEZE.json').read_text())['frozen_at_utc'],
                         previous_history_exploratory=True,planned_fits=100))+'\n')
    return plan


def adapter_predict(model,x,core,mask,ids,device):
    model.eval(); predictions=[]
    with torch.no_grad():
        for sub in np.array_split(ids,max(1,int(np.ceil(len(ids)/8)))):
            predictions.append(model(tensor_on(x[sub],device),tensor_on(core[sub],device),tensor_on(mask[sub],device))[0].cpu().numpy())
    return np.concatenate(predictions)


def job(year,seed):
    verify();plan=json.loads((RUN/'PROTOCOL.json').read_text());data=prepared();base=data['base']
    tr,va,te,h1,h2=partitions(base,year);folder=RUN/str(year)/str(seed);folder.mkdir(parents=True,exist_ok=True)
    device=configure_device('cuda');x,mean,std=scale_inputs(base,tr);groups=historical_groups(base,tr)
    norm=normalized(data,tr)
    prep=folder/'preprocessing.npz'
    if not prep.exists():
        np.savez_compressed(prep,train=tr,validation=va,test=te,h1=h1,h2=h2,mean=mean,std=std,
            **{f'{v}_{k}':values[i] for v,values in norm.items() for i,k in [(1,'mean'),(2,'std')]})
    core_record=folder/'core_record.json'
    if not core_record.exists():
        args=SimpleNamespace(device='cuda',lr=.001,epochs=30,patience=6,batch=8,causal_membership=True)
        model,chosen,epochs=fit_model(base,x,tr,h1,'think_mix',seed,groups,args,folder)
        forecast=predict(model,x,base['mask'],np.arange(len(x)),8).astype('float32')
        np.savez_compressed(folder/'core_predictions.npz',all=forecast,validation=forecast[va],test=forecast[te])
        write(core_record,dict(chosen_epoch=chosen,epochs_run=epochs,
              hashes={p.name:sha(p) for p in [folder/f'think_mix_{seed}.pt',folder/f'think_mix_{seed}_history.json',folder/'core_predictions.npz']}))
        # Explicit freeze; adapters receive detached exported predictions only.
        for parameter in model.parameters():parameter.requires_grad_(False)
        del model
    for p,h in json.loads(core_record.read_text())['hashes'].items():
        if sha(folder/p)!=h:raise ValueError('Core cache changed')
    with np.load(folder/'core_predictions.npz') as saved:core=saved['all'].copy()
    cfg=plan['adapter']
    for cell in CELLS:
        record=folder/f'{cell}_record.json'
        if record.exists():
            for p,h in json.loads(record.read_text())['hashes'].items():
                if sha(folder/p)!=h:raise ValueError('Adapter cache changed')
            continue
        view,kind=cell.split('__');features=norm[view][0];torch.manual_seed(seed)
        model=Correction(features.shape[-1],cfg['hidden']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        history=[];artifacts=[]
        for epoch in range(1,cfg['epochs']+1):
            model.train();losses=[]
            for start in range(0,len(tr),cfg['batch_origins']):
                sub=tr[start:start+cfg['batch_origins']];optimizer.zero_grad(set_to_none=True)
                score,delta,scale=model(tensor_on(features[sub],device),tensor_on(core[sub],device),tensor_on(base['mask'][sub],device))
                loss=correction_loss(score,delta,scale,tensor_on((base['y'][sub]*100).astype('float32'),device),
                     tensor_on(base['mask'][sub],device),tensor_on(base['labelmask'][sub],device),
                     tensor_on(data['spy_active'][sub].astype('float32'),device),kind)
                if not torch.isfinite(loss):raise ValueError('Nonfinite correction loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_norm_cap']);optimizer.step()
                losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,training_loss=float(np.mean(losses))))
            if epoch in cfg['checkpoint_epochs']:
                path=folder/f'{cell}_{epoch}.pt'
                torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},features=features.shape[-1],
                                hidden=cfg['hidden'],epoch=epoch,seed=seed,core_sha256=sha(folder/f'think_mix_{seed}.pt')),path)
                prediction=folder/f'{cell}_{epoch}_predictions.npz'
                vp,tp=[adapter_predict(model,features,core,base['mask'],sub,device) for sub in [va,te]]
                np.savez_compressed(prediction,validation=vp,test=tp);artifacts.extend([path,prediction])
        historypath=folder/f'{cell}_history.json';write(historypath,history);artifacts.append(historypath)
        write(record,dict(cell=cell,year=year,seed=seed,hashes={p.name:sha(p) for p in artifacts}))
        print('Fit correction',year,seed,cell,flush=True)
    return 5


def validation_predictions(year,plan):
    cores=[];adapters={}
    for seed in plan['seeds']:
        folder=RUN/str(year)/str(seed)
        with np.load(folder/'core_predictions.npz') as saved:cores.append(saved['validation'].copy())
        for cell in CELLS:
            for epoch in plan['adapter']['checkpoint_epochs']:
                with np.load(folder/f'{cell}_{epoch}_predictions.npz') as saved:
                    adapters.setdefault((cell,epoch),[]).append(saved['validation'].copy())
    return cores,adapters


def candidate_score(candidate,cores,adapters,mask):
    if candidate is None:return ensemble(cores,mask)
    available=adapters[candidate['cell'],candidate['epoch']];shrink=candidate['shrink']
    return ensemble([c+shrink*(a-c) for c,a in zip(cores,available)],mask)


def validation_stage(data,plan,write_files=True):
    base=data['base'];choices=[];candidates=[dict(cell=cell,epoch=epoch,shrink=shrink) for cell in CELLS
        for epoch in plan['adapter']['checkpoint_epochs'] for shrink in plan['shrink_factors'] if shrink]
    cache={}
    for year in plan['test_years']:
        _,va,_,h1,_=partitions(base,year);local=np.searchsorted(va,h1);cores,adapters=validation_predictions(year,plan)
        mask=base['mask'][va];baseline=candidate_score(None,cores,adapters,mask)
        rows=[proxy(candidate_score(c,cores,adapters,mask)[local],baseline[local],mask[local],
                    base['labelmask'][h1],base['y'][h1]*100) for c in candidates]
        controls={cell:choose([c for c in candidates if c['cell']==cell],
                             [s for c,s in zip(candidates,rows) if c['cell']==cell]) for cell in CELLS}
        choices.append(dict(year=year,primary=choose(candidates,rows),controls=controls,
                       candidates=[dict(candidate=c,statistics=s) for c,s in zip(candidates,rows)]))
        cache[year]=(va,cores,adapters,mask,baseline)
    selection=dict(choices=choices,created_before_h2=not (RUN/'VALIDATION_GATES.json').exists(),
                   created_before_test_metrics=not (RUN/'metrics.csv').exists())
    if write_files:immutable(RUN/'H1_SELECTION_LOCK.json',selection)
    # All years' H1 choices now locked. H2 may only accept/reject those choices.
    gates=[]
    for choice in choices:
        year=choice['year'];_,_,_,_,h2=partitions(base,year);va,cores,adapters,mask,baseline=cache[year];local=np.searchsorted(va,h2)
        results={}
        for name,selected in dict(primary=choice['primary'],**choice['controls']).items():
            candidate=selected['candidate'];score=candidate_score(candidate,cores,adapters,mask)
            statistic=proxy(score[local],baseline[local],mask[local],base['labelmask'][h2],base['y'][h2]*100)
            results[name]=dict(h1_candidate=candidate,h2_statistics=statistic,accepted=accepted(statistic,candidate),
                               deployed_candidate=candidate if accepted(statistic,candidate) else None)
        gates.append(dict(year=year,results=results))
    value=dict(gates=gates,h1_lock_sha256=sha(RUN/'H1_SELECTION_LOCK.json'),
               created_before_test_metrics=not (RUN/'metrics.csv').exists())
    if write_files:immutable(RUN/'VALIDATION_GATES.json',value)
    return choices,gates


def train():
    plan=freeze();data=prepared();count=0
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,year,seed) for year in plan['test_years'] for seed in plan['seeds']]):
            count+=future.result();write(RUN/'PROGRESS.json',dict(stage='training',fits_complete=count,planned=100))
            print(f'{count}/100 fits complete; no new test metrics',flush=True)
    records=list(RUN.glob('*/*/*_record.json'))
    if len(records)!=100:raise ValueError('Incomplete correction fit inventory')
    validation_stage(data,plan)
    files=[p for year in plan['test_years'] for p in (RUN/str(year)).rglob('*') if p.is_file()]
    files.extend([RUN/'H1_SELECTION_LOCK.json',RUN/'VALIDATION_GATES.json'])
    immutable(RUN/'FITS_COMPLETE_LOCK.json',dict(fits=100,checkpoints=260,hashes={str(p.relative_to(RUN)):sha(p) for p in files}))
    write(RUN/'PROGRESS.json',dict(stage='training_and_validation_complete',fits_complete=100))


def complete():
    verify();lock=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())
    for name,value in lock['hashes'].items():
        if sha(RUN/name)!=value:raise ValueError('Frozen fit/selection changed: '+name)
    return json.loads((RUN/'PROTOCOL.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',default='train',choices=['train','evaluate','audit','publish'])
    stage=parser.parse_args().stage
    if stage=='train':train()
    else:subprocess.run([sys.executable,str(ROOT/'report_ndx_corrections.py'),stage],cwd=ROOT,check=True)
