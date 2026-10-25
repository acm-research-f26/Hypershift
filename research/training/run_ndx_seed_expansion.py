"""Fixed 20-seed expansion; preserves the previous five and every new seed."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from pit_research_engine import load_data,fold,historical_groups
from run_ndx_alpaca_training import ROOT,sha,write
from run_robust_research import fit_model,predict,scale_inputs

RUN=ROOT/'runs/ndx_seed_expansion_20261020'
CONFIG=ROOT/'config/ndx_seed_expansion_20261020.json'


def verify():
    for p,h in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        if sha(p)!=h:raise ValueError('Frozen seed-study input changed: '+p)


def freeze():
    RUN.mkdir(parents=True,exist_ok=True);plan=json.loads(CONFIG.read_text())
    lock=json.loads((ROOT/'runs/nasdaq100_alpaca_20261019/BASELINE_DATA_CODE_LOCK.json').read_text())
    for p in [CONFIG,ROOT/'run_ndx_seed_expansion.py',ROOT/'test_ndx_seed_expansion.py',ROOT/'ndx_policy_engine.py']:
        lock[str(p)]=sha(p)
    receipt=json.loads((RUN/'UNIT_TESTS.json').read_text())
    if receipt['status']!='passed' or receipt['runner_sha256']!=sha(__file__) or receipt['test_sha256']!=sha(ROOT/'test_ndx_seed_expansion.py'):raise ValueError('Seed expansion preflight missing/stale')
    path=RUN/'INPUT_CODE_LOCK.json'
    if path.exists() and json.loads(path.read_text())!=lock:raise ValueError('Refusing changed frozen experiment')
    write(path,lock);verify()
    if not (RUN/'FREEZE.json').exists():
        write(RUN/'PROTOCOL.json',plan);snapshot=RUN/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for p in lock:
            path=Path(p)
            if path.suffix=='.py':(snapshot/path.name).write_bytes(path.read_bytes())
        write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(CONFIG),fits_absent=not any(RUN.glob('*/*_record.json'))))
        write(RUN/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,workers=2))
        with (ROOT/'docs/audit/research_ledger.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(experiment=RUN.name,protocol_sha256=sha(CONFIG),planned_fits=80,all_seeds_prespecified=True))+'\n')
    return plan


def job(year,seed):
    from training_device import configure_device
    verify();configure_device('cuda');data=load_data();tr,va,te=fold(data,year);x,mean,std=scale_inputs(data,tr);groups=historical_groups(data,tr)
    folder=RUN/str(year);folder.mkdir(parents=True,exist_ok=True);record=folder/f'{seed}_record.json'
    if record.exists():
        for name,h in json.loads(record.read_text())['hashes'].items():
            if sha(folder/name)!=h:raise ValueError('Seed fit cache changed')
        return 1
    prep=folder/f'{seed}_preprocessing.npz'
    if not prep.exists():np.savez_compressed(prep,train=tr,validation=va,test=te,mean=mean,std=std)
    args=SimpleNamespace(device='cuda',lr=.001,epochs=30,patience=6,batch=8,causal_membership=True)
    model,chosen,epochs=fit_model(data,x,tr,va,'think_mix',seed,groups,args,folder)
    vp,tp=[predict(model,x,data['mask'],ids,8) for ids in [va,te]]
    prediction=folder/f'{seed}_predictions.npz';np.savez_compressed(prediction,validation=vp,test=tp)
    write(record,dict(seed=seed,year=year,epoch=chosen,epochs_run=epochs,hashes={p.name:sha(p) for p in [
        folder/f'think_mix_{seed}.pt',folder/f'think_mix_{seed}_history.json',prediction]}))
    print('Fit expansion',year,seed,flush=True);return 1


def bootstrap_draws(plan):
    n=len(plan['original_seeds'])+len(plan['new_seeds']);cfg=plan['seed_bootstrap']
    return np.random.default_rng(cfg['seed']).integers(0,n,(cfg['replicates'],n))


def complete():
    verify();lock=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())
    if lock['fits']!=80:raise ValueError('Incomplete seed expansion')
    for p,h in lock['hashes'].items():
        if sha(RUN/p)!=h:raise ValueError('Seed fit changed')
    return json.loads((RUN/'PROTOCOL.json').read_text())


def train():
    plan=freeze();done=0
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,y,s) for y in plan['test_years'] for s in plan['new_seeds']]):
            done+=future.result();write(RUN/'PROGRESS.json',dict(stage='training',fits_complete=done,planned=80))
    paths=[p for y in plan['test_years'] for p in (RUN/str(y)).rglob('*') if p.is_file()]
    if len(list(RUN.glob('*/*_record.json')))!=80:raise ValueError('Seed fit inventory failed')
    write(RUN/'FITS_COMPLETE_LOCK.json',dict(fits=80,hashes={str(p.relative_to(RUN)):sha(p) for p in paths}))
    write(RUN/'PROGRESS.json',dict(stage='fits_complete',fits_complete=80));print('All 80 new seed fits locked.',flush=True)


if __name__=='__main__':train()
