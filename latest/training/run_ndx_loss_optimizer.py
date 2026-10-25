"""Finite, frozen loss/optimizer training experiment; no outcome-based extension."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import unittest
import numpy as np

from ndx_loss_optimizer import RUN,PARENT,CONFIG,ROOT,paths,reference,fit_one,checked_record,prepared_fold,candidate_specs,select_learning_rates
from run_ndx_extended_hours import complete as parent_complete
from run_ndx_alpaca_training import sha,write
from training_device import configure_device,device_metadata


def verify():
    for p,h in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        if sha(p)!=h:raise ValueError('Frozen training input changed: '+p)


def freeze():
    plan=json.loads(CONFIG.read_text());RUN.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.loadTestsFromName('test_ndx_loss_optimizer')
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise ValueError('Loss/optimizer preflight failed')
    parent_complete();lock=json.loads((PARENT/'INPUT_CODE_LOCK.json').read_text())
    for name,h in json.loads((PARENT/'FITS_COMPLETE_LOCK.json').read_text())['hashes'].items():lock[str(PARENT/name)]=h
    for p in [CONFIG,Path(__file__),ROOT/'ndx_loss_optimizer.py',ROOT/'test_ndx_loss_optimizer.py',ROOT/'pseudohuber_think_mixer.py',PARENT/'FITS_COMPLETE_LOCK.json']:
        lock[str(p.resolve())]=sha(p)
    target=RUN/'INPUT_CODE_LOCK.json'
    if target.exists() and json.loads(target.read_text())!=lock:raise ValueError('Cannot replace frozen study')
    if not target.exists():
        write(target,lock);write(RUN/'PROTOCOL.json',plan)
        write(RUN/'UNIT_TESTS.json',dict(status='passed',tests=result.testsRun))
        write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),new_fits_absent=not (RUN/'fits').exists(),protocol_sha256=sha(CONFIG)))
        write(RUN/'ENVIRONMENT.json',device_metadata(configure_device('cuda')))
        snapshot=RUN/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for p in [Path(__file__),ROOT/'ndx_loss_optimizer.py',ROOT/'test_ndx_loss_optimizer.py']:shutil.copyfile(p,snapshot/p.name)
    verify();print('Frozen loss/optimizer protocol and passed',result.testsRun,'preflight tests',flush=True)


def parity():
    if (RUN/'PARITY.json').exists():return
    verify();plan=json.loads(CONFIG.read_text());year=2025;seed=7
    data,x,extra,tr,va,te,groups=prepared_fold(year)
    record=fit_one(data,x,extra,tr,va,te,groups,configure_device('cuda'),'mse','adamw',.001,year,seed,RUN/'parity',plan)
    previous=checked_record('mse','adamw',.001,year,seed)
    if record['chosen_epoch']!=previous['chosen_epoch']:raise ValueError('Control selected epoch changed')
    with np.load(RUN/'parity/7_predictions.npz') as a,np.load(paths('mse','adamw',.001,year,seed)[1]) as b:
        for name in ['validation','test']:np.testing.assert_allclose(a[name],b[name],atol=1e-6,rtol=0)
    write(RUN/'PARITY.json',dict(status='passed',year=year,seed=seed,chosen_epoch=record['chosen_epoch'],fresh_control_fits=1))
    print('New trainer exactly reproduced the parent selected epoch and forecasts',flush=True)


def job(spec):
    verify();plan=json.loads(CONFIG.read_text());loss,optimizer,year,rates,seeds=spec
    data,x,extra,tr,va,te,groups=prepared_fold(year);device=configure_device('cuda');count=0
    for lr in rates:
        for seed in seeds:
            if reference(loss,optimizer,lr):checked_record(loss,optimizer,lr,year,seed);continue
            cp,pred,hist,record=paths(loss,optimizer,lr,year,seed)
            if record.exists():checked_record(loss,optimizer,lr,year,seed)
            else:fit_one(data,x,extra,tr,va,te,groups,device,loss,optimizer,lr,year,seed,record.parent,plan)
            count+=1
            print(f'Fit {loss} {optimizer} lr={lr} year={year} seed={seed}',flush=True)
    return count


def run_jobs(jobs,phase):
    count=0
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,spec) for spec in jobs]
        for future in as_completed(futures):
            count+=future.result()
            write(RUN/'PROGRESS.json',dict(stage=phase,phase_fit_cells_completed=count))
    return count


def train():
    verify();parity();plan=json.loads(CONFIG.read_text())
    jobs=[(loss,'adamw',year,[.001],plan['seeds']) for year in plan['years'] for loss in ['huber','pseudohuber']]
    jobs += [('mse',opt,year,rates,plan['pilot_seeds']) for year in plan['years'] for opt,rates in plan['learning_rates'].items()]
    run_jobs(jobs,'loss_and_pilot_fits')
    selection=select_learning_rates(plan)
    target=RUN/'LR_SELECTION.json'
    if target.exists() and json.loads(target.read_text())!=selection:raise ValueError('Validation LR choice changed')
    write(target,selection);print('Locked all optimizer learning rates using validation MSE only',flush=True)
    jobs=[('mse',opt,year,[selection[str(year)][opt]['lr']],plan['seeds']) for year in plan['years'] for opt in plan['learning_rates']]
    run_jobs(jobs,'selected_optimizer_fits')
    specs=candidate_specs(plan,selection);lock={}
    for spec in specs:
        checked_record(*spec)
        for p in paths(*spec):lock[str(p.resolve())]=sha(p)
    lock[str(target.resolve())]=sha(target);lock[str((RUN/'PARITY.json').resolve())]=sha(RUN/'PARITY.json')
    write(RUN/'FITS_COMPLETE_LOCK.json',dict(unique_candidate_fits=len(specs),new_candidate_fits=sum(not reference(*spec[:3]) for spec in specs),parent_controls=40,hashes=lock))
    write(RUN/'PROGRESS.json',dict(stage='fits_complete',new_candidate_fits=sum(not reference(*spec[:3]) for spec in specs),control_reproduction_fits=1))
    print('Completed and locked every fit; no new financial test metrics inspected',flush=True)


def complete():
    verify();record=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())
    for p,h in record['hashes'].items():
        if sha(p)!=h:raise ValueError('Changed completed fit: '+p)
    return json.loads(CONFIG.read_text()),json.loads((RUN/'LR_SELECTION.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','train'])
    globals()[parser.parse_args().stage]()
