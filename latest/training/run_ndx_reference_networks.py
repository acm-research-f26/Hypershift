"""Freeze and run the finite author-network transfer comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import unittest
import numpy as np
from ndx_reference_training import RUN,CONFIG,KINDS,prepare,train_one
from ndx_reference_models import ROOT,REFERENCES
from ndx_loss_optimizer import RUN as LOSS_RUN
from run_ndx_loss_optimizer import verify as loss_verify
from run_ndx_alpaca_training import sha,write


def verify():
    for p,h in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        if sha(p)!=h:raise ValueError('Reference input/source changed: '+p)


def freeze():
    plan=json.loads(CONFIG.read_text());RUN.mkdir(exist_ok=True,parents=True)
    tests=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromName('test_ndx_reference_models'))
    if not tests.wasSuccessful():raise ValueError('Reference adapter preflight failed')
    loss_verify();lock=json.loads((LOSS_RUN/'INPUT_CODE_LOCK.json').read_text())
    files=[CONFIG,Path(__file__),ROOT/'ndx_reference_models.py',ROOT/'ndx_reference_training.py',ROOT/'test_ndx_reference_models.py']
    files+=list(REFERENCES.rglob('*.py'))+list(REFERENCES.rglob('*.yaml'))+list(REFERENCES.rglob('REFERENCE.json'))
    files+=list((ROOT/'.cache/reference_dependencies_20261024').glob('*'))
    files+=list((ROOT/'data/nasdaq_monthly_pit/jkp_themes_20261024').glob('*'))
    for p in files:lock[str(p.resolve())]=sha(p)
    target=RUN/'INPUT_CODE_LOCK.json'
    if target.exists() and json.loads(target.read_text())!=lock:raise ValueError('Cannot replace frozen reference experiment')
    if not target.exists():
        write(target,lock);write(RUN/'PROTOCOL.json',plan)
        write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(CONFIG),candidate_fits_absent=True,tests_passed=tests.testsRun))
        write(RUN/'REFERENCE_REVISIONS.json',{name:json.loads((REFERENCES/name/'REFERENCE.json').read_text())['commit'] for name in ['MASTER','dynamic_hypergraph','PRISM_VQ']})
    verify();print('Reference transfer protocol frozen before candidate training',flush=True)


def job(kind,year):
    verify();plan=json.loads(CONFIG.read_text());p=prepare(year)
    for seed in plan['seeds']:
        result=train_one(p,kind,year,seed,plan)
        print(f'Reference fit {kind}/{year}/{seed}: epochs={result["chosen_epochs"]}',flush=True)
    return kind,year


def train():
    verify();plan=json.loads(CONFIG.read_text());done=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,kind,year) for year in plan['years'] for kind in KINDS]
        for future in as_completed(futures):
            done.append(future.result());write(RUN/'PROGRESS.json',dict(stage='training',completed_annual_network_jobs=done))
    lock={};count=0
    for kind in KINDS:
        for year in plan['years']:
            for seed in plan['seeds']:
                folder=RUN/kind/str(year);record=folder/f'{seed}_record.json';value=json.loads(record.read_text())
                for name,h in value['hashes'].items():
                    p=folder/name
                    if sha(p)!=h:raise ValueError('Reference fit artifact changed')
                    lock[str(p.resolve())]=h
                lock[str(record.resolve())]=sha(record);count+=1
    assert count==120
    write(RUN/'FITS_COMPLETE_LOCK.json',dict(annual_seed_fits=count,prism_representation_stages=40,hashes=lock))
    write(RUN/'PROGRESS.json',dict(stage='fits_complete',fits=count))
    print('All 120 reference fits locked before test financial metrics',flush=True)


def complete():
    verify();lock=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())
    for p,h in lock['hashes'].items():
        if sha(p)!=h:raise ValueError('Changed reference fit')
    return json.loads(CONFIG.read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','train'])
    globals()[parser.parse_args().stage]()
