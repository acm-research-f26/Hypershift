"""Execution-only retry of transient Windows atomic progress-file replacement.

Model math, data, seeds, checkpoint selection and fixed account grid are unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import run_ndx_portfolio_robustness_v2 as study
import run_ndx_rank_retraining as trainer
from run_ndx_retention_replication import sha,write

ROOT=study.ROOT
RUN=study.RUN
CODE=ROOT/'research/training/resume_ndx_portfolio_robustness.py'


def freeze():
    study.verify(); hashes={}
    for folder in (RUN/'fits').glob('*/*/*'):
        record=folder/'RECORD.json'
        if record.exists():
            for name,expected in json.loads(record.read_text())['hashes'].items():assert sha(folder/name)==expected
            paths=[p for p in folder.iterdir() if p.is_file()]
        else:
            checkpoints=sorted(folder.glob('epoch_*.pt'))
            epoch=int(checkpoints[-1].stem.split('_')[-1]) if checkpoints else -1
            paths=checkpoints+[p for p in folder.glob('validation_*.npz') if int(p.stem.split('_')[-1])<=epoch]
        for path in paths:hashes[path.relative_to(RUN).as_posix()]=sha(path)
    check=ROOT/'outputs/seed_robustness_20261030/actual_resume_check/RESULT.json'
    gate=json.loads(check.read_text());assert gate['status']=='passed'
    assert gate['model_max_abs_error']==gate['validation_max_abs_error']==0
    write(RUN/'EXECUTION_AMENDMENT.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        controller_sha256=sha(CODE),protocol_sha256=sha(RUN/'PROTOCOL.json'),
        preserved_prefix_sha256=hashes,completed_fits=len(list((RUN/'fits').rglob('RECORD.json'))),
        resume_check_sha256=sha(check),scope='Execution and logging only; frozen scientific code unchanged',
        cause='Transient Windows progress-file replacement permission error; checkpoint8/history8 persisted before progress write failed',
        change='Retry atomic JSON replace up to50times with0.1second delay for PermissionError. Same bytes and destinations. Cache completed fits; resume incomplete fits from saved optimizer/model/CPU+CUDA RNG.',
        no_seed_exclusion=True))
    shutil.copyfile(check,RUN/'RESUME_CHECK.json')
    print('Frozen execution retry; prefix preserved',len(hashes),flush=True)


def verify():
    study.verify(); amendment=json.loads((RUN/'EXECUTION_AMENDMENT.json').read_text())
    assert sha(CODE)==amendment['controller_sha256']
    assert sha(RUN/'PROTOCOL.json')==amendment['protocol_sha256']
    for name,expected in amendment['preserved_prefix_sha256'].items():assert sha(RUN/name)==expected,name
    return amendment


def retry_atomic(path,value):
    temporary=path.with_suffix('.tmp')
    write(temporary,value)
    for attempt in range(50):
        try:
            temporary.replace(path);return
        except PermissionError:
            if attempt==49:raise
            time.sleep(.1)


def job(year):
    trainer.atomic_json=retry_atomic
    return study.train_year(year)


def train():
    verify();plan=study.verify()
    years=[year for year in plan['years'] if any(not (RUN/'fits/think'/str(year)/str(seed)/'RECORD.json').exists()
            for seed in plan['seed_groups']['fifteen_seed'])]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(job,year) for year in years]
        for future in as_completed(jobs):print('Resumed year complete',future.result(),flush=True)
    records=list((RUN/'fits').rglob('RECORD.json'));assert len(records)==60;hashes={}
    for record in records:
        for name,expected in json.loads(record.read_text())['hashes'].items():
            path=record.parent/name;assert sha(path)==expected;hashes[path.relative_to(RUN).as_posix()]=expected
        hashes[record.relative_to(RUN).as_posix()]=sha(record)
    verify();write(RUN/'FITS_COMPLETE.json',dict(fits=60,hashes=hashes,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),shared_accounts_absent=not (RUN/'shared').exists()))
    print('All60sharedfits locked; all prefix hashes unchanged',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['freeze','train'],required=True)
    args=p.parse_args();freeze() if args.stage=='freeze' else train()


if __name__=='__main__':main()
