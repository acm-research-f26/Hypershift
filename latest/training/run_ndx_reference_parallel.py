"""Execution-only resume: parallel independent missing seeds, frozen fit math.

Stop the old driver and its workers before invoking this. Completed records are
verified and preserved. The original frozen runner performs final inventory and
locking after the missing fits finish. No model/data/selection rule changes.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from functools import lru_cache
import json
from pathlib import Path
import shutil
import subprocess
import sys
from ndx_reference_training import RUN,CONFIG,KINDS,prepare,train_one
from ndx_reference_models import ROOT
from run_ndx_reference_networks import verify
from run_ndx_alpaca_training import sha,write


@lru_cache(maxsize=None)
def prepared(year):return prepare(year)


def job(spec):
    verify();kind,year,seed=spec;plan=json.loads(CONFIG.read_text())
    result=train_one(prepared(year),kind,year,seed,plan)
    print('Completed unchanged reference fit',kind,year,seed,result['chosen_epochs'],flush=True)
    return spec


def main():
    verify();plan=json.loads(CONFIG.read_text());missing=[];existing=0
    for kind in KINDS:
        for year in plan['years']:
            for seed in plan['seeds']:
                record=RUN/kind/str(year)/f'{seed}_record.json'
                if not record.exists():missing.append((kind,year,seed));continue
                for name,h in json.loads(record.read_text())['hashes'].items():
                    if sha(record.parent/name)!=h:raise ValueError('Existing completed fit changed')
                existing+=1
    target=RUN/'EXECUTION_AMENDMENT.json';source=Path(__file__).resolve()
    if target.exists():
        if json.loads(target.read_text())['execution_source_sha256']!=sha(source):raise ValueError('Resume source changed')
    else:
        write(target,dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),type='compute_scheduling_only',
            reason='Final annual PRISM job serializes remaining seeds and leaves GPU capacity unused; parallelize independent seeds without changing frozen model math.',
            preserved_completed_fits=existing,missing_fits=missing,max_workers=4,
            interrupted_incomplete_fit_restarted=True,financial_results_inspected=False,
            original_training_source_unchanged=True,execution_source_sha256=sha(source),
            frozen_protocol_sha256=sha(CONFIG),fit_budget=120))
        snapshot=RUN/'resume_source_snapshot';snapshot.mkdir(exist_ok=True);shutil.copyfile(source,snapshot/source.name)
    done=existing
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(job,spec) for spec in missing]):
            future.result();done+=1
            write(RUN/'PROGRESS.json',dict(stage='training_parallel_resume',completed_seed_fits=done,planned_seed_fits=120))
    assert done==120
    # Reuse the original, hash-locked driver for every final record check and
    # the authoritative FITS_COMPLETE_LOCK. Every fit is now a verified cache hit.
    subprocess.run([sys.executable,str(ROOT/'run_ndx_reference_networks.py'),'train'],cwd=ROOT,check=True)


if __name__=='__main__':main()
