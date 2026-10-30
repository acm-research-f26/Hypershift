"""Execution-only two-process schedule for the unchanged frozen rank study.

Every model/objective/selection function remains the frozen implementation.
Only progress-log destinations are separated to prevent worker logging races.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

import run_ndx_rank_retraining as trainer
from run_ndx_retention_replication import sha, write
from training_device import configure_device
import run_ndx_drawdown_research as parent

RUN = trainer.ROOT/'runs'/trainer.DEFAULT
CODE = trainer.ROOT/'research/training/resume_ndx_rank_retraining.py'


def amendment():
    plan = trainer.verify(RUN)
    if (RUN/'EXECUTION_AMENDMENT.json').exists(): raise FileExistsError('Amendment already exists')
    prefix = {}; completed = 0
    for folder in (RUN/'fits').glob('*/*/*'):
        if not folder.is_dir(): continue
        record = folder/'RECORD.json'
        if record.exists():
            completed += 1
            for name, expected in json.loads(record.read_text())['hashes'].items():
                assert sha(folder/name) == expected
            files = [p for p in folder.iterdir() if p.is_file()]
        else:
            # Atomic .pt snapshots are complete; an interrupted .tmp is ignored.
            checkpoints = sorted(folder.glob('epoch_*.pt'))
            latest = int(checkpoints[-1].stem.split('_')[-1]) if checkpoints else -1
            files = checkpoints+[p for p in folder.glob('validation_*.npz')
                                 if int(p.stem.split('_')[-1]) <= latest]
        for path in files: prefix[path.relative_to(RUN).as_posix()] = sha(path)
    write(RUN/'EXECUTION_AMENDMENT.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        type='Scheduling/progress logging only; frozen model math and budget unchanged',
        reason='Single-process GPU utilization is low; run independent family/year jobs in2isolatedprocesses',
        controller_sha256=sha(CODE), protocol_sha256=sha(RUN/'PROTOCOL.json'),
        completed_fits_preserved=completed, preserved_prefix_sha256=prefix,
        workers=2, test_accounts_scored=0,
        resume_gate='Replay epoch2 of existing THINK2022seed7 from its epoch1 optimizer/model/RNG state; require model/prediction maxabs errors<=1e-7 before parallel jobs',
        logging='Redirect each worker root PROGRESS.json to worker_progress/<family>_<year>.json; all fit histories and scientific outputs unchanged',
        budget=dict(fits=plan['fits'], epochs=plan['epochs'], accounts=plan['accounts'])))
    print('Scheduling amendment preserves', completed, 'completed fits and every complete prefix checkpoint', flush=True)


def verify_amendment():
    data = json.loads((RUN/'EXECUTION_AMENDMENT.json').read_text())
    assert sha(CODE) == data['controller_sha256']
    assert sha(RUN/'PROTOCOL.json') == data['protocol_sha256']
    trainer.verify(RUN)
    for name, expected in data['preserved_prefix_sha256'].items(): assert sha(RUN/name) == expected, name
    return data


def resume_check():
    verify_amendment(); device = configure_device('cuda')
    temporary = RUN/'resume_check'; folder = temporary/'fits/think/2022/7'
    folder.mkdir(parents=True, exist_ok=False)
    source = RUN/'fits/think/2022/7'
    for name in ('epoch_00.pt','epoch_01.pt'): shutil.copyfile(source/name, folder/name)
    prepared = trainer.prepare(2022, device)
    trainer.fit(temporary, prepared, 'think', 2022, 7, dict(epochs=2), device)
    first = torch.load(source/'epoch_02.pt', map_location='cpu', weights_only=False)['state_dict']
    second = torch.load(folder/'epoch_02.pt', map_location='cpu', weights_only=False)['state_dict']
    error = max(float((first[key].double()-second[key].double()).abs().max()) for key in first)
    import numpy as np
    with np.load(source/'validation_02.npz') as a, np.load(folder/'validation_02.npz') as b:
        prediction_error = float(np.max(np.abs(a['validation']-b['validation'])))
    assert error <= 1e-7 and prediction_error <= 1e-7
    write(RUN/'RESUME_CHECK.json', dict(status='passed', model_max_abs_error=error,
        validation_max_abs_error=prediction_error, controller_sha256=sha(CODE),
        scope='Actual resumed training epoch against saved state; no financial account scoring'))
    print('Resume check passed:', error, prediction_error, flush=True)


def job(family, year):
    plan = trainer.verify(RUN); device = configure_device('cuda')
    prepared = trainer.prepare(year, device)
    original = trainer.atomic_json
    progress = RUN/'worker_progress'; progress.mkdir(exist_ok=True)
    def worker_log(path, value):
        if path == RUN/'PROGRESS.json': path = progress/f'{family}_{year}.json'
        original(path, value)
    trainer.atomic_json = worker_log
    for seed in plan['groups']['fifteen_seed']:
        trainer.fit(RUN, prepared, family, year, seed, plan, device)
    return family, year


def parallel():
    verify_amendment()
    gate = json.loads((RUN/'RESUME_CHECK.json').read_text())
    assert gate['status'] == 'passed' and gate['controller_sha256'] == sha(CODE)
    plan = trainer.verify(RUN); data = parent.load_data(); jobs = []
    for year in plan['years']:
        count = len(parent.fold(data, year)[0])
        for family in plan['families']:
            remaining = 0.
            for seed in plan['groups']['fifteen_seed']:
                folder = RUN/'fits'/family/str(year)/str(seed)
                if (folder/'RECORD.json').exists(): continue
                checkpoint = sorted(folder.glob('epoch_*.pt'))
                epoch = int(checkpoint[-1].stem.split('_')[-1]) if checkpoint else 0
                remaining += (plan['epochs']-epoch)/plan['epochs']
            if remaining:
                jobs.append((remaining*count*(.07/8 if family == 'think' else .023), family, year))
    jobs.sort(reverse=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(job, family, year) for _, family, year in jobs]
        for future in as_completed(futures):
            family, year = future.result()
            count = len(list((RUN/'fits').rglob('RECORD.json')))
            trainer.atomic_json(RUN/'PROGRESS.json', dict(stage='parallel_training',
                last_completed_job=[family,year], completed_fits=count, test_accounts_scored=0))
            print('Completed parallel job', family, year, count, '/120', flush=True)
    verify_amendment()
    # Use the original unchanged inventory and sealing implementation. Every fit
    # is now a verified cache hit; this cannot retrain or alter prior records.
    trainer.train(RUN)
    verify_amendment()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['amendment','resume-check','train'],required=True)
    args=parser.parse_args()
    if args.stage=='amendment': amendment()
    elif args.stage=='resume-check': resume_check()
    else: parallel()


if __name__=='__main__': main()
