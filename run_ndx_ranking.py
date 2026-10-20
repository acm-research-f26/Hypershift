"""Run the finite, source-locked NASDAQ-100 target/ranking experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ndx_alpaca_models import inputs, scale
from ndx_ranking_objectives import target_data, fit
from pit_research_engine import load_data, fold, historical_groups
from run_ndx_alpaca_training import ROOT, RUN as PARENT, FEATURE_ROOT, feature_data, sha, write

RUN = ROOT/'runs/ndx_ranking_20261020'
CONFIG = ROOT/'config/ndx_ranking_20261020.json'


def state(stage, **details):
    write(RUN/'PROGRESS.json',dict(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**details))
    print(stage,details,flush=True)


def verify():
    for path,expected in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        if sha(path)!=expected:
            raise ValueError('Locked ranking-experiment input or source changed: '+str(path))


def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    lock = json.loads((PARENT/'BASELINE_DATA_CODE_LOCK.json').read_text())
    lock.update(json.loads((PARENT/'INTRADAY_DATA_LOCK.json').read_text()))
    sources = [CONFIG,ROOT/'run_ndx_ranking.py',ROOT/'ndx_ranking_objectives.py',ROOT/'test_ndx_ranking.py',
               ROOT/'prepare_ndx_alpaca.py',ROOT/'config/alpaca_symbol_identity_20261019.json']
    lock.update({str(p):sha(p) for p in sources})
    tests = json.loads((RUN/'UNIT_TESTS.json').read_text())
    if tests['status']!='passed':
        raise ValueError('Tests must pass before the new experiment is frozen')
    for name,value in tests['source_hashes'].items():
        if sha(ROOT/name)!=value:
            raise ValueError('Test source changed')
        lock[str(ROOT/name)] = value
    path = RUN/'INPUT_CODE_LOCK.json'
    if path.exists() and json.loads(path.read_text())!=lock:
        raise ValueError('Refusing to modify a frozen training experiment')
    write(path,lock)
    verify()
    feature_data()
    if not (RUN/'PROTOCOL.json').exists():
        (RUN/'PROTOCOL.json').write_bytes(CONFIG.read_bytes())
        snapshot = RUN/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in lock:
            p = Path(name)
            if p.suffix=='.py':
                (snapshot/p.name).write_bytes(p.read_bytes())
        write(RUN/'FREEZE.json',dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
              protocol_sha256=sha(CONFIG),all_new_fits_absent=not any(RUN.glob('*/*/*_record.json')),
              prior_history_is_exploratory=True,future_confirmation_not_yet_available=True))
    return json.loads(CONFIG.read_text())


def cached_targets(data):
    target = target_data(data)
    path = RUN/'targets.npz'
    if path.exists():
        with np.load(path) as saved:
            for name,value in target.items():
                np.testing.assert_array_equal(saved[name],value)
    else:
        np.savez_compressed(path,**target)
    return target


def fit_group(view,year,kind):
    verify()
    plan = json.loads((RUN/'PROTOCOL.json').read_text())
    data = load_data()
    feature = feature_data()[0] if view=='intraday_augmented' else None
    x,mean,std = scale(inputs(data,view,feature),data['mask'],fold(data,year)[0])
    train,val,test = fold(data,year)
    target = target_data(data)
    groups = historical_groups(data,train)
    folder = RUN/view/str(year)
    with np.load(folder/'preprocessing.npz') as saved:
        for name,value in dict(mean=mean,std=std,train=train,val=val,test=test).items():
            np.testing.assert_array_equal(saved[name],value)
    for seed in plan['seeds']:
        record_path = folder/f'{kind}_{seed}_record.json'
        prediction = folder/f'{kind}_{seed}_predictions.npz'
        checkpoint = folder/f'{kind}_{seed}.pt'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if sha(prediction)!=record['predictions_sha256'] or sha(checkpoint)!=record['checkpoint_sha256']:
                raise ValueError('Cached model integrity failure')
            if sha(folder/f'{kind}_{seed}_history.json')!=record['history_sha256']:
                raise ValueError('Cached history integrity failure')
        else:
            vp,tp,record = fit(data,x,target,train,val,test,groups,kind,seed,plan,folder)
            np.savez_compressed(prediction,validation=vp,test=tp)
            record.update(view=view,year=year,predictions_sha256=sha(prediction),checkpoint_sha256=sha(checkpoint),
                          history_sha256=sha(folder/f'{kind}_{seed}_history.json'))
            write(record_path,record)
        print(f'Fit {view} {year} {kind} seed {seed}: epoch {record["chosen_epoch"]}',flush=True)
    return len(plan['seeds'])


def train(workers=2):
    plan = freeze()
    data = load_data()
    cached_targets(data)
    for view in plan['views']:
        feature = feature_data()[0] if view=='intraday_augmented' else None
        raw = inputs(data,view,feature)
        for year in plan['test_years']:
            tr,va,te = fold(data,year)
            _,mean,std = scale(raw,data['mask'],tr)
            folder = RUN/view/str(year)
            folder.mkdir(parents=True,exist_ok=True)
            path = folder/'preprocessing.npz'
            values = dict(mean=mean,std=std,train=tr,val=va,test=te)
            if path.exists():
                with np.load(path) as saved:
                    for name,value in values.items():
                        np.testing.assert_array_equal(saved[name],value)
            else:
                np.savez_compressed(path,**values)
    jobs = [(v,y,k) for v in plan['views'] for y in plan['test_years'] for k in plan['objectives']]
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fit_group,*job) for job in jobs]):
            completed+=future.result()
            state('training',fits=completed,planned=plan['planned_fits'],test_metrics_inspected=False)
    records = list(RUN.glob('*/*/*_record.json'))
    if len(records)!=plan['planned_fits']:
        raise ValueError('Incomplete declared neural experiment')
    lock = {str(p.relative_to(RUN)):sha(p) for view in plan['views'] for p in (RUN/view).rglob('*')
            if p.suffix in ['.npz','.pt','.json']}
    lock['targets.npz'] = sha(RUN/'targets.npz')
    write(RUN/'FITS_COMPLETE_LOCK.json',dict(neural_fits=len(records),hashes=lock))
    state('all_fits_locked',fits=len(records))


def complete():
    verify()
    lock = json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())
    if lock['neural_fits']!=120:
        raise ValueError('All 120 fits must precede test evaluation')
    for name,value in lock['hashes'].items():
        if sha(RUN/name)!=value:
            raise ValueError('Fit artifact changed: '+name)
    return json.loads((RUN/'PROTOCOL.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['train','evaluate','audit','publish','all'],default='all')
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    args = parser.parse_args()
    if args.stage in ['train','all'] and not (RUN/'FITS_COMPLETE_LOCK.json').exists():
        train(args.workers)
    stages = ['evaluate','audit','publish'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage=='train':
            continue
        if stage=='evaluate' and (RUN/'EVALUATED.json').exists():
            continue
        if stage=='audit' and (RUN/'AUDIT.json').exists():
            continue
        state(stage)
        subprocess.run([sys.executable,str(ROOT/'report_ndx_ranking.py'),stage],cwd=ROOT,check=True)
    if args.stage in ['all','publish']:
        state('completed_audited_exploratory_development',confirmation='future dates reserved; not evaluated')


if __name__=='__main__':
    main()
