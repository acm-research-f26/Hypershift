"""Freeze and fit the declared 280-model matched NASDAQ-100 intraday study."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from ndx_alpaca_models import inputs, scale, fit
from pit_research_engine import load_data, fold, historical_groups

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs/nasdaq100_alpaca_20261019'
CONFIG = ROOT/'config/nasdaq100_alpaca_20261019.json'
ASSEMBLED = ROOT/'data/nasdaq_monthly_pit/assembled'
FEATURE_ROOT = ROOT/'data/nasdaq_monthly_pit/alpaca_features_20261019'
SOURCES = ['ndx_alpaca_models.py', 'run_ndx_alpaca_training.py', 'pit_research_engine.py',
           'research_models.py', 'research_engine.py', 'think_model.py', 'robust_models.py',
           'robust_engine.py', 'pseudohuber_think_mixer.py', 'run_robust_research.py', 'training_device.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def freeze():
    manifest = json.loads((ASSEMBLED/'manifest.json').read_text())
    lock = {str(ASSEMBLED/name): expected for name, expected in manifest['hashes'].items()}
    lock.update({str(ROOT/name): sha(ROOT/name) for name in SOURCES})
    lock[str(CONFIG)] = sha(CONFIG)
    lock[str(ASSEMBLED/'manifest.json')] = sha(ASSEMBLED/'manifest.json')
    for path, expected in lock.items():
        if sha(path) != expected:
            raise ValueError('Changed input bytes or dependency')
    RUN.mkdir(exist_ok=True)
    path = RUN/'BASELINE_DATA_CODE_LOCK.json'
    if path.exists():
        if json.loads(path.read_text()) != lock:
            raise ValueError('Frozen study inputs/code changed')
    else:
        write(path, lock)
        snapshot = RUN/'source_snapshot'
        snapshot.mkdir()
        for name in SOURCES:
            (snapshot/name).write_bytes((ROOT/name).read_bytes())
        (RUN/'PROTOCOL.json').write_bytes(CONFIG.read_bytes())
    return json.loads(CONFIG.read_text())


def verify():
    for path, expected in json.loads((RUN/'BASELINE_DATA_CODE_LOCK.json').read_text()).items():
        if sha(path) != expected:
            raise ValueError('Frozen study inputs/code changed')


def feature_data():
    review = json.loads((FEATURE_ROOT/'REVIEW.json').read_text())
    if review['status'] != 'passed_research_input_gate':
        raise ValueError('Intraday data review has not passed')
    path = FEATURE_ROOT/'daily_features.npz'
    if sha(path) != review['feature_sha256']:
        raise ValueError('Intraday features changed')
    with np.load(path) as saved:
        feature = saved['features'].copy()
        dates = saved['dates'].copy()
        tickers = saved['tickers'].copy()
    return feature, dates, tickers


def fit_group(view, year, kind):
    verify()
    plan = json.loads((RUN/'PROTOCOL.json').read_text())
    data = load_data(ASSEMBLED)
    feature = None
    if view == 'intraday_augmented':
        feature, dates, tickers = feature_data()
        np.testing.assert_array_equal(dates, data['panel'].dates.strftime('%Y-%m-%d'))
        np.testing.assert_array_equal(tickers, data['panel'].tickers)
    raw = inputs(data, view, feature)
    train, val, test = fold(data, year)
    x, mean, std = scale(raw, data['mask'], train)
    groups = historical_groups(data, train)
    folder = RUN/view/str(year)
    folder.mkdir(parents=True, exist_ok=True)
    preprocessing = folder/'preprocessing.npz'
    if not preprocessing.exists():
        # Main creates this once per fold before launching concurrent model groups.
        raise ValueError('Preprocessing was not frozen')
    with np.load(preprocessing) as saved:
        for field, value in [('train', train), ('val', val), ('test', test), ('mean', mean), ('std', std)]:
            np.testing.assert_array_equal(saved[field], value)
    records = []
    for seed in plan['seeds']:
        prediction = folder/f'{kind}_{seed}_predictions.npz'
        record_path = folder/f'{kind}_{seed}_record.json'
        if prediction.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if sha(prediction) != record['predictions_sha256'] or sha(folder/f'{kind}_{seed}.pt') != record['checkpoint_sha256']:
                raise ValueError('Cached fit changed')
        else:
            vp, tp, record = fit(data, x, train, val, test, groups, kind, seed, plan, folder)
            np.savez_compressed(prediction, validation=vp, test=tp)
            record.update(view=view, year=year, checkpoint_sha256=sha(folder/f'{kind}_{seed}.pt'),
                          predictions_sha256=sha(prediction))
            write(record_path, record)
        records.append(record)
        print(f'Fit {view} {year} {kind} seed {seed}: epoch {record["chosen_epoch"]}', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', choices=['daily_control', 'intraday_augmented', 'both'], default='both')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    plan = freeze()
    views = plan['views'] if args.view == 'both' else [args.view]
    data = load_data(ASSEMBLED)
    for view in views:
        feature = feature_data()[0] if view == 'intraday_augmented' else None
        if feature is not None:
            feature_lock = {str(p): sha(p) for p in [FEATURE_ROOT/'daily_features.npz', FEATURE_ROOT/'REVIEW.json']}
            path = RUN/'INTRADAY_DATA_LOCK.json'
            if path.exists() and json.loads(path.read_text()) != feature_lock:
                raise ValueError('Intraday inputs changed')
            write(path, feature_lock)
        raw = inputs(data, view, feature)
        for year in plan['test_years']:
            train, val, test = fold(data, year)
            _, mean, std = scale(raw, data['mask'], train)
            folder = RUN/view/str(year)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/'preprocessing.npz'
            if path.exists():
                with np.load(path) as old:
                    for field, value in [('mean', mean), ('std', std), ('train', train), ('val', val), ('test', test)]:
                        np.testing.assert_array_equal(old[field], value)
            else:
                np.savez_compressed(path, train=train, val=val, test=test, mean=mean, std=std)
    jobs = [(view, year, kind) for view in views for year in plan['test_years'] for kind in plan['models']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit_group, *job) for job in jobs]
        completed = 0
        for future in as_completed(futures):
            completed += len(future.result())
            print(f'{completed}/{len(jobs)*len(plan["seeds"])} fits available in this stage; no financial test metrics computed', flush=True)
    all_records = [p for view in plan['views'] for p in (RUN/view).glob('*/*_record.json')]
    if len(all_records) == plan['planned_neural_fits']:
        lock = {str(p.relative_to(RUN)): sha(p) for view in plan['views'] for p in (RUN/view).rglob('*')
                if p.suffix in ('.pt', '.npz', '.json')}
        write(RUN/'FITS_COMPLETE_LOCK.json', dict(neural_fits=len(all_records), hashes=lock))
        print('All 280 fits locked. Financial evaluation may now begin.', flush=True)


if __name__ == '__main__':
    main()
