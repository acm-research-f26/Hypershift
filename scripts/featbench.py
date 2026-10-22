"""Study FB runner: walk-forward folds 2019-2023 (never 2024+), one run folder per (year, arm, seed).

  python scripts/featbench.py --years 2019 2020 2021 2022 2023 --arms lgbm:rich ridge:rich block:reversal16 --jobs 3
Arm grammar: model:features:target:horizon:repr (missing fields default to raw:5:cs_z); block arms 'block:<name>'.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, 'src')
ROOT = 'results/FEATBENCH_SP500_20261022'


def work(job):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from hypershift.featbench.models import run_one
    root, arm, year, seed, threads, max_days = job
    try:
        return run_one(root, arm, year, seed, threads, max_days)
    except Exception as e:  # keep the batch going; report the failure
        return f'FAILED {year} {arm} seed {seed}: {type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', required=True)
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[7])
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--max-days', type=int, default=None)
    a = ap.parse_args()
    if a.max_days and 'smoke' not in os.path.basename(os.path.normpath(a.root)):
        sys.exit('--max-days is for smoke runs only; the root folder name must contain "smoke"')
    if max(a.years) > 2023:
        sys.exit('Study FB never scores 2024+ (spent for Study J arms only)')
    threads = max(1, 16 // a.jobs)
    # group by year so each worker reuses its loaded fold
    jobs = [(a.root, arm, y, s, threads, a.max_days) for y in a.years for arm in a.arms for s in a.seeds]
    failed = 0
    if a.jobs == 1:
        for j in jobs:
            r = work(j); print(r, flush=True); failed += r.startswith('FAILED')
    else:
        with ProcessPoolExecutor(a.jobs) as ex:
            for f in as_completed([ex.submit(work, j) for j in jobs]):
                r = f.result(); print(r, flush=True); failed += r.startswith('FAILED')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
