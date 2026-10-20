"""Claim-based work queue over (fold, arm) units so several screening workers can share one grid.

Each unit is claimed atomically by creating <output>/.claims/<year>_<arm>; a claimed unit is run as
`research_screen.py --years <year> --arms <arm> <passthrough...>` in a subprocess (resumable:
complete seeds are reused, interrupted fits reload last.pt). After a crash, delete the stale
claim directories of units that are not finished and relaunch.

  python scripts/screen_queue.py --output DIR --years 2022 2023 --arms static two ... -- --panel P --select rankic ...
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--years', nargs='+', required=True)
p.add_argument('--arms', nargs='+', required=True)
p.add_argument('--worker', default='w')
p.add_argument('rest', nargs=argparse.REMAINDER)
args = p.parse_args()
rest = [a for a in args.rest if a != '--']
out = Path(args.output)
(out/'.claims').mkdir(parents=True, exist_ok=True)
(out/'logs').mkdir(parents=True, exist_ok=True)
for arm in args.arms:          # arm-major: expensive arms listed first finish first
    for year in args.years:
        unit = f'{year}_{arm}'
        try:
            os.mkdir(out/'.claims'/unit)
        except FileExistsError:
            continue
        (out/'.claims'/unit/'worker').write_text(args.worker)
        cmd = [sys.executable, 'scripts/research_screen.py', '--output', str(out), '--years', year,
               '--arms', arm] + rest
        print(args.worker, 'START', unit, flush=True)
        with open(out/'logs'/f'{unit}.log', 'a') as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        (out/'.claims'/unit/('done' if code == 0 else f'exit_{code}')).write_text('')
        print(args.worker, 'END', unit, code, flush=True)
print(args.worker, 'QUEUE EMPTY', flush=True)
