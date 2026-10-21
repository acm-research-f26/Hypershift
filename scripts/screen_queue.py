"""Claim-based work queue over (fold, arm) units so several screening workers can share one grid.

Each unit is claimed atomically by creating <output>/.claims/<year>_<arm>; a claimed unit is run as
`research_screen.py --years <year> --arms <arm> <passthrough...>` in a subprocess (resumable:
complete seeds are reused, interrupted fits reload last.pt). After a crash, delete the stale
claim directories of units that are not finished and relaunch.

  python scripts/screen_queue.py --output DIR --years 2022 2023 --arms static two ... -- --panel P --select rankic ...
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
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
crashes = 0
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
        print(args.worker, 'END', unit, code, flush=True)
        if not 0 <= code <= 255:
            # Windows NTSTATUS exit (e.g. 0xC0000142 DLL init failed during logoff/sleep, or a
            # terminated process): infrastructure, not the experiment. Release the claim so the
            # unit is retried, and stop this worker instead of burning through the queue.
            shutil.rmtree(out/'.claims'/unit, ignore_errors=True)
            crashes += 1
            if crashes >= 3:
                print(args.worker, 'STOPPING after repeated process crashes', flush=True)
                sys.exit(2)
            time.sleep(60)
            continue
        crashes = 0
        (out/'.claims'/unit/('done' if code == 0 else f'exit_{code}')).write_text('')
print(args.worker, 'QUEUE EMPTY', flush=True)
