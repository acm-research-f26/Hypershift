"""Kaggle CPU kernel for the v3 broad screen: runs a disjoint arm subset with 2 claim-queue workers.

Built per kernel by kaggle/screen_v3/launch.sh (ARMS/TAG are substituted). Verifies that the shipped
source hashes equal the local v3 run configs before training, so configs stay identical.
"""
import json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

ARMS = __ARMS__
TAG = '__TAG__'
YEARS = ['2022', '2023']
WORKERS = 2
OUT_NAME = 'results/SCREEN_SP500_20261019_v3'

inp = Path('/kaggle/input')
work = Path('/kaggle/working')
repo = work/'repo'
repo.mkdir(exist_ok=True)
print('input files:', [str(p.relative_to(inp)) for p in inp.rglob('*') if p.is_file()][:20], flush=True)
src = next(inp.rglob('research_screen.py'), None)
if src is not None:                         # Kaggle auto-extracted repo.zip
    root = src.parents[1]
    shutil.copytree(root/'src', repo/'src'); shutil.copytree(root/'scripts', repo/'scripts')
else:
    zipfile.ZipFile(next(inp.rglob('repo.zip'))).extractall(repo)
panel = next(p for p in inp.rglob('screen_sp500_2015_2026.npz*') if p.is_file())
expected = json.loads(next(inp.rglob('expected_source.json')).read_text())
os.chdir(repo)
sys.path.insert(0, str(repo/'src'))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from hypershift.screening.runner import source_manifest
got = {k.replace('\\', '/'): v for k, v in source_manifest().items()}
exp = {k.replace('\\', '/'): v for k, v in expected.items()}
assert got == exp, ('source mismatch', {k for k in set(got) | set(exp) if got.get(k) != exp.get(k)})
import torch
print('source verified; torch', torch.__version__, 'cpus', os.cpu_count(), 'panel', panel, flush=True)

cmd = lambda w: [sys.executable, 'scripts/screen_queue.py', '--worker', f'{TAG}w{w}', '--output', OUT_NAME,
                 '--years', *YEARS, '--arms', *ARMS, '--', '--panel', str(panel), '--threads', '2',
                 '--train-stride', '1', '--validation-stride', '1', '--select', 'rankic',
                 '--frozen-reference', '2022', '--protocol', 'broad-screen-v3-sp500']
procs = [subprocess.Popen(cmd(w)) for w in range(WORKERS)]
start = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(60)
    n = sum(1 for _ in (repo/OUT_NAME).rglob('metrics.json'))
    print(f'{(time.time()-start)/60:.0f} min: {n} complete', flush=True)
print('exit codes', [p.returncode for p in procs], flush=True)

# Ship complete artifacts only (no optimizer checkpoints), plus logs and the environment.
(repo/OUT_NAME/f'kaggle_env_{TAG}.json').write_text(json.dumps({'torch': torch.__version__, 'python': sys.version,
                                                               'cpus': os.cpu_count(), 'arms': ARMS}))
with zipfile.ZipFile(work/f'results_{TAG}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in (repo/'results').rglob('*'):
        if f.is_file() and f.name not in ('last.pt', 'last.tmp'):
            z.write(f, f.relative_to(repo))
shutil.rmtree(repo)
print('zipped', (work/f'results_{TAG}.zip').stat().st_size, flush=True)
