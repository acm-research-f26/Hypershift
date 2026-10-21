"""Kaggle GPU kernel for Study J: zero-shot Kronos scores via scripts/j_kronos.py, one shard per visible GPU.

Built per kernel (MODE/ARGS substituted). Code, weights, panel and aligned OHLCV come from the private dataset
hypershift-kronos-j; no internet. MODE 'timing' prints s/day on pre-2024 days only; MODE 'run' forecasts,
assembles and zips the run folders.
"""
import json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

MODE = '__MODE__'
JOBS = __JOBS__                 # list of extra j_kronos.py argument lists (years, seeds, arm, model, lookback, ...)
TAG = '__TAG__'
OUT_NAME = 'results/SCREEN_SP500_20261021_j'

inp, work = Path('/kaggle/input'), Path('/kaggle/working')
repo = work/'repo'
repo.mkdir(exist_ok=True)
src = next(inp.rglob('j_kronos.py'), None)
if src is not None:                         # Kaggle auto-extracted repo.zip
    shutil.copytree(src.parents[1]/'src', repo/'src'); shutil.copytree(src.parents[1]/'scripts', repo/'scripts')
else:
    zipfile.ZipFile(next(inp.rglob('repo.zip'))).extractall(repo)
print('input files:', [str(p.relative_to(inp)) for p in inp.rglob('*') if p.is_file()][:30], flush=True)
if next(inp.rglob('Kronos-small'), None) is None:   # directory zips not auto-extracted
    for z in inp.rglob('*.zip'):
        if z.name != 'repo.zip':
            zipfile.ZipFile(z).extractall(work/'ext'/z.stem)
look = [inp, work/'ext']
find = lambda pat, ok=lambda p: True: next(p for d in look if d.exists() for p in d.rglob(pat) if ok(p))
root = find('ohlcv_sp500_2015_2026.npz*', lambda p: p.is_file()).parent
kronos = find('kronos.py', lambda p: p.parent.name == 'model').parents[1]
weights = find('Kronos-small', lambda p: p.is_dir()).parent
os.chdir(repo)
import torch
gpus = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(gpus)]
print('torch', torch.__version__, 'gpus', names, flush=True)
env = {'torch': torch.__version__, 'gpus': names, 'python': sys.version, 'tag': TAG}
def base_args(args):
    model = args[args.index('--model')+1] if '--model' in args else 'Kronos-small'
    tok = args[args.index('--tokenizer')+1] if '--tokenizer' in args else 'Kronos-Tokenizer-base'
    rest = [a for i, a in enumerate(args) if a not in ('--model', '--tokenizer') and (i == 0 or args[i-1] not in ('--model', '--tokenizer'))]
    return rest, ['--panel', str(next(root.glob('screen_sp500_2015_2026.npz*'))), '--ohlcv', str(next(root.glob('ohlcv_sp500_2015_2026.npz*'))),
                  '--kronos-dir', str(kronos), '--model', str(weights/model), '--tokenizer', str(weights/tok), '--output', OUT_NAME] + rest


py = [sys.executable, 'scripts/j_kronos.py']
if MODE == 'timing':
    rest, base = base_args(JOBS[0] if JOBS else [])
    for chunk in ('77', '154'):
        subprocess.run(py + ['timing'] + base + ['--chunk', chunk, '--dates', '2021-03-01', '2021-09-01'], check=True)
    sys.exit(0)

for job in JOBS:
    rest, base = base_args(job)
    start = time.time()
    procs = [subprocess.Popen(py + ['run'] + base + ['--shard', str(i), str(gpus), '--device', f'cuda:{i}']) for i in range(gpus)]
    codes = [p.wait() for p in procs]
    print('job', job, 'exit codes', codes, 'minutes', round((time.time()-start)/60, 1), flush=True)
    assert all(c == 0 for c in codes)
    arm = rest[rest.index('--arm')+1] if '--arm' in rest else 'kronos'
    for folder in Path(OUT_NAME).glob(f'*/{arm}/seed_*'):
        if not (folder/'env.json').exists():
            (folder/'env.json').write_text(json.dumps(env))
    subprocess.run(py + ['assemble'] + base + ['--shards', str(gpus)], check=True)
with zipfile.ZipFile(work/f'results_{TAG}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in Path('results').rglob('*'):
        if f.is_file():
            z.write(f, f.relative_to(repo))
os.chdir(work); shutil.rmtree(repo)
print('zipped', (work/f'results_{TAG}.zip').stat().st_size, flush=True)
