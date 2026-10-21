"""Merge remote (Kaggle/Colab) screen result zips into a local screen root. Never overwrites.

Copies a seed folder only if it has metrics.json, is not a smoke run, and its config's source and
dataset hashes equal a reference local config. Marks fully merged (year, arm) claims done.

  python scripts/merge_screen_results.py --root results/SCREEN_SP500_20261019_v3 results_k1.zip ... [--dry-run]
"""
import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('zips', nargs='+')
p.add_argument('--dry-run', action='store_true')
p.add_argument('--ref', help='reference config.json (default: a static config under --root)')
args = p.parse_args()
root = Path(args.root)
root.mkdir(parents=True, exist_ok=True)
ref = json.loads(Path(args.ref).read_text() if args.ref else next(root.glob('*/static/seed_*/config.json')).read_text())
norm = lambda d: {k.replace('\\', '/'): v for k, v in d.items()}
counts = {'merged': 0, 'exists': 0, 'incomplete': 0, 'mismatch': 0}
touched = set()
for zp in args.zips:
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zp).extractall(tmp)
        remote = next(Path(tmp).rglob(root.name), None)
        if remote is None:
            print('no', root.name, 'in', zp); continue
        for f in sorted(remote.glob('*')):
            if f.is_file() and not (root/f.name).exists() and not args.dry_run:
                shutil.copy2(f, root/f.name)          # kaggle_env_*.json, data audits
        for seed in sorted(remote.glob('*/*/seed_*')):
            rel = seed.relative_to(remote)
            dest = root/rel
            if not (seed/'metrics.json').exists():
                counts['incomplete'] += 1; continue
            cfg = json.loads((seed/'config.json').read_text())
            if (cfg.get('smoke') or norm(cfg['source']) != norm(ref['source'])
                    or cfg['dataset_sha256'] != ref['dataset_sha256'] or cfg['protocol'] != ref['protocol']):
                counts['mismatch'] += 1; print('MISMATCH', zp, rel); continue
            if (dest/'metrics.json').exists():
                counts['exists'] += 1; continue
            if dest.exists() and not args.dry_run:
                shutil.rmtree(dest)                    # stale partial local folder (no metrics.json)
            if not args.dry_run:
                shutil.copytree(seed, dest)
            counts['merged'] += 1
            touched.add((rel.parts[0], rel.parts[1]))
for year, arm in sorted(touched):
    claim = root/'.claims'/f'{year}_{arm}'
    if claim.exists() and not args.dry_run:
        (claim/'done').write_text('merged from remote')
print(counts, 'units touched:', len(touched))
