"""Merge a Kaggle results zip into the local results/ folder, never overwriting anything.

  .venv/Scripts/python.exe kaggle/merge_results.py results_s1.zip [--dry-run] [--results results]

Only COMPLETE runs (folder containing metrics.json) are copied. A run is skipped (reported) when:
  conflict  - the local run folder already has metrics.json (the local copy is kept)
  partial   - the local run folder exists without metrics.json (a local job may be writing it; left alone)
Incomplete Kaggle runs (no metrics.json) are reported and ignored. Kaggle logs go to results/logs_kaggle/ (only if absent).
"""
import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("zip")
ap.add_argument("--results", default="results")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
local = Path(a.results)
if not local.is_dir():
    raise SystemExit(f"{local} not found: run from the repo root")

copied, conflicts, partial, incomplete = [], [], [], []
with tempfile.TemporaryDirectory(dir=local.parent) as tmp:
    with zipfile.ZipFile(a.zip) as zf:
        zf.extractall(tmp)
    src_root = Path(tmp) / "results"
    runs = sorted({m.parent for m in src_root.rglob("metrics.json")})
    for d in sorted({p.parent for p in src_root.rglob("config.json")} - set(runs)):
        incomplete.append(d.relative_to(src_root).as_posix())
    for d in runs:
        rel = d.relative_to(src_root)
        dst = local / rel
        if (dst / "metrics.json").exists():
            conflicts.append(rel.as_posix())
        elif dst.exists():
            partial.append(rel.as_posix())
        else:
            copied.append(rel.as_posix())
            if not a.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                part = dst.with_name(dst.name + ".kaggle_tmp")
                shutil.copytree(d, part)
                if dst.exists():                      # appeared meanwhile (local job): back off
                    shutil.rmtree(part)
                    copied.remove(rel.as_posix())
                    partial.append(rel.as_posix())
                else:
                    os.rename(part, dst)
    logs = src_root / "logs"
    if logs.is_dir() and not a.dry_run:
        for f in logs.glob("*"):
            t = local / "logs_kaggle" / f.name
            if not t.exists():
                t.parent.mkdir(exist_ok=True)
                shutil.copy2(f, t)

tag = "DRY-RUN " if a.dry_run else ""
print(f"{tag}copied {len(copied)} complete runs; conflicts (local kept) {len(conflicts)}; local partial (left alone) {len(partial)}; "
      f"kaggle incomplete (ignored) {len(incomplete)}")
for name, lst in (("copied", copied), ("CONFLICT", conflicts), ("LOCAL-PARTIAL", partial), ("KAGGLE-INCOMPLETE", incomplete)):
    for r in lst:
        print(f"  {name:17s} {r}")
