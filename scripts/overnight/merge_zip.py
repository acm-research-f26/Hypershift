"""Fallback/extra for driver.sh: merge a Kaggle results zip WITHOUT os.rename (the PermissionError workaround in
docs/phase1_5/F_learnability.md), and copy any analysis .md found in the zip into docs/phase1_5/.
  merge_zip.py ZIP [--no-merge] [--docs-dir docs/phase1_5]
Complete runs only (folder has metrics.json); an existing local run folder is never touched. Prints 'DOC <path>' per new md.
"""
import argparse, glob, os, sys, zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("zip")
ap.add_argument("--no-merge", action="store_true")
ap.add_argument("--docs-dir", default="docs/phase1_5")
a = ap.parse_args()
for p in glob.glob("results/**/*.kaggle_tmp", recursive=True):   # stale temp dirs from a failed rename
    import shutil
    shutil.rmtree(p, ignore_errors=True)
zf = zipfile.ZipFile(a.zip)
names = zf.namelist()
if not a.no_merge:
    runs = {n.rsplit("/", 1)[0] for n in names if n.startswith("results/") and n.endswith("/metrics.json")}
    copied = skipped = 0
    for r in sorted(runs):
        if Path(r).exists():
            skipped += 1
            continue
        for n in names:
            if n.startswith(r + "/") and not n.endswith("/"):
                Path(n).parent.mkdir(parents=True, exist_ok=True)
                with zf.open(n) as s, open(n, "wb") as o:
                    o.write(s.read())
        copied += 1
    print(f"merge_zip: copied {copied} complete runs, skipped {skipped} existing")
for n in names:   # analysis markdown shipped by the kernel: zip root or results/ root only
    parts = n.split("/")
    if n.endswith(".md") and (len(parts) == 1 or (len(parts) == 2 and parts[0] == "results")):
        dst = Path(a.docs_dir) / parts[-1]
        if not dst.exists():
            dst.write_bytes(zf.read(n))
            print(f"DOC {dst.as_posix()}")
