# Kaggle analysis kernel body (CPU, no GPU quota). launch_analysis.sh prepends a parameter header (R5_PREFIX, R8_PREFIX, NORMS,
# OUT_STEM, DRAWS, BOOT) and ships this as the kernel script. It mounts the code dataset `hypershift-code-an` plus the training
# kernels' outputs (`kernel_sources`), unpacks every results_*.zip they contain (never overwriting), runs scripts/r5f_analysis.py
# and leaves only <OUT_STEM>_results.md, <OUT_STEM>_summary.json, <OUT_STEM>_fig.png and analysis.log in /kaggle/working.
import os, shutil, subprocess, sys, zipfile
from pathlib import Path

INPUT = Path(os.environ.get("HS_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("HS_WORK", "/kaggle/working"))
TEMP = Path(os.environ.get("HS_TEMP", "/kaggle/temp"))
print("input", INPUT, "work", WORK)
for p in sorted(INPUT.rglob("*"))[:60]:
    print("  ", p.relative_to(INPUT))

hits = sorted(INPUT.rglob("r5f_analysis.py"))
assert hits, "scripts/r5f_analysis.py not found: attach the code dataset (it may be an unextracted zip)"
code_root = hits[0].parent.parent
REPO = TEMP / "repo"
shutil.rmtree(REPO, ignore_errors=True)
shutil.copytree(code_root, REPO, ignore=shutil.ignore_patterns("__pycache__"))
for z in list(REPO.glob("*.zip")):       # --dir-mode zip uploads Kaggle did not auto-extract
    zipfile.ZipFile(z).extractall(REPO)
    z.unlink()
assert (REPO / "scripts/r5f_analysis.py").exists() and (REPO / "src/hypershift/eval/stats.py").exists()

RES = TEMP / "res"
(RES / "results").mkdir(parents=True, exist_ok=True)
zips = sorted(INPUT.rglob("results_*.zip"))
print("result zips:", [str(z.relative_to(INPUT)) for z in zips])
n = 0
for z in zips:
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():
            # keep only what the analysis reads (skip logs); never overwrite
            if m.startswith("results/") and not m.endswith("/") and not m.startswith("results/logs") and not (RES / m).exists():
                zf.extract(m, RES)
                n += 1
print("extracted", n, "files; exps:", sorted(p.name for p in (RES / "results").iterdir()))
assert n, "no results found in the mounted kernel outputs (is the source kernel COMPLETE and did it write results_*.zip?)"

env = dict(os.environ, PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
cmd = [sys.executable, "scripts/r5f_analysis.py", "--root", str(RES), "--r5-prefix", R5_PREFIX, "--r8-prefix", R8_PREFIX, "--norms", *NORMS,
       "--draws", str(DRAWS), "--boot", str(BOOT), "--out-md", str(WORK / f"{OUT_STEM}_results.md"),
       "--out-json", str(WORK / f"{OUT_STEM}_summary.json"), "--fig", str(WORK / f"{OUT_STEM}_fig.png")]
print(" ".join(cmd))
r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
(WORK / "analysis.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
print(r.stdout[-3000:], r.stderr[-3000:])
shutil.rmtree(RES, ignore_errors=True)
if r.returncode:
    raise SystemExit(f"analysis failed rc={r.returncode}")
print("outputs:", sorted(p.name for p in WORK.iterdir()))
