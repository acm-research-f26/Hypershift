"""Build the two Kaggle upload folders under kaggle/build/ (git-ignored).

  kaggle/build/hypershift-code/       src/, scripts/ (no queues), configs/, pyproject.toml, requirements-kaggle.txt
  kaggle/build/hypershift-rsr-data/   RSR ticker lists, price CSVs, wiki csv, v2 hypergraph cache,
                                      relation tensors (gzip, stored as .npy.gzb so Kaggle does not auto-extract them into a directory; only RSR-I needs them, the cache covers everything else)

Usage (repo root):  .venv/Scripts/python.exe kaggle/build_bundle.py --username <kaggle-user> [--no-relation] [--markets NYSE NASDAQ] [--code-only]
Run with CUDA_VISIBLE_DEVICES=-1 if you run it next to the GPU queue (it does not import torch anyway).
"""
import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "kaggle" / "build"
RSR = REPO / "data" / "raw" / "rsr" / "data"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".pytest_cache")
REQS = """# Kaggle images already ship torch (CUDA), numpy, scipy, pandas, scikit-learn, matplotlib, pyyaml, tqdm.
# torch is deliberately NOT listed: never reinstall it on Kaggle.
numpy>=1.26
scipy>=1.11
pandas>=2.1
scikit-learn>=1.3
pyyaml>=6
tqdm>=4.66
tabulate>=0.9
"""


def cache_version() -> int:
    txt = (REPO / "src/hypershift/data/hypergraph.py").read_text()
    return int(re.search(r"^HYPERGRAPH_CACHE_VERSION\s*=\s*(\d+)", txt, re.M).group(1))


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024


def du(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def meta(folder: Path, user: str, slug: str, title: str, desc: str):
    # `kaggle datasets create` makes a PRIVATE dataset unless --public is passed.
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": title, "id": f"{user}/{slug}", "subtitle": desc[:80], "description": desc,
        "licenses": [{"name": "other"}]}, indent=2))


def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True).stdout.strip()


def build_code(user: str):
    d = BUILD / "hypershift-code"
    shutil.copytree(REPO / "src", d / "src", ignore=IGNORE)
    shutil.copytree(REPO / "scripts", d / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "queues", "overnight"))
    shutil.copytree(REPO / "configs", d / "configs")
    shutil.copy2(REPO / "pyproject.toml", d / "pyproject.toml")
    (d / "requirements-kaggle.txt").write_text(REQS)
    # results/tuned.json and configs/chosen.yaml change experiment configs (grid.py _geo precedence); never ship them silently.
    (d / "BUNDLE_INFO.txt").write_text(
        f"git HEAD: {sh('git', 'rev-parse', 'HEAD')}\n"
        f"dirty files in src/scripts/configs/pyproject: {sh('git', 'status', '--porcelain', '--', 'src', 'scripts', 'configs', 'pyproject.toml', ':!scripts/queues') or 'none'}\n"
        f"HYPERGRAPH_CACHE_VERSION: {cache_version()}\n"
        f"configs/chosen.yaml present: {(REPO / 'configs/chosen.yaml').exists()}; "
        f"results/tuned.json present locally: {(REPO / 'results/tuned.json').exists()} (not shipped)\n")
    meta(d, user, "hypershift-code", "hypershift-code", "THINK reproduction code (src, scripts, configs).")
    return d


def gz(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, gzip.GzipFile(dst, "wb", compresslevel=6, mtime=0) as fo:
        shutil.copyfileobj(fi, fo, 1 << 24)


def build_data(user: str, markets, relation: bool):
    d = BUILD / "hypershift-rsr-data"
    ver = cache_version()
    for m in markets:
        tick = RSR / f"{m}_tickers_qualify_dr-0.98_min-5_smooth.csv"
        names = [ln.split("\t")[0].strip() for ln in tick.read_text().splitlines() if ln.strip()]
        for f in (tick, RSR / f"{m}_wiki.csv", RSR / f"{m}_aver_line_dates.csv"):
            if f.exists():
                (d).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d / f.name)
        (d / "2013-01-01").mkdir(parents=True, exist_ok=True)
        for t in names:
            shutil.copy2(RSR / "2013-01-01" / f"{m}_{t}_1.csv", d / "2013-01-01" / f"{m}_{t}_1.csv")
        cf = RSR / "hypergraph_cache" / f"{m}_industry-wiki.json"
        got = json.loads(cf.read_text()).get("version") if cf.exists() else None
        if got != ver:
            sys.exit(f"hypergraph cache {cf} has version {got}, code expects {ver}: rebuild it locally first "
                     f"(any run builds it), or the Kaggle run would need the 4 GB relation tensors AND write access.")
        (d / "hypergraph_cache").mkdir(exist_ok=True)
        shutil.copy2(cf, d / "hypergraph_cache" / cf.name)
        # industry ticker json is read by poc_sectors.py (small-scale universe); NYSE only matters but both are tiny.
        sub = d / "relation" / "sector_industry"
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RSR / "relation/sector_industry" / f"{m}_industry_ticker.json", sub / f"{m}_industry_ticker.json")
        if relation:
            gz(RSR / "relation/sector_industry" / f"{m}_industry_relation.npy", sub / f"{m}_industry_relation.npy.gzb")
            wsub = d / "relation" / "wikidata"
            gz(RSR / "relation/wikidata" / f"{m}_wiki_relation.npy", wsub / f"{m}_wiki_relation.npy.gzb")
            shutil.copy2(RSR / "relation/wikidata" / f"{m}_connections.json", wsub / f"{m}_connections.json")
    (d / "DATA_INFO.txt").write_text(f"markets: {markets}\nhypergraph cache version: {ver}\nrelation tensors (gz): {relation}\n"
                                     "Relation .npy.gzb (gzip) are only read by model=rsr_i (R8 RSR-I); the unpack step in run_kaggle handles .npy, .npy.gz/.gzb and auto-extracted directories.\n")
    meta(d, user, "hypershift-rsr-data", "hypershift-rsr-data", "RSR NYSE/NASDAQ prices, wiki csv, hypergraph cache v2, relation tensors.")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default="YOUR_KAGGLE_USERNAME")
    ap.add_argument("--markets", nargs="+", default=["NYSE", "NASDAQ"])
    ap.add_argument("--no-relation", action="store_true", help="skip the (gzipped) relation tensors; RSR-I then cannot run")
    ap.add_argument("--code-only", action="store_true", help="rebuild only hypershift-code (seconds); the data dataset is already uploaded, so skip re-gzipping several GB")
    a = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    # Never wipe kaggle/build wholesale: out_<tag>/ holds downloaded result zips that may not be merged yet (a fetch followed by the next launch used to delete them).
    for sub in ("hypershift-code", "kernel") + (() if a.code_only else ("hypershift-rsr-data",)):
        if (BUILD / sub).exists():
            shutil.rmtree(BUILD / sub)
    c = build_code(a.username)
    print(f"{c}: {human(du(c))}  ({sum(1 for _ in c.rglob('*') if _.is_file())} files)")
    if a.code_only:
        return
    r = build_data(a.username, a.markets, not a.no_relation)
    print(f"{r}: {human(du(r))}  ({sum(1 for _ in r.rglob('*') if _.is_file())} files)")
    for f in sorted(r.rglob("*.gz*")):
        print(f"   {f.relative_to(r)}: {human(f.stat().st_size)}")


if __name__ == "__main__":
    main()
