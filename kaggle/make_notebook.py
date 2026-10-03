"""Convert kaggle/run_kaggle.py (cells separated by '# %%' lines) into an .ipynb.

  python kaggle/make_notebook.py OUT.ipynb [--session 1|2|all|r8f-top|p1f|r5f2|<preset>-s] [--tag TAG] [--limit-h H] [--timeout-h H]

The SESSION and TAG assignments are rewritten when the flags are given.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--session")
ap.add_argument("--tag")
ap.add_argument("--limit-h", help="SESSION_LIMIT_H: launch-guard horizon in hours")
ap.add_argument("--timeout-h", help="KERNEL_TIMEOUT_H: kernel -t timeout in hours (bounds the in-kernel analysis)")
ap.add_argument("--src", default=str(Path(__file__).with_name("run_kaggle.py")))
a = ap.parse_args()
text = Path(a.src).read_text()
if a.session:
    text = re.sub(r'^SESSION = "[^"]*"', f'SESSION = "{a.session}"', text, flags=re.M)
if a.tag:
    text = re.sub(r'^TAG = "[^"]*"', f'TAG = "{a.tag}"', text, flags=re.M)
if a.limit_h:
    text = re.sub(r"^SESSION_LIMIT_H = [0-9.]+", f"SESSION_LIMIT_H = {float(a.limit_h)}", text, flags=re.M)
if a.timeout_h:
    text = re.sub(r"^KERNEL_TIMEOUT_H = [0-9.]+", f"KERNEL_TIMEOUT_H = {float(a.timeout_h)}", text, flags=re.M)
# EH seeds already complete locally (results/R5_g2/EH/seed_<k>/metrics.json) are written into the notebook so they are listed and skipped.
eh_done = sorted(int(p.parent.name.split("_")[1]) for p in (ROOT / "results/R5_g2/EH").glob("seed_*/metrics.json"))
text = re.sub(r"^EH_DONE_LOCAL = \[[^\]]*\]", f"EH_DONE_LOCAL = {eh_done}", text, flags=re.M)
cells = []
for chunk in re.split(r"^# %%", text, flags=re.M):
    if not chunk.strip():
        continue
    head, _, body = chunk.partition("\n")
    if head.strip().startswith("[markdown]"):
        src = "\n".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.strip("\n").splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
    else:
        src = (f"# {head.strip()}\n" if head.strip() else "") + body.strip("\n")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path(a.out).write_text(json.dumps(nb, indent=1))
print(f"{a.out}: {len(cells)} cells")
