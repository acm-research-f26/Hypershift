"""Convert kaggle/run_kaggle.py (cells separated by '# %%' lines) into an .ipynb.

  python kaggle/make_notebook.py OUT.ipynb [--session 1|2|all] [--tag TAG]

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
ap.add_argument("--src", default=str(Path(__file__).with_name("run_kaggle.py")))
a = ap.parse_args()
text = Path(a.src).read_text()
if a.session:
    text = re.sub(r'^SESSION = "[^"]*"', f'SESSION = "{a.session}"', text, flags=re.M)
if a.tag:
    text = re.sub(r'^TAG = "[^"]*"', f'TAG = "{a.tag}"', text, flags=re.M)
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
