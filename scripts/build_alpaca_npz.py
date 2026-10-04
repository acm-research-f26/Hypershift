"""Build the compact Alpaca walk-forward panel (Phase 1.5c): data/raw/rsr/data/alpaca_panel_2016_2023.npz and kaggle/build/hypershift-alpaca-data/.
CPU only; reads the local bars.pkl.gz (never uploads keys or raw bars). Usage: python scripts/build_alpaca_npz.py"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import eval_post2017 as E
from hypershift.data.alpaca_wf import PANEL_NAME, save_panel_npz

data, order, *_ = E.load_panel()
ident = pd.read_csv(E.IDENT).identity_ok.to_numpy()
out = E.RSR / PANEL_NAME
save_panel_npz(data, ident, out)
print(out, out.stat().st_size >> 20, "MB", data.features.shape, "identity", int(ident.sum()))
b = Path("kaggle/build/hypershift-alpaca-data")
b.mkdir(parents=True, exist_ok=True)
shutil.copy2(out, b / PANEL_NAME)
user = os.environ.get("KAGGLE_USER", "tomphamdustry")
(b / "dataset-metadata.json").write_text(json.dumps({
    "title": "hypershift-alpaca-data", "id": f"{user}/hypershift-alpaca-data", "subtitle": "Alpaca split-adjusted daily panel 2016-2023 (THINK 1.5c)",
    "description": "Features/gt/mask/dates/tickers/identity mask in the frozen RSR NYSE node order. No API keys.", "licenses": [{"name": "other"}]}, indent=2))
