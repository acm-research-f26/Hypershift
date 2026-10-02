"""Build kaggle/build/hypershift-rsr-orig-code: the authors' ORIGINAL RSR training files (unmodified, md5 checked) plus
their pretrained NYSE sequence embedding. Source: git clone of fulifeng/Temporal_Relational_Stock_Ranking at the pinned commit,
and the authors' Google Drive tarball (README 'pretrained sequential embedding'), downloaded with gdown.
usage: python build_rsr_orig.py --repo <clone dir> --pretrain <NYSE_rank_lstm_seq-8_unit-32_0.csv.npy> --username <kaggle user>"""
import argparse, hashlib, json, shutil
from pathlib import Path

COMMIT = "cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d"
MD5 = {"evaluator.py": "72b3ff62bc16a4b2f4e2cdc5a9d35a26", "load_data.py": "a00745da248c7a614ee4f5137c14939e",
       "rank_lstm.py": "d29ce7b2431a4e124c73ac0402da20c5", "relation_rank_lstm.py": "2f7115a2e159ba733425c49b543fcd29"}
PRETRAIN_MD5 = "254ec4205ed309bfc40a58e7dde8da52"

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--pretrain", required=True); ap.add_argument("--username", required=True)
a = ap.parse_args()
out = Path(__file__).resolve().parents[1] / "build" / "hypershift-rsr-orig-code"
shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
for n, m in MD5.items():
    src = Path(a.repo) / "training" / n
    assert md5(src) == m, f"{n} differs from the pinned commit {COMMIT}"
    shutil.copy2(src, out / n)
assert md5(a.pretrain) == PRETRAIN_MD5
shutil.copy2(a.pretrain, out / "NYSE_rank_lstm_seq-8_unit-32_0.csv.npy")
shutil.copy2(Path(__file__).with_name("patch_rsr.py"), out / "patch_rsr.py")
(out / "SOURCE.txt").write_text(f"fulifeng/Temporal_Relational_Stock_Ranking @ {COMMIT}\nmd5 {json.dumps(MD5, indent=1)}\npretrain md5 {PRETRAIN_MD5}\n")
(out / "dataset-metadata.json").write_text(json.dumps({"title": "hypershift-rsr-orig-code", "id": f"{a.username}/hypershift-rsr-orig-code",
    "licenses": [{"name": "other"}]}))
print("built", out)
