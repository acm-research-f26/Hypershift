# Kaggle script kernel: the authors' ORIGINAL RSR code (Feng et al. 2019, NYSE, RSR-I implicit, README command) under TF2 compat.
# Datasets attached: hypershift-rsr-data (their price CSVs + relation tensors, same files, checksums verified below) and
# hypershift-rsr-orig-code (their unmodified training/*.py at commit cfbb01b + their pretrained NYSE embedding + patch_rsr.py).
# Output: /kaggle/working/rsr_orig_out/seed_<s>/{test_pred_all,val_pred_all}.npz-style stacks, gt/mask, epochs.jsonl, log.txt
import gzip, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

SMOKE = False                      # PARAM: True = 1 epoch, 1 seed
SEEDS = [123456789, 1, 2, 3, 4]    # first = the authors' hard-coded seed; others via patch P2 (RSR_SEED)
EPOCHS = 50                        # the authors' hard-coded value (their main block)
LIMIT_H = 11.0
EXPECT_EOD_AGG = "276fa52407551397f483a691ffb9e243"      # md5 over "md5 name" lines of the 1737 NYSE ticker files, in ticker-file order (the dataset ships 2763 of the 2817 files; local copy of the full dir == GitHub clone)
EXPECT_REL_MD5 = "c427f780a4290f8fe50fc46a8a4d5fb5"      # NYSE_industry_relation.npy (local copy == relation.tar.gz from the clone)
T0 = time.time()
INPUT, WORK, TEMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/kaggle/temp")
OUT = WORK / "rsr_orig_out"
OUT.mkdir(parents=True, exist_ok=True)
if SMOKE:
    SEEDS, EPOCHS = SEEDS[1:2], 1

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def find(name, kind="any"):
    hits = sorted(p for p in INPUT.rglob("*") if p.name == name)
    assert hits, f"{name} not found under {INPUT}; contents: {[p.name for p in list(INPUT.iterdir())]}"
    return hits[0]

code = find("patch_rsr.py").parent
data_in = find("NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv").parent
print("code dir", code, "data dir", data_in, flush=True)
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True, text=True).stdout, flush=True)

# ---- assemble ../training and ../data like the repo layout
R = TEMP / "rsr"
shutil.rmtree(R, ignore_errors=True)
(R / "training").mkdir(parents=True)
for f in code.glob("*.py"):
    if f.name not in ("patch_rsr.py",):
        shutil.copy2(f, R / "training" / f.name)
D = R / "data"
(D / "relation/sector_industry").mkdir(parents=True)
(D / "pretrain").mkdir()
shutil.copy2(data_in / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv", D)
eod_src = data_in / "2013-01-01"
if not eod_src.exists():
    zs = list(data_in.glob("*.zip"))
    assert zs, "2013-01-01 missing"
    import zipfile
    for z in zs:
        zipfile.ZipFile(z).extractall(D)
else:
    shutil.copytree(eod_src, D / "2013-01-01")   # real copy: the code reads ../data/2013-01-01/../relation, so a symlink would resolve .. into the dataset dir
tk = [l.strip() for l in open(D / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv") if l.strip()]
names = [f"NYSE_{t}_1.csv" for t in tk]
agg = hashlib.md5(chr(10).join(f"{md5(D / '2013-01-01' / n)} {n}" for n in names).encode()).hexdigest()
print("EOD files", len(names), "aggregate md5", agg, "OK" if agg == EXPECT_EOD_AGG else "MISMATCH", flush=True)
assert agg == EXPECT_EOD_AGG

# relation tensor: file, .gz, or a directory auto-extracted by Kaggle (see kaggle/README.md)
rel_dst = D / "relation/sector_industry/NYSE_industry_relation.npy"
cands = sorted(p for p in INPUT.rglob("NYSE_industry_relation.npy*"))
print("relation candidates", [str(c) for c in cands], flush=True)
def place(p):
    if p.is_dir():
        inner = sorted(q for q in p.rglob("*") if q.is_file())
        return place(inner[0])
    if p.name.endswith((".gz", ".gzb")):
        with gzip.open(p, "rb") as fi, open(rel_dst, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 24)
    else:
        shutil.copy2(p, rel_dst)
place(cands[0])
rm = md5(rel_dst)
print("relation md5", rm, "OK" if rm == EXPECT_REL_MD5 else "MISMATCH", flush=True)
assert rm == EXPECT_REL_MD5
emb = find("NYSE_rank_lstm_seq-8_unit-32_0.csv.npy")
os.symlink(emb, D / "pretrain/NYSE_rank_lstm_seq-8_unit-32_0.csv.npy")
print("pretrain md5", md5(emb), flush=True)
print("LAYOUT rel_dst is_file", rel_dst.is_file(), "is_dir", rel_dst.is_dir(), "eod is_symlink", (D / "2013-01-01").is_symlink(), "resolved", (D / "2013-01-01" / ".." / "relation/sector_industry").resolve(), flush=True)

r = subprocess.run([sys.executable, str(code / "patch_rsr.py"), str(R / "training")], capture_output=True, text=True)
print(r.stdout, r.stderr[-2000:], flush=True)
assert r.returncode == 0
tfv = subprocess.run([sys.executable, "-c", "import tensorflow as tf, numpy; print('tf', tf.__version__, 'np', numpy.__version__, tf.config.list_physical_devices('GPU'))"],
                     capture_output=True, text=True)
print(tfv.stdout, tfv.stderr[-800:], flush=True)

if os.environ.get("RSR_REQUIRE_GPU", "1") == "1" and "GPU:0" not in tfv.stdout and not SMOKE:
    raise SystemExit("no GPU visible to TensorFlow in a full run; refusing to run 50 epochs x seeds on CPU (558 s/epoch measured)")
# ---- run seeds sequentially
import numpy as np
for k, seed in enumerate(SEEDS):
    if (time.time() - T0) / 3600 > LIMIT_H - 0.6 and k > 0:
        print("time guard: not starting seed", seed, flush=True)
        break
    od = OUT / f"seed_{seed}"
    if (od / "DONE").exists():
        continue
    shutil.rmtree(od, ignore_errors=True)
    od.mkdir(parents=True)
    env = dict(os.environ, RSR_SEED=str(seed), RSR_EPOCHS=str(EPOCHS), RSR_OUT=str(od), TF_FORCE_GPU_ALLOW_GROWTH="true", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "relation_rank_lstm.py", "-m", "NYSE", "-l", "8", "-u", "32", "-a", "10",
           "-e", "NYSE_rank_lstm_seq-8_unit-32_0.csv.npy", "-g", "1" if "GPU:0" in tfv.stdout else "0"]
    print("RUN seed", seed, " ".join(cmd), flush=True)
    t1 = time.time()
    with open(od / "log.txt", "w") as lf:
        p = subprocess.Popen(cmd, cwd=R / "training", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            lf.write(line)
            if line.startswith(("epoch:", "device", "Valid MSE", "Better", "Best", "	Best", "	 Test", "	 Valid")):
                print(f"[{seed}] {line.rstrip()[:300]}", flush=True)
        p.wait()
    print("seed", seed, "rc", p.returncode, "minutes", round((time.time() - t1) / 60, 1), flush=True)
    if p.returncode != 0:
        print("".join(open(od / "log.txt").readlines()[-40:]), flush=True)
        if k == 0:
            raise SystemExit("first run failed")
        continue
    eps = sorted(od.glob("ep*_test_pred.npy"))
    np.save(od / "test_pred_all.npy", np.stack([np.load(f) for f in eps]))
    np.save(od / "val_pred_all.npy", np.stack([np.load(str(f).replace("test_pred", "val_pred")) for f in eps]))
    for f in eps:
        f.unlink()
        Path(str(f).replace("test_pred", "val_pred")).unlink()
    (od / "DONE").write_text("ok")
    shutil.copy2(R / "training/relation_rank_lstm.py", OUT / "relation_rank_lstm.patched.py")
print("all done, minutes", round((time.time() - T0) / 60, 1), flush=True)
