# %% [markdown]
# # Hypershift on Kaggle
# Attach datasets `hypershift-code` and `hypershift-rsr-data` (and optionally a prior results dataset), enable a GPU
# (T4 x2 recommended, see README), Internet on, then Run All (or Save Version -> Save & Run All for a background run).
# Everything is resumable: a run folder with `metrics.json` is skipped. Output = /kaggle/working/results_<TAG>.zip.

# %% Cell 1: parameters
import gzip, json, os, queue, re, shlex, shutil, signal, subprocess, sys, threading, time, zipfile
from pathlib import Path

T0 = time.time()
SESSION = "1"        # PARAM  "1" = R8 small + R5_g2 EE/HE | "2" = R5_g2 EH (seeds 0-24) then R8 full NYSE | "all" = both | "3" = RSR_I full NYSE seeds 4-9 | "4" = R5_g2 G5 (HH_none, EE_none) + A10 (THINK_nodist) | "custom" = fill COMMANDS yourself
TAG = "s1"           # PARAM  names the output zip: results_<TAG>.zip
N_WORKERS = 3        # parallel training processes (4 vCPU; THINK is launch/CPU bound, ~1.6 GB VRAM each, so 16 GB is not the limit)
SESSION_LIMIT_H = 12.0   # Kaggle hard limit for a GPU session
SAFETY_MIN = 25          # stop launching this long before the limit; leftovers are killed 15 min before the limit; then zip
RETRIES = 2              # extra attempts for a run that exits without metrics.json
REQUIRE_GPU = True
UNPACK_RELATION_FOR = ["NYSE"]   # markets whose relation tensors are gunzipped (only model=rsr_i reads them)
FIX_TORCH_FOR_OLD_GPU = False    # True: reinstall torch cu126 (needs Internet) if the GPU arch is unsupported (P100 problem)
CLEAN_WORKING = True             # after zipping, delete the results/ copy so /kaggle/working holds just the zip

INPUT = Path(os.environ.get("HS_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("HS_WORK", "/kaggle/working"))
TEMP = Path(os.environ.get("HS_TEMP", "/kaggle/temp"))
DRYRUN = os.environ.get("HS_DRYRUN") == "1"   # local simulation: --dry-run on run_grid/poc_sectors, skip pip/GPU/hypershift.run
REPO = WORK / "hypershift"
PY = sys.executable
print("session", SESSION, "tag", TAG, "workers", N_WORKERS, "input", INPUT, "work", WORK, "dryrun", DRYRUN)

# %% Cell 2: copy code, install
def find_dir(pattern, root=INPUT):
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise SystemExit(f"{pattern} not found under {root}: attach the dataset to the notebook (Add Input)")
    return hits[0].parent

code_src = find_dir("pyproject.toml")
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(code_src, REPO, ignore=shutil.ignore_patterns("__pycache__", "dataset-metadata.json"))
for z in list(REPO.glob("*.zip")):                 # --dir-mode zip uploads that Kaggle did not auto-extract (src/, scripts/, configs/)
    zipfile.ZipFile(z).extractall(REPO)
    z.unlink()
for need in ("src/hypershift/run.py", "scripts/run_grid.py", "configs/think_nyse.yaml"):
    assert (REPO / need).exists(), f"{need} missing in the code dataset copy"
os.chdir(REPO)
print((REPO / "BUNDLE_INFO.txt").read_text())
assert not (REPO / "results/tuned.json").exists() and not (REPO / "configs/chosen.yaml").exists(), "unexpected config overrides in bundle"

ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
if not DRYRUN:
    # torch is not a dependency of the package (pyproject.toml), so this never touches Kaggle's torch.
    r = subprocess.run([PY, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "-e", "."], capture_output=True, text=True)
    print("pip install -e . rc", r.returncode, r.stderr[-300:])
    for mod, pipname in [("yaml", "pyyaml"), ("tqdm", "tqdm"), ("tabulate", "tabulate"), ("scipy", "scipy"),
                         ("pandas", "pandas"), ("sklearn", "scikit-learn")]:
        if subprocess.run([PY, "-c", f"import {mod}"], capture_output=True).returncode:
            print("installing missing", pipname)
            subprocess.run([PY, "-m", "pip", "install", "-q", pipname])
chk = subprocess.run([PY, "-c", "import hypershift, numpy, torch; print('hypershift ok, numpy', numpy.__version__, 'torch', torch.__version__)"],
                     capture_output=True, text=True, env=ENV)
print(chk.stdout, chk.stderr[-500:])
if chk.returncode:
    raise SystemExit("import hypershift/numpy/torch failed (see above); all runs would fail, stopping now")

# %% Cell 3: data -> data/raw/rsr/data  (real dir in /kaggle/temp so it is not part of the notebook output)
data_in = find_dir("NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv")
DATA = TEMP / "rsr_data"
if DATA.exists():
    shutil.rmtree(DATA)
DATA.mkdir(parents=True)

def link_or_copy(src, dst):
    try:
        os.symlink(src, dst, target_is_directory=Path(src).is_dir())
    except OSError:                       # e.g. Windows without symlink rights (local simulation)
        shutil.copytree(src, dst) if Path(src).is_dir() else shutil.copy2(src, dst)

for f in data_in.iterdir():
    if f.is_file():
        if f.suffix == ".zip":                       # --dir-mode zip uploads that Kaggle did not auto-extract
            zipfile.ZipFile(f).extractall(DATA)
        else:
            shutil.copy2(f, DATA / f.name)
if (data_in / "hypergraph_cache").exists():
    shutil.copytree(data_in / "hypergraph_cache", DATA / "hypergraph_cache", dirs_exist_ok=True)   # writable copy (v2 caches)
if (data_in / "2013-01-01").exists():
    link_or_copy(data_in / "2013-01-01", DATA / "2013-01-01")
rel_src = data_in / "relation" if (data_in / "relation").exists() else DATA / "relation"
def place_relation(f, dst, unpack):
    """Put one relation file at dst (a real .npy). Kaggle may have left `X.npy.gz` as is, or auto-extracted it into a
    DIRECTORY named `X.npy` (name without .gz, holding the content): that directory was symlinked as the .npy and RSR-I
    died with IsADirectoryError (Phase 1 R8, 2026-10-01). Handle file, .gz and directory."""
    if f.is_dir():                                       # auto-extracted archive: take the payload file(s) inside
        inner = sorted(p for p in f.rglob("*") if p.is_file())
        assert inner, f"{f} is an empty directory"
        for p in inner:
            if p.name.endswith((".npy", ".npy.gz", ".npy.gzb")):
                return place_relation(p, dst, unpack)
        assert len(inner) == 1, f"{f} is a directory with unknown content: {[p.name for p in inner]}"
        return place_relation(inner[0], dst, unpack)
    if f.name.endswith((".gz", ".gzb")):
        if unpack and not DRYRUN:
            with gzip.open(f, "rb") as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 24)
            print("unpacked", dst.name, dst.stat().st_size >> 20, "MB")
    elif not dst.exists():
        link_or_copy(f, dst)


for sub in ("sector_industry", "wikidata"):
    (DATA / "relation" / sub).mkdir(parents=True, exist_ok=True)
    for f in sorted((rel_src / sub).glob("*")):
        dst = DATA / "relation" / sub / re.sub(r"\.gzb?$", "", f.name)
        place_relation(f, dst, f.name.split("_")[0] in UNPACK_RELATION_FOR)
for m in UNPACK_RELATION_FOR:                            # fail here, not hours later inside a training run
    for p in sorted((DATA / "relation").glob(f"*/{m}_*_relation.npy")) if not DRYRUN else []:
        assert p.is_file() and p.stat().st_size > 1_000_000, f"{p} is not a real relation tensor ({p.resolve()})"
for need in ("2013-01-01", "hypergraph_cache"):
    assert (DATA / need).exists(), f"{need} missing in the data dataset"
target = REPO / "data/raw/rsr"
target.mkdir(parents=True, exist_ok=True)
link_or_copy(DATA, target / "data")
print("data ready:", sorted(p.name for p in (target / "data").iterdir()))

# %% Cell 4: optional prior results (attach a dataset holding results_*.zip; never overwrites existing files)
(REPO / "results/logs").mkdir(parents=True, exist_ok=True)
for z in sorted(INPUT.rglob("results_*.zip")):
    n = 0
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():
            if m.startswith("results/") and not m.endswith("/") and not (REPO / m).exists():
                zf.extract(m, REPO)
                n += 1
    print("prior results", z.name, "-> extracted", n, "files")

# %% Cell 5: GPU / torch check (a P100 with a torch build that dropped sm_60 fails here, not 3 hours later)
GPU_TEST = ("import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available());"
            "print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list());"
            "x=torch.randn(64,64,device='cuda');print('cuda op ok', float((x@x).sum()));print(torch.cuda.device_count())")
NGPU = 1
if not DRYRUN:
    t = subprocess.run([PY, "-c", GPU_TEST], capture_output=True, text=True, env=ENV)
    print(t.stdout, t.stderr[-600:])
    if t.returncode and FIX_TORCH_FOR_OLD_GPU:
        subprocess.run([PY, "-m", "pip", "install", "-q", "--force-reinstall", "torch", "--index-url", "https://download.pytorch.org/whl/cu126"])
        t = subprocess.run([PY, "-c", GPU_TEST], capture_output=True, text=True, env=ENV)
        print(t.stdout, t.stderr[-600:])
    if t.returncode and REQUIRE_GPU:
        raise SystemExit("GPU unusable with this torch build. Switch the accelerator to 'GPU T4 x2' (README) or set FIX_TORCH_FOR_OLD_GPU = True.")
    if t.returncode == 0:
        NGPU = max(1, int(t.stdout.strip().splitlines()[-1]))
print("GPUs:", NGPU)

# %% Cell 6: COMMANDS  (same commands as scripts/queues/phase1_gpu_3.sh [EE, HE, EH] and phase1_gpu_r8.sh, one run per command)
SEEDS = range(10)
EH_SEEDS = range(25)             # phase1_gpu_3.sh runs EH on seeds 0-24
EH_DONE_LOCAL = []               # BUILD  seeds with results/R5_g2/EH/seed_<k>/metrics.json locally at build time (make_notebook.py rewrites this line)
R5P = "--set exp=R5_g2 norm=paper epochs=100 patience=1000"
R8C = "--config configs/think_nyse.yaml --set exp=R8_baselines_g2 norm=paper epochs=100 patience=1000 batch_days=8"
# est_min = initial wall-clock guess per run (3050: THINK 21 s/epoch x 100 = 35 min; baselines 17-40 min estimated; small ~2-4 min).
# Replaced by the longest observed duration of that kind once one has finished.
def r5(label, s, est_min=40):
    return dict(kind=f"R5_{label}", est_min=est_min, done=f"results/R5_g2/{label}/seed_{s}",
                cmd=f"{{py}} scripts/run_grid.py E2_geometry --labels {label} --seeds {s} {R5P}")

def r8_small(m, s):
    return dict(kind=f"R8small_{m}", est_min=4, done=f"results/POC_sectors_R8_{m}_g2/{m}/seed_{s}",
                cmd=f"{{py}} scripts/poc_sectors.py run --variant R8_{m}_g2 --seeds {s} --set model={m}")

def r8_full(label, model, micro, s):
    return dict(kind=f"R8_{label}", est_min=40, done=f"results/R8_baselines_g2/{label}/seed_{s}",
                cmd=f"{{py}} -m hypershift.run {R8C} label={label} model={model} micro_batch_days={micro} --seeds {s}")

# EE/HE first (time critical, longest); the short small-scale runs last so they can fill the tail if the time guard stops launches.
# POC_sectors_R8_rsr_i_g2 seed 0 already has metrics.json locally (R8 queue, stopped), so it is not repeated here.
S1 = [r5(l, s) for s in SEEDS for l in ("EE", "HE")] +      [r8_small(m, s) for m in ("rsr_i", "sthgcn") for s in SEEDS if (m, s) != ("rsr_i", 0)]
# Preset 2: EH first (R5_g2, exact phase1_gpu_3.sh EH command; seeds already complete locally are listed and skipped), then R8 full NYSE.
EH_TODO = [s for s in EH_SEEDS if s not in EH_DONE_LOCAL]
print("EH seeds complete locally at build time (skipped):", EH_DONE_LOCAL, "| EH seeds queued:", EH_TODO)
S2 = [r5("EH", s, est_min=30) for s in EH_TODO] +      [c for s in SEEDS for c in (r8_full("RSR_I", "rsr_i", 2, s), r8_full("STHGCN", "sthgcn", 4, s))]
# Preset 3: only full-NYSE RSR-I seeds 4-9 (s1/s2 RSR-I failed on the relation-dir bug; local GPU queue does seeds 0-1).
S3 = [r8_full("RSR_I", "rsr_i", 2, s) for s in range(4, 10)]
# Preset 4 (Phase 1 G5 + A10 on full NYSE, same protocol as R5_g2 THINK/EE): structure=none first (no graph, fast), then THINK without the attention distance term.
# Output goes to results/R5_g2/<label>/seed_<k> so the arms pair by seed with THINK_paperProtocol / EE already there.
def r5x(kind, grid, base_label, label, extra, s, est_min):
    return dict(kind=kind, est_min=est_min, done=f"results/R5_g2/{label}/seed_{s}",
                cmd=f"{{py}} scripts/run_grid.py {grid} --labels {base_label} --seeds {s} {R5P} label={label} {extra}")
S4 = ([r5x("R5_HH_none", "E1_main", "THINK_paperProtocol", "HH_none", "structure=none", s, 15) for s in SEEDS] +
      [r5x("R5_EE_none", "E2_geometry", "EE", "EE_none", "structure=none", s, 15) for s in SEEDS] +
      [r5x("R5_THINK_nodist", "E1_main", "THINK_paperProtocol", "THINK_nodist", "attn_dist=off", s, 35) for s in SEEDS])
# Preset 5 (Phase 1.5 D): known-signal check, scripts/known_signal.py on the 309-stock g2 universe with planted signals.
# One run per command -> results/known_signal_<level>_<mode>/<arm>/seed_<k>. Seeds outermost so a cut-off session still has full rows.
KS_LEVELS, KS_ARMS, KS_MODES, KS_SEEDS, KS_EPOCHS = ("none", "own", "group", "mid"), ("HH_hyper", "EH_hyper", "EE_hyper", "HH_none", "EE_none"), ("level", "relative"), range(5), 30
def ks(level, arm, mode, s, epochs=KS_EPOCHS, exp="known_signal", est_min=6, extra=""):
    return dict(kind=f"KS_{exp}_{arm}", est_min=est_min, done=f"results/{exp}_{level}_{mode}/{arm}/seed_{s}",
                cmd=f"{{py}} scripts/known_signal.py --levels {level} --arms {arm} --modes {mode} --seeds {s} --epochs {epochs} --exp {exp} --device cuda" + (f" --set {extra}" if extra else ""))
S5 = [ks(l, a, m, s) for s in KS_SEEDS for l in KS_LEVELS for m in KS_MODES for a in KS_ARMS]
# Preset 6: the two extra SNR levels (low, high) of the same grid, so SNR is varied and not only the signal source.
S6 = [ks(l, a, m, s) for s in KS_SEEDS for l in ("low", "high") for m in KS_MODES for a in KS_ARMS]
# Preset 7: why does level mode fail? (a) R5-like budget: 100 epochs, no early stop; (b) one day per step like the repos (batch_days=1), 30 epochs.
S7 = ([ks(l, a, "level", s, 100, "ks_long", 10, "patience=1000") for s in range(3) for l in ("own", "high") for a in ("EE_none", "HH_none", "EE_hyper", "HH_hyper")] +
      [ks(l, a, "level", s, 30, "ks_bd1", 8, "batch_days=1 patience=1000") for s in range(3) for l in ("own", "high") for a in ("EE_none", "HH_none")])
# Preset 8 (Phase 1.5 F): learnability factors on the known-signal benchmark, one factor at a time vs the D baseline
# (D baseline = existing known_signal_<level>_relative seeds 0-2: wd 5e-4, lr 1e-3, alpha 1). Relative inputs; HH_hyper + EH_hyper; high + group; seeds 0-2.
F_FACT = {"wd0": "weight_decay=0", "adamw": "decoupled_wd=true", "gain4": "init_gain=4", "head50": "head_scale=50",
          "wd0_lr3e3": "weight_decay=0 lr=0.003", "wd0_lr1e2": "weight_decay=0 lr=0.01", "wd0_a0": "weight_decay=0 alpha=0",
          "wd0_a10": "weight_decay=0 alpha=10", "wd0_res": "weight_decay=0 spatial_residual=true",
          "wd0_bd1": "weight_decay=0 batch_days=1", "wd0_gain4": "weight_decay=0 init_gain=4",
          "wd0_std": "weight_decay=0 input_std=true input_scale=0.3"}
F_COMMON = "log_ic=true"
def fk(name, level, arm, mode, s, extra, epochs=30, est=4):
    return ks(level, arm, mode, s, epochs, f"f_{name}", est, f"{F_COMMON} {extra}")
S8 = ([fk(n, l, a, "relative", s, x, est=(12 if "bd1" in n else 4)) for s in range(3) for n, x in F_FACT.items() for l in ("high", "group") for a in ("HH_hyper", "EH_hyper")] +
      [fk(n, "high", a, "level", s, x) for s in range(3) for n, x in
       (("lvl_wd0", "weight_decay=0"), ("lvl_wd0_std", "weight_decay=0 input_std=true input_scale=0.3"),
        ("lvl_wd0_price", "weight_decay=0 target=price")) for a in ("HH_hyper", "EH_hyper", "EE_none")])
# 8s: 1-epoch smoke of each new job type (all switches on)
S8S = [fk("smoke", "group", a, m, 0, "weight_decay=0 decoupled_wd=true init_gain=2 head_scale=10 spatial_residual=true input_std=true input_scale=0.3 grad_clip=0" + (" target=price" if m == "level" else ""), epochs=1, est=2)
       for a, m in (("HH_hyper", "relative"), ("EH_hyper", "relative"), ("EE_none", "level"), ("HH_none", "relative"), ("HH_hyper", "level"))]
# Preset 9 (Phase 1.5 F stage 2): the winning switches on all arms incl. no-graph, 5 seeds, high + group, relative inputs.
# c1 = wd0; c2 = wd0 + alpha0; c3 = wd0 + spatial_residual (graph arms only: no-graph arms equal c1); c4 = wd0 + alpha0 + residual (graph arms only; no-graph = c2).
F2 = [("f2_c1", "weight_decay=0", ("HH_hyper", "EH_hyper", "HH_none", "EE_none")),
      ("f2_c2", "weight_decay=0 alpha=0", ("HH_hyper", "EH_hyper", "HH_none", "EE_none")),
      ("f2_c3", "weight_decay=0 spatial_residual=true", ("HH_hyper", "EH_hyper")),
      ("f2_c4", "weight_decay=0 alpha=0 spatial_residual=true", ("HH_hyper", "EH_hyper"))]
S9 = [fk(n, l, a, "relative", s, x) for s in range(5) for n, x, arms in F2 for l in ("high", "group") for a in arms]
# Preset 10 (Phase 1.5 F): corrected full-NYSE rerun with the F fix (weight_decay=0) on relative inputs, THINK (HH) and EH,
# norm=paper (R5_f_paper) and norm=train (R5_f_train), log_ic on. Seeds outermost so a cut-off still has paired rows. EE last, 5 seeds.
F10 = "input_mode=relative weight_decay=0 log_ic=true epochs=100 patience=1000"
def r5f(arm, norm, s, exp_prefix="R5_f", epochs=None):
    grid, base = ("E1_main", "THINK_paperProtocol") if arm == "HH" else ("E2_geometry", arm)
    exp = f"{exp_prefix}_{norm}"
    extra = F10 if epochs is None else F10.replace("epochs=100", f"epochs={epochs}")
    return dict(kind=f"R5f_{arm}_{norm}", est_min=35 if arm == "HH" else 30, done=f"results/{exp}/{arm}/seed_{s}",
                cmd=f"{{py}} scripts/run_grid.py {grid} --labels {base} --seeds {s} --set exp={exp} norm={norm} {extra} label={arm}")
S10 = ([r5f(a, n, s) for s in range(10) for n in ("paper", "train") for a in ("HH", "EH")] +
       [r5f("EE", n, s) for s in range(5) for n in ("paper", "train")])
S10S = [r5f(a, "train", 0, "R5_f_smoke", epochs=1) for a in ("HH", "EH", "EE")]
# Preset 11 (Phase 1.5 F): R8 baselines (RSR-I, STHGCN) on full NYSE with the F fix, same config as preset 10 (relative inputs, wd 0, log_ic, 100 epochs),
# plus batch_days=8 and micro_batch_days (2 RSR-I / 4 STHGCN) as in preset 2's R8 runs. norm=train (R8_f_train) and norm=paper (R8_f_paper), 5 seeds, seeds outermost.
def r8f(label, model, micro, norm, s, exp_prefix="R8_f", epochs=None):
    exp = f"{exp_prefix}_{norm}"
    extra = F10 if epochs is None else F10.replace("epochs=100", f"epochs={epochs}")
    return dict(kind=f"R8f_{label}", est_min=40, done=f"results/{exp}/{label}/seed_{s}",
                cmd=f"{{py}} -m hypershift.run --config configs/think_nyse.yaml --set exp={exp} norm={norm} {extra} batch_days=8 label={label} model={model} micro_batch_days={micro} --seeds {s}")
S11 = [r8f(l, m, mi, n, s) for s in range(5) for n in ("train", "paper") for l, m, mi in (("RSR_I", "rsr_i", 2), ("STHGCN", "sthgcn", 4))]
S11S = [r8f(l, m, mi, "train", 0, "R8_f_smoke", epochs=1) for l, m, mi in (("RSR_I", "rsr_i", 2), ("STHGCN", "sthgcn", 4))]
# Preset "5s": smoke test of each job type (1 epoch), separate exp name.
S5S = [ks("group", a, "relative", 0, epochs=1, exp="ks_smoke", est_min=2) for a in ("HH_hyper", "EH_hyper", "EE_hyper", "HH_none", "EE_none")]
COMMANDS = {"1": S1, "2": S2, "3": S3, "4": S4, "5": S5, "6": S6, "7": S7, "5s": S5S, "8": S8, "9": S9, "10": S10, "10s": S10S, "11": S11, "11s": S11S, "8s": S8S, "all": S1 + S2, "custom": []}[SESSION]
print(len(COMMANDS), "commands;", sum(1 for c in COMMANDS if (REPO / c["done"] / "metrics.json").exists()), "already complete")

# %% Cell 7: run with N_WORKERS, time guard
LIMIT_MIN = SESSION_LIMIT_H * 60
est = {c["kind"]: c["est_min"] for c in COMMANDS}
observed = {}
lock, procs, stop = threading.Lock(), set(), threading.Event()
q = queue.Queue()
for c in COMMANDS:
    q.put(c)
log_lines, skipped = [], []

def say(msg):
    line = f"[{(time.time() - T0) / 3600:5.2f} h] {msg}"
    print(line, flush=True)
    log_lines.append(line)

def elapsed_min():
    return (time.time() - T0) / 60

def run_one(c, gpu):
    done = REPO / c["done"]
    name = c["done"].replace("results/", "").replace("/", "_")
    cmd = c["cmd"]
    if DRYRUN:
        if "hypershift.run" in cmd:
            say(f"DRYRUN skip (no --dry-run flag): {cmd}")
            return
        cmd += " --dry-run"
    for attempt in range(1 + RETRIES):
        t = time.time()
        with open(REPO / "results/logs" / f"{name}.log", "ab") as lf:
            p = subprocess.Popen([PY if a == "{py}" else a for a in shlex.split(cmd)], cwd=REPO, env=dict(ENV, CUDA_VISIBLE_DEVICES="-1" if DRYRUN else str(gpu)), stdout=lf,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            with lock:
                procs.add(p)
            p.wait()
            with lock:
                procs.discard(p)
        dur = (time.time() - t) / 60
        ok = (done / "metrics.json").exists() or DRYRUN
        say(f"{'OK  ' if ok else 'FAIL'} rc={p.returncode} {dur:5.1f} min gpu{gpu} {name}")
        if ok:
            with lock:
                observed.setdefault(c["kind"], []).append(dur)
                est[c["kind"]] = max(observed[c["kind"]])
            return
        (done / "failed.json").unlink(missing_ok=True)   # run_grid exits 0 after writing failed.json; clear it and retry
        if stop.is_set() or elapsed_min() + est[c["kind"]] * 1.2 > LIMIT_MIN - SAFETY_MIN:
            return

def worker(i):
    gpu = i % NGPU
    while not stop.is_set():
        try:
            c = q.get_nowait()
        except queue.Empty:
            return
        if (REPO / c["done"] / "metrics.json").exists() and not DRYRUN:
            continue
        if elapsed_min() + est[c["kind"]] * 1.2 > LIMIT_MIN - SAFETY_MIN:
            skipped.append(c["done"])        # may still fit a shorter kind later in the queue
            continue
        try:
            run_one(c, gpu)
        except Exception as e:                  # never let one broken command kill the worker
            say(f"ERROR launching {c['done']}: {e!r}")

def killer():
    while not stop.is_set():
        if elapsed_min() > LIMIT_MIN - 15:
            say("HARD STOP: terminating running processes")
            stop.set()
            with lock:
                for p in list(procs):
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except Exception:
                        pass
            return
        time.sleep(20)

threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_WORKERS)]
threading.Thread(target=killer, daemon=True).start()
for t in threads:
    t.start()
for t in threads:
    t.join()
stop.set()
say(f"finished: {len(skipped)} commands skipped by the time guard (start another session; everything is resumable)")
(REPO / "results/logs/kaggle_driver.log").write_text("\n".join(log_lines) + "\nskipped:\n" + "\n".join(skipped) + "\n")

# %% Cell 8: zip results for download
out = WORK / f"results_{TAG}.zip"
n_done = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted((REPO / "results").rglob("*")):
        if f.is_file():
            zf.write(f, f.relative_to(REPO).as_posix())
            n_done += f.name == "metrics.json"
print(f"{out}: {out.stat().st_size / 1e6:.1f} MB, {n_done} complete runs (metrics.json)")
if CLEAN_WORKING and not DRYRUN:
    shutil.rmtree(REPO, ignore_errors=True)
