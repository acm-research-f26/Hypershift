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
SESSION_LIMIT_H = 12.0   # PARAM  launch guard horizon in hours (12 = Kaggle hard limit; launch.sh sets it per preset: p1f 6, r5f2 8, smokes 1)
KERNEL_TIMEOUT_H = 12.0  # PARAM  kernel `-t` timeout in hours (launch.sh sets limit + 0.5 h for presets with in-kernel analysis); bounds the analysis time
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
if SESSION.startswith("wf"):      # Phase 1.5c: the compact Alpaca panel lives in its own small dataset (hypershift-alpaca-data)
    pn = sorted(INPUT.rglob("alpaca_panel_2016_2023.npz"))
    assert pn or DRYRUN, "alpaca_panel_2016_2023.npz not found under /kaggle/input: attach the hypershift-alpaca-data dataset"
    if pn and not DRYRUN:
        shutil.copy2(pn[0], DATA / "alpaca_panel_2016_2023.npz")
print("data ready:", sorted(p.name for p in (target / "data").iterdir()))

# %% Cell 4: optional prior results (a dataset OR a mounted kernel output holding results_*.zip; never overwrites existing files)
# Kernel sources mount at /kaggle/input/notebooks/<user>/<slug>/results_<tag>.zip (seen in hypershift-run-r5f-an-smoke's log: "result zips: ['notebooks/tomphamdustry/hypershift-run-r5f-smoke/results_r5f_smoke.zip', ...]"),
# datasets at /kaggle/input/<slug>/; the rglob below finds both. Only COMPLETE runs (seed_<k>/ with metrics.json inside the zip) are extracted:
# a run cut off by the previous kernel's hard stop is left out so it is rerun cleanly (its history.jsonl is never half-reused).
(REPO / "results/logs").mkdir(parents=True, exist_ok=True)

def extract_prior(z):
    """Extract complete runs and exp-level files of results zip z into REPO/results. Returns (runs_extracted, partial_skipped, files)."""
    with zipfile.ZipFile(z) as zf:
        names = [m for m in zf.namelist() if m.startswith("results/") and not m.endswith("/") and not m.startswith("results/logs")]
        def run_dir(m):
            parts = m.split("/")
            for i, s in enumerate(parts[:-1]):
                if s.startswith("seed_"):
                    return "/".join(parts[:i + 1])
            return None
        complete = {run_dir(m) for m in names if m.endswith("/metrics.json")} - {None}
        partial = {run_dir(m) for m in names} - complete - {None}
        n = 0
        for m in names:
            rd = run_dir(m)
            if (rd is None or rd in complete) and not (REPO / m).exists():
                try:
                    zf.extract(m, REPO)
                    n += 1
                except Exception as e:                       # one bad member must not stop the rest
                    print("  could not extract", m, repr(e))
    return len(complete), len(partial), n

for z in sorted(INPUT.rglob("results_*.zip")):
    try:
        runs, part, n = extract_prior(z)
        print("prior results", z.relative_to(INPUT), "->", runs, "complete runs,", part, "partial runs skipped,", n, "files extracted")
    except (zipfile.BadZipFile, OSError) as e:              # a truncated or unreadable zip: carry on with whatever else is mounted
        print("prior results", z, "UNREADABLE, ignored:", repr(e))

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
# Preset "r8f-top" (Phase 1.5 F top-up): the SAME command list as preset 11. launch.sh mounts the r8f kernel output (kernel_sources), Cell 4 unpacks its complete runs,
# the worker skips every command whose metrics.json exists, so only the missing R8_f_train / R8_f_paper seeds 0-4 are trained. Worst case (nothing mounted) = a full r8f rerun.
# "r8f-top-s": smoke of the mount + skip + train path: mounts hypershift-run-r8f-smoke (its seed 0 is already complete -> must be skipped), trains seed 1 for 1 epoch.
S_R8F_TOP = S11
S_R8F_TOP_S = [r8f(l, m, mi, "train", s, "R8_f_smoke", epochs=1) for s in (0, 1) for l, m, mi in (("RSR_I", "rsr_i", 2), ("STHGCN", "sthgcn", 4))]

# Preset "p1f" (Phase 1.5 F, small scale): the Phase 1 309-stock g2 arms rerun with the F fix. Single factor vs Phase 1 `rel_g2`: weight_decay 5e-4 -> 0 (+ log_ic, diagnostic only).
# Phase 1 rel_g2 already used input_mode=relative, 30 epochs, patience 10, batch_days 8, lr 1e-3, alpha 1, seeds 0-9: all unchanged here. New exp names carry the `_f` suffix.
# R7 (NASDAQ 3-class F1) is rerun likewise: input_mode=relative + weight_decay=0 vs Phase 1 E11_clf_g2 (level, wd 5e-4), 25 seeds x HH/EH/EE.
P1F_FIX = "weight_decay=0 log_ic=true"
POC_GEO = ("HH", "EE", "EH")      # poc_sectors.py loops geometries in GEOMS order, structures hyper, clique, none
POC_STR = ("hyper", "clique", "none")
def poc(kind, variant, arms, s, est_min, extra="", epochs=30):
    """One poc_sectors.py command = one seed over `arms`; `done` = the last arm in poc_sectors' loop order (arms run sequentially, so it implies the others)."""
    last = max(arms, key=lambda a: (POC_GEO.index(a[:2]), POC_STR.index(a[3:])))
    return dict(kind=kind, est_min=est_min, done=f"results/POC_sectors_{variant}/{last}/seed_{s}",
                cmd=f"{{py}} scripts/poc_sectors.py run --variant {variant} --input-mode relative --arms {' '.join(arms)} --seeds {s} --epochs {epochs} --set {P1F_FIX}{' ' + extra if extra else ''}")
P1F_CHEAP = ("HH_hyper", "HH_none", "EE_hyper", "EE_clique", "EE_none", "EH_hyper")   # ~9 min/seed on the laptop (Phase 1 sec/epoch x epochs run)
P1F_CLIQUE = ("HH_clique", "EH_clique")                                                  # ~13 min/seed (clique = 4897 pairs)
def r8_small_f(m, s, variant_suffix="_f", epochs=30):
    return dict(kind=f"P1F_R8small_{m}", est_min=3, done=f"results/POC_sectors_R8_{m}_g2{variant_suffix}/{m}/seed_{s}",
                cmd=f"{{py}} scripts/poc_sectors.py run --variant R8_{m}_g2{variant_suffix} --input-mode relative --seeds {s} --epochs {epochs} --set model={m} {P1F_FIX}")
def clf_f(arm, first, n, exp="E11_clf_g2_f", est_min=None, epochs=None):
    est = est_min or {"HH": 28, "EH": 18, "EE": 6}[arm]
    return dict(kind=f"P1F_clf_{arm}", est_min=est, done=f"results/{exp}/{arm}/seed_{first + n - 1}",
                cmd=f"{{py}} scripts/run_clf.py --exp {exp} --arms {arm} --first-seed {first} --seeds {n} --set input_mode=relative weight_decay=0" + (f" epochs={epochs}" if epochs else ""))
P1F = ([poc("P1F_poc_cheap", "rel_g2_f", P1F_CHEAP, s, 12) for s in range(10)] +
       [poc("P1F_poc_clique", "rel_g2_f", P1F_CLIQUE, s, 17) for s in range(5)] +
       [clf_f(a, c * 5, 5) for c in range(5) for a in ("HH", "EH", "EE")] +
       [r8_small_f(m, s) for s in range(10) for m in ("rsr_i", "sthgcn")] +
       [poc("P1F_poc_clique", "rel_g2_f", P1F_CLIQUE, s, 17) for s in range(5, 10)])
# "p1f-s": 1 epoch, seed 0, one command of every job type (poc multi-arm, clique, RSR-I small, clf) under *_smoke exps.
P1F_S = [poc("P1F_poc_cheap", "rel_g2_f_smoke", ("HH_hyper", "EH_hyper", "EE_none"), 0, 3, epochs=1),
         poc("P1F_poc_clique", "rel_g2_f_smoke", ("HH_clique",), 0, 3, epochs=1),
         r8_small_f("rsr_i", 0, "_f_smoke", 1),
         clf_f("HH", 0, 1, "E11_clf_g2_f_smoke", 3, epochs=1), clf_f("EH", 0, 1, "E11_clf_g2_f_smoke", 3, epochs=1)]

# Preset "r5f2" (Phase 1.5 F, full NYSE): the r5f config (F fix: relative inputs, weight_decay 0, log_ic, 100 epochs, patience 1000, norm=train) plus ONE extra switch per exp:
#   R5_f2_alpha0_train: alpha=0 (loss = MSE only; the loss form and alpha are INFERRED, not in the paper)   R5_f2_resid_train: spatial_residual=true (DEPARTURE from eq. 15, p. 851)
# HH (THINK) and EH (TConv+DHHAN) get identical switches and seeds 0-4. Seeds outermost so a cut-off session leaves paired rows. Compare with R5_f_train (r5f) locally after merging.
F2_VARIANTS = {"alpha0": "alpha=0", "resid": "spatial_residual=true"}
def r5f2(arm, var, s, exp_prefix="R5_f2", epochs=None):
    grid, base = ("E1_main", "THINK_paperProtocol") if arm == "HH" else ("E2_geometry", arm)
    exp = f"{exp_prefix}_{var}_train"
    extra = F10 if epochs is None else F10.replace("epochs=100", f"epochs={epochs}")
    return dict(kind=f"R5f2_{arm}_{var}", est_min=35 if arm == "HH" else 30, done=f"results/{exp}/{arm}/seed_{s}",
                cmd=f"{{py}} scripts/run_grid.py {grid} --labels {base} --seeds {s} --set exp={exp} norm=train {extra} {F2_VARIANTS[var]} label={arm}")
R5F2 = [r5f2(a, v, s) for s in range(5) for v in F2_VARIANTS for a in ("HH", "EH")]
R5F2_S = [r5f2("HH", "alpha0", 0, "R5_f2s", epochs=1), r5f2("EH", "resid", 0, "R5_f2s", epochs=1),
          r5f2("EH", "alpha0", 0, "R5_f2s", epochs=1), r5f2("HH", "resid", 0, "R5_f2s", epochs=1)]

# Presets "r5f3h" / "r5f3e" (Phase 1.5b, post-2017 plan R2/R3): replicate R5_f2_alpha0_train/HH (resp. /EH) -- identical resolved config (F10 + alpha=0, norm=train, batch_days 8 from
# the shipped configs/global.yaml) -- under the NEW exp R5_f3_alpha0_train with save_weights=true (best_state.pt + epoch_preds/). Seeds 0-4. HH and EH run as two parallel kernels.
# Smoke twins (-s): exp R5_f3s_alpha0_train, 1 epoch, seed 0, same flags. Uses only pre-2017 data (train/val selection); nothing here scores 2018+.
def r5f3(arm, s, exp="R5_f3_alpha0_train", epochs=None):
    grid, base = ("E1_main", "THINK_paperProtocol") if arm == "HH" else ("E2_geometry", arm)
    extra = F10 if epochs is None else F10.replace("epochs=100", f"epochs={epochs}")
    return dict(kind=f"R5f3_{arm}", est_min=40 if epochs is None else 3, done=f"results/{exp}/{arm}/seed_{s}",
                cmd=f"{{py}} scripts/run_grid.py {grid} --labels {base} --seeds {s} --set exp={exp} norm=train {extra} alpha=0 save_weights=true label={arm}")
R5F3H = [r5f3("HH", s) for s in range(5)]
R5F3E = [r5f3("EH", s) for s in range(5)]
R5F3H_S = [r5f3("HH", 0, "R5_f3s_alpha0_train", epochs=1)]
R5F3E_S = [r5f3("EH", 0, "R5_f3s_alpha0_train", epochs=1)]

# Presets "wfh1/2/3" and "wfe1/2/3" (Phase 1.5c walk-forward, docs/phase1_5c/SPEC.md): exp WF_<test year>_alpha0, arm HH (resp. EH), settings identical to R5_f3_alpha0_train
# (F10 + alpha=0, norm=train, batch_days 8 from configs/global.yaml) plus wf_test_year=<year> (Alpaca panel, expanding train window, val = year-1) and save_weights=true.
# est_min scales with the train windows (about 500/754/1006/1258/1510 for 2019..2023) + eval on val and test: 40 min = the 756-window R5 run. Heavy years first (LPT).
# Seeds: wfh1 = 0,1; wfh2 = 2,3; wfh3 = 4 (about 257 est-min per seed, so a 12 h kernel is never at risk). Smoke "wfh-s"/"wfe-s": test years 2023 and 2019, 1 epoch, exp WFs_<year>_alpha0.
WF_EST = {2019: 29, 2020: 40, 2021: 53, 2022: 62, 2023: 73}
def wf(arm, year, s, exp=None, epochs=None):
    grid, base = ("E1_main", "THINK_paperProtocol") if arm == "HH" else ("E2_geometry", arm)
    exp = exp or f"WF_{year}_alpha0"
    extra = F10 if epochs is None else F10.replace("epochs=100", f"epochs={epochs}")
    return dict(kind=f"WF_{arm}_{year}", est_min=WF_EST[year] if epochs is None else 6, done=f"results/{exp}/{arm}/seed_{s}",
                cmd=f"{{py}} scripts/run_grid.py {grid} --labels {base} --seeds {s} --set exp={exp} norm=train {extra} alpha=0 save_weights=true label={arm} wf_test_year={year}")
def wf_set(arm, seeds):
    return [wf(arm, y, s) for y in sorted(WF_EST, reverse=True) for s in seeds]
WFH = {"wfh1": wf_set("HH", (0, 1)), "wfh2": wf_set("HH", (2, 3)), "wfh3": wf_set("HH", (4,))}
WFE = {"wfe1": wf_set("EH", (0, 1)), "wfe2": wf_set("EH", (2, 3)), "wfe3": wf_set("EH", (4,))}
WFS = {"wfh-s": [wf("HH", y, 0, f"WFs_{y}_alpha0", epochs=1) for y in (2023, 2019)],
       "wfe-s": [wf("EH", y, 0, f"WFs_{y}_alpha0", epochs=1) for y in (2023, 2019)]}

# In-kernel analysis (Cell 7b; runs after training, BEFORE zipping; the md goes into the zip root, which scripts/overnight/merge_zip.py copies to docs/phase1_5/,
# and stays next to the zip in /kaggle/working). A failure never blocks the zip. steps = [(title, command, output md)].
def _q(p):                                  # POSIX-style quoted path (Kaggle is Linux; forward slashes also keep the local Windows simulation working under shlex)
    return shlex.quote(Path(p).as_posix())
def _r5f_an(prefix, out, note, boot):
    return (prefix, f"{{py}} scripts/r5f_analysis.py --root {_q(REPO)} --r5-prefix {prefix} --r8-prefix R8_f2_none --norms train --draws 5 --boot {boot} --note {shlex.quote(note)} --out-md {_q(TEMP / out)} --out-json {_q(TEMP / (out + '.json'))}", TEMP / out)
_F2NOTE = "preset r5f2 (kaggle/run_kaggle.py): r5f config (input_mode=relative weight_decay=0 log_ic=true epochs=100 patience=1000 norm=train) plus {} for both HH and EH, seeds 0-4. No R8 baselines in this run."
def _p1f_an(suffix, out):
    return ("p1f", f"{{py}} scripts/p1f_analysis.py --root {_q(REPO)} --suffix {suffix} --summarize --out-md {_q(TEMP / out)}", TEMP / out)
ANALYSIS = {
    "p1f": dict(md="F_p1f_results.md", steps=[_p1f_an("_f", "p1f.md")]),
    "p1f-s": dict(md="F_p1f_smoke_results.md", steps=[_p1f_an("_f_smoke", "p1fs.md")]),
    "r5f2": dict(md="F_r5f2_results.md", steps=[_r5f_an("R5_f2_alpha0", "a0.md", _F2NOTE.format("alpha=0 (exp R5_f2_alpha0_train)"), 3000),
                                                 _r5f_an("R5_f2_resid", "res.md", _F2NOTE.format("spatial_residual=true (exp R5_f2_resid_train)"), 3000)]),
    "r5f2-s": dict(md="F_r5f2_smoke_results.md", steps=[_r5f_an("R5_f2s_alpha0", "a0s.md", _F2NOTE.format("alpha=0 (SMOKE, 1 epoch)"), 200),
                                                         _r5f_an("R5_f2s_resid", "ress.md", _F2NOTE.format("spatial_residual=true (SMOKE, 1 epoch)"), 200)]),
}.get(SESSION)
# Preset "5s": smoke test of each job type (1 epoch), separate exp name.
S5S = [ks("group", a, "relative", 0, epochs=1, exp="ks_smoke", est_min=2) for a in ("HH_hyper", "EH_hyper", "EE_hyper", "HH_none", "EE_none")]
COMMANDS = {"1": S1, "2": S2, "3": S3, "4": S4, "5": S5, "6": S6, "7": S7, "5s": S5S, "8": S8, "9": S9, "10": S10, "10s": S10S, "11": S11, "11s": S11S, "8s": S8S, "all": S1 + S2, "custom": [],
            "r8f-top": S_R8F_TOP, "r8f-top-s": S_R8F_TOP_S, "p1f": P1F, "p1f-s": P1F_S, "r5f2": R5F2, "r5f2-s": R5F2_S,
            "r5f3h": R5F3H, "r5f3h-s": R5F3H_S, "r5f3e": R5F3E, "r5f3e-s": R5F3E_S, **WFH, **WFE, **WFS}[SESSION]
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

# %% Cell 7b: in-kernel analysis (only presets with an ANALYSIS entry; never blocks the zip)
ANALYSIS_MD = None
if ANALYSIS:
    try:
        parts = []
        for title, acmd, outp in ANALYSIS["steps"]:
            left = KERNEL_TIMEOUT_H * 3600 - (time.time() - T0) - 4 * 60          # keep 4 min for zipping
            if left < 60:
                say(f"analysis {title}: skipped, only {left:.0f} s left before the kernel timeout")
                continue
            cmd_l = [PY if a == "{py}" else a for a in shlex.split(acmd)]
            aenv = dict(ENV, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2")
            try:
                r = subprocess.run(cmd_l, cwd=REPO, env=aenv, capture_output=True, text=True, timeout=min(900, left))
                (REPO / "results/logs" / f"analysis_{title}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
                say(f"analysis {title}: rc={r.returncode}")
                print((r.stdout + r.stderr)[-1500:])
                if Path(outp).exists():
                    parts.append(Path(outp).read_text(encoding="utf-8"))
            except subprocess.TimeoutExpired:
                say(f"analysis {title}: TIMEOUT")
        if parts:
            ANALYSIS_MD = WORK / ANALYSIS["md"]
            ANALYSIS_MD.write_text(("\n\n---\n\n".join(parts)), encoding="utf-8")
            say(f"analysis written: {ANALYSIS_MD} ({ANALYSIS_MD.stat().st_size} bytes)")
    except Exception as e:                       # an analysis problem must never cost the results zip
        say(f"analysis FAILED (results are still zipped): {e!r}")

# %% Cell 8: zip results for download (+ the analysis md at the zip root, which scripts/overnight/merge_zip.py copies to docs/phase1_5/)
out = WORK / f"results_{TAG}.zip"
n_done = 0
(REPO / "results/logs/kaggle_driver.log").write_text("\n".join(log_lines) + "\nskipped:\n" + "\n".join(skipped) + "\n")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted((REPO / "results").rglob("*")):
        if f.is_file():
            zf.write(f, f.relative_to(REPO).as_posix())
            n_done += f.name == "metrics.json"
    if ANALYSIS_MD is not None and ANALYSIS_MD.exists():
        zf.write(ANALYSIS_MD, ANALYSIS_MD.name)
print(f"{out}: {out.stat().st_size / 1e6:.1f} MB, {n_done} complete runs (metrics.json)")
if CLEAN_WORKING and not DRYRUN:
    shutil.rmtree(REPO, ignore_errors=True)
