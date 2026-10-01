# Running Hypershift experiments on Kaggle

Free Kaggle GPU for the heavy runs, resumable, same run-folder layout (`results/<exp>/<label>/seed_<k>/`), so results merge straight into the local `results/`.

Checked against Kaggle sources on 2026-10-01 (search results quoting the Kaggle docs and product posts; the docs pages themselves did not render for my fetch tool, so re-read the limits on your account page):

- GPU sessions are limited to 12 h (TPU 9 h); weekly GPU quota is 30 h "or sometimes higher", resets Saturday 00:00 UTC. The quota counts wall-clock session time, so running several training processes in one session is the way to get more out of it.
- Accelerators: 1x P100 or 2x T4, each with 4 CPU cores and 29 GB RAM. T4 x2 counts against the same quota.
- Private datasets: archives (zip, gz) you upload are auto-extracted and the size limits apply after extraction. Ours is about 222 MB as uploaded; if Kaggle auto-extracts the `.gz` relation tensors it grows to about 4.8 GB (2.6 + 0.8 NYSE, 0.8 + 0.36 NASDAQ), still far below the limits.
- **P100 risk**: newer Kaggle images ship a torch build without `sm_60` kernels, so a P100 can fail with "CUDA capability sm_60 is not compatible" ([Kaggle/docker-python#1546](https://github.com/Kaggle/docker-python/issues/1546)). Use **T4 x2** (launch.sh does by default). The notebook runs a CUDA matmul test first and stops with a clear message instead of failing hours later.
- The `kaggle` CLI (v2.2.4, tested only up to `--help`; no token available here) takes `--accelerator NvidiaTeslaT4` (= GPU T4 x2) on `kernels push` and `kernel-metadata.json` accepts `enable_gpu`, `enable_internet`, `dataset_sources`, `machine_shape` ([kernel metadata docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)).

## What is in this folder

| File | Purpose |
|---|---|
| `build_bundle.py` | Builds `kaggle/build/hypershift-code` (src, scripts without queues, configs, pyproject; about 0.2 MB) and `kaggle/build/hypershift-rsr-data` (about 222 MB: NYSE and NASDAQ price CSVs, ticker lists, wiki csv, hypergraph cache v2, gzipped relation tensors for RSR-I). Writes `dataset-metadata.json` for each (datasets are private by default). Refuses to build if the hypergraph cache version differs from `HYPERGRAPH_CACHE_VERSION` in the code. |
| `run_kaggle.py` / `run_kaggle.ipynb` | The notebook (the .ipynb is generated from the .py by `make_notebook.py`; edit the .py). 8 cells: parameters, copy code and install, data layout, optional prior results, GPU check, `COMMANDS`, parallel runner with time guard, zip. |
| `launch.sh`, `fetch.sh` | CLI flow: upload datasets, push and start the notebook; check status, download and merge the result. |
| `merge_results.sh` / `.py` | Merge a results zip into local `results/` without overwriting. |
| `verify_configs.py` | Prints the resolved configs (eq14, paper protocol, exp names) and the hypergraph cache check; used to prove bundle and repo resolve identically. |

## Presets (`SESSION` in the notebook, or `launch.sh <session>`)

- `1`: R8 small scale (`POC_sectors_R8_{rsr_i,sthgcn}_g2`, 10 seeds each) + EE and HE on full NYSE into `R5_g2`, seeds 0-9, paper protocol (the `phase1_gpu_3.sh` commands).
- `2`: first EH on full NYSE into `R5_g2`, seeds 0-24, paper protocol (the `phase1_gpu_3.sh` EH command, one seed per command; seeds with `results/R5_g2/EH/seed_<k>/metrics.json` locally when the notebook is generated are listed in `EH_DONE_LOCAL` and skipped), then R8 full NYSE `R8_baselines_g2`, RSR_I and STHGCN, seeds 0-9 (the `phase1_gpu_r8.sh` commands, without its VRAM gate). 45 runs.
- `all`: both in one session. `custom`: fill `COMMANDS` yourself (Cell 6).

Time estimates, **unmeasured on Kaggle** (serial = one process at a time at local RTX 3050 speed): THINK-type 21 s/epoch x 100 = about 35 min per run (measured locally); RSR-I/STHGCN 17-40 min per run (estimate in `docs/phase1/R8_baselines.md`, not measured); small runs 1-3 min.

| Preset | Runs | Serial time | Wall time with 3 workers (if they scale 2-2.5x; THINK is launch/CPU bound, so it should scale on 4 cores) |
|---|---|---|---|
| 1 | 20 small + 20 full | about 12-13 h | about 5-6 h |
| 2 | 25 EH + 20 R8 (full) | 25 EH x 20-27 min + 20 R8 x 14-32 min = 13-33 h of process time | about 4.3-7.3 h, central about 5.8 h (3 workers; EH 23 min = 1.8x the measured s1 EE 13 min, from local EH 29 vs EE 16 min; R8 per-run time still unmeasured) |
| all | 60 | 18-26 h | about 8-12 h, borderline |

Session 1 runs EE/HE first (time critical), then the short small-scale R8 runs fill the tail. The table is a best case: Kaggle's "4 CPU" may be 2 physical cores, so per-process speed can be below the 3050 and 3-worker scaling below 2x; use `N_WORKERS = 2` if the first completions show slowdowns. Run preset 1, then preset 2 (about 8-12 h of the 30 h weekly quota in total). The time guard does not start a run that cannot finish before `12 h - 25 min`, kills leftovers 15 min before the limit, then zips; anything skipped is picked up by re-running (resumable, via `results_*.zip` as a prior-results input, or by merging and re-launching).

## Steps for you

1. Create a Kaggle account, and **verify your phone** (Settings, Phone verification). GPU and Internet access are locked until you do.
2. Get API access (only for the CLI route):
   - https://www.kaggle.com/settings/api, "Create New Token" (or the legacy "Create Legacy API Key", which downloads `kaggle.json`).
   - Windows location: `C:\Users\Hi\.kaggle\kaggle.json` (legacy) or `C:\Users\Hi\.kaggle\access_token` (new token text only). Alternatively set the env var `KAGGLE_API_TOKEN`. With only an `access_token`, also `export KAGGLE_USER=<your username>`.
   - Nothing to install by hand: `launch.sh` creates a private venv `kaggle/.venv-kaggle` with `pip install kaggle` (the project `.venv` is not touched; it runs the CLI as `python -m kaggle.cli`).
3. **CLI route (recommended)**, from the repo root in Git Bash:
   ```bash
   bash kaggle/launch.sh 1 s1      # build, upload datasets, push + start the notebook (session 1, tag s1)
   bash kaggle/fetch.sh s1         # status; when complete downloads results_s1.zip into kaggle/build/out_s1/
   bash kaggle/merge_results.sh kaggle/build/out_s1/results_s1.zip --dry-run   # then without --dry-run
   ```
   Later: `bash kaggle/launch.sh 2 s2`. The first launch uploads about 0.2 GB; the data dataset is not re-uploaded unless `FORCE_DATA=1`. Other knobs: `ACCEL=NvidiaTeslaT4` (default), `PRIOR_DATASET=<user>/<slug>`, `TIMEOUT_S`.
   `kernels push` starts the run immediately in the background, so you can close the browser. If the CLI complains about `--dir-mode zip`, `--accelerator` or auth, tell the orchestrator the message.
4. **Web route (no CLI)**:
   1. Build the bundle: `.venv/Scripts/python.exe kaggle/build_bundle.py --username <your kaggle username>`.
   2. Kaggle, Datasets, New Dataset: drag in the folder content of `kaggle/build/hypershift-code`, title `hypershift-code`, visibility Private, Create. Repeat for `kaggle/build/hypershift-rsr-data` titled `hypershift-rsr-data` (drag the folder, subfolders are kept).
   3. Notebooks, New Notebook, File, Import Notebook, upload `kaggle/run_kaggle.ipynb`. Add Input: both datasets (search "hypershift"). Settings: Accelerator **GPU T4 x2** (not P100), Internet On (only needed to pip-install a missing package; the run itself works offline), Persistence none.
   4. Edit the first cell if needed (`SESSION = "1"`, `TAG = "s1"`), then Save Version, "Save & Run All (Commit)" (runs in the background, up to 12 h) or Run All interactively (browser must stay open).
   5. When it finishes: the notebook page, Output tab, download `results_s1.zip`; then `bash kaggle/merge_results.sh results_s1.zip`.
5. Optional prior results: put an earlier `results_*.zip` in a dataset, attach it (or `PRIOR_DATASET=` with launch.sh); finished seeds are unpacked first and skipped.

## Merge rules (`merge_results.sh`)

Copies only complete runs (folder has `metrics.json`). Never overwrites: a local run with `metrics.json` is a reported CONFLICT (local kept); a local folder without it is reported LOCAL-PARTIAL and left alone (a local job may be writing). Kaggle runs without `metrics.json` are listed as KAGGLE-INCOMPLETE and ignored. Kaggle logs land in `results/logs_kaggle/`. Always try `--dry-run` first. A LOCAL-PARTIAL with no live local job is stale (for example a killed run): delete that one folder and merge again. Locally `POC_sectors_R8_rsr_i_g2/rsr_i/seed_0` is already complete (from the stopped R8 queue), so session 1 skips it and the merge would report it as a conflict if it were included.

## Avoiding duplicate work with the local queue

The local queue (`phase1_gpu_3.sh`) runs THINK, then EH, then EE/HE. It will reach EE/HE roughly 10 h after 2026-10-01 morning. The local R8 queue has been stopped (R8 is Kaggle's job now). So:

- Merge the Kaggle EE/HE results into `results/R5_g2` **before** the local queue reaches EE/HE: the local run skips every seed that already has `metrics.json`.
- If not merged in time, both machines compute those seeds. The merge then keeps the local copy and reports the conflict (results differ in the last digits between a 3050 and a T4, so never mix: pick one source per seed, which the merge already does).
- A committed kernel's output is only downloadable once the whole version finishes, so merging before the local queue reaches EE/HE needs Kaggle's queue wait plus all of session 1 to end within about 9 h, which is not guaranteed. Safer: stop the local queue after EH and let Kaggle do EE/HE (the orchestrator's call).

## Risks

- Kaggle torch/CUDA: the package does not pin or install torch (`pyproject.toml` lists none), the notebook uses the image's torch. Unverified on a real Kaggle GPU: T4 should work, P100 may not (see above). `FIX_TORCH_FOR_OLD_GPU = True` reinstalls torch cu126 if Internet is on.
- Numerics differ between GPUs and torch versions (different CUDA kernels): Kaggle seeds are valid seeds, not bit-copies of local ones. Do not compare one local seed against the same Kaggle seed expecting equality.
- 12 h limit vs run time: see the table; the guard bounds the damage, `all` may not fit.
- Per-process speed on Kaggle's CPU is unmeasured; the guard replaces its initial per-run estimate by the longest observed duration of each kind.
- `results/tuned.json` and `configs/chosen.yaml` are not shipped; the notebook asserts they are absent, as locally. `configs/global.yaml` (`epochs: 60, batch_days: 8`) is shipped and overridden by the explicit `--set` as in the local queue.
- If Kaggle's dataset upload leaves `.gz` or `.zip` unextracted, the notebook handles both.
