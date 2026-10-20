# THINK Depth Ablation: Two-Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether more THINK hypergraph message-passing depth (0/1/2/3 layers) adds out-of-sample ranking information. Run this on Ethan's 250-stock Alpaca 1-minute data now, without repeating any experiment Ethan already completed. Keep Rishi's Nasdaq track gated on its exact inputs.

**Architecture:** Track B (runnable) uses a new worktree from `origin/ethan`. It reuses Ethan's archive preparation and BASE10 features unchanged, vendors Rishi's `think_model.py` byte-for-byte, and adds a depth wrapper, fold supports, a resumable runner and a paired evaluator. Track A (Rishi Nasdaq, seven arms) keeps the 2026-10-19 plan. It gets one prerequisite check per execution session and records READY or BLOCKED.

**Tech Stack:** Python 3.12 (3.14 fallback), PyTorch 2.14 cu130, NumPy, pandas, exchange-calendars, pytest; laptop RTX 3050 4 GB (Kaggle fallback).

**Spec:** `docs/research_20261019/RESEARCH_AUDIT_AND_FIRST_BATCH.md` (Sections 3–7) plus the 2026-10-20 amendments below. Context: Ethan's `HYPERSHIFT_EXPERIMENT_TEAM_OVERVIEW_2026-10-20_v1.md` (Drive folder `1VSe5dDZKadUHRSQhXQHbDVItOu8NVgG4`), and `deep-research-report(2).md` (residual two-layer depth is literature-supported; dynamic edges, residual targets and listwise loss are later questions). Supersedes `2026-10-19-depth-ablation-research.md` for Track B. Track A keeps that plan's Tasks 2–6.

## Amendments, 2026-10-20 (evidence gathered while writing this plan)

- **Rishi's upload:** `feat-rs` 202f285 adds rank-depth and loss-similarity analysis and docs only. `think_model.py`, `research_models.py`, `robust_models.py`, `pit_research_engine.py` and `run_robust_research.py` are byte-identical to d83c4a4. There are no releases, and Drive holds nothing from Rishi. The Track A blocker (9 inputs, 64 baseline files) is unchanged.
- **Ethan's data is reachable.** Drive folder `1fRdFBwoAy9uehsj3cEvIEDiZEsBVp8Xr` is `local-data`; its `raw/alpaca/us_equities/1min/2021-09-26_2026-09-26/` holds `manifest.json` and `bars/<SYM>/<YYYY-MM>.csv.gz`. The data is SIP and unadjusted, from 2021-09-26 to 2026-09-26, with 250 symbols, 15,250 partitions, 127,690,000 bars and a sha256 for every partition. `C:\Projects\ACM Research\...` is Ethan's machine and does not exist here.
- **Ethan's completed work (do not repeat):**
  - **`four_day` market ablation** (1644/1644 fits, `experiment_complete` 2026-10-04): temporal-only T, eight relation families G/K/L/E/J/M/P/C and their combinations, per-family and all-family rewiring, uniform-all, frozen reference, feature packs R/RV/F, PH context, graph window ×0.5/×2, signed/covariance similarity. It covered 1m/15m/1h/1d, folds 2023/24/25, and 3 core / 2 contrast seeds.
  - **`hyperbolic_design_stockmixer` study:** 13 recipes (caution, identity init, tangent LayerNorm, learned curvature, hybrid), Q2 curved THINK, LN4, and the SM_HYB_LN score mixer. Stage C is still running.
  - Neither study varies message-passing depth.
- **Track B arms are therefore T0/H1/H2/H3 only.**
  - U1 (uniform), R1 (rewired) and M0 (mixer) are dropped on Ethan's data, because his uniform-all, rewire-* and SM_HYB_LN already answer them. The report cites his results.
  - T0 and H1 are not repeats of Ethan's T or Q2. They are the depth-0 and depth-1 anchors of one backbone: Rishi's THINK, with signed raw attention and no residual. Ethan's models use different backbones.

## Global Constraints

- **Branch and worktree.** Track B worktree is `C:/Users/Hi/Projects - Coding/Hypershift-depthB`, branch `research/depth-alpaca-20261020`, created from `origin/ethan` at `9990ca1`.
  - Never commit to `ethan`, `feat-rs` or the Codex worktree.
  - Large data and runs go under the worktree's git-ignored `local-data/` and `local-experiments/`. Verify they are ignored before the first write.
- **Python.** Use a venv `.venv-b` inside the worktree, never under `%TEMP%`.
  - Use Python 3.12 if Ethan's `data.*` and `experiments.market_ablation.dataset` import cleanly, otherwise 3.14.
  - Install torch 2.14 from the cu130 index.
  - CPU-only jobs set `CUDA_VISIBLE_DEVICES=-1`.
- **Model source.** Vendor Rishi's file to `depth_b/think_model.py`. It must have sha256 `ee05058dba2be85fc6bde6d768a5f6397e078108037717dc5512c5dac7da85bb`. Never edit it; tests assert the hash.
- **Data.** Use Ethan's 1-minute archive exactly as published. A partition counts only when its sha256 equals `manifest.json` `partitions[...]["sha256"]`.
  - No price adjustment.
  - No daily or multi-session target on these raw bars: the reverse-split artifacts that rejected Tom's Alpaca panel apply here too.
- **Universe.** Use Ethan's retrospective 2026-09-26 selection of 250 symbols, which is survivorship-biased. Every Track B result is **exploratory**. All evaluation dates were already inspected by Ethan.
- **Protected dates.** Use no origin or target at or after `2026-04-01T00:00:00-04:00` (local Study08 seal), for training or evaluation.
  - Ethan's `later_development64` cohort lies after this date, so it is excluded unless the user lifts the seal.
- **Folds** (Ethan's design-study cutoffs, UTC):
  - `earlier`: cutoff `2025-01-01T00:00:00Z`. Evaluation scope `earlier_OOF` is the sessions 2025-01-02 to 2025-06-25 (119 sessions).
  - `main`: cutoff `2025-06-26T00:00:00Z`. Evaluation scope `initial_validation` is the sessions 2025-06-26 to 2025-09-25 (64 sessions).
- **Origins.** An origin is a regular-session minute `t` with at least 20 completed in-session rows ending at `t`, and with `t+1` in the same session: 370 per full session and 190 on a half day. Training origins have target time ≤ cutoff.
- **Inputs and target.**
  - Inputs are Ethan's BASE10 values plus 10 validity masks: 20 channels × 20 rows, from `MarketDataset.batch` with feature pack `F`, standardized by the fold's `MarketDataset.moments`.
  - Model input is `clip(x, -8, 8) * 0.2` on all 20 channels.
  - The target is the next-minute log close return divided by the fold's TRAIN RMS, computed over eligible TRAIN targets.
  - Loss is masked MSE on target-eligible labels.
- **Architecture.** Rishi's hyperbolic THINK with hidden 16 and kernels `(4, 5)` (20 = 4×5).
  - `depth` ∈ {0,1,2,3} independent `HypergraphAttention(hidden, 250, groups, True, True)` blocks.
  - No residual, no mixer, no extra loss.
  - Build modules in the order `first`, `last`, `spatial[0..d-1]` from `torch.manual_seed(seed)`.
- **Graph.** Rishi's rule on training data only: each node plus its up to 4 highest positive correlations, with joint observations ≥ 30. The correlation uses in-session 1-minute log returns before the fold cutoff, computed with the masked formula in Task 4.
- **Training.**
  - Q32768 TRAIN origins, chosen as `train[floor(i*(N-1)/32767)]` for i = 0..32767.
  - 4 passes. Each pass is a permutation from `np.random.default_rng(seed + 700001)`, used successively.
  - Batch 8 origins, giving 16,384 updates.
  - AdamW lr 0.001, weight_decay 0.001, gradient clip 1.0, float32.
  - Fixed endpoint: no validation peeking and no early stopping.
- **Seeds:** 7, 19, 42, 61, 97. Pilot seeds are 7, 19, 42 on fold `earlier`.
- **Arms:** T0 (depth 0), H1, H2, H3. Total 4 arms × 5 seeds × 2 folds = 40 fits.
- **Primary metric:** per-origin Spearman RankIC across target-eligible stocks, averaged over seeds and then over origins.
  - Primary contrasts: H1−T0, H2−H1, H3−H1, separately per scope. Holm adjustment within each scope's 3 contrasts.
  - Practical threshold: ΔRankIC ≥ 0.005.
  - Verdict words: STRONG / SEED-ROBUST ONLY / NO EVIDENCE / INSUFFICIENT SEEDS.
- **Uncertainty.** Paired moving-block bootstrap over whole sessions with blocks of 5 and 10, 10,000 draws, RNG seeds 2026102005 / 2026102010. Seeds are averaged before resampling.
- **Out of scope for minute data:** portfolio, Sharpe, cost and turnover ledgers. Prediction metrics only.
- **Budget.** Track B ceiling: 6 GPU-hours and 3 GiB of new run output, excluding the prepared dataset of about 10 GB. Before the pilot, project the total from smoke timing; stop if the projection exceeds the ceiling.
- **Track A.** One `ablation_spec.py` prerequisite check per execution session, and no polling loops.
- **Process.** Before any GPU launch, run `nvidia-smi`. Confirm no other app holds VRAM, read the tail of `results/logs/overnight.log`, and leave the Codex screening CPU workers (`scripts/queues/run_screen_v3_w*.cmd`) untouched.

## Review Focus

1. **A truncated or HTML-error Drive download saved as `.csv.gz`.** Expected: the sha256 mismatch is rejected, retried with backoff, and never counted. Covered by the Task 1 test `test_fetch_rejects_bad_hash`.
2. **A history window or target crossing a session boundary,** for example an overnight split gap. Expected: the origin is excluded. Covered by the Task 3 test `test_origins_stay_in_session`.
3. **An inactive or NaN stock leaking into active scores through multi-hop propagation at depth 2–3.** Expected: active outputs are bit-identical when inactive inputs are perturbed. Covered by the Task 4 test `test_inactive_nodes_cannot_leak`.
4. **A resume after a crash diverging from an uninterrupted run.** Expected: predictions are identical. Covered by the Task 5 test `test_resume_matches_uninterrupted`.
5. **Arms scored on different origins or masks, or any date on or after 2026-04-01 used.** Expected: the evaluator refuses. Covered by the Task 6 test `test_evaluator_requires_common_support` and the Task 3 test `test_no_protected_dates`.

---

### Task 0: Track A single prerequisite check

**Files:**
- Create: `docs/research_20261020/TRACK_A_STATUS.json` (main repo, branch `tom-shlom`)

**Interfaces:**
- Consumes: Codex worktree `C:/Users/Hi/.codex/worktrees/ablation-depth-20261019/Hypershift`, CLI `python ablation_spec.py --data-root PATH --baseline-root PATH` (exit 0 = passed, 2 = missing).
- Produces: `{"checked_at": ISO, "feat_rs_head": sha, "passed": bool, "missing_inputs": int, "missing_baseline": int, "verdict": "READY"|"BLOCKED"}`.

- [ ] **Step 1:** `git fetch origin feat-rs` and record the head sha. If the head is beyond 202f285, run `git diff --stat 202f285 origin/feat-rs -- data runs` to look for newly tracked inputs.
- [ ] **Step 2:** In the Codex worktree, run `python ablation_spec.py` with the default roots and with any user-supplied path. Expected now: exit 2, 9 missing inputs, 64 missing baseline files.
- [ ] **Step 3:** Write `TRACK_A_STATUS.json`. If READY, append the line "Track A READY: execute 2026-10-19 plan Tasks 2–6 in the Codex worktree" to `docs/research_20261019/PROGRESS.md` and stop treating it as blocked. If BLOCKED, do not re-check in the same session.
- [ ] **Step 4:** Commit only `docs/research_20261020/TRACK_A_STATUS.json` with `git commit -m "docs(research): Track A prerequisite status 2026-10-20"`.

### Task 1: Worktree, environment and verified archive download

**Files:**
- Create: `depth_b/__init__.py`, `depth_b/archive_fetch.py`, `tests/depth_b/test_archive_fetch.py`
- Create (data, git-ignored): `local-data/raw/alpaca/us_equities/1min/2021-09-26_2026-09-26/{manifest.json,bars/...}`, `local-data/metadata/universes/us_equities_250/2026-09-26/{universe.csv,universe_notes.md}`, `local-data/drive_index_1min.json`

**Interfaces:**
- Produces:
  - `index_drive_folder(folder_id: str, cache: Path) -> dict[str, str]` maps relative paths such as `bars/AAPL/2024-03.csv.gz` to Drive file ids. It lists through `https://drive.google.com/embeddedfolderview?id=`, recurses one level into the symbol folders, and writes and reuses `cache`.
  - `fetch_partitions(index: dict[str, str], manifest: dict, dest: Path, workers: int = 2, max_retries: int = 5) -> dict` returns `{"verified": int, "failed": list[str], "skipped": int}`.
  - CLI: `python -m depth_b.archive_fetch --folder 1S746H5ItQskWnANWCCOnGSCrSGH-yrdJ --manifest-id 17CNga4Yr68knE7FP8MFEw-2E941-AMXT --dest <archive dir>`.
  - Fixed metadata ids: universe csv `1EXyA5BDsCPjuBGzE4G0wWZS4lX7UzlbX`, notes `1dkeQHEvXTlLT5QrCf_GkyPWqR_KFUL1k`.

- [ ] **Step 1:** Create the worktree with superpowers:using-git-worktrees (path and branch from Global Constraints). Check that `git check-ignore local-data/x local-experiments/x` prints both paths.
- [ ] **Step 2:** Create `.venv-b` (3.12) and install torch from `--index-url https://download.pytorch.org/whl/cu130`, then numpy, pandas, exchange-calendars, scipy and pytest. Verify with `.venv-b/Scripts/python.exe -c "import data.calendars, experiments.market_ablation.dataset, torch; print(torch.cuda.is_available())"`. If the import fails on syntax, recreate with `--python 3.14`.
- [ ] **Step 3:** Write the failing tests, using an injected `http_get(url) -> bytes` to fake the downloads:

```python
def test_fetch_rejects_bad_hash(tmp_path):
    manifest = {"partitions": {"bars/AAA/2024-01.csv.gz": {"sha256": sha(b"good")}}}
    calls = iter([b"<html>quota</html>", b"good"])
    report = fetch_partitions({"bars/AAA/2024-01.csv.gz": "id1"}, manifest, tmp_path, http_get=lambda u: next(calls))
    assert report["verified"] == 1 and report["failed"] == []
    assert (tmp_path / "bars/AAA/2024-01.csv.gz").read_bytes() == b"good"

def test_fetch_skips_verified_and_never_keeps_bad_bytes(tmp_path):
    # pre-existing verified file is skipped; a permanently bad file is reported failed and absent on disk
    ...
    assert report["skipped"] == 1 and report["failed"] == ["bars/BBB/2024-01.csv.gz"]
    assert not (tmp_path / "bars/BBB/2024-01.csv.gz").exists()
```

- [ ] **Step 4:** Run `pytest tests/depth_b/test_archive_fetch.py -v` and confirm it fails with ImportError.
- [ ] **Step 5:** Implement the functions. Write to `*.part`, verify sha256, then rename atomically. Back off for 2, 4, 8… s on HTTP 429/5xx or a hash mismatch. Download with `https://drive.google.com/uc?export=download&confirm=t&id=<id>`.
- [ ] **Step 6:** Run the tests. Expected: PASS.
- [ ] **Step 7:** Download `manifest.json` and the universe files, then run the CLI in the background (`run_in_background`). Expected at completion: `verified + skipped == 15250`, `failed == []`, and the summary `bars` total is 127,690,000.
- [ ] **Step 8:** Commit `depth_b/__init__.py`, `depth_b/archive_fetch.py` and the tests with `git commit -m "feat(depth-b): verified resumable Drive archive fetch"`.

### Task 2: Prepare Ethan's 1-minute dataset

**Files:**
- Create: `depth_b/prepare.py`, `tests/depth_b/test_prepare_config.py`
- Output (git-ignored): `local-experiments/runs/depth_b_20261020/dataset/1m/*.npy`

**Interfaces:**
- Consumes: Ethan's `experiments.market_ablation.config.SweepConfig` and `experiments.market_ablation.dataset.prepare, MarketDataset` (unchanged).
- Produces:
  - `RUN_ROOT = Path("local-experiments/runs/depth_b_20261020")`
  - `sweep_config() -> SweepConfig`, which returns `SweepConfig(output=str(RUN_ROOT), intervals=("1m",), lookback=20)`
  - `open_dataset() -> MarketDataset`, which returns `MarketDataset(sweep_config(), "1m")`

- [ ] **Step 1:** Write `test_prepare_config`. It asserts `sweep_config().intervals == ("1m",)`, `lookback == 20`, that the archive path ends with `1min/2021-09-26_2026-09-26`, and that `RUN_ROOT` is git-ignored.
- [ ] **Step 2:** Run it (FAIL), implement it, then run it again (PASS).
- [ ] **Step 3:** Run `CUDA_VISIBLE_DEVICES=-1 .venv-b/Scripts/python.exe -c "from depth_b.prepare import *; from experiments.market_ablation.dataset import prepare; prepare(sweep_config())"` in the background. Expected: `dataset/preparation.json` has `"complete": true`, and `metadata.json` has `adjustment: raw`.
- [ ] **Step 4:** Commit with `git commit -m "feat(depth-b): 1m dataset preparation via Ethan's pipeline"`.

### Task 3: Fold supports and the data gate

**Files:**
- Create: `depth_b/supports.py`, `tests/depth_b/test_supports.py`
- Output: `local-experiments/runs/depth_b_20261020/DATA_GATE.json`

**Interfaces:**
- Consumes: `open_dataset()`.
- Produces:
  - `PROTECTED_FROM = pd.Timestamp("2026-04-01", tz="America/New_York")`
  - `FOLDS = {"earlier": ("2025-01-01T00:00:00Z", "2025-01-02", "2025-06-25"), "main": ("2025-06-26T00:00:00Z", "2025-06-26", "2025-09-25")}`, each entry being cutoff, first evaluation session and last evaluation session
  - `in_session_origins(ds, lookback: int = 20) -> np.ndarray`
  - `fold_supports(ds, fold: str) -> dict[str, np.ndarray]` with keys `train`, `q` (32768), `eval`
  - `target_rms(ds, train: np.ndarray) -> float`
  - `data_gate(ds) -> dict`

**Expected gate values (from Ethan's overview):**

| Quantity | earlier | main |
|---|---:|---:|
| eval origins | 43,996 | 23,500 |
| eval sessions | 119 | 64 |
| eligible eval labels | 10,143,493 | 5,341,404 |
| TRAIN target RMS | 0.0007815717452377552 | 0.0008012143737238865 |
| eligible TRAIN origins (N for Q) | 302,288 (or 302,288 + 222 incl. D/P) | — |

- [ ] **Step 1:** Write the failing tests on a synthetic `MarketDataset`-shaped stub with 2 sessions (one of 390 minutes, one of 210) and 3 stocks:

```python
def test_origins_stay_in_session():
    o = in_session_origins(stub, lookback=20)
    assert len(o) == 370 + 190
    assert (stub.sessions[o] == stub.sessions[o + 1]).all()
    assert (stub.sessions[o - 19] == stub.sessions[o]).all()

def test_no_protected_dates():
    for fold in FOLDS:
        s = fold_supports(real_or_stub, fold)
        for key in ("train", "q", "eval"):
            assert (pd.to_datetime(ds.times[s[key] + 1], utc=True) < PROTECTED_FROM).all()

def test_q_is_chronological_rank_selection():
    train = np.arange(100_000)
    q = q_select(train)
    assert len(q) == 32768 and q[0] == 0 and q[-1] == 99_999 and (np.diff(q) > 0).all()
```

- [ ] **Step 2:** Run them (FAIL), implement, then run again (PASS). Eval origins are origins whose session date falls within the fold's evaluation range. TRAIN origins have `times[o+1] <= cutoff` and at least one target-eligible stock.
- [ ] **Step 3:** Run `data_gate` on the prepared data and write `DATA_GATE.json`. **PASS requires all of:**
  - every count in the table matches exactly
  - both RMS values agree to a relative difference of 1e-6 or less
  - the protected-date check holds
- [ ] **Step 4:** On a mismatch, diagnose for at most 2 hours. Candidates: origin rule, per-origin "any eligible" filter, half-days, RMS over channel 0 versus in-session targets. Record each hypothesis and its counts in `DATA_GATE.json`. If still mismatched:
  - set `"comparable_to_ethan": false`
  - ask the user to request Ethan's `scientific_definitions_v4.json` and eligibility arrays
  - continue with the declared rule, labelled "non-comparable to Ethan's tables"
  - never train on an unrecorded population
- [ ] **Step 5:** Commit `depth_b/supports.py` and its tests, plus a copy of `DATA_GATE.json` at `docs/depth_b_20261020/DATA_GATE.json`, with `git commit -m "feat(depth-b): fold supports and data gate"`.

### Task 4: Depth model, graph and diagnostics

**Files:**
- Create: `depth_b/think_model.py` (vendored verbatim from `git show d83c4a4:think_model.py`), `depth_b/model.py`, `tests/depth_b/test_model.py`

**Interfaces:**
- Consumes: `Temporal`, `HypergraphAttention`, `exp0`, `log0` from `depth_b.think_model`.
- Produces:
  - `class DepthThink(nn.Module)` with `__init__(self, groups: list[set[int]], depth: int, features: int = 20, nodes: int = 250, hidden: int = 16, kernels: tuple[int, int] = (4, 5))`, `forward(self, x, eligible, probe: bool = False)`. Here `x` is `[B,20,250,20]` and `eligible` is `[B,250]`. It returns `score [B,250]`, or `(score, states)` when `probe=True`, where `states` is the list of per-layer `[B,T',250,16]` tensors.
  - `to_model_input(x_flat: np.ndarray) -> np.ndarray` maps `MarketDataset.batch` x `[B,250,400]` to `[B,20,250,20]`, then applies `clip(-8, 8) * 0.2`.
  - `correlation_groups(returns: np.ndarray, valid: np.ndarray, k: int = 4, min_joint: int = 30) -> list[set[int]]`.
    - Let z be each stock's returns standardized over its own valid entries, with 0 where invalid, and m the validity mask.
    - `corr = (zᵀz) / sqrt((z²)ᵀm ⊙ mᵀ(z²))`, accumulated in float64 time chunks.
    - Pairs with `mᵀm < min_joint` are set to NaN.
    - Each node gets itself plus its top-k positive neighbours, with stable tie order.
  - `layer_diagnostics(states, eligible) -> dict` gives, per layer, the active-node variance and the mean pairwise cosine in log0 space, using detached tensors.

- [ ] **Step 1:** Write the failing tests:

```python
def test_vendored_hash():
    assert sha256(Path("depth_b/think_model.py").read_bytes()).hexdigest() == "ee05058dba2be85fc6bde6d768a5f6397e078108037717dc5512c5dac7da85bb"

def test_depth1_matches_rishi_reconstruction():
    torch.manual_seed(0); ref = ThinkReconstruction(20, 6, groups, 16, kernels=(4, 5))
    m = DepthThink(groups, depth=1, nodes=6)
    m.load_state_dict({k.replace("spatial.", "spatial.0."): v for k, v in ref.state_dict().items()})
    assert torch.equal(m(x, elig), torch.where(elig.bool(), ref(x, elig), 0))

def test_common_init_identical_across_depths():
    ms = [build(seed=7, depth=d) for d in range(4)]
    for name in ("first", "last"):
        assert all(torch.equal(p, q) for a in ms[1:] for p, q in zip(getattr(ms[0], name).parameters(), getattr(a, name).parameters()))
    assert all(torch.equal(p, q) for p, q in zip(ms[1].spatial[0].parameters(), ms[3].spatial[0].parameters()))

@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_inactive_nodes_cannot_leak(depth):
    x2 = x.clone(); x2[:, :, inactive] = torch.randn_like(x2[:, :, inactive]) * 50
    x2[:, 0, inactive, 0] = float("nan")
    a, b = model(x, elig), model(torch.nan_to_num(x2), elig)
    assert torch.equal(a[:, active], b[:, active])

@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_finite_forward_backward(depth): ...   # loss.backward(); all grads finite; scores finite

def test_batch_rows_independent(): ...          # perturbing batch row 1 leaves row 0 scores identical

def test_correlation_groups_train_only():
    g1 = correlation_groups(r[:cut], v[:cut]); r2 = r.copy(); r2[cut:] = rng.normal(size=r2[cut:].shape)
    assert g1 == correlation_groups(r2[:cut], v[:cut]) and all(len(s) <= 5 and i in s for i, s in enumerate(g1))
```

- [ ] **Step 2:** Run `pytest tests/depth_b/test_model.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. For depth 0, `forward` skips the spatial path. For depth ≥ 1, blocks are applied in sequence on the hyperbolic `u`, each with the same `eligible`. Masking matches `ThinkReconstruction.forward`. The final score is zeroed at ineligible nodes.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `depth_b/think_model.py`, `depth_b/model.py` and the tests with `git commit -m "feat(depth-b): depth-configurable THINK on vendored Rishi model"`.

### Task 5: Resumable runner, smoke and pilot

**Files:**
- Create: `depth_b/train.py`, `tests/depth_b/test_train.py`
- Output: `local-experiments/runs/depth_b_20261020/fits/<fold>/<arm>/seed_<s>/{config.json,groups.json,ckpt_last.pt,history.jsonl,diagnostics.jsonl,eval_pred.npz,complete.json}` and `.../RUN_MANIFEST.json`

**Interfaces:**
- Consumes: `open_dataset`, `fold_supports`, `target_rms`, `DepthThink`, `to_model_input`, `correlation_groups`, `layer_diagnostics`.
- Produces:
  - `ARMS = {"T0": 0, "H1": 1, "H2": 2, "H3": 3}`
  - `run_fit(arm: str, fold: str, seed: int, root: Path, device: str, max_updates: int | None = None) -> dict`. It returns the `complete.json` content: `{arm, fold, seed, updates, parameters, wall_s, gpu_s, peak_mem_mb, pred_sha256, groups_sha256, code_sha256}`.
  - `eval_pred.npz` holds `origins [E]` and `pred_raw [E,250] float32` (score × RMS).
  - The shared per-fold file `fits/<fold>/eval_support.npz` holds `origins`, `target_raw` and `mask`. It is written once, and each fit checks its hash, which keeps the 40 fits near 1.4 GB.
  - CLI: `python -m depth_b.train --stage smoke|pilot|full [--device cuda]`.
    - smoke: fold `earlier`, seed 7, every arm, `max_updates=256`, nothing written as complete.
    - pilot: fold `earlier`, seeds 7/19/42.
    - full: all remaining fits.
    - Fits that are already complete are skipped after their hashes are verified.

- [ ] **Step 1:** Write the failing tests on the Task 3 stub with a tiny Q (64 origins, 2 passes):

```python
def test_resume_matches_uninterrupted(tmp_path):
    a = run_fit("H2", "stub", 7, tmp_path / "a", "cpu")
    run_fit("H2", "stub", 7, tmp_path / "b", "cpu", max_updates=5)      # interrupted after a checkpoint
    b = run_fit("H2", "stub", 7, tmp_path / "b", "cpu")
    assert a["pred_sha256"] == b["pred_sha256"]

def test_changed_code_hash_refuses_resume(tmp_path): ...   # edit config.json code_sha256 -> RuntimeError
def test_complete_requires_all_eval_origins(tmp_path): ... # truncated eval_pred.npz -> not skipped, refit
def test_nonfinite_loss_writes_failed_json(tmp_path): ...  # injected NaN target -> failed.json, no complete.json
```

- [ ] **Step 2:** Run `pytest tests/depth_b/test_train.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `run_fit`.
  - Checkpoint `{model, optimizer, torch/numpy/python RNG states, pass, position, permutation}` atomically every 1024 updates and at the end.
  - Hold an exclusive lock file per fit directory.
  - Log the pre-clip gradient norm and loss every 256 updates.
  - Every 4096 updates and at the end, log `layer_diagnostics` and the attention-coefficient |α| quantiles (0.5/0.9/0.99 plus the nonfinite count) on a fixed probe of 128 TRAIN origins (`q[::256]`).
  - After training, forecast every eval origin once.
  - `RUN_MANIFEST.json` records the code, data-gate, manifest and universe hashes.
- [ ] **Step 4:** Run the tests. Expected: PASS. Commit with `git commit -m "feat(depth-b): resumable deterministic runner"`.
- [ ] **Step 5:** Run the smoke stage on the GPU after the Process checks. **Gates:** finite loss and gradients for all arms, peak memory ≤ 3,200 MB, and timing written to `SMOKE.json`. Project `40 × 16,384 × s_per_update(arm)` and stop if the total exceeds 6 GPU-hours (then switch to Kaggle per `kaggle/README.md`).
- [ ] **Step 6:** Run the pilot stage (12 fits) in the background. **Engineering gates only:** every fit complete, all predictions finite, no `failed.json`. Do not open RankIC before all 40 fits are complete.
- [ ] **Step 7:** Run the full stage (the remaining 28 fits) in the background. On a crash, relaunch the same command, which resumes.

### Task 6: Paired evaluator and report

**Files:**
- Create: `depth_b/evaluate.py`, `tests/depth_b/test_evaluate.py`
- Output: `docs/depth_b_20261020/{REPORT.md,metrics_by_seed.csv,contrasts.csv,diagnostics.csv,figures/*.png}` (worktree), copied to `docs/research_20261020/depth_b/` on `tom-shlom`

**Interfaces:**
- Consumes: every `complete.json` and `eval_pred.npz`.
- Produces:
  - `origin_metrics(pred, target, mask) -> dict[str, np.ndarray]` gives per-origin `rank_ic`, `ic`, `mse_raw`, `prec5`, `prec10`, `spread5`, `spread10` and `tie_frac`. Ranks use average ties; an origin with fewer than 2 eligible stocks is NaN and is excluded in every arm.
  - `paired_block_ci(diff_by_session: np.ndarray, block: int, draws: int, seed: int) -> tuple[float, float, float]` gives mean, 2.5% and 97.5%. It uses a non-circular moving block over sessions, with draws truncated to the original length.
  - `build_report(root: Path, out_dir: Path) -> None`.

- [ ] **Step 1:** Write the failing tests:

```python
def test_rank_ic_hand_case():
    m = origin_metrics(np.array([[1., 2, 3, 4]]), np.array([[.1, .2, .4, .3]]), np.ones((1, 4), bool))
    assert m["rank_ic"][0] == pytest.approx(0.8)

def test_identical_arms_zero_contrast():
    lo_mean, lo, hi = paired_block_ci(np.zeros(64), 5, 1000, 1)
    assert lo_mean == 0 and lo == 0 and hi == 0

def test_evaluator_requires_common_support(tmp_path):
    # two arms whose eval origins or masks differ -> ValueError naming the arm
    ...

def test_seed_average_before_resampling():
    # 5 seeds x 64 sessions: CI width equals that of the seed-mean series, not a 320-length series
    ...
```

- [ ] **Step 2:** Run them (FAIL), implement, then run again (PASS).
- [ ] **Step 3:** Write `build_report`. It must produce:
  - the per-seed and seed-mean metric table for each arm and scope
  - the three primary contrasts per scope, with block-5 and block-10 CIs, Holm-adjusted p (two-sided sign-flip over sessions, 10,000 draws), paired seed wins, and the verdict word
  - descriptive raw-MSE ratios versus Ethan's published per-seed LN4 and Q2 MSEs (overview Appendix A), labelled unpaired
  - per-layer variance and cosine at 0/4096/16384 updates (oversmoothing evidence), plus attention |α| quantiles
  - parameters, wall time, GPU time and peak memory per arm
  - figures: RankIC by arm with CIs; cumulative per-session ΔRankIC for H1−T0, H2−H1 and H3−H1; representation cosine by depth
  - for each arm, the six questions (what changed, why it should help, what happened, reproducible?, evidence sufficient?, retain/modify/remove)
  - a recommendations section with the best-supported architecture and the next experiment
- [ ] **Step 4:** Run `python -m depth_b.evaluate` and check by hand: for one arm, scope and seed, recompute the mean RankIC with an independent 10-line scipy `spearmanr` loop and confirm it matches the report to 1e-9.
- [ ] **Step 5:** Commit in the worktree with `git commit -m "feat(depth-b): paired evaluator and report"`. Then copy the docs to `tom-shlom` under `docs/research_20261020/depth_b/` and commit only that path.

### Task 7: Handoff and closure

**Files:**
- Modify: `docs/HANDOFF.md` (Running now and Next steps), `docs/research_20261019/PROGRESS.md`

- [ ] **Step 1:** Add `T<n> done 2026-10-.. by claude: ...` lines for Tasks 0–6 to `docs/research_20261019/PROGRESS.md`. Each line gives the result, Track A status and Track B verdicts.
- [ ] **Step 2:** Update `docs/HANDOFF.md` to point to the Track B report and Track A status. Commit only this task's hunks: `git add -p docs/HANDOFF.md`, keeping Codex's pre-existing hunks unstaged.
- [ ] **Step 3:** Run `graphify update .` in the main repo.

## Open items for the user (non-blocking)

- Ask Ethan for a single zip of `raw/alpaca/us_equities/1min/2021-09-26_2026-09-26/`. This would replace about 15k Drive downloads; Task 1 accepts a local path in place of `--folder`. Also ask for `admission_v3/protocol/scientific_definitions_v4.json`, which fixes the Task 3 gate if the counts differ.
- `later_development64` falls after the 2026-04-01 Study08 seal, so it is excluded by default. Lifting the seal would allow a third scope.
- Track A still needs Rishi's original `data/nasdaq_monthly_pit/assembled` and `runs/nasdaq100_historical_monthly_20261011` (exporter: `RECOVERY.md` in the Codex worktree).
