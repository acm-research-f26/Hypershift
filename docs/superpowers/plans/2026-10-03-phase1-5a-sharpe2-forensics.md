# Phase 1.5a: 2017 Sharpe≈2 Forensics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain why the corrected full-NYSE R5_f2 `alpha=0` THINK model (HH, seeds 0-4) gets a 2017 top-5 Sharpe of about 2 when its global ranking metrics are near chance. Decide between four explanations:
- (A) a tie or index-order artifact
- (B) a lucky or concentrated basket, or market/sector/beta exposure
- (C) a data or backtest error
- (D) a genuine top-tail signal

The analysis uses saved artifacts and CPU only.

**Architecture:**
- One pure-numpy module, `src/hypershift/eval/forensics.py`, holds every calculation. Each function is unit-tested on hand-checkable synthetic panels.
- One script, `scripts/forensic_2017.py`, runs the whole pipeline in stages from the saved run folders. It writes:
  - large evidence exports to `results/forensics_1_5a/` (git-ignored)
  - compact json/md/png to `docs/phase1_5a/` and `docs/figures/`
- Nothing under `results/R5_*` is ever written.

**Tech Stack:** Python 3, numpy, pandas, scipy.stats, matplotlib, pytest; existing `hypershift.eval.metrics` and `hypershift.eval.stats`.

**Spec:** the user's request of 2026-10-03 (three pasted briefs: "Investigate why...", "We need to pause...", and the 35-risk addendum), condensed in §Spec below. Where the briefs conflict with repository facts, §Sanity check wins and says why.

---

## Spec (condensed, binding)

1. **Primary case.** `results/R5_f2_alpha0_train/HH/seed_{0..4}`: validation-selected epoch, top-5, equal weight, daily rebalance, 237 test days (2017-01-03 to 2017-12-08).
   - **Primary strategy:** top-5 at the validation-selected epoch. Never replace it with the best k, epsilon, seed or epoch found in 2017.
   - **Diagnostic only:** `test_oracle_sr`, which uses best-test epoch selection.
2. **References**, analysed with the same code:
   - `R5_f_train/{HH,EH}` seeds 0-4 (the cleaner same-normalisation reference)
   - `R5_f2_alpha0_train/EH` seeds 0-4
   - `R5_f_paper/HH` seeds 0-4, secondary only. Its `norm=paper` uses the full-series max, which is a look-ahead input leak. Keep that separate from model-selection leakage.
3. **Milestone 1: lock the evidence.**
   - Inventory and hash the artifacts.
   - Recompute the daily series and Sharpe from the saved arrays.
   - Export tidy stock-day and portfolio-day tables.
4. **Milestone 2: explain the return.**
   - selection persistence and concentration
   - per-stock additive return contribution
   - hindsight exclude-and-reselect and best-day removal
   - tie-day vs non-tie-day split
   - index/tie mechanisms, evaluator-level and model-level
   - exposures, costs and benchmarks, including daily excess series
5. **Milestone 3: nulls and uncertainty.**
   - Nulls: within-day permutation, random daily top-5, fixed basket, common label permutation, industry/beta-matched baskets.
   - Every null keeps the same dates, mask, returns, k, weighting and costs.
   - Empirical `p = (1 + #{null ≥ obs}) / (B + 1)`, with the tail declared in advance.
   - Block bootstrap with sensitivity analysis. A declared multiplicity family.
6. **Milestone 4: do tiny score gaps carry information?** All grids are predeclared:
   - epsilon tie groups
   - jitter
   - fifth-minus-sixth margin buckets
   - k ∈ {1, 5, 10, 20, 50}
   - local IC
   - calibration
7. **Milestone 5: epochs, seeds, controls.**
   - the trajectory available in `history.jsonl`
   - cross-seed overlap
   - the existing planted-signal weight-decay control, stated narrowly
8. **Tests.** Deterministic unit tests plus a fixture integration test and a saved-artifact smoke test, covering the seven acceptance areas of brief 1.
9. **Sequential stopping.** Write up after each cheap phase. If a mechanism decisively reproduces the Sharpe, document it, rule out competitors minimally, then stop expanding. Never call it an "artifact" from IC≈0, small amplitude or the 26% tie rate alone.
10. **Deliverable.** A report with:
    - a per-seed reproduction table, evidence files and plots
    - gross/net benchmark comparisons
    - test results, the artifact inventory and limitations
    - a four-row decision table A–D: for, against, unresolved
    - a verdict on whether Sharpe≈2 is evidence of learned ranking
    - a separate costed proposal for anything needing training or post-2017 data

---

## Sanity check of the briefs (verified 2026-10-03 against the repo; executors must not re-derive these)

| # | Brief says / implies | Repository fact | Consequence for this plan |
|---|---|---|---|
| S1 | "epoch 0" may mean untrained | `loop.py:211-243`: epoch 0 is evaluated **after one full training pass** (740 windows, about 93 Adam steps at `batch_days 8`) | The report says "after the first training pass", never "untrained" |
| S2 | Use "several intermediate checkpoints" (Phase 7); "frozen model" for 2018-2023 (Phase 10) | **No weights or checkpoints exist anywhere** (no `*.pt/*.pth/*.ckpt`, and `loop.py` has no `torch.save`). Per-stock predictions are saved **only at the epoch with the best validation Sharpe so far** (`loop.py:249-255`, overwritten). The last write is the selected epoch | Per-epoch baskets/margins/ties and any frozen-model evaluation are **not recoverable**. Only aggregate per-epoch metrics exist (`history.jsonl`). A 2018+ test needs retraining, i.e. a **replication**, never "frozen". Moved to the costed proposal |
| S3 | Test permutations through the model (inputs/masks/graph reordered, outputs mapped back) | Needs the trained weights, which don't exist | Replaced by (a) a **model equivariance unit test** on random-init THINK (a code property; no training) and (b) a check of every index-dependent step in the data path. The trained-model version goes in the costed proposal |
| S4 | Verify target timing, dates, missing data, extremes, normalisation leak | Already verified and tested in Phase 1.5 A/C:<br>• `tests/test_phase15_eval.py` `test_prediction_at_window_end_is_scored_on_next_day_return`, `test_real_split_and_scored_dates`, `test_real_gt_mask_trace_to_raw_close`, `test_real_daily_return_from_scratch`, `test_norm_*`, `test_tie_break_is_lowest_index_and_matters`, `test_sharpe_matches_authors_up_to_annualisation_constant`<br>• `tests/test_phase15_data.py`: date vector, with an external AAPL 2013-01-24 anchor<br>• `docs/phase1_5/C_data_graph.md`: 8 NYSE stock-days with \|ret\|>0.5, missing-close fill 1.1, masked stocks still enter the graph | **Reuse; do not redo.** Add tests only for the uncovered risks: stale/zero-return runs inside selected baskets, duplicate rows, extreme returns *inside the selected baskets*, mask-vs-fill inside the scoring window, and exact-tie group composition |
| S5 | "Run ≥10,000 within-day permutations" and "random daily top-5" as two nulls | Permuting scores uniformly among valid stocks and taking the top 5 gives a **uniform random 5-subset**, which does not depend on the scores. They are the **same null**, identical for every seed and arm | Computed once, B = 10,000, and reported under both names. The genuinely different nulls are fixed baskets, a common label permutation (keeps score persistence) and industry/beta-matched baskets |
| S6 | Phase 9: run a clean wd 5e-4 vs 0 control | Already done one-factor in `docs/phase1_5/F_learnability.md`: same planted signal, data, code path and seeds; only wd changed. It reports IC/oracle 26% → 63-84%, `\|z\|` norms, prediction sd and loss-vs-decay gradient ratios (10-40×) | **No new run.** The report quotes the narrow conclusion and lists the one open item (level inputs) as not relevant here |
| S7 | Compare against "market benchmark" and "sector/beta-matched" baskets | No index/ETF series in the RSR data. Industry labels exist (`relation/sector_industry/NYSE_industry_ticker.json`). Beta can be estimated causally on training days against the equal-weight hold-all | "Market" = equal-weight hold-all of valid stocks (stated as such). Cap-weighted market = **UNKNOWN, not in data**. Beta uses training-period data only |
| S8 | Momentum baseline "if available" | Computable causally from the close column (feature index 4) at the window end | Included: 20-day past return, top-5, same backtest |
| S9 | Seeds as evidence | Seeds share the market days, ordering and evaluator | Seeds are repeated model runs, not independent samples. Formal tests use an intersection-union rule over seeds (below), never Holm-as-if-independent or pooled-seed n |
| S11 | A near-constant output could become a price-level ranking via the base price | `loop.py:166-167`: `_to_return(out, base, "return")` returns `out` unchanged, so base price never enters the score when `target=return` | No price-level path through the head. Whether the score *proxies* a simple input (MA ratios, past return, volatility, beta, degree, index) is still open, and is tested directly in Task 2B |
| S12 | "Only the final epoch's predictions exist" | Saved arrays are from the **validation-selected** epoch. Pass 2 of the overnight review reported selected epochs [8, 32, 0, 1, 2] for the 5 primary seeds | Three seeds' saved predictions come from the first 1-3 training passes. That allows a **cross-seed** early-vs-late comparison (Task 7), but no within-seed trajectory |
| S10 | Scores dtype | `test_pred.npy` is float32, [1737, 237]; `test_mask.npy` is float32 (0/1); scores are about 1e-4 to 1e-3, so one float32 ulp is about 6e-11 to 1e-10 | The absolute epsilon grid starts at the ulp scale (below) |

---

## Global Constraints

- **CPU only.** Set `CUDA_VISIBLE_DEVICES=-1` for every command. No Kaggle/GPU launches, no retraining, no seed top-ups, no architecture runs in Phase 1.5a.
- **Never write under `results/R5_*`, `results/R8_*`, `results/POC_*`.** Outputs go to:
  - `results/forensics_1_5a/` (large; git-ignored)
  - `docs/phase1_5a/` (md/json, committed)
  - `docs/figures/phase1_5a_*.png` (committed)
- **Predeclared grids** (module constants; never changed after looking at returns):
  - `EPS_ABS = (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)`
  - `EPS_FRAC = (0.001, 0.01, 0.05, 0.10, 0.25)` (fraction of the day's score SD)
  - `JITTER_FRAC = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00)` (noise SD as a fraction of the day's score SD)
  - `K_GRID = (1, 5, 10, 20, 50)`, with `K_PRIMARY = 5`
  - `COST_BPS = (0, 5, 10, 25)` per side, turnover convention as in `metrics.topk_daily_returns_net` (cost = 2 × bps × fraction of names replaced; day 1 = full)
  - `MARGIN_BUCKETS = 5` equal-count quantile buckets of the 5th-minus-6th margin over non-zero-margin days, plus a separate "exact tie (margin 0)" bucket
  - `DROP_DAYS = (1, 3, 5, 10, 20)`, `DROP_STOCKS = (1, 2, 3, 5, 10)`, `TOP_FREQ = (1, 5, 10, 20)`
  - `LOCAL_IC_Q = (0.05, 0.10, 0.20)`, `CALIB_BINS = 10`
  - **Repetitions:**
    - `B_NULL = 10_000` (random daily top-k, fixed basket)
    - `B_PERM = 2_000` (common label permutation, matched baskets)
    - `R_TIE = 1_000` (random exact/epsilon tie order)
    - `R_JIT = 200`
    - `N_BOOT = 5_000`
    - `BLOCK = 10`, sensitivity `(5, 20)`
  - `SEED_RNG = 20261003`; every random draw comes from `np.random.default_rng(SEED_RNG + offset)`, with the offset fixed per diagnostic in `forensics.RNG_OFFSETS`.
- **Score SD ("spread")** = cross-sectional population SD (ddof 0) of valid scores that day. `ptp` is recorded too.
  - **Zero-spread day:** the SD is 0. Jitter and epsilon-fraction use noise/epsilon 0, so selection falls back to the tie rule of the variant (stable, or random under random-tie variants).
  - Such days are counted and reported separately.
- **Effective tie group** (transitive by construction): sort valid scores descending. A new group starts wherever the gap between consecutive sorted scores is `> eps`.
  - Groups can chain. The group size distribution is always reported alongside.
  - Under randomisation, all groups entirely above rank k are kept, and the remaining slots are filled uniformly at random from the group straddling the boundary.
- **Formal tests.** Everything is computed on the primary run; 2017 is an exploratory year. The family has 4 tests with Holm adjustment, each on per-seed Sharpe, upper tail:
  - **F1:** vs the random-daily-top-5 null (≡ within-day permutation).
  - **F2:** vs the beta-quintile-matched null (tests hypothesis B, beta exposure).
  - **F3:** vs the industry-matched null (tests B, sector exposure).
  - **F4:** vs the common-label-permutation null (tests whether score persistence alone explains it).
  - The seed rule is intersection-union: family p for a test = **max** over seeds of the per-seed empirical p (valid for "every seed beats the null"; conservative).
  - Excess over hold-all is **not** a separate formal test: hold-all is the same series for the observed and null portfolios, so it cancels and the test would duplicate F1. It is still reported descriptively.
  - Everything else is labelled **exploratory**.
- **Labels in prose:**
  - hindsight stress tests ("exclude best contributors", "drop best days") are labelled **retrospective**
  - anything not in the paper is `INFERRED (not in paper)` or `UNKNOWN`
  - verdict words are only STRONG / SEED-ROBUST ONLY / NO EVIDENCE / INSUFFICIENT SEEDS, where a project verdict is given
- **Git:** stage only files this plan creates or modifies. Never touch the existing unrelated working-tree changes (`AGENTS.md`/`Codex.md` deletions, `results/POC_*`, logs, `kaggle/queue.txt`). No reset, clean or stash.
- **Commit trailer:** `Co-Authored-By: Claude Sonnet <noreply@anthropic.com>`

## Review Focus

1. **Zero-spread and all-tied days** (e.g., a constant prediction day). Expect: no division by zero, jitter = 0, epsilon-fraction = 0, random variants randomise the whole valid set, and the day is counted in "zero-spread days". Pinned in Task 2 (`test_zero_spread_day_*`).
2. **Days with fewer than k valid stocks, or a stock masked on some test days.** Expect: selection only from valid stocks, a portfolio of `min(k, n_valid)` names, and a fixed-basket null that only samples stocks valid on all 237 days. Pinned in Tasks 1 and 4.
3. **NaN/inf in saved arrays or an extreme |gt|>0.5 inside a selected basket.** Expect: the inventory fails loudly on non-finite values in valid cells; extreme selected returns are listed by ticker/date and their P&L share reported. Pinned in Task 3.
4. **A run folder whose `test_daily.npy` disagrees with recomputation**, e.g. a preset that wrote a different k. Expect: the stage aborts with the seed and the max abs difference; nothing downstream runs on unverified data. Pinned in Task 1.
5. **Positive affine rescaling** (`a·s + b`, a > 0) and the evaluator-level index permutation. Expect identical selections, and for a permutation identical stock identities when there are no ties. Pinned in Task 2.

---

## File structure

| File | Responsibility |
|---|---|
| `src/hypershift/eval/forensics.py` (create) | Pure functions: loading/verification, selection variants, boundary stats, tie groups, jitter, turnover/costs, contributions, reselection, day removal, nulls, empirical p, top-k diagnostics, local IC, calibration, margin buckets, persistence, exposures. Grid constants. No I/O except `load_run` |
| `scripts/forensic_2017.py` (create) | CLI orchestrator: `--stage {inventory,mechanism,proxy,integrity,decompose,nulls,gaps,trajectory,report,all}`, `--runs` (default primary + references), `--out results/forensics_1_5a`. Writes the evidence files, `docs/phase1_5a/forensics.json`, figures, and `docs/phase1_5a/REPORT_2017.md` |
| `tests/test_forensics.py` (create) | Synthetic unit tests (Tasks 1-6) |
| `tests/test_think_equivariance.py` (create) | Random-init THINK node-permutation equivariance (Task 2) |
| `tests/test_forensics_artifacts.py` (create) | `@pytest.mark.data` smoke test on the saved R5_f2 arrays; read-only check (Task 1) |
| `docs/phase1_5a/REPORT_2017.md` (create) | Final deliverable, plus an interim "Gate A" section written after Task 2 |
| `docs/phase1_5a/PROPOSAL_followups.md` (create) | Costed proposal for training-dependent or post-2017 work |
| `.gitignore` (modify) | Add `results/forensics_1_5a/` |
| `docs/phase1_5/PHASE1_5_SUMMARY.md`, `docs/PHASE1_TRACKER.md`, `CLAUDE.md` (modify, Task 8 only) | One pointer line each to Phase 1.5a; no verdict rewrites |

Run identifiers used throughout:

```python
PRIMARY = ("R5_f2_alpha0_train", "HH")
REFERENCES = (("R5_f_train", "HH"), ("R5_f_train", "EH"), ("R5_f2_alpha0_train", "EH"), ("R5_f_paper", "HH"))
SEEDS = (0, 1, 2, 3, 4)
```

---

### Task 1: Inventory, verification and tidy evidence export (Milestone 1)

**Files:**
- Create: `src/hypershift/eval/forensics.py` (constants, `load_run`, `RunArrays`, `selection`, `portfolio`, `boundary_stats`, `turnover`, `perf`)
- Create: `scripts/forensic_2017.py` (stage `inventory`)
- Create: `tests/test_forensics.py`, `tests/test_forensics_artifacts.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `RunArrays(pred: np.ndarray[N,D] float64, gt: np.ndarray[N,D] float64, mask: np.ndarray[N,D] bool, daily: np.ndarray[D], metrics: dict, config: dict, history: list[dict], path: Path)`
  - `load_run(exp: str, label: str, seed: int, root=Path("results")) -> RunArrays`
  - `select(scores_d: np.ndarray, idx: np.ndarray, k: int, order: str = "stable", rng=None) -> np.ndarray`, with `order ∈ {"stable","reverse","random"}`
  - `portfolio(pred, gt, mask, k=5, order="stable", rng=None) -> tuple[np.ndarray[D], list[np.ndarray]]`
  - `hold_all(gt, mask) -> np.ndarray[D]`
  - `boundary_stats(pred, mask, k=5) -> dict[str, np.ndarray]` with keys `s_k, s_k1, margin, exact_tie, sd, ptp, n_valid`
  - `turnover(baskets: list[np.ndarray]) -> np.ndarray[D]`
  - `net(r, to, bps) -> np.ndarray`
  - `perf(r) -> dict` with keys `mean, vol_d, sr, cumret, mdd, n`

- [ ] **Step 1: Write failing tests** (`tests/test_forensics.py`)

```python
import numpy as np
import pytest
from hypershift.eval import forensics as F
from hypershift.eval.metrics import topk_daily_returns, topk_daily_returns_net, sharpe


def panel():
    # 7 stocks x 3 days; stock 6 masked on day 1; day 2 has an exact tie across the k=2 boundary
    pred = np.array([[.9, .1, .5], [.8, .2, .5], [.1, .9, .5], [.2, .8, .1], [.3, .3, .0], [.0, .0, .0], [.5, .95, .2]])
    gt = np.arange(21, dtype=float).reshape(7, 3) / 100
    mask = np.ones((7, 3), bool); mask[6, 1] = False
    return pred, gt, mask


def test_portfolio_matches_existing_evaluator_and_respects_mask():
    pred, gt, mask = panel()
    r, baskets = F.portfolio(pred, gt, mask, k=2)
    np.testing.assert_allclose(r, topk_daily_returns(pred, gt, mask.astype(float), 2))
    assert 6 not in baskets[1] and set(baskets[1]) == {2, 3}


def test_stable_reverse_random_tie_orders():
    pred, gt, mask = panel()
    idx = np.arange(7)
    assert list(F.select(pred[:, 2], idx, 2, "stable")) == [0, 1]       # tie 0,1,2 at .5 -> lowest index
    assert list(F.select(pred[:, 2], idx, 2, "reverse")) == [2, 1]      # tie -> highest index first
    rng = np.random.default_rng(0)
    seen = {tuple(sorted(F.select(pred[:, 2], idx, 2, "random", rng))) for _ in range(200)}
    assert seen == {(0, 1), (0, 2), (1, 2)}


def test_boundary_stats_margin_and_exact_tie():
    pred, _, mask = panel()
    b = F.boundary_stats(pred, mask, k=2)
    assert b["s_k"][0] == .8 and b["s_k1"][0] == .5 and b["margin"][0] == pytest.approx(.3)
    assert b["exact_tie"].tolist() == [False, False, True]
    assert b["n_valid"].tolist() == [7, 6, 7]


def test_turnover_and_net_match_existing_cost_convention():
    pred, gt, mask = panel()
    r, baskets = F.portfolio(pred, gt, mask, k=2)
    to = F.turnover(baskets)
    assert to[0] == 1.0
    np.testing.assert_allclose(F.net(r, to, 10), topk_daily_returns_net(pred, gt, mask.astype(float), 2, 10))


def test_perf_uses_population_sd_and_sqrt252():
    r = np.array([.01, -.005, .02, 0.0])
    p = F.perf(r)
    assert p["sr"] == pytest.approx(r.mean() / r.std(ddof=0) * np.sqrt(252))
    assert p["sr"] == pytest.approx(sharpe(r))
    assert p["n"] == 4


def test_fewer_than_k_valid_stocks():
    pred = np.array([[.3], [.2], [.1]]); gt = np.array([[.01], [.02], [.03]])
    mask = np.array([[True], [False], [False]])
    r, b = F.portfolio(pred, gt, mask, k=5)
    assert list(b[0]) == [0] and r[0] == pytest.approx(.01)
```

- [ ] **Step 2: Run, expect failure** — `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests/test_forensics.py -q` → FAIL (`module 'hypershift.eval.forensics' not found`).

- [ ] **Step 3: Implement** (`src/hypershift/eval/forensics.py`, first part)

```python
"""Phase 1.5a forensic calculations (CPU, pure numpy). Grids are predeclared; do not edit after inspecting returns."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS_ABS = (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)
EPS_FRAC = (0.001, 0.01, 0.05, 0.10, 0.25)
JITTER_FRAC = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00)
K_GRID = (1, 5, 10, 20, 50)
K_PRIMARY = 5
COST_BPS = (0, 5, 10, 25)
MARGIN_BUCKETS = 5
DROP_DAYS = (1, 3, 5, 10, 20)
DROP_STOCKS = (1, 2, 3, 5, 10)
TOP_FREQ = (1, 5, 10, 20)
LOCAL_IC_Q = (0.05, 0.10, 0.20)
CALIB_BINS = 10
B_NULL, B_PERM, R_TIE, R_JIT, N_BOOT, BLOCK, BLOCK_SENS = 10_000, 2_000, 1_000, 200, 5_000, 10, (5, 20)
SEED_RNG = 20261003
RNG_OFFSETS = {"tie": 1, "eps": 2, "jitter": 3, "null_random": 4, "null_fixed": 5, "null_perm": 6,
               "null_matched": 7, "boot": 8, "index_perm": 9}


def rng_for(name: str, extra: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED_RNG + 1000 * RNG_OFFSETS[name] + extra)


@dataclass
class RunArrays:
    pred: np.ndarray
    gt: np.ndarray
    mask: np.ndarray
    daily: np.ndarray
    metrics: dict
    config: dict
    history: list
    path: Path


def load_run(exp: str, label: str, seed: int, root: Path | str = Path("results")) -> RunArrays:
    p = Path(root) / exp / label / f"seed_{seed}"
    pred = np.load(p / "test_pred.npy").astype(np.float64)
    gt = np.load(p / "test_gt.npy").astype(np.float64)
    mask = np.load(p / "test_mask.npy") > 0.5
    if not (np.isfinite(pred[mask]).all() and np.isfinite(gt[mask]).all()):
        raise ValueError(f"non-finite pred/gt in valid cells: {p}")
    hist = [json.loads(line) for line in open(p / "history.jsonl") if line.strip()]
    return RunArrays(pred, gt, mask, np.load(p / "test_daily.npy"), json.loads((p / "metrics.json").read_text()),
                     json.loads((p / "config.json").read_text()), hist, p)


def select(scores_d, idx, k, order="stable", rng=None):
    s = scores_d[idx]
    if order == "stable":
        o = np.argsort(-s, kind="stable")
    elif order == "reverse":
        o = np.lexsort((-np.arange(len(s)), -s))           # last key primary: score desc, then index desc
    elif order == "random":
        o = np.lexsort((rng.random(len(s)), -s))
    else:
        raise ValueError(order)
    return idx[o[:k]]


def portfolio(pred, gt, mask, k=K_PRIMARY, order="stable", rng=None):
    r, baskets = np.zeros(pred.shape[1]), []
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d])[0]
        top = select(pred[:, d], idx, k, order, rng) if len(idx) else idx
        baskets.append(top)
        r[d] = gt[top, d].mean() if len(top) else 0.0
    return r, baskets


def hold_all(gt, mask):
    return np.array([gt[mask[:, d], d].mean() if mask[:, d].any() else 0.0 for d in range(gt.shape[1])])


def boundary_stats(pred, mask, k=K_PRIMARY):
    D = pred.shape[1]
    out = {key: np.zeros(D) for key in ("s_k", "s_k1", "margin", "sd", "ptp")}
    out["exact_tie"], out["n_valid"] = np.zeros(D, bool), np.zeros(D, int)
    for d in range(D):
        s = np.sort(pred[mask[:, d], d])[::-1]
        out["n_valid"][d] = len(s)
        if len(s) == 0:
            continue
        out["sd"][d], out["ptp"][d] = s.std(), s[0] - s[-1]
        if len(s) > k:
            out["s_k"][d], out["s_k1"][d] = s[k - 1], s[k]
            out["margin"][d] = s[k - 1] - s[k]
            out["exact_tie"][d] = s[k - 1] == s[k]
    return out


def turnover(baskets):
    to, prev = np.zeros(len(baskets)), set()
    for d, b in enumerate(baskets):
        cur = set(b.tolist())
        to[d] = 1.0 if not prev else len(cur - prev) / max(len(cur), 1)
        prev = cur
    return to


def net(r, to, bps):
    return r - 2 * bps * 1e-4 * to


def perf(r):
    r = np.asarray(r, dtype=np.float64)
    sd = r.std()
    w = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    return {"mean": float(r.mean()), "vol_d": float(sd), "sr": 0.0 if sd == 0 else float(r.mean() / sd * math.sqrt(252)),
            "cumret": float(w[-1] - 1) if len(w) else 0.0, "mdd": float((w / peak - 1).min()) if len(w) else 0.0,
            "n": int(len(r))}
```

- [ ] **Step 4: Run the tests** → PASS. Also run `.venv/Scripts/python.exe -m pytest tests/test_metrics.py tests/test_phase15_eval.py -q` → PASS (unchanged).

- [ ] **Step 5: Saved-artifact smoke test** (`tests/test_forensics_artifacts.py`)

```python
import hashlib
from pathlib import Path
import numpy as np
import pytest
from hypershift.eval import forensics as F

RUN = Path("results/R5_f2_alpha0_train/HH")
pytestmark = [pytest.mark.data, pytest.mark.skipif(not RUN.exists(), reason="R5_f2 artifacts not present")]


@pytest.mark.parametrize("seed", range(5))
def test_saved_daily_series_and_metrics_reproduce_readonly(seed):
    p = RUN / f"seed_{seed}"
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir()}
    a = F.load_run("R5_f2_alpha0_train", "HH", seed)
    r, _ = F.portfolio(a.pred, a.gt, a.mask)
    # saved arrays are float32 and loop.py averaged float32 gt, so agreement is to float32 precision, not 1e-12
    np.testing.assert_allclose(r, a.daily, atol=1e-7)
    assert a.pred.shape == (1737, 237)
    assert F.perf(r)["sr"] == pytest.approx(a.metrics["test"]["sr"], abs=1e-5)
    # the saved arrays really are from the selected epoch
    assert a.history[a.metrics["best_epoch"]]["test"]["sr"] == pytest.approx(F.perf(r)["sr"], abs=1e-5)
    assert a.config["weight_decay"] == 0.0 and a.config["input_mode"] == "relative" and a.config["alpha"] == 0.0
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir()}
    assert before == after
```

Run: `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests/test_forensics_artifacts.py -q -m data` → PASS. If a daily value differs by more than 1e-7 or the Sharpe by more than 1e-5, Review Focus 4 applies: stop and report the seed and the difference. Selections must match exactly; if they don't, the cause is a dtype issue in a tie, which is a finding.

- [ ] **Step 6: Inventory stage** (`scripts/forensic_2017.py`). Implement `main()` with argparse and the `inventory` stage. For every run in `PRIMARY + REFERENCES` × `SEEDS`:
  1. Load with `F.load_run`.
  2. Assert that recomputed `portfolio` equals `daily` (atol 1e-7, float32 precision of the saved arrays) and `perf.sr` equals both `metrics.test.sr` and `history[best_epoch].test.sr` (abs 1e-5). On failure: `raise SystemExit(f"{run} seed {s}: max|diff|={...}")`.
  3. Record:
     - sha256 of every file in the run folder
     - `git rev-parse HEAD` (via `subprocess`)
     - pred dtype from `np.load(...).dtype`
     - `best_epoch`, `test_oracle_epoch`, `epochs_run`
     - config keys `weight_decay, input_mode, alpha, norm, spatial_residual, topk, seq, seed`
  4. Map test day j to its date: `dates = [l.strip()[:10] for l in open("data/raw/rsr/data/NYSE_aver_line_dates.csv") if l.strip()]`, then `date_j = dates[29 + 1008 + j]`. Assert `date_0 == "2017-01-03"` and `date_236 == "2017-12-08"` (mapping verified by `tests/test_phase15_data.py`).
  5. Tickers come from `hypershift.data.rsr.read_ticker_file(Path("data/raw/rsr/data/NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv"))`. Assert the length is 1737.
  6. Export `results/forensics_1_5a/stockday_<exp>_<label>_s<seed>.csv.gz`, one row per valid stock-day, with columns `date, day, ticker, index, score, ret_next, rank (1 = best, stable order), in_top5, s5, s6, margin5, exact_tie5, seed`.
  7. Export `results/forensics_1_5a/portday_<exp>_<label>.csv` with columns `date, day, seed, ret_gross, ret_hold_all, turnover, exact_tie5, margin5, score_sd, n_valid`.
  8. Write `docs/phase1_5a/inventory.json`. Its list `"not_recoverable"` must contain exactly:
     - `"model weights / checkpoints (none saved; loop.py has no torch.save)"`
     - `"per-stock predictions for non-selected epochs (overwritten; only the last val-improving epoch is on disk)"`
     - `"through-model index permutation of the trained model"`
     - `"frozen-model evaluation outside 2017"`

  Add `results/forensics_1_5a/` to `.gitignore`.

  Run: `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/forensic_2017.py --stage inventory`. Expect 25 run-seeds verified and the files written.

- [ ] **Step 7: Commit**

```bash
git add src/hypershift/eval/forensics.py scripts/forensic_2017.py tests/test_forensics.py tests/test_forensics_artifacts.py .gitignore docs/phase1_5a/inventory.json
git commit -m "feat(phase1.5a): forensic module, inventory/verification stage and tidy 2017 exports"
```

---

### Task 2: Tie and index mechanisms

**Files:**
- Modify: `src/hypershift/eval/forensics.py` (add `tie_groups`, `select_eps`, `jitter`, `evaluator_permutation`)
- Modify: `scripts/forensic_2017.py` (stage `mechanism`)
- Create: `tests/test_think_equivariance.py`
- Test: `tests/test_forensics.py` (append)

**Interfaces:**
- Consumes: `select`, `portfolio`, `perf`, `boundary_stats`, `rng_for` (Task 1)
- Produces:
  - `tie_groups(sorted_desc: np.ndarray, eps: float) -> np.ndarray[int]` (group id per sorted position)
  - `select_eps(scores_d, idx, k, eps, rng) -> np.ndarray` (reference implementation, used by tests)
  - `portfolio_eps(pred, gt, mask, k, eps_abs=None, eps_frac=None, rng=None) -> tuple[np.ndarray, list]` (one draw)
  - `eps_draws(pred, gt, mask, k, R, rng, eps_abs=None, eps_frac=None) -> dict` with keys `returns: [R,D]`, `jaccard: [R]` (mean daily Jaccard vs stable), `amb_frac: float` (days whose boundary group has more than 1 member), `group_size: [D]` (boundary group size). Fast path: sort and group **once per day**, then each draw only refills the boundary group. `eps_abs=0` is exactly the random exact-tie order, so `R_TIE` random-tie draws use this function
  - `jitter(pred, mask, frac, rng) -> np.ndarray`
  - `evaluator_permutation(pred, gt, mask, perm) -> tuple[np.ndarray, np.ndarray, np.ndarray]`

- [ ] **Step 1: Failing tests** (append to `tests/test_forensics.py`)

```python
def test_tie_groups_gap_chaining_is_transitive():
    s = np.array([1.0, 0.95, 0.90, 0.5, 0.49, 0.0])
    assert F.tie_groups(s, 0.06).tolist() == [0, 0, 0, 1, 1, 2]     # 1.0~0.95~0.90 chain into one group
    assert F.tie_groups(s, 0.0).tolist() == [0, 1, 2, 3, 4, 5]


def test_select_eps_keeps_clear_winners_and_randomises_boundary_group():
    scores = np.array([.9, .5, .5005, .4995, .1]); idx = np.arange(5)
    rng = np.random.default_rng(0)
    picks = {tuple(sorted(F.select_eps(scores, idx, 2, 0.001, rng))) for _ in range(300)}
    assert all(0 in p for p in picks) and {p[1] for p in picks} == {1, 2, 3}


def test_zero_spread_day_jitter_is_zero_and_eps_frac_randomises_all():
    pred = np.full((6, 1), .3); mask = np.ones((6, 1), bool)
    np.testing.assert_array_equal(F.jitter(pred, mask, 0.5, np.random.default_rng(0)), pred)
    rng = np.random.default_rng(1)
    seen = {tuple(sorted(F.portfolio_eps(pred, np.zeros((6, 1)), mask, 2, eps_frac=0.1, rng=rng)[1][0])) for _ in range(300)}
    assert len(seen) == 15                                              # C(6,2): all pairs reachable


def test_eps_draws_matches_reference_and_random_ties():
    pred, gt, mask = panel()
    out = F.eps_draws(pred, gt, mask, 2, 400, np.random.default_rng(0), eps_abs=0.0)
    assert out["returns"].shape == (400, 3)
    np.testing.assert_allclose(out["returns"][:, :2], np.broadcast_to(F.portfolio(pred, gt, mask, 2)[0][:2], (400, 2)))
    day2 = {round(x, 10) for x in out["returns"][:, 2]}                 # 3-way tie at .5 -> 3 possible pairs
    assert day2 == {round(gt[list(p), 2].mean(), 10) for p in ((0, 1), (0, 2), (1, 2))}
    assert out["group_size"].tolist() == [1, 1, 3] and out["amb_frac"] == pytest.approx(1 / 3)


def test_jitter_is_seed_deterministic():
    pred = np.random.default_rng(3).normal(size=(50, 4)); mask = np.ones_like(pred, bool)
    a = F.jitter(pred, mask, 0.25, np.random.default_rng(7)); b = F.jitter(pred, mask, 0.25, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_affine_rescaling_leaves_selection_unchanged():
    pred, gt, mask = panel()
    _, b0 = F.portfolio(pred, gt, mask, k=2)
    _, b1 = F.portfolio(1000.0 * pred + 7.0, gt, mask, k=2)
    assert all((x == y).all() for x, y in zip(b0, b1))


def test_evaluator_permutation_preserves_identity_without_ties_and_can_change_with_ties():
    pred, gt, mask = panel()
    perm = np.array([6, 5, 4, 3, 2, 1, 0])
    p2, g2, m2 = F.evaluator_permutation(pred, gt, mask, perm)
    r0, b0 = F.portfolio(pred, gt, mask, k=2); r1, b1 = F.portfolio(p2, g2, m2, k=2)
    assert set(perm[b1[0]]) == set(b0[0]) and r1[0] == r0[0]           # day 0: no tie at boundary
    assert set(perm[b1[2]]) != set(b0[2])                              # day 2: 3-way tie -> lowest *new* index wins
```

`tests/test_think_equivariance.py`:

```python
import torch
from hypershift.data.hypergraph import Hypergraph
from hypershift.models.think import THINK

EDGES = ((0, 1, 2), (2, 3), (4, 5))


def test_think_is_node_permutation_equivariant_random_init():
    torch.manual_seed(0)
    for temporal in ("hyp", "euc"):
        for spatial in ("hyp", "euc"):
            m = THINK(in_dim=5, hidden=8, seq=16, kernel=4, temporal=temporal, spatial=spatial).eval()
            x = torch.rand(2, 7, 16, 5) * 0.1
            perm = torch.tensor([3, 6, 0, 5, 1, 4, 2])          # new position p holds old node perm[p]
            inv = torch.argsort(perm)
            hg = Hypergraph(7, EDGES).to_torch("cpu")
            hg_p = Hypergraph(7, tuple(tuple(int(inv[v]) for v in e) for e in EDGES)).to_torch("cpu")
            with torch.no_grad():
                y = m(x, hg)
                y_p = m(x[:, perm], hg_p)
            torch.testing.assert_close(y_p, y[:, perm], rtol=1e-5, atol=1e-7)
```

- [ ] **Step 2: Run** → the new tests FAIL (missing functions). The equivariance test may PASS immediately. If it FAILS, that is a **finding**: record the failing temporal/spatial combination; it is evidence for hypothesis A through the model path.

- [ ] **Step 3: Implement** (append to `forensics.py`)

```python
def tie_groups(sorted_desc, eps):
    gaps = -np.diff(sorted_desc)
    return np.concatenate([[0], np.cumsum(gaps > eps)]).astype(int)


def select_eps(scores_d, idx, k, eps, rng):
    s = scores_d[idx]
    o = np.argsort(-s, kind="stable")
    if len(o) <= k:
        return idx[o]
    g = tie_groups(s[o], eps)
    gb = g[k - 1]
    above = o[g < gb]
    group = o[g == gb]
    fill = rng.choice(group, size=k - len(above), replace=False)
    return idx[np.concatenate([above, fill])]


def portfolio_eps(pred, gt, mask, k=K_PRIMARY, eps_abs=None, eps_frac=None, rng=None):
    r, baskets = np.zeros(pred.shape[1]), []
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d])[0]
        eps = eps_abs if eps_abs is not None else eps_frac * (pred[idx, d].std() if len(idx) else 0.0)
        top = select_eps(pred[:, d], idx, k, eps, rng) if len(idx) else idx
        if eps_frac is not None and len(idx) and pred[idx, d].std() == 0:
            top = idx[rng.choice(len(idx), size=min(k, len(idx)), replace=False)]   # zero-spread: whole set tied
        baskets.append(top)
        r[d] = gt[top, d].mean() if len(top) else 0.0
    return r, baskets


def eps_draws(pred, gt, mask, k, R, rng, eps_abs=None, eps_frac=None):
    D = pred.shape[1]
    stable = portfolio(pred, gt, mask, k)[1]
    rets, jac = np.zeros((R, D)), np.zeros((R, D))
    gsize = np.zeros(D, int)
    for d in range(D):
        idx = np.nonzero(mask[:, d])[0]
        if not len(idx):
            continue
        s = pred[idx, d]
        sd = s.std()
        o = np.argsort(-s, kind="stable")
        if eps_frac is not None and sd == 0:                       # zero-spread day: whole valid set is one group
            above, group = o[:0], o
        else:
            eps = eps_abs if eps_abs is not None else eps_frac * sd
            g = tie_groups(s[o], eps)
            gb = g[min(k, len(o)) - 1]
            above, group = o[g < gb], o[g == gb]
        gsize[d] = len(group)
        need = min(k, len(o)) - len(above)
        fill = np.argsort(rng.random((R, len(group))), axis=1)[:, :need]       # R independent refills
        picks = np.concatenate([np.broadcast_to(above, (R, len(above))), group[fill]], axis=1)
        rets[:, d] = gt[idx[picks], d].mean(1)
        st = set(stable[d].tolist())
        jac[:, d] = [len(st & set(idx[p].tolist())) / len(st | set(idx[p].tolist())) for p in picks]
    return {"returns": rets, "jaccard": jac.mean(1), "amb_frac": float((gsize > 1).mean()), "group_size": gsize}


def jitter(pred, mask, frac, rng):
    out = pred.copy()
    for d in range(pred.shape[1]):
        i = mask[:, d]
        sd = pred[i, d].std() if i.any() else 0.0
        if sd > 0:
            out[i, d] = pred[i, d] + rng.normal(0.0, frac * sd, size=int(i.sum()))
    return out


def evaluator_permutation(pred, gt, mask, perm):
    """Row p of the output holds old stock perm[p]; the scores travel with their stock."""
    return pred[perm], gt[perm], mask[perm]
```

- [ ] **Step 4: Run** `pytest tests/test_forensics.py tests/test_think_equivariance.py -q` → PASS.

- [ ] **Step 5: Stage `mechanism`.** For the primary run, plus `R5_f_train/HH` and `R5_f_train/EH`, per seed:

  | Variant | Repetitions | Output |
  |---|---|---|
  | `stable` (as saved) | 1 | perf, turnover |
  | `random` exact-tie order (`eps_draws(eps_abs=0)`) | `R_TIE` | Sharpe distribution (mean, sd, 5/50/95%), mean Jaccard of baskets vs stable |
  | `reverse` index tie order | 1 | perf, Jaccard vs stable |
  | evaluator index permutations | 200 random `perm` (rng `index_perm`) | Sharpe distribution, Jaccard vs stable mapped back via `perm`. Under stable sort, a permutation only changes which tied stock wins, so this should match the random-tie distribution. Report it as a consistency check, not as independent evidence |
  | constant scores (`pred*0`) | 1 | perf (= first 5 valid by index) |
  | first-5 / last-5 valid indices | 1 each | perf |
  | random fixed index permutation, then constant scores | 1,000 | Sharpe distribution (= random fixed index basket rule) |
  | exact-tie days vs non-tie days | — | n days, mean, vol, cumulative contribution (sum of daily returns), hit rate (`r>0`), conditional Sharpe |

  **Exact-tie group composition** on tie days: tied-group size, and the fraction of tied stocks that are:
  - (i) graph-isolated, i.e. in no hyperedge (`base_hypergraph` from `hypershift.data.hypergraph` on the NYSE data, as built in `loop.py`)
  - (ii) stale, i.e. relative inputs all equal within the 16-day window (`close` unchanged)
  - (iii) partly masked inside the window

  This tests whether exact ties are produced by identical inputs. Also record the **tied score values**: how many distinct values the boundary-tie groups take across all tie days, and the most common ones. Ties on one repeated value point to output saturation or clamping, not to identical inputs.

  Write `docs/phase1_5a/mechanism.json` and `docs/figures/phase1_5a_mechanism.png` (one panel per run: violin plots of the Sharpe distributions, with the stable value marked and hold-all marked).

  Run: `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/forensic_2017.py --stage mechanism`.

- [ ] **Step 6: Commit**

```bash
git add src/hypershift/eval/forensics.py scripts/forensic_2017.py tests/test_forensics.py tests/test_think_equivariance.py docs/phase1_5a/mechanism.json docs/figures/phase1_5a_mechanism.png
git commit -m "feat(phase1.5a): tie/index mechanism tests and THINK equivariance test"
```

---

### Task 2B: What does the near-flat score encode? (factor proxy; batch 1)

The cheapest high-information test. If the model's score is mostly a monotone proxy for a simple input, and a top-5 on that input reproduces Sharpe≈2, the answer is a **simple factor tilt**: neither an artifact (A) nor learned ranking (D).

**Files:**
- Modify: `src/hypershift/eval/forensics.py` (add `momentum_scores`, `rolling_vol`, `train_beta`, `industry_of`, `graph_degree`, `proxy_features`, `daily_spearman`, `project_scores`)
- Modify: `scripts/forensic_2017.py` (stage `proxy`)
- Test: `tests/test_forensics.py` (append)

**Interfaces:**
- Consumes: `portfolio`, `perf` (Task 1)
- Produces (used by Tasks 4, 5):
  - `momentum_scores(close: np.ndarray[N,T], target_days: np.ndarray, lb=20) -> np.ndarray[N,D]` (close[t]/close[t-lb]-1 at t = target-1)
  - `rolling_vol(close, target_days, lb=20) -> np.ndarray[N,D]` (population SD of daily close-to-close returns over days t-lb+1..t)
  - `train_beta(gt_full, mask_full, valid_index) -> np.ndarray[N]`
  - `industry_of(tickers, json_path) -> list[str]`
  - `graph_degree(edges, n) -> np.ndarray[N]`
  - `proxy_features(features: np.ndarray[N,T,C], gt_full, mask_full, valid_index, target_days, edges) -> dict[str, np.ndarray[N,D]]`. Keys:
    - `ma5_rel, ma10_rel, ma20_rel, ma30_rel` (MA / close at t)
    - `ret1, ret5, ret20`, `vol20`, `beta`, `degree`, `index`, `close_level`

    Every feature uses only data up to t = target-1.
  - `daily_spearman(a, b, mask) -> np.ndarray[D]` (nan on days where either side is constant)
  - `project_scores(pred, feats: dict, mask) -> tuple[np.ndarray, np.ndarray]` (per day, OLS of the score on the standardised features over valid stocks → (fitted, residual))

- [ ] **Step 1: Failing tests**

```python
def test_momentum_and_vol_are_causal():
    close = np.cumprod(np.full((2, 40), 1.01), axis=1); close[1] = 1.0
    m = F.momentum_scores(close, np.array([30]), lb=20)
    assert m[0, 0] == pytest.approx(close[0, 29] / close[0, 9] - 1) and m[1, 0] == 0.0
    close2 = close.copy(); close2[:, 30:] *= 5                          # future change must not move the features
    np.testing.assert_array_equal(F.momentum_scores(close2, np.array([30]), lb=20), m)
    np.testing.assert_array_equal(F.rolling_vol(close2, np.array([30])), F.rolling_vol(close, np.array([30])))
    assert F.rolling_vol(close, np.array([30]))[1, 0] == 0.0


def test_daily_spearman_and_projection():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(50, 4)); mask = np.ones_like(f, bool)
    pred = 3 * f + 1
    np.testing.assert_allclose(F.daily_spearman(pred, f, mask), 1.0)
    fitted, resid = F.project_scores(pred, {"f": f}, mask)
    np.testing.assert_allclose(resid, 0.0, atol=1e-9)
    assert np.isnan(F.daily_spearman(np.zeros_like(f), f, mask)).all()


def test_graph_degree():
    assert F.graph_degree(((0, 1, 2), (2, 3)), 5).tolist() == [1, 1, 2, 1, 0]
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
from scipy.stats import rankdata


def momentum_scores(close, target_days, lb=20):
    t = np.asarray(target_days) - 1
    return close[:, t] / close[:, t - lb] - 1


def rolling_vol(close, target_days, lb=20):
    r = np.zeros_like(close)
    r[:, 1:] = close[:, 1:] / close[:, :-1] - 1
    return np.stack([r[:, t - lb + 1:t + 1].std(axis=1) for t in np.asarray(target_days) - 1], axis=1)


def train_beta(gt_full, mask_full, valid_index):
    g, m = gt_full[:, 1:valid_index], mask_full[:, 1:valid_index] > 0.5
    mkt = np.array([g[m[:, d], d].mean() if m[:, d].any() else 0.0 for d in range(g.shape[1])])
    beta = np.full(g.shape[0], np.nan)
    for i in range(g.shape[0]):
        ok = m[i]
        if ok.sum() >= 250:
            x, y = mkt[ok], g[i, ok]
            beta[i] = np.cov(x, y, ddof=0)[0, 1] / x.var()
    return beta


def industry_of(tickers, json_path):
    lab = {t: ind for ind, ts in json.loads(Path(json_path).read_text()).items() for t in ts}
    return [lab.get(t, "n/a") for t in tickers]


def graph_degree(edges, n):
    deg = np.zeros(n, int)
    for e in edges:
        deg[list(e)] += 1
    return deg


def proxy_features(features, gt_full, mask_full, valid_index, target_days, edges):
    t = np.asarray(target_days) - 1
    close = features[:, :, 4]
    n, D = features.shape[0], len(t)
    out = {f"ma{w}_rel": features[:, t, c] / close[:, t] for c, w in ((0, 5), (1, 10), (2, 20), (3, 30))}
    out.update(ret1=momentum_scores(close, target_days, 1), ret5=momentum_scores(close, target_days, 5),
               ret20=momentum_scores(close, target_days, 20), vol20=rolling_vol(close, target_days, 20),
               close_level=close[:, t])
    out["beta"] = np.repeat(train_beta(gt_full, mask_full, valid_index)[:, None], D, axis=1)
    out["degree"] = np.repeat(graph_degree(edges, n)[:, None].astype(float), D, axis=1)
    out["index"] = np.repeat(np.arange(n, dtype=float)[:, None], D, axis=1)
    return out


def daily_spearman(a, b, mask):
    out = np.full(a.shape[1], np.nan)
    for d in range(a.shape[1]):
        i = mask[:, d] & np.isfinite(a[:, d]) & np.isfinite(b[:, d])
        if i.sum() >= 5 and np.ptp(a[i, d]) > 0 and np.ptp(b[i, d]) > 0:
            out[d] = np.corrcoef(rankdata(a[i, d]), rankdata(b[i, d]))[0, 1]
    return out


def project_scores(pred, feats, mask):
    fitted, resid = np.full_like(pred, np.nan), np.full_like(pred, np.nan)
    names = sorted(feats)
    for d in range(pred.shape[1]):
        X = np.stack([feats[k][:, d] for k in names], axis=1)
        i = mask[:, d] & np.isfinite(X).all(1)
        if i.sum() <= X.shape[1] + 1:
            continue
        Z = (X[i] - X[i].mean(0)) / np.where(X[i].std(0) > 0, X[i].std(0), 1)
        A = np.column_stack([np.ones(i.sum()), Z])
        coef, *_ = np.linalg.lstsq(A, pred[i, d], rcond=None)
        fitted[i, d] = A @ coef
        resid[i, d] = pred[i, d] - fitted[i, d]
    return fitted, resid
```

  In `portfolio`, a NaN score must never be selected. Masks passed to `portfolio` for fitted/residual scores are `mask & isfinite(score)`.

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Stage `proxy`.** Load NYSE via `load_rsr("data/raw/rsr/data", "NYSE", norm="train")`. Target days are `1008 + arange(237)`. Edges come from `base_hypergraph` exactly as `loop.py` builds them for this config. Per primary seed, plus `R5_f_train` HH/EH:
  1. The mean daily Spearman of the model score vs each proxy feature, with a block-bootstrap 95% CI over days. Present this as a table ranked by |mean|.
  2. The per-day R² of `project_scores` (all features): mean and median.
  3. **Proxy portfolios.** For each feature, a top-5 on `sign × feature`, where the sign is the sign of that feature's mean Spearman with the **model score**. This uses no return information, so it is not data-mined. Report perf and Jaccard vs the model baskets.
  4. **Fitted vs residual portfolios:** top-5 on `fitted` and top-5 on `resid`. Report perf and Jaccard vs the model.

  Write `docs/phase1_5a/proxy.json` and `docs/figures/phase1_5a_proxy.png` (bar chart of mean Spearman per feature; Sharpe of the model vs the proxy, fitted and residual portfolios).

- [ ] **Step 6: Commit** — message `feat(phase1.5a): factor-proxy test (what the score encodes)`.

---

### Task 3: Backtest integrity, uncovered risks only

**Files:**
- Modify: `scripts/forensic_2017.py` (stage `integrity`)
- Modify: `src/hypershift/eval/forensics.py` (add `stale_runs`, `duplicate_rows`)
- Test: `tests/test_forensics.py` (append)

**Interfaces:**
- Consumes: `load_run`, `portfolio`
- Produces:
  - `stale_runs(close: np.ndarray[N,T], min_len: int = 3) -> np.ndarray[N,T] bool` (True where close is unchanged for ≥ min_len consecutive days, ending at t)
  - `duplicate_rows(x: np.ndarray[N,T,C]) -> list[tuple[int,int]]` (pairs of stocks with identical full series)

- [ ] **Step 1: Failing tests**

```python
def test_stale_runs_flags_unchanged_close():
    c = np.array([[1, 1, 1, 1, 2], [1, 2, 3, 4, 5]], float)
    s = F.stale_runs(c, min_len=3)
    assert s[0].tolist() == [False, False, True, True, False] and not s[1].any()


def test_duplicate_rows_detects_identical_series():
    x = np.random.default_rng(0).normal(size=(4, 10, 5)); x[3] = x[1]
    assert F.duplicate_rows(x) == [(1, 3)]
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
def stale_runs(close, min_len=3):
    same = np.zeros_like(close, dtype=bool)
    same[:, 1:] = close[:, 1:] == close[:, :-1]
    run = np.zeros(close.shape, dtype=int)
    for t in range(1, close.shape[1]):
        run[:, t] = np.where(same[:, t], run[:, t - 1] + 1, 0)
    return run + 1 >= min_len


def duplicate_rows(x):
    flat = x.reshape(x.shape[0], -1)
    seen, out = {}, []
    for i, row in enumerate(flat):
        h = row.tobytes()
        if h in seen:
            out.append((seen[h], i))
        else:
            seen[h] = i
    return out
```

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Stage `integrity`.** It loads NYSE through `hypershift.data.rsr.load_rsr("data/raw/rsr/data", "NYSE", norm="train")` and reports for the primary run:
  1. Re-run the existing integrity tests and record their output verbatim:

     ```bash
     .venv/Scripts/python.exe -m pytest -m data tests/test_phase15_eval.py tests/test_phase15_data.py -q
     ```
  2. Stale stock-days inside the selected baskets (`stale_runs` on the close column, feature 4), and their P&L share.
  3. Duplicate stock series (`duplicate_rows`).
  4. Every selected stock-day with |gt| > 0.2: ticker, date, return, seed count, and the share of total P&L. Then a neighbouring-day check: an extreme followed by a reversal greater than 50% of its size within 3 days, a candidate split/adjustment error. List these; do not delete.
  5. Selected stock-days whose 17-day window touches a fill value (`mask` is 0 on any window day in the full data). It should be 0, since mask = min over the window; assert this.
  6. Confirm `metrics.sharpe` uses `np.std` ddof 0 and √252, with no rf, by citing `metrics.py:50-52` and the existing test.

  Write `docs/phase1_5a/integrity.json` and add an "Integrity" section to the report. If any integrity check fails in a way that changes returns, stop. Fix it in a **new** analysis variant (never edit the saved runs), rerun Tasks 1-2 on the corrected series, and report both.

- [ ] **Step 6: Commit** — `git add` the changed files plus `docs/phase1_5a/integrity.json` and the report; message `feat(phase1.5a): backtest integrity checks for uncovered risks`.

---

### Task 4: Return decomposition, exposures, costs, benchmarks (Milestone 2)

**Files:**
- Modify: `src/hypershift/eval/forensics.py` (add `selection_freq`, `top_share`, `jaccard_series`, `durations`, `contributions`, `exclude_reselect`, `drop_best_days`)
- Modify: `scripts/forensic_2017.py` (stage `decompose`)
- Test: `tests/test_forensics.py` (append)

**Interfaces:**
- Consumes: `portfolio`, `perf`, `turnover`, `net`, `hold_all` (Task 1); `momentum_scores`, `train_beta`, `industry_of` (Task 2B)
- Produces:
  - `selection_freq(baskets, n) -> np.ndarray[n]`
  - `top_share(freq, top) -> float`
  - `jaccard_series(baskets) -> np.ndarray[D-1]`
  - `durations(baskets) -> np.ndarray` (lengths of consecutive-day holding spells)
  - `contributions(baskets, gt) -> np.ndarray[N]` (sum over days of `gt[i,d]/len(basket_d)`)
  - `exclude_reselect(pred, gt, mask, exclude: np.ndarray, k) -> np.ndarray[D]`
  - `drop_best_days(r, n) -> np.ndarray`

- [ ] **Step 1: Failing tests**

```python
def test_contributions_sum_to_portfolio_return():
    pred, gt, mask = panel()
    r, b = F.portfolio(pred, gt, mask, k=2)
    assert F.contributions(b, gt).sum() == pytest.approx(r.sum())


def test_exclude_reselect_uses_next_best_valid():
    pred, gt, mask = panel()
    r = F.exclude_reselect(pred, gt, mask, np.array([0]), k=2)
    assert r[0] == pytest.approx(gt[[1, 6], 0].mean())


def test_drop_best_days_and_persistence():
    r = np.array([.05, .01, -.02, .03]); assert sorted(F.drop_best_days(r, 2)) == [-.02, .01]
    b = [np.array([0, 1]), np.array([0, 1]), np.array([0, 2]), np.array([3, 4])]
    np.testing.assert_allclose(F.jaccard_series(b), [1.0, 1 / 3, 0.0])
    assert sorted(F.durations(b).tolist()) == [1, 1, 1, 2, 3]
    f = F.selection_freq(b, 5); assert f.tolist() == [3, 2, 1, 1, 1] and F.top_share(f, 1) == pytest.approx(3 / 8)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
def selection_freq(baskets, n):
    f = np.zeros(n, int)
    for b in baskets:
        f[b] += 1
    return f


def top_share(freq, top):
    return float(np.sort(freq)[::-1][:top].sum() / max(freq.sum(), 1))


def jaccard_series(baskets):
    out = []
    for a, b in zip(baskets[:-1], baskets[1:]):
        a, b = set(a.tolist()), set(b.tolist())
        out.append(len(a & b) / max(len(a | b), 1))
    return np.array(out)


def durations(baskets):
    spells, open_ = [], {}
    for d, b in enumerate(baskets + [np.array([], int)]):
        cur = set(b.tolist())
        for s in list(open_):
            if s not in cur:
                spells.append(d - open_.pop(s))
        for s in cur:
            open_.setdefault(s, d)
    return np.array(spells)


def contributions(baskets, gt):
    c = np.zeros(gt.shape[0])
    for d, b in enumerate(baskets):
        if len(b):
            c[b] += gt[b, d] / len(b)
    return c


def exclude_reselect(pred, gt, mask, exclude, k=K_PRIMARY):
    m = mask.copy()
    m[exclude, :] = False
    return portfolio(pred, gt, m, k)[0]


def drop_best_days(r, n):
    return np.delete(r, np.argsort(r)[::-1][:n])
```

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Stage `decompose`** (primary run plus the `R5_f_train` HH/EH reference, per seed and seed-pooled where stated). Every number is labelled exploratory; the hindsight rows are labelled retrospective.
  1. **Persistence:**
     - per-stock top-5 frequency, with the top 30 listed by ticker
     - `top_share` for `TOP_FREQ`
     - mean and median Jaccard
     - the duration distribution
     - the number of distinct stocks ever selected
  2. **Exposure:**
     - industry share of selection slots vs the industry share of the valid universe
     - mean basket beta vs universe mean beta (`train_beta` from the full data, causal)
  3. **Contribution:**
     - top 10 contributors by ticker, with additive return and share of total
     - `exclude_reselect` for `DROP_STOCKS` (retrospective), reporting perf (sr, cumret) after each exclusion
     - `drop_best_days` for `DROP_DAYS` (retrospective), reporting mean, vol, cumret, sr
  4. **Costs and benchmarks:**
     - Strategies: the model, hold-all, momentum top-5 (`momentum_scores` on the close column), and constant-score/first-5.
     - Each at `COST_BPS` with its own turnover: gross and net perf.
     - For model-minus-benchmark **daily** excess series (vs hold-all and vs momentum): mean, Sharpe, and a block-bootstrap 95% CI of the excess mean (`hypershift.eval.stats.stationary_bootstrap_indices`, `BLOCK` with `BLOCK_SENS` sensitivity, `N_BOOT`).
  5. **Seed concentration:** pairwise daily Jaccard between seeds, correlation of daily returns between seeds, and stocks in the top 10 frequency of ≥3 seeds.

  Write `docs/phase1_5a/decompose.json`, the figures `docs/figures/phase1_5a_{persistence,contrib,dropdays,costs}.png`, and a report section.

- [ ] **Step 6: Commit** — message `feat(phase1.5a): return decomposition, exposures, costs, benchmarks`.

---

### Task 5: Nulls and uncertainty (Milestone 3)

**Files:**
- Modify: `src/hypershift/eval/forensics.py` (add `null_random_topk`, `null_fixed`, `null_label_perm`, `null_matched`, `empirical_p`, `sr_rows`)
- Modify: `scripts/forensic_2017.py` (stage `nulls`)
- Test: `tests/test_forensics.py` (append)

**Interfaces:**
- Consumes: `portfolio`, `hold_all`, `train_beta`, `industry_of`, `rng_for`
- Produces:
  - `null_random_topk(gt, mask, k, B, rng, chunk=500) -> np.ndarray[B, D]`
  - `null_fixed(gt, mask, k, B, rng) -> np.ndarray[B, D]`
  - `null_label_perm(pred, gt, mask, k, B, rng) -> np.ndarray[B, D]`
  - `null_matched(baskets, gt, mask, strata: np.ndarray[N] int, B, rng) -> np.ndarray[B, D]`
  - `empirical_p(null: np.ndarray, obs: float, tail: str = "upper") -> float`
  - `sr_rows(R: np.ndarray[B,D]) -> np.ndarray[B]`

- [ ] **Step 1: Failing tests**

```python
def test_empirical_p_formula():
    null = np.arange(99, dtype=float)
    assert F.empirical_p(null, 98.0) == pytest.approx(2 / 100)
    assert F.empirical_p(null, 1000.0) == pytest.approx(1 / 100)
    assert F.empirical_p(null, -1.0, tail="lower") == pytest.approx(1 / 100)


def test_nulls_preserve_mask_and_cross_section_and_are_reproducible():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(30, 12)); mask = np.ones((30, 12), bool); mask[:5, 3] = False; mask[29, :] = False
    a = F.null_random_topk(gt, mask, 3, 400, np.random.default_rng(1))
    b = F.null_random_topk(gt, mask, 3, 400, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    valid_mean = np.array([gt[mask[:, d], d].mean() for d in range(12)])
    np.testing.assert_allclose(a.mean(0), valid_mean, atol=0.25)            # unbiased for the valid cross-section mean
    f = F.null_fixed(gt, mask, 3, 200, np.random.default_rng(2))
    assert f.shape == (200, 12)                                               # stock 29 never valid -> never drawn (no NaN)
    assert np.isfinite(f).all()
    pred = rng.normal(size=(30, 12))
    lp = F.null_label_perm(pred, gt, mask, 3, 50, np.random.default_rng(3))
    assert lp.shape == (50, 12) and np.isfinite(lp).all()


def test_matched_null_keeps_strata_counts():
    gt = np.random.default_rng(0).normal(size=(10, 3)); mask = np.ones((10, 3), bool)
    strata = np.array([0] * 5 + [1] * 5)
    baskets = [np.array([0, 1, 5])] * 3
    R = F.null_matched(baskets, gt, mask, strata, 100, np.random.default_rng(4))
    assert R.shape == (100, 3)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
def null_random_topk(gt, mask, k, B, rng, chunk=500):
    """== within-day score permutation null (a uniform random k-subset of valid stocks); identical for any scores."""
    D = gt.shape[1]
    out = np.zeros((B, D))
    for d in range(D):
        idx = np.nonzero(mask[:, d])[0]
        g = gt[idx, d]
        kk = min(k, len(idx))
        for s in range(0, B, chunk):
            n = min(chunk, B - s)
            pick = np.argpartition(rng.random((n, len(idx))), kk - 1, axis=1)[:, :kk]
            out[s:s + n, d] = g[pick].mean(1)
    return out


def null_fixed(gt, mask, k, B, rng):
    """Random fixed k-baskets of stocks valid on every day, equal weight, daily rebalanced."""
    always = np.nonzero(mask.all(axis=1))[0]
    out = np.zeros((B, gt.shape[1]))
    for b in range(B):
        out[b] = gt[rng.choice(always, size=k, replace=False)].mean(0)
    return out


def null_label_perm(pred, gt, mask, k, B, rng):
    """One stock-label permutation per draw, applied on every day: keeps score persistence, breaks score->stock identity."""
    out = np.zeros((B, gt.shape[1]))
    for b in range(B):
        perm = rng.permutation(pred.shape[0])
        out[b] = portfolio(pred[perm], gt, mask, k)[0]
    return out


def null_matched(baskets, gt, mask, strata, B, rng):
    """Per day, replace each selected stock with a random valid stock of the same stratum (industry or beta bucket)."""
    out = np.zeros((B, gt.shape[1]))
    for d, bk in enumerate(baskets):
        if not len(bk):
            continue
        valid = np.nonzero(mask[:, d])[0]
        pools = {s: valid[strata[valid] == s] for s in np.unique(strata[bk])}
        draws = np.stack([rng.choice(pools[strata[i]], size=B) for i in bk], axis=1)
        out[:, d] = gt[draws, d].mean(1)
    return out


def empirical_p(null, obs, tail="upper"):
    null = np.asarray(null)
    hits = (null >= obs).sum() if tail == "upper" else (null <= obs).sum()
    return float((1 + hits) / (len(null) + 1))


def sr_rows(R):
    sd = R.std(axis=1)
    return np.where(sd > 0, R.mean(1) / np.where(sd > 0, sd, 1) * math.sqrt(252), 0.0)
```

  Note on the matched null: sampling within a stratum is with replacement across draws, but a draw can repeat a stock within one basket. Report this as a stated simplification.

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Stage `nulls`.** Statistics and tails are declared here, before computing:
  - **Statistics:**
    - Sharpe (upper)
    - mean daily return (upper)
    - max drawdown (lower = worse; reported, not tested)
    - turnover (reported)
    - mean daily excess over hold-all (upper)
  - **Nulls:**
    - `null_random_topk` (`B_NULL`, computed once, labelled "within-day permutation ≡ random daily top-5")
    - `null_fixed` (`B_NULL`)
    - `null_label_perm` (`B_PERM`, per seed)
    - `null_matched`: industry strata (`B_PERM`), and beta quintile strata from `train_beta` with NaN as its own stratum (`B_PERM`)
  - For every seed and null: the observed percentile, `empirical_p`, and the null 5/50/95%.
  - **Formal family** F1-F4 as in Global Constraints: per-seed p, IUT family p = max over seeds, Holm over F1-F4 (`hypershift.eval.stats.holm`).
  - **Bootstrap:** per seed, a block-bootstrap 95% CI for Sharpe and for excess-over-hold-all Sharpe at `BLOCK` and `BLOCK_SENS`. Use `stats.sharpe_contrast_ci` where it fits; otherwise `stationary_bootstrap_indices`.
  - Statement in the report: "seeds are repeated runs on the same 237 days; they are not independent samples".

  Write `docs/phase1_5a/nulls.json` and `docs/figures/phase1_5a_nulls.png` (null histograms with the 5 seed values marked, one panel per null).

- [ ] **Step 6: Gate A write-up (end of batch 1).** Create `docs/phase1_5a/REPORT_2017.md` with section `## Gate A`, summarising Tasks 1, 2, 2B and 5 in tables plus a 10-line interpretation. **Predeclared Gate A outcomes**, checked in this order (more than one can hold):
  1. **Tie/index decisive:** in every primary seed, both the random-tie median Sharpe and the reverse-index Sharpe are at or below the hold-all Sharpe of the same days, while the stable Sharpe is above it.
  2. **Exposure (B) sufficient:** in every primary seed, the Sharpe lies inside the central 90% of the beta-matched **or** industry-matched null (per-seed p > 0.05).
  3. **Factor tilt sufficient:** in every primary seed, a proxy or `fitted` portfolio has Sharpe ≥ 0.8 × the model's, **and** the `resid` portfolio lies inside the central 90% of the random null.
  4. **Not decisive:** none of 1-3 hold.

  If 1, 2 or 3 holds: document the mechanism immediately. Batch 2 then runs only Task 3 (integrity), Task 6 items 4-5 (margin buckets and top-k, needed to rule out D) and Task 8. Skip the epsilon/jitter curves and Task 4 items 3-4, and record "skipped by sequential stopping rule" in the report.

  If 4 holds: run batch 2 in full.

  Constant/first-5 baskets reproducing about 2 is **additional** artifact evidence, not proof by itself. Never declare an artifact from IC≈0, small amplitude or the tie rate alone.

  **Stop here.** End of batch 1. Report Gate A to the orchestrator and wait for the go-ahead for batch 2.

- [ ] **Step 7: Commit** — `git add` the changed files plus `docs/phase1_5a/nulls.json`, `docs/phase1_5a/REPORT_2017.md` and `docs/figures/phase1_5a_nulls.png`; message `feat(phase1.5a): nulls, block bootstrap, formal family F1-F4, Gate A`.

---

### Task 6: Do tiny score gaps carry information? (Milestone 4)

**Files:**
- Modify: `src/hypershift/eval/forensics.py` (add `topk_diag`, `local_ic`, `calibration`, `margin_buckets`)
- Modify: `scripts/forensic_2017.py` (stage `gaps`)
- Test: `tests/test_forensics.py` (append)

**Interfaces:**
- Consumes: `portfolio`, `portfolio_eps`, `jitter`, `boundary_stats`, `jaccard_series`, `turnover`, `perf`, `null_random_topk`
- Produces:
  - `topk_diag(pred, gt, mask, k) -> dict` with keys `prec_at_k, hit_top10, hit_top20, ndcg_k`
  - `local_ic(pred, gt, mask, q) -> float` (mean daily Spearman within the predicted top-q fraction; days with <5 names or constant scores inside the slice count as 0, the same convention as `metrics.daily_ic`)
  - `calibration(pred, gt, mask, bins) -> np.ndarray[bins]` (mean realised return per within-day score-rank bin; ties get average ranks)
  - `margin_buckets(margin, r, n) -> list[dict]`

- [ ] **Step 1: Failing tests**

```python
def test_topk_diag_on_hand_panel():
    gt = np.array([[.05], [.04], [.03], [.02], [.01], [0.], [-.01], [-.02], [-.03], [-.04]])
    mask = np.ones_like(gt, bool)
    perfect = F.topk_diag(gt.copy(), gt, mask, 2)
    assert perfect["prec_at_k"] == 1.0 and perfect["hit_top10"] == 0.5 and perfect["hit_top20"] == 1.0
    assert perfect["ndcg_k"] == pytest.approx(1.0)
    worst = F.topk_diag(-gt, gt, mask, 2)
    assert worst["prec_at_k"] == 0.0 and worst["hit_top20"] == 0.0


def test_local_ic_and_calibration():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(200, 5)); mask = np.ones_like(gt, bool)
    assert F.local_ic(gt.copy(), gt, mask, 0.2) == pytest.approx(1.0)
    cal = F.calibration(gt.copy(), gt, mask, 10)
    assert np.all(np.diff(cal) > 0)
    const = np.zeros_like(gt)
    assert F.local_ic(const, gt, mask, 0.2) == 0.0


def test_margin_buckets_exact_tie_bucket_separate():
    margin = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.]); r = np.arange(12) / 100
    rows = F.margin_buckets(margin, r, 5)
    assert rows[0]["bucket"] == "exact_tie" and rows[0]["n"] == 2 and sum(x["n"] for x in rows) == 12
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
from scipy.stats import rankdata, spearmanr
from hypershift.eval.metrics import ndcg_at_k


def topk_diag(pred, gt, mask, k):
    prec, h10, h20 = [], [], []
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d])[0]
        if len(idx) <= k:
            continue
        top = select(pred[:, d], idx, k)
        rk = rankdata(-gt[idx, d], method="average")                  # 1 = best realised
        pos = {s: rk[j] for j, s in enumerate(idx)}
        prec.append(np.mean([pos[s] <= k for s in top]))
        h10.append(np.mean([pos[s] <= 0.10 * len(idx) for s in top]))
        h20.append(np.mean([pos[s] <= 0.20 * len(idx) for s in top]))
    return {"prec_at_k": float(np.mean(prec)), "hit_top10": float(np.mean(h10)), "hit_top20": float(np.mean(h20)),
            "ndcg_k": ndcg_at_k(pred, gt, mask.astype(float), k)}


def local_ic(pred, gt, mask, q):
    ics = []
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d])[0]
        top = select(pred[:, d], idx, max(int(round(q * len(idx))), 1))
        p, g = pred[top, d], gt[top, d]
        if len(top) < 5 or np.ptp(p) == 0:
            ics.append(0.0)
            continue
        c = spearmanr(p, g).correlation
        ics.append(float(c) if np.isfinite(c) else 0.0)
    return float(np.mean(ics))


def calibration(pred, gt, mask, bins=CALIB_BINS):
    acc, cnt = np.zeros(bins), np.zeros(bins)
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d])[0]
        if len(idx) < bins:
            continue
        u = (rankdata(pred[idx, d], method="average") - 0.5) / len(idx)
        b = np.minimum((u * bins).astype(int), bins - 1)
        np.add.at(acc, b, gt[idx, d]); np.add.at(cnt, b, 1)
    return acc / np.maximum(cnt, 1)


def margin_buckets(margin, r, n=MARGIN_BUCKETS):
    rows = [{"bucket": "exact_tie", "n": int((margin == 0).sum()), "mean": float(r[margin == 0].mean()) if (margin == 0).any() else float("nan")}]
    nz = np.nonzero(margin > 0)[0]
    edges = np.quantile(margin[nz], np.linspace(0, 1, n + 1))
    lab = np.clip(np.searchsorted(edges, margin[nz], side="right") - 1, 0, n - 1)
    for b in range(n):
        sel = nz[lab == b]
        rows.append({"bucket": f"q{b + 1}", "n": int(len(sel)), "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "mean": float(r[sel].mean()) if len(sel) else float("nan")})
    return rows
```

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Stage `gaps`** (primary run, plus `R5_f_train` HH/EH for comparison). Every curve is shown in full, and no grid value is singled out.
  1. **Margin and spread:** the distributions of `margin5`, `sd` and `ptp`; zero-spread day count; `margin5 / sd`.
  2. **Epsilon:** for each `EPS_ABS` and each `EPS_FRAC`, `R_TIE` draws via `eps_draws` (the fast path, never `portfolio_eps` in a loop):
     - fraction of days whose boundary group has more than one member
     - group-size distribution at the boundary (median, 90%)
     - number of distinct groups among the top 20
     - whether ranks 1-5 are separated from 6-20 (no shared group)
     - mean Jaccard vs stable
     - Sharpe distribution and mean return

     Plot the curves against epsilon: `docs/figures/phase1_5a_eps.png`.
  3. **Jitter:** for each `JITTER_FRAC`, over `R_JIT` draws: Jaccard vs the original baskets, Spearman of jittered vs original scores (mean daily), the Sharpe distribution, mean return, and `hit_top10`. Plot: `docs/figures/phase1_5a_jitter.png`.
  4. **Margin buckets:** `margin_buckets` per seed and pooled. For each bucket, report the next-day top-5 return (mean ± block-bootstrap CI), Sharpe, `hit_top10/20`. Report the Spearman of margin vs next-day top-5 return (non-tie days) with a block-bootstrap CI. Plot: `docs/figures/phase1_5a_margin.png`.
  5. **Top-k:** for each `K_GRID`:
     - perf (mean, sr, cumret, vol, mdd), turnover, persistence (mean Jaccard)
     - `topk_diag`
     - excess over the random-k null mean (`null_random_topk` with that k, `B_NULL // 10` draws)

     k=5 is labelled primary; the others are diagnostic.
  6. **Ranking diagnostics:**
     - global IC (`metrics.daily_ic`)
     - `local_ic` for `LOCAL_IC_Q`
     - `calibration` deciles: plot `docs/figures/phase1_5a_calib.png`, with the realised mean per predicted decile ± 1.96·SE (day-clustered)
  7. **Reconciliation paragraph** in the report: does top-tail skill (`hit_top10`, the top calibration decile) square with NDCG@5 ≈ random (0.5694 vs 0.5636)? State the arithmetic.

- [ ] **Step 6: Commit** — message `feat(phase1.5a): epsilon/jitter/margin/top-k/local-IC/calibration diagnostics`.

---

### Task 7: Epochs, seeds, controls (Milestone 5)

**Files:**
- Modify: `scripts/forensic_2017.py` (stage `trajectory`)

**Interfaces:**
- Consumes: `RunArrays.history`, `RunArrays.metrics`, outputs of Tasks 4-5

- [ ] **Step 1: Stage `trajectory`.**
  1. From `history.jsonl` for the primary run and `R5_f_train` HH/EH, seeds 0-4, plot per epoch: test `sr`, val `sr`, `test_ic`, `ndcg5`, `test_pred_sd`, `train_loss` → `docs/figures/phase1_5a_trajectory.png`.
     - Caption: "epoch 0 = after the first training pass (loop.py:211-243), not an untrained model".
     - Mark the selected epoch and `test_oracle_epoch` (diagnostic).
  2. Table: test Sharpe at epoch 0, at the selected epoch and at the last epoch; correlation over epochs of test Sharpe with `test_pred_sd` and with `test_ic`.
  3. "Not answerable from artifacts" list, copied verbatim into the report:
     - basket identities, margins and tie rates at non-selected epochs
     - overlap of epoch-0 vs selected baskets
     - through-model index permutation
     - frozen-model 2018+ evaluation
  4. **Seeds:** reuse Task 4 item 5. Add a sentence that agreement across seeds under the same ordering/backtester is not independent evidence.
  5. **Early vs late selected epochs, cross-seed only.** Read each seed's `best_epoch` from the inventory; S12 expects [8, 32, 0, 1, 2]. Compare seeds selected at epoch ≤ 2 against seeds selected at epoch ≥ 8 on:
     - tie rate, score SD, `margin5`
     - top-5 concentration (`top_share`)
     - Jaccard between their baskets
     - Sharpe, and the factor-proxy Spearman from Task 2B

     State plainly that this is a between-seed comparison: it is confounded with seed and is not a within-run trajectory.
  5. **Weight-decay control:** quote `docs/phase1_5/F_learnability.md` lines 7 and 11-12 (one-factor, wd 5e-4 vs 0, same data/seeds/code path, IC/oracle 26% → 63-84%, decay gradient 10-40× the loss gradient, `|z|` shrinkage). State the narrow conclusion: "wd 5e-4 impaired planted-signal learnability; this does not attribute the real-data Sharpe change to wd (input mode, log_ic and seed count also changed)". Recommend no new run.

- [ ] **Step 2: Commit** — message `feat(phase1.5a): epoch trajectory, seed stability, wd-control statement`.

---

### Task 8: Report, decision table, costed proposal, doc pointers

**Files:**
- Modify: `scripts/forensic_2017.py` (stages `report` and `all`)
- Create/complete: `docs/phase1_5a/REPORT_2017.md`, `docs/phase1_5a/PROPOSAL_followups.md`
- Modify: `docs/phase1_5/PHASE1_5_SUMMARY.md`, `docs/PHASE1_TRACKER.md`, `CLAUDE.md` (one pointer line each)
- Test: `tests/test_forensics.py` (append the integration test)

- [ ] **Step 1: Integration test** on a synthetic fixture run folder, written with `tmp_path` in the run-folder contract (`config.json`, `history.jsonl`, `metrics.json`, `test_{pred,gt,mask,daily}.npy`; 40 stocks × 30 days; seed folders 0-1). Run every stage with `--root tmp_path --runs FIX/HH --seeds 0-1 --quick` (`--quick` divides every B/R by 100). Assert that:
  - each stage's json exists and has its top-level keys
  - the fixture's files are unchanged (sha256 before = after)

```python
import hashlib, json, subprocess, sys
import numpy as np


def _fixture(root):
    rng = np.random.default_rng(0)
    for s in (0, 1):
        p = root / "FIX" / "HH" / f"seed_{s}"; p.mkdir(parents=True)
        pred = rng.normal(size=(40, 30)).astype(np.float32); gt = rng.normal(0, .01, size=(40, 30)); mask = np.ones((40, 30), np.float32)
        from hypershift.eval.metrics import topk_daily_returns, evaluate_all
        np.save(p / "test_pred.npy", pred); np.save(p / "test_gt.npy", gt); np.save(p / "test_mask.npy", mask)
        np.save(p / "test_daily.npy", topk_daily_returns(pred, gt, mask, 5))
        (p / "metrics.json").write_text(json.dumps({"best_epoch": 0, "test": evaluate_all(pred, gt, mask), "test_oracle_epoch": 0, "epochs_run": 1}))
        (p / "config.json").write_text(json.dumps({"weight_decay": 0.0, "input_mode": "relative", "alpha": 0.0, "norm": "train", "topk": 5, "seed": s}))
        (p / "history.jsonl").write_text(json.dumps({"epoch": 0, "train_loss": 0.0, "val": {"sr": 0.0}, "test": {"sr": 0.0, "ndcg5": 0.5}, "test_ic": 0.0, "test_pred_sd": 1.0}) + "\n")


def test_forensic_script_end_to_end_on_fixture(tmp_path):
    _fixture(tmp_path)
    h = lambda: {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (tmp_path / "FIX").rglob("*") if f.is_file()}
    before = h()
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "scripts/forensic_2017.py", "--stage", "all", "--root", str(tmp_path), "--runs", "FIX/HH",
                        "--seeds", "0-1", "--quick", "--fixture", "--out", str(out), "--docs", str(out / "docs")],
                       capture_output=True, text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stderr[-2000:]
    for stage in ("inventory", "mechanism", "proxy", "decompose", "nulls", "gaps", "trajectory"):
        assert (out / "docs" / f"{stage}.json").exists(), stage
    assert h() == before
```

  `--fixture` skips the steps that need real NYSE data:
  - date mapping, tickers, integrity, the isolated-node check: replaced by index-based placeholders
  - the `proxy` stage: writes `{"skipped": "fixture"}`
  - beta/industry strata for the matched nulls: replaced by index-based strata (`index % 3`)
  - momentum: replaced by index-based placeholders

  Everything replaced is labelled `fixture` in the json. `--docs` redirects the doc outputs. Add both flags to the script.

- [ ] **Step 2: Report stage** fills `docs/phase1_5a/REPORT_2017.md` from the stage jsons. Sections:
  1. **Executive summary:** at most 10 lines answering "why Sharpe≈2".
  2. **Reproduction table by seed:** selected epoch, Sharpe, mean, vol, mdd, IC, NDCG@5, spread, spread/realised spread, tie rate, turnover, hold-all Sharpe.
  3. **Evidence file index:** paths and hashes.
  4. Gate A (Tasks 1, 2, 2B, 5), Integrity (Task 3), Decomposition (Task 4), Tiny gaps (Task 6), Trajectory/seeds/wd control (Task 7). Each has its figure, a table and a "what this shows / what it does not" pair.
  5. **Decision table**, with exactly 4 rows (A tie/index, B lucky/concentrated/exposure, C data/backtest, D genuine top-tail) and the columns Evidence for | Evidence against | Unresolved.
  6. **Verdict** sentence: "Sharpe≈2 is / is not evidence of learned stock ranking", with the justified uncertainty: formal F1-F4 Holm p, bootstrap CIs, null percentiles.
  7. **Limitations:** the artifact inventory "not recoverable" list; exploratory year; multiplicity statement (formal family F1-F4 only); market = equal-weight hold-all (cap-weighted UNKNOWN).
  8. **Test results:** the exact commands and pass/fail counts from Step 3.

- [ ] **Step 3: Full test run** and record the output verbatim in the report:

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests/test_forensics.py tests/test_think_equivariance.py tests/test_metrics.py tests/test_phase15_eval.py tests/test_loop.py -q
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest -m data tests/test_forensics_artifacts.py tests/test_phase15_eval.py tests/test_phase15_data.py -q
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest -q
```

  Expected: all pass. Any failure is recorded with its output and fixed before the commit.

- [ ] **Step 4: One reproducible command,** documented at the top of the report:

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/forensic_2017.py --stage all
```

- [ ] **Step 5: `docs/phase1_5a/PROPOSAL_followups.md`** contains costed, gated items, each with information value, cost, prerequisites and what it can and cannot claim:
  1. **Instrumented replication of R5_f2 alpha=0 HH**, 5 seeds × 40 epochs, on Kaggle.
     - Requires a `loop.py` option to save `state_dict` and test predictions every epoch, plus a resumable checkpoint.
     - About 24 s/epoch on a Kaggle P100: 16 min/seed, about 1.5 GPU-h including the smoke run.
     - Enables the per-epoch basket trajectory, the through-model index permutation and a frozen-weights artifact.
     - It is a **replication**, not the historical run.
  2. **Post-2017 data-compatibility spec**, CPU, about 1 day of work. It covers:
     - ticker/identity mapping, delistings and mergers
     - close vs adjusted-close semantics, MA5/10/20/30, return definition
     - causal normalisation, and a test that future prices cannot change earlier features
     - survivorship quantification

     Only then a first-pass 2018-2023 evaluation of the model from item 1: per-year and concatenated-daily Sharpe, never the mean of annual Sharpes. A frozen graph and a historically updated graph are kept separate.
  3. **Walk-forward retraining.** A separate experiment; GPU cost about 6 windows × 5 seeds × about 40 min ≈ 20 GPU-h.
  4. **Architecture/geometry controls** (EE/temporal-only/no-graph/pairwise/hypergraph/simple baselines), plus a hyperbolic-op audit against the paper (`docs/phase1_5/B_model.md` is the start). Only after evaluation is trusted.

  Every long run must checkpoint, save per-epoch predictions and logs, and run without any Claude session or watcher.

- [ ] **Step 6: Pointers.** Add one line to each of:
  - `docs/phase1_5/PHASE1_5_SUMMARY.md`: "Phase 1.5a (2017 Sharpe≈2 forensics): see `docs/phase1_5a/REPORT_2017.md`"
  - `docs/PHASE1_TRACKER.md`: the same line
  - the `CLAUDE.md` study status

  Change no verdict text.

- [ ] **Step 7: Commit and push**

```bash
git add scripts/forensic_2017.py src/hypershift/eval/forensics.py tests/test_forensics.py docs/phase1_5a/ docs/figures/phase1_5a_*.png docs/phase1_5/PHASE1_5_SUMMARY.md docs/PHASE1_TRACKER.md CLAUDE.md
git commit -m "docs(phase1.5a): 2017 Sharpe≈2 forensic report, decision table, costed follow-up proposal"
git push origin tom-shlom
```

---

## Self-review (done at authoring)

- **Spec coverage:**

  | Requirement | Task |
  |---|---|
  | M1 | T1 |
  | M2 | T2 (index/tie), T3 (integrity), T4 (decomposition/exposure/costs/benchmarks) |
  | M3 | T5 |
  | M4 | T6 |
  | M5 | T7 |
  | Tests 1-7 | T1 (1, 7), T2 (2, 4), T5 (3), T4 (5), T6 (6), T8 integration |
  | Sequential stop | Gate A at the end of T5 (batch 1) |
  | Deliverable + proposal | T8 |

  Phases 9-12 of brief 2 are deliberately mapped to S6 (already done) or to T8's proposal (gated).
- **Placeholders:** none. Two places name existing functions to reuse instead of re-implementing them (`stats.sharpe_contrast_ci`, `stats.stationary_bootstrap_indices`, `stats.holm`). All exist in `src/hypershift/eval/stats.py`.
- **Type consistency:** masks are bool inside `forensics`; the conversion to float happens only when calling `metrics.*`. `portfolio` returns `(r, baskets)` everywhere.
- **Review Focus:** items 1-5 are pinned in T2, T1/T4 (via `test_fewer_than_k_valid_stocks` and the fixed-null "always valid" test in T5), T3, T1 Step 5 and T2.

## Execution notes (cost)

- **Compute:** CPU only, about 20-40 min wall in total. The heaviest step is `null_label_perm`: 2,000 × 5 seeds × 237 days of sorts over 1,737 stocks. `--quick` exists for debugging.
- **Batching for token efficiency:** batch 1 runs, in this order, T1, T2, T2B, T5 (its matched nulls need `train_beta`/`industry_of` from T2B), then stops at Gate A and reports. Batch 2 runs T3, T4, T6, T7, T8, trimmed per the Gate A outcome. The worker never polls or waits on anything external.
