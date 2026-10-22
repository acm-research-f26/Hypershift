# Simple-Model Feature Benchmark (Study FB) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether simple models (Ridge, LightGBM, small MLP) on richer price/volume/volatility/liquidity features, different targets, horizons and feature representations extract factor-neutral cross-sectional signal on the S&P 500 panel, then freeze one specification for a prospective test.

**Architecture:** A new package `src/hypershift/featbench/` builds causal per-(stock, date) features from the aligned OHLCV array, labels for any horizon, three targets, and per-date transforms. A CLI `scripts/featbench.py` runs walk-forward folds 2019–2023 and writes run folders in the screening contract (`config.json`, `metrics.json`, `predictions.npz`, `daily.npz`). `scripts/featbench_analysis.py` produces the tables and verdicts. Portfolio and subgroup studies reuse frozen predictions without retraining.

**Tech Stack:** Python 3 in `.venv`, numpy, scipy, scikit-learn (Ridge, MLPRegressor), LightGBM (new dependency, CPU), existing `hypershift.screening` (load_panel, rank_metrics, trading).

**Spec:** the user's proposal of 2026-10-22 ("The 7 highest-value experiments HyperShift hasn't properly tested", pasted in session) and the retrospective doc *HyperShift Research Retrospective* (claude.ai artifact U4q9KoJyuyiShReCP63ozr), section H.

## Global Constraints

- **Status: exploratory.** Every year 2015–2023 has been inspected before; nothing scored here is confirmatory. The confirmatory step is the prospective test (Task 9) on data after the 2026-10-22 freeze.
- **Never score 2024 or later** for any FB arm. The Study J amendment spent 2024+ for the J arms only. Code must raise if an evaluation year > 2023 is requested (`--year` guard), and the prospective scorer must refuse signal dates ≤ 2026-10-22.
- Development folds: evaluate 2019, 2020, 2021, 2022, 2023. Fold Y trains on labels maturing before Y−1-01-01, validates on signal dates in Y−1 whose labels mature before Y-01-01, evaluates on signal dates in Y whose labels mature before Y+1-01-01 (`hypershift.screening.data.split_signals`, horizon-aware, training stride 1).
- Universe and eligibility: the 462-stock panel `data/fresh/sp500_daily/screen_sp500_2015_2026.npz` with `identity_ok`, 64-session eligibility (`Panel.eligible`), OHLCV from `data/fresh/sp500_daily/ohlcv_sp500_2015_2026.npz` (columns open, high, low, close, volume; adjusted; aligned to panel order).
- Primary metric: **factor-neutral RankIC** = daily Spearman between prediction and the h-day forward return after removing trailing-126-session beta × equal-weight market return and then demeaning within GICS sector (generalises `Panel.target('sector')` to horizon h). Secondary: raw RankIC, style-neutral RankIC (vs target C), NDCG@10 excess, spread10, turnover, net Sharpe (top-10, 10 bp per side).
- Statistics: per fold, the daily paired ΔRankIC series; stationary bootstrap block max(10, 2h), B = 5000 (`screen_analysis.bootstrap_means`); pooled = equal-weight mean of fold means; two-sided bootstrap p; Holm within each predeclared family.
- LightGBM fallback (predeclared): if `import lightgbm` fails after 3 retries (Smart App Control), `lgbm` means `sklearn.ensemble.HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, min_samples_leaf=1000, l2_regularization=1.0, max_iter=1000)` with the same validation-RankIC early stopping (checked every 10 iterations via `warm_start`).
- Compute: `--jobs 3`; LightGBM `num_threads = 16 // jobs`; runs longer than 10 minutes go in foreground chunks under 10 minutes (resumable via the `metrics.json` skip) or a Task Scheduler queue; MLP arms run last.
- Spec copy: the pasted proposal is saved verbatim to `docs/featbench_20261022/PROPOSAL.md` in Task 1.
- **Verdict (exploratory):** SUPPORTED = Holm p < 0.05, pooled CI excludes 0, and ≥ 4 of 5 fold means share the sign of the pooled mean. Otherwise NO EVIDENCE. Deterministic models have no seed axis; LightGBM uses `deterministic=True`, seed 7, with a 3-seed sensitivity for the primary LightGBM arm.
- Fixed hyperparameters (no tuning beyond what is listed): Ridge alpha chosen from {0.1, 1, 10, 100, 1000} by validation mean neutral RankIC. LightGBM: `objective='regression'`, `num_leaves=15`, `max_depth=4`, `learning_rate=0.05`, `n_estimators=1000`, `min_child_samples=1000`, `subsample=0.7`, `subsample_freq=1`, `colsample_bytree=0.7`, `reg_lambda=1.0`, `deterministic=True`, `force_row_wise=True`, `seed=7`, early stopping after 50 rounds without improvement in validation mean neutral RankIC (evaluated every 10 rounds). MLP: `MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-4, learning_rate_init=1e-3, max_iter=50, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5)`, seeds 7/19/42, mean prediction.
- Default representation `cs_z` (per-date cross-sectional z-score over eligible stocks, clipped ±5, missing → 0). Default target A (raw), default horizon 5.
- Results root `results/FEATBENCH_SP500_20261022`; docs `docs/featbench_20261022/`. Run folders never overwritten (skip if `metrics.json` exists).
- Commit locally on `tom-shlom`; never push. Leave `docs/HANDOFF.md` uncommitted (other sessions' edits). CPU only (`CUDA_VISIBLE_DEVICES=-1`). No paid or restricted data; no new downloads except the LightGBM wheel.
- Run unattended until 2026-10-22 08:00. On a blocker, record it in `docs/featbench_20261022/PROGRESS.md` and continue with the next independent task.

## Review Focus

1. **Look-ahead in features:** a feature at signal t must not change when any data after t changes → `test_features_causal` mutates data after t and asserts features ≤ t are identical.
2. **Label/validation overlap for long horizons:** with h = 20 the training labels must mature before the validation year → `test_split_purged_for_horizon` asserts `max(train + h) < first validation date index`.
3. **Stocks with zero or missing volume:** dollar volume and Amihud must stay finite → `test_features_finite_with_zero_volume`.
4. **A fold whose eval year > 2023 or a prospective date ≤ freeze:** must raise → `test_year_guard` and `test_prospective_guard`.
5. **Constant predictions on a day (e.g. Ridge collapses):** RankIC on that day is 0, not NaN → `test_neutral_rankic_constant_prediction`.

---

### Task 1: Dependency, data audit and power analysis (E0)

**Files:**
- Create: `scripts/featbench_audit.py`
- Create: `docs/featbench_20261022/DATA_AUDIT.md`, `docs/featbench_20261022/POWER.md`, `docs/featbench_20261022/PROGRESS.md`
- Modify: `requirements` file if one exists (add `lightgbm`)

**Interfaces:**
- Produces: `docs/featbench_20261022/POWER.md` with the minimum detectable ΔRankIC (MDE, 80% power, two-sided α 0.05) for 5 folds × ~250 days, and the number of prospective days needed for MDE 0.01 and 0.02.

- [ ] **Step 1:** `.venv/Scripts/python.exe -m pip install lightgbm`; verify `import lightgbm`.
- [ ] **Step 2: Audit script.** Report for the OHLCV array on panel-eligible cells: share of NaN or zero volume, `low ≤ min(open, close)` and `high ≥ max(open, close)` violations, close vs panel close max relative error, ticker/date alignment with the panel after `identity_ok`, split-adjustment check (count of |daily volume log-change| > 3 coinciding with |price log-change| > 0.4). State survivorship (current constituents) and the absence of free point-in-time fundamentals or shares outstanding (so "size" = 20-day mean dollar volume).
- [ ] **Step 3: Power.** From saved daily series of existing deterministic baselines (v3 `reversal` and `ridge`, folds 2022/2023, `results/SCREEN_SP500_20261019_v3/*/{reversal,ridge}/seed_7/daily.npz`), compute the daily ΔRankIC SD and the block-bootstrap SE of its mean; scale to N days; MDE = 2.8 × SE. Write both numbers and the method to POWER.md.
- [ ] **Step 4: Commit** the audit script, the three docs and this plan (the plan is the predeclaration; commit before any FB number).

### Task 2: Features, labels, targets and transforms

**Files:**
- Create: `src/hypershift/featbench/__init__.py`, `src/hypershift/featbench/data.py`
- Test: `tests/test_featbench.py`

**Interfaces:**
- Produces:
  - `FEATURES: list[str]` (20 names, order fixed): `ret_1, ret_3, ret_5, ret_10, ret_20, ret_60, mom_20_60, vol_5, vol_20, vol_60, downvol_20, mdd_60, beta, log_dvol_20, relvol_5_60, volchg_5_20, amihud_20, hl_range_20, sec_ret_5, sec_ret_20`. Definitions: `ret_k` = close_t/close_{t−k} − 1; `mom_20_60` = close_{t−20}/close_{t−60} − 1; `vol_k` = std of the last k daily returns; `downvol_20` = sqrt(mean(min(r, 0)²)) over 20; `mdd_60` = min over 60 sessions of close/running max − 1; `beta` = `Panel.beta`; `log_dvol_20` = log(1 + mean(close × volume) over 20); `relvol_5_60` = log((mean vol 5 + 1)/(mean vol 60 + 1)); `volchg_5_20` = log((mean vol 5 + 1)/(mean vol 20 + 1)); `amihud_20` = mean(|r| / (close × volume + 1)) × 1e9 over 20; `hl_range_20` = mean((high − low)/close) over 20; `sec_ret_k` = ret_k − eligible-sector mean of ret_k at t.
  - `FB(year: int, horizon: int = 5)` dataclass loaded by `load_fb(year: int, horizon: int = 5) -> FB` with fields `panel` (screening `Panel`), `X: np.ndarray [N, T, 20] float32`, `y: np.ndarray [N, T]` (h-day forward return), `y_ok: np.ndarray[bool]`, `neutral: np.ndarray [N, T]` (beta-market then sector demeaned, horizon h), `train, valid, evaluate: np.ndarray[int]` signal indices; raises `ValueError` if `year > 2023`.
  - `target(fb: FB, signals, kind: str) -> (y [S, N], mask [S, N])` for kind ∈ {`raw`, `sector`, `residual`}; `residual` = per-date OLS of the forward return on [1, beta, sector dummies, z(log_dvol_20), z(ret_60), z(vol_60)] over eligible stocks, residual returned.
  - `transform(X_s: [S, N, F], mask: [S, N], kind: str, fb, signals) -> [S, N, F]` for kind ∈ {`raw`, `cs_rank`, `cs_z`, `sector_z`, `volscaled_z`}. `raw` = global standardisation with train-period mean/std (passed via the fb object); `cs_rank` = per-date percentile rank − 0.5; `cs_z` = per-date z clipped ±5; `sector_z` = per-date z after subtracting the sector mean; `volscaled_z` = return features (`ret_*`, `mom_20_60`, `sec_ret_*`) divided by `vol_60` before `cs_z`.
  - `neutral_rankic(pred [S, N], neutral [S, N], mask [S, N]) -> np.ndarray [S]` (daily Spearman; 0 for a constant prediction).

- [ ] **Step 1: Write failing tests** `test_features_causal` (mutate OHLCV for t > 300 in a small synthetic panel; features at t ≤ 300 identical), `test_features_finite_with_zero_volume`, `test_split_purged_for_horizon` (h = 20: `(train + 20).max() < valid.min()` index-wise), `test_year_guard` (`load_fb(2024)` raises), `test_neutral_rankic_constant_prediction` (returns 0), `test_residual_target_orthogonal` (residual has ~0 correlation with beta and sector dummies per date, |corr| < 1e-6), `test_transform_cs_z_moments` (per-date mean ≈ 0, std ≈ 1 on eligible).
- [ ] **Step 2:** Run `pytest tests/test_featbench.py -q`; expect failures (module missing).
- [ ] **Step 3: Implement** `src/hypershift/featbench/data.py`. Features are computed once for the full date range and truncated by the fold through `load_panel(..., year)`. Cache the feature tensor per year under `results/FEATBENCH_SP500_20261022/_cache/` keyed by source hash.
- [ ] **Step 4:** Run the tests; expect PASS. Run the full `pytest -q tests/test_screening.py tests/test_featbench.py`.
- [ ] **Step 5: Commit.**

### Task 3: Models and the run CLI

**Files:**
- Create: `src/hypershift/featbench/models.py`, `scripts/featbench.py`
- Test: `tests/test_featbench.py` (append)

**Interfaces:**
- Consumes: Task 2 `load_fb`, `target`, `transform`, `neutral_rankic`, `FEATURES`.
- Produces:
  - `fit_predict(model: str, Xtr, ytr, mtr, Xva, nva, mva, Xev, seed: int) -> (pred_ev [S, N], info: dict)` for model ∈ {`ridge`, `lgbm`, `mlp`}; selection uses validation mean neutral RankIC (`nva` = neutral validation target).
  - Lag-block scorers for E1: `lagblock(fb, signals, block: str) -> [S, N]` for block ∈ {`l1` (r_{t−1}), `l2_5` (r_{t−5..t−1} excluding the last day, i.e. close_{t−1}/close_{t−5} − 1), `l6_20` (close_{t−5}/close_{t−20} − 1), `l21_60` (close_{t−20}/close_{t−60} − 1)}; sign chosen by the sign of the training-period mean neutral RankIC; plus `reversal16` = −(close_t/close_{t−16} − 1).
  - CLI: `python scripts/featbench.py --years 2019 2020 2021 2022 2023 --arms <arm ...> [--root R]` where an arm name encodes `model:features:target:horizon:repr` (e.g. `lgbm:rich:raw:5:cs_z`, `ridge:lags:raw:5:cs_z`, `block:l1:raw:5:-`). Feature sets: `rich` (all 20), `lags` (the four lag blocks only), `price` (`ret_*`, `mom_20_60`, `vol_*`, `downvol_20`, `mdd_60`, `beta`, `sec_ret_*`: no volume/liquidity). Writes `<root>/<year>/<arm>/seed_7/{config.json, metrics.json, predictions.npz, daily.npz}`; `metrics.json` holds `RankIC`, `neutral_RankIC`, `style_neutral_RankIC`, `NDCG10_excess`, `spread10`, and for h = 5 the `trading()` block; `daily.npz` holds daily `RankIC`, `neutral_RankIC`, `style_neutral_RankIC` and `dates`. Skips existing `metrics.json`. `--jobs n` runs arms in parallel processes.
- [ ] **Step 0: Anchor test** `test_reversal16_matches_v3`: `block:reversal16` at h = 5 on folds 2022 and 2023 reproduces the v3 `reversal` arm's raw RankIC (+0.0153 / +0.0057) and neutral RankIC from `docs/screening_20261019/SCREENING_RESULTS.csv` to 1e-6 (same panel, eligibility, signal dates). Marked slow; runs once.
- [ ] **Step 1: Failing test** `test_cli_smoke_planted`: on a synthetic FB with a planted signal in `ret_5`, `fit_predict('ridge', …)` and `'lgbm'` reach validation neutral RankIC > 0.3, and the CLI writes all four files for one arm (use a `--max-days` smoke option whose root must contain `smoke`).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3: Implement.** Ridge on flattened eligible rows with `sklearn.linear_model.Ridge`; LightGBM via `lightgbm.LGBMRegressor` with a callback computing validation mean neutral RankIC every 10 iterations and keeping the best iteration; MLP via `MLPRegressor`. Training rows: all eligible (stock, signal) pairs with mature labels; targets standardised by the training std.
- [ ] **Step 4:** Run tests; expect PASS. Smoke on real data: `--years 2023 --arms ridge:rich:raw:5:cs_z --max-days 30 --root results/FEATBENCH_smoke`.
- [ ] **Step 5: Commit.**

### Task 4: Lag decomposition (E1) and the feature benchmark (E2)

**Files:**
- Create: `scripts/featbench_analysis.py`
- Create: `docs/featbench_20261022/fb_TABLES.md`, `fb_PAIRED.csv` (generated)

**Interfaces:**
- Consumes: Task 3 CLI and run folders.
- Produces: `featbench_analysis.py --root R --out D --tag T` writing the families below with the Global Constraints statistics; functions `paired(root, folds, a, b, key='neutral_RankIC') -> dict` (per-fold means, pooled, CI, p) and `verdict(holm_p, ci, fold_means) -> str`.

Arms (all h = 5, target raw, repr cs_z unless named):
- E1: `block:l1`, `block:l2_5`, `block:l6_20`, `block:l21_60`, `block:reversal16`, `ridge:lags`.
- E2: `ridge:rich`, `lgbm:rich`, `mlp:rich`, `ridge:price`, `lgbm:price`.

Families:
- **Primary (Holm over 3):** `lgbm:rich − ridge:rich`; `ridge:rich − block:reversal16`; `lgbm:rich − block:reversal16`.
- **Secondary (Holm over 3):** `lgbm:rich − lgbm:price` (value of volume/liquidity); `mlp:rich − ridge:rich`; `ridge:lags − block:reversal16`.
- E1 descriptive: absolute neutral RankIC and CI for each lag block and `ridge:lags`.
- LightGBM 3-seed sensitivity: `lgbm:rich` seeds 19 and 42 (folder `seed_k`), descriptive.

- [ ] **Step 1:** Run E1 + E2 arms for 2019–2023 with `--jobs 3`.
- [ ] **Step 2:** Implement and run the analysis; test `test_paired_matches_manual` (pooled = mean of fold means, recomputed by hand on two folds) in `tests/test_featbench.py`.
- [ ] **Step 3:** Spot-check one number by hand from `predictions.npz` (e.g. `lgbm:rich` 2021 neutral RankIC).
- [ ] **Step 4: Commit** code, tables and PROGRESS line.

### Task 5: Targets (E3 of the proposal: raw vs sector-relative vs factor-residual) and horizons

**Interfaces:**
- Consumes: Task 3 CLI.

Arms: `{ridge,lgbm}:rich:{raw,sector,residual}:5:cs_z` (raw already run) and `{ridge,lgbm}:rich:raw:{1,10,20}:cs_z`.

Families (each Holm over its members; same verdict rule):
- **Targets:** `M:rich:sector − M:rich:raw`, `M:rich:residual − M:rich:raw` for M ∈ {ridge, lgbm} (4 contrasts), all scored on neutral RankIC at h = 5, plus style-neutral RankIC descriptively.
- **Horizons:** descriptive only (contrasts across horizons compare different targets). Report per horizon: neutral RankIC with CI, effective independent observations (days / h), turnover; no cross-horizon verdicts. Each horizon is also compared with `block:reversal16` scored at the same horizon (descriptive).

- [ ] **Step 1:** Run arms (`--jobs 3`).
- [ ] **Step 2:** Extend the analysis to the two families; regenerate tables.
- [ ] **Step 3: Commit.**

### Task 6: Feature representations (E4)

Arms: `{ridge,lgbm}:rich:raw:5:{raw,cs_rank,sector_z,volscaled_z}` (cs_z already run).

Family (Holm over 8): each representation − `cs_z` for each model. Report which representation is best per model, descriptively.

- [ ] **Step 1:** Run arms.  - [ ] **Step 2:** Analysis.  - [ ] **Step 3: Commit.**

### Task 7: Portfolio construction on frozen predictions (E5) and subgroup predictability (E6)

**Files:**
- Create: `src/hypershift/featbench/portfolio.py`, `scripts/featbench_portfolio.py`
- Test: `tests/test_featbench.py` (append)

**Interfaces:**
- Produces: `simulate(weights: [S, N] per signal day, fb, hold: int = 5, cost: float = 0.001) -> dict` returning `gross_Sharpe, net_Sharpe, ann_return, max_drawdown, turnover, beta_to_market` with daily book = mean of the last `hold` sleeve weight vectors (no drift; turnover = Σ|w_t − w_{t−1}|; cost = cost × turnover); and `weights_rule(pred, fb, signals, rule)` for rule ∈ {`A_top10` (equal-weight long top 10), `B_sector_ls` (long top 20%, short bottom 20% within each sector, equal weight, gross 1 per side), `C_volscaled` (top 10 weighted ∝ 1/vol_60), `D_beta_ls` (long top decile, short bottom decile, scaled so net beta = 0), `E_buffer` (top-10 entry, keep until rank > 20)}.
- Subgroups: `subgroup_rankic(pred, fb, signals, char: str) -> {tercile: daily neutral RankIC}` with per-date terciles of `log_dvol_20`, `vol_60`, `beta`, `amihud_20` measured at t.

Predictions used: `lgbm:rich`, `ridge:rich`, `block:reversal16` (h = 5, raw, cs_z), folds 2019–2023. No retraining.

- [ ] **Step 1: Failing tests** `test_simulate_constant_weights_turnover_zero` and `test_beta_ls_zero_beta` (net beta of D weights < 1e-8).
- [ ] **Step 2:** Implement; tests pass.
- [ ] **Step 3:** Run; write `docs/featbench_20261022/fb_PORTFOLIO.md` (rules × models × folds, pooled Sharpe; descriptive) and `fb_SUBGROUP.md` (tercile RankIC with CI; descriptive; flagged as hypothesis-generating only).
- [ ] **Step 4: Commit.**

### Task 8: Learned-hyperedge collapse fix, stage 1 only (E7; time-boxed)

Start only if Tasks 1–7 are done before 05:30. Otherwise record "not started" in PROGRESS.md.

**Files:**
- Modify: `src/hypershift/screening/models.py` (soft assignment: add `temperature`, prototype initialisation from the training clusters, and an entropy-balance penalty, all behind new arm names `soft_t`, so existing arms stay bit-identical)
- Test: `tests/test_screening.py` (append `test_soft_t_recovers_planted_groups`: synthetic panel with 4 planted groups sharing a group-level return signal; after training, assignment entropy < 0.5 × log(16) and the adjusted Rand index of argmax assignments vs planted groups > 0.5)

- [ ] **Step 1:** Failing test.  - [ ] **Step 2:** Implement.  - [ ] **Step 3:** Pass; confirm `static` output on `2022/static/seed_7` is unchanged (bit-identical predictions vs v3).  - [ ] **Step 4: Commit.** Real-data stage 2 is deferred to a separate predeclared study.

### Task 9: Freeze and prospective protocol

**Files:**
- Create: `docs/featbench_20261022/PROSPECTIVE_PROTOCOL.md`, `docs/featbench_20261022/FREEZE_MANIFEST.json`, `scripts/featbench_prospective.py`
- Test: `tests/test_featbench.py` (append `test_prospective_guard`)

**Interfaces:**
- Selection rule (predeclared now): the frozen specification is the **h = 5** arm with the highest pooled neutral RankIC across 2019–2023 among Ridge/LightGBM arms of Tasks 4–6 (other horizons are not comparable and are excluded); ties within 0.001 → the simpler model (Ridge). Comparators: `ridge:rich:raw:5:cs_z` (if not selected), `block:reversal16`, random.
- Hyperparameters are frozen from the development folds (Ridge alpha and LightGBM rounds = median of the 2019–2023 selected values) and recorded in `FREEZE_MANIFEST.json`. The prospective retrain uses **no validation step**, so no FB metric is ever computed on 2024+ (the J-only period).
- Prospective scorer: trains the frozen spec on all labels maturing before 2026-10-23 using a freshly fetched panel extension (yfinance, the same pipeline as `scripts/fetch_fresh.py`), then scores only signal dates > 2026-10-22 once enough days accumulate (minimum N from POWER.md). Refuses earlier dates.

- [ ] **Step 1:** Failing test (scorer raises for a signal date ≤ 2026-10-22).  - [ ] **Step 2:** Implement the scorer (no data fetch tonight).  - [ ] **Step 3:** Write the protocol and the manifest (code hash, spec, selection outcome, primary contrast, MDE, decision rule).  - [ ] **Step 4: Commit.**

### Task 10: Report and handoff

**Files:**
- Create: `docs/featbench_20261022/FB_REPORT.md`
- Modify: `docs/HANDOFF.md` (uncommitted bullet), `docs/screening_20261019/HANDOFF_2026-10-19.md` (append update)

- [ ] **Step 1:** Write the report: answer per proposal item (1–7 + lag decomposition), verdict tables, what is exploratory, what is frozen, caveats (survivorship, no fundamentals, dollar volume as size proxy, all years inspected).
- [ ] **Step 2:** Check two headline numbers by hand.
- [ ] **Step 3:** Run `pytest -q`; commit (not HANDOFF.md).
