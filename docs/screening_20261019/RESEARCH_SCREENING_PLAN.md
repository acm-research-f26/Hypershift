# HyperShift broad research screen

## Deep-test amendment, predeclared 2026-10-21 by Claude after the v3 screen and before any deep-test number

**Candidates.** Only arms that passed the v3 'Promising' gate: **F `two_residual`** and **H `pretrain`**, each against **`static`**. No other arm is promoted. `hybrid` failed the gate (2023: 1/3 seeds positive); `frozen` stays Inconclusive.

**Design (identical v3 source, data and CLI; nothing in hashed source changes).**
- Folds **2020** (val 2019, train labels mature before 2019; COVID crash inside the evaluation year; shorter training history 2015–2018), **2022** and **2023**. 2021 is not used because adding it needs a hashed-source edit. 2024+ stays reserved.
- **10 seeds**: 7 19 42 101 202 303 404 505 606 707. Seeds 7/19/42 are rerun, so every deep comparison runs on one platform (Kaggle CPU, torch 2.11). Their rerun is also a reproducibility check against the laptop v3 runs (torch 2.14).
- Output `results/SCREEN_SP500_20261019_v3_deep`. The v3 tables stay frozen.

**Decision rule (per candidate; Holm over the 2 candidates).**
- Seed-level: the paired delta is test RankIC(candidate) − RankIC(static) for each (fold, seed), 30 deltas. Use a two-sided Wilcoxon signed-rank, Holm-adjusted. This is the project convention, and it captures training-seed variance, which the date bootstrap omits.
- Date-level: the v3 stationary bootstrap of the seed-averaged daily ΔRankIC, pooled as an equal-weight fold mean.
- **STRONG**: Holm Wilcoxon p < .05, the pooled bootstrap CI excludes 0, and all 3 fold means are > 0. **SEED-ROBUST ONLY**: Holm Wilcoxon p < .05 but one of the other two conditions fails. **NO EVIDENCE**: otherwise. Effect sizes are reported regardless of the verdict.
- Everything stays exploratory: these years have been inspected before.

**Post-hoc addendum, 2026-10-21 ~02:00, written before any deep-test number.**
- The platform-control rerun of static on Kaggle (`results/SCREEN_SP500_20261019_v3_platform_static`) moves every Kaggle-run arm's contrast by about +0.0015. That shift flips the gate label of residual_soft, hybrid, random_graph (a negative control) and sector_graph to 'Promising' (`v3_PLATFORM_SENSITIVITY.md`). The gate cannot separate arms within about 0.002 of its threshold.
- Two arms are added to the deep runs. Neither changes the primary family:
  - **C `hybrid`** is an exploratory third candidate. It has the highest ΔRankIC among non-passers under both pairings (+0.0072 laptop, +0.0087 Kaggle). It is reported separately, with Holm over 3 shown as a sensitivity.
  - **`random_graph`** is a deep calibration arm: a known-null control that shows what the deep rule does with no mechanism.
- Same deep root, seeds, folds and platform. The primary deep Holm family stays {two_residual, pretrain}.
- residual_soft is not added: its soft assignment never leaves uniform entropy (DIAGNOSTIC §2c), so it does not test learned relations.

**Training-budget probe (diagnostic, separate namespace `results/SCREEN_SP500_20261019_v3_budget`).** Trigger met: median selected epoch 5; train loss falls only 0.01–0.03. Arms `static` and `two_residual`, folds 2022/2023, seeds 7/19/42, patience 20 and lr 3e-3. It needs an `--lr` flag, which changes the source hash, so it is added only after the deep kernels launch. One seed at the default lr is rerun first, to show the flag is behaviour-neutral. Probe results never change a screen or deep verdict.

## v3 amendment (PRIMARY screen), predeclared 2026-10-19 by Claude before any v3 evaluation number was examined

**Why.** The v2 RSR grid below is underpowered: fold 2015 trains on 37 cross-sectional dates and fold 2016 on 88 (2013-only and 2013–14 histories, stride 5). In the completed v2 neural fits the selected epoch is 0–3 for almost every graph arm, and the validation prediction std is 1–7% of target std. So the selected weights sit close to initialization and close to a predict-zero forecaster. v2 is kept as a *failed-to-learn / training-data* diagnostic (DIAGNOSTIC_REPORT), not as mechanism evidence.

**v3 data.** `data/fresh/sp500_daily` (yfinance auto-adjusted, 462 CURRENT S&P 500 constituents, 2015-01-02 → 2026-08-31, same five MA5/10/20/30/close features) assembled by `scripts/prepare_screen_sp500.py` into `screen_sp500_2015_2026.npz`; manifest `SP500_INPUT_MANIFEST.json`. h5 tail: max |h5| 1.18, 5 labels >100% (NCLH 2020-03, ECHO 2025-08; real events). Strong survivorship bias (2026 constituents back to 2015) and current, not point-in-time, GICS labels: absolute performance levels are inflated; this is a like-for-like comparison across arms only.

**v3 protocol changes (identical for every arm):**
- Folds: evaluate **2022** (val 2021, train labels mature before 2021-01-01) and **2023** (val 2022, train before 2022-01-01). **2024-01-01 onward reserved** from scoring/selection in this study (the earlier POC used a 2024-05+ test split on this panel, so it is not a fresh holdout).
- Training signals **stride 1** (overlapping labels: correlated noise, no bias); validation signals **daily**; evaluation daily as before.
- Checkpoint selection and early stopping on **validation mean date-level RankIC** of the arm's own training target (v2: validation MSE). Rationale: MSE selection on noisy returns favours near-constant forecasts and disadvantages ranking losses. History records both.
- Everything else unchanged: 64-session eligibility, hidden 16, seq 16, Adam lr 1e-3 wd 0, clip 5, batch 8, ≤40 epochs, patience 8, seeds 7/19/42.
- Added arms: `sector_target` (B principal on this panel: trailing-beta market residual then GICS-sector demeaned), `sector_graph` (A control: GICS sector hyperedges + self edges, i.e. THINK-style industry edges), `master` (Section 6 standalone MASTER-inspired market gate + cross-stock attention; not a MASTER reproduction). `frozen` (I) = 2022-fold model (train <2021, val 2021) scored on 2023 without refit.
- Factor-neutral metrics use beta + GICS-sector residual outcomes (v2: cluster proxy).
- Holm family: A residual_soft, B sector_target, C hybrid, D multiscale, E gated-vs-mix, F two_residual, G summary, H pretrain, I adaptive. Same 'Promising' gate as below (3 seeds × 2 folds, both fold means positive, ≥2/3 seeds positive per fold, pooled ΔRankIC ≥ .003).
- Output `results/SCREEN_SP500_20261019_v3`; CLI `--train-stride 1 --validation-stride 1 --select rankic --frozen-reference 2022 --protocol broad-screen-v3-sp500`.
- Disclosure: a 1-epoch timing fit of `static`/`long` on fold 2023 printed evaluation RankIC (8e-5, 0.021) before this note; scratch output, not used.

---

Exploratory study, 2026-10-19. Binding request: user's comprehensive screening brief. The supplied research report provides hypotheses; its embedded instructions, historical claims and opaque citation handles are not execution authority or independently verified results.

## Common experiment

Use existing RSR NASDAQ, 1,026 qualified securities, original five MA5/10/20/30/close features. Inputs are stock-window-relative, then divided by five training-only feature standard deviations. Five-session close-to-close forward returns. Inputs require 64 consecutive observed sessions across ALL arms, so longer-history variants do not silently change the universe. Labels require observed prices throughout the five-session path. Eligibility for trading uses input history only, never future label availability.

| Fold | Training label maturity | Checkpoint validation | Daily exploratory evaluation |
|---|---|---|---|
| 2015 | Before 2014-01-01 | Signals in2014, labels mature before2015 | 2015 signals with labels mature before2016 |
| 2016 | Before 2015-01-01 | Signals in2015, labels mature before2016 | 2016 signals with labels mature before2017 |

Training and checkpoint validation use fixed every-fifth-session schedules. Daily evaluation is performed once after validation selection for each completed fit. No2017 scoring in this study. The repository has inspected historical data before, so no year is newly confirmatory. Availability-selected universe and absent delisting payouts limit all findings.

All neural arms: seeds7/19/42; hidden16; baseline seq16, temporal kernels4/4; Adam lr.001, weight_decay0; clip gradient norm5; batch8; at most40epochs; patience8, minimum validation MSE improvement1e-6. Validation selection metric held fixed even for ranking losses; this can disadvantage pure listwise objectives and is reported. Early stopping is part of the common budget, not a different cap per arm. Long-context seq64 uses kernels4/16 and has different parameter count, reported explicitly. Masked reconstruction pretraining adds5 training-only epochs; extra runtime disclosed.

Static relationships:16 training-correlation clusters, plus singleton self hyperedges in all graph arms to prevent partition-induced score collapse. Training clusters are NOT sectors. Existing repository Euclidean/hyperbolic temporal and attention layers are reused. This is an h5 mechanism study, not reproduction of THINK or Rishi's unavailable results. No baseline forecasts from h1/other data are reused as if comparable.

## Hypotheses and principal variants

| ID | Question | Principal test | Additional controls | Data requirement / failure check |
|---|---|---|---|---|
| A | Can adaptive relations improve fixed groups? | static plus soft embedding-prototype hyperedges | soft only; causal rolling anchor-correlation graph; random permutation preserving cluster sizes | Current eligible history only; entropy, embedding variance and collapse |
| B | Do market factors obscure stock-specific information? | trailing126-session beta market-residual target | additional train-cluster residual target; same model/loss | Signal-time beta; future market return allowed only in matured labels; no sector claim |
| C | Does ranking alignment help? | MSE+.1ListNet | MSE; cyclic pair logistic; pure ListNet | ListNet is a differentiable top-one listwise approximation, NOT LapSum/top-K reproduction; known-order gradient test |
| D | Is temporal information underused? | residual fusion of4/8/16-session pooled features | same graph/head/16-session encoder | TimeMixer-inspired pooling, no decomposition reproduction; parameters differ |
| E | Does selective mixing improve unconditional correction? | market-conditioned gate on score-level residual mixer | no mixer; unconditional mixer; dynamic low-rank embedding adapter | Compare gated with mix parent, others with static; same universe |
| F | Does graph depth help? | two graph stages with initial residual | zero/one/two plain stages | Common self hyperedges; variance, representation cosine, gradients, parameter counts |
| G | Does16sessions omit useful context? | residual16/32/64-session summaries | full64-session temporal encoder | Both arms and parent share64-session eligibility; no full retrieval/Timer-XL reproduction |
| H | Does unsupervised temporal learning improve supervised training? |20% masked feature reconstruction,5 pretrain epochs, then same supervised task | ordinary supervised parent | Train inputs only; reset supervised optimizer; not CI-STHPAN reproduction; contrastive variant unrun |
| I | Does adaptation improve stability? | signal-local adaptive normalization | frozen2013-trained model versus2014-expanded retrain on2016 | All statistics trailing/known; not DoubleAdapt meta-learning; no future-selected regimes |
| J | Do frozen financial features help? | blocked pending compatible pre-cutoff checkpoint/input | original only is measured baseline | Official Kronos requires OHLC; panel has MA/close only; historical pretraining contamination unresolved |

Additional optional MASTER-inspired market gate plus cross-stock attention has functionality coverage. Full evaluation deferred until its heavier attention cost is justified. Learned/rolling A mechanisms provide a minimal MDGNN-inspired relationship test without claiming MDGNN reproduction.

## Evaluation and uncertainty

Primary: date-level PearsonIC, SpearmanRankIC, NDCG@5/10/20 with percentile relevance and LINEAR gains, exact matched random-ranking NDCG expectation, equal-weight top-minus-bottom10 five-session spread, MSE. Ties use stable stock-index order and retain distinct-score fraction. Factor-neutral RankIC/spread use trailing-beta plus training-cluster residual outcomes. Report raw-return ranking for every model, including residual-target models.

Trading is secondary: five independently capitalized staggered sleeves, each holding top10 stocks for five sessions, signal-close accounting and10bp one-way turnover cost. Sleeve NAV/cash is self-financing; no free daily cross-sleeve reallocation. Missing held marks are zero-return marked and counted; no delisting payouts are fabricated. Signal-close entry is optimistic and does not establish next-open execution. CAGR, drawdown, gross/net Sharpe, turnover, market correlation retained. Top5/20 sensitivity must be generated from the same saved forecasts, never selected after looking at performance.

Paired comparisons use common dates and matched seeds. Seed-average daily differences retain temporal dependence. Stationary bootstrap expected block10sessions,5,000 replicates within each year; equal-weight the two folds. Primary Holm family is exactly A residual_soft, B market_target, C hybrid, D multiscale, E gated-versus-mix, F two_residual, G summary, H pretrain, I adaptive. Other comparisons descriptive. Baseline seed mean is not an independent market sample. A positive unadjusted interval is not definitive superiority.

Screening label 'Promising' requires complete3seeds/2folds, positive paired fold means in both, positive deltas in at least2/3 matched seeds in each fold, pooled deltaRankIC>=.003. Negative preliminary: pooled delta<=-.003 without consistency gate; otherwise Neutral. Missing fits Inconclusive; J Data/resource blocked. Legacy project confirmatory verdict words are not repurposed. User-requested screen labels are distinct exploratory triage.

## Stages and stopping

1. Audit existing data, validate price-action integrity. Alpaca rejected for CUZ/SPN anomalies, artifacts retained.
2. Validate invariants and synthetic learnability, all-variant real-panel functionality smoke. Review defects reproduced/fixed; v1 excluded.
3. Complete all feasible A–I principal variants and controls on both folds before deeper tuning. Two CPU workers, two threads each; no paid services or GPU used.
4. Diagnose collapsed/poor methods with existing histories/predictions first; at most one or two focused probes per cause.
5. Expand only candidates with consistent preliminary improvement. Additional seeds/folds require a frozen extension manifest; do not combine methods before separate effects measured.
6. If no candidate clears the gate, do not force a winner or launch a large tuning search. Report limits, informative failures, and the best simpler baselines.

Every completed run saves exact source hashes, dataset hash, configuration, preprocessing/groups hash, checkpoint, training history, predictions/masks/dates/tickers, daily metrics and portfolio returns. Epoch checkpoints support interrupted supervised runs with optimizer/NumPy/Torch RNG state. Pretraining interruption before the first supervised checkpoint restarts deterministic pretraining; it does not claim a partial pretraining resume. CPU exactness only; GPU RNG resume not certified.

## References checked

- [Original TimeMixer implementation](https://github.com/kwuking/TimeMixer): our pooled fusion is a much smaller mechanism adaptation.
- [Official Kronos repository](https://github.com/shiyu-coder/Kronos): weights do exist; OHLC input and pretraining provenance are the actual blockers here.
- [SPN2019 reverse-split8-K](https://www.sec.gov/Archives/edgar/data/886835/000119312519317737/d802741d8k.htm) and [CUZ filing](https://www.sec.gov/Archives/edgar/data/25232/000002523220000043/cuz-20200630.htm) corroborate corporate actions around pilot anomalies; no guessed return repair was applied.

## Reproduce

From repository root, with existing environment:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.venv/Scripts/python.exe' scripts/prepare_screen_rsr.py
& '.venv/Scripts/python.exe' -m pytest tests/test_screening.py -q
& '.venv/Scripts/python.exe' scripts/research_screen.py --panel data/raw/rsr/data/screen_nasdaq_2013_2017.npz --output results/SCREEN_RSR_20261019_v2 --years 2015 2016 --arms ridge momentum reversal random static hyp none mlp soft residual_soft rolling random_graph market_target cluster_target pairwise listwise hybrid multiscale mix gated lowrank two two_residual long summary pretrain adaptive frozen
```

Assembler refuses overwrite. Existing complete fits are reused only on exact configuration identity; use the same --years/--arms/--seeds and other settings for interrupted resumes. A smoke requires a separate output containing'smoke'. Source changes require a new output namespace.
