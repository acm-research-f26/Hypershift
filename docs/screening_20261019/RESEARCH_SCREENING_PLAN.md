# HyperShift broad research screen

## Study J amendment — Kronos foundation model on the reserved period. Predeclared 2026-10-21 by Claude, before any J evaluation number

**Authorisation.** The user said "unblock J Kronos, start doing it" (2026-10-21), which selects option (b) of FINAL_RECOMMENDATIONS: evaluate J on the reserved ≥ 2024-07 period. The reserved spend is limited to the arms listed below. **No other arm may be scored on 2024+ later on the strength of this amendment.** As DATA_AUDIT notes, 2024+ is not a fresh holdout (an earlier POC tested 2024-05 onward).

**Disclosure of one peek.** While timing the smoke test, Kronos-small was scored on one reserved day, 2024-07-01: day RankIC −0.22, correlation +0.29 with 5-day reversal. That number played no part in any choice below. All later timing and consistency checks use pre-2024 days only.

### Model and provenance
- Official repo `shiyu-coder/Kronos` at commit `67b630e`, MIT licence. Weights `NeoQuasar/Kronos-small` (sha `901c26c1`) with `Kronos-Tokenizer-base` (`0e011738`). Sensitivity weights: `Kronos-base` (`2b554741`) and `Kronos-mini` (`f4e68697`) with `Kronos-Tokenizer-2k` (`26966d00`).
- **Pretraining cutoff:**
  - The paper (arXiv 2508.02739, App. D) says the pretraining data "extends up to June 2024". The HF model cards state no cutoff.
  - The HF weights were uploaded 2025-06-30/07-01 and never changed, so the released checkpoints cannot have seen anything after 2025-06-30. That is a hard bound.
  - That the released checkpoints share the paper's June 2024 cutoff is **not confirmed**.
- **Inputs:** adjusted OHLCV from `data/fresh/sp500_daily/bars.csv.gz`, aligned to the panel. close/panel-close is constant per stock to within 1.2e-7, and no bar is missing where the panel is observed. `amount = volume × mean(OHLC)` (the official predictor's fill). Time stamps are the real session dates; future stamps are the next 5 calendar sessions, since the trading calendar is known in advance.
- Per-window z-normalisation and the ±5 clip exactly as in `KronosPredictor`. Inference calls the official `auto_regressive_inference` with T = 1.0, top_k = 0, top_p = 0.9 (the predictor defaults), and **fp16 autocast** (a deviation from the official fp32, adopted for speed).

### Arms scored on the reserved period (complete list)
- `kronos`: zero-shot Kronos-small. Lookback 256 sessions (64 for a stock without 256 observed sessions, in a separate batch, so the universe equals static's eligibility). 5-step forecast. Score = mean over **M = 4** sampled paths of predicted close(t+5) / close(t) − 1.
  - **Why M = 4:** on 4 pre-2024 days (2021), path-to-path rank correlation was 0.72–0.82 for single paths and 0.88–0.94 for 4-path vs 4-path means. A single path would attenuate skill.
  - **Seeds:** Kronos seed k = sampling seed k (7, 19, 42, 101, 202), with the RNG reseeded per (seed, signal day, lookback group). Kronos seed k is paired with static seed k.
  - **Budget fallback:** if the pre-2024 Kaggle GPU smoke shows the M = 4 × 5-seed run will not fit in about 12 GPU-hours, the lookback drops to 128. M is never reduced.
- `kronos_static`: no fitting. Per day, the average of the cross-sectional z-scores of the static seed-k and kronos seed-k predictions over eligible stocks.
- `static` and `none`: the v3 model and CLI (stride 1, validation-RankIC selection), seeds as above, on runner folds 2024, 2025 and 2026. Fold Y trains on labels maturing before Y−1, validates on Y−1 and evaluates Y. Fold 2026 runs to the last mature label in 2026-08.
- `reversal`, `momentum`, `ridge` and `random`: the deterministic and statistical baselines, unchanged.
- **Not run: Kronos embeddings as model inputs** (the feature-input version of J). Training-year features come from inside Kronos's pretraining window. A model trained on them could learn to rely on memorised information, and validation-based selection could not detect it.

### Folds and decision rule
- **Primary folds: 2025 and 2026** (signal dates 2025-01-01 → 2026-08). These have at least a 6-month margin after the paper's stated cutoff.
- **Primary family:** {kronos, kronos_static} vs static, Holm over 2. Two-sided Wilcoxon on the 10 (fold, seed) deltas, plus the stationary date bootstrap (block 10, B = 5000), pooled as the equal-weight mean of fold means. STRONG / SEED-ROBUST ONLY / NO EVIDENCE exactly as in the deep test.
- **Descriptive:**
  - kronos RankIC vs 0 (bootstrap CI);
  - kronos vs none, reversal, momentum and ridge;
  - the 5-seed kronos ensemble;
  - single-path kronos (from path 1 of each seed);
  - neutral RankIC and the trading block.
- **Sensitivities (descriptive):**
  - (s1) add fold 2024H2 (signal dates ≥ 2024-07-01; static is restricted to the same dates);
  - (s2) hard-clean only: signal dates ≥ 2025-07-01;
  - (s3) Kronos-base and Kronos-mini, seed 7, primary folds;
  - (s4) lookback 64, seed 7.
- **Memorisation diagnostic (contaminated, descriptive):** kronos seed 7 on 2022 and 2023, inside the pretraining window. A higher in-sample IC is **not** proof of memorisation, because the regimes differ.
- All absolute and trading numbers are computed from daily series on the common dates of the compared arms, not from `metrics.json` scalars.

### Compute and code
- Runner: an explicit `--reserved-j` flag permits years 2024–2026; without it the old guard stands. `kronos` and `kronos_static` are added to ARMS under experiment J, and `statistical()` raises for them. Because the source hash changes, this is a new root: `results/SCREEN_SP500_20261021_j`.
- static/none/baselines: one Kaggle CPU kernel per seed, as in X1.
- Kronos: `scripts/j_kronos.py` on Kaggle GPU. The Kronos code and weights go in a private Kaggle dataset loaded from a local path. Days are sharded across GPUs, and the per-(seed, day) RNG makes this deterministic. Configs record the GPU model, fp16, lookback, M and the source/weight hashes.

## Study X amendment — pairwise / star expansion (neural) and pairs trading. Predeclared 2026-10-21 by Claude, before any X number

The user asked to test "Pairwise Trading / Star Expansion" and then to "test out all possibilities". The phrase has no definition in the repo, so it is read three ways, and each reading is tested on the v3 S&P 500 panel. 2024+ stays reserved. Everything is exploratory: the years have been inspected before.

Already measured, not rerun:
- the C `pairwise` ranking-loss arm (−0.005, Negative preliminary);
- Phase 1 hypergraph vs clique on 309 RSR stocks (not established).

### X1. Neural graph expansion (Kaggle)

- **Arms**, all with the v3 model, data, CLI and selection rule (validation RankIC):
  - `static`: parent; correlation-cluster hyperedges plus self edges.
  - `clique`: each cluster expanded into all within-cluster pairs, as 2-node hyperedges plus self edges, in the **same** attention layer. Only the graph changes. 8.5–9.6k pairs per fold.
    - Known confound, inherent to clique expansion: softmax over incident edges gives the self edge weight ≈ 1/k instead of ≈ 1/2, so self information is diluted.
  - `star`: explicit star expansion. One hub per cluster: hub = ReLU(W_in · mean of member nodes). Each member then gets ReLU(W_self·h + W_out·hub). **This changes the layer as well as the expansion.** The current static attention layer is itself already a node→hyperedge→node scheme, so a star arm built with the same layer would duplicate static.
  - `none`: no graph (graph-free bar).
  - `random_graph`: random partition; known-null calibration.
- **Design:**
  - Folds 2020, 2022, 2023; seeds 7, 19, 42, 101, 202.
  - **One Kaggle kernel per seed, running all 5 arms, so every paired contrast is session-matched.** `lscpu` is logged.
  - The source edit changes hashes, so this is a new root, `results/SCREEN_SP500_20261021_x`, and static is rerun. Before launch, a laptop rerun of v3 `2022/static/seed_7` under the new source must be bit-identical to v3.
- **Decision rule (as in the deep test):**
  - Primary family {clique, star} vs static, Holm over 2, on a two-sided Wilcoxon of 15 (fold, seed) deltas, plus the date bootstrap.
  - **STRONG** = Holm p < .05, pooled CI excludes 0 and all fold means > 0. **SEED-ROBUST ONLY** = Holm p < .05 only. **NO EVIDENCE** = otherwise.
  - Descriptive only: each arm vs `none`, clique vs star, and random_graph vs static as the noise floor.

### X2. Pairs trading on graph structure (laptop, deterministic, `scripts/pairs_trading.py`)

Gatev–Goetzmann–Rouwenhorst distance method, adapted.

- **Formation:** 252 sessions before each trading start. Stocks must be fully observed in formation. Prices are normalised to 1 at formation start (adjusted closes, as in DATA_AUDIT).
- **Trading:** 6 months. A new portfolio is formed at every month start, and the daily return is the mean over the ≤ 6 overlapping active portfolios. Only returns dated inside 2020, 2022 or 2023 are scored.
- **Groupings, recomputed at each formation from formation data only:**
  - `cluster`: KMeans-16 on standardised formation returns.
  - `sector`: GICS; current labels, which is a caveat.
  - `random`: a random partition with the cluster sizes; 20 draws, averaged.
  - `all`: no grouping.
- **Strategy families:**
  - **Pairwise (clique):** candidates are all within-group pairs, and the 20 with the smallest SSD of normalised prices are selected. Spread = P̃_i − P̃_j.
  - **Star:** candidates are each stock vs its group centre. The centre is a buy-and-hold basket of the group's other members (P̃ mean, excluding the stock), and the 20 stock–centre pairs with the smallest SSD are selected. For `all`, the centre is the rest of the market.
- **Rules (primary):**
  - z = (spread − formation mean) / formation std.
  - Open when |z| > 2; close when z crosses 0 or at the end of the trading period.
  - **Wait one day:** execute at the next close after the signal, for both entry and exit.
  - Hold $1 long the cheap leg and $1 short the rich leg; the pair's daily return is r_long − r_short.
  - Portfolio return is on committed capital (sum / 20).
  - Costs are 10 bp per leg per side.
  - If a stock stops trading, its position is closed at the last observed price.
- **Sensitivities (descriptive):** same-close execution, costs of 0 and 25 bp, top-50 pairs.
- **Tests:** stationary bootstrap (block 10, B = 5000) of daily net returns, with years pooled at equal weight.
  - (a) **Absolute:** mean > 0. Family of 6: {pairwise, star} × {cluster, sector, all}; Holm over 6.
  - (b) **Structure:** grouped minus the random-group null (mean of 20 draws). Family of 4: {pairwise, star} × {cluster, sector}; Holm over 4.
  - (c) Star minus pairwise, per grouping: descriptive.
- **Labels:**
  - **STRUCTURE EFFECT** = (a) and (b) Holm p < .05 with CIs > 0, and net Sharpe > 0 in all 3 years.
  - **POSITIVE ONLY** = (a) passes and (b) fails; for `all`, (a) passes.
  - **NO EVIDENCE** = (a) fails.
- **Survivorship bias is decisive here.** The panel holds only 2026 constituents, so pairs that diverge permanently (bankruptcy, delisting, takeover at a gap) are excluded, and those are exactly the pairs where the distance method loses. Absolute returns are therefore an upper bound, and inference leans on the null-relative test (b).


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
