# Study J — zero-shot Kronos on the reserved period (2026-10-21)

**Exploratory.** This is not confirmatory and not a production recommendation. The design was fixed by the Study J amendment in `RESEARCH_SCREENING_PLAN.md` (commit 823a80e) before any J number was computed. The reserved ≥ 2024-07 period is now **spent for J only**: the arms below are the only ones ever scored on it under that amendment.

Tables: `j_TABLES.md`; per-(fold, seed) deltas: `j_PAIRED.csv`. Code: `scripts/j_kronos.py` (scorer), `scripts/j_analysis.py` (tables). Root: `results/SCREEN_SP500_20261021_j`.

## Answer

**There is no evidence that Kronos helps.**

- **Kronos alone.** Zero-shot Kronos-small ranks the 462 S&P 500 stocks no better than chance on 2025–26: pooled RankIC +0.003 [−0.031, +0.036].
- **Kronos vs the v3 baseline.** It is below static in all 10 (fold, seed) pairs, by −0.0165 [−0.056, +0.024].
- **Kronos averaged with the baseline.** This is also below static, by −0.0059 [−0.029, +0.018], in 9 of 10 pairs.

Under the predeclared rule both primary contrasts come out **SEED-ROBUST ONLY in the harmful direction (−)**: Holm p 0.004, with the date-bootstrap CI including 0.

- **Why the label is weak evidence of harm.** All seeds score the same dates. So the Wilcoxon only shows that the shortfall is not seed noise.
- **Why it is not a claim about other periods.** Nothing in 2025–26 is distinguishable from zero. That includes static (+0.019, p 0.18) and every simple baseline.

The right reading is "Kronos ≈ 0; static nominally higher". It is not "Kronos is harmful".

## What was run

| | |
|---|---|
| Model | Official `shiyu-coder/Kronos` @ 67b630e (MIT); `NeoQuasar/Kronos-small` (sha 901c26c1) + `Kronos-Tokenizer-base` (0e011738), from local copies in a private Kaggle dataset. |
| Input | Adjusted OHLCV aligned to the v3 panel (close/panel spread ≤ 1.2e-7); amount = volume × mean(OHLC). Lookback 256 sessions (64 for stocks without 256 observed sessions). |
| Score | Mean over M = 4 sampled paths of predicted close(t+5)/close(t) − 1. T 1.0, top_p 0.9, per-window z-norm and clip 5 as in `KronosPredictor`. **fp16 autocast** (a deviation from official fp32). |
| Seeds | Sampling seeds 7/19/42/101/202, paired with static seeds; RNG reseeded per (seed, day, lookback group), so the day sharding across GPUs is deterministic (verified: 2 shards ≡ 1 shard bit for bit). |
| `kronos_static` | No fitting. The daily mean of the cross-sectional z-scores of kronos seed k and static seed k. |
| Baselines | v3 static and none (5 seeds; Kaggle CPU, one kernel per seed); reversal, momentum, ridge, random (deterministic, seed 7 kernel). |
| Folds | Primary 2025 (245 days) and 2026 (161 days, to 2026-08-24). Sensitivity 2024H2 (123 days, 2024-07-01 → 12-23). Eligibility is identical across arms (mean 462 stocks/day). |
| Compute | Kaggle T4×2: about 80 min per Kronos seed (2025–26 plus 2024H2); 5 kernels. The sensitivity kernel (memorisation 2022/23, base, mini, lookback 64) ran separately. |

## Primary result (folds 2025 and 2026, vs static, Holm over 2)

| Candidate | Pooled ΔRankIC [95% CI] | 2025 | 2026 | Seeds > 0 | Holm p | Rule output |
|---|---|---|---|---|---|---|
| kronos | −0.0165 [−0.0560, +0.0236] | −0.0182 | −0.0147 | 0/10 | 0.004 | SEED-ROBUST ONLY (−) |
| kronos_static | −0.0059 [−0.0288, +0.0177] | −0.0087 | −0.0031 | 1/10 | 0.004 | SEED-ROBUST ONLY (−) |

**Reading the verdict.** The amendment fixed a two-sided Wilcoxon, and the label ladder (STRONG / SEED-ROBUST ONLY / NO EVIDENCE) only checks whether the CI and fold means are positive. So a significant shortfall still gets "SEED-ROBUST ONLY". `j_analysis.py` marks such cases (−), and `deep_analysis.py` is unchanged so earlier verdicts still reproduce. For the question the study asks, whether Kronos improves on static, the answer is **no evidence of improvement**.

## Absolute RankIC (seed mean, common dates)

| Arm | 2024H2 | 2025 | 2026 | Pooled 2025–26 [95% CI] |
|---|---|---|---|---|
| static | −0.0097 | +0.0216 | +0.0172 | +0.0194 [−0.0087, +0.0482] |
| none (no graph) | −0.0076 | +0.0140 | +0.0189 | +0.0165 [−0.0072, +0.0416] |
| ridge | −0.0193 | +0.0038 | +0.0274 | +0.0156 [−0.0192, +0.0508] |
| kronos_static | −0.0044 | +0.0129 | +0.0141 | +0.0135 [−0.0171, +0.0442] |
| reversal (16-day) | +0.0485 | +0.0091 | +0.0018 | +0.0054 [−0.0235, +0.0344] |
| **kronos** | +0.0032 | +0.0034 | +0.0024 | **+0.0029 [−0.0305, +0.0364]** |
| kronos 5-seed ensemble | — | +0.0037 | +0.0030 | +0.0034 [−0.0306, +0.0375] |
| kronos single path | — | +0.0022 | +0.0013 | +0.0017 [−0.0296, +0.0333] |
| random | +0.0000 | −0.0010 | −0.0013 | −0.0011 [−0.0031, +0.0008] |

momentum is the sign flip of reversal. 2024H2 uses the kronos dates for every arm.

- **More sampling does not help.** Averaging 4 paths moves Kronos from +0.002 to +0.003, and 20 paths (the 5-seed ensemble) to +0.003. The forecast is consistent across samples (see the amendment: 4-path means correlate about 0.9 with each other), but it carries almost no cross-sectional information about 5-day returns.
- **Kronos sits with reversal and below the trained models.** kronos − reversal −0.0025 [−0.030, +0.028]; kronos − none −0.0135 (0/10 seeds positive); kronos − ridge −0.0127 [−0.031, +0.005].
- **The combination only dilutes static.** kronos_static − kronos is +0.0106 (10/10 seeds), but kronos_static − static is −0.0059.

## Sensitivities (descriptive)

| | Pooled ΔRankIC vs static [95% CI] | Per fold | Seeds > 0 |
|---|---|---|---|
| (s1) kronos, add 2024H2 | −0.0067 [−0.0361, +0.0233] | 2024H2 +0.0129 / 2025 −0.0182 / 2026 −0.0147 | 5/15 |
| (s1) kronos_static, add 2024H2 | −0.0022 [−0.0199, +0.0163] | +0.0053 / −0.0087 / −0.0031 | 5/15 |
| (s2) kronos, signal dates ≥ 2025-07-01 | −0.0234 [−0.0647, +0.0166] | 2025H2 −0.0321 / 2026 −0.0147 | 0/10 |
| (s2) kronos_static, ≥ 2025-07-01 | −0.0123 [−0.0365, +0.0120] | −0.0216 / −0.0031 | 1/10 |
| (s3) Kronos-base, seed 7 | S3_BASE | | |
| (s3) Kronos-mini, seed 7 | S3_MINI | | |
| (s4) lookback 64, seed 7 | S4_LB64 | | |

- **2024H2 is the only window where Kronos beats static (+0.013, 5/5 seeds).** It is also the window where static is negative (−0.010) and reversal is strong (+0.049). It contains the one peeked day (2024-07-01; see Disclosures). It also lies within 6 months of the paper's stated pretraining cutoff. Treat it as descriptive only.
- **The hard-clean window (s2) is worse for Kronos, not better.** This window is after 2025-06-30, the upload date of the released weights, so no leakage is possible. A memorisation story would predict the opposite.

## Trading block (primary folds; seed mean; top-10 long, 10 bp; dates checked identical across arms)

| Arm | 2025 net Sharpe | 2026 net Sharpe | Neutral RankIC 2025 / 2026 |
|---|---|---|---|
| static | +0.50 | +1.69 | +0.0029 / +0.0033 |
| none | +0.42 | +1.09 | +0.0044 / +0.0131 |
| ridge | +0.99 | +1.70 | −0.0034 / +0.0120 |
| kronos_static | +0.45 | +1.33 | −0.0033 / +0.0033 |
| reversal | +0.30 | +1.24 | +0.0009 / +0.0265 |
| momentum | +0.57 | +1.83 | −0.0009 / −0.0265 |
| **kronos** | +0.39 | +0.63 | −0.0026 / +0.0017 |
| random | +0.43 | +0.96 | −0.0003 / +0.0009 |

kronos trades no better than random baskets in either year, and factor-neutral RankIC is about 0. Positive Sharpe for every arm, random included, reflects the market and survivorship bias (current constituents), not skill.

## Memorisation diagnostic (inside the pretraining window; contaminated)

MEMO_ROWS

## Disclosures and deviations

- **Peek before predeclaration.** Kronos-small was scored on one reserved day, 2024-07-01: day RankIC −0.22, correlation +0.29 with reversal. It played no part in any choice.
- **Runner smoke test.** The 4-day 2025 smoke run of the `--reserved-j` flag printed static −0.147 and reversal +0.064 before the full runs. These came from 4 days and were not used.
- **Analysis run on partial data.** `j_analysis.py` was debugged on 2 of 5 Kronos seeds after the design was locked. No choice depends on it.
- **fp16 autocast** instead of official fp32.
- **Pretraining cutoff is unconfirmed.** The paper states June 2024. The released checkpoints are bounded only by their 2025-06-30 upload. The primary folds start 2025-01; s2 (≥ 2025-07) is clean under either bound.
- **Hardware.** Kronos ran on Kaggle Tesla T4×2 (torch 2.11.0+cu128). The CPU baselines ran on Kaggle: seed 7 on AMD EPYC 7B12, the other seeds on Intel Xeon 2.20 GHz. This does not affect the kronos contrasts, since Kronos is deterministic per seed and runs separately.
- **Data.** 2026 S&P 500 constituents traced back (survivorship bias); yfinance adjusted OHLCV. An earlier POC tested 2024-05 onward (DATA_AUDIT), so 2024+ was never a pristine holdout.
- **Not run: Kronos embeddings as model inputs** (the feature-input J). Training-year features would come from inside the pretraining window, and validation selection cannot detect reliance on memorised information.

## Checks

- **Kronos 2025 seed 7.** RankIC recomputed by hand from `predictions.npz` = 0.0033016, equal to `metrics.json`.
- **Kronos and static 2025 seed 202.** Hand 0.0021033 and 0.0219921, equal to stored.
- **kronos_static 2026 seed 101.** Rebuilt from the raw kronos and static predictions = 0.0085230; stored 0.0085230 (difference 1e-11).
- **Merging.** 84 runs complete, 0 incomplete. Kernel outputs were merged without overwriting.

## Implication

The one direction that needed the reserved period has now used it, and the result is null. Zero-shot Kronos forecasts carry no usable cross-sectional 5-day signal on this universe. Combining them with the v3 model does not help either. This adds nothing new to the screen's conclusion: the bottleneck is signal in the inputs and targets.
