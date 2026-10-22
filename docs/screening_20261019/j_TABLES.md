# Study J — Kronos zero-shot (j)

Root `results/SCREEN_SP500_20261021_j`. Exploratory; reserved period spent for J only (see the Study J amendment). Bootstrap: stationary, block 10, B=5000 per fold; pooled = equal-weight fold mean. Wilcoxon: two-sided on (fold, seed) deltas.

## Primary: folds 2025, 2026 vs static (Holm over 2)

| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| kronos | -0.0165 [-0.0560, +0.0236] | 2025: -0.0182 / 2026: -0.0147 | 2025: 0/5 / 2026: 0/5 | 10 | 0.002 | 0.004 | SEED-ROBUST ONLY (−) |
| kronos_static | -0.0059 [-0.0288, +0.0177] | 2025: -0.0087 / 2026: -0.0031 | 2025: 0/5 / 2026: 1/5 | 10 | 0.004 | 0.004 | SEED-ROBUST ONLY (−) |

(−) = the predeclared rule (Holm p < .05, not STRONG) fired in the harmful direction: evidence of no improvement, not of a gain. All seeds score the same dates, so the Wilcoxon reflects seed noise; the date-bootstrap CI is the date uncertainty.

## Absolute RankIC on common dates (seed mean; pooled CI over primary folds)

| Arm | 2024 | 2025 | 2026 | pooled 2025-26 [95% CI] | p (mean = 0) |
|---|---|---|---|---|---|
| static | -0.0097 | +0.0216 | +0.0172 | +0.0194 [-0.0087, +0.0482] | 0.180 |
| none | -0.0076 | +0.0140 | +0.0189 | +0.0165 [-0.0072, +0.0416] | 0.187 |
| kronos | +0.0032 | +0.0034 | +0.0024 | +0.0029 [-0.0305, +0.0364] | 0.871 |
| kronos_static | -0.0044 | +0.0129 | +0.0141 | +0.0135 [-0.0171, +0.0442] | 0.384 |
| reversal | +0.0485 | +0.0091 | +0.0018 | +0.0054 [-0.0235, +0.0344] | 0.714 |
| momentum | -0.0485 | -0.0091 | -0.0018 | -0.0054 [-0.0344, +0.0235] | 0.714 |
| ridge | -0.0193 | +0.0038 | +0.0274 | +0.0156 [-0.0192, +0.0508] | 0.379 |
| random | +0.0000 | -0.0010 | -0.0013 | -0.0011 [-0.0031, +0.0008] | 0.272 |
| kronos 5-seed ensemble | — | +0.0037 | +0.0030 | +0.0034 [-0.0306, +0.0375] | 0.856 |
| kronos single path (path 1) | — | +0.0022 | +0.0013 | +0.0017 [-0.0296, +0.0333] | 0.914 |

## Descriptive contrasts, primary folds (no verdicts; raw p)

| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p |
|---|---|---|---|---|---|
| kronos − none | -0.0135 [-0.0433, +0.0165] | 2025: -0.0106 / 2026: -0.0164 | 2025: 0/5 / 2026: 0/5 | 10 | 0.002 |
| kronos_static − kronos | +0.0106 [-0.0082, +0.0300] | 2025: +0.0095 / 2026: +0.0117 | 2025: 5/5 / 2026: 5/5 | 10 | 0.002 |
| none − static | -0.0030 [-0.0181, +0.0115] | 2025: -0.0076 / 2026: +0.0017 | 2025: 1/5 / 2026: 4/5 | 10 | 0.557 |
| kronos − reversal | -0.0025 [-0.0304, +0.0279] | 2025: -0.0057 / 2026: +0.0006 | — | — | boot 0.861 |
| kronos − momentum | +0.0084 [-0.0468, +0.0646] | 2025: +0.0125 / 2026: +0.0043 | — | — | boot 0.763 |
| kronos − ridge | -0.0127 [-0.0310, +0.0054] | 2025: -0.0004 / 2026: -0.0250 | — | — | boot 0.175 |

## Sensitivities (descriptive)

| Candidate | Pooled dRankIC [95% CI] | Fold means | Seeds positive | n deltas | Wilcoxon p |
|---|---|---|---|---|---|
| (s1) kronos, + fold 2024H2 | -0.0067 [-0.0361, +0.0233] | 2024: +0.0129 / 2025: -0.0182 / 2026: -0.0147 | 2024: 5/5 / 2025: 0/5 / 2026: 0/5 | 15 | 0.169 |
| (s1) kronos_static, + fold 2024H2 | -0.0022 [-0.0199, +0.0163] | 2024: +0.0053 / 2025: -0.0087 / 2026: -0.0031 | 2024: 5/5 / 2025: 0/5 / 2026: 1/5 | 15 | 0.252 |
| (s2) kronos, signal dates ≥ 2025-07-01 | -0.0234 [-0.0647, +0.0166] | 2025: -0.0321 / 2026: -0.0147 | 2025: 0/5 / 2026: 0/5 | 10 | 0.002 |
| (s2) kronos_static, signal dates ≥ 2025-07-01 | -0.0123 [-0.0365, +0.0120] | 2025: -0.0216 / 2026: -0.0031 | 2025: 0/5 / 2026: 1/5 | 10 | 0.004 |
| (s3/s4) kronos_base (seed 7; Wilcoxon on 2 deltas, uninformative) | -0.0226 [-0.0687, +0.0233] | 2025: -0.0191 / 2026: -0.0261 | 2025: 0/1 / 2026: 0/1 | 2 | 0.500 |
| (s3/s4) kronos_mini (seed 7; Wilcoxon on 2 deltas, uninformative) | -0.0156 [-0.0512, +0.0180] | 2025: -0.0126 / 2026: -0.0187 | 2025: 0/1 / 2026: 0/1 | 2 | 0.500 |
| (s3/s4) kronos_lb64 (seed 7; Wilcoxon on 2 deltas, uninformative) | -0.0127 [-0.0501, +0.0238] | 2025: -0.0092 / 2026: -0.0163 | 2025: 0/1 / 2026: 0/1 | 2 | 0.500 |

## Trading block, primary folds (seed mean of metrics.json; top-10, 10 bp; dates asserted identical across arms)

| Arm | 2025 net Sharpe | 2026 net Sharpe | neutral RankIC 2025 / 2026 |
|---|---|---|---|
| static | +0.50 | +1.69 | +0.0029 / +0.0033 |
| none | +0.42 | +1.09 | +0.0044 / +0.0131 |
| kronos | +0.39 | +0.63 | -0.0026 / +0.0017 |
| kronos_static | +0.45 | +1.33 | -0.0033 / +0.0033 |
| reversal | +0.30 | +1.24 | +0.0009 / +0.0265 |
| momentum | +0.57 | +1.83 | -0.0009 / -0.0265 |
| ridge | +0.99 | +1.70 | -0.0034 / +0.0120 |
| random | +0.43 | +0.96 | -0.0003 / +0.0009 |

## Memorisation diagnostic (inside pretraining window; contaminated; descriptive)

| Fold | kronos RankIC (seed 7) [95% CI] |
|---|---|
| 2022 | +0.0098 [-0.0241, +0.0429] |
| 2023 | +0.0098 [-0.0162, +0.0354] |
