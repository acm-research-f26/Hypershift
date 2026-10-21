# Screening tables (v3)

Root `results/SCREEN_SP500_20261019_v3`; folds [2022, 2023]. Exploratory screen triage; verdict words are not the project confirmatory verdicts.
Bootstrap: stationary, expected block 10, B=5000, rng seed 0 (per fold). Pooled = equal-weight mean of fold means; p = fraction of bootstrap means with |m*-m| >= |m|. Holm over the primary family only (A residual_soft, B sector_target, C hybrid, D multiscale, E gated vs mix, F two_residual, G summary, H pretrain, I adaptive).

## Direction matrix (principal arms)

| Experiment | Method | Baseline RankIC | New RankIC | dRankIC [95% CI] | dNDCG@10 | dTop-K spread (spread10) | Seeds (per fold) | Runtime (mean s/fit) | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A | residual_soft (vs static) | 0.0071 | 0.0089 | +0.0018 [-0.0142, +0.0187] | +0.0068 | +0.0011 | 2022:3 / 2023:3 | 341 | 1.000 | Neutral |
| B | sector_target (vs static) | 0.0071 | -0.0169 | -0.0240 [-0.0569, +0.0086] | -0.0037 | -0.0021 | 2022:3 / 2023:3 | 435 | 1.000 | Negative preliminary result |
| C | hybrid (vs static) | 0.0071 | 0.0143 | +0.0072 [-0.0032, +0.0181] | +0.0081 | +0.0017 | 2022:3 / 2023:3 | 461 | 1.000 | Neutral |
| D | multiscale (vs static) | 0.0071 | 0.0088 | +0.0017 [-0.0113, +0.0154] | -0.0008 | +0.0000 | 2022:3 / 2023:3 | 361 | 1.000 | Neutral |
| E | gated (vs mix) | -0.0009 | -0.0029 | -0.0020 [-0.0239, +0.0202] | -0.0077 | +0.0003 | 2022:3 / 2023:3 | 336 | 1.000 | Neutral |
| F | two_residual (vs static) | 0.0071 | 0.0162 | +0.0090 [-0.0043, +0.0225] | +0.0017 | +0.0002 | 2022:3 / 2023:3 | 706 | 1.000 | Promising |
| G | summary (vs static) | 0.0071 | 0.0084 | +0.0013 [-0.0169, +0.0214] | -0.0158 | -0.0002 | 2022:3 / 2023:3 | 361 | 1.000 | Neutral |
| H | pretrain (vs static) | 0.0071 | 0.0107 | +0.0036 [-0.0074, +0.0150] | -0.0007 | +0.0005 | 2022:3 / 2023:3 | 652 | 1.000 | Promising |
| I | adaptive (vs static) | 0.0071 | 0.0054 | -0.0017 [-0.0257, +0.0234] | -0.0077 | -0.0001 | 2022:3 / 2023:3 | 315 | 1.000 | Neutral |
| J | Kronos-type frozen financial features | n/a | n/a | n/a | n/a | n/a | 0 | n/a | - | Data/resource blocked |

## Controls and other arms (descriptive, no Holm)

| Experiment | Method | Baseline RankIC | New RankIC | dRankIC [95% CI] | dNDCG@10 | dTop-K spread (spread10) | Seeds (per fold) | Runtime (mean s/fit) | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A | random_graph (vs static) | 0.0071 | 0.0101 | +0.0030 [-0.0055, +0.0113] | -0.0041 | +0.0005 | 2022:3 / 2023:3 | 374 | - | Neutral |
| A | rolling (vs static) | 0.0071 | 0.0066 | -0.0005 [-0.0106, +0.0091] | +0.0004 | -0.0003 | 2022:3 / 2023:3 | 367 | - | Neutral |
| A | sector_graph (vs static) | 0.0071 | 0.0109 | +0.0038 [-0.0058, +0.0129] | +0.0003 | +0.0012 | 2022:3 / 2023:3 | 283 | - | Neutral |
| A | soft (vs static) | 0.0071 | 0.0120 | +0.0049 [-0.0113, +0.0206] | +0.0016 | -0.0005 | 2022:3 / 2023:3 | 108 | - | Neutral |
| B | cluster_target (vs static) | 0.0071 | -0.0101 | -0.0172 [-0.0457, +0.0112] | +0.0007 | -0.0022 | 2022:3 / 2023:3 | 292 | - | Negative preliminary result |
| B | market_target (vs static) | 0.0071 | -0.0193 | -0.0264 [-0.0591, +0.0061] | -0.0052 | -0.0027 | 2022:3 / 2023:3 | 288 | - | Negative preliminary result |
| C | listwise (vs static) | 0.0071 | 0.0130 | +0.0059 [-0.0161, +0.0281] | +0.0021 | +0.0047 | 2022:3 / 2023:3 | 271 | - | Neutral |
| C | pairwise (vs static) | 0.0071 | 0.0026 | -0.0046 [-0.0120, +0.0020] | +0.0001 | -0.0013 | 2022:3 / 2023:3 | 387 | - | Negative preliminary result |
| E | lowrank (vs static) | 0.0071 | 0.0015 | -0.0056 [-0.0170, +0.0055] | +0.0020 | +0.0000 | 2022:3 / 2023:3 | 316 | - | Negative preliminary result |
| E | mix (vs static) | 0.0071 | -0.0009 | -0.0080 [-0.0365, +0.0210] | -0.0150 | -0.0041 | 2022:3 / 2023:3 | 443 | - | Negative preliminary result |
| F | none (vs static) | 0.0071 | 0.0113 | +0.0042 [-0.0111, +0.0195] | +0.0005 | -0.0003 | 2022:3 / 2023:3 | 87 | - | Neutral |
| F | two (vs static) | 0.0071 | 0.0099 | +0.0028 [-0.0071, +0.0129] | +0.0001 | +0.0014 | 2022:3 / 2023:3 | 453 | - | Neutral |
| G | long (vs static) | 0.0071 | 0.0005 | -0.0066 [-0.0259, +0.0133] | -0.0056 | -0.0009 | 2022:3 / 2023:3 | 773 | - | Negative preliminary result |
| I | frozen (vs static) | 0.0071 | 0.0431 | +0.0218 [-0.0113, +0.0579] | +0.0037 | +0.0054 | 2023:3 | 304 | - | Inconclusive |
| baseline | hyp (vs static) | 0.0071 | 0.0119 | +0.0047 [-0.0083, +0.0190] | -0.0057 | -0.0003 | 2022:3 / 2023:3 | 833 | - | Neutral |
| baseline | mlp (vs static) | 0.0071 | 0.0111 | +0.0040 [-0.0153, +0.0228] | -0.0034 | -0.0006 | 2022:3 / 2023:3 | 70 | - | Neutral |
| baseline | momentum (vs static) | 0.0071 | -0.0105 | -0.0176 [-0.0552, +0.0202] | -0.0004 | -0.0003 | 2022:1 / 2023:1 | 1 | - | Inconclusive |
| baseline | random (vs static) | 0.0071 | -0.0002 | -0.0074 [-0.0287, +0.0130] | -0.0152 | -0.0012 | 2022:3 / 2023:3 | 1 | - | Negative preliminary result |
| baseline | reversal (vs static) | 0.0071 | 0.0105 | +0.0034 [-0.0267, +0.0332] | -0.0027 | -0.0023 | 2022:1 / 2023:1 | 1 | - | Inconclusive |
| baseline | ridge (vs static) | 0.0071 | 0.0064 | -0.0007 [-0.0255, +0.0254] | -0.0142 | +0.0004 | 2022:1 / 2023:1 | 6 | - | Inconclusive |
| extra | master (vs static) | 0.0071 | 0.0099 | +0.0028 [-0.0086, +0.0142] | +0.0013 | +0.0009 | 2022:3 / 2023:3 | 1024 | - | Neutral |

## Baseline table: absolute pooled metrics (mean over folds of seed means), sorted by RankIC

| Arm | Exp | Runs | RankIC | IC | NDCG5 excess | NDCG10 excess | NDCG20 excess | spread10 | neutral RankIC | net Sharpe | turnover |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | I | 3 | 0.0431 | 0.0440 | 0.0348 | 0.0269 | 0.0279 | 0.0077 | 0.0007 | 1.46 | 0.251 |
| two_residual | F | 6 | 0.0162 | 0.0178 | 0.0199 | 0.0164 | 0.0160 | 0.0015 | 0.0022 | 0.65 | 0.261 |
| hybrid | C | 6 | 0.0143 | 0.0196 | 0.0303 | 0.0227 | 0.0178 | 0.0030 | 0.0056 | 0.67 | 0.288 |
| listwise | C | 6 | 0.0130 | 0.0202 | 0.0190 | 0.0168 | 0.0132 | 0.0061 | -0.0047 | 0.73 | 0.234 |
| soft | A | 6 | 0.0120 | 0.0144 | 0.0172 | 0.0163 | 0.0150 | 0.0008 | 0.0003 | 0.61 | 0.288 |
| hyp | baseline | 6 | 0.0119 | 0.0140 | 0.0095 | 0.0089 | 0.0099 | 0.0010 | 0.0087 | 0.34 | 0.290 |
| none | F | 6 | 0.0113 | 0.0157 | 0.0178 | 0.0151 | 0.0145 | 0.0010 | 0.0010 | 0.58 | 0.278 |
| mlp | baseline | 6 | 0.0111 | 0.0131 | 0.0086 | 0.0113 | 0.0136 | 0.0008 | -0.0000 | 0.56 | 0.292 |
| sector_graph | A | 6 | 0.0109 | 0.0149 | 0.0158 | 0.0149 | 0.0133 | 0.0026 | 0.0006 | 0.52 | 0.264 |
| pretrain | H | 6 | 0.0107 | 0.0143 | 0.0143 | 0.0140 | 0.0147 | 0.0018 | -0.0005 | 0.54 | 0.272 |
| reversal | baseline | 2 | 0.0105 | 0.0066 | 0.0147 | 0.0120 | 0.0081 | -0.0010 | 0.0058 | 0.55 | 0.239 |
| random_graph | A | 6 | 0.0101 | 0.0130 | 0.0133 | 0.0105 | 0.0108 | 0.0018 | 0.0049 | 0.47 | 0.270 |
| master | extra | 6 | 0.0099 | 0.0144 | 0.0213 | 0.0160 | 0.0147 | 0.0022 | 0.0035 | 0.45 | 0.297 |
| two | F | 6 | 0.0099 | 0.0152 | 0.0203 | 0.0148 | 0.0122 | 0.0027 | 0.0047 | 0.52 | 0.280 |
| residual_soft | A | 6 | 0.0089 | 0.0150 | 0.0294 | 0.0215 | 0.0145 | 0.0024 | 0.0059 | 0.67 | 0.310 |
| multiscale | D | 6 | 0.0088 | 0.0169 | 0.0170 | 0.0138 | 0.0117 | 0.0013 | 0.0046 | 0.45 | 0.265 |
| summary | G | 6 | 0.0084 | 0.0075 | -0.0056 | -0.0012 | 0.0043 | 0.0011 | 0.0025 | 0.31 | 0.180 |
| static | baseline | 6 | 0.0071 | 0.0126 | 0.0202 | 0.0147 | 0.0100 | 0.0013 | -0.0006 | 0.49 | 0.266 |
| rolling | A | 6 | 0.0066 | 0.0098 | 0.0230 | 0.0150 | 0.0105 | 0.0010 | 0.0023 | 0.49 | 0.281 |
| ridge | baseline | 2 | 0.0064 | 0.0126 | -0.0022 | 0.0004 | 0.0060 | 0.0017 | 0.0045 | 0.43 | 0.241 |
| adaptive | I | 6 | 0.0054 | 0.0083 | 0.0077 | 0.0069 | 0.0054 | 0.0012 | -0.0016 | 0.18 | 0.232 |
| pairwise | C | 6 | 0.0026 | 0.0079 | 0.0195 | 0.0148 | 0.0101 | 0.0000 | 0.0001 | 0.49 | 0.260 |
| lowrank | E | 6 | 0.0015 | 0.0083 | 0.0217 | 0.0167 | 0.0122 | 0.0013 | -0.0005 | 0.50 | 0.323 |
| long | G | 6 | 0.0005 | 0.0056 | 0.0113 | 0.0090 | 0.0098 | 0.0005 | 0.0018 | 0.38 | 0.194 |
| random | baseline | 6 | -0.0002 | 0.0001 | 0.0005 | -0.0006 | 0.0005 | 0.0001 | -0.0005 | -0.08 | 0.385 |
| mix | E | 6 | -0.0009 | -0.0048 | 0.0027 | -0.0003 | 0.0002 | -0.0028 | 0.0046 | 0.38 | 0.230 |
| gated | E | 6 | -0.0029 | -0.0033 | -0.0093 | -0.0081 | -0.0057 | -0.0025 | -0.0033 | 0.33 | 0.161 |
| cluster_target | B | 6 | -0.0101 | -0.0047 | 0.0215 | 0.0154 | 0.0104 | -0.0009 | -0.0033 | 0.46 | 0.322 |
| momentum | baseline | 2 | -0.0105 | -0.0066 | 0.0156 | 0.0143 | 0.0074 | 0.0010 | -0.0058 | 0.79 | 0.238 |
| sector_target | B | 6 | -0.0169 | -0.0114 | 0.0167 | 0.0110 | 0.0085 | -0.0008 | 0.0017 | 0.37 | 0.332 |
| market_target | B | 6 | -0.0193 | -0.0132 | 0.0154 | 0.0095 | 0.0054 | -0.0014 | -0.0038 | 0.27 | 0.344 |

Net Sharpe, turnover: top-10, 10 bp (predeclared default). Single-seed statistical arms (ridge, momentum, reversal) are compared with the seed-mean series of their parent; random has its own seeds.
