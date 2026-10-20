# Screening tables (v2)

Root `results/SCREEN_RSR_20261019_v2`; folds [2015, 2016]. Exploratory screen triage; verdict words are not the project confirmatory verdicts.
Bootstrap: stationary, expected block 10, B=5000, rng seed 0 (per fold). Pooled = equal-weight mean of fold means; p = fraction of bootstrap means with |m*-m| >= |m|. Holm over the primary family only (A residual_soft, B market_target, C hybrid, D multiscale, E gated vs mix, F two_residual, G summary, H pretrain, I adaptive).

## Direction matrix (principal arms)

| Experiment | Method | Baseline RankIC | New RankIC | dRankIC [95% CI] | dNDCG@10 | dTop-K spread (spread10) | Seeds (per fold) | Runtime (mean s/fit) | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A | residual_soft (vs static) | 0.0020 | 0.0045 | +0.0025 [-0.0084, +0.0130] | +0.0022 | +0.0002 | 2015:3 / 2016:3 | 31 | 1.000 | Neutral |
| B | market_target (vs static) | 0.0020 | -0.0014 | -0.0033 [-0.0146, +0.0087] | +0.0015 | +0.0017 | 2015:3 / 2016:3 | 63 | 1.000 | Negative preliminary result |
| C | hybrid (vs static) | 0.0020 | 0.0018 | -0.0002 [-0.0009, +0.0006] | +0.0007 | +0.0000 | 2015:3 / 2016:3 | 31 | 1.000 | Neutral |
| D | multiscale (vs static) | 0.0020 | 0.0099 | +0.0079 [-0.0047, +0.0202] | +0.0148 | +0.0029 | 2015:3 / 2016:3 | 32 | 1.000 | Promising |
| E | gated (vs mix) | 0.0007 | -0.0003 | -0.0011 [-0.0068, +0.0046] | -0.0008 | -0.0002 | 2015:3 / 2016:3 | 33 | 1.000 | Neutral |
| F | two_residual (vs static) | 0.0020 | 0.0030 | +0.0010 [-0.0048, +0.0072] | -0.0033 | +0.0002 | 2015:3 / 2016:3 | 52 | 1.000 | Neutral |
| G | summary (vs static) | 0.0020 | 0.0094 | +0.0074 [-0.0028, +0.0186] | +0.0168 | +0.0058 | 2015:3 / 2016:3 | 30 | 1.000 | Neutral |
| H | pretrain (vs static) | 0.0020 | 0.0053 | +0.0033 [-0.0059, +0.0129] | +0.0057 | +0.0011 | 2015:3 / 2016:3 | 37 | 1.000 | Promising |
| I | adaptive (vs static) | 0.0020 | 0.0045 | +0.0025 [-0.0102, +0.0148] | +0.0013 | -0.0005 | 2015:3 / 2016:3 | 43 | 1.000 | Neutral |
| J | Kronos-type frozen financial features | n/a | n/a | n/a | n/a | n/a | 0 | n/a | - | Data/resource blocked |

## Controls and other arms (descriptive, no Holm)

| Experiment | Method | Baseline RankIC | New RankIC | dRankIC [95% CI] | dNDCG@10 | dTop-K spread (spread10) | Seeds (per fold) | Runtime (mean s/fit) | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A | random_graph (vs static) | 0.0020 | 0.0007 | -0.0012 [-0.0044, +0.0019] | +0.0035 | +0.0008 | 2015:3 / 2016:3 | 32 | - | Neutral |
| A | rolling (vs static) | 0.0020 | -0.0008 | -0.0028 [-0.0051, -0.0005] | +0.0012 | -0.0000 | 2015:3 / 2016:3 | 24 | - | Neutral |
| A | soft (vs static) | 0.0020 | 0.0094 | +0.0074 [-0.0077, +0.0219] | +0.0013 | +0.0023 | 2015:3 / 2016:3 | 19 | - | Promising |
| B | cluster_target (vs static) | 0.0020 | 0.0026 | +0.0006 [-0.0113, +0.0108] | +0.0003 | +0.0019 | 2015:3 / 2016:3 | 78 | - | Neutral |
| C | listwise (vs static) | 0.0020 | 0.0004 | -0.0016 [-0.0128, +0.0107] | -0.0065 | -0.0023 | 2015:3 / 2016:3 | 33 | - | Neutral |
| C | pairwise (vs static) | 0.0020 | 0.0020 | +0.0001 [-0.0000, +0.0001] | +0.0005 | +0.0001 | 2015:3 / 2016:3 | 31 | - | Neutral |
| E | lowrank (vs static) | 0.0020 | 0.0052 | +0.0032 [-0.0142, +0.0188] | +0.0020 | -0.0006 | 2015:3 / 2016:3 | 38 | - | Neutral |
| E | mix (vs static) | 0.0020 | 0.0007 | -0.0012 [-0.0091, +0.0067] | +0.0077 | -0.0009 | 2015:3 / 2016:3 | 42 | - | Neutral |
| F | none (vs static) | 0.0020 | 0.0046 | +0.0026 [-0.0046, +0.0104] | +0.0030 | +0.0024 | 2015:3 / 2016:3 | 8 | - | Neutral |
| F | two (vs static) | 0.0020 | 0.0023 | +0.0003 [-0.0077, +0.0077] | +0.0031 | -0.0000 | 2015:3 / 2016:3 | 57 | - | Neutral |
| G | long (vs static) | 0.0020 | 0.0043 | +0.0023 [-0.0135, +0.0174] | +0.0110 | +0.0021 | 2015:3 / 2016:3 | 57 | - | Neutral |
| I | frozen (vs static) | 0.0020 | 0.0006 | -0.0017 [-0.0119, +0.0101] | -0.0042 | -0.0005 | 2016:3 | 26 | - | Inconclusive |
| baseline | hyp (vs static) | 0.0020 | 0.0081 | +0.0061 [-0.0045, +0.0173] | +0.0074 | +0.0016 | 2015:3 / 2016:3 | 36 | - | Promising |
| baseline | mlp (vs static) | 0.0020 | 0.0081 | +0.0061 [-0.0091, +0.0210] | +0.0087 | +0.0037 | 2015:3 / 2016:3 | 12 | - | Promising |
| baseline | momentum (vs static) | 0.0020 | -0.0195 | -0.0215 [-0.0449, +0.0023] | -0.0284 | -0.0118 | 2015:1 / 2016:1 | 0 | - | Inconclusive |
| baseline | random (vs static) | 0.0020 | 0.0006 | -0.0014 [-0.0095, +0.0068] | +0.0008 | -0.0027 | 2015:3 / 2016:3 | 0 | - | Neutral |
| baseline | reversal (vs static) | 0.0020 | 0.0195 | +0.0175 [-0.0047, +0.0394] | +0.0082 | +0.0061 | 2015:1 / 2016:1 | 0 | - | Inconclusive |
| baseline | ridge (vs static) | 0.0020 | 0.0137 | +0.0117 [-0.0072, +0.0299] | +0.0106 | +0.0017 | 2015:1 / 2016:1 | 2 | - | Inconclusive |

## Baseline table: absolute pooled metrics (mean over folds of seed means), sorted by RankIC

| Arm | Exp | Runs | RankIC | IC | NDCG5 excess | NDCG10 excess | NDCG20 excess | spread10 | neutral RankIC | net Sharpe | turnover |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reversal | baseline | 2 | 0.0195 | 0.0238 | 0.0043 | 0.0062 | 0.0087 | 0.0089 | 0.0191 | 1.26 | 0.236 |
| ridge | baseline | 2 | 0.0137 | 0.0149 | 0.0065 | 0.0086 | 0.0096 | 0.0046 | 0.0207 | 1.01 | 0.319 |
| multiscale | D | 6 | 0.0099 | 0.0182 | 0.0110 | 0.0128 | 0.0133 | 0.0058 | -0.0000 | 1.18 | 0.255 |
| summary | G | 6 | 0.0094 | 0.0207 | 0.0136 | 0.0148 | 0.0135 | 0.0086 | -0.0010 | 1.39 | 0.234 |
| soft | A | 6 | 0.0094 | 0.0127 | -0.0064 | -0.0007 | 0.0012 | 0.0052 | 0.0061 | 0.93 | 0.299 |
| hyp | baseline | 6 | 0.0081 | 0.0161 | 0.0075 | 0.0054 | 0.0066 | 0.0044 | -0.0028 | 1.00 | 0.262 |
| mlp | baseline | 6 | 0.0081 | 0.0135 | 0.0045 | 0.0067 | 0.0070 | 0.0066 | 0.0040 | 1.20 | 0.302 |
| pretrain | H | 6 | 0.0053 | 0.0142 | 0.0020 | 0.0037 | 0.0027 | 0.0039 | -0.0055 | 0.91 | 0.286 |
| lowrank | E | 6 | 0.0052 | 0.0085 | 0.0007 | -0.0000 | 0.0004 | 0.0022 | 0.0026 | 0.62 | 0.282 |
| none | F | 6 | 0.0046 | 0.0130 | -0.0001 | 0.0010 | 0.0014 | 0.0052 | -0.0049 | 0.92 | 0.262 |
| residual_soft | A | 6 | 0.0045 | 0.0078 | -0.0010 | 0.0002 | -0.0011 | 0.0031 | 0.0027 | 0.60 | 0.293 |
| adaptive | I | 6 | 0.0045 | 0.0062 | -0.0006 | -0.0007 | 0.0003 | 0.0024 | -0.0053 | 0.63 | 0.293 |
| long | G | 6 | 0.0043 | 0.0113 | 0.0029 | 0.0090 | 0.0100 | 0.0050 | -0.0030 | 0.99 | 0.206 |
| two_residual | F | 6 | 0.0030 | 0.0095 | -0.0071 | -0.0053 | -0.0025 | 0.0030 | -0.0071 | 0.66 | 0.260 |
| cluster_target | B | 6 | 0.0026 | 0.0106 | -0.0050 | -0.0017 | -0.0012 | 0.0047 | 0.0001 | 0.78 | 0.296 |
| two | F | 6 | 0.0023 | 0.0107 | -0.0008 | 0.0011 | 0.0015 | 0.0028 | -0.0105 | 0.66 | 0.278 |
| pairwise | C | 6 | 0.0020 | 0.0104 | 0.0007 | -0.0015 | -0.0021 | 0.0029 | -0.0093 | 0.70 | 0.274 |
| static | baseline | 6 | 0.0020 | 0.0103 | -0.0004 | -0.0020 | -0.0024 | 0.0028 | -0.0094 | 0.68 | 0.274 |
| hybrid | C | 6 | 0.0018 | 0.0102 | -0.0003 | -0.0013 | -0.0019 | 0.0029 | -0.0100 | 0.71 | 0.272 |
| mix | E | 6 | 0.0007 | 0.0043 | 0.0075 | 0.0057 | 0.0032 | 0.0020 | -0.0011 | 0.77 | 0.186 |
| random_graph | A | 6 | 0.0007 | 0.0101 | 0.0029 | 0.0015 | 0.0013 | 0.0036 | -0.0077 | 0.82 | 0.270 |
| random | baseline | 6 | 0.0006 | 0.0014 | -0.0006 | -0.0012 | -0.0009 | 0.0001 | 0.0006 | 0.25 | 0.389 |
| frozen | I | 3 | 0.0006 | 0.0066 | -0.0039 | -0.0026 | -0.0019 | 0.0034 | -0.0153 | 0.99 | 0.261 |
| listwise | C | 6 | 0.0004 | 0.0052 | -0.0093 | -0.0085 | -0.0092 | 0.0005 | -0.0166 | 0.39 | 0.270 |
| gated | E | 6 | -0.0003 | 0.0038 | 0.0056 | 0.0049 | 0.0053 | 0.0018 | -0.0034 | 0.80 | 0.208 |
| rolling | A | 6 | -0.0008 | 0.0092 | 0.0031 | -0.0008 | -0.0013 | 0.0028 | -0.0086 | 0.71 | 0.276 |
| market_target | B | 6 | -0.0014 | 0.0091 | -0.0001 | -0.0005 | -0.0020 | 0.0046 | -0.0114 | 0.82 | 0.277 |
| momentum | baseline | 2 | -0.0195 | -0.0238 | -0.0343 | -0.0304 | -0.0248 | -0.0089 | -0.0191 | -0.45 | 0.240 |

Net Sharpe, turnover: top-10, 10 bp (predeclared default). Single-seed statistical arms (ridge, momentum, reversal) are compared with the seed-mean series of their parent; random has its own seeds.
