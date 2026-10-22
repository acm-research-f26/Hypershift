# A4 — Matched target tests at fixed rounds, seeds and a target ladder

Exploratory audit refits (`scripts/fbaudit_refit.py`). LightGBM with FB parameters and cs_z inputs on the 20 rich features; only the target (and, where stated, the number of rounds) changes. Fixed-round fits use `prospective.fit_fixed` on the FB training window (no validation step); "es" = FB early stopping on validation neutral RankIC. Score: factor-neutral RankIC, pooled mean of 5 fold means, stationary bootstrap block 10.

## T4: residual − raw at fixed rounds (predeclared)

| Series | Pooled [95% CI] | Fold means | Folds > 0 |
|---|---|---|---|
| residual − raw, 50 rounds | +0.0138 [+0.0013, +0.0266] | +0.0089 / +0.0157 / +0.0270 / +0.0139 / +0.0033 | 5/5 |
| residual − raw, 300 rounds | +0.0107 [+0.0000, +0.0212] | +0.0019 / +0.0184 / +0.0256 / +0.0126 / -0.0052 | 4/5 |
| raw, 50 rounds (absolute) | -0.0004 [-0.0120, +0.0109] | +0.0025 / +0.0136 / -0.0169 / -0.0044 / +0.0035 | 3/5 |
| residual, 50 rounds (absolute) | +0.0134 [+0.0030, +0.0242] | +0.0113 / +0.0293 / +0.0102 / +0.0095 / +0.0068 | 5/5 |
| raw, 300 rounds (absolute) | +0.0009 [-0.0101, +0.0114] | +0.0016 / +0.0108 / -0.0163 / -0.0018 / +0.0100 | 3/5 |
| residual, 300 rounds (absolute) | +0.0115 [+0.0025, +0.0205] | +0.0036 / +0.0292 / +0.0093 / +0.0108 / +0.0048 | 5/5 |

**T4 verdict: target effect robust to rounds** (50 rounds: pass; 300 rounds: pass).

## Development estimate of the frozen specification (lgbm:rich:residual, 50 rounds, descriptive)

| Series | Pooled [95% CI] | Fold means | Folds > 0 |
|---|---|---|---|
| frozen spec (50 rounds) vs 0 | +0.0134 [+0.0030, +0.0242] | +0.0113 / +0.0293 / +0.0102 / +0.0095 / +0.0068 | 5/5 |
| frozen spec (50 rounds) − reversal16 | +0.0063 [-0.0060, +0.0188] | -0.0117 / +0.0152 / +0.0232 / +0.0091 / -0.0043 | 3/5 |
| FB early-stopped residual (reported +0.0127) − reversal16 | +0.0056 [-0.0071, +0.0184] | -0.0108 / +0.0101 / +0.0223 / +0.0107 / -0.0043 | 3/5 |

## Seeds (early-stopped, as in FB)

| Series | Pooled [95% CI] | Fold means | Folds > 0 |
|---|---|---|---|
| residual − raw, seed 7 | +0.0140 [+0.0022, +0.0260] | +0.0131 / +0.0134 / +0.0262 / +0.0121 / +0.0051 | 5/5 |
| residual − raw, seed 19 | +0.0113 [-0.0005, +0.0230] | +0.0065 / +0.0181 / +0.0304 / +0.0098 / -0.0082 | 4/5 |
| residual − raw, seed 42 | +0.0121 [+0.0010, +0.0234] | +0.0001 / +0.0167 / +0.0281 / +0.0082 / +0.0073 | 5/5 |
| residual − raw, mean of 3 seeds | +0.0125 [+0.0014, +0.0238] | +0.0066 / +0.0161 / +0.0283 / +0.0100 / +0.0014 | 5/5 |
| residual (mean of 3 seeds) vs 0 | +0.0120 [+0.0025, +0.0214] | +0.0088 / +0.0254 / +0.0105 / +0.0095 / +0.0058 | 5/5 |

## Target ladder

LightGBM at 50 fixed rounds; Ridge with FB alpha selection (raw/sector/residual Ridge rows are the FB runs).

| Series | Pooled [95% CI] | Fold means | Folds > 0 |
|---|---|---|---|
| lgbm raw | -0.0004 [-0.0120, +0.0109] | +0.0025 / +0.0136 / -0.0169 / -0.0044 / +0.0035 | 3/5 |
| lgbm demean | +0.0045 [-0.0098, +0.0188] | +0.0023 / +0.0272 / -0.0240 / +0.0102 / +0.0066 | 4/5 |
| lgbm sector | +0.0086 [-0.0049, +0.0227] | +0.0058 / +0.0296 / -0.0091 / +0.0111 / +0.0058 | 4/5 |
| lgbm sector_beta | +0.0114 [-0.0007, +0.0236] | +0.0122 / +0.0289 / +0.0009 / +0.0088 / +0.0063 | 5/5 |
| lgbm neutral | +0.0248 [+0.0113, +0.0386] | +0.0281 / +0.0368 / +0.0328 / +0.0119 / +0.0144 | 5/5 |
| lgbm residual | +0.0134 [+0.0030, +0.0242] | +0.0113 / +0.0293 / +0.0102 / +0.0095 / +0.0068 | 5/5 |
| lgbm demean − raw | +0.0048 [-0.0060, +0.0153] | -0.0001 / +0.0137 / -0.0072 / +0.0146 / +0.0031 | 3/5 |
| lgbm sector − raw | +0.0090 [-0.0021, +0.0198] | +0.0033 / +0.0160 / +0.0078 / +0.0155 / +0.0023 | 5/5 |
| lgbm sector_beta − raw | +0.0118 [-0.0002, +0.0237] | +0.0097 / +0.0154 / +0.0177 / +0.0133 / +0.0028 | 5/5 |
| lgbm neutral − raw | +0.0251 [+0.0080, +0.0417] | +0.0256 / +0.0232 / +0.0496 / +0.0163 / +0.0109 | 5/5 |
| lgbm residual − raw | +0.0138 [+0.0013, +0.0266] | +0.0089 / +0.0157 / +0.0270 / +0.0139 / +0.0033 | 5/5 |
| ridge raw | -0.0049 [-0.0196, +0.0105] | +0.0025 / +0.0066 / -0.0412 / +0.0023 / +0.0054 | 4/5 |
| ridge demean | -0.0050 [-0.0199, +0.0104] | +0.0014 / +0.0074 / -0.0417 / +0.0026 / +0.0053 | 4/5 |
| ridge sector | +0.0026 [-0.0124, +0.0177] | +0.0060 / +0.0184 / -0.0214 / +0.0070 / +0.0028 | 4/5 |
| ridge sector_beta | +0.0053 [-0.0090, +0.0199] | +0.0160 / +0.0151 / -0.0107 / +0.0072 / -0.0011 | 3/5 |
| ridge neutral | +0.0210 [+0.0056, +0.0366] | +0.0334 / +0.0280 / +0.0281 / +0.0071 / +0.0082 | 5/5 |
| ridge residual | +0.0057 [-0.0089, +0.0202] | +0.0123 / +0.0235 / -0.0070 / +0.0059 / -0.0065 | 3/5 |
| ridge demean − raw | -0.0001 [-0.0005, +0.0002] | -0.0011 / +0.0008 / -0.0005 / +0.0003 / -0.0000 | 2/5 |
| ridge sector − raw | +0.0075 [+0.0014, +0.0133] | +0.0035 / +0.0118 / +0.0198 / +0.0047 / -0.0025 | 4/5 |
| ridge sector_beta − raw | +0.0102 [-0.0002, +0.0202] | +0.0135 / +0.0085 / +0.0305 / +0.0049 / -0.0064 | 4/5 |
| ridge neutral − raw | +0.0258 [+0.0077, +0.0434] | +0.0309 / +0.0213 / +0.0693 / +0.0048 / +0.0028 | 5/5 |
| ridge residual − raw | +0.0105 [-0.0034, +0.0242] | +0.0098 / +0.0169 / +0.0343 / +0.0036 / -0.0118 | 4/5 |
