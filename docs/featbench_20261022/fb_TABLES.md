# Study FB tables (fb)

Root `results\FEATBENCH_SP500_20261022`. Exploratory (2019–2023 inspected before). Primary metric: factor-neutral RankIC (h-day return minus beta × market, sector-demeaned). Paired daily series on common dates; stationary bootstrap block max(10, 2h), B 5000; pooled = mean of 5 fold means. SUPPORTED = Holm p < .05, CI excludes 0, ≥ 4/5 fold means share the pooled sign.

## Absolute (seed 7; h = 5 unless named)

| Arm | Neutral RankIC pooled [95% CI] | Fold means 2019–2023 | Raw RankIC | Style-neutral | Net Sharpe (mean of folds) | Turnover |
|---|---|---|---|---|---|---|
| block:l1 | +0.0107 [+0.0032, +0.0185] | +0.0009 / +0.0317 / +0.0025 / +0.0060 / +0.0124 | +0.0135 | +0.0062 | +0.49 | 0.38 |
| block:l2_5 | +0.0058 [-0.0075, +0.0194] | +0.0042 / +0.0219 / -0.0041 / +0.0045 / +0.0025 | +0.0098 | +0.0016 | +0.79 | 0.38 |
| block:l6_20 | +0.0065 [-0.0078, +0.0203] | +0.0293 / -0.0034 / -0.0007 / -0.0012 / +0.0085 | +0.0193 | +0.0042 | +1.39 | 0.25 |
| block:l21_60 | -0.0070 [-0.0215, +0.0076] | +0.0070 / -0.0135 / -0.0171 / -0.0034 / -0.0080 | -0.0143 | +0.0062 | +0.58 | 0.17 |
| block:reversal16 | +0.0071 [-0.0084, +0.0221] | +0.0230 / +0.0141 / -0.0131 / +0.0004 / +0.0111 | +0.0114 | +0.0047 | +0.93 | 0.25 |
| ridge:lags | +0.0115 [-0.0045, +0.0279] | +0.0177 / +0.0290 / -0.0056 / +0.0063 / +0.0100 | +0.0191 | +0.0065 | +0.97 | 0.34 |
| ridge:price | -0.0068 [-0.0222, +0.0084] | -0.0032 / +0.0075 / -0.0423 / -0.0024 / +0.0066 | +0.0167 | -0.0017 | +0.82 | 0.26 |
| ridge:rich | -0.0049 [-0.0205, +0.0101] | +0.0025 / +0.0066 / -0.0412 / +0.0023 / +0.0054 | +0.0180 | -0.0010 | +0.84 | 0.25 |
| lgbm:price | +0.0076 [-0.0035, +0.0186] | +0.0092 / +0.0280 / -0.0153 / +0.0044 / +0.0115 | +0.0164 | +0.0015 | +0.97 | 0.23 |
| lgbm:rich | -0.0012 [-0.0127, +0.0098] | -0.0009 / +0.0108 / -0.0170 / -0.0009 / +0.0017 | +0.0065 | +0.0028 | +0.95 | 0.24 |

## Family: primary (Holm over 3 of 3 predeclared)

| Contrast | ΔNeutral RankIC pooled [95% CI] | Fold means | Folds > 0 | Boot p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| lgbm:rich − ridge:rich | +0.0036 [-0.0091, +0.0164] | -0.0034 / +0.0042 / +0.0242 / -0.0032 / -0.0037 | 2/5 | 0.575 | 0.575 | NO EVIDENCE |
| ridge:rich − block:reversal16 | -0.0120 [-0.0264, +0.0018] | -0.0205 / -0.0075 / -0.0282 / +0.0019 / -0.0057 | 1/5 | 0.092 | 0.276 | NO EVIDENCE |
| lgbm:rich − block:reversal16 | -0.0084 [-0.0212, +0.0048] | -0.0239 / -0.0033 / -0.0039 / -0.0013 / -0.0094 | 0/5 | 0.197 | 0.394 | NO EVIDENCE |

## Family: secondary (Holm over 2 of 3 predeclared)

| Contrast | ΔNeutral RankIC pooled [95% CI] | Fold means | Folds > 0 | Boot p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| lgbm:rich − lgbm:price | -0.0088 [-0.0180, +0.0004] | -0.0101 / -0.0172 / -0.0017 / -0.0053 / -0.0098 | 0/5 | 0.060 | 0.121 | NO EVIDENCE |
| ridge:lags − block:reversal16 | +0.0044 [-0.0046, +0.0135] | -0.0053 / +0.0149 / +0.0074 / +0.0059 / -0.0011 | 3/5 | 0.348 | 0.348 | NO EVIDENCE |

## Horizons (descriptive; no cross-horizon verdicts)

| Arm | h | Neutral RankIC pooled [95% CI] | Block | Days / h (effective obs.) | vs reversal16 at same h |
|---|---|---|---|---|---|
| ridge:rich | 5 | -0.0049 [-0.0205, +0.0101] | 10 | 247 | -0.0120 [-0.0264, +0.0018] |
| lgbm:rich | 5 | -0.0012 [-0.0127, +0.0098] | 10 | 247 | -0.0084 [-0.0212, +0.0048] |

## LightGBM seed sensitivity (lgbm:rich, descriptive)

| Seed | Neutral RankIC pooled [95% CI] | vs ridge:rich |
|---|---|---|
| 7 | -0.0012 [-0.0127, +0.0098] | +0.0036 [-0.0091, +0.0164] |
| 19 | -0.0004 [-0.0124, +0.0111] | +0.0045 [-0.0077, +0.0167] |
| 42 | +0.0002 [-0.0098, +0.0095] | +0.0051 [-0.0073, +0.0173] |
