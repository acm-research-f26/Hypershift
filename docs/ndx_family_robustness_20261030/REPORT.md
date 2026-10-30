# Model-family isolation — October 30, 2026

The fixed **300-account** follow-up has **0** passing improvement recipes and **0** with SR>1.5 in every group. No new models were fitted in this arm.

![Family comparison](family_comparison.png)

All rows below use retention five, cohort-target averaging, no band, base fees and3%borrow.

| Predictor | Original five SR | Second five SR | Third five SR | Ten SR | Fifteen SR |
|---|---:|---:|---:|---:|---:|
| Fixed 50:50 original blend | 1.168 | 1.269 | 1.345 | 1.267 | 1.330 |
| THINK + stock-ID mixer | 0.814 | 0.938 | 0.877 | 0.882 | 0.883 |
| Dynamic only | 0.735 | 0.584 | 0.854 | 0.696 | 0.779 |
| Shared THINK only | -0.389 | -0.106 | -0.067 | -0.255 | -0.184 |
| Validation-weighted original blend | 1.148 | 1.103 | 1.270 | 1.154 | 1.219 |
| Validation-weighted shared blend | 0.324 | 0.101 | 0.575 | 0.234 | 0.384 |

Both original families alone underperform their fixed50:50blend in this portfolio comparison. Validation-only weight selection also fails to improve it consistently. This does not show one family is universally inferior: rank combination and top-k portfolio decisions are nonlinear. The shared-THINK failure is preserved rather than excluded.

Validation weights are selected independently each year from[0,.25,.5,.75,1] using first-half prior-year meanRankIC−0.5quarterSD across all15seeds. Exact ties prefer0.5thenlowerweight. The diagnostic half is never used to choose weights. Every validation-grid result and every allocation/band/cost case is retained. Group results share model-selection weights and market outcomes; they are not independent market trials.

The follow-up was frozen after the initial540-account study failed its strict goal. This is additional disclosed development search on spent history, not a fresh market test. All300accounts match independent dollar-position replay.

[All accounts](METRICS.csv) · [All recipe/cost summaries](CONSISTENCY.csv) · [Validation grid](VALIDATION.csv) · [Selected family weights](FAMILY_WEIGHTS.json) · [Protocol](PROTOCOL.json)
