# Pooled Sharpe confidence intervals

<!-- benchmark-audit-20261010:start -->
NYSX is the passive Global X NYSE 100 ETF. It launched March 25, 2026 and has no tradable 2022–2025 history. Its index covers technology and growth companies across exchanges, not the 100 largest NYSE-listed companies. NYSX performance is unavailable here. NYSE historical charts use the simulated equal-weight cohort basket (scheduled rebalances and modeled costs), not an ETF. NASDAQ charts retain QQQ; SPY supplies broad-market context. [Issuer and inception](https://www.globalxetfs.com/funds/nysx/) and [prospectus](https://assets.globalxetfs.com/funds/documents/nysx/prospectus-regulatory/Summary-Prospectus_NYSX.pdf).

The refreshed CHARTS.md pages are the current visual record. Earlier ETF-overlay descriptions are superseded by this correction. Audited return paths and training selections remain preserved.
<!-- benchmark-audit-20261010:end -->

<!-- nyse-benchmark:start -->
**Comparison benchmark:** NYSE tables use **NYSE basket — passive benchmark** (saved `equal_universe` portfolio: equal weights across all stocks eligible at each scheduled rebalance, with modeled execution constraints and costs). Earlier tables use their original NYSE cohort; top-100 tables use the new NYSE cohort. This is a simulated portfolio, not an ETF. Nasdaq retains QQQ; SPY remains a secondary reference. A mixed-cohort “benchmark” column means NYSE basket for NYSE rows and QQQ for Nasdaq rows. Paired confidence intervals were recalculated on identical dates using 1,000 circular 20-session block resamples within each annual fold (seed 20261005). Saved model returns and selection choices are unchanged. Embedded charts retain their previous benchmark overlays; this update applies to tables.

<!-- nyse-benchmark:end -->

<!-- huber-ranking:start -->
## Huber plus ranking: updated results and confidence intervals

All ranking results below are **five-session forecasts for 2022–2025**, trained on CPU before CUDA was configured. The models were not retrained for this documentation update. **95% intervals use 1,000 paired circular 20-session block resamples separately within each annual fold** (seed 20261005). Sharpe uses the same 3% annual hurdle. Absolute Sharpe uncertainty and paired differences are distinct. Intervals condition on saved fits, selected policies and observed years; they do not retrain or adjust for repeated testing, multiple comparisons or survivorship bias.

[Full ranking report](HUBER_RANKING.md) | [Ranking metrics and intervals CSV](huber_ranking_pooled.csv) | [Consolidated 100-series metrics](combined_2022_2025_metrics_current.csv) | [Consolidated interval table](pooled_confidence_intervals_current.csv)

### NASDAQ_recent: equal K10

| Model | Net SR | 95% SR CI | ΔSR vs Huber [95% CI] | ΔSR vs SPY [95% CI] | ΔSR vs QQQ [95% CI] |
| --- | ---: | --- | --- | --- | --- |
| THINK+mixer+Huber | 0.532 | [-0.390, 1.484] | +0.000 [0.000, 0.000] | -0.090 [-0.439, 0.262] | -0.094 [-0.545, 0.349] |
| Huber+ranking, weight 0.1 | 0.529 | [-0.399, 1.482] | -0.002 [-0.104, 0.098] | -0.092 [-0.435, 0.276] | -0.096 [-0.519, 0.365] |
| Huber+ranking, weight 0.3 | 0.488 | [-0.435, 1.426] | -0.044 [-0.128, 0.055] | -0.133 [-0.471, 0.247] | -0.138 [-0.573, 0.325] |
| Huber+ranking, weight 1.0 | 0.397 | [-0.496, 1.354] | -0.135 [-0.269, 0.011] | -0.224 [-0.603, 0.174] | -0.229 [-0.671, 0.271] |
| Validation choice of ranking weight | 0.488 | [-0.435, 1.427] | -0.044 [-0.129, 0.053] | -0.133 [-0.470, 0.247] | -0.138 [-0.574, 0.325] |
| SPY | 0.621 | [-0.215, 1.542] | +0.090 [-0.262, 0.439] | +0.000 [0.000, 0.000] | -0.004 [-0.290, 0.271] |
| QQQ | 0.626 | [-0.234, 1.524] | +0.094 [-0.349, 0.545] | +0.004 [-0.271, 0.290] | +0.000 [0.000, 0.000] |

### NASDAQ_recent: validation-selected

| Model | Net SR | 95% SR CI | ΔSR vs Huber [95% CI] | ΔSR vs SPY [95% CI] | ΔSR vs QQQ [95% CI] |
| --- | ---: | --- | --- | --- | --- |
| THINK+mixer+Huber | 0.675 | [-0.264, 1.652] | +0.000 [0.000, 0.000] | +0.053 [-0.396, 0.524] | +0.049 [-0.445, 0.596] |
| Huber+ranking, weight 0.1 | 0.646 | [-0.285, 1.604] | -0.029 [-0.243, 0.195] | +0.024 [-0.380, 0.489] | +0.020 [-0.497, 0.559] |
| Huber+ranking, weight 0.3 | 0.711 | [-0.191, 1.702] | +0.037 [-0.178, 0.252] | +0.090 [-0.336, 0.562] | +0.085 [-0.391, 0.601] |
| Huber+ranking, weight 1.0 | 0.671 | [-0.256, 1.637] | -0.003 [-0.240, 0.238] | +0.050 [-0.385, 0.531] | +0.045 [-0.464, 0.592] |
| Validation choice of ranking weight | 0.638 | [-0.279, 1.633] | -0.037 [-0.138, 0.062] | +0.017 [-0.416, 0.468] | +0.012 [-0.507, 0.529] |
| SPY | 0.621 | [-0.215, 1.542] | -0.053 [-0.524, 0.396] | +0.000 [0.000, 0.000] | -0.004 [-0.290, 0.271] |
| QQQ | 0.626 | [-0.234, 1.524] | -0.049 [-0.596, 0.445] | +0.004 [-0.271, 0.290] | +0.000 [0.000, 0.000] |

### NYSE_recent: equal K10

| Model | Net SR | 95% SR CI | ΔSR vs Huber [95% CI] | ΔSR vs SPY [95% CI] | ΔSR vs NYSE basket [95% CI] |
| --- | ---: | --- | --- | --- | --- |
| THINK+mixer+Huber | 0.089 | [-0.907, 1.059] | +0.000 [0.000, 0.000] | -0.532 [-1.132, 0.030] | -0.090 [-0.481, 0.273] |
| Huber+ranking, weight 0.1 | 0.154 | [-0.755, 1.059] | +0.065 [-0.141, 0.259] | -0.467 [-1.019, 0.094] | -0.025 [-0.377, 0.327] |
| Huber+ranking, weight 0.3 | 0.109 | [-0.845, 1.051] | +0.021 [-0.209, 0.228] | -0.512 [-1.141, 0.088] | -0.069 [-0.479, 0.306] |
| Huber+ranking, weight 1.0 | 0.101 | [-0.859, 1.067] | +0.012 [-0.198, 0.207] | -0.520 [-1.150, 0.032] | -0.078 [-0.504, 0.294] |
| Validation choice of ranking weight | 0.148 | [-0.781, 1.098] | +0.059 [-0.137, 0.260] | -0.473 [-1.072, 0.097] | -0.031 [-0.404, 0.327] |
| SPY | 0.621 | [-0.215, 1.542] | +0.532 [-0.030, 1.132] | +0.000 [0.000, 0.000] | +0.443 [-0.052, 0.942] |
| NYSE basket — passive benchmark | 0.179 | [-0.720, 1.082] | +0.090 [-0.273, 0.481] | -0.443 [-0.942, 0.052] | +0.000 [0.000, 0.000] |

### NYSE_recent: validation-selected

| Model | Net SR | 95% SR CI | ΔSR vs Huber [95% CI] | ΔSR vs SPY [95% CI] | ΔSR vs NYSE basket [95% CI] |
| --- | ---: | --- | --- | --- | --- |
| THINK+mixer+Huber | 0.194 | [-0.716, 1.151] | +0.000 [0.000, 0.000] | -0.427 [-0.929, 0.053] | +0.015 [-0.303, 0.297] |
| Huber+ranking, weight 0.1 | 0.135 | [-0.798, 1.107] | -0.059 [-0.184, 0.064] | -0.486 [-1.020, 0.038] | -0.044 [-0.378, 0.269] |
| Huber+ranking, weight 0.3 | 0.032 | [-0.847, 0.925] | -0.162 [-0.471, 0.117] | -0.589 [-1.182, -0.025] | -0.146 [-0.548, 0.234] |
| Huber+ranking, weight 1.0 | 0.239 | [-0.662, 1.120] | +0.045 [-0.433, 0.473] | -0.382 [-1.042, 0.249] | +0.060 [-0.459, 0.556] |
| Validation choice of ranking weight | 0.189 | [-0.767, 1.131] | -0.005 [-0.232, 0.218] | -0.432 [-1.010, 0.117] | +0.011 [-0.382, 0.370] |
| SPY | 0.621 | [-0.215, 1.542] | +0.427 [-0.053, 0.929] | +0.000 [0.000, 0.000] | +0.443 [-0.052, 0.942] |
| NYSE basket — passive benchmark | 0.179 | [-0.720, 1.082] | -0.015 [-0.297, 0.303] | -0.443 [-0.942, 0.052] | +0.000 [0.000, 0.000] |

**Interpretation:** ranking weight 0.3 reaches NASDAQ Sharpe 0.711 with its selected portfolio, but its advantage over Huber is +0.037, CI [-0.178, 0.252]. NYSE weight 1.0 reaches 0.239 versus Huber 0.194, with gain CI [-0.433, 0.473]. Neither establishes an improvement. The primary fixed-K10 validation-selected ranking procedure changes NASDAQ 0.532→0.488 (gain CI [-0.129, 0.053]) and NYSE 0.089→0.148 (gain CI [-0.137, 0.260]). Secondary validation-chosen ranking weights yield NASDAQ 0.638 and NYSE 0.189. No best test-performing weight or K is promoted to the validation-selected procedure.

[Verified top-5/top-10/top-20 paired intervals](huber_ranking_k_pooled.csv). The original batch’s interval tables follow below.

<!-- huber-ranking:end -->

**These are exploratory 95% percentile block-bootstrap intervals.** Each of 1,000 replicates resamples contiguous 20-session circular blocks separately within each annual test fold. Model and benchmark returns use identical sampled dates. The random seed is 20261005, matching the predeclared analysis. Every previously published pooled model-versus-ETF interval was independently reproduced.

- **Sharpe CI** describes uncertainty in a model’s own pooled net Sharpe.
- **ΔSR vs SPY/QQQ CI** describes the paired difference between the two Sharpes. It is not obtained by subtracting endpoints of separate Sharpe intervals.
- An interval containing zero does not establish an advantage. These intervals condition on the four observed years, fixed trained models and selected policies. They do not rerun training/selection or correct for repeated testing, multiple comparisons, survivorship bias or future regime uncertainty.
- Annual folds reset sizing and omit boundary sessions. All comparisons use identical dates within each horizon. Returns include modeled costs and Sharpe uses a 3% annual hurdle.

## Validation-selected architecture procedures

| Procedure | Cohort | Net SR | 95% SR CI | ΔSR vs SPY | 95% ΔSR CI | ΔSR vs benchmark | 95% ΔSR CI |
| --- | --- | ---: | --- | ---: | --- | ---: | --- |
| Five-session | NASDAQ_recent | 0.379 | [-0.429, 1.152] | -0.242 | [-0.817, 0.288] | -0.247 | [-0.839, 0.348] |
| Five-session | NYSE_recent | -0.540 | [-1.394, 0.318] | -1.161 | [-1.876, -0.474] | -0.718 | [-1.182, -0.233] |
| Daily | NASDAQ_recent | -0.616 | [-1.469, 0.187] | -1.113 | [-1.739, -0.524] | -1.115 | [-1.819, -0.453] |
| Daily | NYSE_recent | -0.192 | [-0.971, 0.682] | -0.688 | [-1.337, -0.018] | -0.277 | [-0.799, 0.253] |
| Joint validation choice | NASDAQ_recent | 0.379 | [-0.429, 1.152] | -0.242 | [-0.817, 0.288] | -0.247 | [-0.839, 0.348] |
| Joint validation choice | NYSE_recent | -0.584 | [-1.387, 0.352] | -1.156 | [-1.817, -0.466] | -0.710 | [-1.195, -0.210] |

## NASDAQ_recent: 5-session models and controls

| Model | Net SR | 95% SR CI | ΔSR vs SPY | 95% ΔSR CI | ΔSR vs QQQ | 95% ΔSR CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| think | 0.265 | [-0.532, 1.167] | -0.356 | [-0.846, 0.191] | -0.361 | [-0.935, 0.279] |
| think_rank | 0.222 | [-0.595, 1.041] | -0.399 | [-0.883, 0.143] | -0.403 | [-0.982, 0.227] |
| think_gate | 0.361 | [-0.455, 1.253] | -0.261 | [-0.739, 0.279] | -0.265 | [-0.873, 0.366] |
| think_mix | 0.356 | [-0.564, 1.299] | -0.265 | [-0.706, 0.263] | -0.270 | [-0.843, 0.358] |
| think_risk | 0.259 | [-0.559, 1.075] | -0.362 | [-0.840, 0.139] | -0.367 | [-0.974, 0.233] |
| mlp | 0.291 | [-0.535, 1.142] | -0.331 | [-0.885, 0.230] | -0.335 | [-0.957, 0.347] |
| lstm | 0.599 | [-0.266, 1.562] | -0.022 | [-0.632, 0.537] | -0.026 | [-0.706, 0.661] |
| stockmixer | 0.381 | [-0.606, 1.386] | -0.241 | [-0.802, 0.379] | -0.245 | [-0.921, 0.499] |
| market_transformer | 0.059 | [-0.851, 0.934] | -0.563 | [-1.140, -0.040] | -0.567 | [-1.153, 0.018] |
| think_huber | 0.228 | [-0.599, 1.067] | -0.393 | [-0.962, 0.200] | -0.398 | [-1.063, 0.259] |
| think_mix_huber | 0.675 | [-0.264, 1.652] | 0.053 | [-0.396, 0.524] | 0.049 | [-0.445, 0.596] |
| stockmixer_huber | 0.364 | [-0.511, 1.380] | -0.257 | [-0.784, 0.287] | -0.261 | [-0.889, 0.404] |
| think_mix_shrink | 0.452 | [-0.430, 1.315] | -0.170 | [-0.663, 0.371] | -0.174 | [-0.778, 0.468] |
| think_mix_shrink_huber | 0.620 | [-0.274, 1.546] | -0.002 | [-0.427, 0.440] | -0.006 | [-0.527, 0.528] |
| multiscale_mse | 0.448 | [-0.455, 1.377] | -0.173 | [-0.658, 0.336] | -0.178 | [-0.784, 0.445] |
| multiscale_huber | 0.419 | [-0.430, 1.339] | -0.202 | [-0.652, 0.260] | -0.206 | [-0.748, 0.376] |
| ridge | 0.194 | [-0.697, 1.133] | -0.427 | [-0.860, 0.046] | -0.432 | [-0.998, 0.147] |
| momentum126 | 0.381 | [-0.405, 1.225] | -0.240 | [-0.812, 0.303] | -0.245 | [-0.863, 0.397] |
| reversal5 | 0.204 | [-0.615, 1.041] | -0.417 | [-1.038, 0.169] | -0.421 | [-1.102, 0.269] |
| buy_hold | 0.341 | [-0.490, 1.255] | -0.281 | [-0.705, 0.161] | -0.285 | [-0.830, 0.274] |
| equal_universe | 0.364 | [-0.472, 1.283] | -0.258 | [-0.695, 0.191] | -0.262 | [-0.816, 0.339] |
| SPY | 0.621 | [-0.215, 1.542] | 0.000 | [0.000, 0.000] | -0.004 | [-0.290, 0.271] |
| qqq | 0.626 | [-0.234, 1.524] | 0.004 | [-0.271, 0.290] | 0.000 | [0.000, 0.000] |

## NYSE_recent: 5-session models and controls

| Model | Net SR | 95% SR CI | ΔSR vs SPY | 95% ΔSR CI | ΔSR vs NYSE basket | 95% ΔSR CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| think | 0.017 | [-0.974, 0.972] | -0.605 | [-1.178, -0.094] | -0.162 | [-0.539, 0.225] |
| think_rank | -0.185 | [-1.204, 0.861] | -0.806 | [-1.419, -0.233] | -0.363 | [-0.831, 0.114] |
| think_gate | 0.096 | [-0.874, 1.018] | -0.525 | [-1.074, -0.001] | -0.082 | [-0.442, 0.303] |
| think_mix | 0.301 | [-0.640, 1.181] | -0.321 | [-0.942, 0.285] | 0.122 | [-0.294, 0.552] |
| think_risk | -0.246 | [-1.179, 0.719] | -0.867 | [-1.455, -0.301] | -0.424 | [-0.811, -0.069] |
| mlp | -0.237 | [-1.067, 0.552] | -0.859 | [-1.549, -0.257] | -0.416 | [-0.778, -0.065] |
| lstm | -0.011 | [-0.849, 0.840] | -0.632 | [-1.170, -0.101] | -0.190 | [-0.483, 0.090] |
| stockmixer | -0.118 | [-0.995, 0.687] | -0.739 | [-1.343, -0.220] | -0.297 | [-0.727, 0.056] |
| market_transformer | -0.328 | [-1.117, 0.461] | -0.949 | [-1.598, -0.308] | -0.506 | [-0.968, -0.070] |
| think_huber | -0.054 | [-1.015, 0.907] | -0.675 | [-1.248, -0.124] | -0.232 | [-0.680, 0.178] |
| think_mix_huber | 0.194 | [-0.716, 1.151] | -0.427 | [-0.929, 0.053] | 0.015 | [-0.303, 0.297] |
| stockmixer_huber | -0.157 | [-1.040, 0.706] | -0.778 | [-1.322, -0.265] | -0.335 | [-0.683, -0.016] |
| think_mix_shrink | 0.075 | [-0.830, 1.021] | -0.546 | [-1.174, 0.061] | -0.104 | [-0.529, 0.348] |
| think_mix_shrink_huber | -0.176 | [-1.059, 0.745] | -0.797 | [-1.417, -0.169] | -0.354 | [-0.797, 0.095] |
| multiscale_mse | -0.114 | [-0.950, 0.691] | -0.735 | [-1.365, -0.183] | -0.293 | [-0.665, 0.025] |
| multiscale_huber | -0.113 | [-1.005, 0.773] | -0.734 | [-1.354, -0.144] | -0.291 | [-0.621, -0.006] |
| ridge | -0.030 | [-0.885, 0.791] | -0.651 | [-1.376, 0.006] | -0.208 | [-0.727, 0.241] |
| momentum126 | 0.393 | [-0.507, 1.381] | -0.228 | [-0.809, 0.328] | 0.215 | [-0.227, 0.656] |
| reversal5 | -0.007 | [-0.873, 0.845] | -0.628 | [-1.366, 0.118] | -0.186 | [-0.711, 0.315] |
| buy_hold | 0.213 | [-0.707, 1.134] | -0.408 | [-0.902, 0.081] | 0.035 | [-0.042, 0.116] |
| NYSE basket — passive benchmark | 0.179 | [-0.720, 1.082] | -0.443 | [-0.942, 0.052] | 0.000 | [0.000, 0.000] |
| SPY | 0.621 | [-0.215, 1.542] | 0.000 | [0.000, 0.000] | 0.443 | [-0.052, 0.942] |

## NASDAQ_recent: 1-session models and controls

| Model | Net SR | 95% SR CI | ΔSR vs SPY | 95% ΔSR CI | ΔSR vs QQQ | 95% ΔSR CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| think | -0.097 | [-0.973, 0.789] | -0.594 | [-1.111, -0.075] | -0.596 | [-1.232, 0.066] |
| think_huber | -0.182 | [-0.998, 0.669] | -0.678 | [-1.172, -0.179] | -0.681 | [-1.264, -0.055] |
| think_mix | 0.383 | [-0.515, 1.373] | -0.113 | [-0.649, 0.400] | -0.115 | [-0.730, 0.517] |
| think_mix_huber | 0.291 | [-0.542, 1.137] | -0.206 | [-0.710, 0.305] | -0.208 | [-0.832, 0.456] |
| stockmixer | 0.055 | [-0.810, 0.989] | -0.442 | [-0.988, 0.080] | -0.444 | [-1.090, 0.199] |
| stockmixer_huber | 0.251 | [-0.609, 1.121] | -0.246 | [-0.726, 0.239] | -0.248 | [-0.836, 0.338] |
| multiscale_mse | 0.161 | [-0.689, 0.987] | -0.335 | [-0.834, 0.156] | -0.337 | [-0.927, 0.257] |
| multiscale_huber | 0.088 | [-0.743, 0.970] | -0.409 | [-0.907, 0.117] | -0.411 | [-1.020, 0.224] |
| ridge | -0.842 | [-1.705, 0.029] | -1.339 | [-1.909, -0.770] | -1.341 | [-1.989, -0.653] |
| momentum126 | -0.004 | [-0.808, 0.826] | -0.501 | [-1.054, 0.051] | -0.503 | [-1.122, 0.142] |
| reversal5 | -0.092 | [-0.840, 0.696] | -0.588 | [-1.189, -0.002] | -0.590 | [-1.250, 0.079] |
| buy_hold | 0.251 | [-0.576, 1.087] | -0.246 | [-0.684, 0.186] | -0.248 | [-0.784, 0.300] |
| equal_universe | 0.255 | [-0.570, 1.090] | -0.241 | [-0.684, 0.197] | -0.243 | [-0.780, 0.316] |
| SPY | 0.497 | [-0.341, 1.360] | 0.000 | [0.000, 0.000] | -0.002 | [-0.264, 0.263] |
| qqq | 0.499 | [-0.333, 1.399] | 0.002 | [-0.263, 0.264] | 0.000 | [0.000, 0.000] |

## NYSE_recent: 1-session models and controls

| Model | Net SR | 95% SR CI | ΔSR vs SPY | 95% ΔSR CI | ΔSR vs NYSE basket | 95% ΔSR CI |
| --- | ---: | --- | ---: | --- | ---: | --- |
| think | -0.262 | [-1.078, 0.606] | -0.759 | [-1.297, -0.200] | -0.348 | [-0.665, -0.010] |
| think_huber | -0.597 | [-1.435, 0.305] | -1.094 | [-1.712, -0.513] | -0.683 | [-1.019, -0.335] |
| think_mix | -0.229 | [-1.018, 0.613] | -0.726 | [-1.386, -0.081] | -0.315 | [-0.809, 0.166] |
| think_mix_huber | -0.180 | [-1.016, 0.745] | -0.676 | [-1.258, -0.079] | -0.265 | [-0.722, 0.263] |
| stockmixer | -0.217 | [-0.965, 0.646] | -0.714 | [-1.349, -0.080] | -0.303 | [-0.693, 0.108] |
| stockmixer_huber | -0.482 | [-1.297, 0.376] | -0.979 | [-1.624, -0.311] | -0.567 | [-0.961, -0.183] |
| multiscale_mse | -0.397 | [-1.195, 0.427] | -0.893 | [-1.523, -0.291] | -0.482 | [-0.816, -0.164] |
| multiscale_huber | -0.329 | [-1.179, 0.579] | -0.825 | [-1.450, -0.218] | -0.414 | [-0.833, 0.003] |
| ridge | -0.939 | [-1.797, -0.031] | -1.436 | [-2.065, -0.798] | -1.025 | [-1.450, -0.595] |
| momentum126 | 0.079 | [-0.823, 0.879] | -0.417 | [-0.980, 0.123] | -0.006 | [-0.475, 0.453] |
| reversal5 | -0.226 | [-1.019, 0.642] | -0.723 | [-1.358, -0.068] | -0.312 | [-0.708, 0.064] |
| buy_hold | 0.139 | [-0.739, 1.021] | -0.358 | [-0.812, 0.116] | 0.053 | [-0.021, 0.128] |
| NYSE basket — passive benchmark | 0.086 | [-0.784, 0.983] | -0.411 | [-0.872, 0.076] | 0.000 | [0.000, 0.000] |
| SPY | 0.497 | [-0.341, 1.360] | 0.000 | [0.000, 0.000] | 0.411 | [-0.076, 0.872] |

## Huber versus MSE: paired differences

The fixed-equal-top-10 comparison holds portfolio construction constant. The validation-selected comparison permits each model’s previously selected portfolio and K to differ.

| Horizon | Cohort | Portfolio | MSE SR | Huber SR | Huber minus MSE | 95% difference CI |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 5 | NASDAQ_recent | validation-selected per model | 0.356 | 0.675 | 0.319 | [0.036, 0.616] |
| 5 | NASDAQ_recent | fixed equal K10 | 0.293 | 0.532 | 0.239 | [-0.033, 0.523] |
| 5 | NYSE_recent | validation-selected per model | 0.301 | 0.194 | -0.107 | [-0.498, 0.266] |
| 5 | NYSE_recent | fixed equal K10 | 0.170 | 0.089 | -0.081 | [-0.408, 0.261] |
| 1 | NASDAQ_recent | validation-selected per model | 0.383 | 0.291 | -0.092 | [-0.565, 0.412] |
| 1 | NASDAQ_recent | fixed equal K10 | 0.332 | 0.103 | -0.229 | [-0.525, 0.063] |
| 1 | NYSE_recent | validation-selected per model | -0.229 | -0.180 | 0.049 | [-0.340, 0.445] |
| 1 | NYSE_recent | fixed equal K10 | -0.231 | -0.171 | 0.060 | [-0.254, 0.423] |

[All model intervals](pooled_confidence_intervals.csv) | [Procedure intervals](procedure_confidence_intervals.csv) | [Huber/MSE intervals](huber_mse_pooled_intervals.csv)
