# NASDAQ-100 strategy comparison

2022-01-03 to 2025-12-31. Net returns at base modeled costs. SR uses a fixed 3% annual hurdle.

95% SR intervals: 10,000 common-date circular-block samples, 20-session primary blocks. These are pointwise fitted-path intervals, not corrected superiority claims.

## Models

| Strategy | Study | Sharpe | CAGR | RankIC | Max drawdown | SR 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SPY | Original daily | 0.498 | 10.85% | N/A | -24.55% | [-0.367, 1.419] |
| QQQ | Original daily | 0.469 | 11.93% | N/A | -34.87% | [-0.408, 1.394] |
| MLP | Original daily | -0.208 | -10.14% | -0.0108 | -44.51% | [-1.009, 0.569] |
| 126-session momentum | Original daily | 0.145 | 2.94% | 0.0139 | -38.37% | [-0.738, 1.084] |
| Monthly equal-weight Nasdaq-100 basket | Original daily | 0.279 | 6.89% | N/A | -30.50% | [-0.567, 1.158] |
| Five-session reversal | Original daily | 0.246 | 5.81% | 0.0183 | -35.78% | [-0.547, 1.046] |
| Ridge | Original daily | -0.058 | -4.99% | -0.0009 | -39.13% | [-0.864, 0.743] |
| StockMixer | Original daily | 0.326 | 8.65% | 0.0186 | -33.99% | [-0.484, 1.169] |
| THINK | Original daily | 0.435 | 12.30% | 0.0206 | -29.00% | [-0.368, 1.271] |
| THINK + Huber | Original daily | 0.475 | 13.66% | 0.0197 | -36.74% | [-0.376, 1.367] |
| THINK + mixer (original five seeds) | Original daily | 0.629 | 19.60% | 0.0074 | -39.44% | [-0.186, 1.505] |
| THINK + mixer + Huber | Original daily | 0.245 | 6.03% | 0.0038 | -41.70% | [-0.603, 1.124] |
| THINK + mixer + pseudo-Huber | Original daily | 0.412 | 11.39% | 0.0045 | -40.70% | [-0.437, 1.293] |
| MLP / Daily padded | Alpaca matched features | -0.098 | -6.24% | -0.0142 | -33.97% | [-0.898, 0.687] |
| Ridge / Daily padded | Alpaca matched features | -0.058 | -4.99% | -0.0009 | -39.13% | [-0.864, 0.743] |
| StockMixer / Daily padded | Alpaca matched features | 0.292 | 7.49% | 0.0155 | -29.59% | [-0.443, 1.031] |
| THINK / Daily padded | Alpaca matched features | 0.286 | 7.32% | 0.0046 | -34.67% | [-0.495, 1.120] |
| THINK + Huber / Daily padded | Alpaca matched features | 0.047 | 0.54% | 0.0038 | -41.86% | [-0.804, 0.920] |
| THINK + mixer / Daily padded | Alpaca matched features | 0.464 | 13.35% | 0.0077 | -40.72% | [-0.365, 1.319] |
| THINK + mixer + Huber / Daily padded | Alpaca matched features | 0.567 | 15.80% | 0.0011 | -31.29% | [-0.247, 1.426] |
| THINK + mixer + pseudo-Huber / Daily padded | Alpaca matched features | 0.398 | 10.73% | 0.0019 | -37.47% | [-0.404, 1.256] |
| MLP / Alpaca intraday | Alpaca matched features | 0.004 | -3.87% | -0.0016 | -43.52% | [-0.795, 0.802] |
| Ridge / Alpaca intraday | Alpaca matched features | 0.016 | -1.91% | -0.0098 | -35.05% | [-0.839, 0.885] |
| StockMixer / Alpaca intraday | Alpaca matched features | 0.192 | 3.56% | 0.0104 | -38.43% | [-0.562, 0.960] |
| THINK / Alpaca intraday | Alpaca matched features | 0.078 | 0.72% | -0.0052 | -37.26% | [-0.694, 0.876] |
| THINK + Huber / Alpaca intraday | Alpaca matched features | -0.068 | -3.17% | -0.0008 | -44.93% | [-0.914, 0.802] |
| THINK + mixer / Alpaca intraday | Alpaca matched features | 0.337 | 9.04% | 0.0061 | -42.98% | [-0.482, 1.197] |
| THINK + mixer + Huber / Alpaca intraday | Alpaca matched features | 0.397 | 10.64% | 0.0027 | -36.09% | [-0.453, 1.282] |
| THINK + mixer + pseudo-Huber / Alpaca intraday | Alpaca matched features | 0.427 | 11.73% | 0.0026 | -38.87% | [-0.398, 1.291] |
| Raw MSE / RankIC stopping / daily padded | Ranking objectives | 0.573 | 17.02% | 0.0211 | -32.95% | [-0.225, 1.418] |
| QQQ-residual MSE / daily padded | Ranking objectives | 0.487 | 13.78% | 0.0151 | -44.04% | [-0.427, 1.432] |
| QQQ-residual + ranking / daily padded | Ranking objectives | 0.453 | 12.61% | 0.0117 | -44.04% | [-0.467, 1.408] |
| QQQ-residual ridge / daily padded | Ranking objectives | 0.245 | 5.97% | 0.0050 | -31.82% | [-0.599, 1.074] |
| Raw MSE / RankIC stopping / Alpaca intraday | Ranking objectives | 0.560 | 17.33% | 0.0157 | -38.41% | [-0.288, 1.453] |
| QQQ-residual MSE / Alpaca intraday | Ranking objectives | 0.364 | 9.83% | 0.0076 | -44.04% | [-0.520, 1.274] |
| QQQ-residual + ranking / Alpaca intraday | Ranking objectives | 0.296 | 7.63% | 0.0083 | -44.04% | [-0.581, 1.195] |
| QQQ-residual ridge / Alpaca intraday | Ranking objectives | 0.194 | 4.32% | -0.0078 | -33.49% | [-0.635, 1.031] |
| Boosted trees (daily, raw) | Broad features | 0.322 | 8.50% | 0.0087 | -46.43% | [-0.534, 1.201] |
| Ridge (daily, raw) | Broad features | 0.359 | 9.85% | 0.0036 | -31.19% | [-0.457, 1.201] |
| Shared neural net (daily, raw) | Broad features | 0.077 | -1.14% | 0.0079 | -49.77% | [-0.715, 0.880] |
| Boosted trees (daily, residual) | Broad features | 0.034 | -1.29% | 0.0038 | -45.51% | [-0.847, 0.865] |
| Ridge (daily, residual) | Broad features | -0.035 | -3.17% | -0.0023 | -38.17% | [-0.906, 0.835] |
| Shared neural net (daily, residual) | Broad features | 0.147 | 2.36% | -0.0054 | -38.43% | [-0.646, 0.922] |
| Equal blend of all 12 models | Broad features | 0.036 | -1.46% | 0.0014 | -44.59% | [-0.785, 0.854] |
| Boosted trees (intraday, raw) | Broad features | 0.283 | 7.13% | 0.0070 | -33.21% | [-0.547, 1.114] |
| Ridge (intraday, raw) | Broad features | 0.149 | 2.79% | 0.0064 | -34.90% | [-0.635, 0.979] |
| Shared neural net (intraday, raw) | Broad features | 0.161 | 1.74% | 0.0082 | -40.32% | [-0.623, 0.969] |
| Boosted trees (intraday, residual) | Broad features | 0.127 | 1.30% | 0.0054 | -43.10% | [-0.726, 0.975] |
| Ridge (intraday, residual) | Broad features | 0.113 | 1.51% | 0.0012 | -32.98% | [-0.696, 0.917] |
| Shared neural net (intraday, residual) | Broad features | 0.146 | 2.01% | 0.0014 | -39.66% | [-0.632, 0.938] |
| Validation best single model | Broad features | -0.115 | -5.63% | -0.0114 | -42.96% | [-0.947, 0.701] |
| Validation top-three blend | Broad features | 0.107 | 1.23% | 0.0060 | -35.93% | [-0.708, 0.914] |
| Intraday-window correction / utility (gated) | Frozen-core corrections | 0.548 | 16.72% | 0.0077 | -39.44% | [-0.244, 1.407] |
| Intraday-window correction / ranking (gated) | Frozen-core corrections | 0.459 | 13.20% | 0.0036 | -39.44% | [-0.307, 1.279] |
| Paired core (H1 validation stopping) | Frozen-core corrections | 0.548 | 16.72% | 0.0077 | -39.44% | [-0.244, 1.407] |
| Validation-gated correction | Frozen-core corrections | 0.510 | 15.11% | 0.0045 | -39.44% | [-0.271, 1.350] |
| Rich-feature correction / utility (gated) | Frozen-core corrections | 0.510 | 15.11% | 0.0045 | -39.44% | [-0.271, 1.350] |
| Rich-feature correction / ranking (gated) | Frozen-core corrections | 0.546 | 16.48% | 0.0064 | -39.44% | [-0.242, 1.389] |
| Intraday-window correction / utility (ungated) | Frozen-core corrections | 0.546 | 16.47% | 0.0060 | -39.44% | [-0.251, 1.402] |
| Intraday-window correction / ranking (ungated) | Frozen-core corrections | 0.452 | 12.94% | 0.0040 | -39.44% | [-0.314, 1.273] |
| Rich-feature correction / utility (ungated) | Frozen-core corrections | 0.453 | 13.09% | 0.0060 | -39.77% | [-0.339, 1.298] |
| Rich-feature correction / ranking (ungated) | Frozen-core corrections | 0.546 | 16.48% | 0.0064 | -39.44% | [-0.242, 1.389] |
| THINK + mixer (25-seed ensemble) | Seed expansion | 0.452 | 12.86% | 0.0080 | -36.78% | [-0.363, 1.330] |
| THINK + mixer (20 new seeds) | Seed expansion | 0.503 | 14.42% | 0.0091 | -34.73% | [-0.336, 1.399] |

## Defensive

| Strategy | Study | Sharpe | CAGR | RankIC | Max drawdown | SR 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SPY | Original daily | 0.498 | 10.85% | N/A | -24.55% | [-0.367, 1.419] |
| QQQ | Original daily | 0.469 | 11.93% | N/A | -34.87% | [-0.408, 1.394] |
| QQQ with fixed trend exposure | Broad features | 0.547 | 11.56% | N/A | -26.13% | [-0.403, 1.493] |
| Boosted trees (daily, raw) | Broad features | 0.187 | 4.70% | 0.0087 | -35.03% | [-0.740, 1.128] |
| Ridge (daily, raw) | Broad features | 0.239 | 5.99% | 0.0036 | -26.69% | [-0.643, 1.160] |
| Shared neural net (daily, raw) | Broad features | -0.021 | -1.43% | 0.0079 | -41.27% | [-0.852, 0.828] |
| Boosted trees (daily, residual) | Broad features | -0.083 | -1.51% | 0.0038 | -34.19% | [-1.020, 0.834] |
| Ridge (daily, residual) | Broad features | -0.294 | -6.27% | -0.0023 | -36.84% | [-1.228, 0.659] |
| Shared neural net (daily, residual) | Broad features | -0.007 | 0.01% | -0.0054 | -30.74% | [-0.840, 0.848] |
| Equal blend of all 12 models | Broad features | -0.100 | -2.15% | 0.0014 | -33.51% | [-0.971, 0.770] |
| Boosted trees (intraday, raw) | Broad features | 0.241 | 6.03% | 0.0070 | -24.11% | [-0.614, 1.117] |
| Ridge (intraday, raw) | Broad features | 0.076 | 2.10% | 0.0064 | -24.93% | [-0.750, 0.919] |
| Shared neural net (intraday, raw) | Broad features | 0.031 | -0.18% | 0.0082 | -34.61% | [-0.793, 0.887] |
| Boosted trees (intraday, residual) | Broad features | 0.032 | 0.59% | 0.0054 | -32.83% | [-0.879, 0.956] |
| Ridge (intraday, residual) | Broad features | -0.031 | -0.42% | 0.0012 | -28.08% | [-0.921, 0.875] |
| Shared neural net (intraday, residual) | Broad features | 0.089 | 2.03% | 0.0014 | -31.68% | [-0.744, 0.933] |
| Original THINK + mixer | Broad features | 0.654 | 17.01% | 0.0074 | -29.04% | [-0.175, 1.560] |
| Daily raw MSE / RankIC stopping | Broad features | 0.609 | 15.78% | 0.0211 | -26.69% | [-0.217, 1.478] |
| Intraday raw MSE / RankIC stopping | Broad features | 0.545 | 14.30% | 0.0157 | -28.15% | [-0.334, 1.444] |
| Validation best single model | Broad features | -0.342 | -7.34% | -0.0114 | -38.33% | [-1.206, 0.520] |
| Validation top-three blend | Broad features | -0.020 | -0.30% | 0.0060 | -29.46% | [-0.881, 0.848] |
| Intraday-window correction / utility (gated) | Frozen-core corrections | 0.553 | 14.34% | 0.0077 | -28.30% | [-0.261, 1.435] |
| Intraday-window correction / ranking (gated) | Frozen-core corrections | 0.452 | 11.37% | 0.0036 | -28.30% | [-0.335, 1.291] |
| Paired core (H1 validation stopping) | Frozen-core corrections | 0.553 | 14.34% | 0.0077 | -28.30% | [-0.261, 1.435] |
| Validation-gated correction | Frozen-core corrections | 0.503 | 12.79% | 0.0045 | -28.30% | [-0.305, 1.365] |
| Rich-feature correction / utility (gated) | Frozen-core corrections | 0.503 | 12.79% | 0.0045 | -28.30% | [-0.305, 1.365] |
| Rich-feature correction / ranking (gated) | Frozen-core corrections | 0.549 | 14.09% | 0.0064 | -28.30% | [-0.264, 1.411] |
| Intraday-window correction / utility (ungated) | Frozen-core corrections | 0.547 | 14.05% | 0.0060 | -29.31% | [-0.266, 1.433] |
| Intraday-window correction / ranking (ungated) | Frozen-core corrections | 0.442 | 11.11% | 0.0040 | -28.30% | [-0.340, 1.282] |
| Rich-feature correction / utility (ungated) | Frozen-core corrections | 0.426 | 10.80% | 0.0060 | -28.50% | [-0.391, 1.285] |
| Rich-feature correction / ranking (ungated) | Frozen-core corrections | 0.549 | 14.09% | 0.0064 | -28.30% | [-0.264, 1.411] |
| THINK + mixer (25-seed ensemble) | Seed expansion | 0.415 | 10.41% | 0.0080 | -26.34% | [-0.423, 1.296] |
| THINK + mixer (20 new seeds) | Seed expansion | 0.505 | 12.51% | 0.0091 | -24.52% | [-0.356, 1.410] |

## Portfolio variants

| Strategy | Study | Sharpe | CAGR | RankIC | Max drawdown | SR 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SPY | Original daily | 0.498 | 10.85% | N/A | -24.55% | [-0.367, 1.419] |
| QQQ | Original daily | 0.469 | 11.93% | N/A | -34.87% | [-0.408, 1.394] |
| MLP / top 20 | Original daily | 0.012 | -1.79% | -0.0108 | -37.77% | [-0.766, 0.800] |
| MLP / top 5 | Original daily | -0.046 | -6.82% | -0.0108 | -40.29% | [-0.872, 0.763] |
| 126-session momentum / top 20 | Original daily | 0.191 | 4.73% | 0.0139 | -31.10% | [-0.694, 1.116] |
| 126-session momentum / top 5 | Original daily | 0.327 | 8.59% | 0.0139 | -40.94% | [-0.549, 1.277] |
| Five-session reversal / top 20 | Original daily | 0.240 | 5.88% | 0.0183 | -35.68% | [-0.546, 1.035] |
| Five-session reversal / top 5 | Original daily | 0.259 | 5.93% | 0.0183 | -43.43% | [-0.530, 1.046] |
| Ridge / top 20 | Original daily | 0.216 | 5.06% | -0.0009 | -34.42% | [-0.626, 1.059] |
| Ridge / top 5 | Original daily | -0.260 | -13.68% | -0.0009 | -56.46% | [-1.140, 0.627] |
| StockMixer / top 20 | Original daily | 0.347 | 9.20% | 0.0186 | -27.74% | [-0.455, 1.183] |
| StockMixer / top 5 | Original daily | 0.659 | 22.52% | 0.0186 | -36.13% | [-0.129, 1.477] |
| THINK / top 20 | Original daily | 0.354 | 9.46% | 0.0206 | -33.91% | [-0.486, 1.233] |
| THINK / top 5 | Original daily | 0.461 | 13.65% | 0.0206 | -34.07% | [-0.373, 1.315] |
| THINK + Huber / top 20 | Original daily | 0.445 | 12.29% | 0.0197 | -34.03% | [-0.407, 1.340] |
| THINK + Huber / top 5 | Original daily | 0.453 | 13.14% | 0.0197 | -38.54% | [-0.473, 1.404] |
| THINK + mixer (original five seeds) / top 20 | Original daily | 0.503 | 14.10% | 0.0074 | -32.52% | [-0.316, 1.363] |
| THINK + mixer (original five seeds) / top 5 | Original daily | 0.415 | 11.91% | 0.0074 | -46.21% | [-0.434, 1.338] |
| THINK + mixer + Huber / top 20 | Original daily | 0.350 | 9.02% | 0.0038 | -33.52% | [-0.478, 1.221] |
| THINK + mixer + Huber / top 5 | Original daily | 0.527 | 15.77% | 0.0038 | -44.27% | [-0.304, 1.378] |
| THINK + mixer + pseudo-Huber / top 20 | Original daily | 0.358 | 9.30% | 0.0045 | -36.48% | [-0.487, 1.236] |
| THINK + mixer + pseudo-Huber / top 5 | Original daily | 0.689 | 22.39% | 0.0045 | -43.59% | [-0.169, 1.575] |
| MLP / Daily padded / top 20 | Alpaca matched features | -0.075 | -4.43% | -0.0142 | -40.08% | [-0.871, 0.724] |
| MLP / Daily padded / top 5 | Alpaca matched features | 0.056 | -2.74% | -0.0142 | -36.64% | [-0.804, 0.908] |
| Ridge / Daily padded / top 20 | Alpaca matched features | 0.216 | 5.06% | -0.0009 | -34.42% | [-0.626, 1.059] |
| Ridge / Daily padded / top 5 | Alpaca matched features | -0.260 | -13.68% | -0.0009 | -56.46% | [-1.140, 0.627] |
| StockMixer / Daily padded / top 20 | Alpaca matched features | 0.403 | 10.99% | 0.0155 | -30.28% | [-0.358, 1.181] |
| StockMixer / Daily padded / top 5 | Alpaca matched features | 0.060 | -1.57% | 0.0155 | -41.01% | [-0.700, 0.798] |
| THINK / Daily padded / top 20 | Alpaca matched features | 0.150 | 3.41% | 0.0046 | -31.48% | [-0.609, 0.935] |
| THINK / Daily padded / top 5 | Alpaca matched features | 0.362 | 9.89% | 0.0046 | -37.48% | [-0.500, 1.274] |
| THINK + Huber / Daily padded / top 20 | Alpaca matched features | 0.284 | 7.20% | 0.0038 | -31.30% | [-0.551, 1.153] |
| THINK + Huber / Daily padded / top 5 | Alpaca matched features | -0.053 | -2.66% | 0.0038 | -47.54% | [-0.938, 0.835] |
| THINK + mixer / Daily padded / top 20 | Alpaca matched features | 0.341 | 8.99% | 0.0077 | -35.06% | [-0.458, 1.175] |
| THINK + mixer / Daily padded / top 5 | Alpaca matched features | 0.350 | 9.51% | 0.0077 | -38.75% | [-0.479, 1.184] |
| THINK + mixer + Huber / Daily padded / top 20 | Alpaca matched features | 0.314 | 7.98% | 0.0011 | -33.27% | [-0.522, 1.175] |
| THINK + mixer + Huber / Daily padded / top 5 | Alpaca matched features | 0.684 | 21.63% | 0.0011 | -29.70% | [-0.144, 1.565] |
| THINK + mixer + pseudo-Huber / Daily padded / top 20 | Alpaca matched features | 0.301 | 7.67% | 0.0019 | -32.66% | [-0.525, 1.157] |
| THINK + mixer + pseudo-Huber / Daily padded / top 5 | Alpaca matched features | 0.574 | 17.81% | 0.0019 | -33.06% | [-0.276, 1.445] |
| MLP / Alpaca intraday / top 20 | Alpaca matched features | 0.006 | -2.37% | -0.0016 | -45.84% | [-0.793, 0.827] |
| MLP / Alpaca intraday / top 5 | Alpaca matched features | -0.017 | -6.87% | -0.0016 | -44.33% | [-0.876, 0.840] |
| Ridge / Alpaca intraday / top 20 | Alpaca matched features | 0.028 | -0.51% | -0.0098 | -37.68% | [-0.817, 0.897] |
| Ridge / Alpaca intraday / top 5 | Alpaca matched features | 0.158 | 1.90% | -0.0098 | -32.93% | [-0.686, 1.016] |
| StockMixer / Alpaca intraday / top 20 | Alpaca matched features | 0.166 | 3.29% | 0.0104 | -36.09% | [-0.593, 0.948] |
| StockMixer / Alpaca intraday / top 5 | Alpaca matched features | 0.166 | 1.34% | 0.0104 | -41.49% | [-0.625, 0.944] |
| THINK / Alpaca intraday / top 20 | Alpaca matched features | 0.082 | 1.36% | -0.0052 | -33.15% | [-0.733, 0.916] |
| THINK / Alpaca intraday / top 5 | Alpaca matched features | -0.139 | -6.70% | -0.0052 | -45.97% | [-0.934, 0.679] |
| THINK + Huber / Alpaca intraday / top 20 | Alpaca matched features | 0.044 | 0.58% | -0.0008 | -41.10% | [-0.798, 0.892] |
| THINK + Huber / Alpaca intraday / top 5 | Alpaca matched features | -0.138 | -5.63% | -0.0008 | -46.79% | [-1.019, 0.768] |
| THINK + mixer / Alpaca intraday / top 20 | Alpaca matched features | 0.380 | 10.21% | 0.0061 | -32.80% | [-0.429, 1.216] |
| THINK + mixer / Alpaca intraday / top 5 | Alpaca matched features | 0.246 | 5.51% | 0.0061 | -46.30% | [-0.586, 1.098] |
| THINK + mixer + Huber / Alpaca intraday / top 20 | Alpaca matched features | 0.247 | 6.18% | 0.0027 | -33.71% | [-0.598, 1.125] |
| THINK + mixer + Huber / Alpaca intraday / top 5 | Alpaca matched features | 0.435 | 12.28% | 0.0027 | -37.45% | [-0.437, 1.340] |
| THINK + mixer + pseudo-Huber / Alpaca intraday / top 20 | Alpaca matched features | 0.268 | 6.78% | 0.0026 | -33.97% | [-0.567, 1.127] |
| THINK + mixer + pseudo-Huber / Alpaca intraday / top 5 | Alpaca matched features | 0.468 | 13.67% | 0.0026 | -40.04% | [-0.395, 1.360] |
| MLP / top 10 / 50% long + 50% cash | Original allocations | -0.295 | -3.78% | -0.0108 | -21.53% | [-1.103, 0.482] |
| MLP / top 10 / 50% long / 50% short | Original allocations | -0.904 | -11.66% | -0.0108 | -39.40% | [-1.739, -0.153] |
| MLP / top 10 / 75% long + 25% cash | Original allocations | -0.235 | -6.62% | -0.0108 | -32.85% | [-1.039, 0.544] |
| MLP / top 10 / 75% long / 25% short | Original allocations | -0.439 | -10.23% | -0.0108 | -37.27% | [-1.237, 0.325] |
| MLP / top 20 / 50% long + 50% cash | Original allocations | -0.089 | 0.25% | -0.0108 | -19.88% | [-0.870, 0.700] |
| MLP / top 20 / 50% long / 50% short | Original allocations | -0.746 | -7.11% | -0.0108 | -27.39% | [-1.502, -0.051] |
| MLP / top 20 / 75% long + 25% cash | Original allocations | -0.022 | -0.48% | -0.0108 | -29.08% | [-0.802, 0.766] |
| MLP / top 20 / 75% long / 25% short | Original allocations | -0.215 | -3.91% | -0.0108 | -26.01% | [-0.986, 0.539] |
| MLP / top 5 / 50% long + 50% cash | Original allocations | -0.124 | -1.58% | -0.0108 | -20.84% | [-0.962, 0.690] |
| MLP / top 5 / 50% long / 50% short | Original allocations | -0.744 | -12.54% | -0.0108 | -41.58% | [-1.637, 0.059] |
| MLP / top 5 / 75% long + 25% cash | Original allocations | -0.071 | -3.74% | -0.0108 | -30.99% | [-0.902, 0.738] |
| MLP / top 5 / 75% long / 25% short | Original allocations | -0.287 | -8.94% | -0.0108 | -32.17% | [-1.140, 0.523] |
| 126-session momentum / top 10 / 50% long + 50% cash | Original allocations | 0.036 | 2.45% | 0.0139 | -20.23% | [-0.848, 0.977] |
| 126-session momentum / top 10 / 50% long / 50% short | Original allocations | -0.053 | 0.47% | 0.0139 | -24.94% | [-0.860, 0.771] |
| 126-session momentum / top 10 / 75% long + 25% cash | Original allocations | 0.107 | 2.94% | 0.0139 | -29.66% | [-0.778, 1.046] |
| 126-session momentum / top 10 / 75% long / 25% short | Original allocations | 0.080 | 2.67% | 0.0139 | -27.54% | [-0.777, 0.975] |
| 126-session momentum / top 20 / 50% long + 50% cash | Original allocations | 0.068 | 3.08% | 0.0139 | -16.50% | [-0.817, 0.991] |
| 126-session momentum / top 20 / 50% long / 50% short | Original allocations | -0.402 | -3.31% | 0.0139 | -28.11% | [-1.243, 0.446] |
| 126-session momentum / top 20 / 75% long + 25% cash | Original allocations | 0.149 | 4.08% | 0.0139 | -24.03% | [-0.736, 1.073] |
| 126-session momentum / top 20 / 75% long / 25% short | Original allocations | -0.027 | 1.37% | 0.0139 | -23.22% | [-0.907, 0.883] |
| 126-session momentum / top 5 / 50% long + 50% cash | Original allocations | 0.231 | 5.62% | 0.0139 | -21.35% | [-0.642, 1.183] |
| 126-session momentum / top 5 / 50% long / 50% short | Original allocations | 0.185 | 4.68% | 0.0139 | -24.51% | [-0.663, 1.073] |
| 126-session momentum / top 5 / 75% long + 25% cash | Original allocations | 0.290 | 7.39% | 0.0139 | -31.52% | [-0.584, 1.243] |
| 126-session momentum / top 5 / 75% long / 25% short | Original allocations | 0.319 | 8.03% | 0.0139 | -29.55% | [-0.530, 1.220] |
| Five-session reversal / top 10 / 50% long + 50% cash | Original allocations | 0.154 | 4.23% | 0.0183 | -18.53% | [-0.648, 0.956] |
| Five-session reversal / top 10 / 50% long / 50% short | Original allocations | -0.338 | -3.15% | 0.0183 | -34.77% | [-1.179, 0.505] |
| Five-session reversal / top 10 / 75% long + 25% cash | Original allocations | 0.215 | 5.36% | 0.0183 | -27.49% | [-0.583, 1.018] |
| Five-session reversal / top 10 / 75% long / 25% short | Original allocations | 0.073 | 2.15% | 0.0183 | -23.99% | [-0.703, 0.843] |
| Five-session reversal / top 20 / 50% long + 50% cash | Original allocations | 0.134 | 3.92% | 0.0183 | -18.71% | [-0.655, 0.932] |
| Five-session reversal / top 20 / 50% long / 50% short | Original allocations | -0.303 | -1.32% | 0.0183 | -23.95% | [-1.112, 0.487] |
| Five-session reversal / top 20 / 75% long + 25% cash | Original allocations | 0.204 | 5.17% | 0.0183 | -27.52% | [-0.584, 1.002] |
| Five-session reversal / top 20 / 75% long / 25% short | Original allocations | 0.093 | 2.97% | 0.0183 | -20.96% | [-0.667, 0.844] |
| Five-session reversal / top 5 / 50% long + 50% cash | Original allocations | 0.168 | 4.46% | 0.0183 | -22.66% | [-0.624, 0.956] |
| Five-session reversal / top 5 / 50% long / 50% short | Original allocations | -0.267 | -3.39% | 0.0183 | -48.91% | [-1.202, 0.641] |
| Five-session reversal / top 5 / 75% long + 25% cash | Original allocations | 0.229 | 5.64% | 0.0183 | -33.45% | [-0.559, 1.017] |
| Five-session reversal / top 5 / 75% long / 25% short | Original allocations | 0.095 | 2.24% | 0.0183 | -41.44% | [-0.715, 0.871] |
| Ridge / top 10 / 50% long + 50% cash | Original allocations | -0.155 | -1.24% | -0.0009 | -20.66% | [-0.969, 0.651] |
| Ridge / top 10 / 50% long / 50% short | Original allocations | -0.778 | -8.07% | -0.0009 | -31.38% | [-1.720, 0.156] |
| Ridge / top 10 / 75% long + 25% cash | Original allocations | -0.094 | -2.87% | -0.0009 | -30.32% | [-0.904, 0.712] |
| Ridge / top 10 / 75% long / 25% short | Original allocations | -0.280 | -5.83% | -0.0009 | -32.64% | [-1.112, 0.567] |
| Ridge / top 20 / 50% long + 50% cash | Original allocations | 0.110 | 3.55% | -0.0009 | -18.04% | [-0.738, 0.959] |
| Ridge / top 20 / 50% long / 50% short | Original allocations | -0.694 | -4.79% | -0.0009 | -23.28% | [-1.698, 0.269] |
| Ridge / top 20 / 75% long + 25% cash | Original allocations | 0.178 | 4.54% | -0.0009 | -26.54% | [-0.666, 1.025] |
| Ridge / top 20 / 75% long / 25% short | Original allocations | -0.026 | 0.65% | -0.0009 | -22.91% | [-0.888, 0.844] |
| Ridge / top 5 / 50% long + 50% cash | Original allocations | -0.352 | -5.62% | -0.0009 | -30.03% | [-1.238, 0.533] |
| Ridge / top 5 / 50% long / 50% short | Original allocations | -1.123 | -15.73% | -0.0009 | -50.21% | [-2.095, -0.169] |
| Ridge / top 5 / 75% long + 25% cash | Original allocations | -0.293 | -9.45% | -0.0009 | -44.16% | [-1.178, 0.589] |
| Ridge / top 5 / 75% long / 25% short | Original allocations | -0.559 | -13.92% | -0.0009 | -49.96% | [-1.467, 0.351] |
| StockMixer / top 10 / 50% long + 50% cash | Original allocations | 0.223 | 5.38% | 0.0186 | -17.70% | [-0.593, 1.066] |
| StockMixer / top 10 / 50% long / 50% short | Original allocations | 0.054 | 2.92% | 0.0186 | -14.99% | [-0.870, 0.950] |
| StockMixer / top 10 / 75% long + 25% cash | Original allocations | 0.293 | 7.39% | 0.0186 | -26.04% | [-0.521, 1.140] |
| StockMixer / top 10 / 75% long / 25% short | Original allocations | 0.269 | 6.59% | 0.0186 | -23.94% | [-0.555, 1.128] |
| StockMixer / top 20 / 50% long + 50% cash | Original allocations | 0.231 | 5.37% | 0.0186 | -14.14% | [-0.575, 1.069] |
| StockMixer / top 20 / 50% long / 50% short | Original allocations | -0.193 | 0.70% | 0.0186 | -15.14% | [-1.022, 0.636] |
| StockMixer / top 20 / 75% long + 25% cash | Original allocations | 0.307 | 7.51% | 0.0186 | -21.07% | [-0.496, 1.146] |
| StockMixer / top 20 / 75% long / 25% short | Original allocations | 0.224 | 5.51% | 0.0186 | -19.84% | [-0.594, 1.061] |
| StockMixer / top 5 / 50% long + 50% cash | Original allocations | 0.583 | 12.64% | 0.0186 | -19.29% | [-0.205, 1.398] |
| StockMixer / top 5 / 50% long / 50% short | Original allocations | 0.704 | 13.52% | 0.0186 | -18.85% | [-0.126, 1.506] |
| StockMixer / top 5 / 75% long + 25% cash | Original allocations | 0.645 | 18.37% | 0.0186 | -28.03% | [-0.142, 1.460] |
| StockMixer / top 5 / 75% long / 25% short | Original allocations | 0.738 | 19.52% | 0.0186 | -22.50% | [-0.051, 1.538] |
| THINK / top 10 / 50% long + 50% cash | Original allocations | 0.338 | 7.20% | 0.0206 | -14.46% | [-0.469, 1.176] |
| THINK / top 10 / 50% long / 50% short | Original allocations | 0.616 | 9.05% | 0.0206 | -20.47% | [-0.356, 1.585] |
| THINK / top 10 / 75% long + 25% cash | Original allocations | 0.405 | 10.10% | 0.0206 | -21.74% | [-0.400, 1.242] |
| THINK / top 10 / 75% long / 25% short | Original allocations | 0.521 | 11.54% | 0.0206 | -20.48% | [-0.322, 1.379] |
| THINK / top 20 / 50% long + 50% cash | Original allocations | 0.250 | 5.67% | 0.0206 | -17.56% | [-0.594, 1.131] |
| THINK / top 20 / 50% long / 50% short | Original allocations | 0.092 | 3.43% | 0.0206 | -19.17% | [-0.963, 1.132] |
| THINK / top 20 / 75% long + 25% cash | Original allocations | 0.320 | 7.84% | 0.0206 | -25.94% | [-0.521, 1.200] |
| THINK / top 20 / 75% long / 25% short | Original allocations | 0.318 | 7.17% | 0.0206 | -21.31% | [-0.589, 1.236] |
| THINK / top 5 / 50% long + 50% cash | Original allocations | 0.378 | 8.20% | 0.0206 | -17.99% | [-0.463, 1.234] |
| THINK / top 5 / 50% long / 50% short | Original allocations | 0.254 | 5.64% | 0.0206 | -26.80% | [-0.673, 1.169] |
| THINK / top 5 / 75% long + 25% cash | Original allocations | 0.438 | 11.43% | 0.0206 | -26.25% | [-0.401, 1.293] |
| THINK / top 5 / 75% long / 25% short | Original allocations | 0.425 | 10.44% | 0.0206 | -25.40% | [-0.440, 1.309] |
| THINK + Huber / top 10 / 50% long + 50% cash | Original allocations | 0.379 | 7.84% | 0.0197 | -19.30% | [-0.479, 1.269] |
| THINK + Huber / top 10 / 50% long / 50% short | Original allocations | 0.325 | 5.81% | 0.0197 | -24.22% | [-0.719, 1.393] |
| THINK + Huber / top 10 / 75% long + 25% cash | Original allocations | 0.445 | 11.08% | 0.0197 | -28.29% | [-0.410, 1.337] |
| THINK + Huber / top 10 / 75% long / 25% short | Original allocations | 0.483 | 10.59% | 0.0197 | -22.93% | [-0.414, 1.419] |
| THINK + Huber / top 20 / 50% long + 50% cash | Original allocations | 0.337 | 6.97% | 0.0197 | -17.67% | [-0.518, 1.230] |
| THINK + Huber / top 20 / 50% long / 50% short | Original allocations | 0.116 | 3.62% | 0.0197 | -20.63% | [-0.967, 1.192] |
| THINK + Huber / top 20 / 75% long + 25% cash | Original allocations | 0.409 | 9.87% | 0.0197 | -26.03% | [-0.444, 1.302] |
| THINK + Huber / top 20 / 75% long / 25% short | Original allocations | 0.405 | 8.65% | 0.0197 | -22.07% | [-0.487, 1.349] |
| THINK + Huber / top 5 / 50% long + 50% cash | Original allocations | 0.367 | 7.83% | 0.0197 | -20.41% | [-0.564, 1.328] |
| THINK + Huber / top 5 / 50% long / 50% short | Original allocations | 0.139 | 3.99% | 0.0197 | -24.31% | [-0.862, 1.126] |
| THINK + Huber / top 5 / 75% long + 25% cash | Original allocations | 0.427 | 10.89% | 0.0197 | -29.74% | [-0.499, 1.383] |
| THINK + Huber / top 5 / 75% long / 25% short | Original allocations | 0.395 | 9.38% | 0.0197 | -24.90% | [-0.553, 1.355] |
| THINK + mixer (original five seeds) / top 10 / 50% long + 50% cash | Original allocations | 0.529 | 10.60% | 0.0074 | -20.91% | [-0.286, 1.405] |
| THINK + mixer (original five seeds) / top 10 / 50% long / 50% short | Original allocations | -0.250 | -0.25% | 0.0074 | -18.72% | [-1.083, 0.579] |
| THINK + mixer (original five seeds) / top 10 / 75% long + 25% cash | Original allocations | 0.595 | 15.39% | 0.0074 | -30.42% | [-0.218, 1.472] |
| THINK + mixer (original five seeds) / top 10 / 75% long / 25% short | Original allocations | 0.442 | 10.18% | 0.0074 | -27.06% | [-0.345, 1.288] |
| THINK + mixer (original five seeds) / top 20 / 50% long + 50% cash | Original allocations | 0.391 | 7.75% | 0.0074 | -16.78% | [-0.431, 1.249] |
| THINK + mixer (original five seeds) / top 20 / 50% long / 50% short | Original allocations | 0.045 | 3.07% | 0.0074 | -11.46% | [-0.870, 0.928] |
| THINK + mixer (original five seeds) / top 20 / 75% long + 25% cash | Original allocations | 0.465 | 11.14% | 0.0074 | -24.80% | [-0.357, 1.327] |
| THINK + mixer (original five seeds) / top 20 / 75% long / 25% short | Original allocations | 0.444 | 9.24% | 0.0074 | -20.78% | [-0.375, 1.288] |
| THINK + mixer (original five seeds) / top 5 / 50% long + 50% cash | Original allocations | 0.320 | 7.15% | 0.0074 | -25.37% | [-0.535, 1.250] |
| THINK + mixer (original five seeds) / top 5 / 50% long / 50% short | Original allocations | -0.333 | -2.22% | 0.0074 | -26.12% | [-1.162, 0.480] |
| THINK + mixer (original five seeds) / top 5 / 75% long + 25% cash | Original allocations | 0.382 | 9.85% | 0.0074 | -36.44% | [-0.471, 1.311] |
| THINK + mixer (original five seeds) / top 5 / 75% long / 25% short | Original allocations | 0.221 | 5.53% | 0.0074 | -32.23% | [-0.600, 1.093] |
| THINK + mixer + Huber / top 10 / 50% long + 50% cash | Original allocations | 0.139 | 3.99% | 0.0038 | -22.46% | [-0.712, 1.016] |
| THINK + mixer + Huber / top 10 / 50% long / 50% short | Original allocations | -1.015 | -7.27% | 0.0038 | -26.98% | [-1.831, -0.210] |
| THINK + mixer + Huber / top 10 / 75% long + 25% cash | Original allocations | 0.210 | 5.30% | 0.0038 | -32.45% | [-0.640, 1.090] |
| THINK + mixer + Huber / top 10 / 75% long / 25% short | Original allocations | -0.082 | -0.05% | 0.0038 | -28.96% | [-0.899, 0.749] |
| THINK + mixer + Huber / top 20 / 50% long + 50% cash | Original allocations | 0.231 | 5.22% | 0.0038 | -17.38% | [-0.597, 1.094] |
| THINK + mixer + Huber / top 20 / 50% long / 50% short | Original allocations | -0.656 | -1.71% | 0.0038 | -14.30% | [-1.462, 0.152] |
| THINK + mixer + Huber / top 20 / 75% long + 25% cash | Original allocations | 0.312 | 7.37% | 0.0038 | -25.52% | [-0.514, 1.181] |
| THINK + mixer + Huber / top 20 / 75% long / 25% short | Original allocations | 0.153 | 4.21% | 0.0038 | -21.15% | [-0.663, 1.000] |
| THINK + mixer + Huber / top 5 / 50% long + 50% cash | Original allocations | 0.423 | 8.78% | 0.0038 | -24.33% | [-0.407, 1.274] |
| THINK + mixer + Huber / top 5 / 50% long / 50% short | Original allocations | -0.332 | -2.26% | 0.0038 | -27.36% | [-1.207, 0.540] |
| THINK + mixer + Huber / top 5 / 75% long + 25% cash | Original allocations | 0.490 | 12.52% | 0.0038 | -34.91% | [-0.340, 1.342] |
| THINK + mixer + Huber / top 5 / 75% long / 25% short | Original allocations | 0.294 | 7.17% | 0.0038 | -34.22% | [-0.544, 1.126] |
| THINK + mixer + pseudo-Huber / top 10 / 50% long + 50% cash | Original allocations | 0.306 | 6.60% | 0.0045 | -21.86% | [-0.542, 1.179] |
| THINK + mixer + pseudo-Huber / top 10 / 50% long / 50% short | Original allocations | -0.586 | -3.24% | 0.0045 | -16.62% | [-1.440, 0.241] |
| THINK + mixer + pseudo-Huber / top 10 / 75% long + 25% cash | Original allocations | 0.377 | 9.28% | 0.0045 | -31.64% | [-0.470, 1.254] |
| THINK + mixer + pseudo-Huber / top 10 / 75% long / 25% short | Original allocations | 0.178 | 4.65% | 0.0045 | -27.80% | [-0.658, 1.010] |
| THINK + mixer + pseudo-Huber / top 20 / 50% long + 50% cash | Original allocations | 0.242 | 5.39% | 0.0045 | -19.23% | [-0.606, 1.112] |
| THINK + mixer + pseudo-Huber / top 20 / 50% long / 50% short | Original allocations | -0.405 | -0.07% | 0.0045 | -10.22% | [-1.219, 0.415] |
| THINK + mixer + pseudo-Huber / top 20 / 75% long + 25% cash | Original allocations | 0.322 | 7.61% | 0.0045 | -28.02% | [-0.525, 1.196] |
| THINK + mixer + pseudo-Huber / top 20 / 75% long / 25% short | Original allocations | 0.218 | 5.23% | 0.0045 | -23.62% | [-0.613, 1.071] |
| THINK + mixer + pseudo-Huber / top 5 / 50% long + 50% cash | Original allocations | 0.595 | 12.06% | 0.0045 | -23.50% | [-0.263, 1.475] |
| THINK + mixer + pseudo-Huber / top 5 / 50% long / 50% short | Original allocations | -0.078 | 0.99% | 0.0045 | -26.42% | [-0.959, 0.802] |
| THINK + mixer + pseudo-Huber / top 5 / 75% long + 25% cash | Original allocations | 0.659 | 17.57% | 0.0045 | -33.91% | [-0.199, 1.540] |
| THINK + mixer + pseudo-Huber / top 5 / 75% long / 25% short | Original allocations | 0.505 | 12.19% | 0.0045 | -32.36% | [-0.362, 1.373] |

## Definitions and limitations

- **Period:** January 3, 2022–December 31, 2025. All strategies use the same 1,003 daily observations.
- **Financial metrics:** Net returns at each study’s base modeled trading costs. Ensemble forecasts; no seed selected by test performance.
- **Sharpe ratio:** Annualized daily SR = sqrt(252) × [mean(daily return) − ((1.03)^(1/252)−1)] / sample SD(daily return).
- **95% Sharpe interval:** Pointwise 2.5th/97.5th percentiles from 10,000 circular moving-block resamples. Primary block: 20 sessions.
- **Dependence:** The same date-block draws apply to every strategy. Stocks, bars and seeds are not treated as independent observations.
- **Sharpe annualization:** Conventional annualized daily ratio, matching existing reports. It is not a separately autocorrelation-adjusted annual-holding-period Sharpe.
- **RankIC:** Mean per-origin cross-sectional Spearman correlation of ensemble scores with realized five-session STOCK returns, among observed eligible labels.
- **RankIC comparability:** Raw-return RankIC throughout, including models trained on QQQ residuals. Approximately 200 complete forecast origins.
- **RankIC for overlays:** RankIC describes the underlying stock forecast. Changing exposure or portfolio size does not change that forecast’s RankIC.
- **N/A:** SPY, QQQ, the monthly equal-weight basket and single-ETF trend timing do not provide stock rankings. Their RankIC is not applicable.
- **Defensive rule:** Trend-half policies target full stock exposure when signal-time QQQ is at/above MA200, half exposure plus cash otherwise.
- **Portfolio variants:** Top-five/top-twenty holdings and original cash/short allocations. Short accounts include 3% annual borrow, with borrow availability assumed.
- **Excluded repetitions:** Reference aliases, individual seeds, fee stresses and seed-bootstrap accounts are omitted. Distinct declared selection policies remain, even when their realized paths coincide.
- **Historical scope:** Active historical monthly Nasdaq-100 only. Retired 2026-selected survivor universes and deferred NYSE studies are excluded.
- **Interpretation:** Previously inspected historical development. Intervals condition on fitted paths and omit full research-search, refit, data-revision and future-regime uncertainty.
- **Comparison warning:** A strategy’s SR interval is not an interval for its advantage over SPY/QQQ. Overlap or zero inclusion alone is not a paired superiority test.
- **Sensitivities:** Five- and 60-session block intervals are included in the accompanying sensitivity CSV. No block chosen for favorable bounds.
- **Source studies:** Audited original monthly/top-k/long-short, matched Alpaca, ranking, broad-feature, frozen-core correction and seed-expansion studies.
- **Block-bootstrap reference:** Politis and Romano, A Circular Block-Resampling Procedure for Stationary Data: https://statistics.stanford.edu/technical-reports/circular-block-resampling-procedure-stationary-data
- **Sharpe reference:** Andrew W. Lo, The Statistics of Sharpe Ratios (2002): https://alo.mit.edu/publications/page/18/
