# Historical-label preference by dated SIC, 2024–2025

Protocol ../SEC_PREFERENCE_PROTOCOL.md was committed as ed65d58, and implementation as 34c7a8b, before this calculation. At each fold's first signal, we joined every eligible ID and its saved training-label Hlabel percentile rank to the separately audited SEC cover/header receipts. Unmatched names have no imputed industry. This is a description of the cheap historical-return control, not the neural network's learned predictions or an account backtest.

| Fold | Covered / eligible | Between-industry share of covered rank variation | Mean rank: covered | Mean rank: uncovered | Covered / uncovered with fewer than 30 training labels |
|---|---:|---:|---:|---:|---:|
| 2024 | 81 / 101 | 13.37% | 0.534 | 0.388 | 5 / 6 |
| 2025 | 79 / 101 | 13.84% | 0.518 | 0.456 | 5 / 8 |

For context, the earlier first-signal 2022 and 2023 analyses found 18.51% and 19.09% among 77 and 80 covered names, respectively; their dated SICs used different historical filings and a retrospective mapping rather than the later SEC cover checks. None of these four snapshots is a complete-universe estimate. Their variation fractions are descriptive squared-deviation partitions, not proportions of portfolio returns or a causal industry effect. Different covered-name rank means, especially in 2024, show why missing names should not be assumed representative. Small and singleton industries make individual group averages unstable; all groups appear in INDUSTRY_PREFERENCE.csv.

The prior source SIC match proves only what was recorded at filing, and the later SEC cover match additionally confirms the tagged symbol at that date. Neither verifies no subsequent corporate or ticker change, NDX price-series entity identity, continually current SIC or as-published membership. SIC/FF12 is not GICS. The Hlabel control substitutes the training cross-sectional median for IDs with fewer than 30 training labels; the last column quantifies those cases. Coverage and decompositions were independently recalculated from all 202 joined rows. Five focused join/date/decomposition tests pass. See JOINED.csv, SUMMARY.csv, COVERAGE_PREFERENCE.csv, INDUSTRY_PREFERENCE.csv and PROVENANCE.json. No financial account, protected-window score, model training or checkpoint gate changed.
