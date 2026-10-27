# NASDAQ-100 research status — October 20, 2026

Continued training has produced a fixed 25-seed model with **12.9% net historical CAGR**, compared with **10.9% for SPY**, **11.9% for QQQ** and **19.6% for the original five-seed reference**. The requested reliable superiority has **not been established**. Previously inspected 2022–2025 results are development evidence, and none can become unseen confirmation by changing the split labels or training more candidates.

| Model / benchmark, full exposure | Net CAGR | Sharpe | Maximum drawdown |
| --- | --- | --- | --- |
| original_five | 19.6% | 0.629 | -39.4% |
| new_twenty | 14.4% | 0.503 | -34.7% |
| all_twenty_five | 12.9% | 0.452 | -36.8% |
| SPY | 10.9% | 0.498 | -24.6% |
| QQQ | 11.9% | 0.469 | -34.9% |
| H1-stopped paired core | 16.7% | 0.548 | -39.4% |
| Validation-gated correction | 15.1% | 0.510 | -39.4% |

The fixed 25-seed primary’s annual arithmetic active mean versus SPY is 4.79 percentage points, with paired 95% interval [-8.65, 18.21] pp and broad-family simultaneous diagnostic [-18.39, 27.96] pp. These condition on fitted paths and do not account for the full prior research search. 12/25 individual seeds beat SPY on CAGR; individual seeds share one market history.

## What the new work established

- Added intraday information, ranking objectives, richer daily features, boosted trees, shared networks and bounded frozen-core adapters were tested under declared protocols. Every unsuccessful model remains visible. Full accounts use next-close execution, continuous cash, modeled fees, liquidity caps and reviewed historical membership/issuer identities.
- The correction batch trained 20 cores and 80 adapters. Chronological validation accepted the primary correction for one annual fold. The resulting strategy returned 15.1% versus 16.7% for the paired core; validation acceptance did not guarantee financial improvement.
- The seed-count batch trained 80 additional models from 20 fixed new seeds and combined all 25 equally in rank space. No seed was selected or removed using test return. It also replayed 128 initialization resamples under two fixed exposure policies, separating initialization sensitivity from independent market history.
- Averaging forecast ranks changes which stocks enter the top ten. The resulting return is not a weighted average of the component ensembles’ portfolio returns; adding seeds can therefore lower CAGR below both component blends.
- Every completed batch has frozen source/data hashes, saved fits and predictions, independent checkpoint reproduction, portfolio replay, cost stresses and paired confidence diagnostics. The seed expansion compares against every eligible ensemble path from all preceding active studies, including negative cells and alternative exposure policies.

## Strengths and remaining gaps

The project now has unusually traceable experiments: point-in-time monthly membership, issuer-reuse quarantine, chronological label purges, train-only transforms/graphs, fixed seed lists and auditable continuous accounts. Matched feature-capacity controls distinguish changing information from changing parameter count. Benchmark-relative losses, missing-label handling, seed sensitivity and execution costs are tested explicitly.

The main limits are effective history and selection uncertainty. Four test years cover one market path, and those years have been inspected across many research rounds. Early folds contain few five-session training origins. Dense stock-ID mixing has changing-universe and cold-start constraints. Better forecast MSE or validation RankIC need not improve top-ten net return. Abrupt selection boundaries, turnover, missing outcomes, vendor revisions and acquisition/total-return approximations remain material. More intraday bars do not turn correlated stocks and sessions into independent years.

## Confidence intervals and the next evidence

Intervals should narrow through lower paired active-return variability, stable training and genuinely additional independent observations. They should not narrow by treating each stock/bar/seed as independent, shortening bootstrap blocks after viewing results, discarding poor seeds or presenting a post-search winner as prespecified. A fixed blend with SPY scales both the active mean and its uncertainty: if strategy return minus SPY is multiplied by w, both its mean and interval endpoints scale by w. That can reduce tracking risk and absolute interval width without improving the evidence that active return is positive.

The next confirmation step needs a fixed candidate/refit/execution policy and reviewed point-in-time membership, issuer support, actions and features on new dates. The separately reserved period begins no earlier than October 21, 2026, with the first formal read after 252 sessions. Those observations are not available in this study; one year can still be underpowered. The larger research search and model-training uncertainty must remain disclosed. No prospective data result, order or recurring job is claimed here.

## Data actually used

- Raw input: 10,839,262 Alpaca SIP 15-minute bars, 2018–2025, in `data/nasdaq_monthly_pit/alpaca_15min/sip_raw_20180101_20260101/`.
- Reviewed regular-session summaries: eight channels per security/day, in `data/nasdaq_monthly_pit/alpaca_features_20261019/daily_features.npz`.
- Economic daily prices, historical monthly membership, tradability, liquidity and ETF labels: `data/nasdaq_monthly_pit/assembled/`.
- Models use daily windows and five-session labels/holding periods. The strongest original and seed-expanded cores use five daily features; other matched models and correction cells test the intraday summaries. This is not a claim of minute-by-minute or 15-minute trading. The [data guide](ALPACA_15MIN.md) gives the exact locations, coverage and limitations.

## Complete reports

| Study | Candidate fits | Disclosed accounts |
| --- | --- | --- |
| [Alpaca matched features](nasdaq_alpaca_20261019/REPORT.md) | 288 | 241 |
| [Ranking / benchmark-residual targets](ndx_ranking_20261020/REPORT.md) | 128 | 69 |
| [Rich features / trees / shared networks](ndx_broad_20261020/REPORT.md) | 208 | 216 |
| [Frozen-core corrections](ndx_corrections_20261020/REPORT.md) | 100 | 144 |
| [Fixed seed-count expansion](ndx_seed_expansion_20261020/REPORT.md) | 80 | 330 |

Counts describe declared model/penalty fits, not independent market observations. Accounts include ensembles, seeds, controls, fee/exposure sensitivities and initialization resamples; repeated paths are not independent trials. All 126 publication artifact hashes were reverified while producing this status page. Older active historical studies and the audit trail remain preserved; the retired 2026-selected universe stays retired.
