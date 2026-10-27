# First mechanism tests: results — October 27, 2026

**The plan was finalized and testing has started.** The first available-data batch is complete: 12 workbook daily paths validated, 18 monthly model accounts reconstructed, six daily policy contrasts and four monthly model contrasts evaluated at their frozen block lengths, and 31 focused tests passed. This report separates what those tests answer from the trained-signal questions still requiring Rishi's original files.

**Main result:** volatility targeting explains most of the latest historical Sharpe improvement among the tested policy increments. The original rank blend also outperforms a simple combination of standalone account paths. Its stock-level mechanism remains unidentified. These are selected-history diagnostics, not untouched confirmation.

## 1. Availability and provenance

The first-batch protocol was committed as `06791ad` before computing these contrasts. No alternative blend weights, new market-period scoring, financial model training or new strategy search was run.

The checked `origin/feat-rs` tree contains no tracked `runs/` payload, `.pt` checkpoints or per-seed prediction NPZ exports. `.gitignore` excludes `data/` and `runs/`; `latest/results/README.md` explicitly describes the committed results as compact summaries and says the large curves, ledgers and checkpoints remain in ignored run locations. The current, Rishi, Rishi-latest and unification checkouts lack the newest required raw inputs/runs. A bounded Downloads filename search found no relevant payload. This does not prove Rishi lacks the files on his machine or elsewhere.

The workbook supplies **literal daily returns** for 12 leader/benchmark paths. These extend the compact CSV evidence, including the original-five 1.199 leader. They do not supply per-stock predictions, shares, fills or checkpoint weights. The original workbook was read without modification; its SHA256 remains `044c8b2e26a769f16722392eef3c9d87e28f8f43aafb41d0d3b182a9b309acd9`.

Validation: 1,003 matching exchange sessions per workbook path; eight nonbenchmark paths overlap the prior CSV and match; ten nonbenchmark accounts match published SR/CAGR/vol/maxDD within **5.25e-14**. Eighteen monthly parent/dynamic/blend accounts cover 48 complete months and match source CAGR within **1.28e-15**, using the source's 1,003-session annualization rather than assuming exactly four 252-session years.

## 2. Test 1: does volatility targeting explain the latest improvement?

These comparisons retain the original top-five ranking/selection rule. The actual accounts still differ in exposure, trade quantities, fees and cash drift; that is the policy effect being measured.

| Seed ensemble / account | SR | CAGR | Annual volatility | Max drawdown |
|---|---:|---:|---:|---:|
| Original five, full exposure | 0.919 | 32.94% | 34.04% | -34.89% |
| Original five, 20% basket-vol target | **1.199** | 28.96% | 20.49% | **-17.78%** |
| Ten seeds, full exposure | 0.896 | 31.87% | 34.05% | -40.94% |
| Ten seeds, 20% basket-vol target | **1.080** | 26.00% | 20.65% | **-21.69%** |

**Interpretation:** the >1 result can arise with the same trained forecasts and stock-ranking rule. Realized risk falls substantially, while CAGR also falls. There is no need to invoke an improvement in the neural predictor to explain this step. Risk reduction does not by itself show an implementable timing advantage over every simpler risk-matched control; those share-account controls still require original inputs.

The fixed 3% annual hurdle and zero-interest idle cash are inherited from the source. This is not a pure zero-risk-free-rate Sharpe scaling identity. Existing source summaries also report lower ten-seed turnover, **30.38 → 22.19** under their annual turnover definition, and fee dollars **$128,159 → $96,118**. Fees/turnover are **source-table values**, not independently reconstructed from trades here; lower exposure and NAV also affect dollar comparisons.

Paired stationary-bootstrap SR differences, expected block 21 sessions / 10,000 draws:

- Original-five volatility control minus full exposure: **+0.280**, pointwise 95% interval **[-0.018, +0.559]**.
- Ten-seed volatility control minus full exposure: **+0.184**, interval **[-0.139, +0.487]**.

Both include zero. Selection across historical recipes is not corrected by these conditional intervals. The effect is descriptively substantial but not statistically established future superiority.

## 3. Test 2: issuer selection and trend control

Starting from the ten-seed original-selection 20% target:

| Policy | SR | CAGR | Max drawdown |
|---|---:|---:|---:|
| Original selections + 20% target | 1.080 | 26.00% | -21.69% |
| Five issuers + 20% target | 1.112 | 26.81% | -21.69% |
| Original selections + 20% target + trend cap | 1.081 | 25.50% | -20.84% |

- **Issuer increment:** +0.032 SR, interval **[-0.091, +0.163]**; +0.81 percentage points CAGR. It is smaller than the volatility-control increment. It also changes estimated basket risk/exposure, so it is not a pure fixed-exposure stock-picking effect.
- **Trend increment:** +0.002 SR, interval **[-0.079, +0.063]**; CAGR declines by 0.50 percentage points. This comparison gives little descriptive support for adding trend control to this already volatility-managed ten-seed account.
- In the original-five group, issuer selection reduces SR by 0.021 and the trend cap reduces SR by 0.010 relative to the no-issuer 20% rule. Neither increment has established superiority across groups.

The six predeclared daily **annual net arithmetic mean** contrasts all have Holm-adjusted one-sided p=1.0. This does not mean risk management failed: the volatility target deliberately gives up expected return to reduce risk. Those p values test positive mean-return increments, not drawdown reduction or positive SR increments. SR intervals above are pointwise and selected-history conditional. See `DAILY_CONTRASTS.csv` for all 5/21/63-session sensitivities.

## 4. Test 3: where does the risk tradeoff appear?

Exact calendar returns for the ten-seed original-selection book:

| Year | Full exposure | 20% target | Target minus full |
|---|---:|---:|---:|
| 2022 | -26.07% | -13.59% | +12.48 pp |
| 2023 | +79.86% | +65.07% | -14.79 pp |
| 2024 | +43.22% | +29.87% | -13.35 pp |
| 2025 | +57.92% | +35.46% | -22.47 pp |

This looks like protection in the inspected bear year purchased with less upside in the subsequent rally years. It is consistent with exposure reduction and risk forecasting; it does not establish directional market forecasting. There is only one full bear-year episode here. These are calendar returns, not annualized partial-year metrics.

## 5. Test 4: is rank blending just ordinary portfolio diversification?

The two operations differ mathematically. Rank blending selects a new basket. A capital split holds two existing baskets. I compared the actual blend's monthly wealth with a fixed **initial-capital** 50:50 combination of published standalone NAV paths, with no periodic rebalancing and no weight search.

| Ten-seed account | Five holdings CAGR | Ten holdings CAGR |
|---|---:|---:|
| Parent standalone | 16.73% | 15.36% |
| Dynamic standalone | 20.53% | 15.32% |
| Algebraic standalone capital pool | 18.68% | 15.34% |
| Actual rank blend | **31.87%** | **23.92%** |

The historical difference is too large to describe as merely averaging the two standalone return series. This supports investigating changed upper-tail selections, persistent preferences and blend boundary crossings.

**Important limitation:** the pool holds the standalone full-size account returns/costs fixed. It does not reconstruct half-AUM fees, participation limits, cash scaling or combined orders. It is a non-executable algebraic comparator, not the exact account-level experiment required for causal attribution. It cannot identify which stocks or information produce the difference.

The blend also exceeds that algebraic pool in both original/new seed halves at k5/k10, with differing margins. These share the same market history, so this is initialization sensitivity rather than independent market replication.

## 6. Test 5: uncertainty in the model-blend increments

Separate monthly diagnostics use annualized arithmetic **monthly** differences, 48 observations, stationary expected block 3 months, 10,000 draws. These endpoints are not the roadmap's daily arithmetic primary statistic; no daily Sharpe is inferred from monthly data.

| Ten-seed contrast | Annualized monthly arithmetic difference | Pointwise 95% interval | Holm p, four contrasts |
|---|---:|---:|---:|
| Top5 blend − parent | +12.26 pp | [-1.15, +26.53] pp | 0.134 |
| Top5 blend − dynamic | +9.91 pp | [-4.95, +26.80] pp | 0.158 |
| Top10 blend − parent | +6.86 pp | [+0.27, +13.44] pp | 0.083 |
| Top10 blend − dynamic | +7.24 pp | [-1.94, +17.23] pp | 0.158 |

The top10 parent contrast has a positive pointwise interval, but does not pass the declared family-level test. None of the four establishes a family-adjusted positive increment at 5%. This does not erase the historical candidate; it limits the strength of the claim. Full earlier strategy search and refitting uncertainty remain outside these intervals. Fixed block-1/6-month sensitivities are retained in `MONTHLY_MODEL_CONTRASTS.csv`.

## 7. Test 6: are identity/masking interventions well defined?

Three new structural tests on the actual parent/dynamic code passed:

1. Jointly reindexing parent features, eligibility, graph node indices and mixer input/output parameters preserves predictions after undoing the permutation. Swapping only features does not. This distinguishes a valid ID bookkeeping audit from an out-of-distribution identity perturbation.
2. Jointly transporting dynamic features, eligibility, wavelet matrices and stock-indexed filters preserves predictions. This shows the adapter can be consistently reindexed; it does not establish that trained filters are useful.
3. Inactive dynamic inputs receive exactly zero gradient from eligible outputs in the tested synthetic case, while active input gradients remain nonzero.

Combined with existing tests for cutoff chronology, normalization, NaNs, settlement, costs, cash and next-close risk execution, the final focused suite passes **31 tests**. These are structural/accounting tests, not proof that either trained architecture learned peer alpha. Synthetic full-learner recovery and matched financial ablations remain conditional stages, not silently declared complete.

## 8. Independent review and preservation

An independent reviewer checked the analysis code, protocol, tests and output arithmetic. It confirmed source input hashes, calendars, stationary paired resampling, Holm families, source-session CAGR and the initial-capital pool calculation. Independent arithmetic differences were below 2.0e-15 for pooling. It found one P2 issue: reruns could overwrite completed output. A failing regression test was added, then the runner was changed to refuse any nonempty destination; the combined suite passed afterward. Existing results are preserved.

The runner therefore refuses to recompute into this completed folder. A separately declared reproduction must select a fresh destination; do not delete or overwrite this evidence to make the default command run again. The reviewer did not independently rerun financial calculations from raw inputs, which remain unavailable.

## 9. Decision and next work

**GO:** pursue saved-forecast blend attribution and training-only identity/simple-signal controls once the original artifacts arrive. The large historical gap versus the algebraic capital pool makes the actual selected names and trade boundaries a high-value question. Keep risk controls as a separate mechanism.

**DEFER:** conclusions about learned peer information, identity-explained P&L, feature importance in the trained networks, exact half-capital execution, or financial component ablations. Those need per-stock forecasts, original checkpoints/preprocessing and reconciled holdings/trades. No new architecture grid is justified by the aggregate tests alone.

The user requested a message to obtain the existing artifacts; a paste-ready request is in `../ARTIFACT_REQUEST.md`. No message was sent. No new financial training, protected-window scoring, capital, push or automation occurred.

Machine-readable evidence: `BATCH_RESULTS.json`, `ARTIFACT_INVENTORY.json`, `WORKBOOK_VALIDATION.csv`, `WORKBOOK_METRICS.csv`, `WORKBOOK_ANNUAL.csv`, `DAILY_CONTRASTS.csv`, `MONTHLY_MODEL_CONTRASTS.csv`, `CAPITAL_POOL_DIAGNOSTIC.csv` and `TEST_LOG_FINAL.txt`.
