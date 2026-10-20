# NASDAQ-100 model and evidence audit

Audit date: October 19, 2026 (America/Chicago).

The project has a strong, auditable research foundation and a promising historical
THINK+mixer result. The available evidence does not yet establish that any model
regularly outperforms both SPY and QQQ. The main constraints are a short, repeatedly
examined market history, unstable concentrated selections, and a mismatch between
the forecast objective and the benchmark-relative investment objective.

This audit inspected the active historical monthly NASDAQ-100 pipeline, its frozen
protocols, model implementations, saved metrics, prior audit records, and relevant
tests. It also inspected the earlier research engines used as dependencies. It is
not a security review or an independent certification of every historical vendor
record. Monthly refers to universe formation: the active models predict and
rebalance every five sessions, with next-close execution.

## Evidence and scope

- Re-ran 36 targeted tests covering membership, timing, inactive-node masking,
  portfolio accounting, independent replay fixtures, pseudo-Huber mathematics,
  and signed portfolios. All passed.
- Rehashed all entries in the parent input/code lock, parent dependency lock, and
  pseudo-Huber extension input/code lock. All matched. This includes the actual
  assembled CSVs referenced by those locks.
- Read the prior parent audit: 120 checkpoints reproduced and 120 continuous
  accounts replayed. Read the extension audit: 20 additional checkpoints and
  204 compared paths, including 120 unchanged parent paths. Those complete
  checkpoint/account audits were not rerun in this review.
- Computed new paired diagnostics from the frozen extension curves, including
  the unchanged parent curves: 10,000 circular-block resamples at each of 5,
  20, and 60 sessions. All declared block lengths and all four comparisons are
  retained in [audit_results.json](audit_results.json).
- Added this report, [the diagnostic script](analyze.py), and its JSON output.
  No forecasting, training, accounting, frozen run, or input data was changed.

The active fallback is the official historical NASDAQ-100 reconstruction, with
101–104 included share classes per month. It excludes financial companies and is
different from a reconstructed ranking of the largest NASDAQ-listed companies.
The assembled panel has 156 security IDs across time, not 156 simultaneous members.

## What the results support

All rows below use the fixed, equal top-10, five-seed rank ensemble, base modeled
costs, and continuous January 3, 2022–December 31, 2025 account, except the ETF rows.
Sharpe uses the existing fixed 3% annual hurdle.

| Strategy | Net CAGR | Net Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| THINK | 12.3% | 0.435 | -29.0% |
| THINK+mixer | 19.6% | 0.629 | -39.4% |
| THINK Huber | 13.7% | 0.475 | -36.7% |
| THINK+mixer Huber | 6.0% | 0.245 | -41.7% |
| THINK+mixer pseudo-Huber | 11.4% | 0.412 | -40.7% |
| StockMixer | 8.6% | 0.326 | -34.0% |
| MLP | -10.1% | -0.208 | -44.5% |
| SPY | 10.9% | 0.498 | -24.6% |
| QQQ | 11.9% | 0.469 | -34.9% |

THINK+mixer deserves further research. Its original paired 95% Sharpe-difference
interval versus QQQ is [-0.262, +0.559]; versus SPY it is [-0.348, +0.549]. Both
include zero. It beats QQQ in 28/48 months and 3/4 calendar years, but beats SPY
in only 24/48 months and 2/4 years. Its five individual-seed CAGRs range from
8.5% to 21.2%, with median 12.4%; only 3/5 beat QQQ. Five initializations share
one market path and are not five independent confirmations.

The top-five extensions are interesting but less stable. StockMixer's 22.5% CAGR
depends heavily on 2024; SMCI, NVDA, and TSLA account for about 85% of that year's
gain in the existing reconciled attribution. Pseudo-Huber top five returns 22.4%
with -43.6% maximum drawdown, while its individual-seed median CAGR is 6.1%.
These are exploratory candidates, not stronger confirmation than the primary.

## Strengths

1. **Historical universe reconstruction addresses a major earlier bias.** The
   active pipeline retains former members and official share classes, rejects
   reused symbols, documents aliases, and explicitly handles acquisitions.
   Earlier 2026-selected NASDAQ models were retired rather than presented as
   historical point-in-time evidence.
2. **Time handling is unusually explicit.** Labels mature before the next split;
   training normalization and correlation groups use training data only. Monthly
   eligibility masks apply to fitting and inference. Perturbation tests check
   that future data and inactive inputs do not change earlier calculations.
3. **Portfolio accounting goes beyond multiplying selected-stock returns.** It
   carries holdings and cash across annual models, freezes share orders at the
   signal close, fills at the next close, applies liquidity caps and costs, and
   settles reviewed acquisitions. Independent replay checks are valuable.
4. **Experiment records expose failures and sensitivity.** Fixed protocols,
   all seeds, K sensitivity, transaction-cost stress, source hashes, checkpoints,
   and separate ranking metrics make it possible to audit an attractive curve.
   MLP, weak robust-loss variants, and disappointing shorts remain visible.
5. **The architecture comparisons are tractable.** THINK has 719 parameters;
   THINK+mixer has 5,883; StockMixer has 6,000. This makes matched ablations and
   reproducibility practical. The documentation correctly calls them local
   adaptations rather than certified paper reproductions.

## Findings and model gaps, ordered by impact

### 1. High research priority: insufficient independent confirmation

The test has 1,003 daily account returns, 200 complete five-session forecast
periods, 48 months, and four calendar years. Those are different views of the
same short history. Correlated stocks do not turn this into tens of thousands
of independent market experiments.

The four fitting folds contain only 75, 125, 176, and 200 training signal dates;
each has 49 validation dates. Cross-sectional pooling is useful, but market and
sector dependence sharply limit effective information. Previously inspecting
2022–2025 and then changing models, losses, K, or allocation rules creates
research selection that chronological split correctness alone cannot remove.

Action: freeze the entire candidate decision rule, log all attempted variants,
and evaluate an untouched future sequence. Acquire a longer reliable historical
panel for broader development and regime stress. Treat backfilled earlier dates
as retrospective evidence; do not claim that expanding the archive erases prior
selection. Apply a declared multiplicity correction to the tested family.
[Bailey and López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
and [Harvey and Liu](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF)
explain the relevance of the full search history and dependent tests.

### 2. High model priority: cold starts in the ticker-specific mixer

`research_models.py:21,34,57` uses dense mappings across the fixed list of stock
IDs. The masks correctly prevent inactive inputs from leaking into active scores.
However, a stock absent from every training eligibility mask has no data-driven
gradient for its mixer input column or output row. Weight decay can still modify
those parameters; the shared forecasting layers do learn from other stocks.

Fresh counts of test-eligible IDs absent from all training masks are 18, 18, 15,
and 12 in the four folds. Examples include SMCI in 2024 and PLTR in 2025. Some
appear in validation; early stopping sees them, but validation does not train
their stock-specific parameters. This is a concrete generalization concern,
not evidence of look-ahead leakage or proof that it caused the reported profits.

Action: add entry-cohort performance and mixer-contribution diagnostics. Use a
shared, permutation-equivariant cross-stock aggregator, or gate the mixer residual
to zero for stocks lacking training support while allowing their shared base
forecast. Refit on a fixed causal rolling schedule after labels mature. Test
future unseen security IDs and membership changes explicitly. The current
fixed 156-node model also needs a policy for IDs outside that historical list.

### 3. High model priority: objective and allocation mismatch

The active features are daily return, distance from the 5/20-session moving
averages, 20-session volatility, and relative dollar volume, over 16 sessions.
These are a sensible small baseline but provide limited explicit medium-term,
market, sector, or event context. The active THINK variants minimize raw return
MSE or robust loss. StockMixer adds a pairwise ranking penalty. Every neural
variant selects its checkpoint with prior-year MSE, even under Huber training.

The investment objective is benchmark-relative net return with controlled risk.
Predicting the common market move can improve MSE while adding little selection
value. A lower MSE need not stabilize the fifth or tenth ranked name. The same
fixed top-K rule buys stocks even when all return forecasts are negative; averaged
rank scores are not calibrated expected-return magnitudes.

Action: compare a tightly scoped benchmark/sector residual-return target against
the existing target, using only lagged exposure estimates. Compare one declared
ranking-aware objective and stable validation rule. Avoid selecting epochs from
raw portfolio Sharpe on just 49 validation origins. Report top-bucket realized
active returns, score calibration, cutoff margins, and RankIC by year and regime.

Extend inputs in controlled groups: 63/126/252-session trends, market and sector
relative returns, downside risk, liquidity changes, and causal market state.
Add earnings/fundamental features only with publication timestamps and correct
historical vintages. Shared longer-window elastic-net and a small tree baseline
would test whether extra information helps before adding architecture complexity.
Research supports examining price trends, liquidity, and volatility as feature
families; it does not establish their success at this study's five-session horizon.
[Gu, Kelly, and Xiu](https://doi.org/10.1093/rfs/hhaa009).

### 4. High portfolio priority: concentration, cutoff instability, and risk exposure

Top five targets roughly 20% per security, and top ten roughly 10%. Abrupt changes
around the cutoff turn modest score changes into large trades. Two share classes
can also concentrate exposure to one issuer. The existing loss-comparison analysis
finds broad cross-model rank agreement but much smaller selected-basket overlap.
Changing robust loss alone does not solve this economic sensitivity.

Fresh single-ETF regressions estimate THINK+mixer top ten's beta at 1.22 to QQQ
and 1.51 to SPY. Its QQQ regression alpha is +6.46% annually, with a normal 95%
HAC interval of approximately [-5.63%, +18.55%]. This is a fixed 3% hurdle,
single-factor exploratory attribution, not a complete factor model or a
selection-adjusted finding. Higher raw CAGR cannot all be identified as stock
selection skill.

Action: test shrunk benchmark-relative scores, issuer/sector limits, a turnover
penalty, and a predeclared retention/no-trade buffer. Use a shrunk covariance
estimate and explicit active-risk budget rather than independent inverse
volatility alone. Compare against beta-matched ETFs, sector/style controls, and
a QQQ-core portfolio with a modest active sleeve. Simulate each full account.
Lowering the sleeve size reduces active returns as well as active volatility;
this can improve suitability without increasing evidence of skill.

### 5. Medium code priority: stale validation export remains in the fit helper

`pseudohuber_think_mixer.py:62–68` restores the selected checkpoint, then returns
`vp` retained from the last training epoch. Test predictions are recomputed from
the selected checkpoint; validation predictions are not.

This issue is already disclosed in the extension audit. That auditor preserves
the original locked arrays and exports selected-checkpoint validation separately
under `selected_validation/`, with `VALIDATION_EXPORT_CORRECTION.json` recording
the correction. Published test paths remain unchanged. The implementation still
repeats the problem for new calls, and a future consumer could tune a policy on
validation scores that do not match its checkpoint.

Action: fix the helper in a new version to recompute validation after restoration;
add a regression in which the best epoch precedes the final epoch. Preserve the
existing locked study and its correction record.

### 6. Medium code priority: fail-fast integrity is inconsistent

`run_historical_ndx.py:23–27` hashes its selected source files and manifest, then
copies the manifest's expected CSV hashes into its lock. `load_data` checks the
review status but does not rehash the CSVs. A modified CSV with an unchanged
manifest can therefore pass this runner's input-lock comparison. The independent
auditor and extension runner do rehash actual files, and the present files all
match. This gap concerns future direct runs, not a demonstrated alteration of
the current study.

Action: verify all input bytes and transitive dependencies before loading or
resuming, hash cached forecasts as well as checkpoints, and replace operational
`assert` gates with explicit exceptions. Python optimization can disable asserts.
Keep the reusable engine separate from run scripts, while retaining immutable
source snapshots for old experiments. Add a small automated test workflow for
the core timing/accounting regressions; no CI workflow was found in this checkout.

### 7. Medium data/execution priority: research approximations remain

Membership is reconstructed rather than a licensed as-published archive. Prices
combine retrospectively adjusted Yahoo and Tiingo series. Economic-price units
approximate reinvested returns; they are not raw executable shares with separate
dividend cash, split processing, and broker settlement. Acquisition cash is
available at completion close. Forced terminal liquidation is a separate
approximation to normal capped orders.

Action: validate action dates, adjustment factors, and outlier returns against
an independent source, with an append-only issue ledger. Add settlement-delay,
fill, and capacity stresses and, for deployment, raw-share corporate-action and
dividend accounting. Use dated cash yields rather than a fixed Sharpe hurdle,
especially when comparing strategies with material cash. Existing 2x/4x costs
are useful: THINK+mixer top ten falls from 19.6% to 17.5% to 13.3% CAGR across
1x/2x/4x costs, with Sharpe falling to 0.459 at 4x.

Shorting is lower priority: prior results show few short sleeves improving on
their matched long-plus-cash controls. Borrow availability and execution are
unverified. Equal long/short dollars also do not impose beta or sector neutrality.

## How to obtain meaningfully narrower confidence intervals

First define the quantity. A CI for mean benchmark-relative return differs from
a CI for Sharpe difference, a CI for compounded relative wealth growth, and a
prediction interval for next month's outcome. The saved parent bootstrap primarily
answers the Sharpe-difference question, not the frequency of future monthly wins.
Its positive-bootstrap fraction is not a posterior probability of future success.

New 20-session, 10,000-replicate diagnostics for THINK+mixer top ten are:

| Quantity | Versus QQQ | Versus SPY |
| --- | ---: | ---: |
| Annualized arithmetic mean daily active return | +8.86 pp | +11.01 pp |
| Exploratory paired 95% interval for that mean | [-3.11, +20.95] pp | [-3.70, +25.68] pp |
| Sharpe difference | +0.160 | +0.131 |
| Exploratory paired 95% Sharpe interval | [-0.263, +0.555] | [-0.343, +0.565] |
| Annualized tracking error | 14.36% | 18.77% |

Arithmetic active means are not differences of CAGRs. The additional relative-
wealth-growth intervals are in the JSON. All examined comparisons retain zero
inside their intervals at all three block lengths. Changing to 10,000 replicates
reduces bootstrap simulation error; it does not add independent observations.

For a rough planning model, the standard error of annual mean active return is
approximately long-run annual active volatility divided by the square root of
years. Four times as much comparable effective history approximately halves
the interval width. Halving genuine active volatility at unchanged expected
active return would also halve it. Stationarity, serial dependence, training
changes, and selection complicate that approximation.

At 14.36% tracking error, an IID illustration gives a 95% half-width near 14
percentage points over four years. A +/-3 percentage point target would require
about 88 years at that same volatility under those simplified assumptions. This
is a warning about statistical power, not a forecast or required waiting period.
The useful response is better signals, lower uncompensated active risk, and more
credible data, rather than a different resampling seed.

For the next evaluation module:

1. Pair all strategies and ETFs on identical dates, and resample whole time
   blocks of the panel jointly. Preserve stock dependence; never bootstrap
   individual stock/date observations as independent market histories.
2. Add studentized time-series bootstrap intervals or HAC inference, plus
   predeclared block-length sensitivity. A circular bootstrap across changing
   annual fitted models is an approximation; report annual/regime results and
   leave-one-year-out sensitivity, without removing bad years from the primary.
   See [Ledoit and Wolf](https://www.ledoit.net/Robust_Sharpe_2008.pdf).
3. Separate fixed-fitted-strategy path uncertainty from training/selection
   uncertainty. Use nested chronological walk-forward evaluation of the entire
   selection/refit rule. Report training-seed sensitivity separately; a carefully
   designed joint block/refit analysis can add an algorithm-uncertainty diagnostic.
4. Track the full experiment family. Use a declared dependent-test adjustment,
   such as a joint max-statistic/stepdown method; DSR is a useful supplementary
   selection diagnostic, not a replacement for benchmark-paired inference.
   An honest correction can widen intervals. That is improved reliability.
5. Define 'regularly' before looking: mean active return, monthly win rate, and
   rolling 12-month performance are different criteria. Overlapping rolling
   windows are useful diagnostics but not independent confirmations.

Larger ensembles, every-day training origins, or more instruments may improve
estimation, but cannot multiply the number of independent market years. Daily
training origins create overlapping five-session labels and require maturity
purging and dependence-aware evaluation. More bootstrap draws and lower confidence
levels do not establish a stronger edge. Conformal prediction does not generally
make financial prediction intervals narrower or guarantee coverage under arbitrary
regime change; it would answer a different uncertainty question.

## Recommended next research sequence

1. **Integrity and diagnostics:** address the stale validation helper and loader
   hash gap in versioned code; add cold-start, membership-entry, attribution,
   turnover, factor, and leave-one-year-out reports. Keep frozen results intact.
2. **One small development batch:** retain THINK and THINK+mixer as references;
   compare a shared/gated mixer, a benchmark-residual target, and one regularized
   allocation rule through matched ablations. Include strong simple baselines.
   Declare the finite search, selection score, training dates, costs, risk budget,
   and stopping rule in advance. Do not tune a large grid on 2022–2025.
3. **Confirmation:** freeze the chosen full pipeline, run forward paper orders
   after the final freeze, and retain all decisions and canceled fills. Historical
   2026 dates already examined anywhere in development are not fresh holdouts.
   Prespecify when to evaluate and how repeated monitoring affects inference.
4. **Promotion criteria:** require positive net active return with a conservative
   lower confidence bound against each named benchmark, multiplicity control,
   acceptable downside/capacity/cost stress, and tolerable seed/entry-cohort
   sensitivity. Set numerical risk and economic-return hurdles before the next
   confirmation run. A narrow interval around negligible or negative alpha fails.

The defensible current candidate is THINK+mixer top ten for further research.
The strongest investment in the project is improved generalization and evaluation
of that candidate, supported by richer causal information and controlled active
risk. The attractive top-five curves merit diagnostic attention, but their
concentration and seed sensitivity make them weaker foundations for a reliability
claim.
