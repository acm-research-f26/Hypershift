> The 2026-selected NASDAQ model artifacts were subsequently retired and deleted by user request. Their numerical audit remains historical evidence. The active replacement is the [historical monthly NASDAQ-100 fallback study](../../nasdaq_monthly_pit/CHARTS.md).

# What the audit establishes

The results reproduce, but the current top-100 study contains future universe
selection. We chose companies using 2026 membership or size and then evaluated
2022–2025. That gives the study knowledge of companies that survived and became
large. Checking feature timestamps does not remove this bias.

The audit independently rebuilt ridge coefficients, momentum scores, input
scalers and fold indices. A separate ledger holding adjusted-price share units
matched 384 saved selected/fixed daily portfolio paths. All 384 neural checkpoints
reproduced their saved validation/test forecasts, and early stopping matched
prior validation MSE records. Altering future prices/volumes left earlier model
features, eligibility, baseline scores, scalers and training correlation graphs
unchanged. Eight complete input-array fingerprints match the fitted runs.
The 25 existing timing, masking, loss and accounting tests also passed.

These checks do not certify the vendor's prices, issuer identities, delisting
returns, historical membership, cost model or a live trading edge. No forecast
timing or portfolio accounting error was found in the audited paths. Future
universe selection and missing-history survivorship remain unresolved.

## Momentum and ridge

126-session momentum computes adjusted close at the signal / adjusted close
126 sessions earlier − 1. It ranks each stock by its past six-month return.
Orders follow at the next session's close; a five-session test then holds until
five sessions after that entry. This version includes the most recent month.
It does not learn weights or use the coming return in its score.

Ridge takes the past 16 sessions of five training-scaled features, flattening
them to 80 inputs. It pools training stock/date observations, fits one shared
linear forecast with an intercept, and penalizes squared coefficient size.
The fixed penalty is 10 in the summed squared-error objective; the intercept
is unpenalized. Targets are expressed in percentage points. It minimizes
forecast error, rather than maximizing portfolio returns. Stocks are ranked by
that forecast. Allocation and K use prior validation, as for momentum.

The five-session selected NASDAQ top-100 momentum path returned 85.6% annualized,
994.2% total on the synthetic fold-compounded account, with a 39.7% drawdown and
Sharpe 1.588. Its annual returns were approximately −13.0%, +72.0%, +204.3%
and +162.4% for 2022–2025. The selected portfolio narrowed to five names in the
last three years, with a confidence-weighted five-name portfolio in 2024.
This is a concentrated strategy inside a retrospectively chosen cohort.

On the original random NASDAQ cohort, momentum returned only 9.3% annualized
with Sharpe 0.381. Ridge returned 4.9% annualized there, versus 41.0% on the
current top-100. On NYSE, ridge changed from −0.5% annualized on the original
cohort to 29.1% on current top-100. It therefore does not perform similarly
across the two samples already available.

Ridge's NYSE top-100 Sharpe advantage over the same-cohort passive basket was
0.165, with an exploratory paired 95% interval [−0.479, 0.769]. Its NASDAQ
advantage over QQQ was 0.437 [−0.151, 1.011]. Those intervals include zero.
NASDAQ momentum's difference was 0.963 [0.163, 1.796], but an interval cannot
correct future membership selection or repeated comparisons.

At four times modeled costs, ridge's NYSE Sharpe fell from 1.139 to 0.351 and
NASDAQ Sharpe from 1.062 to 0.574. NASDAQ momentum remained at 1.483, with about
77.4% annualized return and 40.0% drawdown. Cost stress tests address one
assumption, while leaving universe/data bias intact.

## The mixer comparison

THINK is the local paper-inspired hyperbolic stock-ranking implementation.
THINK+mixer adds a learned cross-stock residual to its scores. Extra parameters
can fit relationships within a cohort and can also overfit. Huber reduces
the influence of extreme training forecast errors. MSE and Huber variants
are separate experiments; their names should not be interchanged.

For current NASDAQ top-100, plain THINK+mixer trails plain THINK at both the
selected portfolio and fixed equal top-10. The Huber mixer beats THINK+Huber
by 0.208 Sharpe with selected allocation and 0.133 with fixed top-10. Their
paired intervals are [−0.395, 0.833] and [−0.366, 0.652]. On NYSE the mixer
differences are negative, but their paired intervals also include zero.
The data do not establish a dependable exchange-specific mixer benefit.

Training seeds, rank ensembling and separate validation choices also affect
curves. A fair architecture comparison holds the loss and portfolio rule fixed
and shows all seeds. The new sample trials do exactly that.

## What additional samples can answer

Three newly frozen 100-name draws per exchange test sensitivity to the ticker
sample. Each of four neural variants has three training seeds across four annual
tests, totaling 288 new fits. The primary portfolio is fixed equal top-10;
the original validation-selected rule is secondary. Missing histories are not
replaced. The charts show each draw and the mean/spread, rather than only the
best outcome.

Overlapping draws observed on the same market dates are dependent. Averaging
their Sharpes is descriptive, and is different from the Sharpe of an investable
combined portfolio. Shared-date block resamples preserve dependence when
calculating paired intervals. More draws can improve the sensitivity estimate,
but another sample of these same years cannot establish future performance.
A stronger confirmation needs a frozen protocol on unseen dates and a
point-in-time universe with original issuers and delisted returns.

NYSX is the passive NYSE 100 ETF, but it launched March 25, 2026 and its growth
index spans exchanges. It has no 2022–2025 ETF history. NYSE charts therefore
use a clearly labeled simulated cohort basket; NASDAQ retains QQQ.
[Issuer](https://www.globalxetfs.com/funds/nysx/).

The fresh trials run on CPU using PyTorch 2.14.0+cu130; the original CPU fits
used 2.14.0+cpu. Within-draw comparisons share a build and backend. The current
top-100 neural fits used CUDA, so changes between cohorts also change backend.

The completed fresh trials did not repeat the extreme top-100 returns. With
fixed equal top-10 allocation, mean annualized net returns were 10.2% NYSE and
9.8% NASDAQ for momentum, and 5.4% NYSE and −2.6% NASDAQ for ridge. These are
arithmetic averages of separate draw statistics. Plain mixer-minus-THINK mean
Sharpe differences were −0.081 NYSE and +0.007 NASDAQ, with intervals
[−0.418, 0.262] and [−0.345, 0.338]. Huber mixer-minus-THINK+Huber differences
were +0.179 NYSE and +0.184 NASDAQ, with intervals [−0.170, 0.529] and
[−0.190, 0.556]. All four mixer intervals include zero.

[Every fresh draw and its descriptive averages](../../random_trials_20261010/CHARTS.md).
