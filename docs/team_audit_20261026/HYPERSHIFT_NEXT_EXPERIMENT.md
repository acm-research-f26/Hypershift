# Definitive next experiment: complementary selection or persistent exposure?

**GO for an immutable saved-prediction replay and attribution gate.** No new neural training is needed initially. The primary research program is a single fixed ensemble; dynamic-architecture causality and market-state conditioning are later, gated questions.

Status: **detailed research proposal with deterministic operating rules; not a preregistered confirmation and not executed by this audit**. Historical 2022–25 outcomes are known. All historical diagnostics below are falsification/attribution, not new evidence of out-of-sample superiority. A prospective protocol must be signed and hash-frozen before its first observation. There is no authorization here to score the protected S&P 2024–September 2026 period.

## 1. Falsifiable research question

Does the fixed 50:50 blend of ten-seed THINK+mixer AH+PM ranks and ten-seed restricted-dynamic ranks improve net Sharpe over its parent **and over replacing the dynamic scores with a causal persistent-stock score**, on identical stocks, dates, execution and costs?

The mechanism is complementary **time-varying** stock selection, especially among the selected tail. It is refuted as an explanation if the gain disappears under a comparable persistent score, matched exposure control or feasible execution. A successful ensemble comparison does not by itself establish that learned dynamic hyperedges are responsible.

## 2. Required immutable export

Rishi exports one bundle tied to `0b06f26` and the original fit/evaluation seals. Keep financial data private where required. Required contents:

| File / table | Required fields |
|---|---|
| `manifest.json` | source Git SHAs; source/checkpoint/preprocessing hashes; schema/version; timestamp; dependency versions; seed order; original protocol/freeze/evaluation seals |
| `security_master.csv` | stable security ID; dated ticker/issuer mapping; membership start/end/announcement availability; corporate-event settlement fields |
| `calendar.csv` | exchange session; feature cutoff; decision; origin; entry; exit; actual horizon; train/validation/test indices per fold |
| `features_provenance.json` | raw vendor versions; adjustment policy; feature formulas; training means/SD/imputation; AH/PM availability/coverage; graph-training rows |
| `forecasts.npz` or partitioned equivalent | `[date,security,model,seed]` raw scores for parent and dynamic; inference eligibility mask distinct from realized-label mask; labels with exact economic endpoints; validation and test kept separate |
| `training_labels.npz` | each fold's already-used training targets/masks/IDs and boundaries; enough to construct identity score without any test outcomes |
| `accounts.csv` | strategy ID, date, NAV/cash/stock value, net/gross return, fees/settlements, traded dollars, exposure |
| `trades.csv` / `positions.csv` | decision/execution timestamp; ID; frozen order; fill proxy; quantity; price; ADV cap; fees; canceled order; corporate-event proceeds |
| `trial_registry.csv` | every attempted method/loss/optimizer/LR/checkpoint selection/k/seed group/overlay and failure; whether primary/secondary/exploratory; include retired branches where available |

The four-year new daily curves/ledgers and per-stock forecasts are necessary; portable summary CSVs alone are insufficient. Do not regenerate predictions with changed code and label them the original artifacts. If originals are gone, mark the claim non-replayable and separately cost a reconstruction.

## 3. Sequential gates and resource budget

| Step | Work / owner | Budget and exit condition |
|---|---|---|
|0 a|Rishi exports existing files; Tom validates schema/hashes|Up to half a day of team effort. Missing payload → DEFER, explicit missing-file list, no training |
|0 b|Tom+Ethan independently replay original accounts|Half to one day CPU. Match calendars/IDs, cash conservation, actual trades/costs and known tables |
|0 c|Tom runs one fixed diagnostic batch; Kavin independently checks identity/mixer interpretation|One to two days CPU after data receipt. No grids, no protected data |
|0 d|Freeze exact candidate and operational prospective contract|One day after written review. Signed gate, code hashes and start timestamp before forecasts |
|1|Log only genuinely new forecasts, orders, marks and costs|No new fitting initially;252 sessions initial review,504 sessions main review; low power can force DEFER |
|2|Only if Stage 1 supports the fixed ensemble: matched architecture mechanism test|New design and compute estimate required; not automatically authorized or launched |

These are effort estimates, not claims that the missing data can be obtained on schedule. A252 session observation window takes roughly one trading year;504 roughly two. No attempt should promise a reliable small-alpha answer within weeks.

## 4. Replay acceptance criteria

Use native full-precision exports, stable IDs and original fills. Target tolerances: daily return absolute difference≤1e−10, NAV relative difference≤1e−8, SR/CAGR/DD≤1e−8. Validate tied-score fractional allocations, signal-vs-label masks, missing fills, membership exits, acquisitions and terminal liquidation. Tolerance misses require a documented reconciliation, not automatic rounding away; lossy exports must explain their precision and adopt a pre-outcome bound before reanalysis.

Confirm every output path is new. Do not overwrite sealed runs. Verify inputs truncated at decision time give identical features, graph and scores. No future membership/labels may alter inference eligibility. Costs and portfolios must be replayable by an implementation independent of the production allocation function. A passed contributor audit does not replace this independent check.

## 5. Fixed arms: no selection among them

Let `R_t` map eligible scores to percentile ranks with average ties. Define:

`B_t = mean_s R_t(parent_s,t)`

`D_t = mean_s R_t(dynamic_s,t)`

`E_t = 0.5*B_t + 0.5*D_t`.

The primary candidate is **E**, ten seeds, equal-weight **top 10**, every five exchange sessions. Existing top 5 is a labeled sensitivity only; it cannot become primary because its historical SR is higher.

Mandatory controls on the identical cross-section:

- **B:** ten-seed parent, same portfolio policy.
- **D:** ten-seed dynamic alone, attribution comparator.
- **H:** per-security average five-session economic label over that fold's training rows only, at least 30 valid observations; no test return, present ticker or future membership. Missing H gets the training cross-sectional median. For prospective logging use the final original fit's training window, then keep H fixed.
- **E-H:** `0.5*B + 0.5*R(H)`, one cheap persistent-score substitute for the dynamic component. This is the mechanistic control, not a retrospectively optimized benchmark.
- **REV5:** negative preceding five-session total return, available by the common decision cutoff; same entry and top 10 rule.
- **EW:** equal-weight eligible universe, rebalance at the close of the first exchange session of each month, eligibility and share orders frozen at 09:20 ET that morning. Apply the same fees, supported universe and mandatory membership exits. If that rebalance is canceled by the common failure rule below, wait until the next month's scheduled rebalance; mandatory exits still occur. QQQ and SPY are external reference paths.

A constant per-security **mean training prediction** is an additional descriptive diagnostic for neural identity preference, not another candidate or promotion test. Compare it with the training-label H because they answer different questions. Do not call a cross-sectional Spearman correlation a variance-explained statistic.

## 6. Historical attribution batch

Run once on the exported, already-spent 2022–25 development record. Do not retrain, tune blend weights, select seeds, choose k, choose regimes, or change costs from the observed answer.

1. Reproduce each original account and the published ranking/seed-half table. Match ten-seed primary, not just original-five/top 5.
2. For E/B/D/H/E-H/REV5 report per-origin RankIC with paired daily/weekly time aggregation, top 10 forward return minus eligible EW, top-decile and bottom-decile returns, top 10 overlap, turnover, concentration, yearly results and every seed's diagnostics. Labels do not decide executable support.
3. Separate stable identity and changing scores using the fixed training-prediction score. Measure per-ID time variation, score-rank persistence, holdings persistence and performance of H/E-H. Report exposure and stock attribution rather than replacing them with a correlation threshold.
4. Estimate QQQ beta and FF5+MOM+ST_Rev Newey–West alpha as **historical attribution**. Report sector, momentum, volatility and size exposures. Current-vintage factors can be used for ex-post description with that limitation; they cannot enter causal trading states. Regressing out exposures using the entire test sample is not an implementable control.
5. Construct one causal QQQ-beta-neutral diagnostic using only trailing 126 daily observations available before each rebalance. Estimate beta by OLS with intercept against QQQ excess returns, clip beta to[0,1.5], retain the underlying stock book and short QQQ notional equal to `beta * contemporaneous account NAV`. Before 126 observations use beta 1. Rebalance the hedge only with the stock schedule; use the same delayed close fill and participation-cost formula. Short proceeds stay in zero-interest collateral; charge 3% annual borrow on short QQQ market value, accrued actual calendar days/365, and disclose hypothetical margin/gross exposure up to 2.5×. Apply this same rule separately to E, B and E-H. This causal diagnostic still requires borrow/margin availability and is not the primary LO strategy or a fitted full-sample hedge.
6. Report original/new seed halves and per-year signs. Rank-average across seeds is primary. Resampled seed bundles estimate initialization sensitivity conditional on this history, not independent market evidence.
7. Cost grid: original participation model×1/2/4 and separately flat one-way 10/20/30 bp. Report fee dollars, turnover and active-return changes; do not mix multipliers with flat bp. If a quote/auction-cost proxy later exists, calibrate its formula without using strategy returns and freeze it before prospective comparison.
8. No regime search. At most report calendar-year partitions and one already-defined causal state for descriptive power estimation; no state-specific success claim.

### Historical decision rule

Historical results can **kill or simplify** a candidate, not confirm future alpha. Material accounting/timing failures → STOP interpretation pending repair. Non-replayable payload → DEFER. If E lacks any positive net increment over B **and** E-H at the base and 20 bp cost views, do not allocate new training to it. If E-H is economically equivalent to E, simplify the proposed mechanism to persistent exposure.

Equivalence requires a paired interval narrow enough to exclude differences outside±2 percentage points/year in net arithmetic mean return; a nonsignificant contrast does not establish equivalence. This margin represents 10% of a 20%-volatility equity book's annual risk, approximately .10 SR if risk were unchanged, rather than a universal alpha cutoff. Use the actual paired covariance and risk when interpreting it. If uncertainty is too wide, the label is **DEFER**, not “identity proven.” The historical gate's resource choice may still favor cheap logging over new fitting.

## 7. Training, model selection and prospective implementation

**No initial new fits.** Preserve the original 2025 annual parent and dynamic checkpoints, training-derived graph, normalization and ten seeds. This tests the frozen system rather than hiding a refit improvement. Track staleness as a limitation. At the first prospective review, do not refit and combine the resulting track record as if unchanged.

Use the common fixed checkpoint security-ID universe intersected with point-in-time eligible/tradable names. Exclude unseen IDs requiring new stock-indexed weights; retain them in the coverage log. All controls use that same intersection. This tests the supported universe, not full future Nasdaq 100. Name changes use stable IDs; remove ineligible holdings at the first feasible declared execution, account for cash/stock consideration, and disclose delayed/unfillable exits.

Freeze the feature/mask pipeline and normalization,16 sessions/25 channels and exact trained objectives. Prices can arrive for feature warmup without scoring historical portfolio performance. AH/PM coverage flags and training-mean imputation stay unchanged; do not improve coverage rules in response to returns. A forecast not timestamped before 09:20 is missing. If either neural family is missing/nonfinite on a scheduled rebalance, cancel discretionary orders for **all arms**, retain existing shares/cash until the next scheduled rebalance, and still execute mandatory membership exits/corporate settlements. Start in cash if the first rebalance fails. Log failure, coverage and every carried position. A held asset without a defensible valuation invalidates that day's economic inference until reconciled; never mark it to zero or delete the day. This common deterministic fallback prevents outcome-dependent substitution.

Start no earlier than **November 2, 2026**, and only after the complete freeze. If readiness slips, start at the next exchange session after approval, with no backfill. Any October 21/22 reservation belongs to its own existing protocol; it cannot make this later choice confirmatory from that date. No automation or live trading is created by this document.

At 09:20ET freeze scores, desired shares and liquidity limits using completed information and preceding regular close marks. Execute at the regular-session close proxy; actual fill records, if available, replace proxies only under a predeclared measurement policy. Weekly signal clock: first prospective session then every fifth exchange session; maintain positions between rebalances, with mandatory membership/corporate-action handling. Cap each trade at 1% of prior 20 session ADV, scale buys to available cash, cancel unavailable fills and do not rerank. These reproduce the candidate's constraints and disclose the distinction between targets and actual holdings. No leverage or additional volatility timing is added to the primary LO account.

Research-account target initial NAV$1 M is for liquidity/cost calculation only; no capital deployment. Cash return and financing policy must be common across arms: primary zero cash interest for exact historical continuity, with observed Treasury cash return reported as a labeled sensitivity. Sharpe uses the fixed 3% hurdle for continuity, and rf 0 plus realized-risk-free versions as separate descriptive columns.

## 8. Economic and statistical evaluation

The **primary statistic** is paired net SR difference `SR(E)-SR(B)` at base costs. For the full scientific claim also require the fixed **E−E-H** mechanism contrast. Treat these two as a predefined family (Holm one-sided 5%, or synchronized simultaneous bounds). Do not promote the cheaper control and relabel its test after seeing outcomes.

Report absolute SR/CAGR/maxDD and paired net arithmetic mean differences, CAGR differences and tracking-error IR. SR difference, active-return IR and alpha t are distinct quantities. Absolute strategy intervals overlapping or not overlapping do not test paired superiority. Bootstrap synchronized date blocks across every arm,10,000 draws, stationary block 21 sessions primary;5/63 sensitivity. Include inactive/cash days. Use weekly-origin inference for five-session ranking labels; do not count overlapping stock targets or seeds as independent market observations. NW lag 5 daily for mean/alpha support, with disclosed sensitivity; long-horizon or different-holding claims require a revised dependence choice before outcomes.

Monthly live monitoring is restricted to data quality, timestamps, accounting, gross limits and missing inputs. Formal performance reads are at 252 and 504 sessions; no monthly “winning model” updates or stopping based on a good Sharpe. Because two looks can create optional-stopping bias, the 252 read is diagnostic only; the 504 read is the single confirmatory test unless a predeclared sequential method is approved before the start.

Before Stage 1, estimate paired long-run covariance from development returns, then simulate synthetic increments with the observed dependence structure. Report power to detectΔSR.10/.20 and net mean increments 1/2% per year. These are scenario effects, not forecasts. **Do not invent a sample size/power result before the raw new return tape arrives.** If the target 2% effect has<80% power at 504 sessions, label the planned horizon underpowered and specify a longer fixed endpoint or an explicitly exploratory pilot before any future outcomes. Do not repeatedly extend the endpoint in reaction to p-values.

### Prospective decision rules

- **STOP takes precedence:** material timing/accounting failures stop interpretation pending a separately versioned repair. Hard execution/capacity infeasibility or a breach of the predeclared drawdown resource limit stops this candidate as an implementation program, without claiming zero predictive information. At a sufficiently powered final endpoint also STOP if the upper 95% bound on net mean improvement versus B is below 2 pp/year, or a family-adjusted upper bound establishes E is no better than E-H. Preserve and report the scientific result when stopping for resources.
- **CONTINUE research only if no STOP condition applies:** at the frozen confirmatory endpoint, E−B and E−E-H have positive family-adjusted lower 95% bounds for SR; point SR increment≥.10 versus B; net annual arithmetic mean increment≥2 pp versus B; net mean versus EW>0; positive point net mean increments versus B and E-H at 20 bp/side; drawdown no more than 5 pp worse than the worse of B/EW. Show risk and capacity, not just thresholds. These are proposal-specific hurdles for a complex equity system, to be ratified before start.
- **DEFER in every remaining case:** this includes unavailable data, inadequate power, broad intervals, a precisely positive but economically borderline gain, failure of the mechanism/cost gates without decisive harm/equivalence, or any valid result that does not satisfy all CONTINUE criteria. DEFER does not trigger an architecture search or a result-driven endpoint extension.

All criteria above must be frozen together. Passing one attractive metric cannot compensate for a failed primary or control. Research continuation does not authorize capital. Statistical success does not establish capacity.

## 9. What would justify architecture or conditional research later?

If the ensemble survives, test the restricted dynamic architecture with **the exact same input/training/portfolio** against a frozen-incidence twin, no-relational-path twin and degree/cardinality-matched placebo. Equal seeds, validation budget and compute; synthetic relational/nonrelational recovery first. Freezing a trained graph at inference is only a perturbation, not a replacement for retrained matched controls. Predeclare whether the target is learned membership, static wavelets or temporal processing; changing all three does not identify one.

Only after time-varying incremental selection is established should one state-conditioned hypothesis be designed. Candidate state must be available before trading, economically justified, and powered for an interaction/full-calendar strategy, including all off-state cash/benchmark days. No stress-state threshold is selected here and no regime data are scored.

## 10. Contribution even if the hypothesis fails

A reusable immutable prediction/ledger interface; a documented separation of cross-sectional ranking, tail selection and stock-ID exposure; paired seed and market-time uncertainty; and evidence about how costly ensemble boundaries turn weak rankings into attractive historical portfolios. A negative result becomes scientifically useful through mechanism identification, complete reporting and independent reproducibility—not by claiming neural equity prediction is impossible.
