# Matched allocation, exposure and execution attribution —2026-10-27

Instantiate Research_Plan §9 on verified saved forecasts. This is the already-authorized saved-output study, not financial fitting. All2022–2025history remains exploratory; strict checkpoint gate and earlier two primary contrasts stay unchanged.

## Frozen cells

- E blend: original/new-five/ten-seed × original top5 versus issuer5 × full,constant.75,20-session basketvol target.20,QQQ/MA200 cap.50,20-session basketvol target.25plus MA200cap.50:30cells.
- B,D,REV5,Hlabel: three seed groups, equal top5, each full/vol20/training-calibrated constant:36cells. REV5/Hlabel seed-group duplicates are retained and disclosed, not independent evidence.
- Eligible EW and QQQ: each full/vol20/training-calibrated constant:6cells. EW equal-weights the same inference-eligible members at each five-session origin (weekly matched policy, not the old monthly index benchmark). QQQ is a one-ETF weekly account using the same reviewed QQQ adjusted/economic marks and trailing20-session dollar ADV; no acquisition events. Calendar/origin/entry axes identical to stock arms.
- E original-top5 training-calibrated constant, three groups:3additional cells. Total75fixed account policies.
- Re-execute every policy at modeled participation costs1×/2×/4×, flat10/20/30bpper side(replacing rather than adding to modeled fees), and zero cost:525accounts. Cash/participation caps/frozen quantity calculation/cancellations/settlement/terminal liquidation remain in every scenario. No leverage or interest. Flat/zero fee adapters are scoped in-memory to the new analysis; frozen source bytes remain unchanged.

## Risk and timing

Use unchanged ndx_drawdown_engine allocations, risk_inputs, exposure_series and simulate, weekly cadence. Risk is trailing20-session sample SD of the currently chosen target-weight basket applied to historical individual returns, including the signal close; nan returns use the source zero convention, annualization252, floor.05, warmup.30. Risk and marks are known before09:20decision, same entry-close fill. PM-derived rankings are frozen at09:20; the engine's preceding-close signal index is a sizing/risk timestamp, not a claim forecasts existed before PM. Exposure is min(1,target/risk), optionally capped by QQQ<MA200. Orders stay frozen until entry close. The inherited five-session membership/cancellation/settlement policy remains unchanged.

Fixed exposure calibration uses only each annual fold's training-origin return windows. At the first test decision of the year, freeze that arm's first selected target basket, measure its sample annual volatility in each20-session window ending at a training origin, retaining windows with≥15days where all selected returns are finite. At least30windows are required; otherwise use disclosed.30risk fallback. Median valid training-window volatility floored.05gives constant min(1,.20/median). Freeze until the next annual fit. The initial basket is known at that first decision; this is training-return risk calibration of the first-decision basket, not a training-only forecast or mean future exposure matching. No later test return/selection calibrates the constant. Record training counts/risks/fallbacks and first decision dates.

## Evidence and statistics

Fresh execution_batch8_risk and results/RISHI_MECHANISM_RISK_20261027. Store all paths/detail ledgers, metrics, calendar returns/proportional log growth, average actual exposure, cost/turnover, QQQ loading and downside covariance. Check all525NAV compounding, cash, terminal exposure; reproduce all30corresponding source base-cost metrics and source baseline returns where available. Describe selection×exposure interactions without additive fractions of Sharpe explained.

For zero-fee original-top5and matched controls, relate scaled-full path mean to mean prior-day actual managed exposure×mean full-frictionless return plus covariance, and explicitly retain managed-account residual. This is a descriptive drift/cap/execution reconciliation, not market-timing causality. First entry session prior exposure is zero. No cash yield is fabricated; interest remains zero in this study.

At ten seeds(for stock arms) or the single benchmark, full-versus-vol20 and fixed-versus-vol20 paired Sharpe contrasts for E,B,D,REV5,Hlabel,EW,QQQ form one14-test descriptive family.10000stationary draws, expected block21,seed27102032;5/63sensitivity. Report95%paired SR/annual arithmetic mean/CAGR intervals and Holm one-sided SR p across14within each block. No primary promotion, neural-alpha inference or post-outcome rule selection. Tests cover forward-mutation invariance of calibration/risk, flat-fee restoration, exact calendar compounding and inherited frozen-fill edge cases. Execution gaps/new entries/missing marks are tested as mechanics, not invented trading strategies.
