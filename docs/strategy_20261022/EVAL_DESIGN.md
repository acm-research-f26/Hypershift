# D. Evaluation design (applies to every candidate)

## Stages

Each stage is gated; a candidate never skips one.

1. **Predeclare.** Commit the candidate, its rules and its thresholds before computing anything.
2. **Explore** on a **point-in-time universe including delisted names** (Norgate or Sharadar). Survivor panels are allowed only for engineering and debugging, never for decisions.
3. **Freeze.** Commit the spec, the code sha256 and the decision rules.
4. **Holdout, scored once:**
   - for price-based strategies, 1990–2014 (about 25 years; includes the preserved 2004–2012);
   - for EDGAR-based strategies, 2024–2026.
5. **Prospective paper trading**, at least 12 months.
6. **Capital:** small and risk-limited, after stages 4 and 5.

## Economic objective and metrics

All definitions are in `src/hypershift/strategy/{engine,study,factors}.py`.

- **Primary.** Net return of the strategy minus a **matched, implementable benchmark**:
  - long-only: the equal-weight universe (and cap-weighted where available), on the same days, with the same costs;
  - long-short: zero.
- **Reported for each:**
  - annualised mean, volatility, IR (or Sharpe);
  - stationary-bootstrap CI (block 21 days);
  - maximum drawdown;
  - worst calendar year.
- **Attribution.** Daily OLS on FF5 + MOM + ST_Rev with Newey–West (10 lags). The alpha and its t-statistic measure incremental value over factors an investor can buy cheaply.
- **Costs:**
  - 10 bp per unit of two-sided traded notional (sensitivities 5/20/30);
  - borrow 50 bp/yr on short gross (sensitivity 100–200 bp/yr for hard-to-borrow names once borrow data exist);
  - next-open execution, with next-close as a sensitivity.
  - **Omitted, disclosed:** market impact beyond linear, financing spreads, taxes.
- **Implementation:** weights drift between trades, a 20%/40% (LO) or 10%/30% (LS) buffer, and rebalances weekly or monthly. Turnover is reported as traded notional per year.
- **Matched exposures.** Each candidate is also compared against a characteristics-only book built from the same style inputs. A model must beat that book, not only zero.

## Success and failure criteria (per stage)

| Stage | Success | Failure → action |
|---|---|---|
| Explore (PIT data) | Net IR (LO active) ≥ 0.30 *or* LS net Sharpe ≥ 0.5 at 10 bp; FF alpha ≥ 0; IR > 0 at 20 bp; worst year > −10%; traded < 300%/yr (LO) or < 1,500%/yr (weekly LS) | Any check fails → **STOP** the candidate (no tuning loop) |
| Holdout (once) | Net Sharpe / IR > 0 **and** FF alpha t ≥ 1 (25-year holdout). For short holdouts (< 5 years), kill-test only: STOP if negative | Negative → **STOP** |
| Paper trading | Implementation matches (≤ 1 bp/week); 12-month net Sharpe > 0; drawdown > −10% | A mismatch pauses; poor performance → review, then STOP |
| Capital | All of the above | — |

## Power: the arithmetic that governs everything

The IR standard error is about 1/√years.

| True IR | Years for 80% power (two-sided α 0.05) |
|---|---|
| 0.3 | ≈ 87 |
| 0.5 | ≈ 31 |
| 0.7 | ≈ 16 |
| 1.0 | ≈ 8 |

**Consequences:**
- realistic large-cap edges (IR 0.3–0.7) cannot be confirmed from short samples;
- long, survivorship-free history is the main lever (25 untouched years give MDE ≈ 0.56);
- paper trading mostly verifies implementation;
- decisions must combine economic prior, cost realism, robustness and long-history evidence, never one short backtest.
