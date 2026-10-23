# Prospective paper-trading specification: style-only characteristic book (frozen 2026-10-22)

**Status.** Frozen by the commit that adds this file and `STRATEGY_MANIFEST.json`, before any post-freeze data exists.
- **What it is:** an implementation check plus slow evidence accumulation.
- **What it is not:** a validation of alpha. It commits no capital.
- **Independence:** it is separate from the FB prospective protocol (v1 / v1-LF), which is untouched.

## Why this candidate

- **The strongest saved candidate** is the style-only LightGBM sector-neutral book. Exploratory results on inspected years 2019–2023 (`STYLE_BOOK.md`, `STYLE_ROBUST.md`):
  - 3-seed net Sharpe +1.19 [+0.48, +1.96] at 10 bp with borrow;
  - mostly *not* explained by FF5 + MOM + ST_Rev (R² of about 0.1).
- **Its edge roughly halves when eligibility is restricted to point-in-time S&P 500 members** (+0.61 [−0.15, +1.47], alpha t 0.87). The short leg loses in beaten-down volatile names, which is the survivorship signature.
- **The historical evidence therefore cannot separate a real edge from universe bias.** A prospective record is survivorship-free by construction: the universe is fixed when each signal is formed.

## Exact procedure

| Item | Specification |
|---|---|
| Universe | The 462 tickers of `data/fresh/sp500_daily/tickers.txt` (2026 S&P 500 constituents with data), restricted at each signal date to current S&P 500 members (fja05680 membership file, refreshed at each rebalance). No additions. A name that leaves the index is exited at the next rebalance |
| Eligibility | 64 consecutive observed sessions (FB rule) and membership at the signal date |
| Features | cs_z of beta (126-day, vs equal weight), log 20-day dollar volume computed from **unadjusted** close × volume, 60-day return, 60-day volatility; plus sector one-hot (`fbaudit.matched.nested_inputs(..., 'style')`; leak-strict `fbaudit.prospective_lf.make_leakfree`) |
| Target | FB residual target (5-day return residualised per date on sector + z(beta, log dollar volume, ret_60, vol_60)) |
| Model | LightGBM, FB `LGBM_PARAMS`, 50 rounds, seeds 7, 19 and 42. Predictions are rank-averaged across seeds (`strategy.paper.style_scores`) |
| Training | At the first rebalance: all signals from 2015 whose labels matured by the signal close. Retrain every 252 trading days. No validation, no tuning |
| Book | `strategy.engine.ls_weights`: sector-neutral long top 10% / short bottom 10% per sector (sectors with ≥ 10 eligible names), 0.5 gross per leg, hold band 30%, plus an equal-weight basket sized to zero ex-ante beta |
| Schedule | First signal at the close of **2026-11-30**; first trade at the **open of 2026-12-01**; then every 5 trading days |
| Fills (paper) | Official open price of the trade day. Assumed cost 10 bp per unit of two-sided traded notional / NAV. Borrow 50 bp/yr on short gross. Cash 0% |
| Record | `scripts/paper_trade.py` writes `results/PAPER_TRADING/targets_<date>.csv` plus a sha256 in `log.csv`. Targets must be generated and logged **before** the trade day's open |
| Outcome computation | Only from logged targets and realised open/close prices, using `strategy.engine.run` on the logged weights. No re-generation of past targets |

## Evaluation

- **Implementation check (monthly).**
  - Recomputing logged targets from archived inputs must reproduce them exactly. LightGBM is deterministic.
  - Paper P&L recomputed independently from prices must match the engine to within 1 bp per week.
- **Descriptive performance:** at 12 and 24 months, report:
  - net Sharpe (10 bp + borrow);
  - FF5 + MOM + ST_Rev attribution;
  - per-leg contributions;
  - turnover;
  - drawdown.
- **Stop rules:**
  - a 12-month net Sharpe below −0.5, or a drawdown worse than −10% of NAV at any time, means stop and review;
  - an implementation mismatch means pause until fixed.
- **Power, stated in advance.** Detecting a true IR of 0.6 at 80% power needs about (2.8/0.6)² ≈ 22 years. One or two years of paper trading **cannot validate alpha**. They can reveal implementation problems, gross failure, and, together with history, whether the survivorship concern was decisive.
- **Capital rule.** No capital is allocated on paper results alone. A capital decision needs both:
  1. a survivorship-free historical re-test (WP-B data) showing a net long-short Sharpe ≥ 0.5 with FF alpha t ≥ 2 over 2016–2023 on point-in-time members **including delisted names**;
  2. at least 12 months of paper trading with net Sharpe > 0 and no implementation breaches.

## What may not change

Universe rule, features, target, model, seeds, rounds, book, schedule, costs, stop rules. Any change requires a new, dated spec and restarts the record.
