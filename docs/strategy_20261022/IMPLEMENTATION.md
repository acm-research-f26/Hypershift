# E. Implementation plan

## Built and tested in this round

| Module / script | Purpose | Tests |
|---|---|---|
| `src/hypershift/strategy/signals.py` | 12-1 momentum, 252-day low volatility, 21-day residual reversal, z-score composite | `test_momentum_lowvol_manual`, `test_composite_clip_mean` |
| `src/hypershift/strategy/engine.py` | LO / LS buffered engine: drift, next-open or next-close execution, linear costs, borrow, EW benchmark, `ls_weights`, trace | `test_open_execution_no_lookahead`, `test_benchmark_costed`, `test_engine_missing_open`, `test_borrow_accrual`, `test_engine_flat_prices_cost_only`, `test_trace_*`, `test_ls_weights_beta_neutral` |
| `src/hypershift/strategy/factors.py` | Ken French daily FF5 + MOM + ST_Rev loader; Newey–West attribution | `test_ff_loader_units`, `test_attribute_recovers_alpha` |
| `src/hypershift/strategy/study.py` | WP-A runner (data load, scores, arms, metrics, point-in-time filter, decision rule) | via scripts |
| `src/hypershift/strategy/paper.py`, `scripts/paper_trade.py` | Paper-trading target generator (style book) with a guard on label maturity | `test_style_predictions_guard_and_shape`, `test_paper_trading_manifest_hashes_unchanged` |
| `scripts/strategy_explore.py`, `strategy_style_book.py`, `strategy_style_robust.py` | WP-A evidence (`EXPLORE.md`, `STYLE_BOOK.md`, `STYLE_ROBUST.md`) | — |

## To build next, in priority order

1. **`src/hypershift/data/norgate.py`** (after the trial). Export the point-in-time S&P 500 membership, prices as traded, adjustment factors, delisting dates and returns into the existing panel format (`features`, `mask`, `dates`, `tickers`, `sectors`) plus a `member[N, T]` array.
   - **Tests:**
     - delisted names keep their final return; there is no forward-fill past delisting;
     - membership on a date matches the vendor's list;
     - truncation invariance of every feature;
     - no adjusted-level features (dollar volume from as-traded prices).
2. **`study.load_data(source='norgate')`.** Point-in-time eligibility as primary; the cap-weighted benchmark where market cap is available.
3. **Re-run `strategy_explore.py` and `strategy_style_robust.py` unchanged** on Norgate data. This is the decisive WP-A check.
4. **`src/hypershift/data/edgar.py`** (only if WP-B proceeds). companyfacts download, an as-first-reported table keyed by (CIK, concept, period, filed), CIK↔ticker history, and a quarterly diluted EPS series.
   - **Tests:**
     - a value is visible only after its `filed` date;
     - a restated value never overwrites the as-first-reported table;
     - SUE uses only filings dated on or before t.
5. **`scripts/paper_trade_ops.md` + a scheduled task.** Weekly: extend data, run `paper_trade.py`, archive inputs and outputs with sha256, and recompute the paper P&L from logged targets.
6. **Holdout runner** (`scripts/strategy_holdout.py`). One-shot, refuses to run unless the spec manifest hash matches, and writes the result once.
