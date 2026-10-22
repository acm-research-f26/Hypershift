# Study BC — leak-free backcast on uninspected 2004–2012 history (proposed specification)

**Status:** proposed on 2026-10-22 by the FB audit. It is **not yet frozen**.

- **Freeze requirement.** It becomes binding only when committed together with its code and the code hashes. That must happen **before any 2000–2012 price is downloaded beyond the coverage count below**.
- **Label:** quasi-confirmatory.
  - The data have never been used by any HyperShift artefact. The RSR panels start 2013-01-01 and the S&P 500 panel starts 2015-01-02.
  - However, the universe is survivor-biased, and the researchers have general knowledge of 2004–2012 market history.

## Why this study and not another

The audit leaves one suggestive positive finding and no feasible way to confirm it prospectively.

- **The finding.** Style-residual factor-neutral RankIC of about +0.011 for leak-free residual LightGBM.
  - Dev CI: [+0.003, +0.019] (A2c).
  - Selection-adjusted range: +0.007 to +0.010 (A3).
- **Why prospective confirmation is infeasible.**
  - The frozen v1 test has about 7% power at θ ≈ 0.006 (A1).
  - Any prospective test needs 1,500–2,500 days (6–10 years) for an MDE of 0.008–0.010 (A2c).
- **What this study offers.** Nine untouched years give about 2,250 days at once, for one day of engineering and about an hour of CPU. That is enough to detect or rule out the development-sized effect.
- **Why not the alternatives:**
  - **More architectures or tuning** cannot change a decision at this power.
  - **A matched raw-vs-residual prospective** would need about 18 years.
  - **Point-in-time fundamentals** are the right next step only if this study finds skill, or if price-based work is abandoned (roadmap P2).

## Hypotheses

**H1 (primary).** The frozen FB model class has positive stock-specific predictive skill on 2004–2012. The model is:
- LightGBM with `LGBM_PARAMS`, 50 rounds, seed 7;
- 20 FB features with **leak-free** dollar volume, per-date z-scored;
- the FB residual target, using the leak-free regressor;
- retrained annually, walk-forward.

The primary metric is the pooled mean of 9 fold means of the daily **style-residual neutral RankIC**. That is the Spearman correlation between predictions and the 5-day factor-neutral return after a per-date OLS on sector dummies plus z(beta, log_dvol_20, ret_60, vol_60), all measured at t with leak-free inputs.

**Falsification.** H1 fails if the 95% CI upper bound is below the threshold δ = 0.008.

### Predeclared secondaries (Holm over the four, descriptive unless stated)

- **S1.** Residual − raw target, both at 50 rounds, on FB neutral RankIC. Tests the "target lever" without the leak.
- **S2.** H1 model − reversal16, on the style-residual metric.
- **S3.** A **residual-reversal scorer**, with zero fitted parameters, vs 0 on the style-residual metric. This is the literature-backed signal FB never tested (Blitz et al. 2013; Da, Liu & Schaumburg 2014).
  - Score = −(5-day and 20-day returns, each residualised per date on sector dummies + z(beta, log_dvol_20, ret_60, vol_60)), averaged and divided by the 60-day residual volatility.
- **S4.** The H1 model at next-close execution (label close t+1 → close t+6).

### Descriptive only

- Buffered market-neutral book (A3 rule D plus a hold buffer: enter the top/bottom decile, exit outside the top/bottom 30%), with net Sharpe at 2, 5 and 10 bp per unit of turnover.
- Results per year.
- A point-in-time S&P 500 membership filter (source to be verified at implementation; if none is reliable, skip it and say so).

## Design

### Data

yfinance `auto_adjust=False` download of Close, Adj Close, Volume, High, Low, Open and Dividends for the 462 current tickers, 2000-01-01..2013-01-10.

- **Total returns** come from Adj Close ratios.
- **Dollar volume** = unadjusted Close × Volume. This is leak-free, because the split factor cancels in Close × Volume.
- **High-low range** uses unadjusted High/Low/Close, a ratio.
- **beta**: trailing 126 days against the equal-weight market.
- **Sectors**: current GICS. This is a disclosed limitation.
- **Eligibility**: 64 sessions, as in FB.
- **Coverage** (checked 2026-10-22; first available quarter only, no prices read): 373 of 462 tickers by 2004, 389 by 2006, 427 by 2011.

### Folds

- Evaluation years 2004–2012: 9 folds.
- For year Y, train on all signals whose 5-day label matures before Y-01-01, starting from 2000 after the 64-day warm-up.
- Fixed hyperparameters with no validation step. Hyperparameters, features and target are exactly as frozen in FB v1, with dollar volume corrected. There is no new selection.

### Statistics

- Daily series per fold, stationary bootstrap (block 10, B 5000), pooled = mean of fold means.
- Also reported:
  - fold-level t (8 df);
  - block 20 and Newey–West 10 sensitivity;
  - non-overlapping subsample spread.

### Controls

- **Negative controls:**
  - a random score, expected about 0;
  - the H1 model trained on labels permuted across stocks within each date, expected about 0. This checks the pipeline cannot manufacture IC.
- **Leak guards** (unit tests, written first):
  - every feature at t equals its value computed from data truncated at t;
  - dollar volume never uses Adj Close;
  - the scorer refuses dates ≥ 2013-01-01.
- **Positive control:** a planted-signal panel (FB's synthetic generator) recovers IC > 0.05.

### Power

From the A2c long-run variance (LRV 0.0201 per day, dev 2019–2023):

| Quantity | Value |
|---|---|
| SE at about 2,250 days | 0.0030 |
| MDE (80%, α .05) | 0.0084 |
| P(detect) at θ = 0.011 (raw dev) | 0.96 |
| P(detect) at θ = 0.008 | 0.76 |
| P(detect) at θ = 0.0068 (dev, shrunk) | 0.62 |
| P(detect) at θ = 0.005 | 0.39 |

The LRV may differ in 2004–2012, with 2008 driving it up. Report the realised LRV.

### Success criterion and stop/go

| Outcome | Rule | Decision |
|---|---|---|
| **GO** | H1 SUPPORTED (p < .05, CI excludes 0), point estimate ≥ 0.008, **and** S4 retains ≥ 50% | Continue price-based research. Next: a cost-aware buffered neutral-book prospective test (v2, own protocol), plus a point-in-time universe with delisting returns (P2) |
| **NO-GO** | 95% upper bound < 0.008 | Rules out the development-sized stock-specific effect in an independent period. Stop price-only large-cap prediction research. Write up the methodology and move to new data (roadmap) |
| **INCONCLUSIVE** | Anything else | Treated as NO-GO for resource allocation. The effect would be too small to trade after costs (A3 break-even analysis) even if real |

**Interpreting failure:**
- **Survivor bias could cut either way.** Survivors had better future fundamentals, which mostly affects style exposure, not the style-residual metric. A NO-GO is therefore informative about large-cap survivor-universe skill. It says nothing about small caps or point-in-time universes.
- **A GO on a survivor universe** still needs the point-in-time replication before any capital claim.

## Implementation plan (to be written as a superpowers plan at freeze)

1. `src/hypershift/backcast/data.py`:
   - leak-free feature builder (FB formulas; dollar volume from unadjusted Close × Volume);
   - guards: refuses dates ≥ 2013-01-01, and raises if Adj Close enters a level feature.

   Tests first, including the truncation-invariance leak test.
2. `scripts/backcast_fetch.py`: download and hash the raw data, written to `data/backcast/`, git-ignored. **Run only after the freeze commit.**
3. `scripts/backcast_run.py`:
   - 9 folds × {H1 model, raw-target twin, permuted-label control} = 27 fixed-round LightGBM fits;
   - plus the scorers (reversal16, residual-reversal, random).
4. `scripts/backcast_analysis.py`: the predeclared table, with verdicts computed by code from the rule above.
5. `FREEZE_MANIFEST_BC.json`: spec, thresholds and code sha256, committed before step 2.

**Cost:** about 1 day of engineering and about 1 h of CPU. No GPU.

> **Superseded (2026-10-22):** after the pre-execution review the decision is **STOP**; see `BC_REVIEW.md`. The corrected conditional design is `STUDY_BC_SPEC_v2.md`.
