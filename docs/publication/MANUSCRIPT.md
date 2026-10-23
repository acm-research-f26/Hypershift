# When Stock-Ranking Models Look Skilful: Optimisation Failure, Adjusted-Price Look-Ahead, Characteristic Exposure and Power in a Reproduction of a Hypergraph Stock Ranker

*Draft 2, 2026-10-22. Changes from draft 1 are listed in `MANUSCRIPT_CHANGES.md`. Bracketed IDs (e.g. [T3]) refer to `CLAIMS_REGISTER.md`. All empirical results are exploratory: every evaluation year had been inspected before. Mathematical definitions are in `APPENDIX.md`.*

---

## Abstract

We reimplemented THINK, a hyperbolic temporal-hypergraph stock ranker (Agarwal et al., ICDM 2022), from its paper; its code repository is empty. We evaluated it, together with simple benchmarks, under protocols that select models on validation data. We report five mechanisms that can make such models appear skilful, each with a diagnostic.

1. **Optimisation failure.** Coupled L2 weight decay under Adam collapsed the stacked hyperbolic layers, and the model recovered about 27% of a planted signal. Removing the decay raised this to 63–84%.
2. **Protocol choices.** In our protocol matrix, published-scale values appear only under test-informed epoch selection, an annualised Sharpe, or an NDCG implementation that random predictors also pass.
3. **Look-ahead in back-adjusted price levels.** Dividend back-adjustment multiplies each historical price by a factor that depends on later distributions *and on later prices*.
   - **Returns** computed from adjusted prices remain valid total returns.
   - **Level features** such as dollar volume do not: they carry this ex-post factor.
   - **In our sample,** the factor's future-price-path component has a rank IC of +0.094 with forward 5-day factor-neutral returns. The full factor has +0.019.
   - **In our benchmark,** correcting the features changed two conclusions.
4. **Exposure to known characteristics.** A learner given only four characteristics plus sector matched a 20-feature learner on beta- and sector-neutral rank IC. Residualising the evaluation target on those characteristics does not remove predictability from their nonlinear functions and interactions.
5. **Limited power.**
   - *Statistical:* the minimum detectable effect over five years is about 0.014–0.019 rank IC.
   - *Learner:* in one synthetic design, the gradient-boosting configuration we used did not exploit a planted incremental feature.

**On the reproduction itself.** After corrections, THINK and its Euclidean ablation were statistically indistinguishable, and neither beat simple baselines out of sample.

**What we do not claim.** We do not claim that returns are unpredictable, and we do not claim to have identified the original authors' procedures.

---

## 1. Introduction

Graph and hypergraph neural networks for cross-sectional stock ranking report gains over sequential and relational baselines. Examples are RSR (Feng et al., 2019), STHAN-SR (Sawhney et al., 2021) and THINK (Agarwal et al., 2022). Such claims are hard to verify:
- code is often incomplete;
- evaluation conventions vary;
- test periods are short;
- seed-to-seed variation can exceed the reported gains.

We set out to reproduce THINK. When its advantage did not appear, we asked which identifiable mechanisms can produce apparently skilful results in this kind of study, and how large each is.

**Research questions:**
1. Is THINK's reported advantage reproduced when models are selected on validation data, and under which protocol readings does it appear?
2. Which optimisation, data and evaluation mechanisms can generate or mask apparent skill?
3. Do factor-neutral metrics measure stock-specific information, or exposure to known characteristics?
4. Do matched model comparisons here have the statistical and learner power to answer feature-information questions?
5. Is any remaining signal economically meaningful after turnover control and costs?

**Contributions:**
- **C1. A quantified demonstration of look-ahead in back-adjusted price levels** [L1–L3]. The dependence of back-adjusted prices on later distributions is known among practitioners (§2). We:
  - make explicit that adjusted *levels* also depend on later *prices*, and exactly which quantities cancel in returns;
  - measure the predictive association of the adjustment term and of its future-price component in a standard data source;
  - show that the contamination altered conclusions of an ML benchmark.
- **C2. Metric interpretation** [S1–S4]. Beta- and sector-neutral rank IC rewards characteristic exposure. Residualising the target does not isolate information beyond characteristics for nonlinear learners. We propose a matched characteristics-only learner as a control.
- **C3. Learner power as a precondition for feature-ablation claims** [M1–M3]. This is preliminary: one synthetic design.
- **C4. An optimisation failure in stacked hyperbolic layers** [T2] (cf. Loshchilov & Hutter, 2019).
- **C5. A reproduction case study** [T1, T3–T8].

**Scope:**
- All development results are exploratory, and all universes are survivor-conditioned (§3, `LEAKAGE_AUDIT.md`).
- One prospective test is frozen and unscored (§11).

## 2. Related literature

**Adjusted prices, data revisions and look-ahead:**
- Practitioner sources describe how backward (CRSP-style) adjustment makes past adjusted prices depend on later events. They recommend forward or "past-only" adjustment, or as-traded prices, for signal construction (e.g. the Portfolio Optimizer note on adjusted prices without look-ahead bias; backtesting-software documentation).
- Vendor histories also change through revisions: Ljungqvist, Malloy and Marston (2009) document non-random retroactive changes to I/B/E/S recommendations that alter back-test results.
- Survivorship-conditioned samples can show spurious predictability (Brown, Goetzmann, Ibbotson & Ross, 1992).
- Kapoor and Narayanan (2023) catalogue leakage in ML-based science.
- We found no peer-reviewed study that quantifies the predictive association of the adjustment term with forward returns, or its effect on an ML benchmark (limited search; `PRIOR_ART.md`).

**Backtest overfitting and multiple testing.**
- White (2000) and Hansen (2005) test the best of many models against a benchmark.
- Bailey and López de Prado (2014) deflate Sharpe ratios for the number of trials.
- Bailey, Borwein, López de Prado and Zhu (2017) estimate the probability of backtest overfitting.
- Harvey, Liu and Zhu (2016) argue for t > 3.
- Hou, Xue and Zhang (2020) find that most published anomalies fail to replicate.

**ML in empirical asset pricing.**
- Gu, Kelly and Xiu (2020) find nonlinear learners improve monthly prediction with about 94 firm characteristics.
- Avramov, Cheng and Metzker (2023) find such gains concentrate in hard-to-arbitrage stocks and shrink under economic restrictions.
- Our setting (large-cap, price/volume only, 5-day horizon) is where both papers lead us to expect small gains.

**Short-term reversal and costs.**
- Jegadeesh (1990) and Lehmann (1990) document short-horizon reversal.
- Lo and MacKinlay (1990) show contrarian profits can arise from cross-autocovariances, not only overreaction.
- Residual reversal improves on raw reversal (Blitz, Huij, Lansdorp & Verbeek, 2013; Da, Liu & Schaumburg, 2014).
- Nagel (2012) relates reversal to liquidity provision.
- Novy-Marx and Velikov (2016) show high-turnover anomalies rarely survive costs, and recommend buffers.

**Model-comparison inference.**
- Diebold and Mariano (1995) test equal predictive accuracy.
- Giacomini and White (2006) test conditional predictive ability.
- Clark and West (2007) adjust MSPE comparisons of nested models for the noise of estimating parameters that are zero under the null.

**Benchmark variance.** Bouthillier et al. (2021) show that ignoring sources of variance makes benchmark conclusions unreliable.

## 3. Data and setting

1. **RSR panels** (Feng et al., 2019): NYSE (1,737 stocks) and NASDAQ, 2013–2017, with industry and Wikidata relations; 2017 is the test year (237 days).
   - The universe consists of stocks trading through the period (survivor-conditioned).
   - The price-adjustment convention of the source files is not documented by us.
2. **NYSE extension** for the RSR stocks, 2018–2023: Alpaca daily bars, split-adjusted only, so returns exclude dividends. Stocks are masked after their last bar.
3. **S&P 500 panel:** 462 current constituents, 2015–2023, current GICS sectors.
   - Prices: yfinance `auto_adjust` (backward dividend and split adjustment).
   - For the adjustment analysis we also downloaded unadjusted (split-only) closes for 2014-06 to 2024-01 (fetch date 2026-10-22).
   - Folds: annual walk-forward 2019–2023; the target is the 5-day return.

**Information-timing status** of each experiment, across selection, normalisation, features, labels and universe, is tabulated in `LEAKAGE_AUDIT.md`. No experiment satisfies all five: the universes are survivor-conditioned throughout.

## 4. Reproduction methodology

- **Model.** Reimplemented from the paper's equations, with every unstated detail labelled as inferred.
- **Arms.** THINK, its Euclidean-temporal ablation EH, Euclidean and no-graph arms, RSR-I and STHGCN.
- **Primary metric.** *Validation-selected* top-5 Sharpe (the epoch with the best validation Sharpe).
- **Diagnostic only.** The *test-selected* Sharpe (the best epoch on the test year).
- **Normalisation is stated per run:**
  - *full-series*: each stock divided by its maximum close over the whole sample, including the test year;
  - *train-period*: divided by the maximum over the training period.
- **Controls:**
  - hold-all equal-weight portfolio;
  - random, beta-matched and industry-matched top-5 baskets;
  - label permutation;
  - shuffled training labels;
  - planted signals.
- **Statistics.** Paired Wilcoxon over seeds with Holm correction, and a stationary bootstrap over dates. Definitions are in `APPENDIX.md`.

## 5. Evaluation and optimisation failures

### 5.1 Coupled weight decay [T2]

With Adam (lr 1e-3) and coupled L2 decay of 5e-4, the decay gradient was 10–40× the loss gradient at initialisation. The HNN++-style Poincaré layer output scales with the norm of its input, so three stacked layers shrank multiplicatively: prediction SD fell from 2.5e-4 to about 4e-12 within 200 steps.

**Planted-signal test** (309 stocks, real graph, oracle IC 0.279):

| Setting | THINK test IC |
|---|---|
| Coupled decay | 0.074 |
| Decay removed | 0.174 |

These are 27% and 63% of the oracle; the best configuration reaches 84%. The setting matches the authors' earlier public code. The THINK paper does not state its optimiser, so we make no claim about it.

### 5.2 Protocol choices [T1, T4, T5, T8]

- **Original setup** (coupled decay, level inputs, **full-series normalisation**), 25 seeds: THINK's validation-selected 2017 Sharpe was 0.089, against the hold-all 1.531 and EH 0.383 (difference −0.29, CI [−1.09, +0.41]) [T1].
- **Test-selected values** reach about 2.1, but shuffled training labels reach the same level (2.50 vs 2.46) [T5].
- **Across 88 protocol readings,** the published ordering never holds with both gaps above 2 standard errors. Published-scale Sharpe values require annualisation, and an NDCG near 0.86 appears only with an implementation that 43% of random predictors also reach [T5].
- **Full-series normalisation leaks future price scale:** a static rank on that factor alone scores Sharpe 4.37 in 2017 [T8].
- **These results bound what the paper's numbers could reflect; they do not identify it.**

### 5.3 After correction [T3, T4, T6, T7]

- **Corrected setup, 10 paired seeds:** with decay removed and relative inputs, THINK and EH were statistically indistinguishable in 2017 [T3]:

  | Normalisation | THINK | EH |
  |---|---|---|
  | Train-period | 1.687 | 1.436 |
  | Full-series | 1.929 | 1.146 |

- **The 2017 Sharpe level:** Sharpe near 2 in that strong year sat at the 88th–93rd percentile of random daily top-5 baskets [T4].
- **Out of sample, 2018–2023:** split-adjusted prices, train-period normalisation, RSR-defined universe. The model tracked a high-beta version of the market, with IC near zero [T6].
- **The RSR authors' public code** reproduced its paper's own metrics, but showed no skill under a validation-selected top-5 Sharpe [T7].

## 6. Look-ahead in back-adjusted price levels [L1–L3]

**Mechanism** (full derivation in `MATH_ADJUSTMENT.md`). With the backward multiplicative convention:

P_adj(t) = P(t) · F(t),  F(t) = ∏_{t<e≤T_f} (1 − y_e),  y_e = D_e / P(e−1).

Here e ranges over ex-dates, D_e is the distribution, P(e−1) is the cum-distribution close, and T_f is the fetch date.

**Returns.** The adjusted gross return over t → t+h is

  P_adj(t+h) / P_adj(t) = [P(t+h) / P(t)] · ∏_{t<e≤t+h} 1/(1 − y_e).

- Factors for ex-dates **after** the window cancel.
- Factors for ex-dates **inside** the window remain. They are what make the ratio a total return: the distribution is treated as reinvested at the theoretical ex-price P(e−1) − D_e.
- Backward-looking return features use only events up to t, so they are causal. Forward labels use only events inside their own window, which is legitimate.

**Levels.** Any feature using adjusted *levels* keeps F(t):

  log(P_adj · V) = log(P · V) + L(t),  L(t) = log F(t).

L(t) is a function of every later distribution amount, every later pre-ex-date price, and the fetch date. Holding later amounts fixed, ∂L / ∂ log P(e−1) ≈ y_e, so higher later prices move L toward zero. Later prices include the label window, so L is algebraically linked to the label.

**Measurement** (462 stocks, 2019–2023 signals, 5-day factor-neutral returns):
- **Dividing L into two parts.** We split L into L_div and L_path (`MATH_ADJUSTMENT.md` §4):
  - **L_div:** the realised later distribution amounts, valued at today's price;
  - **L_path:** the remainder, which exists only because those amounts are divided by later prices.
- **Data used.** The unadjusted download covers 12,989 distribution events.

| Score | Rank IC [95% CI] | Notes |
|---|---|---|
| L | +0.019 [+0.008, +0.030] | |
| L_path | **+0.094 [+0.086, +0.101]** | Positive in all five years; small variance (cross-sectional SD 0.022) |
| L_div | +0.010 [−0.002, +0.021] | Dominates L (rank correlation 0.99) |
| Trailing distribution yield (known at t) | −0.009 (CI includes 0) | |

A per-date regression residual of L on trailing yield scored +0.031. That residual mixes ex-ante information not spanned by the proxy with ex-post information. **It is not an identified "future-only" component,** and we report it only as an association.

**What this establishes:**
- **(i) Mathematically:** adjusted-level features contain a function of post-t prices and distributions.
- **(ii) Empirically, in this sample:** the future-price component of that function is strongly associated with forward returns.
- **(iii) The size of the contamination:** small in variance, which is why the full term's association is modest (rank correlation 0.995 between adjusted and corrected dollar volume).

None of this is a causal estimate of how much any model exploited the contamination. Magnitudes are specific to this vendor, universe and fetch date.

**Effect on our own benchmark** [L2]. Recomputing dollar volume from unadjusted closes:
- the apparent gain from residual-return targets fell from +0.014 to +0.009, and its CI now includes 0;
- an apparent harm from volume features (−0.009, negative in 5 of 5 years) vanished (−0.003).

## 7. Characteristic exposure and metric interpretation [S1–S4]

Four phenomena must be kept apart:
1. **Look-ahead:** information unavailable at prediction time (§6).
2. **Incomplete adjustment:** a metric that claims to remove characteristics but does not.
3. **Legitimate predictability from known characteristics.**
4. **Incremental information beyond characteristics.**

**Neutral rank IC** (5-day return minus beta × market, sector-demeaned) removes only beta and sector. Using adjustment-corrected inputs:

| Learner | Neutral rank IC |
|---|---|
| Four characteristics (beta, log dollar volume, 60-day return, 60-day volatility) + sector | +0.026 |
| All 20 features, neutral target | +0.025 |
| All 20 features, residual target | +0.013 |

This is phenomenon 3 [S1].

**Residualised metrics.** Residualising the target per date on the characteristics, linearly or with quintile dummies, leaves the characteristics-only learner with most of its score:
- synthetic data with purely characteristic-driven returns: about 0.073 of 0.079;
- real data: +0.009 [+0.003, +0.015] linear and +0.006 [+0.001, +0.010] quintile [S2].

This is phenomenon 2. Additive residualisation removes only the span of its regressors, and a tree learns functions outside it. That this explains the real-data placebo is inferred from the synthetic result.

**Further results:**
- The neutral target raised neutral rank IC by about +0.011 for both feature sets [S3].
- Most of the residual-target gain over a raw target was reproduced by neutralising raw predictions ex ante [S4].

## 8. Matched comparisons and learner power [M1–M4]

### 8.1 Fitted-procedure comparison [M1]

**Design: a 2×2.**
- Features: all 20 vs 4 characteristics, both plus sector.
- Target: residual vs neutral.
- Everything else matched: learner, 50 rounds, three seeds, windows, evaluation dates and preprocessing.

| Target | Full − characteristics only [95% CI] |
|---|---|
| Residual | −0.0017 [−0.012, +0.009] |
| Neutral | −0.0011 [−0.010, +0.008] |

As a comparison of these fitted procedures, this is valid: the full-feature procedure did not outperform. Increments of about ±0.01 cannot be excluded.

### 8.2 Not a test of feature information [M2, M3]

**One synthetic design** (400 stocks, 500 evaluation dates, returns nonlinear in four characteristics, with or without a planted feature):
- the matched LightGBM increment was zero or negative when the extra features were uninformative (−0.003 to 0.000; two of three replicates had CIs below zero);
- it did not detect a planted feature with standalone IC 0.046 (−0.004 at 50 rounds; +0.005 at 300 rounds; single-seed probe);
- Ridge recovered the planted feature (+0.038) but not the nonlinear signal;
- linear rank encompassing gave false positives;
- a within-bin permutation control was not validated.

**What this shows.** The configuration's matched nulls are uninformative about feature information at comparable effect sizes. It does not show that gradient boosting in general lacks power.

### 8.3 Nested-model inference [M4]

Clark and West (2007) derive their MSPE adjustment for nested models. An analogous penalty for rank IC is consistent with §8.2, but applying Clark–West to Spearman correlation is an unvalidated adaptation. It would need:
- (i) a derivation or simulation of the null distribution under cross-sectional dependence and overlapping labels;
- (ii) size and power under realistic factor structure;
- (iii) a definition of "extra parameters" for nonlinear learners.

We recommend MSPE-based nested tests and conditional predictive-ability tests (Giacomini & White, 2006) on a continuous loss, validated on synthetic designs, before use.

## 9. Portfolios and economics [E1, E2]

**Turnover.** Daily-rebalanced sector-neutral long/short books built from overlapping 5-day forecasts traded 36–77% of NAV per day. They broke even at 2–14 bp of linear cost per unit of turnover [E1].

**Buffered books.** A weekly book with a 10%/30% buffer cut turnover to about 0.13 per day. The simulator handles drift, next-close and next-open execution, and a costed equal-weight beta hedge; a random-score null centres at zero.

| Book (weekly, buffered) | Net Sharpe at 5 bp |
|---|---|
| Residual LightGBM | +0.56 [−0.32, +1.44] |
| Characteristics-only learner | +1.11 |

No model book outperformed its matched comparators: all paired CIs include zero [E2].

These are post-hoc results on inspected years:
- obtained after selection over arms, schedules and execution rules;
- under linear costs with no borrow, financing or impact;
- on a survivor universe.

We did not attribute the characteristics-only books' returns to factors, and we make no investability claim [UT4, UT5].

## 10. Statistical and data limitations [P1–P3, U1]

- **Power.**
  - Minimum detectable paired differences with five annual folds: about 0.014–0.019 [P1].
  - 77–89% of day-to-day IC variance exceeds the sampling floor [P2].
  - Long-run variance differs about 2.8× across years.
- **Selection.** Selecting the best of 17 arms inflated its estimate: leave-one-year-out +0.0098 vs +0.0127 in-sample. Best-of-pool vs reversal: p = 0.55 [P3].
- **Universe.** Survivor-conditioned throughout. Matching current constituents to a public point-in-time membership list covers about 43–58% of 2004–2012 index members [U1].
  - The figure is a symbol-match count, not a bound: renames lower it; ticker reuse and pre-buyout symbols raise it.
  - We therefore did not run a proposed 2004–2012 backcast.
- **Information timing.** See `LEAKAGE_AUDIT.md`. In particular, the original reproduction used full-series normalisation, and the RSR price-adjustment convention is unknown.
- **Independence.** The many runs share data, years and code; their number is not independent evidence.

## 11. Prospective test

We froze one specification (a residual-target LightGBM) and its comparators before any post-freeze data existed. An adjustment-corrected version (unadjusted dollar volume) was predeclared before any data, and is designated the primary scientific result. At 250 days the test has about 7% power at the development effect [P1], so a null result will be uninformative. No outcome is reported.

## 12. Recommendations

1. **Build level features from as-traded (split-only) prices.** Run two cheap checks:
   - a truncation-invariance test;
   - an *adjustment-term-as-score* test: compute log(P_adj / P) and check its association with your labels.
2. **Report validation-selected results,** and state the normalisation used. Show test-selected values only next to shuffled-label baselines.
3. **Test learnability on planted signals** before interpreting nulls.
4. **Report seed spread;** use paired tests.
5. **Report neutral metrics together with a matched characteristics-only learner.** Do not treat residualised-target metrics as characteristic-free.
6. **Validate any incremental-information test on synthetic data** with known increments. Report false-positive rate and power.
7. **Compute per-contrast power** before freezing confirmatory tests.
8. **Evaluate economics** with turnover control, realistic execution timing, a costed hedge and a random-score null.

## 13. Conclusion

In our setting, apparently skilful results traced to five identifiable mechanisms:
- test-informed selection and flawed metrics;
- an optimisation failure;
- look-ahead through back-adjusted price levels;
- exposure to known characteristics;
- comparisons too weak to detect the effects they targeted.

After controlling for these, the evidence could not identify stock-specific information beyond characteristics and simple reversal, and our tools could not rule it out. The adjustment-level mechanism is algebraic, and simple to detect with the tests in §12. Its empirical size here is specific to our data.

We stopped price-only modelling on this universe as a resource decision, not a claim of unpredictability.

## 14. References

(Verified by search on 2026-10-22; details from publisher or repository records.)

- Agarwal, S., Sawhney, R., Thakkar, M., Nakov, P., Han, J., & Derr, T. (2022). THINK: Temporal Hypergraph Hyperbolic Network. *IEEE ICDM 2022*, 849–854. doi:10.1109/ICDM54844.2022.00096.
- Avramov, D., Cheng, S., & Metzker, L. (2023). Machine learning vs. economic restrictions: Evidence from stock return predictability. *Management Science*, 69(5). doi:10.1287/mnsc.2022.4449.
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4).
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94–107.
- Blitz, D., Huij, J., Lansdorp, S., & Verbeek, M. (2013). Short-term residual reversal. *Journal of Financial Markets*, 16(3), 477–504.
- Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *MLSys 2021*.
- Brown, S. J., Goetzmann, W., Ibbotson, R. G., & Ross, S. A. (1992). Survivorship bias in performance studies. *Review of Financial Studies*, 5(4), 553–580.
- Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291–311.
- Da, Z., Liu, Q., & Schaumburg, E. (2014). A closer look at the short-term return reversal. *Management Science*, 60(3), 658–674.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13, 253–263.
- Feng, F., He, X., Wang, X., Luo, C., Liu, Y., & Chua, T.-S. (2019). Temporal relational ranking for stock prediction. *ACM TOIS*, 37(2), Article 27.
- Giacomini, R., & White, H. (2006). Tests of conditional predictive ability. *Econometrica*, 74(6), 1545–1578.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23, 365–380.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
- Hou, K., Xue, C., & Zhang, L. (2020). Replicating anomalies. *Review of Financial Studies*, 33(5), 2019–2133.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1), 1–28.
- Ljungqvist, A., Malloy, C., & Marston, F. (2009). Rewriting history. *Journal of Finance*, 64(4), 1935–1960.
- Lo, A. W., & MacKinlay, A. C. (1990). When are contrarian profits due to stock market overreaction? *Review of Financial Studies*, 3(2), 175–205.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.
- Nagel, S. (2012). Evaporating liquidity. *Review of Financial Studies*, 25(7), 2005–2039.
- Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their trading costs. *Review of Financial Studies*, 29(1), 104–147.
- Sawhney, R., Agarwal, S., Wadhwa, A., Derr, T., & Shah, R. R. (2021). Stock selection via spatiotemporal hypergraph attention network: A learning to rank approach. *AAAI 2021*, 35(1), 497–504.
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126.
- *Practitioner source (non-peer-reviewed):* Portfolio Optimizer (2022, updated 3 July). Adjusted prices without look-ahead bias. portfoliooptimizer.io/blog.
