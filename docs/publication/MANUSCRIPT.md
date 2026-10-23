# When Stock-Ranking Models Look Skilful: Optimisation Failure, Vendor-Adjustment Leakage, Style Exposure and Power in a Reproduction of a Hypergraph Stock Ranker

*Draft 1, 2026-10-22. Bracketed IDs (e.g. [T3]) refer to `CLAIMS_REGISTER.md`. Every empirical result in this draft is exploratory unless stated otherwise.*

---

## Abstract

We reimplemented THINK, a hyperbolic temporal-hypergraph stock ranker (Agarwal et al., ICDM 2022), from its paper, because its code repository is empty. We then evaluated it and a set of simple benchmarks under leak-free protocols. Throughout, we tried to identify which mechanisms can make such models look skilful.

**Five mechanisms, each documented with saved outputs and diagnostics:**
1. **Optimisation failure.** Coupled L2 weight decay under Adam collapsed the stacked hyperbolic layers. The model recovered about 27% of a planted signal; removing the decay raised this to 63–84%.
2. **Protocol choices.** Published-scale values in our protocol matrix appear only under test-informed epoch selection, an annualised Sharpe, or a flawed NDCG. Shuffled labels reach the same test-selected Sharpe as real labels.
3. **Leakage in vendor-adjusted prices.** Dividend back-adjustment multiplies historical prices by a factor that depends on future prices. Features built on adjusted price levels therefore carry future information: in our data, the future-only part of that factor predicts 5-day factor-neutral returns with a rank IC of +0.031.
4. **Exposure to known characteristics.** A learner given only four style characteristics and sector matched or exceeded a 20-feature learner on beta- and sector-neutral rank IC. Residualising the evaluation target on those characteristics, linearly or by quintile, does not remove predictability from their nonlinear functions and interactions.
5. **Limited power, of two kinds.**
   - *Statistical:* the minimum detectable effect over five years is about 0.014–0.019 rank IC.
   - *Learner:* the gradient-boosting configuration we used did not exploit a planted incremental feature with standalone IC 0.046 in a synthetic test, so matched "features do not help" nulls from it are uninformative.

**Findings on the reproduction itself.** After correcting the optimisation and the inputs, THINK and its Euclidean ablation were statistically indistinguishable, and neither beat simple baselines out of sample.

**What we do not claim.** We do not claim that stock returns are unpredictable, and we do not claim to have identified the original authors' procedures. We argue that, in this setting, the evidence available cannot identify stock-specific information beyond known characteristics and simple reversal, and we show how each mechanism can be diagnosed.

---

## 1. Introduction

Graph and hypergraph neural networks for cross-sectional stock ranking report sizeable gains over sequential and relational baselines. Examples are RSR (Feng et al., 2019), STHAN-SR (Sawhney et al., 2021) and THINK (Agarwal et al., 2022). These claims are hard to check:
- code is often incomplete;
- evaluation conventions vary (top-k Sharpe, NDCG, cumulative return);
- test periods are short, often a single year;
- the target is noisy enough that seed-to-seed spread can exceed the reported gains.

We set out to reproduce THINK. When the published advantage did not appear, we asked a broader question: **which identifiable mechanisms can produce apparently skilful results in this kind of study, and how large is each?** Answering it required a series of studies:
- a fidelity audit;
- out-of-sample tests;
- a simple-model feature benchmark on a second dataset;
- an independent statistical audit;
- a pre-execution review of a proposed historical backcast;
- a matched-control verification.

**Research questions:**
1. Can THINK's reported advantage be reproduced under leak-free evaluation, and if not, under which protocol readings does it appear?
2. Which optimisation, data and evaluation mechanisms can generate or mask apparent skill?
3. Do common "factor-neutral" metrics measure stock-specific information, or exposure to known characteristics?
4. Do matched model comparisons in this setting have enough statistical and learner power to answer feature-information questions?
5. Is any remaining predictive signal economically meaningful after turnover control and costs?

**Contributions,** ordered by how generalisable we judge them:
- **C1. Vendor-adjustment leakage, with a mechanism and a size** [L1, L2]. Back-adjusted price levels encode future prices. We decompose the adjustment factor into a part known at time t and a future-only part, and show that the latter predicts returns.
- **C2. Metric interpretation** [S1–S4]. Beta- and sector-neutral RankIC rewards characteristic exposure. Residualised metrics do not isolate information beyond characteristics when learners are nonlinear. A matched learner restricted to characteristics is a necessary control.
- **C3. Learner power as a precondition for feature-ablation claims** [M1–M3]. A matched comparison is a valid comparison of fitted procedures, but it is only informative about feature information if the learner is shown to exploit increments of the relevant size.
- **C4. An optimisation failure specific to stacked hyperbolic layers** [T2]: coupled weight decay under Adam (cf. Loshchilov & Hutter, 2019).
- **C5. A reproduction case study** [T1, T3–T8] with explicit protocol readings and nulls.

**Scope.** Every development result is exploratory: all evaluation years had been inspected. One prospective test is frozen and pending (§11). The universe is current S&P 500 constituents plus the RSR NYSE/NASDAQ panels, and survivorship applies throughout.

## 2. Related literature

- **Backtest overfitting and multiple testing.** Testing many strategies and choosing the best inflates performance. White's (2000) Reality Check and Hansen's (2005) SPA test the best of many models against a benchmark. Bailey and López de Prado (2014) deflate Sharpe ratios for the number of trials. Bailey, Borwein, López de Prado and Zhu (2017) estimate the probability of backtest overfitting by combinatorially symmetric cross-validation. Harvey, Liu and Zhu (2016) argue that new factors should clear t > 3. Hou, Xue and Zhang (2020) find that most published anomalies fail to replicate under value weighting and stricter hurdles. We apply SPA, RC and CSCV to our own selection step [P3].
- **ML in empirical asset pricing.** Gu, Kelly and Xiu (2020) find that trees and neural networks improve monthly return prediction over linear models using about 94 firm characteristics, mainly through nonlinear interactions. Avramov, Cheng and Metzker (2023) show that such ML gains concentrate in hard-to-arbitrage stocks and shrink under economic restrictions and costs. Our setting (large-cap, price/volume only, 5-day horizon) is the one where both papers lead us to expect small gains. Our results agree with that expectation; they do not test GKX's setting.
- **Short-term reversal.** Short-horizon reversal is long documented (Jegadeesh, 1990; Lehmann, 1990). Lo and MacKinlay (1990) attribute part of it to cross-autocorrelation. Residual reversal ranks on factor-model residual returns and is reported to roughly double risk-adjusted profits (Blitz, Huij, Lansdorp & Verbeek, 2013); Da, Liu and Schaumburg (2014) isolate a within-industry non-fundamental component. Nagel (2012) relates reversal profits to liquidity provision. We use a zero-parameter residual reversal as a benchmark.
- **Costs.** Novy-Marx and Velikov (2016) show that high-turnover anomalies rarely survive costs, and that a buy/hold buffer is the most effective simple mitigation. We apply such a buffer [E1].
- **Survivorship.** Brown, Goetzmann, Ibbotson and Ross (1992) show that survivorship-truncated samples can display spurious predictability. Our universes are survivor-conditioned (§4, [U1]).
- **Model-comparison inference.** Diebold and Mariano (1995) test equal predictive accuracy. Giacomini and White (2006) test conditional predictive ability. Clark and West (2007) adjust MSPE comparisons of nested models for the noise that estimating extra parameters adds under the null. We discuss whether that logic extends to rank-correlation metrics (§9.3).
- **Leakage and benchmark variance in ML.** Kapoor and Narayanan (2023) catalogue leakage across 17 fields and 294 papers and show that it can create spurious advantages for complex models. Bouthillier et al. (2021) show that ignoring variance from data sampling, initialisation and hyperparameters makes benchmark conclusions unreliable. Our seed-to-seed Sharpe spreads (0.3–1.5) exemplify the latter.
- **What is new here.** The individual concerns are known. Our contributions are:
  - **(a)** a concrete, quantified leakage channel in widely used adjusted-price data, with a decomposition separating its ex-ante and future components;
  - **(b)** a demonstration that factor-residualised RankIC does not isolate information beyond characteristics for nonlinear learners, with a matched-learner control;
  - **(c)** evidence that a matched null can be uninformative because of learner power, which feature-ablation studies rarely check;
  - **(d)** a reproduction that maps protocol readings to outcomes.

## 3. Data and setting

1. **RSR panels** (Feng et al., 2019). NYSE (1,737 stocks) and NASDAQ (1,026), 2013–2017, with the authors' industry and Wikidata relations. Test year 2017 (237 days). We rebuilt the hypergraph from the THINK paper's appendix.
2. **Out-of-sample NYSE extension.** Alpaca daily bars (split-adjusted), 2018–2023, 1,647 of the RSR stocks [T6].
3. **S&P 500 panel.** 462 current constituents, daily yfinance bars (`auto_adjust`), 2015–2023 for development, with current GICS sectors. Annual walk-forward folds 2019–2023: train on labels maturing before Y−1, validate on Y−1, test on Y. The target is the 5-day return.

**Inspection status.** All 2015–2023 data have been inspected; 2024 onward was used once, for a separate zero-shot test. A 2004–2012 backcast was proposed and **not** run (§10).

## 4. Reproduction methodology

- **Model.** Reimplemented from the paper's equations, with every unstated detail labelled as inferred: hyperparameters, loss form, attention normalisation, and the operator in its eq. 7.
- **Arms.** THINK (hyperbolic temporal and spatial), its Euclidean-temporal ablation EH (the paper's TConv+DHHAN), Euclidean and no-graph arms, and the RSR-I and STHGCN baselines.
- **Evaluation.**
  - **Leak-free:** the epoch is selected on validation Sharpe.
  - **Diagnostic only:** the "best-test" epoch, the maximum over epochs of test-year Sharpe.
  - **Sharpe:** daily top-5 equal-weight, annualised, no costs. The paper's printed formula has no annualisation, so the levels are not like-for-like [T5].
- **Controls:** hold-all equal-weight; random daily top-5 baskets; beta- and industry-matched baskets; label permutation; shuffled training labels; planted signals.
- **Statistics.** Paired Wilcoxon over seeds with Holm correction, and a stationary bootstrap over dates. Verdict words are predeclared.

## 5. Evaluation and optimisation failures

### 5.1 Optimisation failure under coupled weight decay [T2]

With Adam (lr 1e-3) and coupled L2 decay of 5e-4, the decay gradient on the hyperbolic layers was 10–40× the loss gradient at initialisation. Adam's normalisation then moves each weight toward zero at roughly a constant rate. Because the output of the HNN++-style Poincaré linear layer scales with ‖z‖, three stacked layers shrink multiplicatively, and the prediction SD fell from 2.5e-4 to about 4e-12 within 200 steps.

**Planted-signal test** (309 stocks, real graph):

| Setting | THINK test IC | Share of oracle IC (0.279) |
|---|---|---|
| Coupled decay | 0.074 | about 27% |
| Decay removed | 0.174 | 63% |
| Decay removed, best configuration | — | up to 84% |

This setting matches the authors' earlier public code. The THINK paper does not state its optimiser settings, so we do not claim THINK used them.

### 5.2 Protocol choices [T1, T4, T5, T8]

- **Original setup, 25 seeds:** THINK's validation-selected 2017 Sharpe was 0.089, below the hold-all market (1.531) and below EH (0.383; difference −0.29, CI [−1.09, +0.41]) [T1].
- **Best-test epoch:** THINK reaches 2.1 under test-year selection, but shuffled training labels reach the same value (2.50 vs 2.46) [T5].
- **The 88-cell protocol matrix** (selection × normalisation × Sharpe formula × k × NDCG):
  - the published THINK > EH > STHGCN ordering never holds with both gaps above 2 standard errors;
  - published-scale Sharpe values require annualisation;
  - an NDCG near 0.86 appears only with an index-set implementation under which 43% of random predictors score ≥ 0.86 [T5].
- **Full-series price normalisation** (dividing by each stock's maximum close over the whole period) leaks future price scale. A static ranking on that factor alone scores Sharpe 4.37 in 2017 [T8].
- **These results bound what the paper's numbers could reflect; they do not identify it.** The paper does not state its selection rule, and its code is unavailable.

### 5.3 After correction [T3, T4, T6, T7]

- **2017, corrected optimisation and inputs:** THINK and EH were statistically indistinguishable over 10 paired seeds (train-period normalisation: 1.687 vs 1.436; full-series: 1.929 vs 1.146) [T3]. Sharpe levels near 2 in that strong year sat at the 88th–93rd percentile of random daily top-5 baskets [T4].
- **Out of sample:** the frozen and annually retrained model tracked a high-beta version of the market with IC near zero (2018–2023: 0.43 vs hold-all 0.51; 2019–2023 walk-forward: 0.70 vs 0.69) [T6].
- **The RSR authors' public code** reproduced its own paper's metrics, but showed no skill under a validation-selected top-5 Sharpe [T7].

## 6. Vendor-adjustment leakage [L1, L2]

**Mechanism.** Data vendors back-adjust historical closes for later distributions:

P_adj(t) = P(t) · F(t), where F(t) = ∏ over ex-dates e > t of (1 − D_e / P_{e−1}).

F(t) depends on future dividends **and future prices** P_{e−1}. Returns computed from adjusted prices are correct total returns, because F cancels in ratios. *Level* features do not cancel: dollar volume P_adj · V and Amihud illiquidity |r| / (P_adj · V) carry log F(t). Since ∂ log F / ∂ log P_{e−1} ≈ the cumulative future yield, a stock that rises after t has log F closer to 0. That is an algebraic link to the label.

**Measurement.** We downloaded unadjusted closes and computed L(t) = log(P_adj / P), then split it into:
- **TY(t) = L(t) − L(t−252):** distributions already paid, known at t;
- **the residual of L on TY:** the future-only part.

**Results** (462 stocks, 2019–2023, 5-day factor-neutral returns):

| Score | Rank IC [95% CI] |
|---|---|
| L as a score | +0.019 |
| Trailing yield TY | −0.009 (CI includes 0) |
| Future-only part of L | **+0.031 [+0.024, +0.037]**, positive in all five years |

For comparison, no fitted model in our benchmark exceeded about 0.014 on the same metric. The per-date rank correlation between adjusted and corrected dollar volume was 0.995, so the contamination is invisible to casual inspection.

**Consequences in our own benchmark** [L2]:
- the apparent gain from training on residual returns fell from +0.014 to +0.009, and its CI now includes 0;
- an apparent harm from volume features (−0.009, negative in 5/5 years) vanished (−0.003).

The fix is to compute level features from unadjusted (split-adjusted) price × volume. Magnitudes depend on the vendor, the universe and the fetch date. The mechanism applies to any back-adjusted series.

## 7. Style exposure and metric interpretation [S1–S4]

Four phenomena must be kept apart:
1. **Leakage:** information unavailable at prediction time (§6).
2. **Incomplete statistical adjustment:** a metric that claims to remove characteristics but does not.
3. **Legitimate predictability attributable to known characteristics:** style premia and nonlinear characteristic effects.
4. **Incremental information beyond those characteristics.**

The common "factor-neutral" RankIC (5-day return minus beta × market, sector-demeaned) removes beta and sector only. A LightGBM given only beta, log dollar volume, 60-day return, 60-day volatility and sector scored +0.026 on it. The same learner with all 20 features scored +0.025 on the neutral target and +0.013 on a residual target [S1]. That is phenomenon 3, predictability from characteristics. It is not leakage, and it is not evidence of stock-specific information.

**Residualised metrics.** We residualised the target per date on the characteristics, linearly and with quintile dummies [S2]:
- **Synthetic data:** when returns depend only on nonlinear functions and interactions of characteristics, a characteristics-only learner keeps most of its score (about 0.073 of 0.079).
- **Real data:** the characteristics-only learner scored +0.009 [+0.003, +0.015] linear and +0.006 [+0.001, +0.010] quintile.

This is phenomenon 2. Additive residualisation is orthogonal only to the span of its regressors, while a tree learns functions outside that span (interactions, nonlinearities within quintiles), and a rank correlation does not remove them. For real data, the interaction explanation is inferred from the synthetic result.

**Practical reading:**
- Training on the neutral target raised neutral RankIC by about +0.011 for both feature sets [S3], consistent with steering the learner toward rewarded exposures.
- The residual-target gain over a raw target was mostly reproduced by neutralising the raw model's predictions ex ante [S4].

## 8. Matched-model comparisons and learner power [M1–M4]

### 8.1 A valid comparison of fitted procedures [M1]

Design (2×2): features (all 20 vs 4 characteristics, both plus sector) × target (residual vs neutral). Everything else is the same:
- learner, 50 rounds, three seeds;
- training windows, evaluation dates and preprocessing.

| Target | Full − characteristics only [95% CI] |
|---|---|
| Residual | −0.0017 [−0.012, +0.009] |
| Neutral | −0.0011 [−0.010, +0.008] |

The interaction was not detected. As a comparison of these fitted procedures, this is valid: the full-feature procedure did not outperform the characteristics-only procedure. Increments of about ±0.01 cannot be excluded.

### 8.2 It is not a test of feature information [M2, M3]

**Synthetic design.** 400 stocks, 500 evaluation dates, returns nonlinear in four characteristics, with or without a planted incremental feature. Results:
- The matched LightGBM increment was zero or *negative* when the extra features were uninformative (−0.003 to 0.000 across three replicates; two had CIs below zero).
- It did not detect a planted feature with standalone IC 0.046 (increment −0.004 at 50 rounds; +0.005 at 300 rounds).
- Ridge recovered the planted feature (+0.038) but not the nonlinear characteristic signal.
- Linear rank encompassing gave false positives under the null.
- A within-bin conditional permutation control neither avoided false positives reliably nor detected planted information.

These results come from one synthetic design: three replicates and a single-seed probe. They do not show that gradient boosting in general lacks power. They show that **this configuration's** matched nulls are uninformative about feature information at effect sizes like those of interest (about 0.01). A feature-ablation claim requires evidence that the learner can exploit increments of that size.

### 8.3 Nested-model inference [M4]

Clark and West (2007) show that, for nested models compared by MSPE, the larger model's forecasts are noisier under the null because it estimates parameters that are truly zero. They adjust the MSPE difference so that standard normal inference applies. Our negative increments under the uninformative-feature design are consistent with an analogous penalty for rank IC.

Clark–West's result is derived for MSPE, not for Spearman correlation. Applying it to RankIC would be an unvalidated adaptation. It would need:
- (i) a derivation, or simulation evidence, that the adjusted rank-IC difference has a known null distribution under cross-sectional dependence and overlapping labels;
- (ii) simulation of size and power under realistic factor structure;
- (iii) handling of nonlinear learners, where the "extra parameters" are not well defined.

Until then we recommend MSPE-based nested tests (Clark–West) and conditional predictive-ability tests (Giacomini & White, 2006) on a continuous loss, each validated on synthetic designs like ours before use.

## 9. Portfolio construction and economic implications [E1, E2]

**Turnover.** Daily-rebalanced sector-neutral long/short books built from overlapping 5-day forecasts traded 36–77% of NAV per day. They broke even at 2–14 bp of linear cost per unit of turnover [E1].

**Buffered books.** A weekly book with a 10%/30% entry/exit buffer cut turnover to about 0.13 per day. Our simulator handles weights that drift between trades, next-close and next-open execution, and an equal-weight beta hedge whose trades are costed; a random-score null centres at zero.

| Book (weekly, buffered) | Net Sharpe at 5 bp |
|---|---|
| Residual LightGBM | +0.56 [−0.32, +1.44] |
| Characteristics-only learner | +1.11, positive in all five years |

**No model book outperformed its matched comparators.** The comparators were residual reversal, a neutralised raw model, and the characteristics-only learners. All paired CIs include zero [E2].

These are post-hoc results on inspected years, after selection over arms, schedules and execution rules. The cost model is linear and has no borrow, financing or impact costs. The universe is survivor-conditioned. We did not perform a style-factor attribution of the characteristics-only books, so we do not claim their returns are explained by known premia [UT5]. None of these books is shown to be investable [UT4].

## 10. Statistical limitations [P1–P3, U1]

- **Power.**
  - With five annual folds, minimum detectable paired differences in neutral RankIC were about 0.014–0.019.
  - 77–89% of day-to-day IC variance exceeds the cross-sectional sampling floor, so more years, not more stocks, would raise power [P2].
  - The long-run variance differed about 2.8× across years [P1].
- **Selection.** Picking the best of 17 arms inflated its estimate: leave-one-year-out +0.0098 vs +0.0127 in-sample. A best-of-pool test against reversal had p = 0.55 [P3].
- **Survivorship and coverage.** Matching current constituents to public point-in-time membership lists by ticker covers about 43–58% of 2004–2012 index members [U1].
  - The figure is a symbol-match count, not a bound: renames lower it, while ticker reuse (e.g. a symbol reassigned to a different company) and pre-buyout symbols raise it.
  - Without delisting returns, a backcast could support only claims conditional on survival to 2026.
  - This, together with §8.2, is why we did not run it.
- **Independence.** The many runs share data, years and code. Their number is not independent evidence. Each verdict relies on paired, date-bootstrapped and multiplicity-adjusted tests.

## 11. A prospective test

We froze one specification (a residual-target LightGBM) and its comparators before any post-freeze data existed. Its stated limitations:
- **Power.** At 250 days it has about 7% power at the development effect size [P1], so a null result will be uninformative.
- **Leak-free version.** A version using unadjusted dollar volume (v1-LF) was predeclared and implemented before any post-freeze data. It is the primary scientific result; the original pipeline (v1) is the reproducibility record.

No prospective outcome is reported here.

## 12. Recommendations for researchers

1. **Check every level feature for vendor adjustment.** Compute price-level features from unadjusted prices. Add a truncation-invariance test: a feature at t must not change when data after t are removed.
2. **Report validation-selected results as primary.** Show test-selected values only as diagnostics, next to shuffled-label and random-predictor baselines.
3. **Test learnability on a planted signal** before interpreting real-data nulls.
4. **Report seed spread and use paired tests.** Do not compare a single seed per arm.
5. **Separate exposure from stock-specific information.** Report beta- and sector-neutral metrics together with a **matched characteristics-only learner**. Do not treat residualised-target metrics as characteristic-free.
6. **Validate any feature-ablation or incremental-information test on synthetic data** with known increments of the size of interest. Report its false-positive rate and power.
7. **Compute per-contrast power before freezing a confirmatory test,** and state what a null can rule out.
8. **Evaluate economics** with turnover control, realistic execution timing, a costed hedge and a random-score null.

## 13. Conclusion

In our setting, apparently skilful results arose from identifiable mechanisms:
- test-informed selection and flawed metrics;
- an optimisation failure;
- leakage through vendor-adjusted price levels;
- exposure to known characteristics;
- comparisons too weak to detect the effects they were meant to test.

After controlling for these, we could not identify stock-specific information beyond characteristics and simple reversal, and our tools were not powerful enough to rule it out. We stopped price-only modelling on this universe as a resource decision. That decision rests on these limitations, not on a claim that returns are unpredictable.

The diagnostics we describe (an adjustment-factor decomposition, matched characteristics-only learners, synthetic validation of evaluation procedures, and power calculations) are inexpensive. We suggest they become standard in financial machine-learning benchmarks.

## 14. References (verified by search on 2026-10-22; bibliographic details from publisher or repository records)

- Agarwal, S., Sawhney, R., Thakkar, M., Nakov, P., Han, J., & Derr, T. (2022). THINK: Temporal Hypergraph Hyperbolic Network. *IEEE ICDM 2022*, 849–854. doi:10.1109/ICDM54844.2022.00096.
- Avramov, D., Cheng, S., & Metzker, L. (2023). Machine learning vs. economic restrictions: Evidence from stock return predictability. *Management Science*, 69(5). doi:10.1287/mnsc.2022.4449.
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4). doi:10.21314/JCF.2016.322.
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.
- Blitz, D., Huij, J., Lansdorp, S., & Verbeek, M. (2013). Short-term residual reversal. *Journal of Financial Markets*, 16(3), 477–504.
- Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems 3* (MLSys 2021).
- Brown, S. J., Goetzmann, W., Ibbotson, R. G., & Ross, S. A. (1992). Survivorship bias in performance studies. *Review of Financial Studies*, 5(4), 553–580.
- Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291–311.
- Da, Z., Liu, Q., & Schaumburg, E. (2014). A closer look at the short-term return reversal. *Management Science*, 60(3), 658–674.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13, 253–263.
- Feng, F., He, X., Wang, X., Luo, C., Liu, Y., & Chua, T.-S. (2019). Temporal relational ranking for stock prediction. *ACM Transactions on Information Systems*, 37(2), Article 27.
- Giacomini, R., & White, H. (2006). Tests of conditional predictive ability. *Econometrica*, 74(6), 1545–1578.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23, 365–380.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
- Hou, K., Xue, C., & Zhang, L. (2020). Replicating anomalies. *Review of Financial Studies*, 33(5), 2019–2133.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1), 1–28.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.
- Nagel, S. (2012). Evaporating liquidity. *Review of Financial Studies*, 25(7), 2005–2039.
- Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their trading costs. *Review of Financial Studies*, 29(1), 104–147.
- Sawhney, R., Agarwal, S., Wadhwa, A., Derr, T., & Shah, R. R. (2021). Stock selection via spatiotemporal hypergraph attention network: A learning to rank approach. *AAAI 2021*, 35(1), 497–504.
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126.

*Cited only through secondary sources; verify before submission:* Lo, A. W., & MacKinlay, A. C. (1990), on contrarian profits and cross-autocorrelation.
