# Selection, initialization and past-price characteristics

2026-10-27. Protocol `b35bcaf`, implementation `6c7fb6f`. The pre-outcome count typo was corrected to the four named origin metrics. Completed 1,809 origin rows, 60 paired selection intervals, 54 daily active-return rows, 15 raw covariance decompositions, 120 fixed seed-bundle accounts and 804 as-of style rows. No neural fitting or new primary endpoint.

## Global ranking and top-tail performance differ

Ten-seed means, using 200 origins with defined metrics out of 201 original trading origins; individual stock label availability still varies:

| Candidate | RankIC | Top-10 minus EW label return | Ten-seed top-10 net CAGR |
|---|---:|---:|---:|
| Parent B |.00377|.216 pp / origin|15.36%|
| Dynamic D |.03072|.222 pp / origin|15.32%|
| Blend E |.01889|.361 pp / origin|23.92%|
| Hlabel |.01382|.209 pp / origin|14.77%|
| Parent + Hlabel |.01331|.355 pp / origin|22.71%|
| Reversal |.01827|.161 pp / origin|5.81%|

The dynamic family has the strongest whole-universe RankIC but the blend has stronger top-tail and account point results. E minus B RankIC is +.01512 [+.00573,+.02442]; E minus D is −.01183 [−.02324,−.00100]. These are pointwise descriptive intervals, not multiplicity-adjusted discovery tests. Higher broad ranking accuracy does not automatically identify the best concentrated portfolio.

E minus parent+Hlabel top-10 label advantage is only .0059 pp per origin, with interval [−.2035,+.2069] pp. Its RankIC difference is +.00558 [−.01057,+.02193]. E minus reversal RankIC is +.00062 [−.03314,+.03440]. These estimates do not establish an incremental changing signal or equivalence.

Labels normalize over observed selected weight only after the frozen inference mask chooses targets. The final partial target remains unavailable for full five-session metrics. Never annualize these frictionless origin returns into a tradable net-alpha claim: actual accounts include share drift, costs, settlement, partial terminal handling and missing-label differences. The earlier full-account primary mean/Holm results are unchanged.

## Active performance versus a broad eligible book

Against the matched weekly eligible EW account, top-10 E has active IR 1.205 [.324,2.120] and annual arithmetic active mean +17.20 pp [+4.69,+29.63]. Parent+Hlabel also has positive point performance: IR .982 [.021,1.988], annual mean +17.62 pp [+.40,+34.20]. These are selected-history pointwise date-bootstrap intervals. They do not control for risk/style exposure, historical search or refit uncertainty, and do not identify learned relations. The direct E-minus-identity family remains the relevant incremental mechanism comparison.

Origin intervals use synchronized stationary blocks within each fitted annual stratum: four origins primary, one/thirteen sensitivity. Daily intervals use the separate 21-session date bootstrap, five/63 sensitivity. Annual strata and these ten trained seeds are fixed; neither procedure creates extra independent market regimes.

## Initialization remains consequential

Twenty five-seed subsets were frozen before scoring and used identically across both families. Every B/D/E, top-5/top-10 account is retained; no subset is selected for further use.

| Candidate / holdings | SR min / median / max | CAGR min / max |
|---|---|---|
| B /5 |.175 / .399 / .711|2.89% /23.31%|
| D /5 |.255 / .537 / .731|6.12% /23.90%|
| E /5 |.362 / .678 /1.083|9.97% /43.31%|
| B /10 |.255 / .459 / .750|6.34% /23.35%|
| D /10 |.351 / .515 / .697|9.40% /22.24%|
| E /10 |.432 / .647 / .857|12.12% /29.10%|

The large spread shows sensitivity to which existing initializations are averaged. These overlapping subsets are conditional diagnostics of the original ten seeds, not fresh training replications, independent years or confidence intervals for all possible seeds. The highest row is not a new recommended strategy.

## Why nearly identical errors do not imply identical forecasts

Pooled mean-raw-forecast correlation B/D is .113, while their error correlation is .9979. Trading instead averages per-seed percentile ranks, a different operation. Within annual fits, raw forecast correlations range .076–.429.

The exact identity is:

`cov(fB-y, fD-y) = cov(fB,fD) - cov(fB,y) - cov(fD,y) + var(y)`.

For 2023, target variance is 22.0698 percentage-points squared; error covariance is 22.0643; forecast covariance is only .000164. The shared target term dominates that arithmetic identity. Zero forecasts would also have identical errors. Thus the near-one error correlation is not evidence that the two selectors are interchangeable. Subtracting a common baseline from both forecast and target leaves each error unchanged, verified within 1.8e-15; it does not solve this interpretation problem. Forecast innovations relative to Hlabel are separately recorded without a causal claim.

## The historical-return identity substitute carries style preferences

Mean cross-sectional Spearman correlation of Hlabel preference with as-of QQQ beta is .506/.591/.080/.368 for 2022–2025; with 20-day volatility it is .386/.303/.066/.208, and with log ADV .233/.266/.122/.289. Momentum changes sign: −.204/+.195/+.115/+.084. The cheap historical-return control is not style-neutral. This helps frame competing explanations, but is not a causal style-matched account test or evidence about the neural model's learned identity intercept.

All features end at the origin close. Beta requires at least 126 finite/tradable observations in 252, volatility 15 in 20, and momentum finite positive tradable endpoints. No dated sector classification was supplied; current sectors were not substituted. Pair availability affects only descriptive correlations, not trading names.

Seven tests pass. Independent review prompted a finite-positive momentum endpoint guard; a regression test failed before it, then passed. Recomputing all 804 current style rows shows no changed result. INPUT_AND_OUTPUT_AUDIT.json re-verifies all 1,184 original publication hashes, hashes consumed generated inputs, and checks all 120 bundle paths and covariance identities. Original scored-code hashes remain preserved; the later guard is a non-impacting validation correction.
