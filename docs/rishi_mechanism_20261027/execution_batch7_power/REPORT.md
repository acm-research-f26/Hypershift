# How much would a future paired test be able to detect?

2026-10-27; frozen protocolb6bd69d. Ten-seed/top10 E,B,E_Hlabel daily paths. All216planned rows and36reference intervals computed:10000null reference plus10000independent simulated datasets for each horizon252/504/1008and stationary block21/5/63. No future collection or candidate was activated.

## Two years is weak for the proposed increments

Primary block21,504sessions, Holm5%across E−B/E−E_Hlabel:

| True increment relative to each control | Power E−B | Power E−identity | Power to reject both |
|---|---:|---:|---:|
| Annual net mean+1pp |4.36%|3.48%|1.51%|
| Annual net mean+2pp |6.41%|4.64%|2.32%|
| Sharpe+.10 |9.47%|7.16%|4.29%|
| Sharpe+.20 |23.76%|17.64%|13.20%|

At1008sessions, rejecting both still has only3.42%power for+2pp/year and26.17%for+.20SR. None supports an80%power claim. One- and two-year non-rejection would therefore be weak evidence against these small increments. A future504-session log could support exploratory operational observation, but is not a well-powered confirmation design for these endpoints. Any confirmatory horizon or larger minimum relevant effect must be frozen separately, without extending based on interim p values. This report does not choose or activate that future design.

The zero-effect probability of any family rejection ranges4.06–5.02%across all horizon/block/endpoint cells, consistent with the intended5%level within simulation uncertainty. Monte Carlo standard errors are reported per row (≤.005). They condition on the sampled reference-null distribution and do not include reference estimation, historical-path, regime or model-refit uncertainty.

## What was simulated

Mean differences are centered at zero, then E receives+1/+2ppannual mean relative to both controls. SR baselines are shifted to.5for all arms while retaining centered paired paths and SD; E receives the shift needed for+.10/+.20SR. These are fixed effect-size scenarios, not estimates that the current candidate possesses those increments. Dates remain synchronized across arms. Seeds are not treated as extra market histories. The same inspected tape is sampled circularly; extrapolation to more sessions assumes its dependence/variation remain relevant and provides no new independent years.

## Supportive actual-history estimates

Newey–West annual mean E−B:+7.04pp,95%interval[−1.23,+15.31]pp at lag5and[−.23,+14.30]pp at lag20. E−identity:−.42pp,[−12.35,+11.51]pp at lag5and[−11.46,+10.62]pp at lag20. These normal approximations support the wide-uncertainty reading; unadjusted p values do not replace earlier primary stationary-bootstrap/Holm results. No factor-regression promotion claim follows.

Verification: four tests pass for synchronized/reproducible contiguous blocks, exact paired identities, SR calibration, empirical ties/plus-one p and Holm family ordering. Observed source hashes and simulation settings are in RESULTS.json; the code path in its hash record is the launcher's temporary root alias for research/analysis/rishi_mechanism_power.py. The canonical file's bytes match that recorded hash. All planned effect/horizon/block cells are retained, including zeros and weak results.
