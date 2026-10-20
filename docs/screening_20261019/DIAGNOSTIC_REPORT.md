# Diagnostic report — HyperShift broad screen

Exploratory. Every number comes from saved artifacts: `v2_SCREENING_RESULTS.csv`, `v2_SCREENING_TABLES.md` and `results/SCREEN_RSR_20261019_v2/**/history.json`.

## 1. v2 (RSR NASDAQ, folds 2015/2016): a training-data/budget failure, not mechanism evidence

**Observation.** 153/153 runs are complete with 0 failures, of which 141 are neural fits.
- Selected (best-validation-MSE) epoch: median **1**. 65% of fits select epoch ≤ 2 and 82% select ≤ 5.
- Neural prediction std is 0.001–0.006 in raw 5-day return units, against a target std of about 0.045. That is 2–14% of target scale.
- Pooled neural RankIC lies between −0.003 and +0.010 for every arm.
- Simple baselines on the same dates: reversal(16) 0.011 / 0.028 (2015 / 2016); Ridge 0.027 / 0.0005; random ≈ 0.

**Hypotheses and the checks that separate them:**

| # | Candidate cause | Check | Result |
|---|---|---|---|
| 4 | Inadequate training data/budget | Count training dates; compare selected epoch and validation MSE to a predict-zero forecaster | **Supported.** 37 (2015) and 88 (2016) training dates. Selected validation MSE: 2015 0.953–0.958 vs predict-zero **0.960**; 2016 static 1.051–1.055 vs predict-zero **1.054**. The models are at or barely below the trivial forecaster. |
| 2 | Model fails to learn at all | Synthetic planted-signal tests (`tests/test_screening.py`: repository temporal and static-graph planted-signal learning) | **Rejected** as a code defect. The same layers learn planted signals. |
| 10 | Validation criterion misaligned with ranking | Validation MSE with a date cross-sectional mean oracle: 0.773 (2015), 0.893 (2016), against predict-zero 0.960 / 1.054 | **Supported.** About 15–20% of validation MSE is the day's market-wide move. MSE selection rewards market timing, not cross-sectional ranking, so it picks near-constant forecasts. |
| 13 | Statistical power | Bootstrap CIs | Every principal ΔRankIC CI includes 0. Holm p = 1.0 for all nine primary comparisons. |

**Consequence for the v2 'Promising' labels.**
- The gate labels D multiscale (+0.008), H pretrain (+0.003), and the controls soft, hyp and mlp "Promising". However, the v2 static parent is among the most collapsed arms (RankIC 0.002, prediction std 0.0012).
- Arms whose selected weights moved slightly further from initialisation (mlp, soft, mix and the residual-target arms select later epochs, median 4.5–13) can beat it without any mechanism being responsible.
- These v2 labels are **not carried forward as evidence**. They are noted only as directions to check on v3.

**Action taken (predeclared before any v3 result, see the plan's v3 amendment):**
- move to the S&P 500 panel, which has 1,443 / 1,695 training dates;
- use training stride 1;
- select checkpoints on validation RankIC instead of MSE.

The protocol, seeds, epochs and architecture are otherwise unchanged.

## 2. v3 diagnostics

*Pending v3 completion.* Planned, in order:
- **(a) Selected-epoch and prediction-std check** per arm, same as §1.
- **(b) Training-budget probe.** Static plus the strongest principal arm, at patience 20 and lr 3e-3, in a separate namespace. Run only if most v3 fits stop within ~8 epochs of start. Early v3 static histories show training loss falling only 1.00 → 0.977 over 12 epochs while validation RankIC stays flat (~0.02–0.03 in 2022, ~0 in 2023).
- **(c) Failure-specific probes**, at most 1–2 per important failure:
  - A: soft-assignment entropy against log(16), to detect uniform collapse;
  - C: listwise gradient on known rankings, already unit-tested;
  - G: a standalone check of whether 16/32/64-day summaries carry signal, via Ridge on the summary features alone.
