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

## 2. v3 diagnostics (S&P 500 panel, folds 2022/2023)

Sources: `v3_SCREENING_TABLES.md`, `v3_PAIRED.csv`, and `metrics.json` → `training.history` under `results/SCREEN_SP500_20261019_v3`. 171/171 runs, 0 failed, 159 neural fits. Spot-check: `two_residual` − `static` RankIC from the per-seed `daily.npz` = +0.01771 (2022) and +0.00038 (2023), pooled +0.00904, which matches `v3_PAIRED.csv` exactly.

### (a) Selected epoch, training progress and prediction spread

- The selected epoch (max validation RankIC) has a median of **5** across neural fits; 24% select epoch ≤ 2. Fits stop on patience 8 after a median of 12–22 epochs, and only 2 of 159 reach the 40-epoch cap.
- **The models barely leave initialisation.** Training loss on the unit-variance target falls by only 0.01–0.03 from epoch 0 to the selected epoch, for static and most arms. Validation RankIC at epoch 0 is already +0.004 to +0.012 for most arms (static +0.009).
- Validation RankIC at the selected epoch (≈ 0.02–0.04) is 2–3× the test RankIC (≈ 0.00–0.02). That gap is the expected upward bias from taking a max over ~15 noisy epochs.
- The 2023 fold validates on 2022. There, validation MSE at the selected epoch is ≈ 1.36 versus ≈ 0.75 in the 2022 fold (validation year 2021): the 2022 bear market sits outside the training target scale.

| Arm | Selected epoch (median, max) | Epochs run (median) | Δ train loss (epoch 0 → selected) | Val RankIC at selected | Test RankIC |
|---|---|---|---|---|---|
| static | 3, 15 | 12 | +0.014 | +0.023 | +0.007 |
| two_residual | 10, 15 | 18 | +0.017 | +0.025 | +0.016 |
| pretrain | 10, 15 | 18 | +0.017 | +0.020 | +0.011 |
| none (no graph) | 4, 14 | 13 | +0.014 | +0.021 | +0.011 |
| mlp | 4, 8 | 14 | +0.019 | +0.020 | +0.011 |
| listwise | 1, 3 | 10 | +0.016 | +0.012 | +0.013 |
| gated | 2, 32 | 12 | +0.081 | +0.020 | −0.003 |
| mix | 13, 33 | 22 | +0.277 | +0.041 | −0.001 |

(Means over the 6 fits unless marked median. Full per-arm table in Appendix 2A.)

### (b) Training-budget probe

*Running* (`results/SCREEN_SP500_20261019_v3_budget`: static and two_residual, patience 20, lr 3e-3, seeds 7/19/42, folds 2022/2023, on the laptop like their v3 parents). The `--lr` flag was checked to be behaviour-neutral by rerunning `2022/static/seed_7` at the default (`results/SCREEN_SP500_20261019_v3_lrcheck`). Results will be added here. The probe cannot change any screen or deep verdict.

### (c) Failure-specific findings (no extra training)

| Direction | Finding | Evidence | Consequence |
|---|---|---|---|
| A soft / residual_soft | **Soft hyperedge assignment collapses to uniform** | Assignment entropy = 2.772 at epoch 0 and at the selected epoch in all 12 fits; log 16 = 2.7726 | The learned hyperedges never differentiate. Each one is a near-global mean, so A's soft arms do not test learned relations. A fix needs an assignment temperature, a balance or entropy penalty, or initialising from the training clusters. Not a negative result for learned hyperedges as such. |
| A sector_graph / random_graph / static | No graph benefit is measurable | `none` (no hyperedges) +0.004, `random_graph` +0.003, `sector_graph` +0.004 vs static; all CIs include 0 | Graph structure, whether correlation clusters, GICS sectors or random groups, is not separable from no graph on this panel at this model size. |
| B market / sector / cluster target | Trained on residual targets but scored on raw RankIC | Raw ΔRankIC −0.017 to −0.026. Sector-neutral RankIC: sector_target +0.0013 / +0.0022 vs static −0.0075 / +0.0063 (2022 / 2023) | The predeclared Negative verdict stands, because raw RankIC is the primary metric. On factor-neutral ranking B is mixed, not worse: better in 2022 and worse in 2023. The raw loss is largely the factor tilt that residual targets remove by design. |
| E mix / gated | Overfitting | Training loss falls 0.20–0.45; validation MSE rises to 1.8–2.2 in the 2023 fold; test RankIC ≈ 0 or negative | The pooled-fusion mixer has much more capacity than the rest. Gating does not fix the overfitting of its mix parent (E: −0.002). |
| G long / summary | Longer context does not help | long −0.007, summary +0.001; Ridge on 4/16/64-day window means plus the last observation: RankIC 0.006 (2022 −0.008, 2023 +0.021) | The long-window summaries carry little linear signal here. G's null is consistent with the data, not only with optimisation. |
| C listwise | Selects very early | Median selected epoch 1; ΔRankIC 2022 −0.015, 2023 +0.026 | High fold-to-fold variance. Not promoted. |
| I frozen | The 2022-fold static model scored on 2023 beats the retrained 2023 static by +0.022 raw RankIC (seeds +0.046 / +0.026 / −0.006) | Neutral RankIC: frozen +0.0007 vs retrained +0.0063 | The raw advantage is a factor/sector tilt that worked in 2023, not better stock-specific ranking. It also suggests 2022 is a poor validation year for selection. It stays Inconclusive (one fold, 3 seeds). |

### (d) Headline diagnosis

- No v3 arm is separable from `static`: Holm p = 1.0 for all nine primary comparisons, and every 95% CI includes 0.
- `static` itself (RankIC 0.007) is not separable from no-graph (0.011), mlp (0.011), reversal(16) (0.011) or Ridge (0.006). Pooled neutral RankIC is within ±0.009 for every arm.
- Predictions correlate 0.8–0.9 with the market-wide move, so most realised performance is market or factor exposure.
- Seed-to-seed spread of per-fold ΔRankIC (about ±0.02–0.03) is 3–5× the largest mean effects. The screen is noise-limited at 3 seeds.

### (e) Platform control

The 120 Kaggle-run fits used torch 2.11 CPU; the 51 laptop fits used torch 2.14 + cu130, run on CPU. Static was rerun on Kaggle (seeds 7/19/42 × 2022/2023; `v3_PLATFORM_SENSITIVITY.md`).
- Four of six runs agree with the laptop to within 0.0004 RankIC (prediction correlation ≥ 0.9988).
- Two 2023 runs diverge: −0.0075 and −0.0018, prediction correlation 0.97 and 0.95. Small numeric differences sent them down different training paths.
- That is platform jitter of seed-noise size, not a bias. Mean −0.0015.
- Laptop reruns are bit-identical (`2022/static/seed_7`, lr-check), so the jitter comes from the platform, not from nondeterminism within one machine.

Re-pairing the Kaggle arms against Kaggle static shifts every contrast by ≈ +0.0015. That flips the gate label of residual_soft, hybrid, sector_graph and the negative control random_graph to 'Promising'. **The 3-seed gate cannot separate arms within about 0.002 of its threshold.** The predeclared labels remain primary. Every CI includes 0 under both pairings.

### Appendix 2A. Per-arm training diagnostics (all neural arms)

Columns: n fits; selected epoch (median, max); epochs run (median); median test prediction std (raw 5-day return units); unique fraction; mean validation RankIC at selected epoch; mean validation RankIC at epoch 0; mean test RankIC; mean training-loss drop from epoch 0 to the selected epoch.

```
arm            n best(med,max) ran(med) testPredStd uniq valRIC@best valRIC@0 testRIC  dTrainLoss
adaptive       6    5  12     14    0.0027   1.00  +0.0281   +0.0089  +0.0054  +0.0083
cluster_target 6    4   9     14    0.0006   1.00  +0.0095   +0.0049  -0.0101  +0.0010
frozen         3    3   3     12    0.0008   1.00  +0.0366   +0.0218  +0.0431  +0.0091
gated          6    2  32     12    0.0032   1.00  +0.0204   -0.0018  -0.0029  +0.0813
hybrid         6   11  14     20    0.0025   1.00  +0.0211   +0.0074  +0.0143  +0.0245
hyp            6   10  14     18    0.0017   1.00  +0.0276   +0.0080  +0.0119  +0.0091
listwise       6    1   3     10    0.0076   1.00  +0.0124   +0.0062  +0.0130  +0.0159
long           6    5  14     14    0.0035   1.00  +0.0289   +0.0048  +0.0005  +0.0253
lowrank        6    3  13     12    0.0006   1.00  +0.0220   +0.0122  +0.0015  +0.0156
market_target  6    4  11     12    0.0010   1.00  +0.0172   +0.0042  -0.0193  +0.0013
master         6    4  26     13    0.0013   1.00  +0.0280   +0.0062  +0.0099  +0.0334
mix            6   13  33     22    0.0123   1.00  +0.0411   -0.0027  -0.0009  +0.2773
mlp            6    4   8     14    0.0020   1.00  +0.0197   +0.0062  +0.0111  +0.0186
multiscale     6    8  13     17    0.0023   1.00  +0.0334   +0.0096  +0.0088  +0.0198
none           6    4  14     13    0.0023   1.00  +0.0210   +0.0091  +0.0113  +0.0138
pairwise       6    4  13     13    0.0016   1.00  +0.0242   +0.0089  +0.0026  +0.0114
pretrain       6   10  15     18    0.0026   1.00  +0.0199   -0.0067  +0.0107  +0.0174
random_graph   6   10  15     18    0.0013   1.00  +0.0188   +0.0083  +0.0101  +0.0154
residual_soft  6    8  15     17    0.0012   1.00  +0.0207   +0.0035  +0.0089  +0.0221
rolling        6    6  11     15    0.0012   1.00  +0.0173   +0.0053  +0.0066  +0.0137
sector_graph   6    4  13     12    0.0013   1.00  +0.0189   +0.0102  +0.0109  +0.0124
sector_target  6    6  22     14    0.0008   1.00  +0.0129   -0.0001  -0.0169  +0.0013
soft           6    7   9     16    0.0022   1.00  +0.0231   +0.0110  +0.0120  +0.0180
static         6    3  15     12    0.0013   1.00  +0.0233   +0.0091  +0.0071  +0.0142
summary        6    4  23     14    0.0035   1.00  +0.0272   +0.0102  +0.0084  +0.0172
two            6    8  14     17    0.0015   1.00  +0.0273   +0.0098  +0.0099  +0.0084
two_residual   6   10  15     18    0.0024   1.00  +0.0250   +0.0107  +0.0162  +0.0165
neural fits 159 median best 5.0 frac best<=2 0.24 frac ran<=10 0.16 hit 40 epochs 2
```
