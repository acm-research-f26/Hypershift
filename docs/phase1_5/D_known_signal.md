# D (part 1): known-signal learning check

*Phase 1.5 fidelity audit, 2026-10-02. Code: `scripts/known_signal.py` (generator + runner), `scripts/known_signal_analysis.py` (references, probe, tables), `tests/test_known_signal.py` (fast pytest), Kaggle preset 5/6/7 (`kaggle/launch_known_signal.sh`). Training ran on Kaggle T4 (kernels `hypershift-run-ks`, `-ks2`, `-ks3`; 1-epoch smoke test `-ks-smoke` first), analysis on laptop CPU, single thread. Raw per-run files: `results/known_signal_*`, `results/ks_long_*`, `results/ks_bd1_*` (git-ignored), summary `results/known_signal_analysis.json`.*

## Verdict

**The pipeline does not recover a planted signal under the Phase 1 protocol (`input_mode: level`).** With level inputs, every arm (THINK, EH, EE, no-graph) reaches test IC 0 to 0.05, at most 19% of the oracle IC, even for a strong planted signal (oracle IC 0.28) and even after 100 epochs or one day per optimizer step. A pooled ridge regression on the *same* window tensors gets IC 0.12 to 0.22, so the signal is in the inputs; the models do not find it. With `input_mode: relative` the no-graph models recover 24-81% of the oracle IC, and the graph arms recover less than the no-graph arms at all four levels where the own-lag carries the signal (own, low, mid, high). Consequence for Phase 1: the null results of R5_g2 (level inputs, `norm: paper`) cannot distinguish "no signal in the data" from "this training setup cannot learn a 1-2% daily signal", so they say nothing about hyperbolic vs Euclidean or hyperedges vs none.

Caveats: the planted signal is a one-day linear autoregression plus a hypergraph-lag term, a favourable and simple case; 5 seeds per cell; 30 epochs (100 in the diagnosis); IC is measured at the epoch chosen by validation Sharpe (the protocol), history files keep no IC per epoch, but test MSE and NDCG per epoch show the same absence of learning in level mode (below).

## 1. Design

Real shapes: the small-scale universe (`poc_sectors.universe()`): 309 stocks, the real 558-edge g2 hypergraph, 1245 real dates, the real observation mask, splits 756 / 1008 / 1245. Prices are synthetic.

Daily returns for all stocks, per seed (`plant_signal`, RNG `default_rng(10000 + seed)`):

`r[n,t] = phi * r[n,t-1] + gamma * (A r[:,t-1])[n] + eps[n,t]`, `eps ~ N(0, 0.01^2)`

with `A = Dv^-1 H De^-1 H^T`, the hypergraph random-walk operator (mean of hyperedge means; `H` the incidence matrix). So the next-day return has an own-lag part (a feature visible in the stock's own window), a group part (visible only through hyperedge co-members' windows) and noise. Prices are `cumprod(1 + r)`; features `(ma5, ma10, ma20, ma30, close)` divided by the per-stock training-period max (`norm: train`); `gt` is the return of the synthetic close, missing days get the real mask and fill value 1.1. Everything after that is the real pipeline: `train_one_run` (windows, `gather_batch`, model, `rank_mse_loss`, validation-Sharpe epoch selection, `evaluate_all`).

| level | phi | gamma | note |
|---|---|---|---|
| none | 0 | 0 | null control (no signal) |
| own | 0.15 | 0 | own-lag only; graph should not help |
| low | 0.05 | 0.10 | |
| mid | 0.10 | 0.20 | |
| high | 0.20 | 0.40 | |
| group | 0 | 0.50 | group effect dominates; graph should help |

Arms: HH_hyper (THINK), EH_hyper, EE_hyper, HH_none, EE_none; `input_mode` level and relative; seeds 0-4; 30 epochs, patience 10, `batch_days 8`, `lr 1e-3`, `alpha 1`, `norm train`, otherwise the Phase 1 defaults (`docs/phase1_5/D_resolved_configs.md`). 300 runs on Kaggle (T4, 2-5 s/epoch).

**A subtlety found in review.** `A` has a self-weight: `mean diag(A) = 0.125` (min 0.021, max 0.487; 0 isolated stocks). So even `group` (phi = 0) plants an own-lag term `gamma * A_ii * r_i[t-1]` that a no-graph model can learn, and "own-only" is not zero there. References are therefore computed per seed from the generator's own `A`: **oracle** `phi r + gamma A r`; **no-graph** `(phi + gamma diag(A)) r` (the best a model that sees only its own window can do); **graph-only** `gamma (A - diag A) r`; **lag-1** `r[t-1]` (what a shared-weight no-graph model can at best use). "Graph beats none" is counted only if graph minus none, paired by seed, is clearly positive and compatible with the graph-only gap.

Metrics: **test IC** (mean over the 252 test days of the cross-sectional Pearson correlation between prediction and realised return; a constant-prediction day counts as 0; float64) at the validation-selected epoch is primary; IC / oracle IC is the fraction of signal recovered; Sharpe is reported as top-5 minus the equal-weight daily return ("excess Sharpe"), because the plain Sharpe is dominated by the noise-path market return (hold Sharpe 2.9-5.3 across levels).

## 2. Results

### Table A. References and linear probe (mean over 5 seeds)

| level | oracle IC | no-graph IC | graph-only IC | lag-1 IC | oracle excess SR | no-graph excess SR | ridge probe IC level / relative |
|---|---|---|---|---|---|---|---|
| none | 0.000 | 0.000 | 0.000 | 0.001 | 0.5 | 0.5 | +0.001 / +0.001 |
| own | 0.150 | 0.150 | 0.000 | 0.150 | 13.1 | 13.1 | +0.116 / +0.149 |
| group | 0.146 | 0.110 | 0.096 | 0.068 | 15.7 | 15.0 | +0.038 / +0.068 |
| low | 0.068 | 0.065 | 0.016 | 0.063 | 7.0 | 6.4 | +0.031 / +0.062 |
| mid | 0.135 | 0.130 | 0.033 | 0.125 | 13.5 | 12.9 | +0.091 / +0.125 |
| high | 0.279 | 0.267 | 0.088 | 0.257 | 27.7 | 26.9 | +0.218 / +0.257 |

The probe is a pooled ridge regression of `gt` on the flattened 16 x 5 window, after the pipeline's own `gather_batch` and `apply_input_mode`, fit on train days and scored on test days. It reaches IC 0.12 / 0.22 (own / high) on level inputs and 0.15 / 0.26 on relative inputs, i.e. 77-100% of the lag-1 reference: **the signal survives window construction and both normalizations.**

### Table B. Model test IC at the validation-selected epoch (30 epochs, 5 seeds; mean +- s.e.)

| level | input | arm | n | test IC | IC / oracle IC | IC / no-graph IC | excess Sharpe (top5 - hold) | pred sd |
|---|---|---|---|---|---|---|---|---|
| group | level | EE_hyper | 5 | +0.002 +- 0.003 | 0.02 | 0.02 | +0.208 +- 0.672 | 9.51e-05 |
| group | level | EE_none | 5 | +0.000 +- 0.004 | 0.00 | 0.00 | +0.330 +- 0.384 | 6.38e-04 |
| group | level | EH_hyper | 5 | -0.002 +- 0.003 | -0.01 | -0.02 | -0.165 +- 0.672 | 1.01e-03 |
| group | level | HH_hyper | 5 | +0.003 +- 0.004 | 0.02 | 0.03 | +0.622 +- 0.936 | 3.08e-05 |
| group | level | HH_none | 5 | +0.002 +- 0.001 | 0.01 | 0.02 | +0.252 +- 0.620 | 1.55e-03 |
| group | relative | EE_hyper | 5 | +0.070 +- 0.007 | 0.48 | 0.63 | +6.777 +- 1.011 | 2.82e-06 |
| group | relative | EE_none | 5 | +0.047 +- 0.002 | 0.32 | 0.43 | +5.385 +- 0.382 | 1.44e-05 |
| group | relative | EH_hyper | 5 | +0.062 +- 0.004 | 0.42 | 0.56 | +6.655 +- 0.778 | 2.51e-05 |
| group | relative | HH_hyper | 5 | +0.038 +- 0.009 | 0.26 | 0.35 | +4.253 +- 1.128 | 3.67e-08 |
| group | relative | HH_none | 5 | +0.035 +- 0.002 | 0.24 | 0.32 | +4.419 +- 0.507 | 1.09e-05 |
| high | level | EE_hyper | 5 | +0.001 +- 0.002 | 0.00 | 0.00 | +1.365 +- 1.359 | 2.19e-04 |
| high | level | EE_none | 5 | +0.035 +- 0.012 | 0.13 | 0.13 | +4.075 +- 1.556 | 2.46e-03 |
| high | level | EH_hyper | 5 | +0.001 +- 0.002 | 0.01 | 0.01 | +1.038 +- 0.613 | 6.62e-04 |
| high | level | HH_hyper | 5 | +0.011 +- 0.009 | 0.04 | 0.04 | +2.988 +- 0.931 | 2.65e-05 |
| high | level | HH_none | 5 | +0.046 +- 0.022 | 0.16 | 0.17 | +8.248 +- 3.442 | 8.20e-04 |
| high | relative | EE_hyper | 5 | +0.132 +- 0.011 | 0.47 | 0.50 | +13.741 +- 1.376 | 1.49e-07 |
| high | relative | EE_none | 5 | +0.226 +- 0.002 | 0.81 | 0.85 | +21.359 +- 0.492 | 3.25e-05 |
| high | relative | EH_hyper | 5 | +0.098 +- 0.012 | 0.35 | 0.37 | +11.119 +- 1.043 | 1.45e-05 |
| high | relative | HH_hyper | 5 | +0.074 +- 0.015 | 0.27 | 0.28 | +7.803 +- 1.766 | 2.29e-08 |
| high | relative | HH_none | 5 | +0.187 +- 0.016 | 0.67 | 0.70 | +17.808 +- 1.489 | 3.71e-05 |
| low | level | EE_hyper | 5 | -0.001 +- 0.002 | -0.01 | -0.02 | +0.120 +- 0.589 | 1.31e-03 |
| low | level | EE_none | 5 | +0.001 +- 0.004 | 0.01 | 0.02 | +0.026 +- 0.613 | 1.78e-03 |
| low | level | EH_hyper | 5 | -0.000 +- 0.001 | -0.00 | -0.01 | +0.539 +- 0.361 | 9.10e-04 |
| low | level | HH_hyper | 5 | +0.002 +- 0.001 | 0.03 | 0.03 | +0.689 +- 0.512 | 1.58e-05 |
| low | level | HH_none | 5 | +0.001 +- 0.002 | 0.02 | 0.02 | -0.139 +- 0.509 | 9.33e-04 |
| low | relative | EE_hyper | 5 | +0.009 +- 0.005 | 0.13 | 0.14 | +1.057 +- 0.435 | 1.45e-07 |
| low | relative | EE_none | 5 | +0.034 +- 0.004 | 0.50 | 0.52 | +3.711 +- 0.227 | 1.35e-05 |
| low | relative | EH_hyper | 5 | +0.016 +- 0.002 | 0.23 | 0.24 | +2.016 +- 0.460 | 1.72e-05 |
| low | relative | HH_hyper | 5 | +0.003 +- 0.002 | 0.05 | 0.05 | +0.746 +- 0.658 | 1.16e-09 |
| low | relative | HH_none | 5 | +0.020 +- 0.002 | 0.30 | 0.31 | +2.326 +- 0.691 | 2.26e-07 |
| mid | level | EE_hyper | 5 | -0.001 +- 0.002 | -0.00 | -0.00 | -0.036 +- 0.617 | 1.39e-04 |
| mid | level | EE_none | 5 | +0.008 +- 0.006 | 0.06 | 0.06 | +0.427 +- 0.815 | 2.33e-03 |
| mid | level | EH_hyper | 5 | -0.001 +- 0.001 | -0.01 | -0.01 | +0.274 +- 0.374 | 1.04e-03 |
| mid | level | HH_hyper | 5 | +0.002 +- 0.002 | 0.01 | 0.02 | -0.239 +- 0.732 | 3.15e-05 |
| mid | level | HH_none | 5 | +0.012 +- 0.008 | 0.09 | 0.10 | +1.722 +- 1.088 | 7.49e-04 |
| mid | relative | EE_hyper | 5 | +0.040 +- 0.003 | 0.30 | 0.31 | +4.875 +- 0.508 | 2.14e-07 |
| mid | relative | EE_none | 5 | +0.093 +- 0.007 | 0.69 | 0.71 | +8.680 +- 0.563 | 1.79e-05 |
| mid | relative | EH_hyper | 5 | +0.038 +- 0.007 | 0.28 | 0.29 | +4.419 +- 0.395 | 4.53e-06 |
| mid | relative | HH_hyper | 5 | +0.009 +- 0.006 | 0.07 | 0.07 | +1.202 +- 0.703 | 2.87e-09 |
| mid | relative | HH_none | 5 | +0.058 +- 0.006 | 0.43 | 0.44 | +5.764 +- 0.526 | 1.22e-05 |
| none | level | EE_hyper | 5 | -0.003 +- 0.001 | n/a | n/a | -0.554 +- 0.574 | 8.56e-04 |
| none | level | EE_none | 5 | +0.000 +- 0.002 | n/a | n/a | -0.095 +- 0.465 | 8.34e-04 |
| none | level | EH_hyper | 5 | +0.000 +- 0.001 | n/a | n/a | +0.239 +- 0.268 | 7.87e-04 |
| none | level | HH_hyper | 5 | -0.000 +- 0.001 | n/a | n/a | -0.359 +- 0.519 | 2.82e-05 |
| none | level | HH_none | 5 | -0.000 +- 0.001 | n/a | n/a | -0.048 +- 0.577 | 1.53e-03 |
| none | relative | EE_hyper | 5 | +0.000 +- 0.002 | n/a | n/a | +0.218 +- 0.446 | 3.97e-05 |
| none | relative | EE_none | 5 | +0.001 +- 0.002 | n/a | n/a | -0.282 +- 0.469 | 2.53e-04 |
| none | relative | EH_hyper | 5 | +0.002 +- 0.001 | n/a | n/a | +0.163 +- 0.672 | 4.17e-05 |
| none | relative | HH_hyper | 5 | +0.001 +- 0.001 | n/a | n/a | +0.417 +- 0.401 | 3.74e-12 |
| none | relative | HH_none | 5 | +0.002 +- 0.001 | n/a | n/a | +0.233 +- 0.573 | 6.89e-11 |
| own | level | EE_hyper | 5 | -0.001 +- 0.001 | -0.00 | -0.00 | -0.080 +- 0.624 | 1.38e-03 |
| own | level | EE_none | 5 | +0.010 +- 0.007 | 0.07 | 0.07 | +0.846 +- 0.927 | 2.35e-03 |
| own | level | EH_hyper | 5 | -0.000 +- 0.001 | -0.00 | -0.00 | +0.235 +- 0.399 | 5.11e-04 |
| own | level | HH_hyper | 5 | +0.000 +- 0.001 | 0.00 | 0.00 | -0.423 +- 0.609 | 2.74e-05 |
| own | level | HH_none | 5 | +0.012 +- 0.005 | 0.08 | 0.08 | +2.870 +- 1.750 | 7.17e-04 |
| own | relative | EE_hyper | 5 | +0.024 +- 0.002 | 0.16 | 0.16 | +3.390 +- 0.765 | 5.38e-07 |
| own | relative | EE_none | 5 | +0.106 +- 0.006 | 0.71 | 0.71 | +9.732 +- 0.666 | 1.41e-05 |
| own | relative | EH_hyper | 5 | +0.024 +- 0.002 | 0.16 | 0.16 | +3.269 +- 0.404 | 3.63e-06 |
| own | relative | HH_hyper | 5 | +0.007 +- 0.003 | 0.04 | 0.04 | +1.625 +- 0.832 | 4.22e-09 |
| own | relative | HH_none | 5 | +0.074 +- 0.012 | 0.50 | 0.50 | +5.912 +- 0.984 | 5.50e-06 |

Rows with `n/a` are the null level, where the oracle IC is 0.

### Table C. Graph arm minus its no-graph counterpart, paired by seed (test IC)

| level | input | pair | n | mean IC diff | s.e. | seeds with diff > 0 | graph-only IC (reference gap) |
|---|---|---|---|---|---|---|---|
| group | level | HH_hyper - HH_none | 5 | +0.0015 | 0.0032 | 3/5 | +0.0363 |
| group | level | EE_hyper - EE_none | 5 | +0.0021 | 0.0063 | 3/5 | +0.0363 |
| group | level | EH_hyper - EE_none | 5 | -0.0021 | 0.0046 | 1/5 | +0.0363 |
| group | relative | HH_hyper - HH_none | 5 | +0.0033 | 0.0098 | 3/5 | +0.0363 |
| group | relative | EE_hyper - EE_none | 5 | +0.0225 | 0.0072 | 5/5 | +0.0363 |
| group | relative | EH_hyper - EE_none | 5 | +0.0145 | 0.0058 | 4/5 | +0.0363 |
| high | level | HH_hyper - HH_none | 5 | -0.0348 | 0.0229 | 1/5 | +0.0119 |
| high | level | EE_hyper - EE_none | 5 | -0.0347 | 0.0113 | 1/5 | +0.0119 |
| high | level | EH_hyper - EE_none | 5 | -0.0338 | 0.0107 | 1/5 | +0.0119 |
| high | relative | HH_hyper - HH_none | 5 | -0.1132 | 0.0277 | 0/5 | +0.0119 |
| high | relative | EE_hyper - EE_none | 5 | -0.0938 | 0.0098 | 0/5 | +0.0119 |
| high | relative | EH_hyper - EE_none | 5 | -0.1277 | 0.0122 | 0/5 | +0.0119 |
| low | level | HH_hyper - HH_none | 5 | +0.0006 | 0.0031 | 3/5 | +0.0024 |
| low | level | EE_hyper - EE_none | 5 | -0.0020 | 0.0036 | 3/5 | +0.0024 |
| low | level | EH_hyper - EE_none | 5 | -0.0013 | 0.0039 | 3/5 | +0.0024 |
| low | relative | HH_hyper - HH_none | 5 | -0.0172 | 0.0034 | 0/5 | +0.0024 |
| low | relative | EE_hyper - EE_none | 5 | -0.0249 | 0.0054 | 0/5 | +0.0024 |
| low | relative | EH_hyper - EE_none | 5 | -0.0181 | 0.0054 | 0/5 | +0.0024 |
| mid | level | HH_hyper - HH_none | 5 | -0.0104 | 0.0081 | 2/5 | +0.0049 |
| mid | level | EE_hyper - EE_none | 5 | -0.0083 | 0.0062 | 2/5 | +0.0049 |
| mid | level | EH_hyper - EE_none | 5 | -0.0090 | 0.0063 | 2/5 | +0.0049 |
| mid | relative | HH_hyper - HH_none | 5 | -0.0486 | 0.0032 | 0/5 | +0.0049 |
| mid | relative | EE_hyper - EE_none | 5 | -0.0529 | 0.0068 | 0/5 | +0.0049 |
| mid | relative | EH_hyper - EE_none | 5 | -0.0551 | 0.0101 | 0/5 | +0.0049 |
| none | level | HH_hyper - HH_none | 5 | -0.0003 | 0.0011 | 2/5 | +0.0000 |
| none | level | EE_hyper - EE_none | 5 | -0.0028 | 0.0022 | 1/5 | +0.0000 |
| none | level | EH_hyper - EE_none | 5 | +0.0001 | 0.0013 | 2/5 | +0.0000 |
| none | relative | HH_hyper - HH_none | 5 | -0.0012 | 0.0014 | 1/5 | +0.0000 |
| none | relative | EE_hyper - EE_none | 5 | -0.0005 | 0.0028 | 2/5 | +0.0000 |
| none | relative | EH_hyper - EE_none | 5 | +0.0009 | 0.0019 | 3/5 | +0.0000 |
| own | level | HH_hyper - HH_none | 5 | -0.0117 | 0.0059 | 1/5 | +0.0000 |
| own | level | EE_hyper - EE_none | 5 | -0.0105 | 0.0073 | 2/5 | +0.0000 |
| own | level | EH_hyper - EE_none | 5 | -0.0102 | 0.0065 | 2/5 | +0.0000 |
| own | relative | HH_hyper - HH_none | 5 | -0.0678 | 0.0118 | 0/5 | +0.0000 |
| own | relative | EE_hyper - EE_none | 5 | -0.0820 | 0.0049 | 0/5 | +0.0000 |
| own | relative | EH_hyper - EE_none | 5 | -0.0819 | 0.0080 | 0/5 | +0.0000 |

`EH_hyper` is paired with `EE_none` (EH has Euclidean temporal conv, the matching no-graph model is EE_none). The last column is the reference gap oracle minus no-graph IC, i.e. what a model that perfectly used the hyperedges could gain over the best no-graph model.

## 3. Reading

1. **Null control is clean.** At level `none` every arm has |IC| <= 0.003 and excess Sharpe within noise of 0 (|excess SR| < 1.1 s.e.). The evaluator and the selection protocol do not manufacture signal.
2. **Level inputs: the pipeline fails to learn.** Best level-mode cell: `high`/HH_none IC 0.046 (17% of the oracle 0.279) and EE_none 0.035; the graph arms are at 0.001-0.011. At `own`, `mid`, `low`, `group` level-mode IC is <= 0.012 for every arm. The ridge probe on the same tensors gets 0.12-0.22.
3. **Relative inputs: learning happens, partially.** No-graph arms recover 24-81% of the oracle IC (EE_none: own 71%, mid 69%, high 81%, low 50%, group 32%). The shortfall from 100% is not small; it is the 30-epoch budget and a 5-seed mean, I did not tune.
4. **Graph arms lose to no-graph at every level where own-lag carries the signal** (own, low, mid, high; relative mode: -0.02 to -0.14 IC, 0/5 seeds positive). Only at `group`, where the reference gap is largest, does a graph arm beat none: EE_hyper - EE_none = +0.0225 +- 0.007 (5/5 seeds), against a reference graph-only gap of 0.0363 (and graph-only IC 0.096); HH_hyper - HH_none +0.003 (3/5), EH vs EE_none +0.005 (4/5). So the graph arms use at most about 60% of what is available for EE and almost nothing for THINK. Hypothesis, not tested here: the DHHAN output replaces a node's embedding by an attention-weighted aggregate of hyperedge messages with no self path (paper eq. 15 as implemented), which averages away the stock's own signal; this fits the loss of IC when the signal is own-lag, but I did not run an ablation with a residual self term.
5. **THINK (HH_hyper) is the weakest arm**: in relative mode IC 0.007 (own), 0.009 (mid), 0.074 (high), 0.038 (group); its predictions collapse to a scale of 1e-8 to 1e-12 (`pred sd` column) versus 2e-7 to 4e-5 for HH_none, so the hyperbolic readout with the graph shrinks outputs by about three orders of magnitude. IC is scale-free, so the collapse itself is not the measured loss, but it signals an optimisation difficulty (all runs use the Phase 1 lr 1e-3, no tuning).
6. **The MSE term is nearly uninformative.** The oracle MSE is 0.9989e-4 against a zero-prediction MSE of 1.02e-4 (own) and 1.085e-4 (high): the planted signal explains 2% and 8% of the return variance. In the 100-epoch level-mode runs the minimum test MSE over epochs is within 1% of the zero-prediction MSE (high/level EE_none 1.08e-4 vs 1.085e-4). This is realistic (real daily signals are weaker), but it means the MSE term has almost no gradient towards the signal and the ranking hinge is the only informative part (see `D_resolved_configs.md` section 3 on its gradient).

## 4. Diagnosis of the level-mode failure (Kaggle preset 7, `ks_long`, `ks_bd1`)

Two training-budget explanations were tested on level inputs, levels `own` and `high`, seeds 0-2.

(a) **R5-like budget: 100 epochs, no early stop** (`ks_long`):

| level | input | arm | n | test IC | IC / oracle IC | IC / no-graph IC | excess Sharpe (top5 - hold) | pred sd |
|---|---|---|---|---|---|---|---|---|
| high | level | EE_hyper | 3 | -0.003 +- 0.000 | -0.01 | -0.01 | +0.227 +- 2.161 | 2.03e-04 |
| high | level | EE_none | 3 | +0.052 +- 0.006 | 0.19 | 0.20 | +2.859 +- 0.654 | 2.60e-04 |
| high | level | HH_hyper | 3 | +0.014 +- 0.016 | 0.05 | 0.05 | +2.253 +- 1.564 | 2.88e-05 |
| high | level | HH_none | 3 | +0.049 +- 0.022 | 0.17 | 0.18 | +8.894 +- 4.532 | 7.87e-05 |
| own | level | EE_hyper | 3 | +0.004 +- 0.001 | 0.02 | 0.02 | +1.338 +- 0.290 | 6.06e-12 |
| own | level | EE_none | 3 | +0.020 +- 0.011 | 0.13 | 0.13 | +2.132 +- 1.083 | 1.22e-04 |
| own | level | HH_hyper | 3 | -0.003 +- 0.001 | -0.02 | -0.02 | -0.173 +- 0.981 | 1.51e-09 |
| own | level | HH_none | 3 | +0.011 +- 0.008 | 0.08 | 0.08 | +4.593 +- 0.496 | 2.52e-05 |

(b) **One day per optimizer step, as in the STHAN-SR/RSR repos** (`batch_days 1`, 30 epochs, no-graph arms; 740 steps per epoch):

| level | input | arm | n | test IC | IC / oracle IC | IC / no-graph IC | excess Sharpe (top5 - hold) | pred sd |
|---|---|---|---|---|---|---|---|---|
| high | level | EE_none | 3 | +0.034 +- 0.017 | 0.12 | 0.13 | +2.023 +- 2.128 | 2.19e-04 |
| high | level | HH_none | 3 | +0.029 +- 0.016 | 0.10 | 0.11 | +6.504 +- 3.283 | 6.89e-04 |
| own | level | EE_none | 3 | +0.008 +- 0.010 | 0.06 | 0.06 | +1.489 +- 0.987 | 2.20e-04 |
| own | level | HH_none | 3 | +0.015 +- 0.013 | 0.10 | 0.10 | +3.268 +- 1.355 | 3.78e-04 |

Neither fixes it: IC stays 0.00-0.05 (<= 20% of the oracle); the 30-epoch table above is not an under-training artifact of the epoch count or of `batch_days 8`. What is established: the signal is present in the inputs (probe), the generator and evaluator are sound (null control, oracle Sharpe), relative inputs learn (Table B). What is not established: *why* level inputs fail. Candidates consistent with the data, none isolated: the inputs are unstandardized levels around 0.5-1 so a 1% day-to-day change is a small perturbation of a large offset (CLAUDE.md already notes inputs sit near the Poincare ball boundary for the hyperbolic arms; EE_none uses no ball and also fails); weight decay 5e-4 and clipping 1.0 with `lr 1e-3`. I did not run an ablation that separates them.

## 5. Implications and what I did not test

- **R5_g2 and the small-scale g2 arms with level inputs** (the Phase 1 headline protocol, plus `norm: paper`, which is also level) used a setup that, on this test, cannot detect a strong planted signal. Their null results (IC about 0, NDCG at random) are therefore uninformative about the model comparison. The small-scale `rel_*` runs (relative inputs) are the ones where learning is possible, but even there THINK is the arm that learns least in this test.
- The planted signal is linear in the previous return; it does not test whether THINK can learn nonlinear or longer-horizon structure, nor real-data effects (heavy tails, regime changes). It is a necessary check, not a sufficient one.
- Not tested: tuned lr/alpha per arm on the synthetic data, longer training in relative mode, a residual self term in the graph layer, standardized level inputs.
- `tests/test_known_signal.py` (2 tests, about 30 s CPU): the planted generator gives oracle IC > 0.25 with own-only < oracle, and EE_hyper / HH_hyper in relative mode reach IC > 0.1 on a tiny strong-signal market (8 epochs).
