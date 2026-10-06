# Performance table: meaning and independent scoring check

**THINK Table II is not independently verified.** This report supplies a similar table for our existing experiment, with scores freshly recomputed from saved predictions and the original historical price snapshot. It does not relabel those models as THINK or use their scores to confirm or refute its benchmark.

## What the screenshot measures

The metric varies by column. The geometry rows describe the dataset, whereas the model rows assess prediction or trading outcomes. The task/metric mapping and 25-run count come from [THINK, Section IV and Table II](https://tylersnetwork.github.io/papers/icdm22-think.pdf).

| Columns | Metric | Interpretation |
|---|---|---|
| DTT, CPox, WMill, Risk (CSE) | MSE, lower is better | Average squared numerical prediction error. Scaling of the target determines the units. |
| NYSE, TSE | SR, higher is better | Mean excess portfolio return divided by its standard deviation. |
| NYSE, TSE | NDCG, higher is better | Reward for putting more relevant stocks higher in a ranked list, relative to the ideal ranking. |
| Clf (NASDAQ) | F1, higher is better | Balance of precision and recall for movement classification; the task includes up/down/neutral. |

F1 is not accuracy: for one class, F1 = 2TP/(2TP+FP+FN). Multiclass F1 also requires an averaging rule (macro, weighted or micro). [F1 reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). NDCG depends on the relevance labels, gain function, rank cutoff and tie handling. [NDCG reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html).

MSE = mean((prediction-actual)^2); it is not a percentage and values across differently scaled targets are not comparable. Sharpe is not a percentage return; annualization, risk-free rate, costs and execution timing must be specified.

For example, the screenshot's 1.18 +/- 4e-3 means 1.18 +/- 0.004, not 1.18 percent. The caption identifies the mean over 25 runs and a Wilcoxon signed-rank significance test at p<0.01 for asterisks. It does not clearly define whether the +/- quantity is standard deviation, standard error or another uncertainty measure. We cannot reconstruct paired significance tests from means and error bars alone.

## Independently rescored local experiment

28 model/seed groups; 168 checks against saved metrics passed. Maximum absolute difference: 1.71e-09. The audit imports none of the original evaluation functions.

Setup: 12 US stocks, 21-session horizon, binary up versus down/flat, 377 test origins from 2024-05-31 through 2025-12-01, with outcomes ending 2025-12-31. Learned models have five seeds; deterministic baselines have one result. The original training used earlier data; this audit does not retrain the models.

| Model | Runs | Macro F1 (up) | NDCG@3 (up) | Net annualized Sharpe (up) |
|---|---:|---:|---:|---:|
| Always up / equal-weight basket | 1 | 0.381 | 0.369 | 1.101 |
| Pooled training up-rate | 1 | 0.381 | 0.369 | 1.101 |
| Per-stock training up-rate | 1 | 0.477 | 0.405 | 0.912 |
| Logistic regression | 5 | 0.468 +/- 0.005 | 0.386 +/- 0.002 | 0.937 +/- 0.102 |
| Independent neural network | 5 | 0.455 +/- 0.011 | 0.378 +/- 0.011 | 0.628 +/- 0.314 |
| Ordinary GNN | 5 | 0.448 +/- 0.015 | 0.362 +/- 0.006 | 0.350 +/- 0.327 |
| Euclidean hypergraph | 5 | 0.448 +/- 0.014 | 0.372 +/- 0.006 | 0.396 +/- 0.295 |
| Hyperbolic hypergraph (teaching model) | 5 | 0.450 +/- 0.014 | 0.373 +/- 0.007 | 0.383 +/- 0.277 |

Here +/- is explicitly the sample standard deviation across the five seeds, not a confidence interval. There are no significance stars. The five fits share a market period and do not represent five independent market histories.

The independent checker rebuilt 21-session actual returns and labels from prices, checked target-date alignment, recalculated macro F1 from class counts, recalculated tie-averaged NDCG, and rebuilt portfolio returns from scores and prices. NDCG uses positive realized percentage return as linear gain and excludes dates with no positive gain. Trading buys the top three scores with fractional allocation over ties, enters one session after the signal, holds 21 sessions, and charges 10 basis points on both entry and exit. There are 18 completed, nonoverlapping baskets. Annualization uses sqrt(252/22) and risk-free return is zero.

**These are different tasks and model implementations from THINK.** Our hyperbolic model's F1 must not be compared numerically with the paper's three-class NASDAQ F1 as a replication result. The local experiment also has a short trading window, overlapping classification labels, a selected surviving-stock universe, and a previously examined test period. Rechecking arithmetic does not cure those research limitations or establish predictive validity.

## What can be verified about the paper now?

| Table II task | Available inputs | Remaining obstacle to numerical verification |
|---|---|---|
| CPox | Public graph and series recovered | Exact preprocessing, splits, hypergraph merging and model settings not recovered. |
| DTT | Both public Twitter variants recovered | Variant, snapshot treatment, training/test setup and processed groups unresolved. |
| WMill | Official source mirror recovered | Weight treatment, preprocessing, split and processed groups unresolved. |
| NYSE | Matching-size source stock panel and relation files recovered | Exact graph, ranking loss, top-k, execution, costs, NDCG and Sharpe conventions unresolved. |
| NASDAQ Clf | Matching-size source stock panel recovered | Neutral-label thresholds, F1 averaging, splits and exact trained model unresolved. |
| TSE | Some source loading code available | Matching full processed panel and groups unavailable. |
| Risk/CSE | Related source references available | Matching processed panel and exact risk targets unavailable. |

The [THINK code repository](https://github.com/shivamag125/ICDM22-THINK) is empty as inspected on 2026-09-28. The authors' per-run predictions, scores and checkpoints have not been obtained. None of their Table II cells or significance claims is marked verified by this local rescoring.

A new independent implementation remains possible: freeze a documented reconstruction, implement the full THINK architecture and comparison models, use chronological train/validation/test splits, and train each for 25 seeds. Save every forecast and portfolio return, explicitly define all metric conventions, and evaluate paired differences with a specified multiple-comparison policy. Such an experiment tests the published method under declared assumptions; matching the original table requires resolving the missing inputs and protocol.

## Reproduce this scoring audit

```powershell
.\.venv\Scripts\python.exe verify_performance_table.py
```

Results: `runs/performance_verification/independent_metrics.csv`, `independent_summary.csv` and `verification.json`. Original training results remain unchanged. See `runs/direction/REPORT.md` for the original experiment and `THINK_AUDIT.md` for the separate geometry audit.
