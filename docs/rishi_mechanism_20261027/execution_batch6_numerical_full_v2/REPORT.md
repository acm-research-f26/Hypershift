# Localize strict checkpoint replay failure

2026-10-27, existing NUMERICAL_REPLAY_AMENDMENT. Fixed parent2022/seed7 weights, original inputs and canonical batch8inference. No fitting, parameter change or tolerance adjustment. Earlier CPU-failed forecasts and all failed attempts remain preserved.

## Repeatability and hardware/precision sensitivity

On local CUDA, complete validation and test inference repeat bit-for-bit. They nevertheless differ from published forecasts: maximum validation error1.43051147e-5pp, test.00151824951pp. Worst test pair is origin186/ADBE; worst validation107/ANSS. Isolating the CPU-worst canonical batch179–185gives exactly the same output as that batch in the full call. The repeated full calls do not support stochastic evaluation in these runs. Only that batch was isolated; the batch containing the full CUDA maximum186was not isolated, so context effects at that maximum are not excluded.

The diagnostic batch containing the **CPU** worst pair(origin185/ADBE) reproduces prior CPU outputs exactly. That batch matches published exports on CUDA within2.38419e-7pp, while CPU32differs by.00308084pp and CPU64by.00073016pp. This local match does not promote the full CUDA forecast gate; the full-call maximum occurs in the next batch. FP64is a diagnostic, not the source reference or a replacement acceptance rule.

## Where local CPU/CUDA differences grow

Maximum output differences in the observed batch:

| Module boundary | CPU32 versus CUDA32 |
|---|---:|
| First temporal block |4.17233e-7|
| Hypergraph spatial block |6.73782e-6|
| Final temporal tensor |2.50340e-6|
| Base scalar score after log0 |.00279617|
| Core including indexed mixer |.00306273|

The spatial representation reaches last-axis norm.99999006, near the configured Poincare boundary. The base score applies log0, whose atanh operation is sensitive near unit norm; the observed amplification is consistent with this mathematical conditioning. The indexed mixer and auxiliary correction then carry the discrepancy into final scores. This localizes the CPU/CUDA sensitivity within the actual checkpoint path. It does not uniquely identify which original-versus-local GPU operation caused the remaining cross-device difference; module traces from original hardware are absent.

Source hardware is RTX4070Laptop, capability8.9; local RTX3050Laptop,8.6. Torch2.14.0+cu130/CUDA13/CuDNN92400, deterministic algorithms and TF32off match. Deterministic local execution does not guarantee bit-identical output across hardware. Do not claim this hardware distinction alone proves the cause. Full80checkpoint reproduction and dependent Hpred/changing-score interpretation remain unresolved. Tested earlier failed cases have unchanged ranks/top-k, but this is narrower than verifying every checkpoint or intervention.

DIAGNOSTIC.json retains per-module shape/precision summaries and both complete-split errors. Its summary fields worst_origin_id/worst_stock are the CPU reference pair185/ADBE for every path, not each path's argmax; FIELD_CLARIFICATION.json preserves this metadata correction without changing original receipts or results. Future code uses explicit reference_cpu_worst names. The separate full_cuda fields correctly identify each complete-split argmax. Norms for score vectors and ordinary auxiliary activations are not Poincare radii; only the named spatial representation supports the radius observation. First full-diagnostic attempt failed before receipt write due variable shadowing; corrected variable name only and reran in this fresh destination. The earlier successful batch-only receipt remains in execution_batch6_numerical. Independent review prompted these field/scope clarifications and found no substantive issue in the rank-behavior/power analyses.
