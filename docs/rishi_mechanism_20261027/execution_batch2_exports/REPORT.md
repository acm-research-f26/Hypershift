# Original saved forecasts: account replay and stock-signal controls

2026-10-27. Publication865dbb3; protocol0a70456 and EXPORTED_FORECAST_STAGE amendmentfd9796e. This stage successfully reconstructs the saved-export strategy. Selected-checkpoint numerical regeneration remains unresolved and is a separate gate.

## What was verified

All339contract files and1184published artifacts match their hashes. Canonical preparation reproduces saved axes, scalers and annual splits. All18published B(parent), D(restricted dynamic) and E(50:50rank blend) accounts reproduce daily returns (maximum error9.99e-17), NAV, aggregate cash/holdings/cost/settlement ledgers and metrics. Fifty-four fixed account cells were evaluated across three seed groups and two holding counts. No neural fitting, original-artifact edits or protected-window scoring occurred.

## Fixed ten-seed results

| Candidate | Top5 Sharpe | Top5 CAGR | Top10 Sharpe | Top10 CAGR |
|---|---:|---:|---:|---:|
| Parent B | .531 |16.73%|.519|15.36%|
| Dynamic D |.657|20.53%|.530|15.32%|
| Equal rank blend E |.896|31.87%|.755|23.92%|
| Training-label stock means Hlabel |.453|14.00%|.480|14.77%|
| Parent + Hlabel, equal ranks |.621|21.91%|.675|22.71%|
| Five-session reversal |.259|5.93%|.246|5.81%|
|126-session momentum |.327|8.59%|.145|2.94%|
| Fixed ridge on latest five daily channels |.220|3.09%|.121|−.01%|
| Fixed ridge on six coverage/activity channels |−.101|−4.48%|−.113|−4.01%|

Hlabel averages only valid training labels per stable ID (minimum30observations; frozen cross-ID median fallback). It never reads future labels. The blend applies the same percentile-rank and execution rules. Fixed ridge controls are narrow untuned controls; their weakness does not exclude nonlinear activity signals. Top5 and top10 are separately frozen cells, not a license to select whichever supports an interpretation.

## Primary daily contrasts

Ten seeds/top10, annual arithmetic mean differences, paired stationary bootstrap10000draws, expected block21sessions, Holm across two primary tests:

| Contrast | Difference/year |95%interval | Holm one-sided p |
|---|---:|---:|---:|
| E−B |+7.04pp|[−.22,+14.42]pp|.0604|
| E−(B+Hlabel)/2 |−.42pp|[−11.35,+10.10]pp|.5362|

Neither passes the frozen5%family threshold. The second interval is far wider than the±2pp equivalence margin: similar CAGRs do **not** establish equivalence or prove identity explains the model. E−B passes at block63only (Holm.0400), but not the primary21or sensitivity5(.0898); no promotion of the sensitivity endpoint. E−identity remains unresolved at every block.

The ten-seed mean RankIC is B.00377,D.03072,E.01889,Hlabel.01382,REV5.01827. Highest RankIC and highest portfolio performance belong to different cells. Thus broad cross-sectional accuracy alone cannot explain the concentrated rank-blend return.

## Checkpoint gate and interpretation

First parent2022/seed7 regenerated CPU errors reach.00308084percentagepoints on test; source-config CUDA on local hardware reaches.00151825pp, both above1e-6pp. Prepared input/scaler reconstruction is exact; disabling MKLDNN does not fix it. First CPUparent eligible ranks correlate1.0 and no top5/top10 origin changes. Twenty dynamic CUDA checkpoints pass before2024/seed7fails validation at1.54972e-6pp (test8.64267e-7pp); that failure also changes no eligible ranks or top-k allocations. We retain the strict gate, failed receipts and partial results; do not claim all80checkpoints reproduced. Hardware-sensitive numerical behavior is a hypothesis, not a resolved cause.

Full training-prediction Hpred and within-ID changing-score controls were therefore not produced. Saved exports independently support account reconstruction and Hlabel comparisons; they do not validate regenerated training predictions or causal component explanations.

Subsequent stock trace is in ../execution_batch3_trace/REPORT.md. Its result supersedes the RESULTS.json historical note that per-stock replay was still needed. Parent artifacts contain aggregate ledgers, not original broker fills. Raw15minute source chunks and first-provider-publication timestamps remain unavailable, limiting independent latency certification. All2022–2025analysis remains exploratory and intervals omit historical search/refit uncertainty.
