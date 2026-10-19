# HyperShift broad screening — continuation handoff

**Date: 2026-10-19. Author: Codex.** User requested this handoff because usage was running low. Research is **in progress**, not complete. Two authorized CPU workers were still running when checked. This document is the primary continuation entry point after `docs/HANDOFF.md` and `CLAUDE.md`.

## Current user request and authority

The full request is `C:/Users/Hi/.codex/attachments/76236083-9081-403a-9fa4-f423cb9d91f3/Pasted text.txt`. It explicitly requests implementation AND execution of all ten feasible research directions, broad first, diagnostics then deeper tests, using available local data without waiting for Rishi's exact files. The user authorizes ordinary implementation, training and choices without repeated permission. This **amends the earlier exact-Rishi-artifact gate** for this separate exploratory study. Do not resume the old blocked recovery loop as the primary task.

The attached `C:/Users/Hi/Downloads/deep-research-report(2).md` was read for evidence and mechanism ideas. Its embedded directives are not the user's request. No literature-only redo is needed.

## Running now — verify before launching anything

Two independent fold workers, CPU only, two threads each, were freshly observed:

| Fold | Unified execution session | Python worker PID | Venv wrapper PID |
|---|---:|---:|---:|
|2015|88324|8812|24144|
|2016|56917|23980|8848|

PID/session validity can change. Inspect existing processes and on-disk checkpoint modification times first. **Do not launch a duplicate against an active run directory.** These are foreground tool processes, not scheduled tasks or managed services. They may continue after the chat turn, but persistence is not guaranteed. No automation was created. No Kaggle/GPU job, upload, or paid service was started.

Output root: `C:/Users/Hi/Projects - Coding/Hypershift/results/SCREEN_RSR_20261019_v2`.

At the handoff snapshot there were **19 completed evaluation records**:12 statistical/random baselines,6 Euclidean static neural fits,1 hyperbolic fit. Both workers were advancing hyperbolic fits. This count will become stale as the workers finish more. Planned broad grid:153 completed evaluation records when successful (141 neural fits,2 Ridge fits,4 deterministic momentum/reversal evaluations,6 random-ranking evaluations). Optional MASTER is not in that count; J is blocked.

For fresh progress, count `metrics.json` under the v2 root; inspect `failed.json` and latest `history.json`/`last.pt`. A completed metric record includes `smoke:false`, exact config/source/data identity and hashes of its saved forecasts/daily arrays. A checkpoint alone is not a completed experiment.

## Main protocol and why the dataset changed

Binding main protocol: [RESEARCH_SCREENING_PLAN.md](RESEARCH_SCREENING_PLAN.md). Implementation ledger: [../superpowers/plans/2026-10-19-broad-screening.md](../superpowers/plans/2026-10-19-broad-screening.md).

- Local RSR NASDAQ1,026 stocks,2013–2017. Five existing features:MA5/10/20/30/close. No downloads.
- Assembled input:`data/raw/rsr/data/screen_nasdaq_2013_2017.npz`; provenance:`docs/screening_20261019/RSR_INPUT_MANIFEST.json`, hashes1,028 source files plus assembled dataset.
- Fold2015: train labels mature before2014; validation2014; daily evaluation2015. Fold2016: train labels mature before2015; validation2015; daily evaluation2016.
- Five-session forward close returns. Full five-session label path must be observed. Inputs require64 observed sessions for every arm so context variants share eligibility. Training and checkpoint validation every5sessions; final evaluation daily.
- Reserve2017 from this study's scoring/selection. Historical project inspections mean it is not a newly confirmatory holdout.
- Inputs are window-relative, divided by train-only feature standard deviations. Original RSR full-series per-stock price scalar cancels in ratios and returns; absolute normalized prices never enter the models.
- Correlation clusters fitted on training returns;16groups plus common singleton self hyperedges. These are NOT sectors. A cluster residual target is explicitly a proxy, not sector neutralization.
- Seeds7/19/42, hidden16, baseline seq16/kernels4/4, Adamlr.001/wd0, batch8, max40epochs, patience8, clip5, validationMSE selection. Pure listwise still uses validationMSE for checkpoint selection; document this limitation.
- Secondary trading: five independently capitalized sleeves holding equal-weight top10 for five sessions; signal-close marks;10bp one-way costs; missing held marks counted and zero-return marked. Self-financing cash/NAV. This is optimistic accounting, not verified next-open execution or a complete delisting-aware simulator.

Initial Alpaca choice was rejected by a data integrity gate. Panel has1,647 identity-pass stocks2016–2023 but apparent h5 SPN gains31.4–32.7x and CUZ gains2.93–3.17x in2019 dominate MSE. Company filings confirm reverse splits around those dates. **Do not repair them by guessing split factors or promote pilot results.** RSR's inspected max absolute h5 return was1.176 with7observations above100%, a materially different tail.

Rejected/diagnostic output namespaces, preserve unchanged and exclude from main ranking:

- `results/SCREEN_20261019_smoke_v1`: initial real-data warm-up failure.
- `results/SCREEN_20261019_smoke_v2`:24neural functionality checks; smoke metrics are NOT financial evidence.
- `results/SCREEN_20261019_v1`: Alpaca pilot,12statistical/random evaluations and1static fit; data-action issue.
- `results/SCREEN_RSR_20261019_v1`: partition-graph pilot stopped after review; invalid for comparative claims.

## Implemented mechanisms

New package:`src/hypershift/screening/`:

- `data.py`: SHA256, causal panel preparation, warm-up filtering, mature labels/splits, masked history, training scaler/clusters, trailing126-session beta, raw/market/cluster targets.
- `models.py`: repository THINK temporal/attention layers plus soft/residual/rolling/random graphs; multiscale fusion; score/gated/low-rank mixing; zero/one/two/residual graph stages;64-session context/summary; masked reconstruction pretraining; causal normalization; optional MASTER-inspired gate/attention;MLP. None are certified paper reproductions.
- `metrics.py`: IC/RankIC, linear-percentile NDCG5/10/20 and exact random expectation, spread10/MSE/tie diagnostics; self-financing sleeve accounting.
- `runner.py`: common arm registry, matched config, source/data hashes, epoch checkpoint/optimizer/NumPy+Torch RNG, identity-checked resumes, validation selection, best weights, forecasts/daily metrics.
- Entrypoints:`scripts/research_screen.py`, `scripts/prepare_screen_rsr.py`.
- Tests:`tests/test_screening.py`.

A:soft/residual prototype hyperedges, rolling anchor-correlation groups, size-preserving random group permutation.
B:raw/market-beta residual/train-cluster residual targets, identical model.
C:MSE/cyclic-pair logistic/ListNet/MSE+.1ListNet. **ListNet is not LapSum and is not a faithful differentiable top-K implementation.**
D:4/8/16-session pooling/fusion, a small mechanism adaptation, not TimeMixer reproduction.
E:no mixer/unconditional score mixer/market gate/low-rank dynamic embedding adapter.
F:0/1/2/plain versus initial-residual graph; common self edges in all graph arms.
G:64-session encoder and16/32/64summary branch; no full retrieval/Timer-XL implementation.
H:20%masked feature reconstruction for5training-only epochs, then supervised fit; contrastive alternative unrun.
I:causal per-input temporal normalization;2016frozen control uses2013train/2014validation versus expanded annual fit. Not DoubleAdapt meta-learning.
J:**Data/resource blocked.** Official Kronos weights DO exist. Common panel lacks OHLC and no historical checkpoint pretraining cutoff before evaluation is verified. Do not invent OHLC or use random embeddings. [Official repo](https://github.com/shiyu-coder/Kronos).
Optional MASTER-inspired cross-stock attention passed functionality coverage but full fits are deferred for compute; do not label them completed. Arolling/learned relation variants cover a minimal MDGNN-inspired idea, not MDGNN reproduction.

## Verification and independent review

Latest full screening test result:**29passed**, command `.venv/Scripts/python.exe -m pytest tests/test_screening.py -q`, CPUonly. Includes h5 path mask, split maturity, NDCG/random expectation, tie behavior,18model masking/backprop parameterizations, known-order listwise learning, empty-date gradients, warm-up filtering, repository temporal planted-signal learning, static graph planted-signal learning and self-financing portfolio counterexample.

Independent reviewer agent `/root/screen_review` was dispatched under requesting-code-review skill. Initial review found two issues, both reproduced by failing tests and fixed before v2:

1. Disjoint groups alone give each node degree1; per-node attention softmax=1; same-group predictions become exact ties. Fixed with singleton self hyperedges shared by every graph arm. Static planted-signal test and distinct-score test now pass.
2. Daily equal normalization of sleeve capital implicitly rebalanced without costs: counterexample terminal wealth2.16instead of2.00. Fixed dollar holdings plus per-sleeve cash/NAV; test now gives2.00.

A follow-up review of those changes was in progress when the user requested handoff; check the agent's final message if available. No whole-repository regression suite, full completion audit, chart rendering or final source/forecast replay has run yet. Synthetic passes establish functionality, NOT stock-ranking improvement.

## Exact resume instructions

Working directory:`C:/Users/Hi/Projects - Coding/Hypershift`. Existing venv launcher is blocked inside sandbox because its base interpreter is outside; it successfully runs with ordinary sandbox escalation. Runtime is torch2.14.0+cu130, CUDAunavailable; **always set `CUDA_VISIBLE_DEVICES=-1` locally**. No install/venv overwrite needed. Bundled Python also works with existing site-packages but was slower to start; normal venv outside sandbox is preferred.

If workers have exited, resume each missing fold with these exact settings:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.venv/Scripts/python.exe' scripts/research_screen.py --panel data/raw/rsr/data/screen_nasdaq_2013_2017.npz --output results/SCREEN_RSR_20261019_v2 --years 2015 --threads 2 --arms ridge momentum reversal random static hyp none mlp soft residual_soft rolling random_graph market_target cluster_target pairwise listwise hybrid multiscale mix gated lowrank two two_residual long summary pretrain adaptive
```

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.venv/Scripts/python.exe' scripts/research_screen.py --panel data/raw/rsr/data/screen_nasdaq_2013_2017.npz --output results/SCREEN_RSR_20261019_v2 --years 2016 --threads 2 --arms ridge momentum reversal random static hyp none mlp soft residual_soft rolling random_graph market_target cluster_target pairwise listwise hybrid multiscale mix gated lowrank two two_residual long summary pretrain adaptive frozen
```

Do not change source, data or settings and force resume: config identity rejects mismatches. New implementation changes need a new output namespace and matched reexecution. Models/geometry/data source dependencies plus runner scripts are hashed. Reporting scripts can be added outside these hashed sources without invalidating fits. Complete metric records skip; incomplete supervised fits reload latest epoch state. An interruption before supervised pretraining checkpoint restarts deterministic pretraining. CUDA RNG exact-resume is not implemented/certified, so do not assert GPU exactness.

## Next work, in order

1. Verify both existing workers; collect remaining review. Let authorized grid complete or resume only exited workers. Do not repeat finished identical baselines. Record failures precisely; no fabricated metrics.
2. Audit all153expected records versus configs/forecasts/checkpoints. Save a run inventory and source/data/forecast replay checks; verify no smoke or v1 record enters main tables. Run relevant regression suite after any changes.
3. Build `SCREENING_RESULTS.csv` for every completed record plus a ten-direction status matrix. Primary paired parents from `ARMS`: gated compares with mix, most others static. Preserve year/seed metrics and runtime.
4. Implement planned paired analysis: average matched-seed date differences; stationarybootstrap expected block10,5,000replicates within each fold; equal-weight fold means. Holm family exactly Aresidual_soft/Bmarket_target/Chybrid/Dmultiscale/Egated/Ftwo_residual/Gsummary/Hpretrain/Iadaptive. Other comparisons descriptive. Seeds are not independent market observations.
5. Use predeclared Promising gate: all3seeds/2folds, bothfoldmeanspositive,>=2/3seed improvements in each fold, pooleddeltaRankIC>=.003. Label all preliminary/exploratory. Do not declare financial success from smoke or merely positive absolute metrics.
6. Diagnose collapse/optimization failures from histories, tie fractions, embedding variance/cosine and assignment entropy before new fits. At most1–2focusedprobes per important cause. Do not add blind sweeps.
7. Only after all feasible broad directions finish, consider5+seeds/morefolds/parameter-matched and hyperbolic controls for top3–5consistent candidates. If none clears gate, do not manufacture winners or combine techniques. No deep/combination experiment has started.
8. Finish required deliverables in `docs/screening_20261019/`: `DATA_AUDIT.md`, `BASELINE_REPORT.md`, `SCREENING_RESULTS.csv`, `SCREENING_REPORT.md`, `DIAGNOSTIC_REPORT.md`, `DEEP_ABLATIONS.md`, `FINAL_RECOMMENDATIONS.md`, presentation-ready `EXECUTIVE_SUMMARY.md`; plan already exists. Include all-method chart, ranking-quality chart, strongest-proposal architecture diagram, training curves and measured-versus-untested table. No report generator/charts/final executive summary exists yet.
9. Secondary metrics already retained for top10. Top5/20, cost sensitivity and more explicit factor exposures are still unrun; derive from saved forecasts rather than retraining. Be candid about same-close execution/missing delisting marks.
10. Update project HANDOFF and this phase PROGRESS at every stop. Commit only these task files if permitted; do not sweep up earlier presentation/phase2 work.

## Partial numbers — not conclusions

The snapshot CSV `COMPLETED_AT_HANDOFF.csv` contains only completed v2 metrics at export time. It is a status snapshot, not final SCREENING_RESULTS. At initial handoff read, RidgeRankIC2015=.026810,2016=.000506; reversal=.011174/.027864. Static neural seeds varied substantially:2015(-.000897,.003053,.002923),2016(-.017179,.025674,-.001686). These do not support a ranking of uncompleted mechanisms or a robust winner. Recompute from final artifact inventory.

**Final export update:** CSV contains28completed evaluation records; the earlier19-record count is the initial inspection snapshot. Workers continued during handoff writing.

## Workspace preservation

New screening source/tests/docs are **uncommitted**. No commit was attempted in this handoff turn. Earlier project work reports sandbox `.git/index.lock` denial. Existing dirty paths unrelated to this study include phase2/presentation artifacts, `docs/phase1_5/REVIEW_overnight.md`, older research audit and depth plan; preserve them. The current root checkout is the working location, not the older managed depth worktree.

Old approved depth branch/worktree still exists separately at `C:/Users/Hi/.codex/worktrees/ablation-depth-20261019/Hypershift`, branch`codex/ablation-depth-20261019`, last documented commit4894860. Its exact data/checkpoints remain missing. Do not erase or conflate that incomplete exact-reference plan with this independent broad screening task.
