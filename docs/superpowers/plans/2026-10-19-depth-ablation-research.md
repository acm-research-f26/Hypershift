# HyperShift depth ablation implementation plan

> **SUPERSEDED for Track B (Ethan Alpaca) by `2026-10-20-depth-ablation-two-track.md`.** Track A (Rishi Nasdaq) Tasks 2–6 below remain valid once its prerequisites pass.

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task after the user's approval. Execute inline; no additional agent delegation is required. Steps use checkbox syntax.

**Goal:** Run one bounded, reproducible exploratory depth study that isolates graph propagation and its essential controls without repeating completed history.
**Architecture:** Extend pinned feat-rs's masked historical monthly engine and THINK model with configurable depth and weighting, an exact topology placebo, and a no-graph score mixer. Preserve source-era outputs and reuse verified H1 forecasts. Keep new resume, analysis and accounting artifacts separate from historical outputs.
**Tech stack:** Existing Python/PyTorch/NumPy/pandas environment on Kaggle for substantive fits; Windows CPU-only for tiny fixtures and metadata.
**Spec:** ../../research_20261019/RESEARCH_AUDIT_AND_FIRST_BATCH.md, Sections3–7. This plan and all budget values are proposed, not execution approval.

## Global constraints

- User approval and required data/runtime prerequisites precede implementation and execution.
- New research branch proposed: codex/ablation-depth-20261019 from d83c4a4c4711d826b8ac6757e01d4fd32d8760a8; preserve existing checkout edits.
- GPU work on Kaggle, full batch only after its successful smoke; no protected confirmation access, trades or purchases.
- New output directory only; seven fixed arms T0/H1/H2/H3/U1/R1/M0, h5, lookback16, hidden16, kernels4/4.
- Seeds7/19/42/61/97; pilot7/19/42 in2022 fold; all old dates exploratory.
- AdamW.001/wd.001, clip1,batch8 chronological dates,30epochs/patience6, validation MSE improvement>1e-7; pure MSE for all seven.
- Maximum12GPU-hours including pilot/smoke; maximum120new fits if20 H1 reusable, otherwise140 with prior documented approval of the revised count and budget feasibility.
- Exact data and protocol hashes are required inputs, not values to invent.

## Review focus

1. Later/inactive nodes, including NaNs, must not change active predictions or scaler statistics.
2. Initialization and serialization changes must preserve a loaded historical depth1 model's scores and existing account output.
3. Depth/rewiring/membership cannot silently change per-date eligible populations or date/ID alignment.
4. A crash mid-epoch cannot silently restart with different randomness or treat incomplete predictions as completed.
5. Full-horizon labels and final terminal-account marks have different support; neither may leak across chronological boundaries.

## Task 1: Restore and freeze the matched inputs

**Files:** Create config/depth_ablation_20261019.json, ablation_spec.py, test_ablation_spec.py; inspect existing data/nasdaq_monthly_pit/assembled, all dependency modules and the source INPUT_CODE_LOCK.
**Interfaces:** Frozen AblationSpec holds arm,depth,weighting,graph_control,mixer_strength,hidden,lookback,kernels,seeds,years,epochs,patience,lr,weight_decay,batch,output,code_hashes,data_hashes,budget. validate_spec(spec) returns the validated spec or raises; hashes identify every referenced file and input order.

- [ ] Write fixtures rejecting reused output paths, missing/mismatched hashes, overlapping split labels, unstable security IDs, future membership and a changed scaler.
- [ ] Run the fixture tests; observe failures before adding the guard implementation.
- [ ] Implement spec validation and create-only manifest serialization; expose data availability and reviewed source limitations explicitly.
- [ ] Reconcile H1's20existing checkpoint/prediction/scaler/fold/graph records against exact code/data hashes; count reusable jobs only after successful comparisons.
- [ ] Verify fixture passes and write a prerequisite verdict listing blocked inputs. Stop if any required source is unavailable.
- [ ] Commit only the new spec/validation files; never stage old results or unrelated edits.

## Task 2: Depth, weighting and matched mixing

**Files:** Modify think_model.py, research_models.py, robust_models.py; create test_depth_ablation.py.
**Interfaces:** ThinkReconstruction adds spatial_depth:int=1 and weighting:str='learned'; RankModel accepts the same settings plus mixer_strength:float. Factory receives the frozen spec. Old calls preserve depth1 behavior.

- [ ] Write tests: depth0 bypass; depth1 loaded historical state gives identical scores;0/1/2/3 shape and finite gradients; independently perturbed batch date cannot affect another; inactive NaNs cannot affect active nodes.
- [ ] Assert common encoder/head initial parameters are identical across depths using separate deterministic component streams; adding a layer must not redraw the head.
- [ ] Implement independent ModuleList spatial blocks, explicit geometry conversions and masked uniform incidence means. Preserve original layer-one centroid and learned coefficient computation.
- [ ] Test a synthetic incidence matrix with node incident counts1/2/3; uniform coefficients sum1over active incidences. Inactive incidences receive0; record isolated-node behavior explicitly.
- [ ] Implement T0+N→16→N market correction as M0 with input/output masks and the same pure MSE; test strength0equals T0 exactly.
- [ ] Run targeted tests then the relevant existing THINK/membership tests in the restored environment; commit only these source/tests.

## Task 3: Exact graph placebo and diagnostics

**Files:** Create ablation_graph.py, ablation_diagnostics.py, test_ablation_graph.py; modify model's optional diagnostic return path.
**Interfaces:** rewire_incidence(incidence,seed,attempts) returns (incidence,report); diagnostics are detached summaries, never used to update weights.

- [ ] Write independent assertions for exactly preserved node degrees/edge sizes, no duplicate incidence, deterministic output and unchanged node order.
- [ ] Implement double-edge switches with fixed per-fold seeds; require nonzero accepted switches and changed-incidence fraction≥.1for this placebo. If topology prevents that, fail the placebo feasibility gate without changing its rule after returns.
- [ ] Test insufficient-switch failure, singleton/masked-edge behavior, and diagnostic calls leaving predictions/gradients unchanged.
- [ ] Save per-layer active-node variance/cosine summaries, coefficient sign/magnitude/clamp diagnostics, unclipped/clipped gradient norms and runtime/memory. Don't label signed raw coefficient statistics “probability entropy.”
- [ ] Verify targeted tests and commit source/tests.

## Task 4: Deterministic resume and bounded Kaggle launch

**Files:** Create run_depth_ablation.py, ablation_checkpoint.py, test_ablation_resume.py, kaggle/run_depth_ablation.py, smoke preset and full preset in the chosen Kaggle runner.
**Interfaces:** run_job(spec,fold,seed) returns a completion manifest; last-state contains model,optimizer,RNG,batch position,epoch,best-state,scaler,graph and all identity hashes. Completion requires verified arrays and histories.

- [ ] Write interruption fixtures at mid-epoch and immediately after best-state save; resumed predictions and updates must equal uninterrupted execution. A changed input/code/spec hash must refuse resume.
- [ ] Implement atomic checkpoint writes, exclusive per-run writer lock, partial-output tracking, immutable failure records and completion verification.
- [ ] Implement independent stage commands validate/smoke/pilot/run/analyze. The training command must not calculate test performance each epoch; test forecast publication follows fixed family checkpoint locks.
- [ ] Add actual accelerator timing, peak memory and output-byte accounting; estimate remaining cost before each job. Enforce12hour and5GiB limits plus20%device-memory headroom.
- [ ] Run tiny CPU tests with CUDA_VISIBLE_DEVICES=-1; after approval launch Kaggle one-epoch smokes through a recorded smoke/full gate. Do not invoke existing scripts/time_budget.py as a harmless estimate.
- [ ] Publish manifest, timing/memory and projected total before pilot; commit launcher/recovery changes.

## Task 5: Paired evaluation and independent accounting

**Files:** Create ablation_evaluation.py, audit_depth_ablation.py, test_ablation_evaluation.py.
**Interfaces:** evaluate_family(manifest) writes per-seed forecast metrics, daily ledgers and paired contrasts; independent audit reads raw outputs without importing numerical ledger helpers.

- [ ] Write small hand-calculated IC/RankIC/NDCG-percentile/tie fixtures, and distinguish forecast-date observations from daily returns.
- [ ] Write signal-close-unit/next-close-fill/cancelled-fill/fees/capital/annual-continuity/action fixtures; require old H1 account equality.
- [ ] Implement the spec's three primary contrasts and three mechanism contrasts,5000joint stationary-block draws within years,20session expected blocks,10/40descriptive sensitivities, Holm families, and explicit five-seed exact-test limits.
- [ ] Test identical arms give zero contrasts; seed averaging cannot multiply market sample size; identical dates/node masks enforced; final incomplete h5label excluded.
- [ ] Produce every seed's result, gross/net wealth/drawdown overlays, ranking/top5/top10 diagnostics, parameters/runtime/memory and code/data/config/output hashes.
- [ ] Run independent reconstruction checks; retain all failures and verdict uncertainty. Commit source/tests and bounded report metadata only.

## Task 6: Execute only the approved stages and report

- [ ] Verify written approval and Task1prerequisites; repair deviations before launch.
- [ ] Run synthetic correctness, successful real-size smoke, then18new pilot fits (or documented21); do not select depths from2022returns.
- [ ] If gates and remaining budget pass, complete102remaining new jobs under exact H1reuse, otherwise report the precise limited execution without claiming completion.
- [ ] Report what each arm changed, expected mechanism, actual result, reproducibility, evidence sufficiency and retain/modify/remove recommendation. No architecture promotion from a favorable seed/test SR alone.
- [ ] Update docs/HANDOFF.md and active PROGRESS.md; retain the original audit and append any approved protocol amendments before outcomes are inspected.

## Self-review

The initial audit covers existing work, unresolved questions, code changes, compute envelope, prerequisites and first batch. Execution is bounded to depth and three essential controls. Lookback/geometry/loss/sector-size/residual/capacity follow-ups are specified as later questions, not silently added to this batch. Prediction quality, implementation economics and independent confirmation remain separate. Input hashes and runtime are explicit blockers; nothing substitutes invented market records.

