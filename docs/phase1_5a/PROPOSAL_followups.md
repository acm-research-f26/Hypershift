# Phase 1.5a follow-up proposal (costed, gated; nothing here has been run)

Decision for the post-2017 question (2026-10-03): carry out the CPU data-compatibility gate first; do not allocate GPU while the only local later-year panel is a current-S&P survivor subset with incompatible adjusted-price semantics. The planned strict replication uses five seeds x 100 epochs, about 3.3 serial-equivalent GPU-hours plus overhead, because 40 epochs changes the historical selection horizon. See docs/superpowers/plans/2026-10-03-post2017-frozen-sharpe.md.

Context: `docs/phase1_5a/REPORT_2017.md` finds that the 2017 Sharpe near 2 of `R5_f2_alpha0_train/HH` is not distinguishable from random 5-stock selection (F1-F4 not rejected, Holm 0.32 to 0.64), with low power (seed-0 Sharpe CI 0.28 to 3.48). A genuine weak signal (hypothesis D) is neither supported nor excluded. Every item below needs training or data that Phase 1.5a did not use, so it is a separate decision. Costs are estimates (Kaggle P100 about 24 s/epoch for THINK full NYSE; laptop RTX 3050 is for small jobs only). Every long run must checkpoint, save per-epoch predictions and logs, and run without any Claude session or watcher.

Updated gate for the post-2017 question: verify compatible data access and the CPU specification first. Item 1 training then supplies saved weights for later-year inference; the original five x 40-epoch option below is retained as the historical estimate, while the strict post-2017 plan specifies five x 100 epochs.

## 1. Instrumented replication of R5_f2 alpha=0 HH

- **What:** 5 seeds x 40 epochs on Kaggle, with a `loop.py` option that saves `state_dict` and test predictions every epoch and a resumable checkpoint.
- **Cost:** about 24 s/epoch on a P100, 16 min per seed, about 1.5 GPU-h including the smoke run (1 epoch of each job type first). Code change about half a day plus a unit test.
- **Prerequisites:** the `loop.py` option and its test; a new exp name (never write into `results/R5_*`).
- **Information value:** enables the per-epoch basket, margin and tie trajectory, the through-model index permutation test, and a frozen-weights artifact.
- **Can claim:** behaviour of a *replication* with the same config. **Cannot claim:** anything about the historical run (different random streams), or skill.

## 2. Post-2017 data-compatibility specification (CPU, about 1 day of work)

- **What:** a written and tested spec covering ticker/identity mapping, delistings and mergers; close vs adjusted-close semantics, MA5/10/20/30 and return definition; causal normalisation with a test that future prices cannot change earlier features; survivorship quantification.
- **Then (only after the spec passes):** a first-pass 2018-2023 evaluation of the item 1 model: per-year Sharpe and concatenated-daily Sharpe, never the mean of annual Sharpes; a frozen graph and a historically updated graph kept separate.
- **Cost:** about 1 day CPU, no GPU for the spec; the evaluation is inference only after item 1.
- **Information value:** the only route to more than 237 days of out-of-sample evidence for the same model; it is what would raise power.
- **Can claim:** whether the replication's ranking generalises to later years under a stated data convention. **Cannot claim:** that the original run generalises (it is a different training run), or anything before the spec's tests pass.

## 3. Walk-forward retraining

- **What:** a separate experiment, about 6 windows x 5 seeds x about 40 min.
- **Cost:** about 20 GPU-h (Kaggle, split across sessions).
- **Prerequisites:** items 1 and 2.
- **Information value:** independent-in-time estimates of Sharpe and IC with fresh training per window; the first design in which seeds and years are not tied to one 237-day sample.
- **Can claim:** out-of-sample behaviour of the protocol. **Cannot claim:** attribution to hyperbolic geometry or hyperedges (needs item 4).

## 4. Architecture and geometry controls

- **What:** EE, temporal-only, no-graph, pairwise, hypergraph and simple baselines, plus a hyperbolic-op audit against the paper (`docs/phase1_5/B_model.md` is the start), each with the same tuning budget and seeds.
- **Cost:** order of 10 to 30 GPU-h depending on the arm count; do not start before evaluation is trusted (items 1-3).
- **Information value:** whether any THINK-specific component matters once the evaluation is credible.
- **Can claim:** relative effects under the validated protocol. **Cannot claim:** anything about the authors' original results.

## Not proposed

- A new weight-decay control: `docs/phase1_5/F_learnability.md` already answers it narrowly (planted-signal learnability), and the real-data change also involved input mode, loss and seed count.
- More seeds on the 2017 sample alone: seeds share the days, ordering and evaluator, so they add repeated runs, not independent evidence.
