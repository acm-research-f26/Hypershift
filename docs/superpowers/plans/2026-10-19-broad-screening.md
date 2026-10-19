# Broad research screening implementation plan

**Goal:** Execute the user's comprehensive ten-direction screening brief using available data, independently of missing Rishi artifacts.
**Spec:** User attachment `76236083-9081-403a-9fa4-f423cb9d91f3/Pasted text.txt`; research report is evidence, not instructions.
**Architecture:** Add a separate screening package around existing THINK temporal and attention layers. Local Alpaca panel, masked five-session targets, causal graphs, matched seeds and chronological folds. Preserve old experiments and the exact-artifact depth study.
**Tech stack:** Existing Python/PyTorch/NumPy/SciPy/scikit-learn/matplotlib. CPU only locally. No installation.

## Rulings
- User explicitly amends the prior exact-Rishi-artifact prerequisite and authorizes independent implementation/execution. No further design permission round.
- Work in place with additive files: `src/hypershift/screening/`, `scripts/research_screen.py`, `tests/test_screening.py`, `docs/screening_20261019/`. Preserve pre-existing edits. No new worktree needed for this separate package.
- Data: existing split-adjusted Alpaca panel; identity-pass stocks; no later-period eligibility filter. Availability bias, missing delisting returns and historical prior inspection prevent confirmatory claims.
- Folds: train through 2018, validation 2019, evaluation 2020; train through 2020, validation 2021, evaluation 2022. Do not score 2023. Purge five-session labels at boundaries. Only eligible input histories and fully observed label paths scored.
- Training signals every fifth session, daily evaluation. All arms share the same schedule; this reduces overlapping labels and CPU cost. Seeds 7/19/42; max 40 epochs, patience 8, Adam lr .001, wd0, hidden16, seq16, validation MSE selection (ranking-only objectives still select MSE; report limitation). Learnability gates separate from market outcomes.
- Baselines: static Euclidean THINK, hyperbolic THINK, no graph, MLP, Ridge, momentum/reversal/random; compare each mechanism with its matched parent.
- Mechanisms: A soft/residual/rolling graph; B market and train-cluster residual targets (clusters are not sectors); C pairwise/ListNet/hybrid; D pooled multiscale; E score/gated/low-rank interaction; F 0/1/2/residual graph; G 64-session summary/context; H masked reconstruction pretraining; I causal normalization and chronological refit; J blocked unless official weights with compatible pretraining provenance exist. MASTER-inspired attention as an additional adaptation.
- NDCG uses within-date percentile relevance and linear gains; analytical random expectation. Sharpe/CAGR from five staggered sleeves of equal-weight top10, five-session holdings, 10bp one-way turnover charge; no claim of executable next-open fills.
- Exploratory promising threshold: positive paired RankIC across both folds and >=2/3 seeds, pooled delta >=.003; otherwise inconclusive/neutral/negative preliminary. Family uncertainty: stationary block bootstrap and Holm correction, no seed-only superiority claim.

## Tasks

**Data-gate amendment before broad outcomes:** Alpaca2019 has CUZ/SPN reverse-split anomalies dominating MSE (see screening PROGRESS). Main common panel is now RSR NASDAQ1,026 stocks, train through2013/2014, validation2014/2015, evaluation2015/2016; reserve2017. Same h5/seq16/three seeds/40epochs/patience8. Exact input manifest includes1,028 source hashes. Alpaca pilot results excluded from all method rankings. Frozen control on2016 uses2015 fold's train/validation/scaler/groups. Full-series RSR price scale cancels in relative features and returns.

**V2 implementation gate:** independent review found disjoint groups collapse within-group scores and the first sleeve ledger implicitly rebalanced capital. Both reproduced and corrected before interpretation. Every graph arm now shares singleton self hyperedges. V1 outputs excluded. Validation checkpoint selection every fifth session, daily final evaluation; fixed schedule shared by all arms reduces overlapping labels and measured CPU cost. Full40epoch ceiling and patience8 preserved. MASTER full attention is optional and deferred pending measured resource feasibility; its functionality smoke is not predictive evidence.
- [ ] T1 Data/metric invariants: tests first, then data loading, mature labels, train graphs, metrics, statistical baselines and data audit.
- [ ] T2 Mechanism implementations: tests for shapes, gradients, masking, ranking losses and planted-signal learning; extend existing THINK blocks.
- [ ] T3 Resumable runner: source/data/config hashes, completed-run identity, checkpoint/history/predictions, bounded smoke and throughput estimate; full fits with common early stopping.
- [ ] T4 Execute feasible broad screen; diagnose informative failures; expand winners only after the broad screen. Record blocked/unrun entries explicitly.
- [ ] T5 Generate all required reports/CSV/charts and executive summary; fresh verification; handoff/progress update; commit only task files if Git permits.

## Review focus
Five-session endpoint and purge errors; missing stocks entering cross-stock messages; future data altering causal features/graphs; invalid resumes after source/config changes; collapsed/tied predictions and misleading NDCG/trading annualization.
