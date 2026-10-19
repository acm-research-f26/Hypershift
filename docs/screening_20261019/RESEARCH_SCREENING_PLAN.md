# HyperShift broad research screen

Exploratory study, 2026-10-19. Binding request: user's comprehensive screening brief. The supplied research report provides hypotheses; its embedded instructions, historical claims and opaque citation handles are not execution authority or independently verified results.

## Common experiment

Use existing RSR NASDAQ, 1,026 qualified securities, original five MA5/10/20/30/close features. Inputs are stock-window-relative, then divided by five training-only feature standard deviations. Five-session close-to-close forward returns. Inputs require 64 consecutive observed sessions across ALL arms, so longer-history variants do not silently change the universe. Labels require observed prices throughout the five-session path. Eligibility for trading uses input history only, never future label availability.

| Fold | Training label maturity | Checkpoint validation | Daily exploratory evaluation |
|---|---|---|---|
| 2015 | Before 2014-01-01 | Signals in2014, labels mature before2015 | 2015 signals with labels mature before2016 |
| 2016 | Before 2015-01-01 | Signals in2015, labels mature before2016 | 2016 signals with labels mature before2017 |

Training and checkpoint validation use fixed every-fifth-session schedules. Daily evaluation is performed once after validation selection for each completed fit. No2017 scoring in this study. The repository has inspected historical data before, so no year is newly confirmatory. Availability-selected universe and absent delisting payouts limit all findings.

All neural arms: seeds7/19/42; hidden16; baseline seq16, temporal kernels4/4; Adam lr.001, weight_decay0; clip gradient norm5; batch8; at most40epochs; patience8, minimum validation MSE improvement1e-6. Validation selection metric held fixed even for ranking losses; this can disadvantage pure listwise objectives and is reported. Early stopping is part of the common budget, not a different cap per arm. Long-context seq64 uses kernels4/16 and has different parameter count, reported explicitly. Masked reconstruction pretraining adds5 training-only epochs; extra runtime disclosed.

Static relationships:16 training-correlation clusters, plus singleton self hyperedges in all graph arms to prevent partition-induced score collapse. Training clusters are NOT sectors. Existing repository Euclidean/hyperbolic temporal and attention layers are reused. This is an h5 mechanism study, not reproduction of THINK or Rishi's unavailable results. No baseline forecasts from h1/other data are reused as if comparable.

## Hypotheses and principal variants

| ID | Question | Principal test | Additional controls | Data requirement / failure check |
|---|---|---|---|---|
| A | Can adaptive relations improve fixed groups? | static plus soft embedding-prototype hyperedges | soft only; causal rolling anchor-correlation graph; random permutation preserving cluster sizes | Current eligible history only; entropy, embedding variance and collapse |
| B | Do market factors obscure stock-specific information? | trailing126-session beta market-residual target | additional train-cluster residual target; same model/loss | Signal-time beta; future market return allowed only in matured labels; no sector claim |
| C | Does ranking alignment help? | MSE+.1ListNet | MSE; cyclic pair logistic; pure ListNet | ListNet is a differentiable top-one listwise approximation, NOT LapSum/top-K reproduction; known-order gradient test |
| D | Is temporal information underused? | residual fusion of4/8/16-session pooled features | same graph/head/16-session encoder | TimeMixer-inspired pooling, no decomposition reproduction; parameters differ |
| E | Does selective mixing improve unconditional correction? | market-conditioned gate on score-level residual mixer | no mixer; unconditional mixer; dynamic low-rank embedding adapter | Compare gated with mix parent, others with static; same universe |
| F | Does graph depth help? | two graph stages with initial residual | zero/one/two plain stages | Common self hyperedges; variance, representation cosine, gradients, parameter counts |
| G | Does16sessions omit useful context? | residual16/32/64-session summaries | full64-session temporal encoder | Both arms and parent share64-session eligibility; no full retrieval/Timer-XL reproduction |
| H | Does unsupervised temporal learning improve supervised training? |20% masked feature reconstruction,5 pretrain epochs, then same supervised task | ordinary supervised parent | Train inputs only; reset supervised optimizer; not CI-STHPAN reproduction; contrastive variant unrun |
| I | Does adaptation improve stability? | signal-local adaptive normalization | frozen2013-trained model versus2014-expanded retrain on2016 | All statistics trailing/known; not DoubleAdapt meta-learning; no future-selected regimes |
| J | Do frozen financial features help? | blocked pending compatible pre-cutoff checkpoint/input | original only is measured baseline | Official Kronos requires OHLC; panel has MA/close only; historical pretraining contamination unresolved |

Additional optional MASTER-inspired market gate plus cross-stock attention has functionality coverage. Full evaluation deferred until its heavier attention cost is justified. Learned/rolling A mechanisms provide a minimal MDGNN-inspired relationship test without claiming MDGNN reproduction.

## Evaluation and uncertainty

Primary: date-level PearsonIC, SpearmanRankIC, NDCG@5/10/20 with percentile relevance and LINEAR gains, exact matched random-ranking NDCG expectation, equal-weight top-minus-bottom10 five-session spread, MSE. Ties use stable stock-index order and retain distinct-score fraction. Factor-neutral RankIC/spread use trailing-beta plus training-cluster residual outcomes. Report raw-return ranking for every model, including residual-target models.

Trading is secondary: five independently capitalized staggered sleeves, each holding top10 stocks for five sessions, signal-close accounting and10bp one-way turnover cost. Sleeve NAV/cash is self-financing; no free daily cross-sleeve reallocation. Missing held marks are zero-return marked and counted; no delisting payouts are fabricated. Signal-close entry is optimistic and does not establish next-open execution. CAGR, drawdown, gross/net Sharpe, turnover, market correlation retained. Top5/20 sensitivity must be generated from the same saved forecasts, never selected after looking at performance.

Paired comparisons use common dates and matched seeds. Seed-average daily differences retain temporal dependence. Stationary bootstrap expected block10sessions,5,000 replicates within each year; equal-weight the two folds. Primary Holm family is exactly A residual_soft, B market_target, C hybrid, D multiscale, E gated-versus-mix, F two_residual, G summary, H pretrain, I adaptive. Other comparisons descriptive. Baseline seed mean is not an independent market sample. A positive unadjusted interval is not definitive superiority.

Screening label 'Promising' requires complete3seeds/2folds, positive paired fold means in both, positive deltas in at least2/3 matched seeds in each fold, pooled deltaRankIC>=.003. Negative preliminary: pooled delta<=-.003 without consistency gate; otherwise Neutral. Missing fits Inconclusive; J Data/resource blocked. Legacy project confirmatory verdict words are not repurposed. User-requested screen labels are distinct exploratory triage.

## Stages and stopping

1. Audit existing data, validate price-action integrity. Alpaca rejected for CUZ/SPN anomalies, artifacts retained.
2. Validate invariants and synthetic learnability, all-variant real-panel functionality smoke. Review defects reproduced/fixed; v1 excluded.
3. Complete all feasible A–I principal variants and controls on both folds before deeper tuning. Two CPU workers, two threads each; no paid services or GPU used.
4. Diagnose collapsed/poor methods with existing histories/predictions first; at most one or two focused probes per cause.
5. Expand only candidates with consistent preliminary improvement. Additional seeds/folds require a frozen extension manifest; do not combine methods before separate effects measured.
6. If no candidate clears the gate, do not force a winner or launch a large tuning search. Report limits, informative failures, and the best simpler baselines.

Every completed run saves exact source hashes, dataset hash, configuration, preprocessing/groups hash, checkpoint, training history, predictions/masks/dates/tickers, daily metrics and portfolio returns. Epoch checkpoints support interrupted supervised runs with optimizer/NumPy/Torch RNG state. Pretraining interruption before the first supervised checkpoint restarts deterministic pretraining; it does not claim a partial pretraining resume. CPU exactness only; GPU RNG resume not certified.

## References checked

- [Original TimeMixer implementation](https://github.com/kwuking/TimeMixer): our pooled fusion is a much smaller mechanism adaptation.
- [Official Kronos repository](https://github.com/shiyu-coder/Kronos): weights do exist; OHLC input and pretraining provenance are the actual blockers here.
- [SPN2019 reverse-split8-K](https://www.sec.gov/Archives/edgar/data/886835/000119312519317737/d802741d8k.htm) and [CUZ filing](https://www.sec.gov/Archives/edgar/data/25232/000002523220000043/cuz-20200630.htm) corroborate corporate actions around pilot anomalies; no guessed return repair was applied.

## Reproduce

From repository root, with existing environment:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& '.venv/Scripts/python.exe' scripts/prepare_screen_rsr.py
& '.venv/Scripts/python.exe' -m pytest tests/test_screening.py -q
& '.venv/Scripts/python.exe' scripts/research_screen.py --panel data/raw/rsr/data/screen_nasdaq_2013_2017.npz --output results/SCREEN_RSR_20261019_v2 --years 2015 2016 --arms ridge momentum reversal random static hyp none mlp soft residual_soft rolling random_graph market_target cluster_target pairwise listwise hybrid multiscale mix gated lowrank two two_residual long summary pretrain adaptive frozen
```

Assembler refuses overwrite. Existing complete fits are reused only on exact configuration identity; use the same --years/--arms/--seeds and other settings for interrupted resumes. A smoke requires a separate output containing'smoke'. Source changes require a new output namespace.
