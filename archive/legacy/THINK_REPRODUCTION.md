# THINK reproduction attempt: fidelity and limits

**An exact one-to-one reproduction has not been achieved.** The earlier 12-stock
model was an educational implementation, not THINK. This attempt adds a separate
implementation mapped to the paper's architecture and a full-universe NYSE daily
ranking pipeline. The default runner refuses to call the experiment an exact
reproduction while required settings are unresolved.

## Why the old scores do not contradict the ablation

Our earlier scores came from a different experiment:

| Ingredient | Earlier local experiment | THINK specification / new attempt |
|---|---|---|
| Universe | 12 chosen surviving US stocks | NYSE source panel: 1,737 stocks |
| Horizon | 21 trading sessions | One future session for this ranking reconstruction |
| Target | Binary up versus down/flat | Predict a return score and rank the universe |
| Groups | Training-return correlations | Industry and Wikidata relations |
| Temporal encoder | Linear layer over lag/summary features | Temporal convolution on each side of hypergraph aggregation |
| Hyperbolic layers | Simplified mean/distance operations | Poincare FC, beta-concatenation, Einstein midpoint, distance-modulated attention |
| Optimized loss | Binary cross-entropy | Return MSE plus pairwise ranking loss borrowed from cited baseline |
| Trading | Monthly top-3, delayed execution, costs | Daily top-5 gross research diagnostic; exact THINK conventions unresolved |
| Repeated fits | Five seeds | Paper reports 25; short execution check below is explicitly a pilot |

The architectural/task descriptions are from [THINK](https://tylersnetwork.github.io/papers/icdm22-think.pdf).
Our old implementation is in `hypergraph.py`; the new one is in `think_model.py`.
Even a successful ablation establishes a relative advantage under that experiment's
conditions, not that every hyperbolic network beats every simpler model on every
market. Our earlier result does not isolate a causal explanation for its poor
performance. Weak predictive features, a different horizon, graph construction,
model simplification, optimization and sampling variability are plausible factors.

## What was recovered

- The exact public PDF was downloaded and its equations inspected visually.
- The cited NYSE panel has 1,737 stocks, 1,245 rows and five features per row.
- The prior audit's reconstructed industry/Wikidata memberships are reused:
  4,372 groups and 9,727 node/group memberships. They are not the missing author incidence file.
- The cited STHAN-SR `train_nyse.py` specifies split boundaries at rows 756 and
  1,008, a default lookback of four and a return-regression/pairwise-ranking loss.
  These are evidence about the cited baseline, not proof of THINK's settings.
- [THINK's repository](https://github.com/shivamag125/ICDM22-THINK) is empty.
  Both public issues request the code and had no replies when checked on 2026-09-28.
- The original [HNN++ implementation](https://github.com/mil-tokyo/hyperbolic_nn_plusplus)
  provides a reference for the Poincare fully connected and concatenation operations.

Downloaded sources are retained as text, not imported or executed. Source versions:
STHAN-SR `05a23fca0f737f7a6d6924a38ce57e96c8aff04f`; HNN++
`28737e22822562ac18d5ea03f8c3e3929e945a83`. The price/relation provenance is
recorded in `data/think_audit/stock_sources.json` and the earlier audit.

## New implementation and explicit assumptions

The model follows the temporal/spatial/temporal ordering in Eq.17. It includes
origin exponential/logarithmic maps, Mobius addition, hyperbolic distance,
Poincare fully connected layers, beta-concatenation, Einstein-midpoint group
aggregation and the tangent-space weighted update. Sparse membership indexing
avoids constructing a dense batch-by-stock-by-hyperedge tensor.

Four variants are implemented:

1. `think`: both temporal blocks and the spatial block are hyperbolic.
2. `euclidean_temporal`: replaces the two temporal blocks with Euclidean layers,
   retaining hyperbolic spatial aggregation.
3. `no_distance`: removes the distance modulation of attention.
4. `euclidean`: Euclidean temporal and spatial operations.

Their names indicate reconstruction variants, not certified implementations of
the authors' ablation checkpoints. We had to make these choices:

| Choice | Implementation | Why it prevents an exact claim |
|---|---|---|
| Printed Eq.7 | Standard Mobius scalar multiplication for distance modulation | The printed expression has ambiguous symbols/shapes and uses tan/arctan notation inconsistent with the usual Poincare operation. |
| Printed Eq.10 | Normalize the direction of z inside the inner product, following HNN++ code | That normalization is absent from the literal printed expression. |
| Attention | Signed coefficients, with no added softmax | Eq.15 writes a weighted sum; no recoverable normalization setting is provided. |
| Temporal shape | Two nonoverlapping kernel-2 blocks; lookback four | Kernel sizes/stride used in the experiment are not recovered. |
| Width and optimizer | Width 16; Adam, lr 0.001, weight decay 0.0005 | Not identified as the paper's tuned settings. |
| Isolated stocks | Spatial update is zero when there is no membership | No invented singleton hyperedges or residual path; authors' treatment unknown. |
| Numerical stability | Ball projection, finite-value checks, gradient clipping; cap FC sinh argument at 15 | Explicit implementation safeguards that may affect saturated activations. |
| Epoch selection | Lowest validation MSE | Author checkpoint-selection rule unavailable. |
| NDCG | Top-5; positive realized return as relevance; tie averaging | Author relevance and tie conventions unavailable. |

This is deliberately not described as a literal transcription of every printed
symbol or an exact recovery of the original experiment.

## Problems found in cited baseline code

These findings concern the public **STHAN-SR baseline release**, not recovered
THINK code. They cannot establish that THINK inherited the same behavior.

The pinned [evaluator](https://github.com/midas-research/sthan-sr-aaai/blob/05a23fca0f737f7a6d6924a38ce57e96c8aff04f/training/evaluator.py):

- Passes the numeric IDs from two sets of top-five stocks to `ndcg_score`.
  These are not relevance values and scores aligned to the same items. Relabeling
  stocks can therefore change the purported ranking quality.
- Assigns the NDCG field inside the day loop, without accumulating it: only the
  last day's value survives.
- References undefined `sharpe_li` after defining `sharpe_li5`, which raises an
  error if this unmodified function reaches that line.

A reproducible 64-stock counterexample changes only the node numbering. The
source's NDCG expression changes from **0.774704 to 0.791466**, while correct
positive-return NDCG remains **0.208052** in both cases. The example is synthetic
and demonstrates an evaluator property, not market performance. See
`audit_cited_evaluator.py` and
`data/think_reproduction_sources/evaluator_counterexample.json`.

The pinned [NYSE trainer](https://github.com/midas-research/sthan-sr-aaai/blob/05a23fca0f737f7a6d6924a38ce57e96c8aff04f/training/train_nyse.py)
shuffles starting offsets from 0 through 755, then uses a subset for training.
With lookback four and horizon one, offset 755 has target row 759, beyond the
validation boundary at row 756. Shuffling/selecting fewer offsets does not guarantee
those boundary-crossing examples are excluded. Our pipeline partitions by target
row instead: training targets end at 755, validation begins at 756, and test begins
at 1,008. It does not reproduce that leakage pathway.

The source loader also mutates sentinel prices before later checks. Our loader
preserves a separate missingness mask before replacement and requires every price
in the lookback/target window to be valid. Borrowing useful source conventions
does not require importing execution errors or invalid evaluation logic.

## Execution check, not a benchmark conclusion

`runs/think_nyse_pilot/` is a two-epoch, seed-7 check of `think` and
`euclidean_temporal` on the full NYSE universe. It exercises data loading,
backpropagation, validation selection, checkpoint saving and held-out scoring.
Two epochs and one seed are not a 25-run ablation study and must not be used to
confirm or reject the paper's numerical findings. A small pilot is useful here
because unresolved protocol differences remain even if training is extended.

Both runs completed and produced finite predictions for **237 held-out target
rows by 1,737 stocks**. The two-epoch/one-seed budget was fixed before test results
were inspected. The recorded diagnostics are:

| Pilot variant | Return MSE | Gross annualized Sharpe | NDCG@5 |
|---|---:|---:|---:|
| Hyperbolic temporal + hyperbolic spatial | 0.270088 | -0.008583 | 0.125790 |
| Euclidean temporal + hyperbolic spatial | 0.301678 | 0.775366 | 0.140951 |

These models are severely undertrained: a zero-return forecast has return MSE
**0.000225710** on the same masked targets. That baseline is an execution-check
diagnostic, not an additional tuned model. The pilot establishes that the pipeline
runs; it does not establish a useful fitted model or an ablation conclusion.
The `no_distance` and fully Euclidean variants were unit-tested but not trained
in this pilot. No full 25-seed experiment or significance test has been completed.

The pipeline saves all test return predictions, masks, target row indices,
checkpoints, loss histories, data/code hashes and the unresolved-setting ledger.
Its portfolio score is a gross close-to-close diagnostic with same-close entry;
it is not an executable trading simulation or evidence of profits after costs.

## Tests

The implementation tests exp/log round trips, beta-concatenation dimensions,
Poincare FC numerical gradients, finite gradients for every ablation, checkpoint
restoration, stock-relabeling equivariance, exact chunked ranking loss and gradients
against a dense reference, correct/tie-aware NDCG, and target-boundary separation.
The separate existing suite continues to cover the previous forecasting pipeline.
All **40 combined tests passed**. The default exact-reproduction gate was also
checked separately.

## Reproduction commands

The default command reports unresolved prerequisites and exits before training:

```powershell
.\.venv\Scripts\python.exe run_think_reconstruction.py
```

The explicitly labeled pilot is:

```powershell
.\.venv\Scripts\python.exe run_think_reconstruction.py --allow-reconstruction --pilot --seeds 7 --variants think euclidean_temporal --epochs 2 --patience 2 --hidden 16 --batch-size 8 --out runs\think_nyse_pilot_new
```

For a larger documented reconstruction, the runner supports
25 seeds, four variants and up to 50 epochs by default. This configuration is
provided for reproducibility, **not claimed to have been run**:

```powershell
.\.venv\Scripts\python.exe run_think_reconstruction.py --allow-reconstruction --out runs\think_nyse_25seed_reconstruction
```

Choose a fresh output directory; the runner refuses to overwrite an experiment.

## What is needed for one-to-one verification

We still need the exact author implementation/configuration, processed incidence
matrices, data/target preprocessing, split dates/indices, attention convention,
loss and optimizer settings, checkpoint-selection procedure, metric definitions,
and per-run outputs or enough information to rerun all 25 matched experiments.
The table's means and error bars are not sufficient to reconstruct these inputs
or verify its paired significance tests. No contact with the authors has been sent.
