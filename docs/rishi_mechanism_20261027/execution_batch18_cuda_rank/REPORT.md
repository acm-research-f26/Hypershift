# All-checkpoint local CUDA rank sensitivity

The fixed, non-promoting protocol ../CUDA_ALL_RANK_DIAGNOSTIC_PROTOCOL.md was committed as 1bd7fc2 and implementation as ca7bb71 before execution. Local source-config CUDA inference completed for all 80 original selected B and D checkpoints, on validation and test only: 160 checkpoint-split comparisons. No training predictions, model interventions, financial account returns or new fits were generated. The original saved values remain the reference.

| Model | Checkpoints exceeding 1e-6 pp on either split | Maximum value error | Changed eligible-rank origins across validation/test comparisons | Changed top-5 / top-10 target origins |
|---|---:|---:|---:|---:|
| Parent B | 30 / 40 | 0.00609875 pp | 11 | 0 / 0 |
| Dynamic D | 1 / 40 | 0.00000154972 pp | 30 | 0 / 0 |

For each model the split comparisons comprise 1,960 validation and 2,010 test origins across the 40 checkpoints; these repeat the same market origins across seeds and are not independent samples. Parent maximum is 2024/seed 7 test; dynamic maximum is 2024/seed 7 validation. Some rank changes occur even where the strict value tolerance passes, because small differences can break ties. Rank changes are reported at any eligible position, not exclusively the portfolio boundary.

Across all 36 fixed test-ensemble cells (four years × three seed groups × parent/dynamic/blend), 33 of 1,809 cell-origin comparisons have a changed full eligible rank vector; **none changes the top-five or top-ten target weights**. The same market origin is repeated across ensemble cells; these are counts of comparisons, not 1,809 independent dates. No account-return equivalence was computed, and this fixed unmodified-inference finding says nothing about how interventions or training predictions would behave.

An independent audit rebuilt all 160 split counts and 36 ensemble cells using the saved generated arrays, verified security and date axes, and checked all 80 original checkpoint and export hashes. Two focused tests cover rank changes within and across a target boundary, ties and inactive values. CHECKPOINT_SPLITS.csv, ENSEMBLES.csv, RESULTS.json and INDEPENDENT_AUDIT.json retain the complete evidence; generated arrays are in the ignored `results/RISHI_CUDA_RANK_DIAGNOSTIC_20261028` directory. This is one local RTX 3050 GPU, not the original RTX 4070. The unchanged top-k weights do **not** pass the preregistered value threshold: strict checkpoint reproduction, Hpred, changing-score controls, surrogate and trained-forward intervention gates remain unresolved. No tolerance was widened.
