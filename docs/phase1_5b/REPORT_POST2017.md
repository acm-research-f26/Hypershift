# Post-2017 frozen THINK test (Phase 1.5b): report

STATUS: predeclaration committed before the full analysis run (results appended below after the run).

## Predeclaration (fixed before computing the formal statistics)

- Data and models: locked pass of `R5_f3_alpha0_train` HH and EH, seeds 0-4, `best_state.pt` selected on pre-2017 validation Sharpe (FREEZE_MANIFEST.md). Panel = Alpaca `adjustment=split`, 1,647 identity-pass nodes, masked after last bar, 2018-01-02..2023-12-29.
- Statistic: pooled Sharpe of the concatenated daily top-5 returns (mean/std ddof 0 x sqrt(252)); never a mean of annual Sharpes.
- Primary formal family (Holm over 5 tests): F1 HH vs random daily top-5; F2 HH vs beta-quintile-matched baskets (beta from the RSR 2013-2015 training period only); F3 HH vs industry-matched baskets; F4 HH vs stock-label permutation; F5 HH vs EH (Wilcoxon on per-seed pooled Sharpes, paired by seed). F1-F4 use one-sided empirical upper-tail p per seed and the intersection-union rule (max p over the 5 seeds).
- Null sizes: B_NULL = 2000 (random top-5), B_PERM = 500 (beta, industry, label-permutation), N_BOOT = 5000 stationary-bootstrap draws, mean block 10 days, common day blocks across all series. Random streams from `hypershift.eval.forensics.rng_for`.
- Verdict words for F5 follow the repo rule (`aggregate.py`): INSUFFICIENT SEEDS if m * 2^(1-n) >= 0.01 (n = 5 seeds, m = 5 tests gives 0.3125).
- Descriptive only (no test): per-seed and seed-averaged pooled Sharpe, annual and leave-one-year-out, hold-all on the same mask, gross vs 5/10/25 bp-per-side net, IC, NDCG@5, top/bottom decile hits, basket beta, eligible names per year.
- Two separate conclusions will be stated: (1) Sharpe persistence, (2) evidence of learned ranking.
