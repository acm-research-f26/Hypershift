# C. Figures and tables specification

Every item is built from saved outputs only; no item needs a new experiment or a new data download. Where no plotting script exists yet, the item names one to write (`scripts/pubfig_*.py`). Each such script only reads files.

## Main tables

| ID | Content | Source files | Regenerate |
|---|---|---|---|
| Tab 1 | Reproduction summary: THINK, EH, EE, RSR-I and STHGCN; leak-free vs best-test Sharpe; hold-all; seeds (2017 NYSE, old and corrected setups) [T1, T3] | `results/R5_g2/*/seed_*/metrics.json`; `results/R5_f_{train,paper}/*/seed_*/metrics.json`; `results/R8_f_*` (if present locally) | `scripts/r5f_analysis.py`, `scripts/r5_g2_analysis.py`; values cross-checked in `CLAIMS_VERIFICATION.md` (T1, T3) |
| Tab 2 | Protocol matrix (88 cells): where published-scale values appear [T5] | `docs/phase2/protocol_matrix.md` | Phase 2 W3 scripts (Kaggle-era outputs). Mark as doc-sourced |
| Tab 3 | Planted-signal recovery under weight-decay settings [T2] | `results/known_signal_analysis.json`, `results/f_*/*/seed_*/known_signal.json` | `scripts/f_analysis.py` |
| Tab 4 | Leakage decomposition: L, TY, future-only part; per-year IC [L1] | `data/audit/yf_unadjusted/unadjusted.npz` plus FB `predictions.npz` labels | `scripts/pub_verify.py` (pooled values); per-year: `scripts/fbaudit_dividends.py yield` |
| Tab 5 | Matched 2×2 (features × target) with per-year and per-seed values [S1, S3, M1] | `results/FBAUDIT_20261022/matched/` | `scripts/closure_matched.py report` |
| Tab 6 | Synthetic validation: metrics and tests under S/I scenarios; learner probe [S2, M2, M3] | deterministic; no saved arrays | `scripts/closure_synth.py`, `scripts/closure_synth_perm.py` (the probe is inline in A12: move it into `scripts/pubfig_probe.py`) |
| Tab 7 | Paired economics of buffered books [E1, E2] | regenerated from predictions | `scripts/closure_econ.py`, `scripts/bcreview_cost.py` |
| Tab 8 | Power and selection: MDE by contrast, prospective power, LOFO, SPA/RC [P1–P3] | FB `daily.npz` | `scripts/fbaudit_stats.py`, `scripts/fbaudit_selection.py` |

## Figures

| ID | Content | Source | Script to write |
|---|---|---|---|
| Fig 1 | Leak-free vs best-test Sharpe by arm, seeds as dots, hold-all and random-basket band (2017) | Tab 1 sources; `docs/phase1_5a/nulls.json` | `scripts/pubfig_repro.py` |
| Fig 2 | Collapse mechanism: per-layer norm and prediction SD over steps, coupled decay vs none | `scripts/f_diag.py` output (recompute: CPU, minutes; this is a probe, not a training run) | `scripts/pubfig_collapse.py` |
| Fig 3 | Adjustment factor: per-year cross-sectional SD of L; IC bars for L, TY and the future-only part | Tab 4 sources | `scripts/pubfig_leak.py` |
| Fig 4 | 2×2 cells with CIs, plus the characteristics-only placebo on the neutral, linear-residual and quintile-residual metrics | Tab 5 sources; A7 | `scripts/pubfig_matched.py` |
| Fig 5 | Synthetic S/I: increment and encompassing CIs by replicate; learner probe bars | Tab 6 | `scripts/pubfig_synth.py` |
| Fig 6 | Turnover vs net Sharpe across schedules and costs; random-null band | Tab 7 | `scripts/pubfig_cost.py` |
| Fig 7 | MDE vs number of days, per contrast; dev effect marked | Tab 8 | `scripts/pubfig_power.py` |

**Conventions:**
- Every figure caption states: the data period; that the result is exploratory (inspected years); seeds; the CI method (stationary bootstrap, block 10, unless stated); and the claim ID.
- Colour and palette follow the repository's dataviz guidance.
- No figure shows best-test values without the shuffled-label reference.

## Status (revision 2, 2026-10-22)

- **Generated** from saved outputs by `scripts/pubfig.py`:
  - Fig 1 / Tab 1 (reproduction);
  - Fig 2 / Tab 2 (planted signal);
  - Fig 3 / Tab 3 (adjustment term: TY, L, residual, L_div, L_path, L(end));
  - Fig 4 / Tab 4 (matched 2×2);
  - Fig 5 / Tab 5 (power).

  Files are in `docs/publication/figures/` and `docs/publication/tables/`.
- **Not generated** (they need deterministic re-execution, not saved outputs):
  - protocol-matrix table;
  - collapse-probe figure;
  - synthetic panel;
  - economics figure.

  See `REPRODUCE.md`.
- **Numbering change.** The planned Figs 2/5/6/7 were renumbered to match the manuscript; this file's earlier table is superseded where it conflicts.
