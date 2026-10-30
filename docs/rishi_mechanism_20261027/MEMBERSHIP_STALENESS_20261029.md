# Monthly membership compared with recorded effective dates

At **31 of 201 test decisions**, the study's monthly membership differs from the daily-effective membership reconstructed from the same pinned third-party event source. The source's true accuracy and announcement timing are not independently certified. Across those decisions, **50 origin/security rows** are eligible under the monthly mask but absent from the source's daily-effective membership. These are repeated observations, not 50 distinct stocks.

| Test year | Decisions with a difference | Monthly-only eligible rows |
|---|---:|---:|
| 2022 | 7 of 51 | 16 |
| 2023 | 8 of 50 | 19 |
| 2024 | 8 of 50 | 8 |
| 2025 | 8 of 50 | 7 |

In the existing AH+PM-parent/dynamic issuer-five target schedules, monthly-only selections occurred on **2022-12-30 (NTES, original five, 10% target)** and **2024-07-23 (WBA, original five and ten seeds, 20% target each)**. These are target weights, not executed positions or performance contributions. No monthly-only selections occurred in the new-five schedule. The five later 2025 entrants ALNY, FER, INSM, MPWR and STX are absent from the saved model's security axis, so comparisons restricted to that axis undercount daily-effective entrants. The documented SOLS effective-date disagreement remains unresolved.

This diagnostic does not show whether a daily-updated index universe would improve or worsen returns. A different eligible population can alter peer aggregation and cross-sectional ranks, and a valid model-level counterfactual remains gated by strict checkpoint reproduction. No model was rerun, fitted, or scored on a new window. The exact results can be reproduced using `python -m hypershift run rishi_membership_staleness` and the [membership source check](DAILY_RAW_RECONSTRUCTION_20261029.md).

## Reconciled account trace of those selections

The existing reconstructed risk-managed stock traces show positive end-of-entry-day exposure for all three cases. From the entry day through the day before the next scheduled fill, the sum of daily net stock contributions was **+0.756 percentage points** for original-five NTES, **+0.749 pp** for original-five WBA and **+0.728 pp** for ten-seed WBA. The respective initial realized exposure was **6.21%, 11.53% and 11.20%** of equity. These sums use daily net P&L divided by each day's prior NAV, include fees recorded in that window, and exclude the next fill day's return and any exit fee. They are neither complete trade returns nor annualized performance contributions. Run `python -m hypershift run rishi_membership_lag_trace` to reproduce.

The traces are reconstructed economic fills under the source accounting model, not broker orders. These positive historical stock contributions do not measure the impact of the monthly rule: another selection might have earned more or less, and peer/rank changes remain untested. This follow-up was chosen after the cases were observed and has no independent confirmation value.
