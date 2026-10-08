# Training dates and leakage audit

**Result:** the 16 cohort/horizon/year folds passed the chronology and preprocessing checks. No future-observation leakage was detected in these audited code paths. This is not a certification that the public source data are point-in-time or that the research history is free of test reuse.

| Test year | Actual training-signal years | Validation year |
|---|---|---|
| 2022 | 2019–2020 | 2021 |
| 2023 | 2019–2021 | 2022 |
| 2024 | 2019–2022 | 2023 |
| 2025 | 2020–2023 | 2024 |

Raw prices start in 2018. A 260-session warmup moves the first model-training signal to **2019-01-15**. Models refit for each annual test. Earlier test years can become training or validation data for later years because they are then in the past. The 2025 fit never contributes predictions to the 2022 curve. NASDAQ and NYSE remain separate cohorts.

## Exact saved signal dates

Dates below apply to both cohorts. Labels mature after signal dates, as shown in the next table.

| Horizon | Test year | Training signals | Validation signals | Test signals |
|---|---|---|---|---|
| 5 | 2022 | 2019-01-15 to 2020-12-16 | 2021-01-08 to 2021-12-21 | 2022-01-05 to 2022-12-19 |
| 5 | 2023 | 2019-01-15 to 2021-12-21 | 2022-01-05 to 2022-12-19 | 2023-01-04 to 2023-12-18 |
| 5 | 2024 | 2019-01-15 to 2022-12-19 | 2023-01-04 to 2023-12-18 | 2024-01-03 to 2024-12-16 |
| 5 | 2025 | 2020-01-06 to 2023-12-18 | 2024-01-03 to 2024-12-16 | 2025-01-08 to 2025-12-16 |
| 1 | 2022 | 2019-01-15 to 2020-12-29 | 2021-01-04 to 2021-12-29 | 2022-01-03 to 2022-12-28 |
| 1 | 2023 | 2019-01-15 to 2021-12-29 | 2022-01-03 to 2022-12-28 | 2023-01-03 to 2023-12-27 |
| 1 | 2024 | 2019-01-15 to 2022-12-28 | 2023-01-03 to 2023-12-27 | 2024-01-02 to 2024-12-27 |
| 1 | 2025 | 2020-01-02 to 2023-12-27 | 2024-01-02 to 2024-12-27 | 2025-01-02 to 2025-12-29 |

| Horizon | Test year | Last training outcome | First validation signal | Last validation outcome | First test signal |
|---|---|---|---|---|---|
| 5 | 2022 | 2020-12-24 | 2021-01-08 | 2021-12-30 | 2022-01-05 |
| 5 | 2023 | 2021-12-30 | 2022-01-05 | 2022-12-28 | 2023-01-04 |
| 5 | 2024 | 2022-12-28 | 2023-01-04 | 2023-12-27 | 2024-01-03 |
| 5 | 2025 | 2023-12-27 | 2024-01-03 | 2024-12-24 | 2025-01-08 |
| 1 | 2022 | 2020-12-31 | 2021-01-04 | 2021-12-31 | 2022-01-03 |
| 1 | 2023 | 2021-12-31 | 2022-01-03 | 2022-12-30 | 2023-01-03 |
| 1 | 2024 | 2022-12-30 | 2023-01-03 | 2023-12-29 | 2024-01-02 |
| 1 | 2025 | 2023-12-29 | 2024-01-02 | 2024-12-31 | 2025-01-02 |

## Checks performed

- Reconstructed train/validation/test indices exactly match saved preprocessing files.
- Training labels mature before validation; validation labels mature before testing.
- Training-only feature means and standard deviations exactly match the saved scalers.
- Changing all future prices, volume and the benchmark leaves earlier inputs, eligibility, liquidity, training/validation labels, scalers and correlation groups unchanged.
- Saved portfolio winners equal the best recorded validation candidate. Architecture selection uses only validation scores from the current or earlier folds.
- Frozen training-source hashes match. Inputs remain chronological, with next-close execution and one- or five-session holding periods.

## Remaining limitations

Yahoo adjusted histories can reflect later corporate-action revisions. The fixed research cohorts and unavailable delisted securities create selection/survivorship bias. Correlation groups use all available price history through the last training signal, including older history before a rolling training window. That is historical information, not future information. The 2024–2025 tests were previously inspected and the wider period was chosen retrospectively. Thus the results are exploratory. Pooled curves concatenate separate annual accounts, reset sizing each year and omit boundary sessions.

[Machine-readable fold audit](chronology_audit.csv) | [Audit status](audit/chronology_audit.json)
