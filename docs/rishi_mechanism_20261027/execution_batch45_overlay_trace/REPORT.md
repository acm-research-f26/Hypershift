# Parent-view overlay attribution — exploratory accounting

The previously highlighted new-five, five-issuer, 20%-volatility-target comparison gained **3.56 percentage points per year in arithmetic mean net return** when the separately trained AH+PM parent replaced the daily parent in the same dynamic blend. The existing 21-day-block interval is **[0.44, 7.10] pp/year**; the corresponding Sharpe difference is **+0.179 [0.023, 0.352]**. This subgroup was highlighted after 24 pointwise comparisons on repeatedly examined history. These intervals are unadjusted and do not establish a discovery or a causal feature effect. Seeds share one market history.

The difference reconciles to an annual gross contribution increase of **3.54 pp** and a modeled fee advantage of **0.02 pp**. Mean realized exposure was 66.86% for the daily-parent account and 67.31% for AH+PM; these are descriptive averages, not an exposure-only counterfactual. Annual turnover was 27.24 and 26.96 respectively. The dynamic model uses AH+PM in both accounts, and the parents were trained separately, so this comparison cannot isolate the information learned from extended-hours inputs.

The table decomposes each pair's daily net return differences by issuer, using each account's own prior NAV. Values are annualized arithmetic percentage points, rounded; other includes every issuer outside the six listed. These are post-hoc accounting contributions, not effects of removing issuers or proof of a learned signal.

| Seed group | Three largest positive contributions | Three largest negative contributions | All others | Total |
|---|---|---|---:|---:|
| Original five | MRNA +1.03; NVDA +0.63; AMD +0.49 | CDNS -0.75; MELI -0.58; ADBE -0.45 | +0.45 | +0.80 |
| New five | MRNA +1.18; MCHP +0.90; TSLA +0.89 | INTU -0.45; SNPS -0.44; ROST -0.28 | +1.75 | +3.56 |
| Ten seeds | AVGO +0.85; MRNA +0.76; TSLA +0.56 | MSFT -1.05; Alphabet -0.77; MELI -0.51 | +0.86 | +0.70 |

For original five, the existing paired mean interval is **[-2.35, 4.08] pp/year** and Sharpe difference **+0.034 [-0.118, 0.190]**. For ten seeds, they are **[-3.35, 5.05] pp/year** and **+0.029 [-0.169, 0.235]**. New-five annualized daily-mean differences for 2022–2025 respectively were **+3.74, -2.29, +6.31, +6.45 pp**; these are not calendar-year CAGR differences. The direction varied, and the three seed groups are overlapping diagnostics, not independent replications.

## Verification and scope

The [fixed protocol](../PARENT_OVERLAY_TRACE_PROTOCOL.md) was committed as `c94f5dd` after the prior comparison but before this issuer breakdown; implementation was committed as `43bf701`. All six saved account daily return paths reproduce exactly; maximum normalized NAV error was 1.40e-15 and maximum absolute cash error $1.40e-9, with zero share error. An independent reconstruction verified six paths, three issuer/exposure comparisons, all 12 group-year stock-contribution partitions and 566 changed target rows, with maximum error 4.93e-16. Ten focused source trace and risk tests passed. Full account, stock, issuer, calendar, exposure and target tables accompany this report; target changes are not broker fills. The original numerical checkpoint gate, source-data limitations and primary inference decisions remain unchanged. No model was fitted or intervened upon, and no protected period was scored.
