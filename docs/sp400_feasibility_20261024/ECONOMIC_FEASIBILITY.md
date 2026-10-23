# E. Economic feasibility of S&P 400 filing-text avoidance (2026-10-24)

No post-signal returns are used. Inputs: HyperShift's own S&P 500 run (`../strategy_lazyprices_20261023/EVAL.md`),
the verified literature (`LITERATURE_CORRECTIONS.md`), and cost studies (`LIT_SP400_VERIFY.md` §5). Figures marked
**assumption** are not measured.

## Costs

| Component | Large caps | Mid caps (S&P 400) | Source |
|---|---|---|---|
| Market impact, one-way | 11.2 bp | about 15–21 bp | Frazzini–Israel–Moskowitz 2012 draft: large 11.21 vs small 21.27 bp (patient institutional execution) |
| Half-spread | a few bp | **assumption:** about 5–10 bp | No verified S&P 400 spread statistic found; a Corwin–Schultz estimate needs the mid-cap panel that was not built |
| Commissions | ~0 | ~0 | Retail and institutional brokers |
| **Total per unit traded** | 10 bp (as used) | **20 bp is plausible, possibly optimistic for the least liquid names** | — |

**Capacity (assumption).** At $10–100M AUM an equal-weight 400-name book holds $25–250k per name. An overlay trade
of that size is a small fraction of a typical mid cap's daily dollar volume, so impact is not binding at this scale.

## Turnover

- **The overlay** added **1.19× NAV/yr** of trading in the S&P 500. It is driven by the filing calendar, so a
  similar figure is expected in mid caps.
- **Index churn:** about 48 logged S&P 400 changes a year, against about 20–25 for the S&P 500. This turnover is
  common to the benchmark and the overlay, so it barely affects the active result.

## What gross improvement is needed

- **Cost:** 1.2× NAV/yr × 20 bp ≈ **0.24%/yr**.
- **Required gross:** clearing the predeclared bar of **0.30%/yr net** needs about **0.54%/yr active gross**.
- **Required spread:** avoidance captures about one quarter of the changer-versus-rest spread (it removes 20% of
  names and spreads their weight over the other 80%). So mid-cap **changers would have to underperform the rest of
  their industry by about 2.2%/yr after publication decay**.

## Is that plausible?

| Evidence | Implied changer shortfall |
|---|---|
| HyperShift S&P 500, 2016–2023 (measured) | about −0.4 to −0.5%/yr against the benchmark (IR −0.12) |
| Sadlo, S&P 1500, 1996–2020 (abstract) | least-similar quintile alpha "up to −5.12%/yr", the best of several specifications, mostly pre-publication |
| Cheng et al., size split (optionable, 1996–2017) | non-S&P 500 Q5−Q1 is about 1.4× the S&P 500 estimate; there is no difference test |
| McLean–Pontiff post-publication haircut | about −58% |

- **Literature reading.** Taking Sadlo's best figure, halving it for decay, and allowing some extra effect in mid
  caps gives about −2.5%/yr at most. That is just at the requirement.
- **Our own data.** The changer shortfall measured on 2016–2023 is about a fifth of what is needed. The literature
  suggests a mid-cap effect perhaps 1.4× larger, not 4–5× larger.
- **Expected net active return:** roughly **−0.1 to +0.2%/yr at 20 bp**, below the bar.

## Statistical power and decision value

| Quantity | Value |
|---|---|
| Calendar years | 7.5 (2016-07..2023-12). More stocks add breadth, **not** independent years |
| Tracking error of the overlay | about 0.7%/yr in the S&P 500; about 1%/yr assumed for mid caps |
| IR standard error | about 0.37 |
| IR needed to clear the bar | about 0.3–0.5 |
| Power if the true effect sits exactly at the bar | about 25–35% |
| Multiple testing | This would be HyperShift's fourth strategy test on the same 2016–2023 window. Any pass is exploratory at best |

**Verdict on economics:**
- The plausible effect falls short of the cost-adjusted bar.
- The test could not distinguish a small real effect from noise.
- A pass would be weak evidence because of prior use of the window.

**Expected decision value is low even if the data problem were solved.**
