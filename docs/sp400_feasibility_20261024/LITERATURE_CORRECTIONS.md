# B. Literature corrections and the S&P 400 question (2026-10-24)

Detailed verification with access levels: `LIT_SP400_VERIFY.md` (delegated agent). The Sadlo abstract was
re-checked directly through the Semantic Scholar API.

## Corrections to earlier HyperShift reports

| Earlier statement | Correction | Source and access |
|---|---|---|
| "Sadlo (2020) could not be found" (`LIT2_SYNTHESIS.md`, overnight review) | **Wrong.** Sven Sadlo, "Copy-Paste Outperformance: Lazy Investors and Copied Reports", SSRN WP, DOI 10.2139/ssrn.3748216 (2020; Crossref posting 2021). Sample: S&P 1500 constituents, 1996 to June 2020, 3,668 firms, 208,425 reports. The least-similar quintile earns 3- and 5-factor alpha of up to −5.12%/yr; changers underperform, but no-changers do **not** outperform. "Changers continue to underperform in the … universe of S&P 500 firms" (no magnitude given) | ABSTRACT ONLY (Semantic Scholar and Crossref agree; SSRN 403; no full text) |
| "Cheng, Liu, Qiao & Wang show the effect only outside the S&P 500, so it lives there" (`REPORT.md`, `ALTERNATIVES.md`) | **Overstated.** Table 12 (value-weighted, optionable stocks, 1996–2017) reports **separate subsamples, with no formal test of a difference.** Q5−Q1 is 0.21–0.26%/month in the S&P 500 (t 1.06–1.30) and 0.30–0.36 outside it (t 1.43–2.18). The point estimates are similar; insignificance in one group and significance in the other is not a significant difference | FULL TEXT (accepted manuscript; `-raw` text reconstruction) |
| Author list | Settled: **four authors**, Hua Cheng, Steve Liu, Zheng Qiao, Z. Jay Wang (title page of the JFQA accepted manuscript). "Cheng, Liu and Qiao" omits Wang | FULL TEXT |
| Lazy Prices size evidence | Cohen–Malloy–Nguyen report **no size split**. Quintile mean market values are $3.5B (Q1) and $2.5B (Q5). No S&P 400 or mid-cap study was found anywhere | FULL TEXT (NBER WP) |

## Ruling: is moving from the S&P 500 to the S&P 400 likely to improve a filing-text signal net of costs?

**No. The evidence is too weak to justify the move.**

| Level | What the evidence supports |
|---|---|
| Statistical predictability | Yes in broad samples ending 2014 (Cohen–Malloy–Nguyen) and 2020 (Sadlo, S&P 1500, abstract only) |
| Gross portfolio returns | Changer-side alphas exist in those samples. Size dependence is weak: Cheng's large vs small split is 0.19 vs 0.52–0.62%/month, but the S&P 500 vs non-S&P 500 split is only 0.21–0.26 vs 0.30–0.36, with no difference test |
| Net returns | No paper reports net-of-cost returns for any universe |
| Tradability | Our own S&P 500 test of 2016–2023 measured a changer shortfall of only about −0.4%/yr against the benchmark. Mid caps would need several times that to clear costs (`ECONOMIC_FEASIBILITY.md`) |

**Economic source of the effect, as the evidence reads:**
- **Changer underperformance** is the robust leg (CMN event time; Sadlo's asymmetry).
- **Stable-reporter outperformance** is not robust (Sadlo: none).
- **Small-firm inefficiency** fits the large- vs small-cap gradient in Cheng.
- **Options-market information** concentrates the effect in optionable stocks (Cheng).
- **Liquidity and short-sale constraints** are plausible but not tested in these papers.

None of this singles out S&P 400 names as the place where a long-only avoidance overlay would clear costs.
