# F. Next research directions, ranked by expected decision value (2026-10-24)

Evidence: `LIT_SP400_VERIFY.md`, `../strategy_ear_20261023/LIT_EAR_VERIFY.md`,
`../overnight_20261024/OVERNIGHT_RESEARCH.md`. None of these was implemented in this round.

| Rank | Direction | Modern evidence | Timing quality | Costs and liquidity | Free history | Engineering | Probable net | Can it be evaluated independently? |
|---|---|---|---|---|---|---|---|---|
| **1** | **Decide on data before any further signal.** Norgate Platinum ($630/yr; needs your authorisation) gives survivorship-free prices from 1990 and dated S&P 400/500/600 membership | n/a | n/a | n/a | It *is* the missing history | About 1 week to integrate | n/a | Yes. 1990–2015 is untouched by HyperShift (literature-overlapping, so labelled "historical replication"). 2024–2026 stays protected |
| 2 | Filing-text changer avoidance, S&P 1500 (Sadlo's investable universe), **only with paid membership and long history** | Sadlo: changers underperform to 2020, including in the S&P 500 (abstract). HyperShift S&P 500 2016–23: about 0 net | Exact (EDGAR pipeline built) | About 1.2× NAV/yr overlay; 10–20 bp | 1996+ text (free); prices and membership need rank 1 | Small (pipeline exists) | Small, possibly 0 net | With 1996–2015, a power of about 0.6 for IR 0.4, unlike the 7.5 years now |
| 3 | Opportunistic insider purchases (Form 4, free XML from 2003; S&P 500 PIT panel exists) | Cohen–Malloy–Pomorski value-weighted 82 bp/month, 1989–2007. **Gangopadhyay & Yook (2021, J. Financial Regulation and Compliance 30(1) 43–59; abstract only): significantly less profitable after Dodd-Frank.** Post-2008 replications 60–70% weaker (secondary) | Filing within 2 business days; exact | Low turnover; sparse signals in large caps | Free from 2003 | 1–2 weeks (Form 4 parser, routine-vs-opportunistic classifier) | Likely below costs in the S&P 500 after the post-2010 decline | Only 2016–2023 for the S&P 500 (already used window) |
| 4 | Profitability / quality tilt (XBRL fundamentals) | A factor premium (FF RMW); post-2010 mid-cap evidence not verified. Buyback announcement returns vanished after 2003 (Fu & Huang 2016, per agent) | Exact with XBRL `filed` dates | Low turnover | XBRL from about 2009 | About 1 week | Cheap to buy through ETFs, so not an information edge | — |

**Ranking logic:**
- The binding constraint across five HyperShift strategy rounds has been **data, not ideas**: 7.5 explored years
  and one survivor-free universe. No free-data signal left on the list has a credible post-2010 large-cap net
  edge.
- Another free-data S&P 500 test (rank 3) would most likely end in another underpowered STOP.
- The highest-value next decision is therefore whether to buy long-history, survivorship-free data with dated
  constituents (rank 1). If not, pause strategy research instead of testing weaker signals on the same window.

**S&P 400 filing-text avoidance on free data is not on the list.** It fails the membership gate and is economically
marginal (`SP400_DATA_FEASIBILITY.md`, `ECONOMIC_FEASIBILITY.md`).
