# BC pre-execution review request (abridged copy of the user's prompt, 2026-10-22)

**Task:** critically re-examine the proposed Study BC before any unused 2004–2012 price data are downloaded or scored. Do not defend it.

**Questions:**
1. Is the remaining style-residual RankIC real stock-specific information, reversal, an exposure-neutralisation effect, or an artefact? In particular, check the interaction between OLS residualisation and Spearman RankIC.
2. Correct the prospective-protocol claims: the direction of the leak bias, and whether agreement between v1 and v1-LF validates v1. Set a reporting hierarchy that puts uncontaminated evaluation first. Is v1-LF adequately frozen?
3. Stress-test the universe: survivorship, membership, sectors, ticker identity, corporate actions. Is point-in-time membership available? Define the restricted estimand.
4. Benchmarks: reversal16, a zero-parameter residual reversal, and the neutralised raw model. Use the smallest set with multiplicity control, and separate criteria for information, incremental value and economics.
5. Power and regime sensitivity: 2008, coverage, year dependence; time-series plus year-level inference; concentration in a few years. Do not equate a failed backcast with impossibility.
6. Economics: turnover control (buffer, lower frequency, next-open/next-close), an implementable hedge, matched exposures, realistic costs.
7. Decide APPROVE, REVISE or STOP.
8. Deliverables 1–10, ending with a freeze checklist.

**Constraints:**
- Do not download or score the 2004–2012 prices.
- Keep it inexpensive.
- No architecture search.
- Do not invent evidence.
