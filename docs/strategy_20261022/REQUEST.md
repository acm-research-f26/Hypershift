# Strategy-pivot request (abridged copy of the user's prompt, 2026-10-22)

**Mission:** discover, validate and eventually deploy a strategy with positive, incremental, risk-adjusted returns after realistic costs. Reuse the repository; no architecture focus.

**Track A.** The simplest economically useful strategy, built from styles, momentum, reversal, residual reversal, volatility management and simple combinations:
- slower turnover;
- long-only and hedged versions;
- matched benchmarks and exposures;
- realistic costs, execution delay, drift, borrow and liquidity;
- reuse saved signals.

**Track B.** New information: point-in-time fundamentals and events. Assess uniqueness, point-in-time availability, revisions, coverage, delistings, licence and cost. Recommend the acquisition with the most information per dollar.

**Horizon.** Reconsider horizons; monthly or multi-month for fundamentals.

**Models.** Start simple. Validate ML on synthetic data first. Complexity must earn its place.

**Evaluation.**
- Net portfolio performance against a matched benchmark is primary.
- Report factor exposures, regimes, CIs and power.
- Keep exploratory and independent evidence separate.

**Data integrity:**
- point-in-time universe for any investability claim;
- no future adjustment in level features;
- preserve 2004–2012;
- do not reuse inspected years as confirmation;
- do not touch the frozen protocols.

**Work packages:**
- WP-A: a minimal viable strategy plus a frozen paper-trading spec.
- WP-B: the best new-data source plus one hypothesis, with its MDE and cost.

**Stopping rules.** Stop when:
- the edge vanishes under timing or costs;
- benchmark exposures explain it;
- the effect is too small for the data;
- the data is unreliable or too expensive;
- further tweaks would not resolve the issue.

**Deliverables:**
- A: assets
- B: best strategy
- C: new data
- D: evaluation
- E: implementation
- F: priorities
- G: BUILD / INVESTIGATE / DEFER / STOP decisions
