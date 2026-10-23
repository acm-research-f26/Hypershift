# Prior art and novelty assessment: adjusted-price look-ahead

**Search scope.** Web searches on 2026-10-22 covering:
- dividend and corporate-action back-adjustment;
- look-ahead bias in adjusted prices;
- price-level features;
- vendor revisions;
- point-in-time versus restated data.

The search was limited. It was not a systematic review, so absence of evidence is reported as such.

## What is already known

| Source (verified) | Type | What it establishes | Overlap with our claim |
|---|---|---|---|
| Portfolio Optimizer blog, "Adjusted Prices Without Look-Ahead Bias" (updated 2022-07-03) | Practitioner | Backward (CRSP-style) adjustment makes past adjusted prices depend on later dividends. They change when new events occur (SPY example). Proposes forward adjustment | **The mechanism (dependence on future distributions) is known.** It does not mention future *prices*, and it does not measure predictive content |
| RealTest user guide; QuantInsti forum; ETF Trends; Financial Modeling Prep | Practitioner / vendor | Pre-adjusted data embed future corporate actions; level-based indicators are distorted; recommend as-traded or "past-only" adjustment for signals | Same general warning; no quantification |
| Ljungqvist, Malloy & Marston (2009), "Rewriting History", *J. Finance* 64(4), 1935–1960 | Peer-reviewed | Undisclosed retroactive changes in I/B/E/S recommendations (1.6%–21.7% of matched records across downloads) that are non-random and alter back-test results | Related phenomenon (vendor history changes over time); different data and mechanism |
| Kothari, Shanken & Sloan (1995), *J. Finance* 50(1), 185–224 | Peer-reviewed | Selection bias in COMPUSTAT-based book-to-market results (as described in secondary sources) | Universe / backfill bias; different mechanism |
| Brown, Goetzmann, Ibbotson & Ross (1992), *RFS* 5(4), 553–580 | Peer-reviewed | Survivorship-truncated samples can show spurious predictability | Our universe limitation |
| Kapoor & Narayanan (2023), *Patterns* 4(9) | Peer-reviewed | Taxonomy of ML leakage across 17 fields (294 papers) | General leakage framing |

Not found: a peer-reviewed study that
- (a) derives the dependence of back-adjusted *levels* on future *prices*;
- (b) measures the predictive association of the adjustment term with forward returns; or
- (c) shows the adjustment term changing conclusions of an ML return-prediction benchmark.

This is the result of a limited search, not proof of absence.

## Novelty judgment

| Element | Status | How to position it |
|---|---|---|
| Back-adjustment depends on future distributions | **Known** (practitioner literature) | Cite it as known; do not claim |
| Back-adjusted *levels* also depend on future **prices** through y_e = D_e / P(e−1), with an explicit sensitivity | **Possibly new as an explicit derivation**; likely implicit in practice | Present as a short derivation, with modest language ("we make explicit") |
| Which quantities are and are not contaminated (returns with in-window factors vs levels) | Known in practice; rarely written down precisely | Teaching value |
| A quantified predictive association of the adjustment term and its price-path component (+0.019; +0.094 for the price-path part) in a cross-sectional ML benchmark | **Not found in prior work searched** | The main empirical contribution, framed as a demonstration in one dataset |
| Sensitivity of benchmark conclusions to the contamination (two conclusions changed by about 0.005–0.006) | Not found | Supporting evidence. Modest in size |
| Diagnostics: as-traded level features, a truncation-invariance test, an adjustment-term-as-score test | Practitioner practice partly covers the first two; the third appears new | Present as a checklist |

**Strongest original contribution:** the **quantified demonstration**. The adjustment term, and specifically its future-price component, is predictive of forward returns in a standard data source. It visibly altered an ML benchmark's conclusions. It is detectable with a cheap scoring test. The mechanism itself is not new, and we position it as known.
