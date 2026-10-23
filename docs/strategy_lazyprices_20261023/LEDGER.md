# Ledger: filing-text round (2026-10-23), append-only

Each ruling: what was decided, why, and what it costs if wrong.

**Style-book closure and checks**
- Ruling: closed the style book with an append-only addendum to `STRATEGY_PIT_REPORT.md`. The earlier wording ("survivor artefact", "true edge about 0") is corrected to "not distinguishable from zero; several confounded causes".
  - Why: user instruction and the advisor's review.
  - Cost if wrong: none.
- Ruling: infrastructure checks found sector-proxy look-ahead affecting 18 entities and 201 stock-signals in 2018–2023. Not rerun; `sectors_causal` added; the next strategy uses FF12 from SIC.
  - Why: the look-ahead cannot move +0.11 to 0.60.
  - Cost if wrong: none for the decision.
- Ruling: opening fills verified (`price_index`, tested): the open never uses the same day's close. Rebuilding the panel with the tested function matches the stored data exactly.
  - Cost if wrong: none.

**Data access**
- Ruling: SEC filing documents require a contact email in the User-Agent. The user authorised using their account email, sent only to sec.gov and stored only in the user environment variable `SEC_USER_AGENT`, never committed.
  - Cost if wrong: none.
- Ruling: SEC's submissions-JSON acceptance time is Eastern time double-converted (+8 h EDT / +10 h EST). Verified on 8 index pages and inverted in code; the availability rule (next trading day) also absorbs any residual hour.
  - Cost if wrong: a one-day timing error on a few filings, which cannot matter for a monthly formation.

**Literature and implementation choice**
- Ruling: the implementation is long-only, FF12-industry-neutral changer avoidance on an equal-weight S&P 500 benchmark, chosen before any outcome.
  - Why: the changer leg is the persistent one in event time (Lazy Prices, NBER p. 17).
  - Cost if wrong: the stable-reporter leg is under-used; it is reported as a diagnostic.
- Ruling: full-document cosine similarity, not the Risk Factors section.
  - Why: the only post-2015 support for Risk Factors is an unreviewed replication inside our test window, so choosing it would be selection on outcomes.
  - Cost if wrong: a possibly stronger section is ignored.
- Ruling: within-form ranking (a departure from the paper's pooled ranking).
  - Why: avoids a seasonal 10-K tilt in the excluded set.
  - Cost if wrong: small differences from the paper's construction.
- Ruling: CONTINUE licenses only a protected 2024–2026 scoring plus an implementation-only paper trade. It does not license a Norgate purchase.
  - Why: Cheng et al. report the S&P 500 effect insignificant even in-sample, so a 1997–2015 replication cannot justify $630.
  - Cost if wrong: a delayed long-history replication.

**Identity mapping**
- Ruling: CIKs come from monthly Wikipedia revisions (as listed at the time) with SEC verification, not from today's ticker table.
  - Why: today's table splices reused tickers (DOW) and misses delisted names.
  - Cost if wrong: Wikipedia CIK errors, which the name and filing-coverage checks catch.
- Ruling: three wrong-company matches were found and fixed by stricter name rules before any signal was used (Churchill Downs, Endonovo, Paramount Gold). Fallback candidates require a strict name match, and an entity's own verified CIKs are tried first.
  - Cost if wrong: residual mis-maps among about 5 fallback CIKs.

**Overnight assignment**
- Ruling: the user's overnight plan was launched now as a separate agent in worktree `../Hypershift-overnight` (branch `overnight-20261024`), not after this experiment.
  - Why: the plan requires an agent that has not seen the experiment's results.
  - Its limits: no access to `EVAL.md`, `PRECHECKS.md`, `results/` or signals; at most 30 SEC requests.
  - Cost if wrong: none; the deliverables are unchanged.

**Pre-outcome checks and amendments (2026-10-23, daytime)**
- Ruling: the first pre-check run failed. 4,096 documents with an XML encoding declaration parsed to zero tokens. Fixed and re-fetched under amendment A1 before any return; the independent overnight review found the same defect (its C1).
  - Cost if wrong: none.
- Ruling: SEC's JSON acceptance time is inconsistently converted (double-converted for some filings, correct UTC for others). The signal uses only the availability day. The filingDate guard makes after-17:30 filings available one day later than the index-page time would; P1 counts this as conservative (same 9, later 3, earlier 0).
  - Cost if wrong: none (later, never earlier).
- Ruling: amendment A2 is adopted from the overnight review.
  - I3: the latest filing without a comparison now gives no signal.
  - I2: checks run on valid rows only.
  - I1: formatting is reported as a diagnostic (filing-agent switch ratio 1.30 among changers), not changed; the text rule stays frozen.
  - I5 (the IR entity carries Gardner Denver prices 2017-05 to 2020-03) is noted as a panel limitation and not fixed: one entity of about 500.
  - Cost if wrong: small.
Suite: .venv/Scripts/python.exe -m pytest -q -> 568 passed, 2 xfailed (I1, M5 known limitations) in 667.61s
