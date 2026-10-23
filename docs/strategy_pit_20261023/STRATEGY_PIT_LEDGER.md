# Ledger: PIT validation round (2026-10-23)

Each ruling: what was decided, why, and what it costs if wrong.

**Data and spec**
- Ruling: used Alpaca SIP plus fja05680 membership for a free 2016–2023 PIT panel instead of waiting for a Norgate trial.
  - Why: the probe showed delisted names (XLNX, TWTR, SIVB, FRC), `asof` rename mapping, and raw and adjusted bars; the Norgate trial carries only 2 years of history.
  - Cost if wrong: Alpaca data errors bias the test. Mitigated by checks E1–E3.
- Ruling: predeclared the kill test (`PREREG.md`, commit be76ca7) before building outcomes. The thresholds are STOP < 0.30 and CONTINUE ≥ 0.60 with alpha t ≥ 1.5 and P − R ≥ 0, framed as a kill test (MDE about 1.1).
  - Cost if wrong: none.
- Ruling: did not edit `engine.py`, which is hashed by the paper-trading manifest. Wrote `pit_engine.run_pit` instead: exits gated by tradability, delisted names liquidated at their last close.
  - Why: the frozen engine keeps weights of names that leave eligibility. It is exact against `engine.run` when every name is always eligible.
  - Cost if wrong: none.
- Ruling: one sector rule for all names (return correlation to leave-one-out GICS baskets).
  - Why: GICS for survivors plus a proxy for failed names would correlate sector assignment with survival.
  - Cost if wrong: 73% agreement weakens sector neutrality. A post-hoc GICS sensitivity was run and the decision is unchanged.
- Ruling: Amendment A1, before scoring. Returns are rebuilt from RAW bars, because Alpaca records spin-offs as reverse splits.
  - 12 member-days are zeroed.
  - The fraction 2/3 was removed from the share-ratio list after it falsely matched the SIVB and SEDG crash gaps.
  - Neutralised days get a zero return, because the ARNC bars of that day are themselves spliced.
  - Cost if wrong: the small true market move is lost on 12 member-days.
- Ruling: re-fetched three intervals dropped by multi-symbol `asof` requests one at a time, and aliased CCEP to CCE (fja05680 labels Coca-Cola Enterprises with its successor's ticker for 2016-01 to 2016-05).
  - Cost if wrong: none.
- Ruling: Ridge alpha fixed at 1.0. With about 10⁵ training rows its regularisation is immaterial.
  - Cost if wrong: none.

**Results**
- Ruling: decision **STOP**. P net Sharpe +0.11 is below 0.30.
- Ruling: arm R (Ridge) is reported as uninformative by construction. The residual target is residualised on Ridge's own inputs, so P − R is not cited as evidence.
  - Cost if wrong: none.
- Ruling: ran one post-hoc sensitivity, GICS sectors for survivors (not predeclared, labelled as such).
  - Why: the sector rule was the only modelling choice that could plausibly flip the reading.
  - Result: P −0.17, S −0.34. Decision unchanged.

**Downstream**
- Ruling: paper trade retired before its first signal.
  - Why: its own capital rule required this PIT re-test to pass, and it failed. Separately, `paper_trade.py` omits the membership filter that its spec requires (code ≠ spec).
  - The frozen spec, manifest and code are left untouched as the record.
  - Cost if wrong: a slow, bias-free record of a probably-null book is forgone.
- Ruling: H-B (PEAD/SUE) withdrawn.
  - Why: Martineau 2022, read in full text, finds no large-cap drift since 2006.
  - Cost if wrong: a contested effect is deferred (2025 rebuttals not verified).
- Ruling: the alternative directions are re-ranked by turnover and years of free point-in-time history. Lazy Prices comes before insider purchases, which come before the announcement premium.
  - Why: the announcement premium was first mislabelled as low-turnover (advisor). A long-announcers book runs about 20× NAV per year.
  - Cost if wrong: a better candidate is ordered lower.
- Ruling: no data purchased and no vendor account created. Norgate Platinum is recommended only after a frozen F1 spec passes the free 2016–2023 kill screen.
  - Cost if wrong: about two weeks' delay to a decisive holdout.
- Ruling: literature verification delegated to a Sonnet agent. Two headline facts were re-checked independently: Martineau's citation and abstract, and McLean–Pontiff 26%/58%.
  - Cost if wrong: an unverified number in the synthesis. Every claim carries its access level.

Final review: self-review plus two advisor passes (no separate reviewer subagent).
Suite: .venv/Scripts/python.exe -m pytest -q -> 497 passed, 3 warnings in 643.83s
