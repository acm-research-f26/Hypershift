# Morning decision template: S&P 500 filing-text changer avoidance

Fill it in from `EVAL.md` and `PRECHECKS.md`; it takes about 10 minutes. The rules come from
`docs/strategy_lazyprices_20261023/PREREG.md`. **No new pass criteria are added here.** Step 0 checks that the run is
the preregistered run; Step 1 is the official verdict; Step 2 is exploratory.

## Step 0: was the run valid? (spec conformance, not a pass criterion)

| Check | Source | Answer |
|---|---|---|
| 0a. C1 fixed before scoring: XML-declaration documents parsed, errored documents re-fetched, `PRECHECKS.md` regenerated afterwards (`OVERNIGHT_RISK_REVIEW.md` C1) | Amendments section of PREREG; error count in `signals.csv` | yes / no |
| 0b. Remaining parse errors explained, i.e. not a filing-agent or period cluster; P4 | PRECHECKS | yes / no |
| 0c. P1–P7 passed; P3 ≥ 75% coverage in each year 2017–2023 | PRECHECKS | yes / no |
| 0d. Every deviation from PREREG logged as a dated amendment **before** `lazy_eval.py` ran | git log order: amendment commit before the EVAL commit | yes / no |
| 0e. `lazy_eval.py` ran once, and the factor regression covered the whole window (`att['n']` = number of days; M6) | EVAL / rerun log | yes / no |

- **All yes:** go to Step 1.
- **0a or 0d is no:** the verdict is **not interpretable**. Go to the INCONCLUSIVE row. If EVAL was produced before
  C1 was fixed, it scored a vendor- and period-selected subsample, whatever its sign.

## Step 1: the official preregistered verdict (copy the numbers)

| Quantity | Value |
|---|---|
| Net active IR, A − B, at 10 bp [stationary bootstrap 95% CI] | ____ [____, ____] |
| Active FF5 + MOM + ST_Rev alpha, %/yr (Newey–West t) | ____ (t ____) |
| **Rule:** IR ≥ 0.30 **and** t ≥ 1.0 → **CONTINUE**; otherwise **STOP** | **____** |

When reading the verdict (`OVERNIGHT_RISK_REVIEW.md` I4):
- The IR standard error is about 0.37, and t ≥ 1 needs a residual IR of about 0.37 or more.
- The chance of CONTINUE is about 16% at a true IR of 0, 28% at 0.15 and 64% at 0.5.
- So a CONTINUE at IR 0.3–0.5 is roughly a coin flip between a real effect and noise.
- A STOP is the expected outcome even if the true IR is 0.1–0.2, the literature prior.

## Step 2: exploratory observations (non-gating; they never overturn Step 1)

| Question | Where to read it | Note it as |
|---|---|---|
| Positive after costs? | Active return at 10, 20, 30 bp | sign at each cost |
| Beats the designated benchmark? | Same as Step 1 (the benchmark B is the equal-weight PIT S&P 500 with the same costs) | — |
| Survives factor controls? | Alpha t; exposures (is it mostly ST_Rev or MOM?) | alpha share of active return |
| Economically large? | Active %/yr at 10 bp. The ceiling stated before scoring was 0.3–0.4%/yr gross (`LIT2_SYNTHESIS.md`) | above / below the ceiling. Far above means suspect: re-check I1/I6 |
| Stable enough for a follow-up test? | Per-year active returns: sign count out of 8; share from 2020–2021 (I6) | e.g. "5/8 positive; 2020–21 = 70%" |
| Turnover realistic? | A vs B turnover; overlay turnover. Synthetic expectation is 1.6–2.3× NAV/yr one-way (`IMPLEMENTABILITY.md`) | within / outside the range |
| Concentrated? | Changers per formation; maximum active weight (M3); legs: changers − B vs stable − B | which leg carries it |
| Driven by formatting? (I1) | Share of changers whose filing agent changed, against the base rate | ratio |
| Data reliable and affordable? | Parse-error count after the fix; CIK coverage 0.998 | free; reliable yes/no |
| Supports prospective evaluation? | Only if Step 1 says CONTINUE | — |

## Step 3: outcome and the exact next action

| Outcome | When | Exact next action |
|---|---|---|
| **STOP** | Step 0 all yes and Step 1 fails | Close S&P 500 changer avoidance. Do **not** try a tone-signed or Risk-Factors variant on 2016–2023: that would select on the outcome. Start `NEXT_RESEARCH_SPEC.md` (earnings-announcement-return drift). The protected 2024-01..2026-09 window is released from this strategy; the next spec says how it may be used |
| **INCONCLUSIVE** | Step 0 fails (0a or 0d). PREREG has no inconclusive statistical outcome, so this arises only from an invalid run | Would more information resolve it? **Yes, but only one thing:** a single valid scoring under a dated amendment (C1 fix and re-fetch), reported as a deviation together with the invalid first run. More analysis of 2016–2023 cannot resolve a statistical ambiguity, because the minimum detectable IR is about 1.0 |
| **CONTINUE** | Step 0 all yes and Step 1 passes | Not validation. Licensed, nothing else: (i) **one** scoring of the protected window 2024-01..2026-09 under the frozen spec. Fetch Alpaca bars and filings for that window only, rebuild the PIT panel extension with the same code, and run `lazy_eval` once with START/END changed by a dated amendment and nothing else. (ii) Implementation-only paper trading (next row). No data purchase |
| **IMPLEMENT FOR PAPER TRADING** | Only together with CONTINUE (PREREG licenses both at once) | Daily EDGAR poll of S&P 500 10-K/10-Q acceptances; frozen extraction and similarity; monthly target weights at the first trading day's open; append-only signal and order log with document sha256 and the code hash (see `IMPLEMENTABILITY.md`). No capital. Verify that live acceptances, parse rate and turnover match the backtest mechanics |

Capital deployment is not an outcome of this test.
