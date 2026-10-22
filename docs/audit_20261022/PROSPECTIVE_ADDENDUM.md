# Study FB prospective protocol — audit addendum (2026-10-22, before any post-freeze data exists)

This addendum **does not modify** `docs/featbench_20261022/PROSPECTIVE_PROTOCOL.md` (v1) or `FREEZE_MANIFEST.json`. Both stay byte-identical; `tests/test_fbaudit.py::test_freeze_hashes_unchanged` checks the code hashes.

- **What v1 keeps.** v1 runs once, as declared: frozen − reversal16 on FB neutral RankIC, at least 250 post-freeze days, stationary bootstrap block 10.
- **What this addendum adds.** It classifies the defects the audit found and predeclares **one** leak-free sensitivity run (v1-LF).
- **Timing.** Today is the freeze date and no post-freeze data has been fetched, so the addendum cannot be informed by outcomes.

## Defects found and their classification

| # | Defect | Evidence | Class |
|---|---|---|---|
| D1 | **Dividend look-ahead in `log_dvol_20` / `amihud_20`.** The v1 scorer refetches `auto_adjust=True` prices at scoring time. Scoring-date dollar volume then carries the adjustment for distributions paid between each scoring date and the fetch date. That factor depends on **future prices**, so it is look-ahead. The frozen model was also trained on features carrying up to 11 years of this factor. | A5: the future-only part of the factor has dev neutral RankIC +0.0305 [+0.024, +0.037]. The frozen-spec dev result is insensitive to correction (+0.0127 → +0.0136). | **(2): needs a separately versioned run.** It does not invalidate v1 in advance. The bias on the frozen model is upward or zero, and reversal16 is unaffected. |
| D2 | **v1 primary power.** P(SUPPORTED) at 250 days is about 0.07 if the true effect equals the dev contrast (+0.006), and about 0.28 at +0.020. | A1 prospective power table. | Not a validity defect. v1 stays as declared. **A NO EVIDENCE outcome is uninformative and must be reported that way.** Do not extend v1, and do not add interim looks. |
| D3 | **FB "factor-neutral" RankIC removes only beta × market and sector.** It rewards size, momentum and volatility exposure. | A4b: a model trained on the metric's own target scores +0.0248, and +0.0047 once style-neutralised. | **(2):** add a style-residual RankIC row to v1-LF as a predeclared *descriptive* secondary. v1's primary metric is unchanged. |
| D4 | `load_range` crashes when the extended panel ends in the first days of a year. | FB ledger, deferred minor. Not re-verified by the audit. | **(1): repairable without a spec change.** Fix it only if it occurs, and record the new sha256 here. The manifest is never rewritten. |
| D5 | **Universe drift.** Stocks acquired or delisted after the freeze may break the ticker-identity assert, or drop out of the extended panel. | Code reading (`prospective.load_range` asserts ticker identity). Unverified. | **(1)** if handled by the observed mask (a missing stock becomes ineligible). Record any code change and its hash here. |
| D6 | **Same-close execution.** | Not applicable: v1 is an IC test, not a trading test. The audit's A2 shows the frozen spec keeps 89% of its IC at next-close execution. | No action. |

**None of D1–D6 invalidates the v1 claim in advance.** v1 is classified **Compromised** after the fact if v1 and v1-LF reach different verdicts.

## v1-LF: predeclared leak-free sensitivity (run once, at the same time as v1, on the same dates)

Everything is identical to v1 (model, parameters, comparators, rule, minimum days, universe), except for three changes:

1. **Leak-free dollar volume.** `log_dvol_20` and `amihud_20` are computed from **unadjusted Close × Volume**, from a `auto_adjust=False` download made in the same fetch as v1. This applies everywhere those columns enter: the training features, the scoring features and the residual-target regressor. Returns stay total-return (Adj Close).
2. **Leak-free frozen model.** The frozen model is retrained on these leak-free features, with the same 50 rounds and the same seed.
3. **Extra descriptive secondary.** The style-residual neutral RankIC of the frozen model and of reversal16, computed with leak-free exposures.

**Data capture.**
- At the v1 fetch, also save the raw `auto_adjust=False` download (Close, Adj Close, Volume, Dividends) with its sha256, under `data/prospective/`.
- No further snapshots are needed: unadjusted Close is as-traded apart from splits, which cancel in Close × Volume.

**Reporting.**
- Report v1 and v1-LF side by side.
- If both are SUPPORTED, or both NO EVIDENCE, report v1 as the protocol result.
- If they differ, v1 is Compromised and v1-LF is the leak-free answer. Because v1-LF was predeclared here, it is not a post-hoc choice.

## Not justified: a separate matched raw-vs-residual prospective protocol

- **Effect size.** The leak-free development contrast is +0.0089 [−0.001, +0.019] (A5).
- **Mechanism.** The audit attributes most of it to tilt removal, which is available without retraining (A2, A2c).
- **Days needed.** Detecting 0.009 at 80% power needs about 0.047 × (2.8 / 0.009)² ≈ 4,500 post-freeze trading days, roughly 18 years.
- **Better route.** The backcast in `NEXT_STUDY_SPEC.md` answers the more important question (is there stock-specific skill at all?) with untouched data now.

## Code-change log (append-only)

| Date | File | Reason (defect #) | New sha256 |
|---|---|---|---|
| — | — | none yet | — |

---

## Correction (2026-10-22, BC pre-execution review; still before any post-freeze data exists)

This section supersedes the statements quoted below. Nothing above is deleted, so the audit trail stays intact.

1. **Withdrawn: "the bias on the frozen model is upward or zero".**
   - The direction of the leak's effect is **unknown**. In development, contaminated inputs *lowered* both LightGBM arms: raw −0.0012 contaminated vs +0.0048 leak-free, residual +0.0127 vs +0.0136 (A5).
   - A model can learn a spurious mapping from a leaked level, so the leak can hurt as easily as help.
   - The sign of v1 − v1-LF is therefore not predictable and must not be interpreted as a bias estimate.
2. **Withdrawn: "if both agree, report v1 as the protocol result".**
   - Both runs have about 7% power at the development effect (A1). They will most likely agree on NO EVIDENCE whatever the truth, so agreement validates nothing.
   - **Reporting hierarchy (binding from this commit):**
     - **v1-LF is the scientific result.** It is the only uncontaminated evaluation.
     - **v1 is the reproducibility record** of the frozen pipeline. It is always reported, and it is never cited as evidence of skill on its own.
     - If the two differ in verdict, v1 is labelled **Compromised**.
     - Either way, a NO EVIDENCE result is reported as **uninformative (underpowered)**, not as evidence of no effect.
3. **v1-LF is now fully specified and frozen** (it was not before this commit: no code, no raw-data format).
   - **Code:**
     - `src/hypershift/fbaudit/prospective_lf.py` (`load_range_lf`, `make_leakfree`, `score_lf`, which is v1's `score` unchanged, including the freeze guard);
     - `src/hypershift/fbaudit/leakfree.py` (`corrected_columns`, the FB formulas with unadjusted Close × Volume).

     Hashes are in the code-change log below.
   - **Raw data:** in the same session as the v1 fetch, a yfinance `auto_adjust=False` download of the same 462 tickers, saved as npz with arrays `tickers`, `dates` and `close`, where close is the unadjusted, split-adjusted Close aligned to the panel dates. Record its sha256 in this file before scoring.
   - **Everything else is identical to v1:** model, 50 rounds, seed, training window, comparators, primary contrast, test, at least 250 days, universe.
4. **Removed: the descriptive style-residual secondary** declared above for v1-LF.
   - The BC review (A7) shows that a style-only placebo model scores positive on both the linear and the quintile style-residual metrics. They therefore do not isolate stock-specific information, and reporting them would invite misreading.
   - v1-LF reports exactly the quantities v1 reports.

### Code-change log (continued; sha256 of the file with CRLF normalised to LF, the FREEZE_MANIFEST convention)

| Date | File | Reason | sha256 |
|---|---|---|---|
| 2026-10-22 | `src/hypershift/fbaudit/prospective_lf.py` | v1-LF implementation (D1) | `b02da95d1abe954d39264a35da37d9669aaf3ef2ac3441b0438251edc7f88f0e` |
| 2026-10-22 | `src/hypershift/fbaudit/leakfree.py` | leak-free dollar-volume formulas (D1) | `0ddc3aa34acc7a7bf6bac91d3cc350996a9a6b1284bd8607007d2d2855075b4c` |
| 2026-10-22 | `src/hypershift/fbaudit/prospective_lf.py` | **Supersedes the row above.** No fallback to the adjusted close: a stock-day whose 20-day window has a missing unadjusted close becomes ineligible, and the count (`fb.lf_dropped`) is reported at scoring | `3c76023a1e3d581b666c99af77e2d1105b1c50e6fce703b27e13919fb39ae3e0` |
| 2026-10-22 | `src/hypershift/fbaudit/prospective_lf.py` | **Supersedes the rows above (closure review).** After leak-strict dropping, `neutral` is recomputed with the post-LF eligibility mask exactly as `load_range` builds it, so dropped stocks no longer enter the market mean or sector demeaning. Dev emulation: about 490 dropped stock-days per year; pooled effect on neutral RankIC −0.00026; max daily 0.022 (A13). Class (1) implementation fix before any post-freeze data; the v1-LF specification is unchanged, so no new protocol version is needed. Test `test_lf_neutral_uses_post_lf_mask` | `060303473aa39b4c2727ec6e878407fe65427199a7b0c791b4975f12ae996f23` |
