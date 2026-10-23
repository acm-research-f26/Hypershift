# Overnight handoff (2026-10-24)

Worktree `Hypershift-overnight`, branch `overnight-20261024` (from a14fd5d), not pushed. All deliverables are in
`docs/overnight_20261024/`.

## 1. Executive summary: the five most important findings

1. **CRITICAL defect in the running test (C1, verified).** `edgar.extract_tokens` raises `ValueError` on inline-XBRL
   documents whose XML declaration names an encoding. Both Microsoft 10-Ks checked fail: 0001564590-22-026876 and
   0000950170-23-035122, both DFIN (`<?xml version="1.0" encoding="ASCII"?>`). Apple's Workiva files parse.
   - `lazy_signals.py` stores such documents as errors, so their pairs get **no signal**.
   - The scored sample is therefore selected by filing agent and by period.
   - A plain rerun re-fetches nothing, because errored token files exist.
   - Fix: `patches/C1_xml_declaration_and_refetch.patch`.
2. **The pre-checks would mis-measure (I2).** P4–P6 include errored rows (similarity 0, 0 tokens). "Parse error" is
   not an acceptable P4 explanation. Patch: `patches/I2_prechecks_exclude_errored_rows.patch`.
3. **The signal is fragile to formatting (I1) and to mass rewrites (I6).** Whitespace between HTML block elements
   changes tokens.
   - On Apple's FY2023 10-K, formatting alone moves 1 − cosine by about 40% of its year-over-year change (0.99968 vs
     0.99921).
   - COVID-2020 and the November 2020 Regulation S-K amendments cluster "changes" in 2020–2021.
   - Plus a spec deviation (I3): `signals_at` falls back to an older filing when the latest has no comparison.
4. **What the decision rule can detect (I4).** Alpha t ≥ 1 is the binding gate (residual IR ≳ 0.37).
   - The chance of CONTINUE is about 16% / 28% / 64% at a true IR of 0 / 0.15 / 0.5.
   - The literature prior (net IR 0.1–0.2) makes STOP the expected outcome even if the effect is real.
   - Synthetic overlay turnover is 1.6–2.3× NAV/yr, i.e. 16–23 bp/yr of costs against a 30–40 bp/yr gross ceiling.
5. **The literature points away from S&P 500 filing-text** and towards one testable gap.
   - Cheng et al. (JFQA 2026): no effect in the S&P 500. Recent large-cap practitioner tests are null. The flagship
     LLM-disclosure paper was withdrawn after a failed replication. Sadlo's paper could not be found under that title.
   - The best-supported free alternative is **earnings-announcement-return drift**:
     - Daniel–Hirshleifer–Sun (full text): big-firm leg 0.38%/month over 1972–2014.
     - Dickerson–Julliard–Mueller (full text, 2026): robust to 2022 and "not driven by micro-cap stocks".
     - Contradicted for SUE-based drift by Martineau.

## 2. Best alternative hypothesis

**An earnings-announcement-return drift overlay in the PIT S&P 500**, keyed on 8-K Item 2.02 acceptance times
(`NEXT_RESEARCH_SPEC.md`; ranked first of five in `OVERNIGHT_RESEARCH.md`). It is the only candidate that meets all
three constraints:
- free, exact point-in-time timestamps;
- applies to every S&P 500 member;
- has post-2010 evidence that is not driven by microcaps.

It needs no new data: the submissions JSON is already cached, and the PIT panel and engine are reused. Its stated
failure mode is that large-cap drift has been arbitraged away (Martineau) or is masked by one-month reversal.

**Caveat:** the signal itself is a price return. The non-price information is the dated disclosure event (8-K 2.02
acceptance). The spec's declared placebo diagnostic, the same 3-day return without event conditioning, tests whether
that event timing adds anything beyond price-only momentum or reversal.

## 3. Critical risks for the current experiment

| ID | Status | Location | Must act before interpreting? |
|---|---|---|---|
| C1: encoding-declaration parse failure | **Verified defect** | `edgar.py:86-90`; `lazy_signals.py:27-31, 61` | **Yes.** Count the errors (one-liner in `OVERNIGHT_RISK_REVIEW.md`), patch, re-fetch the errored documents, rerun `lazy_signals` and `lazy_checks`, log a dated amendment |
| I2: P4–P6 include errored rows | Verified | `lazy_checks.py:90, 99, 108, 118` | Yes (rerun the checks) |
| I3: fallback to an older filing | Verified (test) | `lazyprices.py:21` with `lazy_signals.py:84` | Count it; amend before scoring if > ~2% of signals |
| I1: formatting sensitivity | Mechanism verified; size on changers unknown | `edgar.py:103` | Diagnostic: share of changers whose filing agent changed, against the base rate |
| I4: power and the binding gate | Arithmetic | PREREG decision rule | Interpretation only |
| I5: `IR` entity priced with Gardner Denver 2017-05..2020-03 | Verified in the panel | PIT panel / fja05680 ticker reuse | No for this verdict; yes before reusing the panel |
| I6: 2020–2021 mass rewrites | General knowledge | — | Read per-year active returns |

## 4. Engineering completed

- `tests/test_sec_integrity.py`: **28 passed, 6 strict xfail** (C1 ×3, I1, I3, M5).
  - With the related suites: 78 passed, 6 xfailed.
  - With the C1 patch applied to a scratch copy of a14fd5d, the C1 cases XPASS and `test_edgar.py` passes.
- `patches/`: C1 and I2 patches. Both pass `git apply --check` against a14fd5d; use `--ignore-whitespace` if line
  endings differ. They were generated by `tools/make_patches.py` without touching the frozen sources.
- `tools/`:
  - `extraction_probe.py`: real-document C1/I1 measurements;
  - `turnover_sim.py`: synthetic turnover and coverage;
  - `sec_head_probe.py` and `sec_fetch_docs.py`: the 8 SEC requests used.

## 5. Decision tree for the running experiment (details in `MORNING_DECISION_TEMPLATE.md`)

```
Was EVAL produced after the C1 fix, with all deviations logged as amendments beforehand?
├─ No  → INCONCLUSIVE (invalid run): fix C1, re-fetch, rerun checks, one valid scoring under a dated amendment
└─ Yes → Net active IR at 10 bp >= 0.30 AND FF alpha t >= 1.0 ?
         ├─ No  → STOP: close S&P 500 changer avoidance; no tone or Risk-Factors variants on 2016-2023;
         │        start NEXT_RESEARCH_SPEC.md (EAR drift)
         └─ Yes → CONTINUE (not validation): one scoring of 2024-01..2026-09 under the frozen spec
                  + implementation-only paper trade (append-only signal and order log); no capital.
                  Check per-year concentration (2020-21) and the filing-agent-switch share first.
```

## 6. One recommended next action

**Before anything else**, in the main checkout:
1. Run the error-count one-liner from `OVERNIGHT_RISK_REVIEW.md` C1 on `data/edgar/signals.csv` (it counts errors,
   not returns).
2. If the count is nonzero, apply `patches/C1_xml_declaration_and_refetch.patch` and
   `patches/I2_prechecks_exclude_errored_rows.patch`.
3. Commit a dated PREREG amendment.
4. Rerun `scripts/lazy_signals.py` (only errored documents are re-fetched) and `scripts/lazy_checks.py`.
5. Only then run `scripts/lazy_eval.py`.

## 7. Repository changes

- **Commits on `overnight-20261024`:**
  - `436cf49` tests: SEC data-integrity tests
  - `0af0f71` docs: risk review, C1/I2 patches, tools
  - `d4c9e32` docs: literature review, turnover simulation, risk-review additions
  - `4821df4` docs: morning template, implementability, next spec, tests report
  - plus this handoff commit
- **Files:**
  - `tests/test_sec_integrity.py`
  - `docs/overnight_20261024/{OVERNIGHT_RESEARCH, OVERNIGHT_RISK_REVIEW, OVERNIGHT_TESTS, MORNING_DECISION_TEMPLATE, IMPLEMENTABILITY, NEXT_RESEARCH_SPEC, OVERNIGHT_HANDOFF}.md`
  - `patches/*.patch`, `tools/*.py`
- **Tests run:** full suite in this worktree: 523 passed, 35 skipped, 6 xfailed, 3 failed. The same 3 failures occur without the new file (missing data in the worktree: `test_baselines_r8`, `test_featbench`). The related suites alone: 78 passed, 6 xfailed.
- **SEC usage:** 8 requests to www.sec.gov (head probes of 4 documents, then a fetch of 4 documents), ≥ 2.5 s apart,
  with the User-Agent from the environment, never printed. Raw documents are kept outside the repository (`/tmp`).
- **Read but not modified:** the PIT panel (`member`, `observed`, `raw_close` of a few entities for identity checks).
- **Deliberately left unchanged:**
  - every frozen source and spec (`edgar.py`, `lazyprices.py`, `lazy_*.py`, `PREREG.md`): fixes are delivered as
    patches;
  - the main checkout (no reads of `EVAL.md`, `PRECHECKS.md`, `signals.csv`, token files, `cik_map.json` or
    `results/`);
  - `data/`;
  - `docs/HANDOFF.md` (owned by the main agent; not updated from this branch).
- **Tool note:** an attempted temporary edit of `edgar.py` in this worktree (to export a patch) was blocked by the
  permission classifier after one hunk had applied. It was reverted immediately with `git checkout`, and the patches
  were generated from copies instead.
