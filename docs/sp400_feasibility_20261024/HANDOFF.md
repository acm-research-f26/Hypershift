# H. S&P 400 feasibility round: handoff (2026-10-24)

## Decision: **STOP: FREE S&P 400 APPROACH NOT VIABLE**

No strategy was scored. No protected data was used. Nothing was purchased.

## The six questions

1. **Can HyperShift rebuild a point-in-time S&P 400 universe from free data?** **No.**
   - Monthly Wikipedia revisions are always about 400 rows, but **lag the index**: of 93 companies promoted to the
     S&P 500 (dates from fja05680), 41% stay on the S&P 400 list 2–26 months too long.
   - The dated change log has the matching removal for only 81 of 93 (87%); the misses cluster in 2016–2019.
   - 46% of month-to-month snapshot changes have no logged cause, and 26% of logged changes never appear in the
     snapshots within 2 months.
2. **Can deleted companies and their prices be included?** Prices yes: 20/20 removed names have Alpaca bars.
   Identity only partly: 11/20 are verified. Two reused tickers resolve to the wrong company (TW→Tradeweb,
   SUNE→SUNation), and seven need name search because the source has **no CIK field** (its "SEC filings" links carry
   tickers).
3. **Can the SEC-text signal be applied without future information?** Yes: the frozen S&P 500 filing-text pipeline
   (timing, originals only, year-ago comparisons) transfers unchanged. The blocker is the universe, not the signal.
4. **Are mid-cap costs compatible with a useful net edge?** Marginal.
   - Clearing the 0.30%/yr net bar at about 1.2× NAV/yr turnover and 20 bp needs about 0.54%/yr active gross, i.e.
     mid-cap changers trailing the rest by about 2.2%/yr after decay.
   - HyperShift measured about 0.4–0.5%/yr in the S&P 500 over the same years. The literature suggests at most
     about 1.4× more outside the S&P 500, with no formal difference test.
5. **Is there enough evidence to spend the next cycle on this?** No.
   - The premise was overstated. Cheng, Liu, Qiao & Wang (four authors) report separate S&P 500 and non-S&P 500
     subsamples with similar point estimates and **no difference test**.
   - Sadlo (2020, S&P 1500; the paper exists, an earlier report was wrong) finds that changers underperform even in
     the S&P 500 (abstract only).
6. **A better alternative?** The highest-value next decision is about data, not another free-data signal
   (`ALTERNATIVE_RANKING.md`).

## Files

**Docs** (`docs/sp400_feasibility_20261024/`):
- `CURRENT_STATUS.md` (A)
- `LITERATURE_CORRECTIONS.md` (B) and `LIT_SP400_VERIFY.md`
- `SP400_PILOT.md` (C)
- `SP400_DATA_FEASIBILITY.md` (D)
- `ECONOMIC_FEASIBILITY.md` (E)
- `ALTERNATIVE_RANKING.md` (F)
- this file (H)

There is no `SP400_PREREG_DRAFT.md` (G): the feasibility gate failed.

**Scripts:** `scripts/sp400_membership.py`, `scripts/sp400_pilot_deleted.py`.

**Data (git-ignored):** `data/sp400/` (Wikipedia revision cache, probe outputs).

**Context updated for new sessions:**
- `docs/START_HERE.md` (new, authoritative);
- the `CLAUDE.md` top pointer;
- `AGENTS.md`;
- the head of `docs/HANDOFF.md`;
- memory.

## Tests

No production code changed this round; the suite stands at 575 passed, 2 xfailed (last full run on the EAR round).

## Unresolved

- The S&P 400 spread level is an assumption: no verified statistic, and no panel was built.
- The Gangopadhyay & Yook magnitudes rest on the abstract only.

## One next action

Ask the user to decide: **authorise Norgate Platinum ($630/yr)** for dated S&P 400/500/600 membership and
survivorship-free prices from 1990, then run a long-history test of filing-text changer avoidance in the S&P 1500 as
a labelled historical replication; **or pause** strategy research.
