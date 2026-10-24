# START HERE: HyperShift session handoff (authoritative; updated 2026-10-24)

Read this file first in every new session. It holds the current state, the decisions already taken, how the work
is done, and the traps already found. `CLAUDE.md` holds durable environment and code rules. `docs/HANDOFF.md` is
the older live log; its first table mirrors this file.

## 1. Mission

**Goal:** find an economically useful, implementable US-equity strategy that earns incremental return **after
realistic costs**.

The project began as a reimplementation of the THINK hypergraph paper; that line is closed. Since 2026-10-22 the
work has been **evidence-driven strategy research**. Publication, model complexity and backtest counts are not
goals.

## 2. Current state

| Item | Status |
|---|---|
| Running processes | None |
| Last completed round | **Cross-branch unification with Rishi's `feat-rs`** (2026-10-24). Branch **`unify-20261024`** (worktree `../Hypershift-unify`, unpushed, not merged). Start with `docs/unify_20261024/FINAL_DECISION.md` on that branch. Verdict: no learned model in either branch beats 5-session reversal on whole-universe RankIC. feat-rs's top-5 headlines are best-of-k, best-seed-group cells (DSR 0.28–0.45), with QQQ beta about 1.2 and P&L concentrated in a few AI names. **GO: Stage 0** (re-score Rishi's saved predictions plus an identity-tilt control; no training). Then a shared prospective log from 2026-11 |
| Previous round | S&P 400 feasibility (`docs/sp400_feasibility_20261024/`): STOP, the free approach is not viable |
| Open decisions for the user | (1) Ask Rishi to export the Stage 0 files (list in `docs/unify_20261024/IMPLEMENTATION_PLAN.md`, task 1). (2) Whether to merge `unify-20261024` into `tom-shlom`. (3) Whether to spend the S&P 500 2024-01..2026-09 protected window on Stage A. (4) Still open: Norgate Platinum ($630/yr) or pause. Nothing has been purchased |
| Protected data | 2024-01..2026-09: unscored for every strategy since the pivot. Pre-2016: needs paid data |
| Capital | None authorised. Capital deployment is never an outcome of a research round |

## 3. Closed decisions (do not reopen by retuning on inspected periods)

| Strategy | Universe / period | Decision | Key number | Report |
|---|---|---|---|---|
| THINK hypergraph, price-only ML screens, Study FB | RSR NYSE/NASDAQ; S&P 500 survivors | STOP | No leak-free edge | `docs/phase2/REPORT_CLOSURE.md`, `docs/audit_20261022/FINAL_DECISION.md` |
| PRICE3 / momentum / low-vol / RR21 | S&P 500 survivors, 2016–23 | STOP | Active IR −0.74 | `docs/strategy_20261022/STRATEGY_REPORT.md` |
| Style-only LightGBM LS book | PIT S&P 500, 2018–23 | STOP | Net Sharpe +0.11 [−0.61, +0.90] | `docs/strategy_pit_20261023/STRATEGY_PIT_REPORT.md` (read its addendum) |
| Filing-text changer avoidance (Lazy Prices) | PIT S&P 500, 2016–23 | STOP | Active IR −0.01 | `docs/strategy_lazyprices_20261023/REPORT.md` |
| Earnings-announcement-return overlay | PIT S&P 500, 2016–23 | STOP | Active IR −0.47; no event information beyond its matched control | `docs/strategy_ear_20261023/STRATEGY_DECISION.md` |
| Stable reporters (post-hoc diagnostic) | — | ARCHIVED | Selected diagnostic; no independent evidence | `docs/strategy_ear_20261023/ALTERNATIVES.md` |
| S&P 400 filing-text expansion | Free Wikipedia membership | STOP (free approach not viable) | Promotion lag: 41% of names stay listed 2–26 months too long; change log 87% complete | `docs/sp400_feasibility_20261024/HANDOFF.md` |
| Paper trade of the style book | — | Retired before start | Its capital rule failed | `docs/strategy_pit_20261023/STRATEGY_PIT_REPORT.md` |
| feat-rs THINK / StockMixer / THINK+mixer headlines (Rishi) | NDX 2022–25 (spent) | Not evidence of skill | StockMixer RankIC − reversal5 +0.0003; top-5 blend 22.5% vs seeds 1.6–11.1%; 25-seed THINK+mixer 12.9% | `docs/unify_20261024/` on branch `unify-20261024` |
| Graph, residual targets, intraday/AH features, trees, overlays | Both branches | STOP (tested null on both) | — | `docs/unify_20261024/CROSS_BRANCH_OPPORTUNITIES.md` |

**Lessons that set priorities:**
- Large-cap edges after costs are small (IR 0.1–0.3 at best), and 7.5 years cannot confirm them (IR SE ≈ 0.37).
- **The binding constraint is data** (history length, universe), not models or ideas.

## 4. Next step options (choose with the user)

0. **Current GO (from the unification round):** Stage 0, then the shared prospective log.
   - Spec: `docs/unify_20261024/UNIFIED_EXPERIMENT_SPEC.md` §1 and §6 on branch `unify-20261024`.
   - Reusable code: `src/hypershift/unify/` (rank blend identical to feat-rs, deciles, DSR, prediction contract, buffered top-k).
   - feat-rs is inspected read-only in the detached worktree `../Hypershift-rishi`. Its data are not in git.

1. **If Norgate Platinum is authorised.** Build `data/norgate.py`: dated S&P 400/500/600 membership,
   survivorship-free prices from 1990, delisting dates. Then test the best-documented signal in the long sample:
   filing-text changer avoidance in the S&P 1500 (Sadlo 2020's investable universe). Label 1996–2015 a historical
   replication (it overlaps the literature) and keep 2024–2026 protected.
2. **If not.** Pause strategy research. Opportunistic insider purchases on free Form 4 data rank next, but the
   post-Dodd-Frank decline (Gangopadhyay & Yook 2021) makes another underpowered STOP likely.

Details: `docs/sp400_feasibility_20261024/ALTERNATIVE_RANKING.md`.

## 5. How work is done here (the standard the user expects)

**Assignments.** The user sends long research assignments written with another AI. They expect:
- autonomy;
- decisive GO / DEFER / STOP outcomes;
- numbered deliverables written to a dated folder `docs/<topic>_<yyyymmdd>/`;
- a final report.

When they ask "what to send other AI", reply with one paste-ready status block: results, decisions, open questions,
file paths.

**Experiment discipline (every strategy test):**
1. **Feasibility gate first.** Check timing, data integrity, identity, economics (cost vs plausible gross edge) and
   decision value. If the gate fails, STOP without scoring.
2. **Freeze `PREREG.md` and commit it before any forward return is computed.** It contains the hypothesis,
   universe, signal timing, portfolio, benchmark, costs, primary metric, CONTINUE/STOP thresholds (IR, factor alpha
   t, economic bar), a matched control where relevant, power and operating characteristics, and sample labels
   (literature discovery / HyperShift-explored / protected / prospective).
3. **Pre-outcome checks** (text, timing, coverage, identity) must pass. Fixes are dated amendments, **made before
   scoring**.
4. **Exactly one scoring run**, then the predeclared verdict. No variants on the same years after an outcome. A
   STOP closes that implementation only.
5. Report net of costs at 10/20/30 bp, with a stationary bootstrap CI (block 21), FF5+MOM+ST_Rev Newey–West alpha,
   per-year results, turnover and exposures.

**Literature standard:**
- Delegate verification to a Sonnet subagent that writes `LIT_*_VERIFY.md` with access levels (FULL TEXT /
  ABSTRACT ONLY / SECONDARY / NOT VERIFIED).
- Then **re-check the headline numbers yourself** in the PDF (`pdftotext -layout`), or through the Semantic
  Scholar API for SSRN abstracts.
- Never invent citations or numbers.
- A non-significant subsample next to a significant one is not a significant difference.

**Reviews.** Use the `advisor` tool before freezing a spec and before declaring a round done. Use an isolated
worktree agent for independent reviews that must not see results.

**Git:**
- Branch `tom-shlom`; **never push**.
- Commit with **explicit paths** only. Other sessions leave modified files: `docs/phase1_5/REVIEW_overnight.md`,
  `docs/phase2/PROGRESS.md`, untracked `docs/research_20261019/`, `kaggle/build_screen/` and the
  `scripts/queues/*.cmd` files; never commit them.
- `docs/HANDOFF.md` is edited but traditionally left uncommitted.
- Commit messages end with the Co-Authored-By and Claude-Session lines from the system reminder.

**Replies** use the caveman style the user enabled (terse; full words in code and commits).

## 6. Infrastructure you can reuse (all tested; full suite `pytest -q` → 575 passed, 2 xfail)

| Asset | Where | Notes |
|---|---|---|
| PIT S&P 500 panel 2016-01-04..2023-12-29 (678 entities, includes delisted names) | `scripts/pit_build_panel.py` → `data/pit/pit_sp500_2016_2023.npz` (git-ignored) | Alpaca SIP bars fetched with `asof` per fja05680 membership interval; total-return index rebuilt from raw bars (`data/pit_sp500.clean_returns`, `price_index`) |
| PIT loader, engines | `strategy/pit.py` (`load`), `strategy/pit_engine.py` (`run_pit`, `run_targets`) | Next-open fills; exits on index removal; delisting liquidation. Frozen `strategy/engine.py` must not be edited (paper-trade manifest hash) |
| SEC EDGAR | `data/edgar.py` (`SecClient`, `extract_tokens`, `cosine`, `match_prior`, `name_match`, `ff12`) | Needs the env var `SEC_USER_AGENT` (user-level Windows env; the user authorised their email for SEC only; never commit or print it). ≤ 8 req/s |
| Dated entity→CIK map | `scripts/lazy_cik_map.py` → `data/edgar/cik_map.json` | From monthly Wikipedia S&P 500 revisions (CIK column), SEC-verified; 99.8% of member-months |
| Filing similarity signals | `scripts/lazy_signals.py` → `data/edgar/signals.csv`, `data/edgar/tokens/` | Term counts plus sha256 provenance |
| 8-K Item 2.02 events, dated SIC | `scripts/ear_timing_audit.py` → `data/edgar/ear_events.json`; `scripts/ear_sic.py` → `data/edgar/sic_hist.json` | — |
| Strategy modules | `strategy/lazyprices.py`, `strategy/ear.py` | Signal timing, bins, industry-neutral tilts with cap |
| Factors | `strategy/factors.py` (Ken French daily FF5+MOM+ST_Rev to 2026-08) | — |
| S&P 400 probe | `scripts/sp400_membership.py`, `scripts/sp400_pilot_deleted.py` → `data/sp400/` | Free membership failed; kept for reference |

## 7. Traps already found (do not rediscover)

- **Survivorship.** The yfinance 462-name panel is survivors only (its equal-weight alpha is +3.1%, against −0.7% on
  PIT data). Use it only for engineering.
- **Alpaca corporate-action errors.** Spin-offs are recorded as reverse splits (TGNA, WRK, UTX, DLPH), which
  corrupts `all`/`split` series. Rebuild returns from `raw` bars.
- **Alpaca `asof` splices.**
  - `IR` carries Gardner Denver prices 2017-05..2020-03: exclude it before 2020-03-03.
  - Renames can leave duplicate entities; FBHS→FBIN looks like a delisting.
  - Multi-symbol requests can drop symbols: re-fetch singly.
- **SEC `acceptanceDateTime` in the submissions JSON has an inconsistent time zone** (sometimes double-converted).
  Use the **filing date** plus a next-trading-day rule. Never use the clock time.
- **lxml rejects str with an XML encoding declaration.** `extract_tokens` strips it. Re-fetch any documents parsed
  before that fix.
- **Name matching.** Ticker reuse and look-alike names cause wrong-company matches (Churchill Downs, Endonovo,
  Paramount Gold, Tradeweb, SUNation). Require SEC filing coverage and a strict name rule.
- **Wikipedia index lists** are reliable for S&P 500 CIK snapshots, but **not** for S&P 400 timing (stale for
  months).
- **The engine keeps non-eligible names** unless you use `pit_engine`.
- **Sector proxies from returns leak.** Use dated SIC from EDGAR headers.

## 8. Where the detail lives

- **Strategy rounds:**
  - `docs/strategy_20261022/` (pivot)
  - `docs/strategy_pit_20261023/` (PIT and style book)
  - `docs/strategy_lazyprices_20261023/` (filing text)
  - `docs/strategy_ear_20261023/` (EAR)
  - `docs/sp400_feasibility_20261024/` (S&P 400)
  - `docs/overnight_20261024/` (independent review, alternatives, morning template)
- **Literature verification:**
  - `docs/strategy_pit_20261023/LIT_VERIFY.md`
  - `docs/strategy_lazyprices_20261023/LIT2_VERIFY.md`
  - `docs/strategy_ear_20261023/LIT_EAR_VERIFY.md`
  - `docs/sp400_feasibility_20261024/LIT_SP400_VERIFY.md`
- **THINK history:** see `CLAUDE.md` "Study status" and `docs/PHASE1_TRACKER.md`.
