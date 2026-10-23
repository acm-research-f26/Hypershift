# HyperShift strategy pivot: report and decisions (2026-10-22)

**Bottom line.** HyperShift has no validated trading strategy.
- **PRICE3:** the predeclared price-based candidate failed its gate decisively. It underperformed equal weight by 5.5%/yr net, with factor alpha −4.6% (t −2.9).
- **The style-only characteristic book:** the best existing candidate looks good on the survivor panel. About half of its edge disappears under point-in-time membership, and the rest cannot be separated from survivorship.
- **Plan:** fix the data first (survivorship-free prices with index history, about $630/yr), re-test unchanged, and run the frozen paper trade in parallel. Do not deploy capital.

Evidence files in this folder:
- `EXPLORE.md`
- `STYLE_BOOK.md`
- `STYLE_ROBUST.md`
- `PAPER_TRADING.md` (frozen) and `STRATEGY_MANIFEST.json`
- `NEW_DATA.md` (C)
- `EVAL_DESIGN.md` (D)
- `IMPLEMENTATION.md` (E)

Plan and ledger: `docs/superpowers/plans/2026-10-22-strategy.md`, `STRATEGY_LEDGER.md`.

## A. Existing assets: keep, reuse, retire

| Keep (reusable now) | Why |
|---|---|
| `strategy/` engine, signals, factors, paper | New, tested, cost-realistic and execution-correct |
| `fbaudit/leakfree.py`, `matched.py`, `refit.py`, `stats.py`, `diag2.py` | Leak-free dollar volume; matched controls; bootstrap / Newey–West tools |
| `screening/data.py` panel loader, eligibility, beta | Correct causal plumbing |
| `featbench/data.py` features, `prospective.fit_fixed` | Feature set and fixed-config training (read-only; frozen) |
| S&P 500 daily panel (2015–2026) | **Engineering and debugging only.** Survivor universe; not for decisions |
| fja05680 membership file; Ken French factors | Point-in-time membership (survivors only); attribution |
| Test suite (483 tests), freeze manifests, ledgers | Process |

| Retire or archive | Why |
|---|---|
| THINK / hypergraph models (`models/`, `data/hypergraph.py`), screening architectures, Kronos | No economic case. Kept in git, no further work |
| RankIC- and style-residual-based model selection | Shown not to measure economic value (A7, A12) |
| Daily-rebalanced, unbuffered books | Turnover destroys the gross edge |

## B. Best immediate strategy

**Candidate: style-only characteristic book.**
- **Book:** sector-neutral long/short, top/bottom 10% per sector, 30% hold band, beta-hedged with an equal-weight basket, weekly.
- **Signal:** LightGBM, 50 rounds, 3 seeds rank-averaged, on four characteristics (beta, as-traded log dollar volume, 60-day return, 60-day volatility) plus sector, trained on 5-day factor-residual returns.
- **Execution and costs:** next-open, 10 bp per unit traded, borrow 50 bp/yr.

**Economic rationale (hypothesis).** Short-horizon, sector-relative mispricing that depends nonlinearly on risk and liquidity characteristics: liquidity provision and reversal conditioned on volatility and size. It is not captured by FF5 + MOM + ST_Rev (R² about 0.1).

**Evidence status (exploratory; 2019–2023 inspected; survivor-trained):**

| Test | Net Sharpe | 95% CI | FF alpha | t |
|---|---|---|---|---|
| Survivor universe, 3 seeds | +1.19 | [+0.48, +1.96] | +3.9% | 2.4 |
| Point-in-time members only | +0.61 | [−0.15, +1.47] | +1.5% | 0.9 |

- Sharpe at 20 bp: +0.89 on survivors, +0.34 on point-in-time members.
- Turnover is about 12.5× NAV per year.

**Benchmark:** zero (LS). The LO variant is compared against equal weight.

**Status: paper trading only (frozen spec; first trade 2026-12-01). No capital.**

**Not chosen: PRICE3** (momentum + low-volatility + residual reversal, monthly LO). NOT GO on all five predeclared checks:

| Measure | Value |
|---|---|
| Active IR | −0.74 [−1.30, −0.11] |
| Alpha | −4.6% (t −2.9) |
| Traded | 9× NAV per year |
| Worst year | −11% |

The LS version: net Sharpe −0.20. Single-factor tilts (MOM, LOWVOL) show no net edge over equal weight. RR21 has IR +0.19 at 10 bp but turns negative at 20 bp.

## F. Resource allocation

Ranked by expected decision-relevant information per unit of cost.

| Rank | Action | Cost | Information value |
|---|---|---|---|
| 1 | **Norgate trial, then Platinum; re-run WP-A unchanged on point-in-time data with delisted names** | $0–630/yr; 3–5 days of engineering | **Decisive.** Confirms or kills the only surviving candidate, and unlocks a 25-year untouched holdout |
| 2 | **Run the frozen paper trade** (weekly ops) | About 1 h/week, no data cost | Implementation proof and slow, bias-free evidence |
| 3 | Freeze and run a long-history holdout (1990–2014) for any candidate passing rank 1 | 1–2 days | MDE IR about 0.56: the first adequately powered test |
| 4 | EDGAR SUE hypothesis (H-B), or a Sharadar quote | 2–3 weeks; $0 or vendor fee | Tests genuinely new information; detects only large improvements |
| 5 | Learner-power simulation for ML signals | 1–2 days | Only needed if ML signals are pursued beyond the style book |
| — | Architectures, hypergraphs, RankIC optimisation, more price-signal tweaks on survivor data | — | None. STOP |

## G. Decisions

| Direction | Decision | Justification |
|---|---|---|
| Survivorship-free data (Norgate; Sharadar quote) | **BUILD** | The binding constraint on every historical claim; cheap; turns underpowered tests into testable ones |
| Paper trading of the style book | **BUILD** (frozen) | Survivorship-free by construction; low cost; implementation proof. Not alpha validation |
| Style book on point-in-time data | **INVESTIGATE** (rank 1) | The only candidate with a net edge on survivors; halved under point-in-time; the decisive test is cheap |
| EDGAR earnings-surprise drift (H-B) | **INVESTIGATE**, after rank 1 | New information not spanned by FF factors; short XBRL history limits power |
| Capital deployment | **DEFER** | No candidate has passed a survivorship-free test or a holdout |
| 2024–2026 pseudo-holdout; 2004–2012 history | **DEFER** (preserved) | Spend only on a candidate frozen after survivorship-free exploration |
| PRICE3 composite; MOM / LOWVOL tilts as alpha | **STOP** | Failed predeclared gates. Factor tilts are cheaply available elsewhere |
| Short-term reversal (RR21) alone | **STOP** | Edge vanishes at 20 bp; turnover about 15× NAV per year |
| FB residual LightGBM (full features) as a product | **STOP** | No increment over the style-only learner (A11); 20–30× NAV per year turnover |
| THINK / hypergraph / architecture research | **STOP** | No economic case after five studies |

## Rulings made in this round

See `STRATEGY_LEDGER.md`.
- PRICE3 NOT GO.
- The style book was re-tested as a saved candidate.
- The pseudo-holdout was not spent.
- The paper trade was frozen instead.
- Consolidated deliverables A/B/F/G into this report.

**Correction to `EXPLORE.md`'s capacity sentence.** At $100M the least-liquid names would need up to about 13% of a day's dollar volume, which is material. "Not binding" holds at about $10M (≤ 1.3%) but not at $100M without participation limits.
