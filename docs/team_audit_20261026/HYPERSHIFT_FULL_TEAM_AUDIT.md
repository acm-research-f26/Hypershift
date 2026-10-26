# HyperShift — complete team research audit

Audit date: October 26, 2026 (America/Chicago). Recommendation: **GO for one bounded saved-prediction audit of the fixed complementary rank ensemble; DEFER claims of alpha or dynamic-relationship causality.** No training, protected-data scoring, purchase, merge, push or trading was performed.

## Scope and evidence standard

The connected GitHub plugin verified the repository, all six published branches (including pagination), branch heads and pull requests. A subsequent fetch downloaded current objects without switching or changing any working tree. The connector found no open or closed PRs in its repository search. Commit search returned only default-branch history; branch-specific Git histories were therefore inspected separately. Remote discovery is complete at this snapshot, not a promise that nobody pushed later.

Evidence labels used throughout:

- **RECOMPUTED:** this audit calculated a number from available committed artifacts.
- **PUBLISHED SUMMARY:** a committed table/report supports the claim, but the underlying predictions/ledger were unavailable for independent replay.
- **LOCAL RESEARCH:** a local commit supports the claim; no public GitHub file link is implied.
- **UNVERIFIED:** supplied discussion or unavailable research artifacts only.

The audit inventories every branch and research worktree, examines the important reports and relevant model/training/accounting/statistical implementations, and records missing artifacts. It does **not** claim line-by-line review of every historical source file, reproduction of every experiment, or inspection of private Discord/Drive content. Old negative findings were checked where they affect the next decision; the newest Rishi and Ethan updates received priority.

## 1. Branch-by-branch inventory

| Branch | Pinned head | Latest verified state | Audit implication |
|---|---|---|---|
| `main` (default) | `1274dff86c5f26e749275d7ec22b3e604aff61c7` | September 15 initial commit; README only | Default branch is not the research record |
| `feat-rs` | `0b06f26e81eb198fed222bac172fcbc60e7daeee` | October 26 presentation update; current model research through October 24 | Strongest historical economic candidate; portable summaries now published |
| `ethan` | `def988de1751d3321c58499b608f1b2ef06798ae` | October 26 fixed-point projection correction; preceding `5d8a72a` adds execution-aware strategy search | New infrastructure, no tracked performance report establishing its economic success |
| `kavin` | `6be3bf5f344e084704dea6886d02afa73c010050` | September 21 Cora GNN and Cora data | Root alignment/equity claims absent from published branch |
| remote `tom-shlom` | `26cd4758e6f58eb81e3ea70fb6d4af822d87c9f3` | October 6 reproduction closure/code search | Far behind local research; public branch alone misses the pivot |
| `team-report-20261023` | `38360b13f26b0cd1357775d428301c592520a37e` | October 23 PIT/style/filing/EAR/SP400 record and authoritative handoff | Published late-stage Tom research snapshot |
| local `tom-shlom` | `778f739` at audit start | 121 commits ahead of remote; latest handoff points to dynamic-HG STOP | Reports newer than public Tom branch; not all research branches merged |
| local `unify-20261024` | `0172a34e84e08b9b9d2a7a35bd5c276100f2b531` | October 24 reconciliation and 20-seed experiment | Unmerged into local Tom; report-count correction concerns CAGR, not RankIC |
| local `alpha-20261024` | `f40f4ce` | PIT fundamentals DEFER; later external-claims review | Superseded selection argument must not be repeated |
| local `gate-20261025` | `d7bf86ebf10edb57c0807546264f8f3367417df8` | Artifact gate; corrected selection discussion | Some summary-access limitations improved in latest feat-rs, raw-prediction block remains |
| local `dynhg-20261025` | `5ff3104e16fc97e3322f371db238258ae05c20eb` | Correlation-HG/no-graph/placebo evaluation; STOP tested implementation | Not a reproduction of Rishi's full transferred architecture |
| local `overnight-20261024` | `a74bee15c9b15e7085e98f1e729d24bcdc6b9a6c` | Independent feasibility/literature review | Ancestor of local Tom; already integrated |
| local `research/depth-alpaca-20261020` | `5a08176` | Latest committed milestone is cloud smoke and projected runtime | Infrastructure does not establish a completed depth result |
| local `codex/ablation-depth-20261019` | `4126fbe` | Exact-artifact dependency marked blocked | No completed positive finding verified |

Published branch links: [feat-rs](https://github.com/acm-research-f26/Hypershift/tree/0b06f26e81eb198fed222bac172fcbc60e7daeee), [Ethan](https://github.com/acm-research-f26/Hypershift/tree/def988de1751d3321c58499b608f1b2ef06798ae), [Kavin](https://github.com/acm-research-f26/Hypershift/tree/6be3bf5f344e084704dea6886d02afa73c010050), [published team report](https://github.com/acm-research-f26/Hypershift/tree/38360b13f26b0cd1357775d428301c592520a37e).

The required unification references exist: `a045a80` → `c38f822` (protocol) → `5c829bb` (results) → `0172a34` (corrections). Neither `0172a34` nor `5ff3104` is an ancestor of local Tom. The latest local Tom documentation references these side branches without integrating their whole contents.

### Research worktrees

The main checkout is local Tom. `../Hypershift-unify` is now on `dynhg-20261025`, despite its older name. `../Hypershift-rishi` stays detached at `be3523d` for old import compatibility; `../Hypershift-rishi-latest` stays at `06e2460` and is **not** latest anymore. `../Hypershift-overnight`, `../Hypershift-depthB`, the app-managed ablation-depth worktree, and `.claude/worktrees/team-report-20261023` are present. Seven older agent worktrees contain model, evaluator, data and experiment implementation work (`a566…`, `a578…`, `a706…`, `a894…`, `aa783…`, `ab231…`, `af88…`); they are historical engineering branches, not seven new economic studies. None was reset or modified.

`BRANCH_INVENTORY.json` provides exact refs, paths and recent changes. The working tree already contained modified handoff/review/progress files and unrelated untracked work; those were preserved.

## 2. Rishi — newest dynamic blend, resolved from code

### What the two networks actually are

1. **Parent:** equation-based THINK reconstruction, an N→16→N cross-stock mixer, and bounded AH+PM correction. Hyperbolic daily core width 16; 16-session input. The auxiliary 320→16→1 branch is centered across eligible stocks and bounded relative to detached core-score SD.
2. **Restricted dynamic model:** transferred author layers from `kijeong22/ijcai2025-spf@9477001ad9b85beb3455234675535e26c3d33e44`. Feature embedding width 32, four attention heads, temporal/intensity processing, two wavelet graph layers, 64 learned dynamic hyperedges, and learned fusion of own/dynamic/static representations. One training-only correlation snapshot replaces the author's broader sector/lead-lag graph inputs. It is neither PRISM-VQ nor Tom's simpler correlation-HG experiment.

Both consume 25 channels: five daily, ten after-hours, ten premarket. Normalization/imputation uses training data. Daily information ends at the preceding regular close; premarket ends 09:15 ET; decision 09:20; fills at that session's regular close. Rebalance every five exchange sessions. This is a close-executed five-session economic-return target, not minute trading.

**The October 26 MSE description needs a qualification.** The parent uses masked MSE on raw economic returns ×100, AdamW LR .001, decay .001, batch 8. The dynamic branch uses per-date `sqrt(masked MSE+1e-8)`, Adam LR .0001, zero decay. Both select the earliest prior-year validation-MSE improvement >1e-7, maximum 30 epochs, patience 6, gradient cap 1. MSE and per-date RMSE have related targets but different gradient weighting; they must not be called the same training objective.

Each seed's forecasts become eligible-stock percentile ranks with average ties. Average ranks within each family, then `0.5*parent_average + 0.5*dynamic_average`, without re-ranking the family means before averaging. One top-k portfolio follows. Raw forecast units therefore do not create arbitrary family weights. Boundary ties receive fractional weights. These are score blends, not two half-capital accounts.

Exact frozen seeds: **7,19,42,61,97,1828801906,1868046758,239154507,287727767,981756941**. Original five and new five are diagnostics; all ten form the declared primary estimator. Annual test folds 2022–25 use prior-year validation and up to five prior years of training, lower-bounded July 2019; outcomes crossing boundaries are purged. No test-year checkpoint selection is declared.

Sources: [model adapter](https://github.com/acm-research-f26/Hypershift/blob/0b06f26e81eb198fed222bac172fcbc60e7daeee/latest/models/ndx_reference_models.py), [training implementation](https://github.com/acm-research-f26/Hypershift/blob/0b06f26e81eb198fed222bac172fcbc60e7daeee/latest/training/ndx_reference_training.py), [frozen protocol](https://github.com/acm-research-f26/Hypershift/blob/0b06f26e81eb198fed222bac172fcbc60e7daeee/latest/config/ndx_reference_networks_20261024.json), [current charts](https://github.com/acm-research-f26/Hypershift/blob/0b06f26e81eb198fed222bac172fcbc60e7daeee/charts.md).

### Published economic results: one matched protocol

All following rows: historical-monthly NDX membership, January 3, 2022–December 31, 2025, 1,003 sessions, five-session forecast/rebalance, long-only share/cash accounts starting $1 M, net of the branch's modeled trading costs. Sharpe subtracts the daily equivalent of a fixed 3% annual hurdle and annualizes by √252. Drawdown is signed. Intervals are absolute pointwise 95%, circular bootstrap, 10,000 draws, block 20. **PUBLISHED SUMMARY**, not independently replayed here.

| Strategy | Seeds / k | Net SR [95%] | CAGR | Max DD | RankIC |
|---|---|---|---|---|---|
| Parent THINK+mixer AH+PM | ten /10 primary | .519 [−.345,1.425] |15.36%|−38.32%|.0038|
| Restricted dynamic alone | ten /10 |.530|15.32%|−36.62%|.0307|
| **Fixed dynamic blend** | **ten /10 primary** |**.755 [−.095,1.649]**|**23.92%**|**−34.88%**|**.0189**|
| Same blend |ten /5 secondary|.896 [.001,1.829]|31.87%|−40.94%|.0189|
| Historical selected leader |original five /5|.919 [.041,1.833]|32.94%|−34.89%|.0180|
| Same blend |original five /10|.848 [−.006,1.739]|27.79%|−35.04%|.0180|
| Same blend |new five /10|.467 [−.392,1.374]|13.27%|−35.76%|.0191|
| Parent |new five /10|.480|13.69%|−34.40%|.0067|
| Parent |ten /5|.531|16.73%|−44.05%|.0038|
| Dynamic alone |ten /5|.657|20.53%|see committed metrics|.0307|
| Tuned momentum SGD parent |ten /5 secondary|.719 [−.227,1.715]|25.05%|−45.29%|.0238|
| Tuned momentum SGD parent |ten /10|.626|19.22%|−33.81%|same estimator|
| MASTER transfer |ten /10|.164|2.99%|−41.13%|.0197|
| Parent + MASTER rank blend |ten /10|.382|10.58%|−42.38%|.0146|
| PRISM-VQ transfer |ten /10|−.037|−4.00%|−56.63%|−.0160|
| Parent + PRISM rank blend |ten /10|.522|15.95%|−39.55%|.0006|
| SPY |buy/hold|.498 [−.367,1.419]|10.85%|−24.55%|N/A|
| QQQ |buy/hold|.469 [−.408,1.394]|11.93%|−34.87%|N/A|

All important fields for these and cost-stress rows are retained in `RECOMPUTED_EVIDENCE.json` directly from [reference metrics](https://github.com/acm-research-f26/Hypershift/blob/0b06f26e81eb198fed222bac172fcbc60e7daeee/latest/results/reference_networks/metrics.csv) and [346-row comparison](https://github.com/acm-research-f26/Hypershift/blob/0b06f26e81eb198fed222bac172fcbc60e7daeee/outputs/01a15702-86c3-7463-ae6c-24606895c7cf/strategy_comparison_20261020/all_strategies.csv). Missing numbers above are not zero or estimated.

The MASTER transfer adds market-context channels to the shared stock inputs, using the pinned author network, width 256, temporal/spatial attention heads 4/2, Adam LR1e−5 and validation MSE. The PRISM transfer uses a two-stage VQ/factor-loading system and current-vintage JKP inputs with explicit publication-vintage limitations. Its feature set, horizon, normalization and training budget differ from the original paper. These standalone/blended negative or weak results count in the seven-method comparison family. They establish neither replication nor refutation of the original papers; they make another immediate architecture port a lower priority.

### Costs and uncertainty change the interpretation

The code charges one-way traded dollars at `multiplier*(.0007+.0003*sqrt(participation/.01))`, with signal-time ADV, target-order caps of 1% ADV, cash constraints, canceled unavailable fills and acquisition settlement. Base cost is **at least 7 bp**, participation-dependent, not a universal fixed 7–8 bp. Settlement timing and full fractional close fills remain approximations; no observed historical quote/auction impact calibration establishes capacity.

Ten-seed blend top 10: annual traded-notional/NAV turnover 24.91×, SR .755/.695/.577 and CAGR 23.92/21.69/17.37% at base/2×/4× modeled costs. Top 5: turnover 30.38×, SR .896/.830/.699 and CAGR 31.87/28.96/23.35%. Thus this daily candidate is **not** analogous to an intraday edge destroyed by a quarter-bp fee; selection and exposure attribution remain the larger unresolved issues. Cost stress preserves hypothetical fills and is not proof of execution feasibility.

Published paired top 10 SR differences, block 20:

| Contrast | Estimate | Paired 95% | Within-study simultaneous 95% (23 comparisons) |
|---|---|---|---|
| Blend − parent |+.236|[+.004,+.490]|[−.131,+.602]|
| Blend − dynamic alone |+.225|[−.116,+.567]|[−.299,+.748]|
| Blend − SPY |+.257|[−.161,+.649]|[−.351,+.864]|
| Blend − QQQ |+.285|[−.045,+.606]|[−.205,+.776]|

Source: [paired intervals](https://github.com/acm-research-f26/Hypershift/blob/0b06f26e81eb198fed222bac172fcbc60e7daeee/latest/results/reference_networks/paired_intervals.csv). These are conditional on fitted paths, not correction for every earlier experiment. The narrow positive unadjusted parent contrast is worth investigating, not confirmation. New-five top 10 blend−parent is −.0135 SR. Top 5 improves over the parent in both seed halves, but that secondary pattern cannot replace the declared primary.

Dynamic standalone RankIC .0307 is stronger than the earlier reversal estimate near .0183, but there is no available synchronized per-date contrast/CI against reversal on these exact new inputs. Do not retain the blanket claim that no learned model even descriptively exceeds reversal. Conversely, stronger whole-universe RankIC plus weak standalone economics shows why RankIC cannot substitute for tail and portfolio evaluation. Blend RankIC is lower than dynamic alone.

Rank correlation between families≈.410; top 10 overlap≈29%; raw errors correlate≈.998, net returns≈.911, QQQ-active returns≈.427. Different selections support a complementarity hypothesis, but not a proven distinct alpha source. Stable stock-ID filters, mixer weights and exposures are alternatives. Correlation .47 with an identity score does **not** mean 47% of prediction variance is explained.

### Reproducibility boundary

Latest portable audit asserts 120 checkpoint exports,160 stage checks and 172 independent portfolio replays, with largest NAV difference ~$7e−8. Those are **the contributor's recorded checks**, not this auditor's reruns. The October 26 overview claims reproduction of 346 daily paths, but its `analysis_paths.npz` is absent from Git. New daily curves, ledgers, weights, source caches, per-stock forecasts and original seals remain ignored/private. A manifest hash alone cannot provide the missing payload.

This audit independently replayed **204 older top-k/pseudo-Huber paths**, matching SR/CAGR/DD to maximum absolute errors 4.3e−15/7.9e−15/1.3e−14. That confirms older arithmetic, not the new dynamic leader. `recompute_audit.py` records this distinction explicitly.

## 3. Earlier Rishi results and historical selection

Older NDX results, same 2022–25 history but differing study inputs/objectives: StockMixer-inspired top 5 .66 SR/22.5% CAGR; its primary top 10 .33/8.6%; pseudo-Huber THINK+mixer top 5 .69/22.4%; original-five daily THINK+mixer top 10 .629/19.60%; 25-seed version .45/12.9%. These are selected historical strategy cells, not equal evidence units. Older whole-universe StockMixer RankIC .0186 vs reversal .0183 gives paired difference+.0003 [−.034,+.037].

The StockMixer-inspired five seed CAGRs 1.6–11.1% versus ensemble 22.5% are mathematically possible: rank averaging and top-k are nonlinear. They suggest sensitivity, not an accounting contradiction. The original-five THINK+mixer ensemble was above the 97.5 th percentile of 128 resampled 25-seed blends. This is a within-history seed-resampling diagnostic; sampling trained models is not a new market replication, and 128 draws do not precisely identify an extreme tail.

Older StockMixer top 5 P&L concentration: NVDA≈26%, top five names≈78%; omitting best three months reduces CAGR 22.5→9.2%. Older selected books had QQQ beta≈1.2 and best reported FF alpha t≈1.58. These figures belong to the older books, not automatically the new blend. Month deletion is retrospective influence analysis, not a trading rule. Full drawdown/exposure/cost tables are in local unification reports; new exposure attribution remains blocked.

Earlier AH+PM .629→.700 original-five improvement becomes .519 at ten seeds; new-five deterioration and sparse session coverage prevent a causal feature-value claim. Current MSE/RMSE blend, prior Huber/pseudo-Huber books and optimizer variants must stay separate.

**Selection correction:** 272 independent trials with a zero-SR null does not justify declaring .919 the expected maximum. That claim was withdrawn in local `gate-20261025:docs/gate_20261025/SELECTION_BIAS_ASSESSMENT.md`. The later record has 344 nonbenchmark summary rows,901 older tracked daily paths, active-return participation-ratio effective dimension≈5.5 and 58 eigenvectors explaining 90% variance. These are different measures, not an identified number of independent trials. Old DSR .28–.45 is assumption-dependent, not a probability that skill exists. The 346-row current catalog is exhaustive only for its declared active comparison scope, not every informal or retired study.

## 4. Tom — reconciled scientific record

All following results are **LOCAL RESEARCH** unless noted. No shared Sharpe leaderboard is valid across these protocols. Source paths are exact Git object paths; use `git show <ref>:<path>` where a side branch is required. NR means not reported in the specific ranking test or not independently extracted here, not zero.

| Study / source | Data, period, horizon | Result and comparator | SR / CAGR / DD | Costs and status |
|---|---|---|---|---|
| THINK reproduction; `docs/phase2/REPORT_CLOSURE.md`, local Tom | RSR NYSE/NASDAQ, original split;1 day | Paper advantage not reproduced under validation selection; ambiguous settings | Protocol-specific gross SR; not comparable net | No realistic costs in original evaluator; STOP that reproduction program |
| Star; `docs/screening_20261019/X_PAIRWISE_STAR_REPORT.md` |462 survivors;2020/22/23;5 day | Star−static+.0080 [−.0014,+.0176], Holm .146; star−none−.0001 [−.0102,+.0107], raw p .978 | NR: predictive experiment | Exploratory; no graph-specific result |
| Pairwise trading; same report |survivor panel;252 day formation/126 day trade | Cluster pairs 1.8%/yr [−1.8,6.0], random 2.6%; star−.2%; Holm 1.0 | Annual return, **not** a Sharpe statistic |10 bp per leg/side; NO EVIDENCE |
| Feature benchmark; `docs/featbench_20261022/FB_REPORT.md` |survivor S&P;2019–23 folds;5 day | Residual−raw LightGBM RankIC+.014, Holm .065 | NR | Corrected feature comparison+.0089 is a different test; original learner lacked planted-feature power |
| PIT style; `docs/strategy_pit_20261023/STRATEGY_PIT_REPORT.md` |PIT S&P;2018–23 weekly next-open | Net SR .11 [−.61,.90], alpha t .15 | CAGR/DD see study tables; no validated excess performance |10 bp/side plus 50 bp/yr borrow; STOP |
| Filing avoidance; `docs/strategy_lazyprices_20261023/REPORT.md` |PIT S&P 2016–23;monthly | Active IR−.01 [−.66,.68], ~zero annual active return | Active IR, **not** SR |10/20/30 bp study grid; STOP tested changer avoidance |
| Stable reporter diagnostic; `docs/strategy_ear_20261023/ALTERNATIVES.md` |same explored history |+1.19% active/yr, IR .34 [−.43,1.15]; LS+1.09%, IR .19, alpha t .42 | Post-hoc diagnostic |ARCHIVED; not a reopened successful filing strategy |
| EAR overlay; `docs/strategy_ear_20261023/STRATEGY_DECISION.md` |PIT S&P 2016–23;monthly delayed | Active IR−.47 [−1.11,.14];−.52%/yr; incremental IR vs price control−.08 | Active IR, not total SR |Study cost grid; STOP implementation |
| Kronos; `docs/screening_20261019/J_KRONOS_REPORT.md` |survivor S&P 2025–26;5 day | RankIC .003 vs static≈.019; paired−.0165,10/10 negative | NR |Only these J arms spent 2024+; does not authorize new strategies there |
| 20-seed transfer; `unify-20261024:docs/unify_20261024/SEED20_RESULTS.md` |PIT S&P 2020–23;202 weekly next-close 5 day origins | Reversal RankIC .01320; seed counts below |Median SR StockMixer/THINK/mix≈.22/.36/.28; CAGR 1.62/6.73/3.86%; median DD≈−48/−53/−43% |10 bp/side; EW10.8% CAGR is gross; seed experiment, not NDX replication |
| Fundamentals; `alpha-20261024:docs/alpha_20261024/FINAL_RECOMMENDATION.md` |first-reported XBRL/PIT S&P 2019–23;21 day | RankIC increment+.0184 [−.004,+.042],4/5 years,5/5 seeds; industry-neutral+.009 |Aug−price+2.4%/yr [−3.7,9.1], but−1.3% vs EW; alpha−4.8%,t−1.49 |20 bp figure; no book beats EW; DEFER |
| Independent correlation-HG; `dynhg-20261025:docs/dynhg_20261025/RESULTS.md` |PIT S&P 2020–23;201 weekly 5 day origins |Graph−self+.0013 [−.0037,.0062]; graph−random−.0001; graph−reversal−.007 |**Economics not run because gates failed** |STOP tested architecture; cannot invent a net SR |

The 20-seed CSV was recomputed in this audit: **RankIC > reversal =3/5/0**, in StockMixer/THINK/THINK+mixer order. **CAGR > gross EW≈10.8% =1/6/4**. The later 3/6/4 wording combines columns; no single metric has that vector. Median CAGRs from the CSV are 1.619/6.726/3.855%. Nested seed blends move THINK13.5→8.0→−.1% and THINK+mixer .4→7.0→12.0%; portfolio boundary changes explain why averaging more seeds need not improve CAGR.

The independent dynamic study did pass a **16-day** correlation synthetic gate, while its market configuration uses 60 days. Its learned-incidence reconstruction failed synthetic recovery. Neither result proves that the author's implementation or Rishi's transferred learned-incidence network fails. The small corr−self result equally present with random peers supports a pooling/context explanation for **that** experiment; it is not proof that real correlations can never help.

## 5. Kavin — root alignment remains artifact-blocked

The public branch contains README, `cora_gnn.py` and Cora data. No equity root-alignment implementation, yearly table, return series, seed register or HRCF experiment is present. The October 26 statement—returns improved without better ranking and inconsistently by year—is **UNVERIFIED** against repository artifacts. Exact changed holdings, cost treatment, exposure shifts and paired geometry ablations cannot be inferred.

Root alignment in the relevant HRCF literature centers embeddings in tangent space; a separate radial regularizer encourages spread. It could change optimization, ties, score scale or selected tails without improving global RankIC. That makes Kavin's observation plausible, not validated. Required exports: before/after code/config/checkpoint hashes, aligned forecasts/masks/security IDs/dates, all seeds, realized positions/trades/returns, exact centering formula/manifold, yearly and paired metrics. No new HRCF equity run is justified before those existing artifacts are available.

## 6. Ethan — what the latest code does and does not establish

The quote “SR 3.4 at .001 bp, negative at .25 bp” remains **UNVERIFIED**: no corresponding daily return tape, costs/turnover table, exact historical code hash or report was tracked. This audit cannot calculate its actual turnover or confidently diagnose a coding error from two Sharpe values. P&L mean, volatility and the trade tape are required.

Important code/description mismatch: current `market_ablation/portfolio.py` selects about the top**10%**, with positive-score eligibility for LO, and its path is not the quoted sector-relative top 20% policy. It applies delayed intraday origin+2 open execution, carries unquoted positions, compounds minute returns into daily returns and computes daily √252 Sharpe at rf 0. Costs are charged per dollar of **one-way** turnover; cash/holdings drift and closing liquidation matter. Do not attribute the discussion result to this implementation without its run manifest.

The October 26 `strategy_search` is newer and separate:66 configurations across 10 families,10 fees 0–5 bp =660 policy/fee combinations, plus cohorts/shadows/stresses. First 60earlier sessions fit normalization/PCA/calibration; five chronological development blocks select fixed candidates before later replay. `c_SR1` is the largest tested fee retaining SR≥1 under declared development checks; it is neither break-even alpha nor a statistical significance level. Later history was previously inspected.

New replay retains shares/cash, uses completed-close decisions and delay 2 (stress 3/5), reserves liquidation turnover, caps stocks 5%/gross 100%/sector 25% LO, uses tighter operating bounds, and includes preceding 20 session 10% volatility scaling. Warmup stays cash. The final patch stops projection at an exact fixed point **while retaining infeasibility for audit**, rather than silently calling it valid. This is a useful correction, not evidence of strategy profitability.

Limitations explicitly disclosed in code: static 2026 universe/sectors, fractional full fills at minute opens, absent bid/ask/partial fills/queue/impact/participation/capacity, assumed borrow and zero cash interest. Its risk and execution machinery is more careful than the original anecdote suggests, but its economics still need artifacts and quote-aware validation.

For a fixed trade tape, define `T_d=sum(abs(delta_shares)*fill_price)/NAV_start_of_day`; costs are `sum_j(abs(trade_dollars_j)*c_j/10000)`, including both entry and exit. A fully replaced 100% portfolio has about 2× one-way traded notional, before drift. Net returns must be recomputed from the share/cash ledger at each cost, as fees change NAV, sizing and potentially future risk scaling. `gross_edge_bps_per_traded_dollar` is a useful fixed-tape diagnostic, not a complete nonlinear replay or executable bid/ask bound. The two quoted SRs alone do not identify this quantity.

Sources: [assumptions](https://github.com/acm-research-f26/Hypershift/blob/def988de1751d3321c58499b608f1b2ef06798ae/experiments/strategy_search/assumptions.json), [execution](https://github.com/acm-research-f26/Hypershift/blob/def988de1751d3321c58499b608f1b2ef06798ae/experiments/strategy_search/execution.py), [statistics](https://github.com/acm-research-f26/Hypershift/blob/def988de1751d3321c58499b608f1b2ef06798ae/experiments/strategy_search/statistics.py), [older portfolio evaluator](https://github.com/acm-research-f26/Hypershift/blob/def988de1751d3321c58499b608f1b2ef06798ae/experiments/market_ablation/portfolio.py).

## 7. Missing evidence and consequence

| Missing | What remains unanswered | Owner/action |
|---|---|---|
| New Rishi predictions, daily ledgers, checkpoints, inputs/cache/source seals | Independent new-result replay, exposures, identity/tail attribution | Rishi: immutable export, no retraining |
| Complete cross-study trial registry | Dependence-aware full-search adjustment | Rishi+Tom: include failures/retired variants |
| Kavin root-alignment equity files | Any numerical or causal conclusion | Kavin: existing run bundle first |
| Ethan quoted-run manifest and daily/trade tapes | Exact turnover, cost threshold, source of 3.4 | Ethan: run-specific reconciliation |
| Private Drive/Discord artifacts | Additional unpublished results | No Drive/Discord connector or authorized artifact link available in this task; not inspected |
| Historical quotes/actual auction fills | Investable execution and capacity | Ethan: quote/capacity gate if daily candidate survives |

No blanket finding of “no alpha anywhere” follows. The defensible state is **one stronger exploratory ensemble candidate, several bounded nulls, useful methodological infrastructure, and incomplete replication payloads.**
