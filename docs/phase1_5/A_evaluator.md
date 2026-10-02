# Phase 1.5 A: evaluator, target alignment, normalization leakage

*2026-10-01. CPU only. Code read: `src/hypershift/eval/metrics.py`, `src/hypershift/train/loop.py` (`window_offsets`, `gather_batch`, `predict_split`), `src/hypershift/data/rsr.py`. Data: `data/raw/rsr/data` (NYSE, 1737 stocks, T = 1245). Tests: `tests/test_phase15_eval.py` (13 tests; real-data ones are marked `data`). Paper = `docs/paper/icdm22-think.pdf`. The authors' reference code (fetched 2026-10-01 from GitHub, not in the repo): STHAN-SR `training/evaluator.py`, `training/load_data_nyse.py`, `training/train_nyse.py` (github.com/NDS-VU/STHAN-SR-AAAI21, commit 8d7861c) and the RSR repo `training/evaluator.py` (fulifeng/Temporal_Relational_Stock_Ranking, master). The THINK repo is empty, so no paper-specific evaluator exists.*

## Findings, ranked by impact on results

1. **Exact prediction ties are broken by ticker order, and ties are common (our code, not the paper).** At the selected epoch, THINK has an exact tie at the top-5 boundary on 72.8 of 237 test days on average (max 237, i.e. fully constant) over the 25 R5_g2 seeds; EE_none 52.2; HH_none 0.1. `topk_daily_returns` uses a stable sort, so ties go to the lowest index (alphabetical ticker). A constant prediction scores Sharpe **0.368** with lowest-index-first and **2.271** with highest-index-first (same 237 days, same masks). Re-scoring the saved selected-epoch predictions (25 seeds each, `results/R5_g2`):

   | tie rule | HH (THINK) | EH | HH - EH (Wilcoxon p) |
   |---|---|---|---|
   | lowest index first (current) | 0.089 ± 0.300 | 0.383 ± 0.758 | -0.294 (0.182) |
   | highest index first | 0.051 ± 0.307 | 0.418 ± 0.825 | -0.367 (0.182) |
   | random tie-break (mean of 100 draws per seed) | -0.006 ± 0.259 | 0.379 ± 0.699 | -0.385 (0.052) |

   The leak-free headline (HH 0.089) moves by about 0.1 across tie rules, which is small against the seed std of 0.3 to 0.76. The verdicts do not change (NO EVIDENCE for HH vs EH; the raw Wilcoxon p-value with random ties, 0.052, is borderline but was not Holm-corrected). The best-test Sharpe cannot be re-scored this way: only the selected-epoch predictions are on disk, so I did not test the "constant-prediction artefact" statements for best-test. The `all-tied 0.368` baseline quoted in `R5_full_nyse_g2.md` is tie-rule dependent (0.368 to 2.271), so it is not a meaningful reference. No tie rule is unambiguously correct and `metrics.py` is unchanged. Recommended reporting: add a random-tie-break score as a robustness column.
2. **`norm=paper` leaks the future into the inputs, and the leak carries signal (confirmed).** All R5/R8 full-NYSE runs used `norm: paper`. Section 4.
3. **The paper's Sharpe is not reproduced by our evaluator's formula in level, and the tracker misattributes it.** The RSR repo evaluator computes no Sharpe (`btl` only); the Sharpe constant comes from STHAN-SR (`* 15.87`). Differences are listed in Section 1. Numerically our SR equals the STHAN-SR SR for tie-free predictions (max |diff| in HH_none seeds 0-9 after the 15.87/sqrt(252) constant: 0, to 5 decimals). Only ties differ.
4. **Alignment is correct.** Prediction at window end t is scored against close(t+1)/close(t) - 1, with t+1 the first scored target 2017-01-03 and last 2017-12-08, 237 scored days. Section 3.
5. **No bug found in `metrics.py`, `loop.py` or `rsr.py`.** Minor properties (not bugs): the eligibility rule conditions on the target day being observed (59 of 410,364 candidate stock-days, 0.014%, dropped for a missing target; the authors do the same); `sharpe()` returns 0 when std is 0 (authors' code gives inf/nan).

## 1. Sharpe: exact specification

Code: `eval/metrics.py:10-36, 82-94`; called per epoch from `train/loop.py:200-203` on arrays from `predict_split` (`loop.py:145-157`).

| item | what the code does | verified |
|---|---|---|
| k | `cfg.topk = 5` (`config.py:44`) | yes |
| weights | equal, 1/len(top). `topk_daily_returns` = `gt[top, d].mean()` | `len(top)` = 5 on every test day (min eligible stocks per day 1726) |
| daily portfolio return | `mean over the 5 chosen stocks of gt[i, d]`, gt = (close_{t+1} - close_t) / close_t, so buy at close t, sell at close t+1, no costs, no slippage | recomputed from the raw CSV close column for all 237 days: max abs diff 3.7e-9 vs the saved `test_daily.npy` (R5_g2 THINK seed 0) |
| rebalance | every day, whole portfolio reset to the new top-5 | by construction; no turnover cost in `evaluate_all` (a net variant exists, `topk_daily_returns_net`, not used for headline numbers) |
| R_f | 0 (not subtracted) | `metrics.py:34-36` |
| annualisation | `* sqrt(252)` (`periods_per_year=252`) | |
| std | `np.std(r)`, ddof = 0. With ddof = 1 the Sharpe would be lower by sqrt(236/237) = 0.9979 (R5 seed 0: std 0.012638 vs 0.012665) | |
| eligible stocks | `mask[:, d] > 0.5`, where `mask` = min over the 16 input days AND the target day (`loop.py:46-47`) | 1726 to 1736 eligible per test day, mean 1731.2 of 1737 |
| missing / masked | a stock with a missing price anywhere in its 17-day window is excluded from the ranking that day; if no stock is eligible the day's return is 0 (never happens in test) | Section 3 |
| ties | stable sort of `-pred`, lowest index first (`metrics.py:13`) | finding 1 |
| scored days | validation 252 (targets 2016-01-04 to 2016-12-30), test **237** (targets 2017-01-03 to 2017-12-08) | yes, 237 |
| equal to hold-all | the same arrays give hold-all Sharpe 1.531 (mean 0.000422, std 0.004374 per day) | reproduces the doc value |

**Differences from the paper (p852 Sec. IV-B, Table II)**, as far as the paper states anything:
- The paper writes `SR = E[R_a - R_f] / std[R_a - R_f]` (no annualisation shown). Ours multiplies by sqrt(252). Whether the paper annualised is UNKNOWN.
- R_f: the paper does not give a value. Ours = 0. UNKNOWN.
- k: the paper says "top-k stocks" and gives no value. Ours = 5. UNKNOWN.
- Weights: the paper does not say. Ours equal-weight. INFERRED from the RSR / STHAN-SR code, not from the paper.
- Std ddof and test period length are not stated. Footnote 1 on p852 says the code is at github.com/shivamag125/ICDM22-THINK; that repo is empty.
- The paper says "buy the top-k at the closing price on day t ... sold at the closing market price of day t+1". Our gt is exactly close(t+1)/close(t) - 1, so the buy price is the day-t close. The paper does not name the buy price explicitly; close of t is the reading that matches the sell price statement. INFERRED.

**Differences from the authors' code** (STHAN-SR `evaluator.py`, the only authors' evaluator with a Sharpe; RSR's own evaluator has none):
- Authors: `sharpe5 = mean/np.std(sharpe_li5) * 15.87` (comment: "To annualize"). Ours: `* sqrt(252)` = 15.8745. Ratio 0.99972; negligible (HH_none seed 0: 1.69719 vs 1.69767).
- Authors divide by a constant 5; we divide by `len(top)`. Identical here (at least 1726 eligible stocks every day).
- Ties: authors use `np.argsort(pred)` (default quicksort, not stable) and read from the end, so among tied stocks they take the highest indices and, in a tie group larger than the cut, an arbitrary subset. On an all-zero vector it takes indices 1736..1732; ours takes 0..4. On the 25 THINK seeds the authors' evaluator gave a mean Sharpe of -0.070 vs our 0.089 (per seed differences up to 0.9) because of ties; with no ties (HH_none seeds 0-9) the two agree exactly after the constant.
- As shipped at commit 8d7861c, `evaluate()` raises `NameError` at `sharpe_li = np.array(sharpe_li)` (`sharpe_li` is never defined; the list is `sharpe_li5`). I removed that one line in a local copy to run the comparison above. So the published numbers cannot have come from this exact file.
- The authors' `btl5` is a sum of daily returns (`bt_long5 += ...`), not a product.
- Authors' `ndcg_score(list(gt_top5), list(pre_top5))` is the known NDCG bug (HANDOFF E3); ours is `ndcg5`, the correct one. Both are reported.

## 2. NDCG: exact specification

Code: `metrics.py:53-64`.
- Cutoff k = 5 (`cfg.topk`), via `sklearn.metrics.ndcg_score(..., k=5)`.
- Relevance: the day's true return minus the minimum true return among that day's eligible stocks (`rel = gt - gt.min()`, so all relevances are >= 0; negative returns are shifted, not clipped). Gains are **linear** in relevance (sklearn uses the raw relevance, not 2^rel - 1), with a log2(rank + 1) discount. Verified against a hand-written NDCG in `test_ndcg_definition_linear_gain_shifted_relevance_tie_averaged`.
- Ties: sklearn's default `ignore_ties=False` averages the gain over tied predicted scores (so NDCG is tie-neutral, while Sharpe is not, finding 1). Ties in the true relevance are handled in the ideal DCG by sorting.
- Eligible stocks: `mask > 0.5` that day. Days with fewer than 2 eligible stocks or `rel.max() <= 0` (all returns equal) are skipped.
- Daily aggregation: unweighted mean over scored days of the per-day NDCG@5.
- The paper reports "NDCG" (Table II) with no k, gains or relevance stated. All of the above is OUR choice and is UNKNOWN for the paper.
- Random baseline: `scripts/r5_g2_analysis.py:67-69` calls the same `ndcg_at_k(rng.standard_normal(G.shape), G, K)` on the saved test gt / mask, 100 draws, seed 0. Same function, same definition, same days and masks (`scripts/r8_analysis.py:88` likewise). Confirmed. A relabelling-invariance test of the baseline is in the test file.

## 3. Alignment (real data)

Raw layout: `NYSE_<T>_1.csv` columns `[date_idx, ma5, ma10, ma20, ma30, close]`, T = 1245 rows. `NYSE_aver_line_dates.csv` has 1274 rows; the last 1245 are the trading days (row 29 = 2013-01-02, row 1273 = 2017-12-08). This mapping is **INFERRED** (counting rows: 1274 - 1245 = 29 warm-up rows, last date equal to the RSR dataset's stated end); I did not verify a price against an external source.

Mapping, from the data (test `test_real_split_and_scored_dates`):

| split | input days | target days (scored) | windows |
|---|---|---|---|
| train | 2013-01-02 to 2015-12-30 | 2013-01-25 to 2015-12-31 | 740 |
| val | 2015-12-09 to 2016-12-29 | 2016-01-04 to 2016-12-30 | 252 |
| test | 2016-12-08 to 2017-12-07 | 2017-01-03 to 2017-12-08 | **237** |

- **Boundaries refer to target dates.** `window_offsets` (`loop.py:33-39`): `valid_index = 756` (2016-01-04) and `test_index = 1008` (2017-01-03) are the index of the first TARGET day of that split. Train targets are `< 756`; the last train input day is 754. Validation and test windows' inputs reach back 16 days into the previous period (history), but their targets never do. This matches the authors' loop (`train_nyse.py`: test offsets `range(test_index - seq - steps + 1, T - seq - steps + 1)`, target = `offset + seq + steps - 1`, steps = 1; same 237 and 252 counts).
- **Trace.** `gather_batch` (`loop.py:42-51`): inputs = days `offset..offset+15`, `base` = close at `offset+15` (= t, the window end), `gt = data.gt[:, offset+16]` = (close_{t+1} - close_t) / close_t, and the mask = min over days `offset..offset+16`.
  - First test window: offset 992, window start 2016-12-08, window end t = 2016-12-30, target 2017-01-03.
  - Day 0 top-5 of THINK seed 0: BBL (close_t 0.436520, close_t+1 0.454281, return +0.040688), NRP (0.134248 to 0.137781, +0.026317), A (0.645417 to 0.658592, +0.020413), AAN, AAP. Mean of the five gt = +0.01867118; saved `test_daily[0]` = +0.01867118. (Close values are on the shipped paper-normalised scale; the ratio is scale-free.)
  - Stock AKO.A has a missing close on 2016-12-20, 2016-12-28 and 2016-12-29 (4 missing days inside the first test windows). Its raw closes at t and t+1 are present and its `gt` is non-zero (-0.0063), but `mask` = 0 on days 0 to 2 (window contains a missing price), so it is excluded from ranking and never scored.
- **Independent check.** Recomputing the daily top-5 return straight from the raw CSV close column, with the rule "eligible = no -1234 in days t-15..t+1", for all 237 days gives max abs difference 3.7e-9 against the saved `test_daily.npy`; the Sharpe is 0.262266 in both (`test_real_daily_return_from_scratch` repeats this on seeded random predictions). `test_gt/test_mask` saved in the run folder equal `gather_batch` output exactly.
- **Missing prices.** All 410,305 eligible test stock-days: `gt` equals close_{t+1}/close_t - 1 computed from observed raw closes (0 mismatches, tolerance 1e-5), every eligible window has no -1234 in days t-15..t+1, and no observed day carries the 1.1 fill value in any feature. `parse_eod` sets `gt = 0` for a return into or out of a gap, and those zeros are only ever stored for stock-days whose window contains a gap, which the mask removes. 878 ineligible stock-days have a non-zero gt (target day observed, but an earlier input day missing): excluded, as intended. 13,820 eligible stock-days have gt exactly 0; these are real zero returns in the raw closes, not gaps (they match the raw recomputation).
- **Stale / forward-filled prices (property of the data, cannot be ruled out).** In the test period 13,993 of 411,442 observed consecutive pairs (3.4%) are identical closes; 1,766 stock-days sit in a run of at least 3 identical closes and 405 in a run of at least 5, spread over 27 stocks. The RSR files do not distinguish a halted or forward-filled price from a genuinely flat one. Our code cannot detect this, so such stocks are eligible with zero return. This is the same data the authors used.
- **Look-ahead in eligibility.** A stock is eligible only if its target-day price exists. This drops 59 of 410,364 stock-days (0.014%). The authors' evaluator does the same via `mask_batch`. Not a bug; it is a negligible selection on future availability.
- **Authors' code vs ours**: `gather_batch` reproduces `get_batch` of `train_nyse.py` (`mask min over [offset, offset+seq+steps)`, price at `offset+seq-1`, gt at `offset+seq+steps-1`) and `parse_eod` reproduces `load_EOD_data` (ground truth 0 and mask 0 on a gap, fill 1.1, NASDAQ last row dropped).

## 4. Normalization leakage

`norm=paper` (`rsr.py:81-105`) is the identity on the shipped files: every file divides all five columns by the stock's **full-series** max close. Verified: `max(close) = 1` for all 1737 stocks over all 1245 days (`test_real_paper_norm_uses_full_series_max`). `norm=train` (`renormalize_train`) divides by the max close over days `< valid_index`.

**Test (`test_norm_leak_paper_exists_train_none`, `test_norm_train_scale_uses_only_train_period`).** Synthetic RSR-format panel (trailing MAs, raw closes), written in the shipped format (divided by the full-series max); then future RAW prices are perturbed (a stock's level x3 from day s, a one-day x1.5 spike on another), the files are regenerated and the full load (`load_rsr`, then `gather_batch` windows with both `level` and `relative` inputs) is rerun:
- **paper**: features of days before s change. The perturbed stock's earlier features are rescaled by a constant factor (the leak is a per-stock constant, a function of the future max); the unperturbed stock is unchanged. Leak exists. s = 800 (after `valid_index`) and s = 1008 (at `test_index`).
- **train**: features, base prices and gt before s are unchanged for every stock (tolerance 1e-6 for float32/text rounding), and all val/test windows whose input and target lie before s are identical under `level` and `relative`. No leak from the future into val/test inputs.
- Caveat test: perturbing a price inside the training period (day 400) does change earlier days under `train` norm. That is a within-train look-ahead of the scale (max of 3 training years), it does not reach val/test inputs, and it cannot be removed except by `relative` inputs.

**How big is the leak in the real data (my own analysis, not a test of exploitability by the model):**
- For 38.2% of stocks the full-series max falls in the test year (index >= 1008); for 50.2%, in the validation or test years (index >= 756). For those stocks the paper divisor was set after the training period.
- The per-stock factor `log(max_full / max_train)` has Spearman +0.415 (p = 1.8e-73) with the stock's 2017 buy-and-hold return. A no-training rule that ranks by this static number scores Sharpe **4.37** on the 237 test days (top-5, same evaluator). That is an upper bound on what the leak CONTAINS; the models never see `max_train` directly. What a model can see is the window-end close level: ranking by LOWEST paper-norm close at window end scores 0.194, highest -12.3 (train-norm: 0.194 lowest, 0.281 highest). So the observable level is a weak signal, and whether THINK exploits the stock-level offset is UNKNOWN. The C1 shuffled-label result (same best-test Sharpe 2.5 with random labels) suggests the high best-test numbers do not come from this channel.
- Effect on our results: R5/R8 and the small-scale paper-protocol runs used `norm: paper`, so they carry this input leak. It cannot explain the leak-free result being poor (the leak would favour a good score). It does mean the paper-protocol numbers (and, if the authors used the same files, the paper's) are not leak-free in the inputs.

**Scope of C2 (honest statement).** C2 (`tests/test_controls.py`) mutates features, mask, gt and base price **after** they are built, from the target day onward, and checks predictions. It proves the model and window layer have no look-ahead, for both `level` and `relative`. It is **structurally blind** to normalization leakage, because the leak is in how the arrays are created (the divisor), upstream of anything C2 touches. It also does not cover the correlation-hyperedge builder (`base_hypergraph` uses days `1:valid_index` of gt, intended) or `select_universe`. The new tests close the normalization gap for RSR-format data; they do not test `data/fresh.py`.

## Tests added (all pass)

`tests/test_phase15_eval.py`: norm leak under paper and none under train (2 start days), train-scale caveat, missing-price handling and window exclusion, alignment on the synthetic market (t scored against t+1; train targets < valid; test last target = last day; history in val/test inputs), Sharpe equals the authors' loop up to the 15.87 constant on tie-free data, lowest-index tie rule and its sensitivity, NDCG against a hand-written linear-gain tie-averaged NDCG, degenerate-day skip and baseline function, and four real-data tests (scored dates and the 237 count, gt/mask trace to raw closes, from-scratch daily return, paper-norm full-series max).

## UNKNOWN / not verified
- Paper's k, R_f, annualisation, std ddof, buy price (INFERRED close t), NDCG k, relevance and gains: none stated in the PDF. UNREADABLE parts: none.
- The row-29 date mapping (INFERRED above).
- Whether the paper's authors' final evaluator tie-breaking matches either tie rule (their STHAN-SR evaluator is not stable; the THINK evaluator is unavailable).
