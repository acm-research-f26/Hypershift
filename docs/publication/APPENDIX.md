# Technical appendix: definitions and evaluation procedures

Notation: stocks i = 1..N; trading days t. A mask m_{i,t} marks eligible stocks.

## A1. Returns, labels and features

- **h-day label:** y_{i,t} = P_adj,i(t+h) / P_adj,i(t) − 1.
  - Defined only if the stock is observed on every day t..t+h; h = 5 in the S&P 500 studies.
  - This is a total return under the vendor's backward adjustment (`MATH_ADJUSTMENT.md` §2).
- **Factor-neutral label:** ỹ_{i,t} = y_{i,t} − β_{i,t} · ȳ_t, then demeaned within sector across eligible stocks.
  - ȳ_t is the equal-weight mean of y over eligible stocks.
  - β_{i,t} is the trailing 126-day beta against the equal-weight market.
  - Sector = current GICS.
- **Return features:** r^{(k)}_{i,t} = P_adj(t)/P_adj(t−k) − 1. These are causal.
- **Dollar-volume features (FB):**
  - log_dvol_20 = log(1 + mean_{20d}(P · V));
  - amihud_20 = 10^9 · mean_{20d}(|r^{(1)}| / (P · V + 1)).
  - As run in FB, P = P_adj (contaminated). In the corrected runs, P = unadjusted (split-only) close.
- **Per-date standardisation (cs_z):** z = (x − mean_t) / sd_t over eligible stocks, clipped at ±5.

## A2. Metrics

- **Rank IC on date t:** the Spearman correlation between predictions and labels over eligible stocks (computed if at least 20 are eligible). "Neutral rank IC" uses ỹ.
- **Pooled value:** the mean of annual-fold means; folds are 2019–2023.
- **Top-k Sharpe (THINK studies):**
  - daily return r_t = the mean of the 1-day returns of the k = 5 highest-scored eligible stocks;
  - Sharpe = mean(r) / std(r) · √252, with std using ddof = 0;
  - no risk-free rate, no costs;
  - ties broken by lowest index.
- **Style-residual metrics:**
  - LIN residualises ỹ per date on sector dummies + z(β, log_dvol_20, ret_60, vol_60);
  - NL adds quintile dummies of each of those four variables.

## A3. Inference

- **Stationary bootstrap** (Politis–Romano):
  - mean block length 10 for 1- and 5-day horizons, 2h for longer horizons; B = 5000;
  - pooled CIs average the fold bootstrap means;
  - two-sided p = Pr*(|m* − m| ≥ |m|).
- **Seeds.** Paired Wilcoxon signed-rank across seeds, with Holm correction within predeclared families.
- **Selection over a pool:** White's Reality Check and Hansen's SPA (consistent, studentised), plus CSCV PBO with 10 blocks.
- **MDE** = 2.8 × SE (80% power, two-sided α 0.05). For a single series of n days, SE = √(LRV / n), where the long-run variance is estimated by the block-10 bootstrap.

## A4. Adjustment decomposition (`MATH_ADJUSTMENT.md`)

- **The adjustment term:**
  - L(t) = log(P_adj / P);
  - y_e = 1 − exp(−(L(e) − L(e−1))) at detected ex-dates;
  - D_e = y_e · P(e−1).
- **The two components:**
  - L_div(t) = L(T_end) + Σ_{t<e≤T_end} log(1 − D_e / P(t));
  - L_path = L − L_div.
- **Trailing distribution yield:** TY(t) = L(t) − L(t−252).

## A5. Portfolio simulator (A8, A13)

- **Book.** Sector-neutral long top 10% / short bottom 10% (0.5 gross per leg; sectors with fewer than 10 eligible stocks skipped), plus an equal-weight basket of eligible stocks sized to zero ex-ante beta.
- **Schedules:** daily; weekly; weekly with a buffer (exit when outside the top/bottom 30%).
- **Execution.** Close t+1, or open t+1 (OHLCV opens); weights drift between trades.
- **Costs.** Net return = gross − c × traded notional / NAV.
- **Omitted:** borrow, financing and impact costs.

## A6. Learner configuration (FB, A11, A12)

- **LightGBM:** num_leaves 15, max_depth 4, learning_rate 0.05, min_child_samples 1000, subsample 0.7 (freq 1), colsample_bytree 0.7, reg_lambda 1, deterministic. Rounds: 50 fixed, or early-stopped on validation neutral rank IC.
- **Ridge:** α chosen on validation from {0.1, 1, 10, 100, 1000}.
