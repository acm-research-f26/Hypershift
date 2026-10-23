# Back-adjusted prices: derivation, what cancels, and what the decomposition can claim

## 1. Definitions

Let P(t) be the as-traded close, adjusted for splits only. Splits cancel in everything below provided price and volume are split-adjusted consistently. Distribution events (cash dividends, and anything else the vendor treats as a distribution) occur on ex-dates e with amount D_e. The cum-distribution close on the day before is P(e−1), and the per-event ratio is

  y_e = D_e / P(e−1),  0 ≤ y_e < 1.

**Backward multiplicative adjustment** (the CRSP-style convention used by Yahoo Finance's "Adj Close"; the vendor's exact implementation is not documented by us beyond its observable behaviour). Adjustment is relative to the fetch date T_f:

  F(t) = ∏_{e : t < e ≤ T_f} (1 − y_e),  P_adj(t) = P(t) · F(t).

F(t) ≤ 1, with F(t) = 1 once no distributions remain after t. **F(t) depends on y_e for every e > t: on future distribution amounts D_e, and on future prices P(e−1).** It also depends on T_f, so the same historical date gets a different P_adj in a later download.

## 2. Returns computed from adjusted prices

For a window t → t+h with h ≥ 1:

  P_adj(t+h) / P_adj(t) = [P(t+h) / P(t)] · [F(t+h) / F(t)].

Split the factors of F by event date:
- **ex-dates after the window (e > t+h)** appear in both F(t) and F(t+h) and **cancel**;
- **ex-dates inside the window (t < e ≤ t+h)** appear only in F(t), so they **remain**:

  F(t+h) / F(t) = ∏_{t < e ≤ t+h} 1 / (1 − y_e).

Therefore

  R_adj(t, t+h) = [P(t+h) / P(t)] · ∏_{t < e ≤ t+h} P(e−1) / (P(e−1) − D_e) − 1.

**Why this is a total return.** With one distribution in the window, the gross adjusted return equals the gross price return times P(e−1) / (P(e−1) − D). This is the gross return of a position that receives D at e and reinvests it at the price P(e−1) − D, the theoretical ex-distribution price. If the actual ex-date price equals P(e−1) − D, it is exactly the total return with reinvestment at the ex-date. Otherwise it approximates the total return, and the convention, not market data, determines the reinvestment price.

**What this means for information timing:**
- **Labels (forward returns from t):** use y_e only for e in (t, t+h], i.e. inside the label window. That is legitimate ex-post label construction. Distributions after the window cancel exactly.
- **Return features (backward returns ending at t, e.g. t−k → t):** use only y_e for e in (t−k, t]. They are **causal**: no information after t enters.
- **Correction to manuscript draft 1.** Draft 1 said the factor "cancels in ratios". Only the factors for events after the window cancel; factors inside the window remain, and that is what makes the ratio a total return.

## 3. Level features computed from adjusted prices

Any feature that uses the *level* of P_adj without dividing it by another adjusted price from the same factor regime keeps F(t). Examples:
- dollar volume P_adj · V;
- Amihud illiquidity |r| / (P_adj · V);
- price thresholds, such as a "price > $5" filter;
- round-number effects;
- log price levels.

  log(P_adj(t) · V(t)) = log(P(t) V(t)) + L(t),  L(t) ≡ log F(t) = Σ_{t < e ≤ T_f} log(1 − y_e).

A 20-day mean of P_adj · V averages F over the window. F changes only at ex-dates, so the contamination term is L evaluated across the averaging window. **L(t) is a function of data after t by definition:** of every later D_e and P(e−1), and of the fetch date.

**Sensitivity to the future price path.** Hold the future distribution amounts D_e fixed. Then

  ∂L / ∂ log P(e−1) = y_e / (1 − y_e) ≈ y_e.

A future price rise lowers future y_e and moves L toward 0. Every P(e−1) with e − 1 > t contains the price path from t onward, *including the label window*, so L is algebraically linked to the label. This is a statement about the construction, not an empirical claim.

## 4. What the trailing-yield decomposition can and cannot claim

**Quantities:**

| Symbol | Definition | Known at t? |
|---|---|---|
| L(t) | Full ex-post adjustment term | No. It is a function of post-t data |
| TY(t) = L(t) − L(t−252) = −Σ_{t−252 < e ≤ t} log(1 − y_e) | Distributions in the past year | **Yes** (the ratio cancels the future factors) |
| L̂(t) | Linear projection of L on TY, per date across stocks | It is a statistical summary, not an information set |
| L − L̂ | Regression residual | Not uniquely "future-only" (see below) |

**Why "future-only" is not justified for L − L̂:**
- **The residual is not the future component.** It equals L minus the part *linearly* predictable from one ex-ante proxy. Ex-ante information that TY misses (dividend growth history, payout policy announced but not yet paid, payment frequency, special-dividend history, sector norms) stays in the residual.
- **The future component is not removed by the proxy either.** Whatever part of the future distribution path is correlated with TY is absorbed into L̂.
- **Conclusion.** The residual is a mixture of ex-ante information not spanned by TY and ex-post information not correlated with TY. The +0.0305 rank IC of the residual is an **empirical predictive association of that mixture**. It is not an estimate of "the predictive content of future information".

**What is demonstrably unavailable at t:**
- the realised D_e and P(e−1) for e > t;
- hence L itself, and any construction that uses them.

What is *not* identified is the split of L's *predictive association* between "future" and "ex-ante-correlated" parts.

**A cleaner, construction-based decomposition** (added in this revision; `LEAK_MECHANISM.md`, `scripts/pub_leak_mechanism.py`). Both terms below use only post-t data, so neither is an ex-ante proxy. Each isolates one channel by construction.

  L_div(t) = L(T_end) + Σ_{t < e ≤ T_end} log(1 − D_e / P(t)),  L_path(t) = L(t) − L_div(t).

- L_div keeps the realised future distribution amounts but values them at today's price.
- L_path is the part that exists only because those amounts are divided by **future** prices.
- T_end = 2024-01-11 is the end of our unadjusted download. The tail L(T_end), for distributions between T_end and T_f, is reported separately.

**Results** (462 stocks, 2019–2023 signals, 5-day factor-neutral returns; exploratory, years inspected):

| Score | Rank IC [95% CI] | Mean cross-sectional SD |
|---|---|---|
| L | +0.0189 [+0.0080, +0.0298] | 0.082 |
| L_div | +0.0096 [−0.0015, +0.0210] | 0.089 |
| **L_path** | **+0.0935 [+0.0857, +0.1014]**, positive in all five years | 0.022 |
| L(T_end) | +0.0171 [+0.0069, +0.0273] | — |
| TY (ex ante) | −0.009 (CI includes 0) | — |
| L − L̂ (TY residual) | +0.0305 [+0.0242, +0.0373] | — |

**Strongest conclusions the evidence supports:**
1. **(Mathematical.)** Level features built from back-adjusted prices are functions of post-t distributions and post-t prices. Returns built from adjusted prices are not contaminated; features built from adjusted *levels* are.
2. **(Empirical, this sample.)** The future-price-path component of the adjustment term has a strong positive association with forward 5-day factor-neutral returns (+0.094). This is expected, because it is a yield-weighted function of future price changes that include the label window.
3. **(Empirical, this sample.)** Its contribution to the variance of the full term is small. The full term L is dominated by the distribution-amount component (rank correlation 0.99 with L_div), so L as a whole carries a smaller association (+0.019).
4. **(Empirical, this sample.)** The distribution-amount component's association is weak and not distinguishable from zero (+0.010, CI includes 0).
5. **(Not established.)** We do not establish:
   - how much of a fitted model's performance came from the contamination (our corrected refits changed headline contrasts by about 0.005; `CLAIMS_REGISTER.md` L2);
   - that the association is causal in any economic sense;
   - that the magnitudes generalise to other vendors, universes or fetch dates.

**Assumptions to state in the paper:**
- the vendor uses the multiplicative backward convention;
- ex-dates are identified from positive jumps of L (12,989 events over 2014-06 to 2024-01 for 462 stocks; 9 negative jumps, not modelled);
- unadjusted closes from the same vendor are split-adjusted and consistent with the adjusted series (checked: constant ratio to stored adjusted closes for 460 of 462 stocks).
