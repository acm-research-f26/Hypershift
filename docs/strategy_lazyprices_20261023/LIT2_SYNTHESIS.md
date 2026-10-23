# B. Filing-text research synthesis and choice of implementation (2026-10-23)

Evidence base: `LIT2_VERIFY.md`, written by a delegated literature agent with an access level on every item. Two
headline facts were re-checked against the PDFs:
- Lazy Prices, NBER WP 25084, p. 17 and Table II;
- Cheng–Liu–Qiao–Wang, JFQA accepted manuscript, Table 12 and p. 40.

Anything resting on an abstract or a secondary source is marked as such.

## What the evidence says

| Source | Access | What it shows | What it means for us |
|---|---|---|---|
| **Cohen, Malloy & Nguyen, "Lazy Prices"** (NBER WP 25084, rev. 2019; JF 2020) | Full text (NBER version) | 1995–2014, all 10-K/10-Q filings. Each is compared with the same report one year earlier, using four similarity measures (cosine on term frequencies, Jaccard, min-edit, simple diff). Quintiles are formed monthly; stocks enter the month after filing and are held 3 months. Value-weighted Q5−Q1 earns **0.34%/month (t 2.53)** on cosine and up to 0.58% on other measures. **Both legs** carry alpha in calendar time: VW cosine Q1 −0.12 (t −1.38), Q5 +0.23 (t 2.23); Jaccard −0.23 / +0.23. In **event time** "any positive alpha on the Q5 long side quickly reverts to zero, while the negative alpha persists and increases up to 6 months out". There is no announcement-day reaction. Changes are 86% negative in tone. The paper has no large-cap or S&P 500 table, no net-of-cost result and no Sharpe ratio | A documented historical anomaly, broad universe, ending 2014. The durable component is the **changer (avoid) side** |
| **Cheng, Liu, Qiao & Wang, "Attentive Options Traders"** (JFQA, accepted March 2026; four authors, not three as listed in the request) | Full text (accepted manuscript); table layout scrambled, mapped through the text | Option traders react to 10-K changes (volatility smirk). The stock effect is concentrated in optionable firms. Table 12 (VW, 1996–2017): **"no significant difference in returns between Q1 and Q5 for large-cap stocks … the effect is significant only for non-S&P 500 stocks"**. S&P 500 Q5−Q1 alphas are about **+0.22 to +0.26%/month, t ≈ 1.1–1.3** (cell-to-panel mapping inferred from the scrambled layout) | **The strongest direct prior for our universe: the effect was weak and insignificant in S&P 500 stocks even in-sample** |
| Sadlo (2020/21), "Copy-Paste Outperformance" | Listing only; SSRN returned 403 | — | **Not verified.** Not used |
| Post-publication tests | None peer-reviewed | Two unreviewed GitHub replications on current constituents (survivorship-biased). Full-document 10-K cosine: S&P 100 Sharpe −0.11; S&P 500 L/S −10 bp/month (t −1.2). One S&P 500 Risk-Factors-only variant: +15 bp alpha (p 0.08) | No credible post-publication evidence either way. **The Risk Factors section is not chosen**: its only post-2015 support comes from an outcome inside our test window, so choosing it would be selection on outcomes |
| Brown & Tucker (2011, JAR) | Abstract only | MD&A modification scores rise with economic change, but the price reaction to them weakened over the decade | Changes carry information, and attention to them has been decaying |
| Adosoglou et al. (2021, ESWA), embeddings | Abstract only | Long non-changers; large alphas claimed, 1998–2018 | Not used (no sweeping of encoders) |
| McLean & Pontiff (2016) | Full text (earlier round) | Returns 58% lower post-publication on average | Haircut the prior by about half |

## Answers to the five questions in the request

1. **Changer underperformance:** yes. It is the persistent side in event time, and the larger, more significant side
   in value weight for Jaccard and Simple (CMN). In Cheng et al.'s S&P 500 panel the Q1 alphas are negative and the
   Q5 alphas near zero (mapping inferred).
2. **Stable-reporter outperformance:** present in calendar time, but it "quickly reverts to zero" in event time.
3. **Both legs:** yes in calendar time. The durable part is the changer side.
4. **Characteristics:** CMN controls for size, value, momentum and liquidity and finds a robust effect. Q1 firms are
   somewhat larger (mean market value $3.5B vs $2.5B). No profitability or sector decomposition was verified.
   **Sector neutrality is imposed in our design rather than assumed.**
5. **Options:** concentrated in *optionable* stocks (Cheng et al.), but within them the effect lives in small-cap and
   non-S&P 500 names. Nearly all S&P 500 stocks are optionable and heavily monitored, which is the segment where the
   effect is weakest.

**Anomaly vs implementable profit.**
- The record is a documented 1995–2014 anomaly in a broad universe.
- It has **no** net-of-cost, large-cap or post-publication confirmation.
- Expected value in our universe after decay is small (ceiling below).

## Implementation: chosen before any outcome

| | A. Long-only avoidance (**chosen**) | B. Long-short disclosure spread (diagnostic only) |
|---|---|---|
| Economic logic | Uses the persistent changer side. An S&P 500 holder can implement it as an overlay at almost no extra complexity | Needs the stable-reporter leg, which reverts quickly in event time |
| Expected turnover | Only the excluded set changes, as roughly a third of firms file each month: about 1–3× NAV/yr of active trading (to be measured, not assumed) | Two legs of about 20% each, turned over quarterly, plus a hedge |
| Data | The same in both | The same in both |
| Market exposure | Matched to the equal-weight benchmark; active beta about 0 by construction | Needs beta hedging |
| Capacity | High: S&P 500 names, underweights only | Short side in large caps is feasible, but borrow and recall risk |
| Operations | One monthly reweighting | Borrow, locate, financing |
| Expected size | Active ≈ ¼ × the spread between changers and the rest | Full spread, but more costs and risk |

**Choice: A, the long-only, sector-neutral avoidance overlay on an equal-weight S&P 500 benchmark.** B is computed
once as a declared diagnostic, together with the leg decomposition. This choice is fixed in `PREREG.md` before any
filing text or outcome is examined.

## Economic ceiling (stated before scoring)

- Cheng et al.'s S&P 500 spread is about 0.22–0.26%/month in 1996–2017 (VW, insignificant). Halved for
  post-publication decay, about 0.12%/month.
- Avoidance captures about a quarter of the changers-vs-rest spread, giving an active return of roughly **0.3–0.4%/yr
  before costs**.
- Tracking error of excluding 20% of an equal-weight S&P 500 is roughly 1–1.5%/yr, so a gross active IR of about
  0.2–0.3.
- Costs of about 0.1–0.3%/yr leave a **net active IR of roughly 0.1–0.2**. That is the realistic expectation.

**The 2016–2023 test cannot confirm such an edge.** Its decisive use is to stop the implementation cheaply, or to
license two free next steps: one scoring of the protected 2024–2026 window, and an implementation-only paper trade.

## Where the effect plausibly lives (not tested here)

Non-S&P 500, small and mid-cap optionable stocks: Cheng et al. find about 60 bp/month for small caps, and CMN's
equal-weight universe agrees. Our point-in-time data covers only S&P 500 members. A broader PIT universe would need
Alpaca's full symbol history including delisted small caps, a causal liquidity-ranked membership rule, and a borrow
and cost model for less liquid names. That is listed under H as INVESTIGATE, with its cost.
