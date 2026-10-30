# Stock attribution of the matched historical-label comparison

Protocol ../INDUSTRY_SIMPLE_TRACE_PROTOCOL.md (ec530bc) and code ab267e9 were committed before inspecting these contributions. Four existing paths were reconstructed: E/top10 and E/industry-matched Hlabel, plus the analogous E_Hlabel pair. The source share-account engine and modeled costs are unchanged. Each stock's daily contribution is normalized by **its own account's prior NAV** before the two accounts are compared; all security differences sum back to the already-reported paired arithmetic mean.

For E minus its matched Hlabel control, the whole-period difference is **+4.52 percentage points/year**. The largest observed positive and negative issuer contributions are:

| Post-hoc issuer attribution | Annualized contribution to paired difference |
|---|---:|
| Alphabet (GOOG + GOOGL) | +4.84 pp |
| Zoom | +1.54 pp |
| Moderna | +1.24 pp |
| KLA | -3.51 pp |
| Take-Two | -1.36 pp |
| Synopsys | -1.13 pp |
| All other issuers combined | +2.90 pp |

Alphabet's contribution exceeds the total net difference because losses and other offsets partly cancel it. These are **post-hoc accounting contributions**, not estimates of what would happen if an issuer were removed and the portfolio retraded. They are not shares of CAGR or Sharpe and do not identify a causal feature or model mechanism. Issuer groupings use the original engine map; all names and share-class components remain in PAIRED_ISSUERS.csv and PAIRED_STOCKS.csv.

The paired annualized arithmetic differences by calendar year are **+22.13 pp in 2022, -13.72 pp in 2023, -0.36 pp in 2024, and +10.01 pp in 2025**. Annualized subperiod means cannot be added as calendar returns. Across the full period, E's gross arithmetic contribution is 27.99% versus 22.58% for the control, and modeled fee drag is 1.82% versus 0.94%; their respective net means are 26.16% and 21.64%. The comparisons mix different holdings, timing, exposures, turnover and fees. The primary paired interval from batch26 remains **[-5.03, +14.03] pp/year**, so neither superiority nor equivalence is established on this repeatedly inspected sample.

All four traced paths exactly reproduce the saved daily returns; maximum normalized NAV error is 1.40e-15 and maximum cash error $7.41e-10, with zero share error and terminal units. A separate calculation rebuilt four paths, stock gross/fee/net contributions and exposures, and issuer aggregation from the saved per-stock arrays. Independent review found no substantive arithmetic issue and identified four consumed-ledger hashes, now recorded in ADDITIONAL_INPUT_HASHES.json. Five focused tests passed. STOCKS.csv, ISSUERS.csv, PAIRED_STOCKS.csv, PAIRED_ISSUERS.csv and PAIRED_CALENDAR_STOCKS.csv contain all records; raw per-stock arrays remain under ignored `results/RISHI_INDUSTRY_SIMPLE_TRACE_20261028/`. The matched controls inherit model industry/position budgets; SIC timing, ticker/price identity, membership vintage and reconstructed-not-broker execution remain limitations. No model fitting, protected scoring, account change or checkpoint-gate amendment occurred.
