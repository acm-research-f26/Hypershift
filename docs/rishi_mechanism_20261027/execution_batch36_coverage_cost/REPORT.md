# Coverage-control gap under fixed cost scenarios

The blend's historical advantage over the matched linear coverage control remains large in the **zero-fee simulation**. With fees set to zero and the share accounts rerun, the blend's CAGR was **26.20%** versus **4.53%** for the control; their annual arithmetic net-return difference was **+19.38 percentage points**, with a block-21 interval of **[+7.71, +31.49] points**. Zero fees are an optimistic diagnostic, not a claim about achievable trading costs.

Protocol 2562d4a and code 8dedb43 fixed four cost views after the earlier comparison. Each view reuses the same original or coverage-control target selections but recalculates share quantities, cash and fees under the source execution engine. The table shows the ten-seed, top-ten E comparison:

| Cost view | E CAGR / Sharpe | Matched coverage control CAGR / Sharpe | Annual arithmetic difference | 95% block-21 interval |
|---|---:|---:|---:|---:|
| Zero | 26.20% / 0.814 | 4.53% / 0.195 | +19.38 pp | [+7.71, +31.49] pp |
| Modeled 1× | 23.92% / 0.755 | -0.11% / 0.038 | +22.10 pp | [+10.51, +34.26] pp |
| Modeled 2× | 21.69% / 0.695 | -4.53% / -0.117 | +24.80 pp | [+13.22, +37.00] pp |
| Modeled 4× | 17.37% / 0.577 | -12.75% / -0.425 | +30.16 pp | [+18.59, +42.38] pp |

The widening gap as assumed fees rise is consistent with the coverage control's much greater turnover; both books deteriorate as costs increase. The zero-fee gap shows that fees alone do not account for this particular historical comparison. It does not isolate the effect of a model feature: the books hold different stocks and have different realized risks, while the control inherits the blend's industry schedule. The selected 2022–2025 history and unadjusted pointwise intervals do not establish future alpha or a new primary-gate pass.

The separately matched E_Hlabel comparison also has a +17.90 pp zero-fee annual arithmetic difference, block-21 interval [+3.91, +31.98] pp. All **16** fixed accounts, **24** paired block-5/21/63 intervals and calendar returns are in `METRICS.csv`, `CONTRASTS.csv` and `CALENDAR.csv`; ledgers and paths are in the ignored `results/RISHI_COVERAGE_COST_20261028/` directory. Original 1× paths and NAV passed the replay gate before other scenarios. An independent calculation verified all 16 metric/ledger paths and 24 paired mean intervals; maximum metric error 3.62e-14, normalized NAV error 6.52e-15, interval error 9.72e-17. Twelve source risk and execution tests passed. The initial independent verification script had a pandas column-name collision and was corrected before verification; no account was changed.

SEC classification, price identity, provider arrival, real auction liquidity and broker execution remain uncertified. No model fitting, trained-checkpoint intervention, protected S&P scoring, capital, or change to the original mechanism gate occurred.
