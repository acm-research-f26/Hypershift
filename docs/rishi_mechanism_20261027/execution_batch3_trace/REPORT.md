# Stock and issuer profit attribution

2026-10-27. Frozen protocole4af428. Ten original-export account cells: B,D,E,Hlabel,E_Hlabel ×top5/top10, ten seeds. Canonical accounting code is unchanged; Python line tracing records executed share changes, economic settlement flows, fees and daily positions. Files are reconstructed records under the source assumptions, not broker records. Unfilled/cancelled instructions are not a complete submitted-order blotter.

## Reconciliation

All ten accounts match preceding saved-export daily returns within1e-16. Share reconciliation error is exactly0; cash error≤9.10e-12dollars. Daily summed stock net contributions match account returns within1e-10; stock fees match daily account fees; all terminal positions are zero. Tests also cover terminal liquidation, acquisition settlement and cancellation of unavailable fills. These establish internal accounting consistency, not independent execution-policy validation.

STOCK_CONTRIBUTIONS.csv and ISSUER_CONTRIBUTIONS.csv provide additive annual arithmetic contributions:252×daily mean net contribution. Larger daily arrays and fill records live in results/RISHI_MECHANISM_TRACE_20261027. Dollar P&L includes changing capital; neither its fraction nor additive arithmetic contribution is a fraction of CAGR.

## Concentration of E's historical profit

| Issuer | Top5 annual contribution | Top10 annual contribution |
|---|---:|---:|
| NVIDIA |11.91pp|5.97pp|
| Alphabet (GOOG+GOOGL) |5.70pp|4.84pp|
| Broadcom |4.06pp|4.00pp|
| Tesla |3.21pp|See complete table|
| AMD |See complete table|3.14pp|
| Total account arithmetic mean/year |33.46%|26.16%|

Top3issuers contribute64.74%of total net arithmetic return in top5 and56.62%in top10. Top5issuers contribute79.81%and74.30%, respectively. These are fractions of net contributions, with other names' gains/losses included in the denominator, not gross-profit concentration statistics. NVIDIA is held1002of1003daily sessions in both cells; the remaining session is terminal liquidation. A strategy that continually owns a historical winner can earn high returns without successfully timing that winner. This establishes persistent exposure, not how the neural network learned to select it.

## What differs from the identity substitute

E minus E_Hlabel contributes+5.69pp/year arithmetic in top5 but−.42pp in the prespecified top10 primary cell. Contributions are mixed across stocks. In top10, relative contributions include MRNA+1.83pp, combined Alphabet+3.22pp, MSFT+1.12pp; offsets include KLAC−3.40pp,LRCX−2.41pp,TEAM−2.01pp and ASML−1.92pp. These are descriptive account differences, not causal feature effects or hypothetical profits from deleting stocks. No named-stock removal strategy was tested.

Persistent stock exposure and concentration in a few growth/semiconductor winners account for substantial realized returns. Holdings alone do not isolate learned stable identity preference from repeatedly favorable changing signals. Whether changing AH/PM features or learned relations add useful information beyond persistent preferences remains open: Hlabel equivalence is unproven, checkpoint training-prediction controls are gated, and no trained feature/relation intervention has yet isolated their increment. Risk-overlay gains must still be compared on matched simple and identity-controlled books.

Independent read-only review found no accounting bug and prompted the narrower exposure interpretation above. AXES.json accompanies large daily arrays with ordered dates/security IDs; issuer aggregation uses the source ndx_drawdown_engine.ISSUERS mapping, now preserved in the trace entry point. These metadata additions do not rescore accounts.
