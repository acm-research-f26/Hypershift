# Historical cash-yield accounting sensitivity

The previously reported accounts assumed cash earned zero. A separate retrospective sidecar credits the previous close's saved cash balance at the repository's historical Fama–French risk-free rate, with accumulated interest kept outside the trading account. On the ten-seed issuer-limited volatility-targeted book, this raises the *accounting-view* Sharpe from 1.112 to 1.161 and CAGR from 26.81% to 27.76%. It does not revise the original result or establish that those cash returns were actually available to the strategy.

| Ten-seed blend book | Original Sharpe | Sidecar Sharpe | Original CAGR | Sidecar CAGR | Original / sidecar max drawdown | Accumulated interest |
|---|---:|---:|---:|---:|---:|---:|
| Original top 5, full | 0.896 | 0.896 | 31.87% | 31.87% | -40.94% / -40.94% | $553 |
| Original top 5, 20% vol target | 1.080 | 1.128 | 26.00% | 26.94% | -21.69% / -21.48% | $75,045 |
| Original top 5, training-calibrated fixed | 0.894 | 0.941 | 23.09% | 24.19% | -28.65% / -28.50% | $82,565 |
| Issuer 5, full | 0.925 | 0.925 | 33.37% | 33.38% | -40.94% / -40.94% | $577 |
| Issuer 5, 20% vol target | 1.112 | 1.161 | 26.81% | 27.76% | -21.69% / -21.48% | $76,904 |
| Issuer 5, constant 75% | 0.893 | 0.925 | 25.43% | 26.23% | -31.73% / -31.62% | $62,886 |

The sidecar accumulates separately and never affects sizing, fills, fees or the original cash ledger. First-day interest is omitted, and later credits use the previous close cash plus accumulated sidecar. The RF series is the 202608 CRSP-vintage Fama–French daily risk-free proxy, not a point-in-time broker yield; rate precision, overnight timing, settlement, taxes and real cash access are not certified. The same unchanged 3% annual hurdle is used for both Sharpe views. Differences in CAGR are compounding effects, not a simple annual-interest subtraction. The sidecar cannot be spent or traded and therefore is not a fully reinvested strategy backtest.

The full `SUMMARY.csv` covers all 24 fixed account/policy tapes, including parent, dynamic, reversal, identity, equal-weight and QQQ controls. The QQQ vol-target Sharpe changes 0.581 to 0.594; REV5 changes 0.180 to 0.245. These are selected-history comparisons, not significance tests. `DAILY_LEDGER.csv` retains all 24,072 dated rows; input hashes and the zero-rate check are in `RESULTS.json`. Independently reconstructing the sidecar by a closed-form cumulative-product expression gave maximum balance error $7.57e-10 and terminal error $6.99e-10. Zero RF reproduced NAV exactly and daily returns within 6.20e-16; two focused tests passed.

Protocol 9438dc7, pre-run precision amendment 99f12a1, code 07cd057. The first attempt stopped without output because the issuer5/fixed source tape does not exist; before any result, amendment fde7aae substituted the existing issuer5/constant75 tape and preserved the empty failed directory. No new financial model fitting, protected-window scoring, changed trading policy or checkpoint gate change occurred.
