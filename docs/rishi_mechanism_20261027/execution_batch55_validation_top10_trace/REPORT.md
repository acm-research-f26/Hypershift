# Ten-stock changing-score versus fixed-validation account attribution

For the ten-seed full-exposure accounts, the changing-score model's **+6.17 percentage-point annual arithmetic net difference** is distributed across many issuers. The largest positive contributions are Applied Materials **+1.48 pp**, AMD **+1.42 pp**, and Lam Research **+1.27 pp**; the largest negative contributions are Intuit **-1.13 pp**, Adobe **-0.99 pp**, and ASML **-0.93 pp**. All other issuers contribute **+5.04 pp**. These are paired accounting contributions using each account's own prior NAV, not single-stock removal effects.

| Seed group | Three largest positive contributions | Three largest negative contributions | All others | Total |
|---|---|---|---:|---:|
| Original five | AVGO +3.33; LRCX +2.80; MRNA +1.28 | AMD -2.41; ADBE -1.28; ZS -0.50 | +6.08 | +9.30 |
| New five | AMD +1.11; MELI +1.02; NVDA +0.89 | ADBE -1.66; Alphabet -1.23; DXCM -1.06 | -0.56 | -1.48 |
| Ten seeds | AMAT +1.48; AMD +1.42; LRCX +1.27 | INTU -1.13; ADBE -0.99; ASML -0.93 | +5.04 | +6.17 |

Values are rounded percentage points per year. Ten-seed annualized daily-mean differences are **+13.40, -2.02, +3.19 and +10.10 pp** across 2022–2025; these are not calendar CAGR differences. The account gross contribution difference is **+7.78 pp/year**, partly offset by **+1.61 pp** of modeled fee drag. Gross retains fee-affected holdings and NAV; it is not a zero-fee counterfactual. The full-period paired mean interval remains **[-2.81, +16.11] pp**, so the result does not establish superiority or equivalence. The source period is repeatedly inspected, seed groups overlap, and validation selected the checkpoints.

Protocol `87b54f5` and implementation `bda5df5` were committed before issuer outcomes. All six account daily paths exactly reproduce the saved paths; maximum normalized NAV error 1.40e-15, cash error $9.32e-10 and share error zero. An independent reconstruction verified six paths, three issuer tables, twelve calendar partitions and **3,577** changed target rows, maximum error 4.58e-16. All securities and target differences are retained. These are reconstructed economic fills, not broker records; no neural inference, fitting, new account policy or protected scoring occurred. Existing gates remain unchanged.
