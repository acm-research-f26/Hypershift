# Which stocks explain the parent-view blend differences?

The small historical difference between the AH+PM-parent blend and the daily-parent blend reflects **offsetting contributions across a few changing holdings**, rather than a simple fee difference. For the ten-seed top-ten accounts, AH+PM minus daily was **−1.59 percentage points/year** in arithmetic net return; the earlier paired 95% interval was **[−4.95, +1.47] points**. Gross contributions differed by **−1.66 points**, partly offset by **0.07 points lower fee drag** in the AH+PM account. This is accounting on the actual simulated paths, not a causal feature decomposition.

The largest observed positive and negative issuer contributions to each fixed top-ten comparison are below. Each number is the difference of the issuers' marked net P&L divided by each account's own preceding NAV, annualized over the full path. The issuer rankings are post-hoc; they are not separate significance tests or estimates from removing an issuer.

| AH+PM-parent blend minus daily-parent blend | Contribution |
|---|---:|
| NetEase | +1.18 pp/year |
| Lam Research | +0.52 pp/year |
| Microsoft | +0.43 pp/year |
| Alphabet | -0.99 pp/year |
| KLA | -0.89 pp/year |
| PDD | -0.54 pp/year |
| All other issuers | -1.30 pp/year |

| AH+PM-parent blend minus AH-only-parent blend | Contribution |
|---|---:|
| NetEase | +0.49 pp/year |
| PDD | +0.30 pp/year |
| Apple | +0.29 pp/year |
| Alphabet | -0.64 pp/year |
| Broadcom | -0.55 pp/year |
| MercadoLibre | -0.42 pp/year |
| All other issuers | +0.30 pp/year |

The AH+PM-minus-daily annualized arithmetic differences for 2022–2025 were **−5.47, +0.07, −2.35 and +1.41 points**, respectively; these are annualized daily means, not additive calendar CAGRs. The source 95% interval for the AH+PM-minus-AH comparison is **[−2.33, +1.78] points**. Neither interval establishes equivalence or a reliable improvement.

Protocol 00c4eb6 and code 51cfa6f fixed all six ten-seed parent-view/top-k traces after the account comparison. The two parent forecasts are separately trained packages, while the dynamic family uses AH/PM in every blend. Both accounts typically hold nearly the same names, but changes at the hard top-k boundary alter specific target weights and subsequent fills. `TARGET_CHANGES.csv` records desired weight differences; they are not trades or a causal isolation of AH/PM price channels. The selected historical period, source issuer mapping, membership, data publication and reconstructed economic-fill limitations remain.

All six account returns exactly match the earlier saved paths; maximum normalized NAV error is 1.40e-15, cash error $1.06e-9 and share error zero. Independent reconstruction checked all six traces, four issuer comparisons and **717** changed target rows with maximum error 4.46e-16; five focused trace/engine tests passed. Full security, issuer, fee, exposure, target and calendar tables and input hashes are in this folder. No new financial fitting, trained-checkpoint intervention, protected-window scoring or gate change occurred.
