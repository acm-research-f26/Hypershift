# Which issuers account for the matched coverage-control gap?

Several issuers made large positive contributions to the blend's historical return difference over its matched coverage control, while others offset it. The three largest positive observed contributions were **Alphabet +4.71**, **AMD +3.58**, and **NVIDIA +3.21 percentage points per year**. The largest negative contributions were **Meta -2.54**, **Palo Alto Networks -1.54**, and **Marvell -1.36 points**. These are accounting contributions to the observed **+22.10-point annual arithmetic net-return difference**, not estimates of what would happen if those issuers were removed and the portfolios retraded.

| Post-hoc issuer contribution to E minus coverage6 | Percentage points/year |
|---|---:|
| Alphabet | +4.71 |
| AMD | +3.58 |
| NVIDIA | +3.21 |
| Meta | -2.54 |
| Palo Alto Networks | -1.54 |
| Marvell | -1.36 |
| All other issuers combined | +16.04 |

The four calendar-year annualized arithmetic differences are **+22.99, +15.74, +15.41 and +34.32 points** for 2022–2025, respectively. They are annualized daily means for subperiods, not additive calendar CAGRs. The original block-21 95% interval for the whole-period difference is **[+10.51, +34.26] points**; the issuer rankings were examined after the account result and have no separate significance claim.

Protocol 93f9487 and code 4170acf fixed this attribution before issuer contributions were computed. The four existing E/E_Hlabel original and matched coverage accounts were reconstructed from saved target weights with the source modeled-fee engine. Stock P&L and fees are normalized by each account's own preceding NAV; stock and issuer differences sum to their account-level paired arithmetic differences. The two selectors inherited matching industry budgets and position-size distributions, but their names, exposure, turnover, cash and fees differ. SIC/identity and provider-publication limitations persist, and source accounting uses reconstructed economic units rather than certified broker records. None of these figures establishes a neural feature or peer-information cause.

All four daily returns match the saved paths exactly; maximum normalized NAV error is 1.40e-15, cash reconciliation error $1.15e-9 and share error zero. Gross, net and fee totals match the earlier account-level decomposition. A separate verification rebuilt all four daily paths and stock/issuer summaries with maximum error 4.79e-16; five trace and accounting tests passed. Complete issuer, security, exposure, fee and calendar tables, hashes and reconstruction receipts are in this directory. No new strategy, model fitting, checkpoint intervention, protected-window scoring or gate change occurred.
