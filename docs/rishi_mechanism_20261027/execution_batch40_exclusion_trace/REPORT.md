# What changed in the training-selected issuer screen?

The ten-seed top-ten blend's **three-issuer screen did not preserve the original winners unchanged**. Relative to the original account, NVIDIA's contribution fell by **2.98 percentage points per year** and Tesla's by **0.53 points**, while changes across other holdings offset those reductions. The overall screened-minus-original annual arithmetic net difference was **+0.75 points**, with the earlier 95% block-21 interval **[-5.44, +7.03] points**. The interval does not establish an improvement.

| E top-ten screen minus original: largest observed issuer changes | Annual arithmetic net contribution |
|---|---:|
| Microchip Technology | +0.95 pp |
| Moderna | +0.87 pp |
| Broadcom | +0.87 pp |
| NVIDIA | -2.98 pp |
| AMD | -1.05 pp |
| Tesla | -0.53 pp |
| All other issuers combined | +2.63 pp |

These issuer rows compare the two accounts' actual marked economic contributions, each normalized by its own preceding NAV. They combine changes in holdings, return paths, costs and cash; they are not causal effects of the issuers or estimates of a retraded single-stock deletion. The training-selected issuer set changes by fold, so an issuer can be excluded in some periods and available in others. The source execution rules can also delay exits. No statement that the network was recomputed without those peers follows.

As an example of **target changes**, Intuit's target weight increased by an average of **1.69 percentage points per decision** across all 201 origins; AMD, Amazon, Broadcom and Alphabet were the next four largest mean increases. Target weights describe desired allocations, not executed trades or realized P&L. AMD's observed contribution change was negative despite its increased target weight. Complete per-origin target changes and all issuer/security contribution and exposure changes are retained, including offsetting and zero entries.

The E screen-minus-original annualized arithmetic differences by calendar year were **+9.91, -9.14, +4.94 and -2.77 points** in 2022–2025. These are annualized daily means for subperiods, not additive calendar CAGRs; their changing sign reinforces the uncertainty. The results are post-outcome descriptions of repeatedly examined development history, not a reason to choose the historical screen. Issuer mapping, membership vintage, provider publication timing and source execution remain imperfectly certified.

Protocol a75de66 and code d1e9607 fixed all six original/exclude-one/exclude-three E and E_Hlabel top-ten traces before this attribution. All six daily paths match saved returns exactly; maximum normalized NAV error is 2.33e-15, cash error $1.93e-9 and share error zero. An independent reconstruction verified all six traces, all four issuer comparisons and **2,276** changed target-weight rows with maximum error 4.79e-16; five trace/engine tests passed. Detailed outputs and hashes are in this folder, with daily arrays in ignored `results/RISHI_TRAINING_EXCLUSION_TRACE_20261028/`. No new fit, model intervention, protected scoring, capital or research-gate change occurred.
