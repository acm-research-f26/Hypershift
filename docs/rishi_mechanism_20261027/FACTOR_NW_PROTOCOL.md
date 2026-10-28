# Supportive exposure/spanning regressions — 2026-10-27

Research_Plan §10 supportive Newey–West statistics; no new predictive/trading model or financial-training gate. Freeze seven descriptive regressions on the same1003saved-account sessions, each with intercept and one predictor:

1. E ten-seed/top10 net return minus daily3%hurdle on raw economic QQQ return minus hurdle.
2. E_Hlabel ten-seed/top10 excess on the same QQQ excess.
3. E−B ten-seed/top10 active return on QQQ excess; no hurdle subtracted from an active difference.
4. E−E_Hlabel ten-seed/top10 active return on QQQ excess.
5. E ten-seed/equal5/vol20 excess on its equal5/full excess.
6. E ten-seed/issuer5/vol20 excess on its issuer5/full excess.
7. QQQ matched weekly/vol20 excess on its weekly/full excess.

OLS coefficients with HAC Bartlett covariance, primary lag5and lag20sensitivity. Sandwich uses sum of score outer products plus weighted symmetrized lag products, multiplied by n/(n−p). Reject rank-deficient design; report coefficient/SE/normal95%interval/two-sidednormalp, R² and n. Annual intercept=252×daily intercept is arithmetic attribution, not CAGR or a tradable alpha claim. No p-value promotion or family replacement; no cherry-picked factor/lag. QQQ is the sole supplied factor, not comprehensive sector/style adjustment. Cash remains zero-interest; the3%hurdle is unchanged.

Use fresh execution_batch13_factor. Tests include hand-computed intercept-only HAC at lag0/1, exact linear slope recovery and singular-design rejection. Match account/QQQ calendars and record consumed-file/code/protocol hashes. Source risk rule differs from inverse-variance academic recipes; a selected-history spanning intercept does not certify useful timing or learned relations. No new positions, protected scoring, prospective activation or capital.
