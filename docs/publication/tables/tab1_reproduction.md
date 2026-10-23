# Table 1. THINK reproduction, NYSE 2017 (source: metrics.json)

| Setup | Arm | Seeds | Validation-selected Sharpe (mean ± SD) | Test-selected Sharpe (mean, diagnostic) |
|---|---|---|---|---|
| Original setup (wd 5e-4, level, full-series norm) | THINK | 25 | 0.089 ± 0.306 | 2.115 |
| Original setup (wd 5e-4, level, full-series norm) | EH | 25 | 0.383 ± 0.774 | 2.158 |
| Corrected (wd 0, relative, train-period norm) | THINK | 10 | 1.687 ± 0.333 | 2.925 |
| Corrected (wd 0, relative, train-period norm) | EH | 10 | 1.436 ± 0.592 | 2.739 |
| Corrected (wd 0, relative, full-series norm) | THINK | 10 | 1.929 ± 0.595 | 2.891 |
| Corrected (wd 0, relative, full-series norm) | EH | 10 | 1.146 ± 0.697 | 2.757 |

Hold-all equal-weight Sharpe 1.531 (recomputed, `CLAIMS_VERIFICATION.md`). Test-selected values select on the test year (diagnostic only).
