# Retention strategy replication in feat-rs — October 29, 2026

The pinned tom-shlom result was reproduced from feat-rs saved forecasts and reviewed market inputs. All 96 source accounts, targets, daily paths, detailed ledgers and 384 calendar returns passed comparison. Twelve existing feat-rs daily-path baselines also reproduced. This is replication of selected, repeatedly inspected 2022–2025 development results, not prospective confirmation.

| Group | Selection | Hedge cap | Sharpe | CAGR | Max drawdown | Annual traded/NAV |
|---|---|---:|---:|---:|---:|---:|
| original_five | issuer | 50% | 1.454 | 27.79% | -14.56% | 31.41 |
| original_five | retention | 0% | 1.383 | 33.71% | -18.99% | 17.13 |
| original_five | retention | 50% | 1.724 | 32.97% | -12.24% | 21.07 |
| original_five | retention | 75% | 1.731 | 33.11% | -13.36% | 21.61 |
| new_five | issuer | 50% | 0.811 | 15.12% | -17.02% | 31.65 |
| new_five | retention | 0% | 0.597 | 13.98% | -17.19% | 19.22 |
| new_five | retention | 50% | 0.643 | 12.26% | -17.11% | 23.16 |
| new_five | retention | 75% | 0.648 | 12.31% | -17.11% | 23.69 |
| ten_seed | issuer | 50% | 1.325 | 24.83% | -14.78% | 26.84 |
| ten_seed | retention | 0% | 1.158 | 28.22% | -17.81% | 15.95 |
| ten_seed | retention | 50% | 1.374 | 26.25% | -13.69% | 19.97 |
| ten_seed | retention | 75% | 1.381 | 26.35% | -13.79% | 20.62 |

![Matched retention comparisons](retention_comparison.png)

## Exact strategy

Each rebalance adds 0.02 to the percentile-derived blend score of every security with positive weight in the previous target. The five highest adjusted scores receive equal weights, with fractional boundary ties and current eligibility enforced. State continues across annual fits. Retention uses previous targets, not actual filled holdings. Share classes occupy separate security slots; this is not issuer-five selection.

The unchanged 50:50 THINK+mixer AH+PM / restricted dynamic-hypergraph rank blend supplies the scores. The selected basket receives a 20-session/20% volatility target. The always-on QQQ short is min(cap, unused gross budget, 20%/QQQ trailing volatility). A 75% cap is a ceiling, not a constant 75% short. Decisions are every five exchange sessions at 09:20 ET after the 09:15 feature cutoff; share sizing and risk use prior-close data and fills use the regular close.

## Costs and interpretation

The original-five retention/75% account at 6% borrowing and 2× trading costs has SR 1.570, CAGR 29.85% and drawdown -15.15%. All 96 accounts are retained, including weaker seed groups and unhedged controls. The grid is the source study’s exact 3 groups × 2 selection rules × 4 caps × 2 borrow rates × 2 cost multipliers; no new tuning was performed. Borrow is charged on short value by calendar day; cash/rebate is zero. Costs, liquidity, settlement and dividend treatment remain inherited simulation assumptions. Broker locates/recalls, full impact and margin enforcement are not established.

The original-five ensemble is substantially stronger than the new-five half. The ten-seed ensemble overlaps both halves. Retention changes holdings, subsequent basket risk and exposure as well as turnover; this replay alone does not attribute the improvement solely to saved transaction costs. The published outcomes were known before this replication protocol was frozen. No untouched-period superiority or trained market-timing claim follows.

## Verification and provenance

Source commit: `fca66ed0b577444928443847a6ec7f7fe21b102a` on `tom-shlom`. Maximum source daily-return error: 0; maximum independent dollar-position replay error: 8.88e-16; maximum source target error: 0; maximum ledger dollar discrepancy: $4.66e-10. All source archive members used were checked against the committed SHA-256 manifest. Existing local forecasts were checked against their publication seals, and inputs/code were hashed before replay. This verifies saved forecasts and accounting, not numerical regeneration of trained checkpoints or independent vendor/broker correctness.

[All metrics](METRICS.csv) · [Account checks](ACCOUNT_CHECKS.csv) · [Audit](AUDIT.json) · [Protocol and input hashes](PROTOCOL.json) · [Baseline parity](BASELINE_PARITY.csv) · [Annual returns](CALENDAR.csv)

## Reproduce

From the repository root:

```powershell
git fetch origin tom-shlom:refs/remotes/origin/tom-shlom
.\.venv\Scripts\python.exe -m hypershift run run_ndx_retention_replication --stage source
.\.venv\Scripts\python.exe -m hypershift run run_ndx_retention_replication --stage freeze --name ndx_retention_replication_20261029_fresh
.\.venv\Scripts\python.exe -m hypershift run run_ndx_retention_replication --stage run --name ndx_retention_replication_20261029_fresh
.\.venv\Scripts\python.exe -m hypershift run run_ndx_retention_replication --stage report --name ndx_retention_replication_20261029_fresh
```

Original run outputs refuse overwrite. Full daily paths, targets and ledgers are under `runs/ndx_retention_replication_20261029/`. No model was retrained, 2026 returns scored, capital deployed or prospective policy changed.
