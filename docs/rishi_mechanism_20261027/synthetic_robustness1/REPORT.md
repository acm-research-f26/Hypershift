# Missingness and membership learned-recovery results

October 27, 2026. Frozen protocol `bc9d646`. **24 fits completed**, fixed800updates each, in **146.66seconds**; no recipe changes or financial data. The initial20-update resource projection was150.72seconds. Combined verification after the export-checker fixes: **45 tests passed** (`TEST_LOG_FINAL.txt`).

## Results across seeds7/19/42

| Synthetic case | Full pinned dynamic adapter R² | Temporal-only path R² | Frozen screen |
|---|---:|---:|---|
| Own signal, increased nuisance missingness | .972–.980 | .977–.988 | Both recover |
| Coverage-only null, latent signal removed | -.0065 to -.0024 | -.0152 to -.0030 | Neither recovers unavailable information |
| Own signal, changing membership | .955–.976 | .964–.975 | Both recover |
| Peer signal, changing membership | **.629–.813** | -.0576 to -.0316 | Full recovers; restricted path does not |

R² is held-out improvement over a zero predictor, not a financial performance statistic. Every intended recovery arm exceeds .5 for all three seeds; no impossible-signal arm exceeds the .2 leakage-warning threshold. The peer recovery range is broader than in the earlier stationary-membership toy, but the generators differ, so this is not a paired causal estimate of the cost of membership change.

## What was actually varied

The own scalar remained visible in feature0 except in the explicit null. Missingness affected nuisance channels5–23, with channel24 indicating availability. Realized missing rates were25.9%→55.9% for the observable case and24.3%→57.6% for the null. This shows the architecture can preserve an obvious signal through this particular nuisance shift; it does not certify robustness to real vendor coverage, information-bearing missingness or missing the signal itself.

Membership cases use6–8active stocks in training and4–6in held-out dates. IDs0/1 always remain active to ensure one complete pair. All slots are existing fixed IDs; this is **not unseen-ID cold-start testing**. The supplied static wavelet pair graph already encodes the peer relation. Full-model success does not identify learned dynamic incidence as the useful branch.

Training/held-out supervision and inference coverage:

| Case | Train supervised / eligible | Held-out supervised / eligible |
|---|---:|---:|
| Observable missingness | 1024 / 1024 | 512 / 512 |
| Coverage-only null | 1024 / 1024 | 512 / 512 |
| Membership own | 893 / 893 | 320 / 320 |
| Membership peer | 764 / 883 | 212 / 318 |

Peer inference retains all current eligible nodes. Supervision only uses nodes whose paired peer is also eligible; this mask is determined by current availability, not future returns. `GENERATOR_AUDIT.json` records counts/ranges and applicability of each check.

## Masking evidence

Each trained membership fit produced identical held-out predictions after inactive inputs changed from NaN to large finite garbage. The two all-active cases also record the equality flag, but that assertion is vacuous there; treat it as not applicable, not evidence of inactive robustness.

An additional structural test on the generated shifted-membership mask checks both actual paths: input gradients are finite, exactly zero on inactive inputs and nonzero on active inputs. That test passed. It uses fresh initialized models; the runner does not claim per-fit trained-gradient verification. Existing tests also cover label-mask separation and unsupervised NaNs in metrics.

## Research implication and limits

The actual imported model has now shown learnability under simple known peer relations, fixed identity preferences, shifted nuisance coverage and changing eligibility. The controls do not manufacture information after removing the planted signal. This strengthens the feasibility of interpreting later controlled financial ablations, but does not establish what Rishi's trained models learned, improve their historical results, or authorize an architecture search.

The unresolved central requirement remains original per-stock predictions/checkpoints/preprocessing and share-account records. These synthetic passes cannot substitute for identity-P&L attribution, blend boundary analysis, feature information tests, independent raw replay or future confirmation.

## Review and reproducibility

An independent read-only reviewer found no substantive generator/training defect and checked the scientific limits above. Protocol/code hashes and output hashes are in `MANIFEST.json`; raw fit records and `RESULTS.json` retain every outcome. Per-update runtime checks preserve completed records and mark an unfinished fit if the cap is reached; evaluation can extend slightly beyond the check. The cap did not bind. Partial progress metadata would not be a resumable checkpoint.

The related Phase-A integrity checker was separately reviewed. Two issues were reproduced and corrected: equivalent path aliases now collide after normalization, and receipt creation uses exclusive file creation to prevent overwrites even if an earlier existence check is stale. Tests failed before both fixes, then the entire45-test suite passed. Existing numerical evidence and the earlier missing-file receipt were not overwritten.

New checker usage after receiving an export retaining `runs/` and `data/` relative paths:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
.\.venv\Scripts\python.exe -m hypershift run rishi_artifact_verify --root "C:/path/to/export" --out "docs/rishi_mechanism_20261027/new_integrity_receipt.json"
```

Exit0 means every listed file hash matches, **not** scientific reproduction. Exit2 means missing or mismatched originals. The current receipt `../artifact_integrity_gate1.json` reports339missing,0matched. Output files are never overwritten.
