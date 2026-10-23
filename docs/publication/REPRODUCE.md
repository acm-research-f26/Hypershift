# Reproducibility instructions

## Environment (recorded 2026-10-22)

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200), Git Bash |
| Python | 3.12.10 (venv `.venv`) |
| Key packages | numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, torch 2.14.0+cu130, pandas 3.0.6, matplotlib 3.11.2 |
| Full lock | `docs/publication/requirements-frozen.txt` (`pip freeze`) |
| CPU | Intel 64 Family 6 Model 154 (16 logical cores), 16 GB RAM |
| GPU (THINK laptop runs) | NVIDIA GeForce RTX 3050 Laptop, 4 GB, driver 592.82 |
| Other THINK runs | Kaggle T4 GPUs (platform per run in `docs/PHASE1_TRACKER.md`) |
| Repository state | branch `tom-shlom`; the commit used for this package is recorded in `PUBLICATION_LEDGER.md` |

**Determinism:**
- LightGBM uses `deterministic=True`; refits reproduced saved series to 3e-9.
- THINK GPU training is not bitwise deterministic: same-seed repeats differ by up to about ±0.6 Sharpe (`docs/phase2/REPORT_CLOSURE.md`).

## Clean-clone procedure

This procedure has **not yet been executed from a clean clone**; that is a listed readiness item.

```bash
git clone <repo> hypershift && cd hypershift
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r docs/publication/requirements-frozen.txt
.venv/Scripts/python.exe -m pip install -e .[featbench]
.venv/Scripts/python.exe -m pytest -q                      # 469 tests at this revision
bash scripts/download_rsr.sh                               # RSR data (AGPL-3.0 repository)
.venv/Scripts/python.exe scripts/fetch_fresh.py --source yf --kind daily --start 2015-01-01 --end 2026-09-01 --name sp500_daily
.venv/Scripts/python.exe scripts/fbaudit_dividends.py fetch     # unadjusted closes (yfinance)
```

- **Fetched values differ from ours.** yfinance back-adjusted values depend on the fetch date (manuscript §6). Expect adjusted levels, and the adjustment-term results, to differ slightly. Return-based results should be stable up to vendor revisions.
- **Re-training is not needed for the published numbers.** Saved per-run outputs (`results/…`) are what the verification and figure scripts read.
- **Some outputs are not in git.** These are the FEATBENCH and FBAUDIT results, the matched runs and the audit downloads. They must either be released as derived data (see licensing) or regenerated:
  - `scripts/featbench.py`;
  - `scripts/fbaudit_refit.py run main|corrected`;
  - `scripts/closure_matched.py run`.

## Figure and table commands (saved outputs only)

| Output | Command | Inputs |
|---|---|---|
| Fig 1, Tab 1 (reproduction) | `python scripts/pubfig.py` | `results/R5_g2`, `results/R5_f_*` |
| Fig 2, Tab 2 (planted signal) | same | `results/known_signal_analysis.json`, `results/f_f2_c1_high_relative` |
| Fig 3, Tab 3 (adjustment term) | `python scripts/pub_leak_mechanism.py && python scripts/pubfig.py` | `data/audit/yf_unadjusted/unadjusted.npz`, FB labels |
| Fig 4, Tab 4 (matched 2×2) | `python scripts/pubfig.py` | `results/FBAUDIT_20261022/matched`, FB labels |
| Fig 5, Tab 5 (power) | same | FB `daily.npz` |
| Claims verification | `python scripts/pub_verify.py` | as listed in the output |

## Not regenerable from saved outputs alone

These need deterministic re-execution of existing scripts on the same inputs. That is re-execution, not a new experiment, and it has not been done in this revision.

| Result | Why | Command |
|---|---|---|
| Economics tables (A8, A13): net Sharpe, turnover | Daily book returns were not saved | `scripts/bcreview_cost.py`, `scripts/closure_econ.py` |
| Synthetic validation (A12) and learner probe | Synthetic arrays were not saved; the probe was run inline | `scripts/closure_synth.py`, `scripts/closure_synth_perm.py` (the probe needs to be scripted) |
| Collapse probe (layer norms) | Diagnostic run, not saved | `scripts/f_diag.py` |
| Protocol matrix (88 cells), 2017 nulls, post-2017 tests | Kaggle-era outputs; only report tables are local | Phase 2 / 1.5a / 1.5b scripts listed in those reports |
| THINK full-NYSE training | GPU training, non-deterministic | `python -m hypershift.run …` (see `CLAUDE.md`) |

## Release audit (what may be published)

| Content | In git? | Licence | Action |
|---|---|---|---|
| Source code, tests, docs | yes | project | release |
| `results/**/test_{gt,pred,daily,mask}.npy` (270 `test_gt` files): daily returns derived from RSR data | **yes** | RSR repository is **AGPL-3.0** | Confirm AGPL compliance (attribution and licence notice), or remove `test_gt.npy` from the release and provide a regeneration script |
| yfinance prices (adjusted and unadjusted) | no (`data/` git-ignored) | Yahoo terms restrict redistribution | Do not release; release the fetch scripts |
| Alpaca bars | no | Alpaca terms | Do not release |
| S&P 500 membership (fja05680/sp500) | no | MIT | Link, or redistribute with notice |
| THINK PDF (`docs/paper/icdm22-think.pdf`) | **yes** | IEEE copyright | **Remove from any public release** |
| Derived daily metric series and predictions (FB/FBAUDIT) | no | derived from Yahoo data | Release daily IC series (aggregate statistics); check vendor terms before releasing per-stock predictions |
