# Codex.md: live handoff for Codex CLI

Last updated: 2026-10-02 (Claude Code session). Claude keeps this file current at every milestone; if it is stale, trust `git log` and the RESUME HERE section linked below.

## Read in this order

1. `CLAUDE.md`: environment, commands, architecture, pitfalls. It is written for Claude but applies to you too.
2. `docs/phase1_5/PHASE1_5_SUMMARY.md`: where the study stands now.
3. `docs/phase1_5/F_learnability.md`, section **RESUME HERE**: the exact fetch, merge and analysis commands for the running Kaggle jobs.
4. `docs/PHASE1_TRACKER.md`: the full record (R/A/G/C/PA/U entries, verdicts).
5. `docs/HANDOFF.md` is **stale** (2026-09-28). Use it for background only.

## Where we are

- **Phase 1** (reproduce THINK, ICDM 2022) is 26/26 done. The verdict was "not reproduced", but it is now **superseded**: every Phase 1 run used coupled `weight_decay=5e-4`, which collapses the model (Phase 1.5 F).
- **Phase 1.5** (fidelity audit) has audits A–F done. The fix is `weight_decay=0` plus `input_mode=relative`. The final verdict waits on the two Kaggle reruns below.
- **Phase 2** (diagnose why) has not started.

## Running now (Kaggle, user `tomphamdustry`)

| Kernel | What | Output exps |
|---|---|---|
| `hypershift-run-r5f` (preset 10) | Full-NYSE THINK (HH) and EH, seeds 0-9; EE, seeds 0-4. Fix applied. | `R5_f_paper`, `R5_f_train` |
| `hypershift-run-r8f` (preset 11) | RSR-I and STHGCN baselines, seeds 0-4. Fix applied. | `R8_f_paper`, `R8_f_train` |

Check status:

```bash
KAGGLE_USER=tomphamdustry kaggle/.venv-kaggle/Scripts/python.exe -m kaggle.cli kernels status tomphamdustry/hypershift-run-r5f
```

## Next steps

1. When r5f is COMPLETE:
   - `bash kaggle/fetch.sh r5f`, then merge. See RESUME HERE for the PermissionError workaround.
   - Analyse THINK vs EH under both norms, using the R5_g2 stats: paired Wilcoxon, Holm, block bootstrap; leak-free and best-test Sharpe; NDCG@5; IC.
   - Write `docs/phase1_5/F_r5f_results.md`.
   - Fill the PENDING section 5 of `PHASE1_5_SUMMARY.md` and update the tracker verdict.
2. Do the same for r8f: the baselines vs THINK.
3. Open questions:
   - Rerun the small-scale Phase 1 arms with the fix?
   - Test `alpha=0` and `spatial_residual` on full NYSE?
4. Then Phase 2.

## Rules (non-negotiable)

- **Paper source of truth is `docs/paper/icdm22-think.pdf`** (pp. 849–854). Cite the page and section, equation or table.
  - Anything the PDF doesn't state is labelled `INFERRED (not in paper)` or `UNKNOWN`. Never guess.
  - If something is unreadable, ask the user.
- **Heavy GPU runs go on Kaggle** (`kaggle/README.md`, presets in `kaggle/run_kaggle.py`). The laptop is CPU-only for analysis: prefix Python with `CUDA_VISIBLE_DEVICES=-1`.
- Before any Kaggle launch, smoke-test 1 epoch, using a preset ending in `s`.
- Never write into an existing `results/<exp>/` folder. Use new exp names.
- Commit on branch `tom-shlom`, only the files you changed. Pushing is OK after commits; the user wants GitHub kept current.
- Windows + Git Bash:
  - Call `.venv/Scripts/python.exe` directly and never activate the venv.
  - Smart App Control sometimes blocks DLLs; retry the command.
- Tests: `.venv/Scripts/python.exe -m pytest -q` (~1 min, 292 pass).
- **Update this file** (Running now, Next steps) when you finish a step, so Claude can pick up again.
