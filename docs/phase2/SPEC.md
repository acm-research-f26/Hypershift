# Phase 2: exhaustiveness closure ("can THINK's paper results be recreated at all?")

Written 2026-10-05 by the Claude Code orchestrator, before any new runs. The user's goal: be exhaustive; establish whether there is **any** reasonable route to the THINK paper's claims (Table II p. 852: NYSE THINK SR 1.18, TCONV+DHHAN 1.14, STHGCN 1.10; NDCG 0.86/0.81/0.78), and which routes fail.

## Workstreams

**W1. RSR rung closure (CPU).** Compare the authors' own RSR-I run (Phase 1.5 E, `docs/phase1_5/E_rsr_original_scores.json`, the kernel logs) with the **RSR paper's** reported NYSE numbers (Feng et al. 2019, TOIS, arXiv 1809.09441: MSE, MRR, IRR). Record the exact table values with a page citation, the metric definitions, and whether their code's output lands in the paper's range under the paper's own metric. Pass/fail is decided before computing: "reproduces" means the 5-seed mean IRR is within the 5-seed range of the paper's value, or within ±1 seed SD.

**W2. STHGCN official code (GPU, Kaggle).** Run `midas-research/sthgcn-icdm` (the THINK authors' earlier hypergraph model) on its own data (HATS preprocessing, `dmis-lab/hats`), with minimal porting to the Kaggle Python/PyTorch/PyG versions. Every port change is listed. Compare with the numbers its paper reports (cite page/table). This answers whether a THINK-family method reproduces on its own benchmark. Budget: ≤ 6 GPU-h. If the data or code cannot be obtained or run, record the exact blocker (that is a valid finding).

**W3. Protocol matrix on existing predictions (CPU, no training).** For each corrected model (R5_f HH/EH/EE, R5_f2 alpha0, R8_f RSR-I/STHGCN, the RSR original, WF), rescore the saved predictions under every documented reading of the paper's protocol:
- epoch selection: validation vs best-test (oracle), using per-epoch predictions where saved, else `history.jsonl`
- normalisation: train vs paper (leaky)
- Sharpe formula: ours (×√252, rf 0) vs the paper's printed `E[R_a − R_f]/std[R_a − R_f]` without annualisation, with R_f = 0 and with a 2017 T-bill daily rate (INFERRED value, cited source)
- k ∈ {1, 5, 10}
- NDCG: correct vs the STHAN-SR buggy evaluator

Output: a matrix of which protocol combinations reproduce (a) the paper's **values** and (b) the paper's **ordering** THINK > TCONV+DHHAN > STHGCN. Every combination is reported; none is cherry-picked.

**W4. Closure report (orchestrator writes).** `docs/phase2/REPORT_CLOSURE.md`: every route tried (Phase 1 to Phase 2), what reproduces and under which protocol, what does not, remaining UNKNOWNs (unpublished THINK code, the authors' selection rule, the Sharpe definition), and the final statement. Headline numbers are re-checked independently by the orchestrator.

## Rules

- CPU first. Never write into existing `results/` folders. Predeclare before computing.
- Paper claims are cited from `docs/paper/icdm22-think.pdf` (or the cited paper's PDF, with page) or labelled INFERRED/UNKNOWN.
- Verdict words per CLAUDE.md.
