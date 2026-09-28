# THINK (Temporal Hypergraph Hyperbolic Network) Reproduction + Ablation Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement THINK (Agarwal, Sawhney et al., ICDM 2022) from scratch. Reproduce its stock-ranking results on NYSE/NASDAQ (Sharpe ratio, NDCG, IRR) and the paper's ablations. Then answer seven questions with statistically defensible experiments:
1. How are stocks grouped?
2. Does hyperbolic space help?
3. Do hyperedges help?
4. Do hyperedges and hyperbolic space help together (interaction)?
5. Does the number of stocks matter?
6. Daily data: does the result hold out-of-period?
7. Hourly data: does hourly data help?

**Architecture:** Python package `hypershift` (src layout) with separate modules for data loading, hypergraph construction, Poincaré-ball math, model layers, training loop, metrics, statistics, and an experiment grid runner. The runner writes one results folder per run (`results/<exp>/<label>/seed_<k>/`). An aggregation script then turns those folders into paper-style tables, figures and a written report. Every experiment is one config, and every question maps to a fixed list of configs plus a comparison rule.

**Tech Stack:** Python 3.12, PyTorch 2.x (CUDA wheels matched to the driver, see D0; RTX 3050 4 GB), numpy, scipy, pandas, scikit-learn, matplotlib, pyyaml, pytest. Optional phase: yfinance, alpaca-py. No PyTorch-Geometric: all hypergraph ops use plain `index_add` and `scatter_reduce`.

**Spec:** There is no separate spec. The spec is the paper itself plus this document's "Paper Primer" section.
- Local copy: `C:\Users\Hi\Downloads\05-Hypershift-OA.pdf` (5 pages, no appendix).
- Author copy with appendix: https://tylersnetwork.github.io/papers/icdm22-think.pdf (6 pages; the appendix has the s-distance algorithm, δ_rel, and hypergraph construction).
- The official code repo https://github.com/shivamag125/ICDM22-THINK **is empty** (verified 2026-09-27), so everything here is a reimplementation.

## Global Constraints

- Python `>=3.11` (machine has 3.12.10). Torch from a CUDA wheel index `https://download.pytorch.org/whl/cuXXX` chosen in D0 (the newest one pytorch.org lists whose CUDA version is ≤ the "CUDA Version" printed by `nvidia-smi`; `cu124` is the fallback).
- Poincaré ball curvature `c = 1` everywhere (the paper uses the unit ball).
- Numerical guards:
  - Project every ball point to norm ≤ `1 - 1e-5`.
  - Clamp `artanh` input to `[-1 + 1e-7, 1 - 1e-7]`.
  - Clamp the `sinh` argument in the Poincaré FC layer to `[-15, 15]`.
- Data splits for the RSR data (NYSE and NASDAQ), from the original code:
  - `valid_index = 756`, `test_index = 1008`.
  - NYSE has T = 1245 days. NASDAQ raw files have 1246 rows, and the last row is all-missing and dropped, so NASDAQ also has T = 1245.
- Default hyperparameters (from STHAN-SR/RSR commands and the paper). `lr` and `alpha` get tuned per geometry in Task 14 Phase C; everything else stays fixed:
  - `seq = 16`, `kernel K = 4` (so 16 → 4 → 1), `hidden = 32`
  - `lr = 1e-3`, `weight_decay = 5e-4`, `epochs = 100`, `patience = 20`
  - `alpha` (ranking-loss weight): NYSE 1.0, NASDAQ 0.1
  - `top-k = 5`, `grad_clip = 1.0`
  - `batch_days = 1` (days per optimizer step) and `micro_batch_days = 0` (0 = no split). `micro_batch_days` splits a step into gradient-accumulated chunks. It is mathematically identical to the unsplit step and only saves GPU memory, so it may differ between arms. `batch_days` may **not** differ between compared arms.
- Metric definitions are fixed (Task 7). **"Short ratio" in the request = Sharpe Ratio (SR)** in the paper's Table II:
  - SR = `mean(daily top-5 return) / std(daily top-5 return) * sqrt(252)`.
  - No risk-free rate, no transaction costs (paper-compatible).
  - The constant `15.87` in the STHAN-SR code is `sqrt(252)`.
- Model selection is by the **validation** Sharpe ratio. Test metrics are never used to pick epochs or hyperparameters. The paper-compat "best test epoch" number is logged only as a diagnostic, labelled `test_oracle_sr`.
- Default feature normalization is `norm="train"`: each stock is divided by its max close over the training period. The paper's full-series max (`norm="paper"`) leaks future price levels into the features. Use it only in the E1 comparison.
- Fairness rule: each of the four geometries (HH, HE, EH, EE) gets **its own tuning run with the same budget** (E_tune: lr × alpha, 3 seeds each), and every arm uses the same seeds. Never tune only the "winning" variant.
  - Structure and grouping variants (clique, none, industry-only, …) reuse their geometry's tuned setting and are not tuned separately. That slightly favours the default "hyper" structure, which is the one the tuning ran on. The report must state this next to Q1/Q3.
- Seeds:
  - Any number in a final answer table uses **25 seeds** (paper protocol).
  - Verdict-bearing sweeps (E4, E9, E10) use **15 seeds**.
  - Screening runs and descriptive curves (E_attn, E6, E7, E8) use 5 seeds and never get a verdict.
  - Why 15: with n seeds and m comparisons in a Holm family, the smallest attainable Holm-adjusted Wilcoxon p is `m · 2^(1−n)`. With 10 seeds and the 8-comparison Q1 family that is 0.0156, so no verdict could ever pass p < 0.01. `tests/test_grid.py::test_holm_floor_allows_verdicts` pins this.
- Statistics:
  - Paired Wilcoxon signed-rank test by seed, with Holm correction per question family.
  - Plus a stationary block bootstrap 95% CI on the Sharpe difference of the daily return series averaged over the **seeds common to both arms**. Its estimand is the Sharpe of the seed-ensemble portfolio, not the mean per-seed SR, so the report labels it that way.
- Out of scope, and why: the DTT, CPox, WMill and CSE-risk datasets (not stock ranking; the user asked about stocks). TSE is out of scope because its data (Li et al., IJCAI'20) has no public download; see decision node D14. The paper's GNN baselines (GConvGRU, DCRNN, TGCN, …) are out of scope for v1; see Task 18.
- Output locations:
  - `data/` and `results/` are git-ignored.
  - Code lives in `src/hypershift/` and scripts in `scripts/`.
  - Configs live in `configs/`, docs in `docs/`.
- Shell: Windows machine. Every command below is given in Git Bash syntax. Activate the venv with `source .venv/Scripts/activate`.
- **Keep the venv inside the project (`.venv`), never under `%TEMP%`.** Windows Application Control on this machine blocks pandas' compiled DLLs when they load from the Temp folder ("An Application Control policy has blocked this file").
- Library versions the plan's code was verified against (2026-09-27): torch 2.14, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1. All test code in this plan passed there: 109 passed, and 2 skipped for lack of the full dataset. The real NYSE hypergraph build was re-checked with the on-disk cache: 312 edges, max size 500, max degree 37, and the cached copy is identical (cold 9 s, cached < 0.1 s). (`scripts/aggregate.py`, `plots.py` and `run_grid.py` were smoke-run on fabricated results.)

## Review Focus

1. **Look-ahead leakage in windows.** The target day must be strictly after the last input day, and train targets must be `< valid_index`. The original STHAN-SR loop let train targets reach up to day 771. Pinned by `test_window_offsets_no_leak` (Task 8).
2. **Stocks with missing prices (`-1234` sentinel).** They must be masked out of loss, ranking and top-k selection. They must never be bought. Pinned by `test_topk_skips_masked` (Task 7) and `test_parse_eod_missing_and_gt` (Task 2).
3. **Nodes that belong to no hyperedge.** This happens with about 17 NYSE stocks, after hub removal, and in small universes. They must pass through the spatial layer unchanged, not become NaN or zero. Pinned by `test_isolated_node_passthrough` (Task 5).
4. **Points near the ball boundary.** Near norm 1, `artanh` and `sinh` overflow, and one NaN silently kills a 25-seed run. Pinned by `test_near_boundary_finite_grads` (Task 4) and the non-finite-loss guard in `test_train_raises_on_nan` (Task 8).
5. **Changing universe size changes the portfolio.** Top-5 of 50 stocks and top-5 of 1737 stocks are different bets. Raw SR is therefore not comparable across N. Every N-sweep number is reported next to the Random-5 and Oracle-5 baselines for the same universe. Pinned by `test_baselines_same_universe` (Task 9).
6. **Seed counts must make a verdict possible.** The Holm-adjusted Wilcoxon floor is `m · 2^(1−n)`. Pinned by `test_holm_floor_allows_verdicts` (Task 12). Comparisons also use only the seeds common to both arms (Task 13 `_common_seeds`).
7. **Memory workarounds must not change the experiment.** `micro_batch_days` (gradient accumulation) must equal the unsplit step. Pinned by `test_micro_batch_matches_full_batch` (Task 8).

---

## Part 0 — Paper Primer (read this first; the executor does not have the PDF)

### 0.1 What THINK is

THINK predicts next-day returns for every stock, ranks the stocks, buys the top 5 at today's close and sells them at tomorrow's close. Architecture (paper eq. 17):

```
ŷ = log_0( TConv_hyp_2( DHHAN( TConv_hyp_1( exp_0(X) ), G ) ) )
```

- `X` has shape `[N stocks, τ=16 days, C=5 features]`.
- `TConv_hyp`: the hyperbolic temporal convolution. It takes non-overlapping windows of K steps, Poincaré β-concatenates them, then applies a Poincaré fully connected layer.
- `DHHAN`: distance-aware hyperbolic hypergraph attention. It is applied independently at each remaining time step.
- `G`: the stock hypergraph (static).

### 0.2 Equations (c = 1). Implement exactly these.

| Name | Formula |
|---|---|
| Conformal factor | `λ_x = 2 / (1 - ‖x‖²)` |
| Möbius addition (eq 3) | `x ⊕ y = ((1 + 2⟨x,y⟩ + ‖y‖²) x + (1 - ‖x‖²) y) / (1 + 2⟨x,y⟩ + ‖x‖²‖y‖²)` |
| Distance (eq 4) | `d(x,y) = 2 artanh(‖(-x) ⊕ y‖)` |
| Exp map at 0 (eq 5, x=0) | `exp_0(v) = tanh(‖v‖) v/‖v‖` |
| Log map at 0 (eq 6, x=0) | `log_0(y) = artanh(‖y‖) y/‖y‖` |
| Möbius scalar mult (the `M = r·I` case of eq 7's Möbius matrix-vector mult; used by eq 13) | `r ⊗ x = tanh(r · artanh(‖x‖)) x/‖x‖` |
| Poincaré FC (eq 9-10, HNN++) | For output unit k with params `z_k ∈ R^n`, `r_k ∈ R`: `v_k(x) = 2‖z_k‖ asinh( λ_x ⟨x, z_k/‖z_k‖⟩ cosh(2 r_k) − (λ_x − 1) sinh(2 r_k) )`; `w = sinh(v(x))`; output `y = w / (1 + sqrt(1 + ‖w‖²))` |
| β-concat (eq 11) | Inputs `x_i ∈ B^{n_i}`, total `n = Σ n_i`, `β_m = B(m/2, 1/2)` (Beta function): `y = exp_0( [ (β_n/β_{n_1}) log_0(x_1), …, (β_n/β_{n_M}) log_0(x_M) ] )` |
| Hyperbolic temporal conv (eq 12) | Input length τ = nK. For each of the n windows, β-concat the K points of shape `[N, C]` into `[N, KC]`, then apply Poincaré FC. Output length is n. |
| Node→hyperedge (eq 13, gyromidpoint) | `z_i = ½ ⊗ ( Σ_{k∈e_i} λ_{u_k} u_k / Σ_{k∈e_i} (λ_{u_k} − 1) )` |
| Attention (eq 14) | `α_ij = softmax over {i : v_j ∈ e_i} of  a^T (u_j ⊕ z_i) · d(u_j, z_i)`. The PDF glyph between `u_j` and `z_i` is lost. It is most likely ⊕ (Möbius add), since the paper defines ⊕ in eq 3. Alternatives are handled in decision node D8. |
| Hyperedge→node (eq 15) | `u'_j = exp_0( ReLU( Σ_{i: v_j∈e_i} α_ij · log_0( FC(z_i) ) ) )` |
| Per-timestep (eq 16) | Apply the same layer to every time slice. |

### 0.3 Paper numbers to compare against (Table II, 25-run mean, NYSE unless noted)

| Model | NYSE SR | NYSE NDCG | TSE SR | TSE NDCG | NASDAQ Clf F1 |
|---|---|---|---|---|---|
| RSR-I | 1.05 | 0.75 | 0.99 | 0.72 | 0.38 |
| STHGCN | 1.10 | 0.78 | 1.07 | 0.74 | 0.40 |
| TCONV + DHHAN (Euclidean temporal conv, hyperbolic hypergraph) | 1.14 | 0.81 | 1.11 | 0.76 | 0.44 |
| **THINK** | **1.18 ± 4e-3** | **0.86 ± 9e-4** | **1.19** | **0.81** | **0.49** |

The table reports hyperbolicity for NYSE as δ_hg = 0.5 and δ_rel = 0.087. For NASDAQ it is δ_hg = 1.0 and δ_rel = 0.107. The paper reports **no NASDAQ Sharpe**: its NASDAQ column is movement-classification F1 only (Task 18).

**Which of our numbers is comparable to the paper's 1.18.** The paper's protocol follows STHAN-SR, whose training loop evaluates the test set after every epoch and has no validation-based checkpointing. The paper also uses the full-series-max feature normalization. The apples-to-apples number is therefore `norm=paper` + all 100 epochs + best test epoch. That is `test_oracle_sr` of `E1_main/THINK_paperProtocol`, a dedicated arm with no early stopping: an early-stopped run would take its max over fewer epochs and understate the protocol. That number is optimistic by construction. Our honest headline is the validation-selected `test_sr` of `E1_main/THINK` (`norm=train`). Report both, side by side (D9).
- The paper's ± values (e.g. ±4e-3 over 25 runs) are far smaller than any realistic seed spread. Expect our seed std to be roughly 0.1–0.5, and do not treat that as a reproduction failure.

- **Fig 2** (THINK vs HHN, where HHN = THINK without distance attention) uses only DTT/CPox. We run the same ablation on NYSE instead.
- **Fig 3a** (NYSE): the hyperedge-decomposition x-axis runs 500, 15, 9, 5, 3. SR falls from about 1.2 to about 0.9 for THINK, and the Euclidean curve sits below it.
- **Fig 3b** (NYSE): the hub-removal x-axis is node degree 31, 28, 22, 16, 2. SR falls from about 1.1 to about 0.8.

**NDCG warning:** the published STHAN-SR evaluator (the paper family's code) computes `ndcg_score` on *sets of ticker indices* and keeps only the last test day. The paper's NDCG therefore cannot be reproduced meaningfully. We report a correct NDCG@5 as `ndcg5`, plus a re-implementation of the buggy one (`ndcg_sthan`) for reference only. **Compare SR, not NDCG, against the paper.**

### 0.4 How the paper groups stocks (answer to Q1; verified against the data)

The paper follows STHAN-SR [38] and appendix B. Hyperedges come from the RSR dataset (Feng et al. 2019, `relation.tar.gz`):

1. **Industry hyperedges.** One hyperedge per industry, containing all stocks in that industry. The source tensor is `relation/sector_industry/NYSE_industry_relation.npy`, shape `[1737, 1737, 108]`. The last channel is a self-loop channel and is ignored. The largest NYSE industry has **500 stocks**, which matches the "500" on the Fig 3a axis. That confirms the construction.
2. **Wiki corporate hyperedges.**
   - First-order (`X –R1→ Y`): one hyperedge = a source stock plus all target stocks linked to it by the same Wikidata relation.
   - Second-order (`X –R2→ Z ←R3– Y`): pairwise.
   - The source is `relation/wikidata/NYSE_wiki_relation.npy`, shape `[1737, 1737, 33]`, last channel = self. The channel-to-path mapping was built from an unordered Python `set`, so first- and second-order channels cannot be separated reliably. Only 29 of 758,189 paths are first-order.
   - Our construction ("star per (source, relation)"): for each channel k and source i, hyperedge = `{i} ∪ {j : rel[i,j,k] = 1}`. Second-order relations with a single target become pairs, as the paper describes.
3. Merge both lists and **deduplicate** identical hyperedges. Drop hyperedges with fewer than 2 nodes.

Expected NYSE result (prototype-verified): **312 hyperedges**, 107 of them industry. Max hyperedge size is 500. Max node degree is 37. 17 stocks have no hyperedge. Expected NASDAQ result: 162 hyperedges, max size 156.

The authors' own `hypergraph_nyse.npy` was never published, so an exact match is impossible. Tolerances are given in decision node D3.

### 0.5 Question → experiment → decision rule map

| # | User question | Experiment(s) | Primary comparison | "Yes" requires |
|---|---|---|---|---|
| Q0 | Can we reproduce the paper's numbers? | E1_main | THINK SR vs 1.18 | paper-protocol SR (`THINK_paperProtocol` `test_oracle_sr`) in [1.03, 1.33]; report the validation-selected `norm=train` SR next to it as the honest number (D9) |
| Q1 | How are stocks grouped, and does the grouping choice matter? | E4_grouping, plus E1_main/THINK and E2_geometry/EE as the "both" arms | both (industry+wiki) vs industry / wiki / random-matched / correlation-clusters; "none" is covered by Q3 | "both" beats "random" with a STRONG verdict ⇒ semantic grouping matters |
| Q2 | Is hyperbolic space actually good? | E2_geometry (2×2 temporal×spatial) | HH vs EE; HH vs EH (temporal effect); HH vs HE (spatial effect) | STRONG verdict (defined below) |
| Q3 | Do hyperedges matter? | E5_structure, E6_decompose, E7_hubs | hyper vs clique vs none (same geometry) | hyper > clique STRONG ⇒ higher-order matters; hyper > none ⇒ relations matter at all |
| Q4 | Do hyperedges and hyperbolic space both matter (interaction)? | E2 + E5 cells | interaction `(HH_hyper − HH_clique) − (EE_hyper − EE_clique)` per seed | Wilcoxon on the interaction < 0.01 ⇒ they reinforce each other |
| Q5 | Does the number of stocks matter? | E8_universe (N = 50…1000, 3 random universes each) + E1/E2 as the N = 1737 point | SR, and SR minus Random-5 SR, vs N; the HH−EE gap vs N (`results/tables/universe.md`) | Spearman ρ over the (N, universe) cells with p < 0.01 ⇒ "STRONG trend", otherwise "NO EVIDENCE of a trend". Always show `num_edges` and covered-node fraction per N: random subsets thin the hypergraph, so N and hypergraph density are confounded, and the report must say so |
| Q6 | Daily data (paper setting, plus out-of-period) | E1/E2 on RSR daily; E9_fresh_daily (2015–2026) | same comparisons as Q2/Q3 on new data | same verdict rules |
| Q7 | Hourly data | E10_hourly (3 arms, same source and window) | `hday` (hourly bars, 16-day look-back = 112 bars, first kernel = 28 bars = 4 days) vs `daily` (daily bars, 16-day look-back, first kernel 4 days): same look-back, horizon, trades and architecture; only input granularity differs. Plus the hourly-trading arm, net of costs | same verdict rules; costs reported at 0/5/10 bps |
| — | Distance-aware attention (paper Fig 2) | E3_hhn | THINK vs HHN | same verdict rules |
| — | Hyperbolicity (paper Table I) | Task 11 script | δ_hg, δ_rel for NYSE/NASDAQ and every E8 subset | within the D16 tolerances; Spearman(δ, HH−EE gain) |

**Verdict rules (used everywhere):**

- **STRONG:** Holm-adjusted Wilcoxon p < 0.01 **and** the bootstrap 95% CI of the SR difference excludes 0.
- **SEED-ROBUST ONLY:** Holm p < 0.01 but the CI includes 0. The difference is reliable across random seeds but smaller than market noise over the approx. 237-day test window, where the SR standard error is about 1.0.
- **NO EVIDENCE:** otherwise.
- **INSUFFICIENT SEEDS:** written by the aggregator instead of NO EVIDENCE when fewer than 6 common seeds exist, or when the Holm floor `m · 2^(1−n)` (the `p_floor` column) is ≥ 0.01. In that case the test could not have passed however large the effect. Fix it by running more seeds (`run_grid.py --labels … --seeds …`), not by rewording.

The written report must use these exact words.

---

## Part 1 — Master Decision Tree

The executor walks this tree top to bottom. Each node says what to check, what to do when the check fails, and where to go next. Tasks refer to node IDs.

**D0 — Environment**
- Choosing the torch wheel index:
  1. Run `nvidia-smi` and read "CUDA Version: X.Y" in the header.
  2. Open https://pytorch.org/get-started/locally/ (Stable, Pip, Python, CUDA) and take the newest listed `cuXXX` index whose version is ≤ X.Y. For example, driver 12.8 → `cu128`.
  3. If `nvidia-smi` is missing or the page can't be read, use `cu124`.
- Run `python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`.
  - `True`: continue to D1. Any torch ≥ 2.2 is fine; the code uses only `index_add`, `scatter_reduce` and basic ops.
  - `False`: reinstall with `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/<index>`, trying the next-older index each time: cu128 → cu126 → cu124 → cu121.
    - Still `False` (driver problem): **STOP and ask the user.** CPU-only is not a viable mode for this study. The measured CPU cost is 0.17 s/step × ~740 steps/epoch plus evaluation, about 2.5 min/epoch, or about 2.5 h per full-NYSE run. The full grid would take over 1000 h. Offer the user three options:
      - (a) fix the NVIDIA driver;
      - (b) use a cloud GPU;
      - (c) a reduced CPU study: E1_main (THINK, THINK_paperNorm) + E2_geometry EE + E5_structure HH_none, 15 seeds each, `universe_size: 500` for all of them (about 45 min/run, about 45 h in total). Run it via `run_grid.py <exp> --labels <label> --seeds 0-14 --set device=cpu universe_size=500`. The report must then say "N = 500 random subset" everywhere.
    - If the user can't be reached, do (c) and record it.

**D1 — Data download** (Task 2)
- The sparse git clone fails or is very slow: download `https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/archive/refs/heads/master.zip`. Extract only `data/2013-01-01/`, `data/*.csv` and `data/relation.tar.gz` into `data/raw/rsr/data/`.
- `relation.tar.gz` fails to extract: download it directly from `https://raw.githubusercontent.com/fulifeng/Temporal_Relational_Stock_Ranking/master/data/relation.tar.gz` (7.3 MB).

**D2 — Data validation** (Task 2 integration test)
- NYSE must be `features.shape == (1737, 1245, 5)`. NASDAQ must be `(1026, 1245, 5)`.
- Mismatch: STOP. Print the ticker count and the row count of the first file, and report to the user. Do not "fix" it by truncating.
- The folder has more files than tickers (1769 NYSE and 1048 NASDAQ files). This is expected: always use the ticker list file.

**D3 — Hypergraph sanity** (Task 3 integration test)
- NYSE target: #edges in [250, 400], max size = 500, max node degree in [20, 60].
- Off target: check that the last (self) channel is excluded, that the diagonal is zeroed for wiki, and that deduplication happens. After that, still off by more than 30%: continue anyway, but record the actual numbers in `docs/report.md` under "Deviations".
- `build_rsr_hypergraph` caches its edge list in `data/raw/rsr/data/hypergraph_cache/`. The NYSE industry tensor is 2.6 GB, so it is loaded only once. **Delete that folder whenever you change any construction code**, or you will keep getting the old hypergraph.
- The first build needs about 4 GB of free RAM. On `MemoryError`, close other programs and retry.

**D4 — Math unit tests** (Tasks 4–6)
- Any failure blocks all later tasks. Never skip or loosen a geometry test to make it pass.
- Fix the implementation. If a test's expected value looks wrong, re-derive it by hand in a comment before changing it.

**D5 — NaN or inf during training** (Task 8 guard raises `FloatingPointError`)
Apply these in order, re-running 1 seed after each change. Put each change in `configs/global.yaml`, which overrides `results/tuned.json`, so that it applies to **every arm**:
1. Confirm that `project()` is called after every `expmap0`, `mobius_add` result used as a point, `mobius_scalar` and PoincareLinear output.
2. Lower `lr` to 5e-4.
3. Set `grad_clip = 0.5`.
4. Run the model in float64. In `train_one_run`, append `.double()` to `build_model(...).to(device)`. In both `train_one_run` and `predict_split`, pass `dtype=torch.float64` to every `torch.as_tensor(...)` of a batch. This costs about 2× time, and it applies to every arm.
5. Still NaN: set `attn_dist = neg`. This is bounded, unlike `mult`, which can blow up with large distances.

Record which step fixed it in the report.

**D6 — Smoke/overfit** (Task 8): the synthetic-signal test must show the loss decreasing.
- If not: check the gradient flow. Look for parameters with `p.grad is None` after `backward()`, then check the sign of the ranking loss.

**D7 — Compute budget** (Task 12, `scripts/time_budget.py`)
Measure seconds per epoch `t_e` for HH on the full NYSE, with `batch_days = 1`. Then:
- `t_run = t_e × 60` (a typical early-stopped run length).
- `t_run ≤ 6 min`: use the plan as written.
- `6 < t_run ≤ 15 min`: set `batch_days: 4` in `configs/global.yaml` (so it applies to every experiment uniformly) and re-measure. Keep lr unchanged.
- `t_run > 15 min`:
  1. Set `batch_days: 8` and `epochs: 60` in `configs/global.yaml`.
  2. Edit `src/hypershift/experiments/grid.py` to set `SEEDS_SCREEN = tuple(range(3))`. Final seeds stay 25 for E1/E2/E3/E5, and sweep seeds stay 15 for E4/E9/E10.
- Clique and decomposition arms are much slower. The measured cost on CPU for full NYSE is 0.17 s per step for the hypergraph vs 3.4 s for the clique (145,547 pairs). So `time_budget.py` also times one `HH_clique` epoch. The grid already gives clique and decomposition arms `micro_batch_days: 1`, so they fit in 4 GB whatever `batch_days` is.
  - If a clique epoch takes more than 10× a hypergraph epoch, run the clique arms on 15 seeds instead of 25. That leaves the Holm floor for Q3 at 4 × 2^-14 ≈ 2e-4:
    ```bash
    python scripts/run_grid.py E5_structure --labels HH_none EE_none
    python scripts/run_grid.py E5_structure --labels HH_clique EE_clique --seeds 0-14
    ```
    Also run `E6_decompose` with sizes {15, 5} only, for both schedules: `python scripts/run_grid.py E6_decompose --labels HH_large_15 HH_large_5 EE_large_15 EE_large_5 HH_small_15 HH_small_5 EE_small_15 EE_small_5`.
  - Keep every other setting identical to the hypergraph arms, and record the change in the report.
  - Once you have used `--labels`/`--seeds` for an experiment, **use the same flags every time you re-run that experiment** in later phases. A bare `run_grid.py <exp>` would schedule the runs you cut.
- Then estimate the total GPU hours; `time_budget.py` prints the estimate, and `scripts/run_grid.py --dry-run` gives the run counts.
  - More than 72 h: ask the user to approve the cuts below, **in this order**. Stop cutting as soon as the estimate is ≤ 72 h. If the user can't be reached, apply the cuts yourself in the same order and record them.
    1. Drop the two NASDAQ arms from E1_main (50 runs; the paper has no NASDAQ Sharpe anyway): `python scripts/run_grid.py E1_main --labels THINK THINK_paperNorm THINK_paperProtocol THINK_shuffled`.
    2. Cut E8 sizes to {100, 500, 1000}: use `--labels` with the matching `HH_N…`/`EE_N…` labels.
    3. Drop the E6 `small_first` schedule.
    4. Set `SEEDS_SCREEN = tuple(range(3))` if it isn't already.
  - **Never cut the E1/E2/E3/E5 seeds below 25, or the E4/E9/E10 seeds below 15, and never drop `THINK_paperProtocol`**: it is the user's headline Sharpe comparison. Budget it at 100 epochs per run, not 60.
  - Still more than 72 h after all four cuts: tell the user the new estimate, then continue in phase order (C → D → E → F → G → Task 16). Runs are resumable, and the phase order puts the core questions (Q0, Q2, Q3, Q4) first, so stopping early loses only the later sweeps.

**D8 — Attention variant** (Task 14 Phase D, from E_attn with 5 seeds)
- Pick `(attn_score, attn_dist)` with the best mean **validation** SR among mobius/concat × mult/neg.
- If `mobius_mult` (the literal reading of the paper) is within 1 std (the top variant's seed std) of the top variant, pick `mobius_mult` instead. `aggregate.py --select-attn` implements exactly this rule.
- Write the choice into `configs/chosen.yaml`. All later HH/EH experiments use it. Euclidean spatial variants use the same `(score, dist)` names in their Euclidean form.

**D9 — Main result vs paper** (Task 14 Phase E, E1)
Two numbers matter (Part 0.3):
- `P` = `test_oracle_sr_mean` of `E1_main/THINK_paperProtocol`: the paper-protocol number (100 epochs, no early stopping).
- `H` = `test_sr_mean` of `E1_main/THINK`: the validation-selected `norm=train` headline.

Rules:
- `P` in [1.03, 1.33]: **reproduced (under the paper's protocol)**. Report `P` and `H` side by side, and continue.
- `P` below 1.03: walk this list, one change at a time. Use 5 seeds each on the cheaper early-stopped `THINK_paperNorm`, via `python scripts/run_grid.py E1_main --labels THINK_paperNorm --seeds 0-4 --set exp=E1_debug_<n> <key>=<value>`. The **trigger** is `P`, but **adopt** a change only if it raises the mean **validation** SR over seeds 0–4, compared with `E1_main/THINK_paperNorm` **seeds 0–4 only**, so that both sides average the same five seeds. That way test data never picks a setting. Mean validation SR over seeds 0–4 of any `<exp>/<label>`:
  ```bash
  python -c "import json,sys; print(sum(json.load(open(f'results/{sys.argv[1]}/seed_{s}/metrics.json'))['val']['sr'] for s in range(5)) / 5)" E1_main/THINK_paperNorm
  ```
  1. `target=price` (the STHAN-SR style output).
  2. Tuned hyperparameters: make sure E_tune ran and `results/tuned.json` is applied.
  3. `batch_days=1` if it had been raised.
  4. `hidden=64`.
  5. `seq=36 kernel=6` instead of `seq=16 kernel=4`.
  6. Check the metric. In `results/baselines/NYSE.json`:
     - Oracle-5 SR must be very large (> 10).
     - Random-5 `irr` must be within ±0.15 of market `irr`. Both have the same expected daily return, since random-5 is a uniform draw of valid stocks.
     - Random-5 SR is usually *below* market SR because 5 stocks are poorly diversified, so do not compare the SRs.
     - If either check fails, the evaluation is broken; go to D10.
  - If `P` is still low after all of that: accept it and report "not reproduced", with the gap. Continue to the ablations anyway, since the relative comparisons are still informative.
- `H` above 1.33, or any single seed's `test_sr` above 4: suspect leakage and go to D10 before anything else. (`P` above 1.33 alone just means test-epoch selection is optimistic; note it.)

**D10 — Leakage audit** (mandatory once, in Task 14 Phase E, whatever the numbers are)
1. Shuffled-label run (`E1_main/THINK_shuffled`, 3 seeds). Inside the training split only, the labels are permuted **across stocks within each day**. That destroys every learnable link between a stock's features and its return, including legitimate stock-level drift. Compare against `results/baselines/NYSE.json` → `random` (`sr` and `sr_std`):
   - **Leak (STOP):** the shuffled mean test SR is > `random.sr + 2 · random.sr_std`, or any shuffled seed's test SR is > 4. The model is reading future information through the features or windows. Re-check `window_offsets`, `gather_batch` and the feature normalization, and re-run the Task 8 tests.
   - **Pass:** otherwise. Do not STOP if the shuffled model is merely somewhat above or below random; the 3 seeds' picks are correlated, so their mean is noisy.
   - Also record whether `E1_main/THINK` beats the shuffled mean. If it doesn't, write "THINK is not distinguishable from a no-signal model" in the report.
2. Compare `norm=paper` vs `norm=train` SR in E1. A gap > 0.3 means the full-series-max normalization leaks. Report both, and base all conclusions on `norm=train`.
3. Compare `test_oracle_sr` vs the selected `test sr`. A gap > 0.2 means test-epoch selection would inflate numbers. State this in the report as a possible reason the paper's numbers are higher.

**D11 — Interpreting each question.** Use the verdict rules in Part 0.5. A result opposite to the paper is a **finding, not a bug**, provided D5/D6/D10 passed. Never re-tune only the losing variant.

**D12 — Hourly data source** (Task 15)
- If the user has Alpaca keys (env `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`):
  - Use Alpaca 30-minute bars with `feed = sip` from 2016-01-01, resampled to 1-hour RTH bars anchored at 9:30.
  - HTTP 403 or "subscription does not permit": switch to `feed = iex`. Then also require a mean daily IEX volume ≥ 10k shares per ticker; drop tickers below that.
- No keys: ask the user once whether they want to create a free Alpaca account at alpaca.markets, and show the two env var names.
  - They decline, or can't be reached: use yfinance `interval = "60m"`, `period = "730d"` (hard limit of about 730 days back).
  - Tell the user the window is then about 2 years and the test set is about 100 trading days, so conclusions will be weak.
- Either way, **the daily comparison bars are aggregated from the same hourly bars**, so the source, universe and calendar are identical.

**D13 — Hourly data cleaning** (Task 15)
- Keep only regular trading hours, 09:30 ≤ t < 16:00 America/New_York.
- Drop timestamps where fewer than 50% of tickers have a bar (half-days, glitches).
- Drop tickers missing more than 5% of the remaining bars.
- For the rest: forward-fill prices, and set `mask = 0` for filled bars so they are never traded or scored.
- Fewer than 300 tickers survive: lower the threshold to 10% and report it.

**D14 — TSE.** Skip. The report says: "TSE data (Li et al. IJCAI'20) is not publicly downloadable; not reproduced."
- If the user later provides the files: add a loader returning `MarketData`, then rerun E1/E2 with `market: TSE`.

**D15 — GPU out of memory** (`failed.json` contains `OutOfMemoryError`, or a direct run crashes with it)
- Typical triggers: clique expansion (about 145k pairs on NYSE), large-edge decomposition, or `batch_days > 1`.
- Fix in this order:
  1. Set `micro_batch_days=1` for the failing labels: `python scripts/run_grid.py <exp> --labels <labels> --set micro_batch_days=1`. This does not change the math, so it needs no matching change elsewhere, and the comparison stays valid.
  2. Still OOM: set `hidden=16` **for every arm of every experiment that the failing arm is compared with** (see `FAMILIES` in grid.py). In practice that means putting `hidden: 16` in `configs/global.yaml`, deleting every `results/E*` folder except `results/E0_budget`, and re-running from Phase C.
  3. Still OOM: ask the user. Do not move one arm to a different `batch_days`, because that would confound the comparison.
- Record the change in the report.

**D16 — Hyperbolicity tolerance** (Task 11)
- δ_hg must be a multiple of 0.5 (distances are integers). Target NYSE 0.5 and NASDAQ 1.0; ±0.5 is acceptable.
- δ_rel depends on the feature choice; target NYSE 0.087 ± 0.05.
- Outside tolerance: report the value, note that the feature definition may differ, and continue.

**D17 — Fresh data looks too good** (SR > 3 on daily data)
- Check survivorship: the universe is the *current* S&P 500 list. This bias inflates all arms equally, so relative comparisons remain valid.
- Absolute SR must be labelled "survivorship-biased" in the report.

**D18 — Hourly look-back** (Task 16 Step 4). The rule is given there.

**D19 — Anything not covered above**
Examples: an exception type no node names, a test that starts failing after an unrelated change, or a result file that is missing.
1. Re-run `pytest -q`. If a test fails, fix the code until it passes, following D4's rule: never loosen a test to make it pass.
2. If the tests pass but the problem persists, do not improvise changes to the model, loss, metrics or splits. Save the traceback, the command and the config to `docs/report.md` under "Run log", then ask the user.
3. Changes that only touch plumbing (paths, flags, printing) are fine without asking, but record them.

---

## Part 2 — File Structure

```
pyproject.toml                         # package + deps (Task 1)
.gitignore                             # + data/, results/, .venv/ (Task 1)
configs/
  think_nyse.yaml                      # default HH config NYSE (Task 8)
  think_nasdaq.yaml                    # default HH config NASDAQ (Task 8)
  chosen.yaml                          # written in Phase D (D8)
scripts/
  download_rsr.sh                      # RSR data fetch (Task 2)
  baselines.py                         # Random/Momentum/Reversal/Market/Oracle (Task 9)
  hyperbolicity.py                     # Table I numbers (Task 11)
  time_budget.py                       # D7 timing (Task 12)
  run_grid.py                          # run an experiment by name, resumable (Task 12)
  aggregate.py                         # tables + stats (Task 13)
  plots.py                             # figures (Task 13)
  fetch_fresh.py                       # yfinance/alpaca daily & hourly (Task 15)
src/hypershift/
  __init__.py
  config.py                            # RunConfig dataclass, yaml io (Task 8)
  run.py                               # CLI: python -m hypershift.run (Task 8)
  data/__init__.py
  data/rsr.py                          # MarketData, load_rsr, renormalize_train (Task 2)
  data/hypergraph.py                   # Hypergraph + all constructions/transforms (Task 3)
  data/universe.py                     # random universe subsets (Task 3)
  data/fresh.py                        # fresh daily/hourly panels, GICS hyperedges (Task 15)
  geometry/__init__.py
  geometry/poincare.py                 # ball math (Task 4)
  geometry/hyperbolicity.py            # delta_hg, delta_rel (Task 11)
  models/__init__.py
  models/layers.py                     # PoincareLinear, beta_concat, temporal convs (Task 5)
  models/attention.py                  # segment_softmax, Hyp/Euc hypergraph attention (Task 5)
  models/think.py                      # THINK with geometry/structure switches (Task 6)
  train/__init__.py
  train/loss.py                        # MSE + pairwise ranking loss (Task 8)
  train/loop.py                        # windows, batches, train_one_run (Task 8)
  eval/__init__.py
  eval/metrics.py                      # SR, IRR, NDCG, MSE, MDD, costs (Task 7)
  eval/baselines.py                    # non-learned rankers (Task 9)
  eval/stats.py                        # Wilcoxon, Holm, bootstrap (Task 10)
  experiments/__init__.py
  experiments/grid.py                  # all experiments + comparisons (Task 12)
tests/
  conftest.py                          # synthetic market fixture (Task 2, extended Task 8)
  test_env.py  test_rsr.py  test_hypergraph.py  test_poincare.py  test_layers.py
  test_attention.py  test_think.py  test_metrics.py  test_loop.py  test_baselines.py
  test_stats.py  test_hyperbolicity.py  test_grid.py  test_fresh.py
docs/
  report.md                            # final answers (Task 17)
```

## Part 3 — Tasks

### Task 1: Project scaffold and environment

**Files:**
- Create: `pyproject.toml`, `src/hypershift/__init__.py`, `tests/test_env.py`
- Modify: `.gitignore` (append lines)

**Interfaces:**
- Produces: installable package `hypershift` with `__version__ == "0.1.0"`.

- [ ] **Step 1: Write the failing test** `tests/test_env.py`

```python
def test_imports():
    import numpy, scipy, sklearn, torch  # noqa: F401
    import hypershift
    assert hypershift.__version__ == "0.1.0"
```

- [ ] **Step 2: Create the venv and install torch** (decision node D0 applies)

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
nvidia-smi                      # read "CUDA Version: X.Y"; pick <index> as in D0 (e.g. cu128, fallback cu124)
pip install torch --index-url https://download.pytorch.org/whl/<index>
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```
Expected: something like `2.x.x+cu128 True`. If you get `False`, go to D0.

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "hypershift"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26", "scipy>=1.11", "pandas>=2.1", "scikit-learn>=1.3",
  "pyyaml>=6", "matplotlib>=3.8", "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8"]
fresh = ["yfinance>=0.2.40", "alpaca-py>=0.30", "lxml>=5"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["data: needs downloaded RSR data", "slow: long running"]
```

`src/hypershift/__init__.py`:
```python
__version__ = "0.1.0"
```

Create empty `__init__.py` files in `src/hypershift/{data,geometry,models,train,eval,experiments}/`.

Append to `.gitignore`. The leading `/` is required: an unanchored `data/` would also ignore the source package `src/hypershift/data/`, and every later `git add` of it would fail.
```
/data/
/results/
/.venv/
```
Check it: `git check-ignore -v src/hypershift/data/__init__.py` must print nothing.

- [ ] **Step 4: Install the package and run the test**

```bash
pip install -e ".[dev]"
pytest tests/test_env.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "chore: scaffold hypershift package"
```

---

### Task 2: RSR data download and loader

**Files:**
- Create: `scripts/download_rsr.sh`, `src/hypershift/data/rsr.py`, `tests/conftest.py`, `tests/test_rsr.py`

**Interfaces:**
- Produces:
  - `MarketData` (dataclass) with fields:
    - `tickers: list[str]`
    - `features: np.ndarray [N,T,C] float32`
    - `mask: np.ndarray [N,T] float32`
    - `gt: np.ndarray [N,T] float32` (return from t−1 to t)
    - `base_price: np.ndarray [N,T] float32`
    - `valid_index: int`, `test_index: int`
    - `timestamps: np.ndarray | None = None`
    - `eligible_ends: np.ndarray | None = None`
  - `MarketData` properties `num_nodes`, `num_steps`, and method `subset(idx) -> MarketData`.
  - `load_rsr(root, market, norm="train") -> MarketData`
  - `renormalize_train(data) -> MarketData`
  - `parse_eod(raw, drop_last) -> (feats, mask, gt, base)`
  - Constants `MISSING = -1234.0`, `FILL = 1.1`.
  - Fixture `synthetic_market` in `tests/conftest.py`.

- [ ] **Step 1: Write the download script** `scripts/download_rsr.sh`

```bash
#!/usr/bin/env bash
# Fetches only what we need from the RSR repo (~100 MB instead of ~340 MB). See decision node D1 on failure.
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash would otherwise rewrite "/data/..." below into a Windows path
DEST=data/raw/rsr
mkdir -p data/raw
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/fulifeng/Temporal_Relational_Stock_Ranking.git "$DEST"
fi
cd "$DEST"
git sparse-checkout init --no-cone
git sparse-checkout set "/data/2013-01-01/*" "/data/*.csv" "/data/relation.tar.gz"
git checkout master
tar xzf data/relation.tar.gz -C data/
ls data/relation/sector_industry data/relation/wikidata
```

Run it:
```bash
bash scripts/download_rsr.sh
```
Expected: the output lists `NYSE_industry_relation.npy`, `NASDAQ_industry_relation.npy`, `NYSE_wiki_relation.npy` and `NASDAQ_wiki_relation.npy`. The data root for all later code is `data/raw/rsr/data`.

- [ ] **Step 2: Write failing tests** `tests/test_rsr.py`

```python
from pathlib import Path
import numpy as np
import pytest
from hypershift.data.rsr import FILL, load_rsr, parse_eod, renormalize_train

REAL = Path("data/raw/rsr/data")


def test_parse_eod_missing_and_gt():
    raw = np.array([
        [0, .5, .5, .5, .5, .50],
        [1, .5, .5, .5, .5, .55],
        [2, -1234, -1234, -1234, -1234, -1234],
        [3, .5, .5, .5, .5, .60],
    ])
    f, m, g, b = parse_eod(raw, drop_last=False)
    assert m.tolist() == [1, 1, 0, 1]
    assert g[1] == pytest.approx(0.1)
    assert g[2] == 0 and g[3] == 0          # current missing / previous missing -> 0
    assert f[2].tolist() == pytest.approx([FILL] * 5)
    assert b[2] == pytest.approx(FILL)
    assert f.shape == (4, 5)


def test_parse_eod_drop_last():
    raw = np.tile(np.array([[0, .5, .5, .5, .5, .5]]), (3, 1))
    f, m, g, b = parse_eod(raw, drop_last=True)
    assert f.shape == (2, 5)


def _write_market(root: Path, market: str, tickers, T):
    (root / "2013-01-01").mkdir(parents=True)
    (root / f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv").write_text("\n".join(tickers) + "\n")
    rng = np.random.default_rng(0)
    for t in tickers:
        close = np.linspace(0.2, 0.9, T) + rng.normal(0, 0.01, T)
        rows = np.column_stack([np.arange(T), close, close, close, close, close])
        np.savetxt(root / "2013-01-01" / f"{market}_{t}_1.csv", rows, delimiter=",", fmt="%.6f")


def test_load_rsr_synthetic_and_train_norm(tmp_path):
    _write_market(tmp_path, "NYSE", ["AAA", "BBB"], T=1100)
    d = load_rsr(tmp_path, "NYSE", norm="paper")
    assert d.features.shape == (2, 1100, 5)
    assert (d.valid_index, d.test_index) == (756, 1008)
    dt = renormalize_train(d)
    train_close_max = dt.features[:, :756, -1].max(axis=1)
    np.testing.assert_allclose(train_close_max, 1.0, rtol=1e-5)
    np.testing.assert_allclose(dt.gt, d.gt)  # returns are scale-invariant


@pytest.mark.data
@pytest.mark.skipif(not REAL.exists(), reason="RSR data not downloaded")
def test_real_shapes():
    ny = load_rsr(REAL, "NYSE", norm="paper")
    assert ny.features.shape == (1737, 1245, 5)
    na = load_rsr(REAL, "NASDAQ", norm="paper")
    assert na.features.shape == (1026, 1245, 5)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_rsr.py -v`
Expected: FAIL with `ModuleNotFoundError: hypershift.data.rsr`.

- [ ] **Step 4: Implement** `src/hypershift/data/rsr.py`

```python
"""Loader for the RSR (Feng et al. 2019) NYSE/NASDAQ daily dataset.

Each file NYSE_<T>_1.csv has rows: date_index, ma5, ma10, ma20, ma30, close, all divided by the
stock's full-series max close ("paper" normalization). -1234 marks a missing day.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

MISSING = -1234.0
FILL = 1.1  # value the original code writes into missing cells
SPLITS = {"NYSE": (756, 1008), "NASDAQ": (756, 1008)}


@dataclass
class MarketData:
    tickers: list[str]
    features: np.ndarray    # [N, T, C] float32
    mask: np.ndarray        # [N, T] float32, 1 = price observed on day t
    gt: np.ndarray          # [N, T] float32, return from t-1 to t (0 if either day missing)
    base_price: np.ndarray  # [N, T] float32, normalized close at t
    valid_index: int
    test_index: int
    timestamps: np.ndarray | None = None
    eligible_ends: np.ndarray | None = None  # allowed "last input step" indices; None = all

    @property
    def num_nodes(self) -> int:
        return self.features.shape[0]

    @property
    def num_steps(self) -> int:
        return self.features.shape[1]

    def subset(self, idx: np.ndarray) -> "MarketData":
        idx = np.asarray(idx)
        return replace(
            self,
            tickers=[self.tickers[i] for i in idx],
            features=self.features[idx],
            mask=self.mask[idx],
            gt=self.gt[idx],
            base_price=self.base_price[idx],
        )


def read_ticker_file(path: Path) -> list[str]:
    return [ln.strip().split("\t")[0] for ln in Path(path).read_text().splitlines() if ln.strip()]


def parse_eod(raw: np.ndarray, drop_last: bool):
    if drop_last:
        raw = raw[:-1]
    miss = np.abs(raw[:, -1] - MISSING) < 1e-8
    close = raw[:, -1]
    gt = np.zeros(len(raw), dtype=np.float32)
    ok = (~miss[1:]) & (~miss[:-1])
    gt[1:][ok] = (close[1:][ok] - close[:-1][ok]) / close[:-1][ok]
    feats = raw[:, 1:].copy()
    feats[np.abs(feats - MISSING) < 1e-8] = FILL
    base = feats[:, -1].copy()
    return feats.astype(np.float32), (~miss).astype(np.float32), gt, base.astype(np.float32)


def renormalize_train(data: MarketData) -> MarketData:
    """Divide each stock by its max observed close over the training period (removes look-ahead)."""
    close = np.where(data.mask > 0, data.features[:, :, -1], 0.0)
    scale = close[:, : data.valid_index].max(axis=1)
    scale = np.where(scale > 0, scale, 1.0)
    feats = data.features / scale[:, None, None]
    base = data.base_price / scale[:, None]
    miss = data.mask < 0.5
    feats[miss] = FILL
    base[miss] = FILL
    return replace(data, features=feats.astype(np.float32), base_price=base.astype(np.float32))


def load_rsr(root: Path | str, market: str, norm: str = "train") -> MarketData:
    root = Path(root)
    tickers = read_ticker_file(root / f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv")
    parts = [
        parse_eod(
            np.loadtxt(root / "2013-01-01" / f"{market}_{t}_1.csv", delimiter=",", dtype=np.float64),
            drop_last=(market == "NASDAQ"),
        )
        for t in tickers
    ]
    vi, ti = SPLITS[market]
    data = MarketData(
        tickers=tickers,
        features=np.stack([p[0] for p in parts]),
        mask=np.stack([p[1] for p in parts]),
        gt=np.stack([p[2] for p in parts]),
        base_price=np.stack([p[3] for p in parts]),
        valid_index=vi,
        test_index=ti,
    )
    if norm == "train":
        return renormalize_train(data)
    if norm != "paper":
        raise ValueError(f"unknown norm {norm!r}")
    return data
```

- [ ] **Step 5: Add the synthetic fixture** `tests/conftest.py` (used by Tasks 8, 9, 12)

```python
import numpy as np
import pytest

from hypershift.data.rsr import MarketData


def make_synthetic_market(N=12, T=80, C=5, valid_index=40, test_index=60, seed=0, signal=0.5):
    """Random-walk prices; next-day return partly predictable from last-day return (planted signal)."""
    rng = np.random.default_rng(seed)
    ret = np.zeros((N, T), dtype=np.float32)
    ret[:, 0] = rng.normal(0, 0.01, N)
    for t in range(1, T):
        ret[:, t] = signal * ret[:, t - 1] + rng.normal(0, 0.01, N)
    close = np.cumprod(1 + ret, axis=1)
    close = close / close[:, :valid_index].max(axis=1, keepdims=True)
    feats = np.repeat(close[:, :, None], C, axis=2).astype(np.float32)
    mask = np.ones((N, T), dtype=np.float32)
    mask[0, 50] = 0.0  # one missing day
    gt = np.zeros((N, T), dtype=np.float32)
    gt[:, 1:] = close[:, 1:] / close[:, :-1] - 1
    gt[0, 50] = gt[0, 51] = 0.0
    return MarketData([f"S{i}" for i in range(N)], feats, mask, gt, close.astype(np.float32), valid_index, test_index)


def make_synthetic_hypergraph(N=12):
    from hypershift.data.hypergraph import Hypergraph  # lazy: Task 3 creates this module
    return Hypergraph(N, ((0, 1, 2), (2, 3, 4, 5), (6, 7), (8, 9, 10)))  # node 11 isolated


@pytest.fixture
def synthetic_market():
    return make_synthetic_market()


@pytest.fixture
def synthetic_hypergraph():
    return make_synthetic_hypergraph()
```

(`Hypergraph` comes from Task 3. The import sits inside `make_synthetic_hypergraph`, so `tests/test_rsr.py` runs before Task 3 exists.)

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_rsr.py -v -m "not data"`, then `pytest tests/test_rsr.py -v -m data`
Expected: all PASS. If `test_real_shapes` fails, go to **D2**.

- [ ] **Step 7: Commit**

```bash
git add scripts/download_rsr.sh src/hypershift/data/rsr.py tests/test_rsr.py tests/conftest.py
git commit -m "feat(data): RSR loader with train-only renormalization"
```

---

### Task 3: Hypergraph construction and transforms

**Files:**
- Create: `src/hypershift/data/hypergraph.py`, `src/hypershift/data/universe.py`, `tests/test_hypergraph.py`

**Interfaces:**
- Consumes: `MarketData` (Task 2).
- Produces (all in `hypershift.data.hypergraph`):
  - `Hypergraph(num_nodes: int, edges: tuple[tuple[int, ...], ...])` (frozen dataclass) with methods:
    - `.incidence() -> (node_idx, edge_idx)` (np.int64 arrays)
    - `.node_degree() -> np.ndarray`
    - `.edge_sizes() -> np.ndarray`
    - `.to_torch(device) -> TorchHypergraph`
  - `TorchHypergraph` fields: `node_idx`, `edge_idx` (LongTensor [P]), `num_nodes`, `num_edges`, `has_edge` (BoolTensor [N]), `edge_size` (FloatTensor [E]).
  - Construction and transform functions:
    - `canonical(edges, min_size=2) -> tuple[tuple[int, ...], ...]`
    - `industry_hyperedges(rel) -> list[tuple]`
    - `wiki_hyperedges(rel) -> list[tuple]`
    - `build_rsr_hypergraph(root, market, sources=("industry","wiki"), cache=True) -> Hypergraph`. It caches the edge list in `<root>/hypergraph_cache/<market>_<sources>.json`, because the NYSE industry tensor is 2.6 GB and would otherwise be reloaded on every run.
    - `decompose(hg, mode, size) -> Hypergraph`, where mode ∈ {"none","large_first","small_first"}
    - `clique_expand(hg) -> Hypergraph`
    - `drop_hub_edges(hg, min_degree) -> Hypergraph`
    - `hub_schedule(hg, ranks=(1,5,20,100)) -> list[int]`
    - `random_like(hg, seed) -> Hypergraph`
    - `correlation_hyperedges(returns, mask, n_clusters) -> Hypergraph`
    - `induced_subgraph(hg, keep) -> Hypergraph`
  - `hypershift.data.universe.select_universe(data, hg, size, seed) -> (MarketData, Hypergraph)`

- [ ] **Step 1: Write the failing tests** `tests/test_hypergraph.py`

```python
from pathlib import Path
import numpy as np
import pytest
from hypershift.data.hypergraph import (
    Hypergraph, canonical, clique_expand, correlation_hyperedges, decompose, drop_hub_edges,
    hub_schedule, induced_subgraph, industry_hyperedges, random_like, wiki_hyperedges,
    build_rsr_hypergraph,
)
from hypershift.data.universe import select_universe

REAL = Path("data/raw/rsr/data")


def _industry_rel():
    N = 5
    rel = np.zeros((N, N, 3), int)
    for k, mem in enumerate([[0, 1, 2], [3, 4]]):
        for i in mem:
            for j in mem:
                rel[i, j, k] = 1
    for i in range(N):
        rel[i, i, -1] = 1
    return rel


def test_industry_hyperedges():
    assert sorted(industry_hyperedges(_industry_rel())) == [(0, 1, 2), (3, 4)]


def test_wiki_star_hyperedges_ignore_self_channel():
    N = 5
    rel = np.zeros((N, N, 3), int)
    rel[0, 1, 0] = rel[0, 2, 0] = 1
    rel[3, 4, 1] = 1
    rel[2, 2, 0] = 1                  # diagonal inside a real channel must be ignored
    for i in range(N):
        rel[i, i, -1] = 1
    assert sorted(canonical(wiki_hyperedges(rel))) == [(0, 1, 2), (3, 4)]


def test_canonical_dedup_and_min_size():
    assert canonical([(2, 1), (1, 2), (3,), (4, 5, 6)]) == ((1, 2), (4, 5, 6))


def test_incidence_and_degree():
    hg = Hypergraph(4, ((0, 1, 2), (2, 3)))
    node, edge = hg.incidence()
    assert node.tolist() == [0, 1, 2, 2, 3] and edge.tolist() == [0, 0, 0, 1, 1]
    assert hg.node_degree().tolist() == [1, 1, 2, 1]
    assert hg.edge_sizes().tolist() == [3, 2]


def test_decompose_modes():
    hg = Hypergraph(6, ((0, 1, 2), (3, 4), (0, 1, 2, 5)))
    lf = decompose(hg, "large_first", 3)          # only size > 3 split
    assert (0, 1, 2) in lf.edges and (0, 5) in lf.edges and (0, 1, 2, 5) not in lf.edges
    sf = decompose(hg, "small_first", 3)          # size <= 3 split
    assert (0, 1) in sf.edges and (0, 1, 2) not in sf.edges and (0, 1, 2, 5) in sf.edges
    assert decompose(hg, "none", 0) == hg
    ce = clique_expand(hg)
    assert all(len(e) == 2 for e in ce.edges)
    assert len(ce.edges) == len({(0, 1), (0, 2), (1, 2), (3, 4), (0, 5), (1, 5), (2, 5)})


def test_drop_hub_edges_and_schedule():
    hg = Hypergraph(5, ((0, 1), (0, 2), (0, 3), (3, 4)))
    assert hg.node_degree()[0] == 3
    out = drop_hub_edges(hg, 3)
    assert out.edges == ((3, 4),)
    assert drop_hub_edges(hg, 0) == hg
    sched = hub_schedule(hg, ranks=(1, 2))
    assert sched[0] == 3 and sched[-1] == 2 and sched == sorted(sched, reverse=True)


def test_random_like_preserves_sizes():
    hg = Hypergraph(20, ((0, 1, 2), (3, 4), (5, 6, 7, 8)))
    r = random_like(hg, seed=0)
    assert sorted(r.edge_sizes().tolist()) == [2, 3, 4]
    assert r != hg


def test_correlation_hyperedges_groups_correlated():
    rng = np.random.default_rng(0)
    f1, f2 = rng.normal(size=200), rng.normal(size=200)
    ret = np.stack([f1 + 0.01 * rng.normal(size=200) for _ in range(3)]
                   + [f2 + 0.01 * rng.normal(size=200) for _ in range(3)])
    hg = correlation_hyperedges(ret, np.ones_like(ret), n_clusters=2)
    assert sorted(hg.edges) == [(0, 1, 2), (3, 4, 5)]


def test_induced_subgraph_and_universe(synthetic_market, synthetic_hypergraph):
    sub = induced_subgraph(Hypergraph(5, ((0, 1, 2), (2, 3, 4))), np.array([1, 2, 4]))
    assert sub.num_nodes == 3 and sub.edges == ((0, 1), (1, 2))
    d, h = select_universe(synthetic_market, synthetic_hypergraph, size=6, seed=0)
    assert d.num_nodes == 6 and h.num_nodes == 6
    d2, _ = select_universe(synthetic_market, synthetic_hypergraph, size=6, seed=0)
    assert d.tickers == d2.tickers  # deterministic


def test_build_rsr_hypergraph_cache(tmp_path):
    (tmp_path / "relation" / "sector_industry").mkdir(parents=True)
    (tmp_path / "relation" / "wikidata").mkdir(parents=True)
    np.save(tmp_path / "relation" / "sector_industry" / "NYSE_industry_relation.npy", _industry_rel())
    wiki = np.zeros((5, 5, 2), int)
    wiki[0, 3, 0] = 1
    np.save(tmp_path / "relation" / "wikidata" / "NYSE_wiki_relation.npy", wiki)
    first = build_rsr_hypergraph(tmp_path, "NYSE")
    assert first.edges == ((0, 1, 2), (3, 4), (0, 3))
    assert list((tmp_path / "hypergraph_cache").glob("*.json"))
    (tmp_path / "relation" / "sector_industry" / "NYSE_industry_relation.npy").unlink()
    assert build_rsr_hypergraph(tmp_path, "NYSE") == first          # served from the cache
    assert build_rsr_hypergraph(tmp_path, "NYSE", ("wiki",), cache=False).edges == ((0, 3),)


@pytest.mark.data
@pytest.mark.skipif(not REAL.exists(), reason="RSR data not downloaded")
def test_real_nyse_hypergraph_stats():
    hg = build_rsr_hypergraph(REAL, "NYSE")
    assert hg.num_nodes == 1737
    assert 250 <= len(hg.edges) <= 400             # prototype: 312
    assert hg.edge_sizes().max() == 500            # matches paper Fig 3a axis
    assert 20 <= hg.node_degree().max() <= 60      # prototype: 37
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_hypergraph.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/data/hypergraph.py`

```python
"""Stock hypergraphs: construction from RSR relations (paper appendix B) and ablation transforms."""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Hypergraph:
    num_nodes: int
    edges: tuple[tuple[int, ...], ...]

    def incidence(self) -> tuple[np.ndarray, np.ndarray]:
        node = np.fromiter((v for e in self.edges for v in e), dtype=np.int64)
        edge = np.repeat(np.arange(len(self.edges), dtype=np.int64), [len(e) for e in self.edges])
        return node, edge

    def node_degree(self) -> np.ndarray:
        deg = np.zeros(self.num_nodes, dtype=np.int64)
        node, _ = self.incidence()
        np.add.at(deg, node, 1)
        return deg

    def edge_sizes(self) -> np.ndarray:
        return np.array([len(e) for e in self.edges], dtype=np.int64)

    def to_torch(self, device) -> "TorchHypergraph":
        node, edge = self.incidence()
        return TorchHypergraph(
            node_idx=torch.as_tensor(node, device=device),
            edge_idx=torch.as_tensor(edge, device=device),
            num_nodes=self.num_nodes,
            num_edges=len(self.edges),
            has_edge=torch.as_tensor(self.node_degree() > 0, device=device),
            edge_size=torch.as_tensor(self.edge_sizes(), dtype=torch.float32, device=device),
        )


@dataclass
class TorchHypergraph:
    node_idx: torch.Tensor
    edge_idx: torch.Tensor
    num_nodes: int
    num_edges: int
    has_edge: torch.Tensor
    edge_size: torch.Tensor


def canonical(edges, min_size: int = 2) -> tuple[tuple[int, ...], ...]:
    seen, out = set(), []
    for e in edges:
        t = tuple(sorted({int(v) for v in e}))
        if len(t) >= min_size and t not in seen:
            seen.add(t)
            out.append(t)
    return tuple(out)


def industry_hyperedges(rel: np.ndarray) -> list[tuple[int, ...]]:
    """rel [N,N,R]; last channel is the self-relation and is ignored."""
    out = []
    for k in range(rel.shape[2] - 1):
        members = np.nonzero(rel[:, :, k].sum(axis=1) > 0)[0]
        out.append(tuple(int(v) for v in members))
    return out


def wiki_hyperedges(rel: np.ndarray) -> list[tuple[int, ...]]:
    """Star hyperedge per (source stock, wiki relation channel): {i} U {j : rel[i,j,k]=1}."""
    out = []
    for k in range(rel.shape[2] - 1):
        a = rel[:, :, k].copy()
        np.fill_diagonal(a, 0)
        for i in np.nonzero(a.sum(axis=1) > 0)[0]:
            out.append((int(i), *(int(j) for j in np.nonzero(a[i])[0])))
    return out


def build_rsr_hypergraph(root, market: str, sources=("industry", "wiki"), cache: bool = True) -> Hypergraph:
    root = Path(root)
    cache_file = root / "hypergraph_cache" / f"{market}_{'-'.join(sorted(sources))}.json"
    if cache and cache_file.exists():
        d = json.loads(cache_file.read_text())
        return Hypergraph(d["num_nodes"], tuple(tuple(e) for e in d["edges"]))
    edges: list[tuple[int, ...]] = []
    n = None
    if "industry" in sources:
        rel = np.load(root / "relation" / "sector_industry" / f"{market}_industry_relation.npy")
        n = rel.shape[0]
        edges += industry_hyperedges(rel)
    if "wiki" in sources:
        rel = np.load(root / "relation" / "wikidata" / f"{market}_wiki_relation.npy")
        n = rel.shape[0]
        edges += wiki_hyperedges(rel)
    if n is None:
        raise ValueError("sources must include industry and/or wiki")
    hg = Hypergraph(n, canonical(edges))
    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"num_nodes": n, "edges": [list(e) for e in hg.edges]}))
    return hg


def decompose(hg: Hypergraph, mode: str, size: int) -> Hypergraph:
    """Replace selected hyperedges by all their pairwise edges (paper Fig 3a).

    large_first: split hyperedges with |e| > size (matches the Fig 3a axis 500 -> 3).
    small_first: split hyperedges with |e| <= size (matches the text "in increasing order").
    """
    if mode == "none":
        return hg
    out: list[tuple[int, ...]] = []
    for e in hg.edges:
        split = len(e) > size if mode == "large_first" else len(e) <= size
        if mode not in ("large_first", "small_first"):
            raise ValueError(mode)
        if split and len(e) > 2:
            out.extend(itertools.combinations(e, 2))
        else:
            out.append(e)
    return Hypergraph(hg.num_nodes, canonical(out))


def clique_expand(hg: Hypergraph) -> Hypergraph:
    max_size = int(hg.edge_sizes().max()) if hg.edges else 0
    return decompose(hg, "small_first", max_size)


def drop_hub_edges(hg: Hypergraph, min_degree: int) -> Hypergraph:
    """Remove every hyperedge touching a node whose degree >= min_degree (paper Fig 3b). 0 = keep all."""
    if min_degree <= 0:
        return hg
    hubs = set(np.nonzero(hg.node_degree() >= min_degree)[0].tolist())
    return Hypergraph(hg.num_nodes, tuple(e for e in hg.edges if hubs.isdisjoint(e)))


def hub_schedule(hg: Hypergraph, ranks=(1, 5, 20, 100)) -> list[int]:
    deg = np.sort(hg.node_degree())[::-1]
    th = [int(deg[min(r, len(deg)) - 1]) for r in ranks] + [2]
    out: list[int] = []
    for t in th:
        if t >= 2 and t not in out:
            out.append(t)
    return sorted(out, reverse=True)


def random_like(hg: Hypergraph, seed: int) -> Hypergraph:
    """Same hyperedge sizes; members drawn proportional to original node degree (grouping control)."""
    rng = np.random.default_rng(seed)
    deg = hg.node_degree().astype(float)
    p = deg / deg.sum()
    edges = tuple(
        tuple(sorted(int(v) for v in rng.choice(hg.num_nodes, size=len(e), replace=False, p=p)))
        for e in hg.edges
    )
    return Hypergraph(hg.num_nodes, edges)


def correlation_hyperedges(returns: np.ndarray, mask: np.ndarray, n_clusters: int) -> Hypergraph:
    """Average-linkage clusters on 1 - corr(train returns); each cluster (size >= 2) is a hyperedge."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    r = np.where(mask > 0, returns, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.nan_to_num(np.corrcoef(r))
    d = np.clip(1.0 - c, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(z, n_clusters, criterion="maxclust")
    edges = [tuple(np.nonzero(labels == c)[0].tolist()) for c in np.unique(labels)]
    return Hypergraph(returns.shape[0], canonical(edges))


def induced_subgraph(hg: Hypergraph, keep: np.ndarray) -> Hypergraph:
    remap = {int(old): new for new, old in enumerate(np.asarray(keep).tolist())}
    edges = [tuple(remap[v] for v in e if v in remap) for e in hg.edges]
    return Hypergraph(len(remap), canonical(edges))
```

`src/hypershift/data/universe.py`:
```python
import numpy as np

from hypershift.data.hypergraph import Hypergraph, induced_subgraph
from hypershift.data.rsr import MarketData


def select_universe(data: MarketData, hg: Hypergraph, size: int, seed: int) -> tuple[MarketData, Hypergraph]:
    """Uniform random subset of `size` stocks (sorted indices) + induced sub-hypergraph. size<=0 = all."""
    if size <= 0 or size >= data.num_nodes:
        return data, hg
    idx = np.sort(np.random.default_rng(seed).choice(data.num_nodes, size=size, replace=False))
    return data.subset(idx), induced_subgraph(hg, idx)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_hypergraph.py -v`
Expected: all PASS. The data-marked test runs only if the data exists; if it fails, go to **D3**.

- [ ] **Step 5: Commit**

```bash
git add src/hypershift/data/hypergraph.py src/hypershift/data/universe.py tests/test_hypergraph.py
git commit -m "feat(data): hypergraph construction, decomposition, hub removal, random/corr controls"
```

---

### Task 4: Poincaré-ball math

**Files:**
- Create: `src/hypershift/geometry/poincare.py`, `tests/test_poincare.py`

**Interfaces:**
- Produces (all operate on the last dim, c = 1):
  - `project(x)`, `artanh(x)`, `expmap0(v)`, `logmap0(y)`
  - `mobius_add(x, y)`, `poincare_dist(x, y) -> [...]` (last dim reduced)
  - `lambda_x(x) -> [...,1]`, `mobius_scalar(r: float, x)`
  - `gyromidpoint(x, node_idx, edge_idx, num_edges) -> [..., E, D]`, where x is `[..., N, D]`
  - Constant `MAX_NORM = 1 - 1e-5`

- [ ] **Step 1: Write the failing tests** `tests/test_poincare.py`

```python
import math
import torch
import pytest
from hypershift.geometry.poincare import (
    MAX_NORM, expmap0, gyromidpoint, lambda_x, logmap0, mobius_add, mobius_scalar,
    poincare_dist, project,
)

torch.manual_seed(0)


def rand_ball(*shape, scale=0.5):
    return expmap0(torch.randn(*shape, dtype=torch.float64) * scale)


def test_exp_log_inverse():
    v = torch.randn(10, 4, dtype=torch.float64)
    torch.testing.assert_close(logmap0(expmap0(v)), v, atol=1e-6, rtol=1e-6)


def test_project_bounds():
    x = torch.tensor([[2.0, 0.0], [0.1, 0.1]])
    assert project(x).norm(dim=-1).max() <= MAX_NORM + 1e-7
    torch.testing.assert_close(project(x)[1], x[1])


def test_mobius_identity_and_left_cancellation():
    x, y = rand_ball(5, 3), rand_ball(5, 3)
    torch.testing.assert_close(mobius_add(x, torch.zeros_like(x)), x)
    torch.testing.assert_close(mobius_add(-x, mobius_add(x, y)), y, atol=1e-9, rtol=1e-7)


def test_distance_properties():
    x, y = rand_ball(6, 3), rand_ball(6, 3)
    torch.testing.assert_close(poincare_dist(x, y), poincare_dist(y, x))
    assert torch.all(poincare_dist(x, x) < 1e-6)
    zero = torch.zeros_like(x)
    torch.testing.assert_close(poincare_dist(zero, x), 2 * torch.atanh(x.norm(dim=-1)))


def test_mobius_scalar():
    x = rand_ball(4, 3)
    torch.testing.assert_close(mobius_scalar(1.0, x), x)
    two = mobius_scalar(2.0, x)
    torch.testing.assert_close(two, mobius_add(x, x), atol=1e-9, rtol=1e-7)


def test_lambda():
    x = torch.tensor([[0.0, 0.0], [0.6, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(lambda_x(x).squeeze(-1), torch.tensor([2.0, 2 / (1 - 0.36)], dtype=torch.float64))


def test_gyromidpoint_single_point_is_identity_and_symmetric_pair_is_origin():
    x = rand_ball(3, 4)                          # nodes 0,1,2
    node = torch.tensor([0, 1, 2])
    edge = torch.tensor([0, 1, 2])               # each node its own hyperedge
    torch.testing.assert_close(gyromidpoint(x, node, edge, 3), x, atol=1e-9, rtol=1e-7)
    p = rand_ball(1, 4)
    pair = torch.cat([p, -p])
    m = gyromidpoint(pair, torch.tensor([0, 1]), torch.tensor([0, 0]), 1)
    assert m.norm() < 1e-9


def test_gyromidpoint_batched_leading_dims():
    x = rand_ball(2, 7, 5, 3)                    # [B,T,N,D]
    node = torch.tensor([0, 1, 2, 2, 3, 4])
    edge = torch.tensor([0, 0, 0, 1, 1, 1])
    m = gyromidpoint(x, node, edge, 2)
    assert m.shape == (2, 7, 2, 3)
    assert m.norm(dim=-1).max() < 1


def test_near_boundary_finite_grads():
    v = (torch.randn(8, 4) * 20).requires_grad_(True)   # exp map lands at the boundary
    x = expmap0(v)
    y = expmap0(torch.randn(8, 4))
    loss = poincare_dist(x, y).sum() + logmap0(x).pow(2).sum()
    loss.backward()
    assert torch.isfinite(v.grad).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_poincare.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/geometry/poincare.py`

```python
"""Poincare ball (curvature -1) operations. All functions act on the last dimension."""
from __future__ import annotations

import torch

MAX_NORM = 1.0 - 1e-5
_MIN = 1e-15


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x.norm(dim=-1, keepdim=True).clamp_min(_MIN)


def project(x: torch.Tensor) -> torch.Tensor:
    n = _norm(x)
    return torch.where(n > MAX_NORM, x / n * MAX_NORM, x)


def artanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def expmap0(v: torch.Tensor) -> torch.Tensor:
    n = _norm(v)
    return project(torch.tanh(n) * v / n)


def logmap0(y: torch.Tensor) -> torch.Tensor:
    n = _norm(y)
    return artanh(n) * y / n


def mobius_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xy = (x * y).sum(-1, keepdim=True)
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / den.clamp_min(_MIN)


def poincare_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 2 * artanh(mobius_add(-x, y).norm(dim=-1))


def lambda_x(x: torch.Tensor) -> torch.Tensor:
    return 2.0 / (1.0 - (x * x).sum(-1, keepdim=True)).clamp_min(_MIN)


def mobius_scalar(r: float, x: torch.Tensor) -> torch.Tensor:
    n = _norm(x)
    return project(torch.tanh(r * artanh(n)) * x / n)


def gyromidpoint(x: torch.Tensor, node_idx: torch.Tensor, edge_idx: torch.Tensor, num_edges: int) -> torch.Tensor:
    """Paper eq. 13. x: [..., N, D] -> [..., E, D]."""
    lam = lambda_x(x)                                           # [..., N, 1]
    lead = x.shape[:-2]
    num = x.new_zeros(*lead, num_edges, x.shape[-1]).index_add(-2, edge_idx, (lam * x).index_select(-2, node_idx))
    den = x.new_zeros(*lead, num_edges, 1).index_add(-2, edge_idx, (lam - 1).index_select(-2, node_idx))
    return mobius_scalar(0.5, num / den.clamp_min(_MIN))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_poincare.py -v`
Expected: all PASS. Any failure: go to **D4**. Do not loosen tolerances except to float32-appropriate ones, and write down the reason.

- [ ] **Step 5: Commit**

```bash
git add src/hypershift/geometry/poincare.py tests/test_poincare.py
git commit -m "feat(geometry): Poincare ball ops incl. gyromidpoint"
```

---

### Task 5: Layers: Poincaré FC, β-concat, temporal convolutions, hypergraph attention

**Files:**
- Create: `src/hypershift/models/layers.py`, `src/hypershift/models/attention.py`, `tests/test_layers.py`, `tests/test_attention.py`

**Interfaces:**
- Consumes: `hypershift.geometry.poincare` (Task 4) and `TorchHypergraph` (Task 3).
- Produces:
  - In `layers`:
    - `beta_fn(a, b) -> float`
    - `beta_concat(xs: list[Tensor]) -> Tensor`
    - `PoincareLinear(in_dim, out_dim)`
    - `HypTemporalConv(in_dim, out_dim, kernel)`: input `[B,T,N,C]` on ball, output `[B,T//K,N,out]` on ball.
    - `EucTemporalConv(in_dim, out_dim, kernel, activation=True)`: same shapes, Euclidean.
  - In `attention`:
    - `segment_softmax(scores[...,P], index[P], num_segments) -> [...,P]`
    - `HypHypergraphAttention(dim, score="mobius", dist="mult")` and `EucHypergraphAttention(dim, score="mobius", dist="mult")`.
    - `forward(u [B,T,N,D], hg: TorchHypergraph) -> [B,T,N,D]`
    - `score ∈ {"mobius","concat"}`, `dist ∈ {"mult","neg","off"}`

- [ ] **Step 1: Write failing tests** `tests/test_layers.py`

```python
import torch
from hypershift.geometry.poincare import expmap0
from hypershift.models.layers import EucTemporalConv, HypTemporalConv, PoincareLinear, beta_concat, beta_fn

torch.manual_seed(0)


def test_beta_fn_known_values():
    assert abs(beta_fn(0.5, 0.5) - 3.141592653589793) < 1e-9
    assert abs(beta_fn(1.0, 1.0) - 1.0) < 1e-12


def test_poincare_linear_inside_ball_and_grads():
    fc = PoincareLinear(6, 4)
    x = expmap0(torch.randn(10, 6) * 3).requires_grad_(True)
    y = fc(x)
    assert y.shape == (10, 4) and y.norm(dim=-1).max() < 1
    y.sum().backward()
    assert torch.isfinite(fc.z.grad).all() and torch.isfinite(fc.r.grad).all()


def test_beta_concat_single_input_identity():
    x = expmap0(torch.randn(3, 4))
    torch.testing.assert_close(beta_concat([x]), x, atol=1e-6, rtol=1e-5)


def test_hyp_temporal_conv_matches_manual_window():
    conv = HypTemporalConv(in_dim=3, out_dim=5, kernel=4)
    x = expmap0(torch.randn(2, 8, 6, 3))                  # B=2, T=8, N=6, C=3
    y = conv(x)
    assert y.shape == (2, 2, 6, 5)
    manual = conv.fc(beta_concat([x[:, 4 + s] for s in range(4)]))   # second window, [B,N,5]
    torch.testing.assert_close(y[:, 1], manual, atol=1e-5, rtol=1e-4)


def test_euc_temporal_conv_shapes():
    conv = EucTemporalConv(3, 5, 4, activation=False)
    assert conv(torch.randn(2, 16, 6, 3)).shape == (2, 4, 6, 5)
```

`tests/test_attention.py`:
```python
import pytest
import torch
from hypershift.data.hypergraph import Hypergraph
from hypershift.geometry.poincare import expmap0
from hypershift.models.attention import EucHypergraphAttention, HypHypergraphAttention, segment_softmax

torch.manual_seed(0)
HG = Hypergraph(6, ((0, 1, 2), (2, 3), (1, 2, 3, 4)))   # node 5 isolated


def test_segment_softmax_sums_to_one():
    s = torch.randn(2, 7)
    idx = torch.tensor([0, 0, 1, 1, 1, 3, 3])
    a = segment_softmax(s, idx, 4)
    sums = torch.zeros(2, 4).index_add(-1, idx, a)
    torch.testing.assert_close(sums[:, [0, 1, 3]], torch.ones(2, 3))


@pytest.mark.parametrize("cls", [HypHypergraphAttention, EucHypergraphAttention])
@pytest.mark.parametrize("score", ["mobius", "concat"])
@pytest.mark.parametrize("dist", ["mult", "neg", "off"])
def test_attention_shapes_and_finite(cls, score, dist):
    layer = cls(4, score=score, dist=dist)
    u = torch.randn(2, 3, 6, 4) * 0.5
    if cls is HypHypergraphAttention:
        u = expmap0(u)
    out = layer(u, HG.to_torch("cpu"))
    assert out.shape == u.shape and torch.isfinite(out).all()
    out.sum().backward()


@pytest.mark.parametrize("cls", [HypHypergraphAttention, EucHypergraphAttention])
def test_isolated_node_passthrough(cls):
    layer = cls(4)
    u = expmap0(torch.randn(1, 1, 6, 4) * 0.5)
    out = layer(u, HG.to_torch("cpu"))
    torch.testing.assert_close(out[..., 5, :], u[..., 5, :])


def test_hyp_attention_permutation_equivariant():
    layer = HypHypergraphAttention(4)
    u = expmap0(torch.randn(1, 2, 6, 4) * 0.5)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])              # new position i holds old node perm[i]
    inv = torch.argsort(perm)
    hg_p = Hypergraph(6, tuple(tuple(int(inv[v]) for v in e) for e in HG.edges))
    out = layer(u, HG.to_torch("cpu"))
    out_p = layer(u[..., perm, :], hg_p.to_torch("cpu"))
    torch.testing.assert_close(out_p, out[..., perm, :], atol=1e-5, rtol=1e-4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_layers.py tests/test_attention.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/models/layers.py`

```python
"""Poincare FC (HNN++), beta-concatenation, hyperbolic and Euclidean temporal convolutions (paper eq. 9-12)."""
from __future__ import annotations

import math

import torch
from torch import nn

from hypershift.geometry.poincare import expmap0, lambda_x, logmap0, project


def beta_fn(a: float, b: float) -> float:
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def beta_concat(xs: list[torch.Tensor]) -> torch.Tensor:
    n = sum(x.shape[-1] for x in xs)
    bn = beta_fn(n / 2, 0.5)
    parts = [logmap0(x) * (bn / beta_fn(x.shape[-1] / 2, 0.5)) for x in xs]
    return expmap0(torch.cat(parts, dim=-1))


class PoincareLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.z = nn.Parameter(torch.randn(in_dim, out_dim) * (2 * in_dim * out_dim) ** -0.5)
        self.r = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = lambda_x(x)                                           # [..., 1]
        z_norm = self.z.norm(dim=0).clamp_min(1e-15)                # [out]
        inner = (x @ self.z) / z_norm                               # <x, z_k/|z_k|>
        v = 2 * z_norm * torch.asinh(lam * inner * torch.cosh(2 * self.r) - (lam - 1) * torch.sinh(2 * self.r))
        w = torch.sinh(v.clamp(-15, 15))
        return project(w / (1 + torch.sqrt(1 + (w * w).sum(-1, keepdim=True))))


def _windows(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """[B,T,N,C] -> [B,T//K,N,K*C] (non-overlapping windows, time-major inside the window)."""
    b, t, n, c = x.shape
    if t % kernel:
        raise ValueError(f"sequence length {t} not divisible by kernel {kernel}")
    return x.reshape(b, t // kernel, kernel, n, c).permute(0, 1, 3, 2, 4).reshape(b, t // kernel, n, kernel * c)


class HypTemporalConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.scale = beta_fn(kernel * in_dim / 2, 0.5) / beta_fn(in_dim / 2, 0.5)
        self.fc = PoincareLinear(kernel * in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(expmap0(_windows(logmap0(x) * self.scale, self.kernel)))


class EucTemporalConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, activation: bool = True):
        super().__init__()
        self.kernel = kernel
        self.activation = activation
        self.lin = nn.Linear(kernel * in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(_windows(x, self.kernel))
        return torch.relu(y) if self.activation else y
```

`src/hypershift/models/attention.py`:
```python
"""Distance-aware hypergraph attention (DHHAN, paper eq. 13-15) and its Euclidean mirror."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hypershift.data.hypergraph import TorchHypergraph
from hypershift.geometry.poincare import expmap0, gyromidpoint, logmap0, mobius_add, poincare_dist
from hypershift.models.layers import PoincareLinear


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    idx = index.expand_as(scores)
    shape = (*scores.shape[:-1], num_segments)
    mx = scores.new_full(shape, float("-inf")).scatter_reduce(-1, idx, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - mx.detach().gather(-1, idx))
    den = scores.new_zeros(shape).index_add(-1, index, ex)
    return ex / den.gather(-1, idx).clamp_min(1e-16)


class _Base(nn.Module):
    def __init__(self, dim: int, score: str, dist: str):
        super().__init__()
        if score not in ("mobius", "concat") or dist not in ("mult", "neg", "off"):
            raise ValueError((score, dist))
        self.score, self.dist = score, dist
        self.a = nn.Parameter(torch.randn(dim if score == "mobius" else 2 * dim) * dim ** -0.5)
        self.gamma = nn.Parameter(torch.zeros(()))

    def _combine(self, base: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.dist == "mult":
            return base * d
        if self.dist == "neg":
            return F.leaky_relu(base, 0.2) - F.softplus(self.gamma) * d
        return F.leaky_relu(base, 0.2)


class HypHypergraphAttention(_Base):
    def __init__(self, dim: int, score: str = "mobius", dist: str = "mult"):
        super().__init__(dim, score, dist)
        self.fc = PoincareLinear(dim, dim)

    def forward(self, u: torch.Tensor, hg: TorchHypergraph) -> torch.Tensor:
        z = gyromidpoint(u, hg.node_idx, hg.edge_idx, hg.num_edges)          # eq 13
        uj = u.index_select(-2, hg.node_idx)
        zi = z.index_select(-2, hg.edge_idx)
        if self.score == "mobius":
            base = mobius_add(uj, zi) @ self.a
        else:
            base = torch.cat([logmap0(uj), logmap0(zi)], dim=-1) @ self.a
        alpha = segment_softmax(self._combine(base, poincare_dist(uj, zi)), hg.node_idx, hg.num_nodes)  # eq 14
        msg = logmap0(self.fc(z)).index_select(-2, hg.edge_idx) * alpha.unsqueeze(-1)
        agg = u.new_zeros(u.shape).index_add(-2, hg.node_idx, msg)
        out = expmap0(F.relu(agg))                                             # eq 15
        return torch.where(hg.has_edge[:, None], out, u)


class EucHypergraphAttention(_Base):
    def __init__(self, dim: int, score: str = "mobius", dist: str = "mult"):
        super().__init__(dim, score, dist)
        self.fc = nn.Linear(dim, dim)

    def forward(self, u: torch.Tensor, hg: TorchHypergraph) -> torch.Tensor:
        lead = u.shape[:-2]
        z = u.new_zeros(*lead, hg.num_edges, u.shape[-1]).index_add(-2, hg.edge_idx, u.index_select(-2, hg.node_idx))
        z = z / hg.edge_size[:, None]
        uj = u.index_select(-2, hg.node_idx)
        zi = z.index_select(-2, hg.edge_idx)
        base = (uj + zi) @ self.a if self.score == "mobius" else torch.cat([uj, zi], dim=-1) @ self.a
        alpha = segment_softmax(self._combine(base, (uj - zi).norm(dim=-1)), hg.node_idx, hg.num_nodes)
        msg = self.fc(z).index_select(-2, hg.edge_idx) * alpha.unsqueeze(-1)
        out = F.relu(u.new_zeros(u.shape).index_add(-2, hg.node_idx, msg))
        return torch.where(hg.has_edge[:, None], out, u)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_layers.py tests/test_attention.py -v`
Expected: all PASS. On failure go to **D4**.

- [ ] **Step 5: Commit**

```bash
git add src/hypershift/models/layers.py src/hypershift/models/attention.py tests/test_layers.py tests/test_attention.py
git commit -m "feat(models): Poincare FC, beta-concat, temporal convs, distance-aware hypergraph attention"
```

---

### Task 6: THINK model with geometry and structure switches

**Files:**
- Create: `src/hypershift/models/think.py`, `tests/test_think.py`

**Interfaces:**
- Consumes: Task 5 layers and `TorchHypergraph`.
- Produces: `THINK(in_dim=5, hidden=32, seq=16, kernel=4, temporal="hyp", spatial="hyp", structure="hyper", attn_score="mobius", attn_dist="mult", out_dim=1)`.
  - `forward(x [B,N,T,C] float, hg: TorchHypergraph) -> [B,N]` if `out_dim == 1`, otherwise `[B,N,out_dim]`.
  - `temporal ∈ {"hyp","euc"}`, `spatial ∈ {"hyp","euc"}`, `structure ∈ {"hyper","clique","none"}`.
  - `"clique"` only changes the hypergraph that is passed in (Task 8). `"none"` skips the spatial layer.

- [ ] **Step 1: Write the failing tests** `tests/test_think.py`

```python
import itertools
import pytest
import torch
from hypershift.data.hypergraph import Hypergraph
from hypershift.models.think import THINK

HG = Hypergraph(7, ((0, 1, 2), (2, 3), (4, 5))).to_torch("cpu")  # node 6 isolated


@pytest.mark.parametrize("temporal,spatial,structure",
                         list(itertools.product(["hyp", "euc"], ["hyp", "euc"], ["hyper", "none"])))
def test_forward_backward_all_variants(temporal, spatial, structure):
    torch.manual_seed(0)
    m = THINK(in_dim=5, hidden=8, seq=16, kernel=4, temporal=temporal, spatial=spatial, structure=structure)
    x = torch.rand(3, 7, 16, 5)
    y = m(x, HG)
    assert y.shape == (3, 7) and torch.isfinite(y).all()
    y.pow(2).mean().backward()
    for name, p in m.named_parameters():
        if name.endswith("gamma"):          # only used when attn_dist == "neg"
            continue
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_structure_none_ignores_graph():
    torch.manual_seed(0)
    m = THINK(hidden=8, structure="none")
    x = torch.rand(1, 7, 16, 5)
    empty = Hypergraph(7, ()).to_torch("cpu")
    torch.testing.assert_close(m(x, HG), m(x, empty))


def test_graph_changes_output():
    torch.manual_seed(0)
    m = THINK(hidden=8)
    x = torch.rand(1, 7, 16, 5)
    empty = Hypergraph(7, ()).to_torch("cpu")
    assert not torch.allclose(m(x, HG), m(x, empty))


def test_multiclass_head():
    m = THINK(hidden=8, out_dim=3)
    assert m(torch.rand(2, 7, 16, 5), HG).shape == (2, 7, 3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_think.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/models/think.py`

```python
"""THINK (paper eq. 17): log0(TConv2(DHHAN(TConv1(exp0(X)), G))) with ablation switches."""
from __future__ import annotations

import torch
from torch import nn

from hypershift.data.hypergraph import TorchHypergraph
from hypershift.geometry.poincare import expmap0, logmap0
from hypershift.models.attention import EucHypergraphAttention, HypHypergraphAttention
from hypershift.models.layers import EucTemporalConv, HypTemporalConv


class THINK(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 32, seq: int = 16, kernel: int = 4,
                 temporal: str = "hyp", spatial: str = "hyp", structure: str = "hyper",
                 attn_score: str = "mobius", attn_dist: str = "mult", out_dim: int = 1):
        super().__init__()
        if seq % kernel:
            raise ValueError("seq must be a multiple of kernel")
        k2 = seq // kernel
        self.temporal_hyp = temporal == "hyp"
        self.spatial_hyp = spatial == "hyp"
        self.use_spatial = structure != "none"
        if self.temporal_hyp:
            self.tconv1 = HypTemporalConv(in_dim, hidden, kernel)
            self.tconv2 = HypTemporalConv(hidden, out_dim, k2)
        else:
            self.tconv1 = EucTemporalConv(in_dim, hidden, kernel, activation=True)
            self.tconv2 = EucTemporalConv(hidden, out_dim, k2, activation=False)
        if self.use_spatial:
            cls = HypHypergraphAttention if self.spatial_hyp else EucHypergraphAttention
            self.spatial = cls(hidden, score=attn_score, dist=attn_dist)

    def forward(self, x: torch.Tensor, hg: TorchHypergraph) -> torch.Tensor:
        h = x.permute(0, 2, 1, 3)                      # [B,T,N,C]
        if self.temporal_hyp:
            h = expmap0(h)
        h = self.tconv1(h)                             # [B,T/K,N,H]
        if self.use_spatial and hg.num_edges > 0:
            if self.spatial_hyp and not self.temporal_hyp:
                h = self.spatial(expmap0(h), hg)
                h = logmap0(h)
            elif self.temporal_hyp and not self.spatial_hyp:
                h = expmap0(self.spatial(logmap0(h), hg))
            else:
                h = self.spatial(h, hg)
        h = self.tconv2(h)                             # [B,1,N,out]
        if self.temporal_hyp:
            h = logmap0(h)
        h = h[:, 0]
        return h.squeeze(-1) if h.shape[-1] == 1 else h
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_think.py -v`
Expected: all PASS. The test skips `spatial.gamma` because it only receives a gradient when `attn_dist == "neg"`. Any other missing gradient means the forward pass is not wired correctly; see D6.

- [ ] **Step 5: Commit**

```bash
git add src/hypershift/models/think.py tests/test_think.py
git commit -m "feat(models): THINK with temporal/spatial geometry and structure switches"
```

---

### Task 7: Metrics (Sharpe, IRR, NDCG, MSE, drawdown, costs)

**Files:**
- Create: `src/hypershift/eval/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces (in all of these, `pred`, `gt` and `mask` have shape `[N, D]`, where D = number of test days):
  - `topk_daily_returns(pred, gt, mask, k=5) -> np.ndarray [D]`
  - `topk_daily_returns_net(pred, gt, mask, k, cost_bps) -> np.ndarray [D]`
  - `sharpe(r, periods_per_year=252) -> float`
  - `irr(r) -> float`: simple sum, the STHAN-SR `btl5 - 1`
  - `cumulative_return(r) -> float`
  - `max_drawdown(r) -> float`
  - `ndcg_at_k(pred, gt, mask, k=5) -> float`
  - `ndcg_sthan_compat(pred, gt, mask, k=5) -> float`
  - `masked_mse(pred, gt, mask) -> float`
  - `evaluate_all(pred, gt, mask, k=5, periods_per_year=252) -> dict` with keys `sr, irr, cumret, mdd, ann_vol, ndcg5, ndcg_sthan, mse, n_days`

- [ ] **Step 1: Write the failing tests** `tests/test_metrics.py`

```python
import math
import numpy as np
import pytest
from hypershift.eval.metrics import (
    cumulative_return, evaluate_all, irr, masked_mse, max_drawdown, ndcg_at_k, sharpe,
    topk_daily_returns, topk_daily_returns_net,
)


def test_topk_picks_highest_predictions():
    pred = np.array([[0.9, 0.1], [0.5, 0.8], [0.1, 0.9]])
    gt = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
    r = topk_daily_returns(pred, gt, np.ones_like(gt), k=2)
    np.testing.assert_allclose(r, [(0.01 + 0.03) / 2, (0.04 + 0.06) / 2])


def test_topk_skips_masked():
    pred = np.array([[9.0], [1.0], [0.5]])
    gt = np.array([[0.5], [0.01], [0.02]])
    mask = np.array([[0.0], [1.0], [1.0]])
    np.testing.assert_allclose(topk_daily_returns(pred, gt, mask, k=1), [0.01])


def test_sharpe_matches_sthan_constant():
    r = np.array([0.01, -0.005, 0.02, 0.0])
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * math.sqrt(252))
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * 15.8745, rel=1e-4)
    assert sharpe(np.zeros(5)) == 0.0


def test_irr_cumret_mdd():
    r = np.array([0.1, -0.5, 0.2])
    assert irr(r) == pytest.approx(-0.2)
    assert cumulative_return(r) == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_ndcg_perfect_and_reversed():
    gt = np.array([[0.05], [0.03], [0.01], [-0.02], [0.0], [0.04]])
    m = np.ones_like(gt)
    assert ndcg_at_k(gt, gt, m, k=5) == pytest.approx(1.0)
    assert ndcg_at_k(-gt, gt, m, k=5) < 0.6


def test_mse_masked():
    assert masked_mse(np.array([[1.0, 5.0]]), np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])) == pytest.approx(1.0)


def test_costs_reduce_returns_by_turnover():
    pred = np.array([[1.0, 1.0], [0.0, 0.0]])
    gt = np.array([[0.01, 0.01], [0.0, 0.0]])
    net = topk_daily_returns_net(pred, gt, np.ones_like(gt), k=1, cost_bps=10)
    np.testing.assert_allclose(net, [0.01 - 2 * 10e-4, 0.01])   # day 1 full turnover, day 2 none


def test_evaluate_all_keys():
    rng = np.random.default_rng(0)
    p, g = rng.normal(size=(20, 30)), rng.normal(0, 0.01, size=(20, 30))
    out = evaluate_all(p, g, np.ones_like(g))
    assert set(out) == {"sr", "irr", "cumret", "mdd", "ann_vol", "ndcg5", "ndcg_sthan", "mse", "n_days"}
    assert out["n_days"] == 30
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/eval/metrics.py`

```python
"""Portfolio and ranking metrics. SR follows STHAN-SR: mean/std(daily top-k return)*sqrt(252), no rf, no costs."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import ndcg_score


def _topk_sets(pred, mask, k):
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d] > 0.5)[0]
        yield d, idx[np.argsort(-pred[idx, d], kind="stable")[:k]]


def topk_daily_returns(pred, gt, mask, k=5) -> np.ndarray:
    out = np.zeros(pred.shape[1])
    for d, top in _topk_sets(pred, mask, k):
        out[d] = gt[top, d].mean() if len(top) else 0.0
    return out


def topk_daily_returns_net(pred, gt, mask, k, cost_bps) -> np.ndarray:
    """Equal-weight top-k; cost = 2 sides * bps * fraction of names replaced (first period = full)."""
    out, prev = np.zeros(pred.shape[1]), set()
    for d, top in _topk_sets(pred, mask, k):
        cur = set(top.tolist())
        turnover = 1.0 if not prev else len(cur - prev) / max(len(cur), 1)
        out[d] = (gt[top, d].mean() if len(top) else 0.0) - 2 * cost_bps * 1e-4 * turnover
        prev = cur
    return out


def sharpe(r, periods_per_year=252) -> float:
    sd = float(np.std(r))
    return 0.0 if sd == 0 else float(np.mean(r)) / sd * math.sqrt(periods_per_year)


def irr(r) -> float:
    return float(np.sum(r))


def cumulative_return(r) -> float:
    return float(np.prod(1 + np.asarray(r)) - 1)


def max_drawdown(r) -> float:
    wealth = np.cumprod(1 + np.asarray(r))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.min(wealth / peak - 1))


def ndcg_at_k(pred, gt, mask, k=5) -> float:
    """Mean over days of NDCG@k with relevance = true return shifted to be >= 0 that day."""
    vals = []
    for d in range(pred.shape[1]):
        idx = np.nonzero(mask[:, d] > 0.5)[0]
        if len(idx) < 2:
            continue
        rel = gt[idx, d] - gt[idx, d].min()
        if rel.max() <= 0:
            continue
        vals.append(ndcg_score(rel[None, :], pred[idx, d][None, :], k=k))
    return float(np.mean(vals)) if vals else 0.0


def ndcg_sthan_compat(pred, gt, mask, k=5) -> float:
    """Re-implementation of the (buggy) STHAN-SR evaluator: ticker-index sets, last day only. Reference only."""
    last = pred.shape[1] - 1
    idx = np.nonzero(mask[:, last] > 0.5)[0]
    gt_top = list(set(idx[np.argsort(-gt[idx, last], kind="stable")[:k]].tolist()))
    pr_top = list(set(idx[np.argsort(-pred[idx, last], kind="stable")[:k]].tolist()))
    if len(gt_top) != len(pr_top) or len(gt_top) < 2:
        return 0.0
    return float(ndcg_score(np.array(gt_top)[None, :], np.array(pr_top)[None, :]))


def masked_mse(pred, gt, mask) -> float:
    return float(np.sum(((pred - gt) * mask) ** 2) / max(np.sum(mask), 1.0))


def evaluate_all(pred, gt, mask, k=5, periods_per_year=252) -> dict:
    r = topk_daily_returns(pred, gt, mask, k)
    return {
        "sr": sharpe(r, periods_per_year),
        "irr": irr(r),
        "cumret": cumulative_return(r),
        "mdd": max_drawdown(r),
        "ann_vol": float(np.std(r) * math.sqrt(periods_per_year)),
        "ndcg5": ndcg_at_k(pred, gt, mask, k),
        "ndcg_sthan": ndcg_sthan_compat(pred, gt, mask, k),
        "mse": masked_mse(pred, gt, mask),
        "n_days": int(pred.shape[1]),
    }
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hypershift/eval/metrics.py tests/test_metrics.py
git commit -m "feat(eval): Sharpe/IRR/NDCG/MSE/drawdown/cost-aware metrics"
```

---
### Task 8: Config, loss, training loop, CLI

**Files:**
- Create:
  - `src/hypershift/config.py`, `src/hypershift/train/loss.py`, `src/hypershift/train/loop.py`, `src/hypershift/run.py`
  - `configs/think_nyse.yaml`, `configs/think_nasdaq.yaml`
  - `tests/test_loop.py`

**Interfaces:**
- Consumes: `load_rsr`, `MarketData` (Task 2); hypergraph functions and `select_universe` (Task 3); `THINK` (Task 6); `evaluate_all`, `topk_daily_returns` (Task 7).
- Produces:
  - `RunConfig` (dataclass; fields below). Methods `.run_dir() -> Path` and `.to_dict() -> dict`.
  - `config_from_dict(d) -> RunConfig`, `load_yaml(path) -> dict`, `apply_overrides(d, ["k=v", ...]) -> dict`
  - `rank_mse_loss(pred, gt, mask, alpha) -> (loss, reg, rank)`, where every input is `[B,N]`.
  - In `hypershift.train.loop`:
    - `window_offsets(data, seq, split) -> np.ndarray`
    - `gather_batch(data, offsets, seq) -> (x[B,N,seq,C], mask[B,N], base[B,N], gt[B,N])`
    - `prepare(cfg, data=None, hg=None) -> (MarketData, Hypergraph)`
    - `build_model(cfg, in_dim) -> THINK`
    - `predict_split(model, data, thg, cfg, split, device) -> (pred[N,D], gt[N,D], mask[N,D])`
    - `train_one_run(cfg, data=None, hg=None) -> dict`
  - Run-folder contract (read by Tasks 12–13): `results/<exp>/<label>/seed_<k>/` contains:
    - `config.json`
    - `history.jsonl` (one JSON per epoch: `epoch, train_loss, sec, val{...}, test{...}`)
    - `metrics.json` (keys `best_epoch, val, test, test_oracle_sr, test_oracle_epoch, epochs_run, sec_per_epoch, num_nodes, num_edges, covered_frac, config`)
    - `test_pred.npy`, `test_gt.npy`, `test_mask.npy`, `test_daily.npy`
    - `failed.json` (only when a run fails)

- [ ] **Step 1: Write the failing tests** `tests/test_loop.py`

```python
import json
import numpy as np
import pytest
import torch
from dataclasses import replace
from hypershift.config import RunConfig, apply_overrides, config_from_dict
from hypershift.train.loop import gather_batch, prepare, train_one_run, window_offsets
from hypershift.train.loss import rank_mse_loss


def small_cfg(tmp_path, **kw):
    base = dict(exp="t", label="x", seq=8, kernel=2, hidden=8, epochs=4, patience=50, device="cpu",
                out_root=str(tmp_path), lr=5e-3)
    base.update(kw)
    return RunConfig(**base)


def test_window_offsets_no_leak(synthetic_market):
    d, seq = synthetic_market, 8
    tr, va, te = (window_offsets(d, seq, s) for s in ("train", "val", "test"))
    assert (tr + seq).max() < d.valid_index                     # train targets strictly before validation
    assert (va + seq).min() == d.valid_index and (va + seq).max() == d.test_index - 1
    assert (te + seq).min() == d.test_index and (te + seq).max() == d.num_steps - 1
    assert tr.min() == 0


def test_gather_batch_alignment(synthetic_market):
    d, seq = synthetic_market, 8
    offs = np.array([40, 45])
    x, m, b, g = gather_batch(d, offs, seq)
    assert x.shape == (2, d.num_nodes, seq, 5)
    np.testing.assert_allclose(x[1, :, -1], d.features[:, 45 + seq - 1])   # last input day
    np.testing.assert_allclose(g[1], d.gt[:, 45 + seq])                   # target = next day
    np.testing.assert_allclose(b[1], d.base_price[:, 45 + seq - 1])
    assert m[1, 0] == 0.0          # stock 0 missing on day 50, inside window 45..53
    assert m[0, 0] == 1.0          # window 40..48 is clean


def test_rank_mse_loss():
    gt = torch.tensor([[0.03, 0.01, -0.02]])
    m = torch.ones_like(gt)
    _, _, rank_good = rank_mse_loss(gt * 2, gt, m, 1.0)
    _, _, rank_bad = rank_mse_loss(-gt, gt, m, 1.0)
    assert rank_good.item() == 0.0 and rank_bad.item() > 0
    m2 = torch.tensor([[1.0, 1.0, 0.0]])
    loss, reg, _ = rank_mse_loss(torch.tensor([[0.03, 0.01, 9.0]]), gt, m2, 0.0)
    assert reg.item() == pytest.approx(0.0)


def test_overrides_and_roundtrip():
    d = apply_overrides({"label": "a"}, ["lr=0.01", "sources=industry", "shuffle_train_labels=true", "epochs=3"])
    cfg = config_from_dict(d)
    assert cfg.lr == 0.01 and cfg.sources == ("industry",) and cfg.shuffle_train_labels is True and cfg.epochs == 3
    assert config_from_dict(cfg.to_dict()) == cfg
    with pytest.raises(KeyError):
        config_from_dict({"nope": 1})


def test_prepare_structures(synthetic_market, synthetic_hypergraph, tmp_path):
    _, hg = prepare(small_cfg(tmp_path, structure="clique"), synthetic_market, synthetic_hypergraph)
    assert all(len(e) == 2 for e in hg.edges)
    _, hg = prepare(small_cfg(tmp_path, drop_hub_degree=2), synthetic_market, synthetic_hypergraph)
    assert 2 not in {v for e in hg.edges for v in e}               # node 2 has degree 2
    d, hg = prepare(small_cfg(tmp_path, universe_size=6, universe_seed=1), synthetic_market, synthetic_hypergraph)
    assert d.num_nodes == 6 and hg.num_nodes == 6
    d, _ = prepare(small_cfg(tmp_path, shuffle_train_labels=True), synthetic_market, synthetic_hypergraph)
    assert not np.allclose(d.gt[:, : d.valid_index], synthetic_market.gt[:, : d.valid_index])
    np.testing.assert_allclose(d.gt[:, d.valid_index:], synthetic_market.gt[:, d.valid_index:])
    # cross-sectional shuffle: each training day keeps the same multiset of returns
    np.testing.assert_allclose(np.sort(d.gt[:, : d.valid_index], axis=0),
                               np.sort(synthetic_market.gt[:, : d.valid_index], axis=0))


def test_train_one_run_end_to_end_and_resume(synthetic_market, synthetic_hypergraph, tmp_path):
    cfg = small_cfg(tmp_path)
    m = train_one_run(cfg, synthetic_market, synthetic_hypergraph)
    out = cfg.run_dir()
    for f in ("config.json", "history.jsonl", "metrics.json", "test_pred.npy", "test_daily.npy"):
        assert (out / f).exists(), f
    assert len((out / "history.jsonl").read_text().splitlines()) == 4
    assert {"best_epoch", "val", "test", "test_oracle_sr", "sec_per_epoch", "covered_frac"} <= set(m)
    assert m["covered_frac"] == pytest.approx(11 / 12)                   # node 11 is isolated
    assert np.load(out / "test_pred.npy").shape == (12, 20)
    mtime = (out / "metrics.json").stat().st_mtime
    train_one_run(cfg, synthetic_market, synthetic_hypergraph)            # resume: no retrain
    assert (out / "metrics.json").stat().st_mtime == mtime


@pytest.mark.parametrize("temporal,spatial", [("hyp", "hyp"), ("euc", "euc")])
def test_train_loss_decreases(synthetic_market, synthetic_hypergraph, tmp_path, temporal, spatial):
    cfg = small_cfg(tmp_path, epochs=15, temporal=temporal, spatial=spatial, label=temporal + spatial)
    train_one_run(cfg, synthetic_market, synthetic_hypergraph)
    hist = [json.loads(l) for l in (cfg.run_dir() / "history.jsonl").read_text().splitlines()]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]


def test_micro_batch_matches_full_batch(synthetic_market, synthetic_hypergraph, tmp_path):
    """Gradient accumulation must be mathematically identical to the unsplit batch (D15)."""
    runs = {}
    for label, micro in (("full", 0), ("micro", 1)):
        cfg = small_cfg(tmp_path, label=label, batch_days=4, micro_batch_days=micro, epochs=3)
        train_one_run(cfg, synthetic_market, synthetic_hypergraph)
        runs[label] = [json.loads(l)["train_loss"] for l in (cfg.run_dir() / "history.jsonl").read_text().splitlines()]
    np.testing.assert_allclose(runs["micro"], runs["full"], rtol=1e-4)


def test_train_raises_on_nan(synthetic_market, synthetic_hypergraph, tmp_path):
    bad = replace(synthetic_market, features=synthetic_market.features.copy())
    bad.features[:, 5, :] = np.nan
    with pytest.raises(FloatingPointError):
        train_one_run(small_cfg(tmp_path, label="nan"), bad, synthetic_hypergraph)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_loop.py -v`
Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/config.py`

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    exp: str = "debug"
    label: str = "THINK"
    market: str = "NYSE"                    # NYSE | NASDAQ | FRESH
    data_root: str = "data/raw/rsr/data"
    fresh_name: str = ""                    # folder under data/fresh when market == FRESH
    norm: str = "train"                     # train | paper
    sources: tuple = ("industry", "wiki")   # industry, wiki, corr, sector, subindustry, random
    corr_clusters: int = 100
    structure: str = "hyper"                # hyper | clique | none
    decompose_mode: str = "none"            # none | large_first | small_first
    decompose_size: int = 0
    drop_hub_degree: int = 0
    universe_size: int = 0
    universe_seed: int = 0
    temporal: str = "hyp"                   # hyp | euc
    spatial: str = "hyp"                    # hyp | euc
    attn_score: str = "mobius"              # mobius | concat
    attn_dist: str = "mult"                 # mult | neg | off
    target: str = "return"                  # return | price
    shuffle_train_labels: bool = False
    seq: int = 16
    kernel: int = 4
    hidden: int = 32
    lr: float = 1e-3
    weight_decay: float = 5e-4
    alpha: float = 1.0
    epochs: int = 100
    patience: int = 20
    batch_days: int = 1                     # days per optimizer step (must match across compared arms)
    micro_batch_days: int = 0               # >0: split each step into gradient-accumulated chunks (memory only)
    topk: int = 5
    periods_per_year: int = 252
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"
    out_root: str = "results"

    def run_dir(self) -> Path:
        return Path(self.out_root) / self.exp / self.label / f"seed_{self.seed}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sources"] = list(self.sources)
        return d


def config_from_dict(d: dict) -> RunConfig:
    names = {f.name for f in fields(RunConfig)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    d = dict(d)
    if "sources" in d:
        d["sources"] = tuple(d["sources"])
    return RunConfig(**d)


def load_yaml(path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def apply_overrides(base: dict, pairs: list[str]) -> dict:
    out = dict(base)
    defaults = RunConfig()
    for p in pairs:
        k, v = p.split("=", 1)
        cur = getattr(defaults, k)
        if isinstance(cur, bool):
            out[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            out[k] = int(v)
        elif isinstance(cur, float):
            out[k] = float(v)
        elif isinstance(cur, tuple):
            out[k] = tuple(s for s in v.split(",") if s)
        else:
            out[k] = v
    return out


def dump_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
```

`src/hypershift/train/loss.py`:
```python
import torch
import torch.nn.functional as F


def rank_mse_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, alpha: float):
    """STHAN-SR/RSR objective: masked MSE on returns + alpha * pairwise hinge on mis-ordered pairs. Inputs [B,N]."""
    reg = (mask * (pred - gt) ** 2).mean()
    dp = pred[:, :, None] - pred[:, None, :]
    dg = gt[:, :, None] - gt[:, None, :]
    mm = mask[:, :, None] * mask[:, None, :]
    rank = F.relu(-dp * dg * mm).mean()
    return reg + alpha * rank, reg, rank
```

`src/hypershift/train/loop.py`:
```python
from __future__ import annotations

import functools
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from hypershift.config import RunConfig, dump_json
from hypershift.data.hypergraph import (
    Hypergraph, build_rsr_hypergraph, canonical, clique_expand, correlation_hyperedges,
    decompose, drop_hub_edges, random_like,
)
from hypershift.data.rsr import MarketData, load_rsr
from hypershift.data.universe import select_universe
from hypershift.eval.metrics import evaluate_all, topk_daily_returns
from hypershift.models.think import THINK
from hypershift.train.loss import rank_mse_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_offsets(data: MarketData, seq: int, split: str) -> np.ndarray:
    lo, hi = {"train": (seq, data.valid_index), "val": (data.valid_index, data.test_index),
              "test": (data.test_index, data.num_steps)}[split]
    offsets = np.arange(max(lo, seq), hi) - seq
    if data.eligible_ends is not None:
        offsets = offsets[np.isin(offsets + seq - 1, data.eligible_ends)]
    return offsets


def gather_batch(data: MarketData, offsets: np.ndarray, seq: int):
    offsets = np.asarray(offsets)
    idx = offsets[:, None] + np.arange(seq)[None, :]
    x = data.features[:, idx].transpose(1, 0, 2, 3)
    midx = offsets[:, None] + np.arange(seq + 1)[None, :]
    mask = data.mask[:, midx].min(axis=2).T
    base = data.base_price[:, offsets + seq - 1].T
    gt = data.gt[:, offsets + seq].T
    return (np.ascontiguousarray(x, dtype=np.float32), mask.astype(np.float32),
            base.astype(np.float32), gt.astype(np.float32))


@functools.lru_cache(maxsize=4)
def _load_market_cached(market: str, data_root: str, norm: str, fresh_name: str) -> MarketData:
    """One parse per process: run_grid trains hundreds of runs on the same market. Never mutate the result."""
    if market in ("NYSE", "NASDAQ"):
        return load_rsr(data_root, market, norm)
    if market == "FRESH":
        from hypershift.data.fresh import load_panel
        return load_panel(Path("data/fresh") / fresh_name)
    raise ValueError(market)


def load_market(cfg: RunConfig) -> MarketData:
    return _load_market_cached(cfg.market, cfg.data_root, cfg.norm, cfg.fresh_name)


def base_hypergraph(cfg: RunConfig, data: MarketData) -> Hypergraph:
    srcs = tuple(cfg.sources)
    edges: list[tuple[int, ...]] = []
    rsr = tuple(s for s in srcs if s in ("industry", "wiki"))
    if rsr:
        edges += build_rsr_hypergraph(cfg.data_root, cfg.market, rsr).edges
    if "corr" in srcs:
        tr = slice(1, data.valid_index)
        edges += correlation_hyperedges(data.gt[:, tr], data.mask[:, tr], cfg.corr_clusters).edges
    for level in ("sector", "subindustry"):
        if level in srcs:
            from hypershift.data.fresh import gics_hypergraph
            edges += gics_hypergraph(Path("data/fresh") / cfg.fresh_name, level).edges
    hg = Hypergraph(data.num_nodes, canonical(edges))
    if "random" in srcs:
        hg = random_like(hg, seed=1000 + cfg.seed)
    return hg


def shuffle_train_labels(data: MarketData, seed: int) -> MarketData:
    """Leakage null (D10.1): permute returns across the observed stocks of each training day.

    Unlike a permutation of days, this also destroys stock-level drift, so no learnable signal remains.
    """
    rng = np.random.default_rng(seed)
    gt = data.gt.copy()
    for t in range(data.valid_index):
        ok = np.nonzero(data.mask[:, t] > 0)[0]
        gt[ok, t] = data.gt[rng.permutation(ok), t]
    return replace(data, gt=gt)


def prepare(cfg: RunConfig, data: MarketData | None = None, hg: Hypergraph | None = None):
    data = data if data is not None else load_market(cfg)
    hg = hg if hg is not None else base_hypergraph(cfg, data)
    data, hg = select_universe(data, hg, cfg.universe_size, cfg.universe_seed)
    hg = decompose(hg, cfg.decompose_mode, cfg.decompose_size)
    if cfg.structure == "clique":
        hg = clique_expand(hg)
    hg = drop_hub_edges(hg, cfg.drop_hub_degree)
    if cfg.shuffle_train_labels:
        data = shuffle_train_labels(data, cfg.seed)
    return data, hg


def build_model(cfg: RunConfig, in_dim: int) -> THINK:
    return THINK(in_dim=in_dim, hidden=cfg.hidden, seq=cfg.seq, kernel=cfg.kernel, temporal=cfg.temporal,
                 spatial=cfg.spatial, structure=cfg.structure, attn_score=cfg.attn_score, attn_dist=cfg.attn_dist)


def _to_return(out, base, target):
    return out if target == "return" else (out - base) / base


@torch.no_grad()
def predict_split(model, data, thg, cfg, split, device):
    model.eval()
    offs = window_offsets(data, cfg.seq, split)
    step = max(1, cfg.micro_batch_days or cfg.batch_days)
    preds, gts, masks = [], [], []
    for i in range(0, len(offs), step):
        x, m, b, g = gather_batch(data, offs[i:i + step], cfg.seq)
        out = model(torch.as_tensor(x, device=device), thg)
        preds.append(_to_return(out, torch.as_tensor(b, device=device), cfg.target).cpu().numpy())
        gts.append(g)
        masks.append(m)
    return np.concatenate(preds).T, np.concatenate(gts).T, np.concatenate(masks).T


def train_one_run(cfg: RunConfig, data: MarketData | None = None, hg: Hypergraph | None = None) -> dict:
    out = cfg.run_dir()
    if (out / "metrics.json").exists():
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    dump_json(cfg.to_dict(), out / "config.json")
    set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    data, hg = prepare(cfg, data, hg)
    thg = hg.to_torch(device)
    model = build_model(cfg, data.features.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_offs = window_offsets(data, cfg.seq, "train")
    rng = np.random.default_rng(cfg.seed)
    best, bad, epoch_secs, test_srs = None, 0, [], []
    with open(out / "history.jsonl", "w") as hist:
        for epoch in range(cfg.epochs):
            t0 = time.time()
            model.train()
            rng.shuffle(train_offs)
            losses = []
            micro = cfg.micro_batch_days if cfg.micro_batch_days > 0 else cfg.batch_days
            for i in range(0, len(train_offs), cfg.batch_days):
                batch = train_offs[i:i + cfg.batch_days]
                opt.zero_grad()
                step_loss = 0.0
                for j in range(0, len(batch), micro):   # gradient accumulation == one unsplit step
                    chunk = batch[j:j + micro]
                    x, m, b, g = (torch.as_tensor(a, device=device) for a in gather_batch(data, chunk, cfg.seq))
                    pred = _to_return(model(x, thg), b, cfg.target)
                    loss, _, _ = rank_mse_loss(pred, g, m, cfg.alpha)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss: epoch {epoch}, step {i} (see decision node D5)")
                    (loss * (len(chunk) / len(batch))).backward()
                    step_loss += loss.item() * len(chunk) / len(batch)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                losses.append(step_loss)
            vp, vg, vm = predict_split(model, data, thg, cfg, "val", device)
            tp, tg, tm = predict_split(model, data, thg, cfg, "test", device)
            vmet = evaluate_all(vp, vg, vm, cfg.topk, cfg.periods_per_year)
            tmet = evaluate_all(tp, tg, tm, cfg.topk, cfg.periods_per_year)
            sec = time.time() - t0
            epoch_secs.append(sec)
            test_srs.append(tmet["sr"])
            hist.write(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), "sec": sec,
                                   "val": vmet, "test": tmet}) + "\n")
            hist.flush()
            if best is None or vmet["sr"] > best["val"]["sr"]:
                best = {"best_epoch": epoch, "val": vmet, "test": tmet}
                bad = 0
                np.save(out / "test_pred.npy", tp)
                np.save(out / "test_gt.npy", tg)
                np.save(out / "test_mask.npy", tm)
                np.save(out / "test_daily.npy", topk_daily_returns(tp, tg, tm, cfg.topk))
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    metrics = {
        **best,
        "test_oracle_sr": float(max(test_srs)),
        "test_oracle_epoch": int(np.argmax(test_srs)),
        "epochs_run": len(test_srs),
        "sec_per_epoch": float(np.mean(epoch_secs)),
        "num_nodes": data.num_nodes,
        "num_edges": len(hg.edges),
        "covered_frac": float((hg.node_degree() > 0).mean()),     # share of stocks in >= 1 hyperedge
        "config": cfg.to_dict(),
    }
    dump_json(metrics, out / "metrics.json")
    return metrics
```

`src/hypershift/run.py`:
```python
"""python -m hypershift.run --config configs/think_nyse.yaml --seeds 0-4 --set epochs=50 exp=manual"""
from __future__ import annotations

import argparse
import json

from hypershift.config import apply_overrides, config_from_dict, load_yaml
from hypershift.train.loop import train_one_run


def parse_seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    base = apply_overrides(load_yaml(args.config), args.set)
    for s in parse_seeds(args.seeds):
        m = train_one_run(config_from_dict({**base, "seed": s}))
        print(json.dumps({"seed": s, "best_epoch": m["best_epoch"], "val_sr": m["val"]["sr"],
                          "test_sr": m["test"]["sr"], "test_ndcg5": m["test"]["ndcg5"]}))


if __name__ == "__main__":
    main()
```

`configs/think_nyse.yaml`:
```yaml
exp: manual
label: THINK
market: NYSE
data_root: data/raw/rsr/data
alpha: 1.0
```

`configs/think_nasdaq.yaml`:
```yaml
exp: manual
label: THINK_NASDAQ
market: NASDAQ
data_root: data/raw/rsr/data
alpha: 0.1
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_loop.py -v`
Expected: all PASS.
- `test_train_loss_decreases` fails: go to D6.
- `test_train_raises_on_nan` fails: the NaN guard is missing or placed after `backward()`.
- `test_micro_batch_matches_full_batch` fails: each chunk's loss must be scaled by `len(chunk) / len(batch)` before `backward()`, and `opt.zero_grad()` must come once per full step, not once per chunk.

- [ ] **Step 5: Real-data smoke run** (needs GPU and data)

```bash
python -m hypershift.run --config configs/think_nyse.yaml --seeds 0 --set exp=smoke epochs=2
```
Expected: one JSON line with finite `val_sr` and `test_sr`, and `results/smoke/THINK/seed_0/metrics.json` exists.
- NaN: go to D5.
- Out of memory: go to D15.

Then delete `results/smoke/`.

- [ ] **Step 6: Commit**

```bash
git add src/hypershift/config.py src/hypershift/train src/hypershift/run.py configs tests/test_loop.py
git commit -m "feat(train): config, rank+MSE loss, leak-free windows, training loop with val selection, CLI"
```

---

### Task 9: Non-learned baselines

**Files:**
- Create: `src/hypershift/eval/baselines.py`, `scripts/baselines.py`, `tests/test_baselines.py`

**Interfaces:**
- Consumes: `window_offsets`, `gather_batch` (Task 8); `evaluate_all`, `sharpe` and friends (Task 7).
- Produces:
  - `baseline_scores(data, seq, split, kind, seed=0, lookback=5) -> (pred, gt, mask)` with `kind ∈ {"oracle","random","momentum","reversal"}`.
  - `market_daily_returns(gt, mask) -> np.ndarray`
  - `evaluate_baselines(data, seq=16, split="test", k=5, periods_per_year=252, random_seeds=25) -> dict[str, dict]`, keyed by `random`, `momentum`, `reversal`, `market`, `oracle`.
  - File `results/baselines/<name>.json`.

- [ ] **Step 1: Write the failing tests** `tests/test_baselines.py`

```python
import numpy as np
from hypershift.data.universe import select_universe
from hypershift.eval.baselines import evaluate_baselines, market_daily_returns


def test_baselines_ordering(synthetic_market):
    res = evaluate_baselines(synthetic_market, seq=8, random_seeds=5)
    assert set(res) == {"random", "momentum", "reversal", "market", "oracle"}
    assert res["oracle"]["sr"] > res["random"]["sr"]
    assert res["oracle"]["irr"] >= res["momentum"]["irr"]


def test_baselines_same_universe(synthetic_market, synthetic_hypergraph):
    sub, _ = select_universe(synthetic_market, synthetic_hypergraph, size=6, seed=0)
    full = evaluate_baselines(synthetic_market, seq=8, random_seeds=5)
    part = evaluate_baselines(sub, seq=8, random_seeds=5)
    assert full["oracle"]["irr"] >= part["oracle"]["irr"]          # bigger universe -> better oracle top-5


def test_market_returns_equal_weight():
    gt = np.array([[0.02, 0.0], [0.04, 0.1]])
    mask = np.array([[1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(market_daily_returns(gt, mask), [0.03, 0.1])
```

- [ ] **Step 2: Run the tests to verify they fail.** Run: `pytest tests/test_baselines.py -v`. Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/eval/baselines.py`

```python
import numpy as np

from hypershift.data.rsr import MarketData
from hypershift.eval.metrics import cumulative_return, evaluate_all, irr, max_drawdown, sharpe
from hypershift.train.loop import gather_batch, window_offsets


def baseline_scores(data: MarketData, seq: int, split: str, kind: str, seed: int = 0, lookback: int = 5):
    offs = window_offsets(data, seq, split)
    _, m, _, g = gather_batch(data, offs, seq)
    gt, mask = g.T, m.T
    ends = offs + seq - 1
    if kind == "oracle":
        pred = gt.copy()
    elif kind == "random":
        pred = np.random.default_rng(seed).standard_normal(gt.shape)
    elif kind in ("momentum", "reversal"):
        past = data.base_price[:, np.maximum(ends - lookback, 0)]
        mom = data.base_price[:, ends] / np.maximum(past, 1e-8) - 1
        pred = mom if kind == "momentum" else -mom
    else:
        raise ValueError(kind)
    return pred, gt, mask


def market_daily_returns(gt, mask) -> np.ndarray:
    return (gt * mask).sum(axis=0) / np.maximum(mask.sum(axis=0), 1.0)


def evaluate_baselines(data, seq=16, split="test", k=5, periods_per_year=252, random_seeds=25) -> dict:
    res = {}
    for kind in ("oracle", "momentum", "reversal"):
        res[kind] = evaluate_all(*baseline_scores(data, seq, split, kind), k=k, periods_per_year=periods_per_year)
    runs = [evaluate_all(*baseline_scores(data, seq, split, "random", seed=s), k=k,
                         periods_per_year=periods_per_year) for s in range(random_seeds)]
    res["random"] = {key: float(np.mean([r[key] for r in runs])) for key in runs[0]}
    res["random"]["sr_std"] = float(np.std([r["sr"] for r in runs]))
    _, gt, mask = baseline_scores(data, seq, split, "oracle")
    mr = market_daily_returns(gt, mask)
    res["market"] = {"sr": sharpe(mr, periods_per_year), "irr": irr(mr), "cumret": cumulative_return(mr),
                     "mdd": max_drawdown(mr)}
    return res
```

`scripts/baselines.py`:
```python
"""Baselines for every universe used in the study. Usage: python scripts/baselines.py [--fresh NAME ...]"""
import argparse
import json
from pathlib import Path

from hypershift.config import RunConfig
from hypershift.data.hypergraph import build_rsr_hypergraph
from hypershift.data.rsr import load_rsr
from hypershift.data.universe import select_universe
from hypershift.eval.baselines import evaluate_baselines

ap = argparse.ArgumentParser()
ap.add_argument("--fresh", nargs="*", default=[])
args = ap.parse_args()
out = Path("results/baselines")
out.mkdir(parents=True, exist_ok=True)
root = RunConfig().data_root
for market in ("NYSE", "NASDAQ"):
    data = load_rsr(root, market, "train")
    (out / f"{market}.json").write_text(json.dumps(evaluate_baselines(data), indent=2))
    if market == "NYSE":
        hg = build_rsr_hypergraph(root, market)
        for n in (50, 100, 250, 500, 1000):
            for u in (0, 1, 2):
                sub, _ = select_universe(data, hg, n, u)
                (out / f"NYSE_N{n}_u{u}.json").write_text(json.dumps(evaluate_baselines(sub), indent=2))
for name in args.fresh:
    from hypershift.data.fresh import load_panel
    data = load_panel(Path("data/fresh") / name)
    ppy = 1764 if name.endswith("_hourly") else 252
    (out / f"{name}.json").write_text(json.dumps(evaluate_baselines(data, periods_per_year=ppy), indent=2))
print("wrote", sorted(p.name for p in out.glob("*.json")))
```

- [ ] **Step 4: Run the tests.** Run: `pytest tests/test_baselines.py -v`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add src/hypershift/eval/baselines.py scripts/baselines.py tests/test_baselines.py
git commit -m "feat(eval): random/momentum/reversal/market/oracle baselines"
```

---

### Task 10: Statistics (Wilcoxon, Holm, block bootstrap, verdict)

**Files:**
- Create: `src/hypershift/eval/stats.py`, `tests/test_stats.py`

**Interfaces:**
- Produces:
  - `wilcoxon_paired(a, b) -> float` (p-value)
  - `wilcoxon_one_sample(x) -> float`
  - `holm(pvals: dict[str, float]) -> dict[str, float]`
  - `stationary_bootstrap_indices(n, mean_block, rng) -> np.ndarray`
  - `sharpe_contrast_ci(series: list[np.ndarray], weights: list[float], mean_block=10, n_boot=5000, alpha=0.05, seed=0, periods_per_year=252) -> dict(est, lo, hi, p_boot)`
  - `sharpe_diff_ci(ra, rb, **kw)`: the same thing with weights `[1, -1]`.
  - `verdict(p_holm, lo, hi) -> str`, returning one of `"STRONG"`, `"SEED-ROBUST ONLY"`, `"NO EVIDENCE"`.
- Note: with n seeds, the smallest possible two-sided Wilcoxon p is `2 / 2^n`. That is 0.0625 for n = 5, so **5-seed screens can never yield a verdict**.
  - After Holm correction over a family of m comparisons, the floor becomes `m · 2^(1−n)`. With 10 seeds and m = 8 that is 0.0156, which also can never pass.
  - This is why verdict-bearing arms use ≥ 15 seeds. `test_grid.py::test_holm_floor_allows_verdicts` checks every family.

- [ ] **Step 1: Write the failing tests** `tests/test_stats.py`

```python
import numpy as np
import pytest
from hypershift.eval.stats import (
    holm, sharpe_contrast_ci, sharpe_diff_ci, stationary_bootstrap_indices, verdict,
    wilcoxon_one_sample, wilcoxon_paired,
)


def test_holm_known_example():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_wilcoxon():
    rng = np.random.default_rng(0)
    b = rng.normal(size=25)
    assert wilcoxon_paired(b, b) == 1.0
    assert wilcoxon_paired(b + 1 + 0.01 * rng.normal(size=25), b) < 0.01
    assert wilcoxon_one_sample(np.full(25, 0.5) + 0.01 * rng.normal(size=25)) < 0.01


def test_bootstrap_indices_valid():
    idx = stationary_bootstrap_indices(100, 10, np.random.default_rng(0))
    assert idx.shape == (100,) and idx.min() >= 0 and idx.max() < 100


def test_sharpe_diff_ci():
    rng = np.random.default_rng(0)
    rb = rng.normal(0, 0.01, 250)
    same = sharpe_diff_ci(rb, rb, n_boot=300)
    assert same["est"] == 0 and same["lo"] == 0 and same["hi"] == 0
    better = sharpe_diff_ci(rb + 0.005, rb, n_boot=300)
    assert better["est"] > 0 and better["lo"] > 0


def test_contrast_matches_diff():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.001, 0.01, 200), rng.normal(0, 0.01, 200)
    d = sharpe_diff_ci(a, b, n_boot=200, seed=3)
    c = sharpe_contrast_ci([a, b], [1, -1], n_boot=200, seed=3)
    assert d == c


def test_verdict():
    assert verdict(0.001, 0.1, 0.5) == "STRONG"
    assert verdict(0.001, -0.1, 0.5) == "SEED-ROBUST ONLY"
    assert verdict(0.2, 0.1, 0.5) == "NO EVIDENCE"
```

- [ ] **Step 2: Run the tests to verify they fail.** Run: `pytest tests/test_stats.py -v`. Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/eval/stats.py`

```python
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from hypershift.eval.metrics import sharpe


def wilcoxon_paired(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a - b, 0):
        return 1.0
    return float(sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def wilcoxon_one_sample(x) -> float:
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        return 1.0
    return float(sps.wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    restart = rng.random(n) < 1.0 / mean_block
    fresh = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = fresh[t] if restart[t] else (idx[t - 1] + 1) % n
    return idx


def sharpe_contrast_ci(series, weights, mean_block=10, n_boot=5000, alpha=0.05, seed=0, periods_per_year=252) -> dict:
    """CI for sum_i w_i * Sharpe(series_i), resampling the same day-blocks for all series (keeps pairing)."""
    series = [np.asarray(s, float) for s in series]
    n = min(len(s) for s in series)
    series = [s[-n:] for s in series]                      # align on the last n days if lengths differ
    est = float(sum(w * sharpe(s, periods_per_year) for w, s in zip(weights, series)))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        boots[b] = sum(w * sharpe(s[idx], periods_per_year) for w, s in zip(weights, series))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"est": est, "lo": float(lo), "hi": float(hi), "p_boot": float(min(p, 1.0))}


def sharpe_diff_ci(ra, rb, **kw) -> dict:
    return sharpe_contrast_ci([ra, rb], [1.0, -1.0], **kw)


def verdict(p_holm: float, lo: float, hi: float) -> str:
    if p_holm < 0.01 and (lo > 0 or hi < 0):
        return "STRONG"
    if p_holm < 0.01:
        return "SEED-ROBUST ONLY"
    return "NO EVIDENCE"
```

- [ ] **Step 4: Run the tests.** Run: `pytest tests/test_stats.py -v`. Expected: PASS. In `test_sharpe_diff_ci`, "same" gives exactly 0 because both series are resampled with identical indices.
- [ ] **Step 5: Commit**

```bash
git add src/hypershift/eval/stats.py tests/test_stats.py
git commit -m "feat(eval): paired Wilcoxon, Holm, stationary block bootstrap on Sharpe contrasts"
```

---

### Task 11: Hyperbolicity (δ_hg on hypergraph s-distance, δ_rel on features)

**Files:**
- Create: `src/hypershift/geometry/hyperbolicity.py`, `scripts/hyperbolicity.py`, `tests/test_hyperbolicity.py`

**Interfaces:**
- Consumes: `Hypergraph` (Task 3), `load_rsr` (Task 2), `select_universe` (Task 3).
- Produces:
  - `s_distance_matrix(hg, s=1) -> np.ndarray [N,N]` (inf where disconnected)
  - `largest_component(D) -> np.ndarray`
  - `gromov_delta(D, base=0) -> float`
  - `sampled_delta(D, sample=1000, repeats=5, seed=0) -> dict(delta_max, delta_mean, diam, delta_rel)`
  - `hypergraph_hyperbolicity(hg, s=1, sample=1000, repeats=5, seed=0) -> dict` (adds `lcc_size`)
  - `feature_hyperbolicity(X, sample=1000, repeats=5, seed=0) -> dict`
  - Output file `results/hyperbolicity.json`.
- Definitions (paper eq. 1-2 and appendix A):
  - Gromov product `(y,z)_x = ½(d(x,y) + d(x,z) − d(y,z))`.
  - δ is the smallest value such that `(x,z)_w ≥ min((x,y)_w, (y,z)_w) − δ` for all x, y, z.
  - Computed with Khrulkov et al.'s max-min matrix trick, using a fixed base point per sample.
  - `δ_rel = 2δ / diam`.
  - The s-walk distance with s = 1 means two stocks are adjacent if they share ≥ 1 hyperedge (appendix Algorithm 1).

- [ ] **Step 1: Write the failing tests** `tests/test_hyperbolicity.py`

```python
import numpy as np
import pytest
from hypershift.data.hypergraph import Hypergraph
from hypershift.geometry.hyperbolicity import (
    feature_hyperbolicity, gromov_delta, hypergraph_hyperbolicity, s_distance_matrix, sampled_delta,
)


def test_tree_has_zero_delta():
    path = Hypergraph(5, ((0, 1), (1, 2), (2, 3), (3, 4)))
    assert gromov_delta(s_distance_matrix(path)) == 0.0


def test_four_cycle_delta_is_one():
    c4 = Hypergraph(4, ((0, 1), (1, 2), (2, 3), (3, 0)))
    assert gromov_delta(s_distance_matrix(c4)) == pytest.approx(1.0)


def test_single_hyperedge_is_clique_zero_delta():
    D = s_distance_matrix(Hypergraph(4, ((0, 1, 2, 3),)))
    assert D[0, 3] == 1 and gromov_delta(D) == 0.0


def test_s_parameter():
    hg = Hypergraph(3, ((0, 1), (0, 1, 2)))
    assert s_distance_matrix(hg, s=2)[0, 1] == 1
    assert np.isinf(s_distance_matrix(hg, s=2)[0, 2])


def test_hypergraph_hyperbolicity_uses_lcc():
    hg = Hypergraph(7, ((0, 1), (1, 2), (2, 3), (3, 0), (5, 6)))
    out = hypergraph_hyperbolicity(hg, sample=10, repeats=2)
    assert out["lcc_size"] == 4 and out["delta_max"] == pytest.approx(1.0)


def test_points_on_a_line_have_zero_rel_delta():
    X = np.linspace(0, 1, 30)[:, None]
    assert feature_hyperbolicity(X, sample=30, repeats=1)["delta_rel"] == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail.** Run: `pytest tests/test_hyperbolicity.py -v`. Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/geometry/hyperbolicity.py`

```python
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist

from hypershift.data.hypergraph import Hypergraph


def s_distance_matrix(hg: Hypergraph, s: int = 1) -> np.ndarray:
    node, edge = hg.incidence()
    H = csr_matrix((np.ones(len(node)), (node, edge)), shape=(hg.num_nodes, max(len(hg.edges), 1)))
    C = (H @ H.T).tolil()
    C.setdiag(0)
    A = (C.tocsr() >= s).astype(float)
    return shortest_path(A, method="D", unweighted=True, directed=False)


def largest_component(D: np.ndarray) -> np.ndarray:
    finite = np.isfinite(D).sum(axis=1)
    return np.nonzero(np.isfinite(D[int(np.argmax(finite))]))[0]


def gromov_delta(D: np.ndarray, base: int = 0) -> float:
    D = np.asarray(D, dtype=np.float32)
    row = D[base]
    A = 0.5 * (row[:, None] + row[None, :] - D)
    M = np.empty_like(A)
    for i in range(0, len(A), 16):
        M[i:i + 16] = np.minimum(A[i:i + 16, :, None], A[None, :, :]).max(axis=1)
    return float(max((M - A).max(), 0.0))


def sampled_delta(D: np.ndarray, sample: int = 1000, repeats: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(D)
    vals = []
    for _ in range(repeats):
        idx = rng.choice(n, size=min(sample, n), replace=False)
        vals.append(gromov_delta(D[np.ix_(idx, idx)], base=0))
    diam = float(D[np.isfinite(D)].max())
    dmax = float(max(vals))
    return {"delta_max": dmax, "delta_mean": float(np.mean(vals)), "diam": diam,
            "delta_rel": 2 * dmax / diam if diam > 0 else 0.0}


def hypergraph_hyperbolicity(hg: Hypergraph, s: int = 1, sample: int = 1000, repeats: int = 5, seed: int = 0) -> dict:
    D = s_distance_matrix(hg, s)
    comp = largest_component(D)
    out = sampled_delta(D[np.ix_(comp, comp)], sample, repeats, seed)
    out["lcc_size"] = int(len(comp))
    return out


def feature_hyperbolicity(X: np.ndarray, sample: int = 1000, repeats: int = 5, seed: int = 0) -> dict:
    return sampled_delta(cdist(X, X), sample, repeats, seed)
```

`scripts/hyperbolicity.py`:
```python
"""Paper Table I analogue. delta_rel features = each stock's training-period daily return series."""
import json
from pathlib import Path

import numpy as np

from hypershift.config import RunConfig
from hypershift.data.hypergraph import build_rsr_hypergraph
from hypershift.data.rsr import load_rsr
from hypershift.data.universe import select_universe
from hypershift.geometry.hyperbolicity import feature_hyperbolicity, hypergraph_hyperbolicity

root = RunConfig().data_root
res = {}
for market in ("NYSE", "NASDAQ"):
    data = load_rsr(root, market, "train")
    hg = build_rsr_hypergraph(root, market)
    universes = [("full", data, hg)]
    if market == "NYSE":
        for n in (50, 100, 250, 500, 1000):
            for u in (0, 1, 2):
                universes.append((f"N{n}_u{u}", *select_universe(data, hg, n, u)))
    for name, d, h in universes:
        X = np.where(d.mask[:, 1:d.valid_index] > 0, d.gt[:, 1:d.valid_index], 0.0)
        res[f"{market}_{name}"] = {"hg": hypergraph_hyperbolicity(h), "rel": feature_hyperbolicity(X),
                                   "num_edges": len(h.edges)}
        print(market, name, res[f"{market}_{name}"]["hg"]["delta_max"], res[f"{market}_{name}"]["rel"]["delta_rel"])
Path("results").mkdir(exist_ok=True)
Path("results/hyperbolicity.json").write_text(json.dumps(res, indent=2))
```

- [ ] **Step 4: Run the tests.** Run: `pytest tests/test_hyperbolicity.py -v`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add src/hypershift/geometry/hyperbolicity.py scripts/hyperbolicity.py tests/test_hyperbolicity.py
git commit -m "feat(geometry): Gromov delta hyperbolicity for hypergraphs and features"
```

---

### Task 12: Experiment grid, resumable runner, compute budget

**Files:**
- Create: `src/hypershift/experiments/grid.py`, `scripts/run_grid.py`, `scripts/time_budget.py`, `tests/test_grid.py`

**Interfaces:**
- Consumes: `RunConfig`, `config_from_dict`, `apply_overrides`, `load_yaml` (Task 8); `train_one_run` (Task 8); `build_rsr_hypergraph`, `hub_schedule` (Task 3).
- Produces:
  - `EXPERIMENTS: list[str]`
  - `experiment(name) -> list[RunConfig]`
  - `FAMILIES: dict[str, list[tuple[str, str]]]` (pairs of `"exp/label"`)
  - `INTERACTIONS: dict[str, list[tuple[str, str, str, str]]]`
  - `GEOMS: dict[str, dict]`
  - Constants `FRESH_DAILY_NAME = "sp500_daily"`, `FRESH_HOURLY_NAME = "sp500_1h"`, `SEEDS_FINAL/SWEEP/SCREEN` (25/15/5) and `MEMORY_LIGHT`.
  - `scripts/run_grid.py <exps…> [--labels L…] [--seeds 0-14] [--set k=v…] [--dry-run]`. It uses `parse_seeds` from `hypershift.run` (Task 8). `--labels` filters on the grid's own labels before `--set` is applied, and `--seeds` replaces each label's seed list.
  - Optional config files it reads:
    - `configs/global.yaml`: budget overrides from D7/D15, applied to every experiment.
    - `results/tuned.json`: from Phase C, shaped `{"HH": {...}, "HE": {...}, "EH": {...}, "EE": {...}}` with keys `lr`, `alpha`.
    - `configs/chosen.yaml`: attention choice from D8.
  - Precedence, later wins: geometry defaults < `tuned.json` < `global.yaml` < `chosen.yaml` < the experiment's own settings. So a D5/D7/D9/D15 fix written to `global.yaml` really applies to every arm.
  - Clique and decomposition arms get `micro_batch_days=1` automatically. This is memory only; see Global Constraints.

- [ ] **Step 1: Write the failing tests** `tests/test_grid.py`

```python
from collections import Counter
import pytest
from hypershift.experiments.grid import EXPERIMENTS, FAMILIES, INTERACTIONS, experiment

NO_DATA_NEEDED = [e for e in EXPERIMENTS if e != "E7_hubs"]


@pytest.mark.parametrize("name", NO_DATA_NEEDED)
def test_experiment_unique_runs(name):
    cfgs = experiment(name)
    assert cfgs
    keys = Counter((c.exp, c.label, c.seed) for c in cfgs)
    assert max(keys.values()) == 1


def test_sizes():
    assert len(experiment("E2_geometry")) == 3 * 25
    assert len({c.label for c in experiment("E_attn")}) == 6
    assert len(experiment("E_tune")) == 4 * 2 * 3 * 3            # 4 geometries x lr x alpha x 3 seeds
    assert {c.label.split("_")[0] for c in experiment("E_tune")} == {"HH", "HE", "EH", "EE"}
    hday = [c for c in experiment("E10_hourly") if c.label.startswith("hday_")]
    assert all((c.seq, c.kernel) == (112, 28) for c in hday)      # 16 days x 7 bars; 4-day first kernel
    assert all(c.micro_batch_days == 1 for c in experiment("E5_structure") if c.structure == "clique")


def _seeds(key):
    exp, label = key.split("/")
    return {c.seed for c in experiment(exp) if c.label == label}


def test_holm_floor_allows_verdicts():
    """Smallest attainable Holm-adjusted Wilcoxon p = m * 2^(1-n); it must leave room below 0.01."""
    for fam, items in {**FAMILIES, **INTERACTIONS}.items():
        m = len(items)
        for keys in items:
            n = len(set.intersection(*(_seeds(k) for k in keys)))
            assert m * 2.0 ** (1 - n) <= 1e-3, (fam, keys, n)


def test_comparisons_reference_existing_labels():
    labels = {f"{c.exp}/{c.label}" for e in NO_DATA_NEEDED for c in experiment(e)}
    refs = [x for pairs in FAMILIES.values() for p in pairs for x in p]
    refs += [x for quads in INTERACTIONS.values() for q in quads for x in q]
    missing = [r for r in refs if r not in labels]
    assert not missing, missing
```

- [ ] **Step 2: Run the tests to verify they fail.** Run: `pytest tests/test_grid.py -v`. Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/experiments/grid.py`

```python
"""Every experiment in the study, as lists of RunConfig. Labels are referenced as "<exp>/<label>"."""
from __future__ import annotations

import json
from pathlib import Path

from hypershift.config import RunConfig, load_yaml

SEEDS_FINAL = tuple(range(25))
SEEDS_SWEEP = tuple(range(15))   # verdict-capable: min p = 6e-5, Holm floor for m = 8 is 5e-4
SEEDS_SCREEN = tuple(range(5))   # descriptive only (min p = 0.0625)
MEMORY_LIGHT = {"micro_batch_days": 1}   # for clique/decomposition arms; identical math, less GPU memory
FRESH_DAILY_NAME = "sp500_daily"
FRESH_HOURLY_NAME = "sp500_1h"
GEOMS = {
    "HH": {"temporal": "hyp", "spatial": "hyp"},
    "HE": {"temporal": "hyp", "spatial": "euc"},
    "EH": {"temporal": "euc", "spatial": "hyp"},   # = paper's "TCONV + DHHAN"
    "EE": {"temporal": "euc", "spatial": "euc"},   # ~ STHGCN-style Euclidean hypergraph model
}
EXPERIMENTS = ["E0_budget", "E_tune", "E_attn", "E1_main", "E2_geometry", "E3_hhn", "E4_grouping",
               "E5_structure", "E6_decompose", "E7_hubs", "E8_universe", "E9_fresh_daily", "E10_hourly"]


def _read(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text()) if p.suffix == ".json" else load_yaml(p)


def _geo(g: str, **kw) -> dict:
    """Precedence (later wins): geometry < tuned.json[g] < global.yaml < chosen.yaml < experiment kwargs."""
    tuned = _read("results/tuned.json").get(g, {})
    return {**GEOMS[g], **tuned, **_read("configs/global.yaml"), **_read("configs/chosen.yaml"), **kw}


def _mk(exp: str, label: str, seeds, **kw) -> list[RunConfig]:
    return [RunConfig(exp=exp, label=label, seed=s, **kw) for s in seeds]


def _hub_thresholds() -> list[int]:
    from hypershift.data.hypergraph import build_rsr_hypergraph, hub_schedule
    return hub_schedule(build_rsr_hypergraph(RunConfig().data_root, "NYSE"))


def experiment(name: str) -> list[RunConfig]:
    E: list[RunConfig] = []
    if name == "E0_budget":
        E += _mk(name, "HH", (0,), **_geo("HH", epochs=3, patience=100))
        E += _mk(name, "HH_clique", (0,), **_geo("HH", epochs=1, patience=100, structure="clique", **MEMORY_LIGHT))
    elif name == "E_tune":
        for g in GEOMS:                               # every geometry gets the same budget (fairness rule)
            for lr in (5e-4, 1e-3):
                for alpha in (0.1, 1.0, 10.0):
                    kw = {**GEOMS[g], **_read("configs/global.yaml"), "lr": lr, "alpha": alpha}
                    E += _mk(name, f"{g}_lr{lr}_a{alpha}", (0, 1, 2), **kw)
    elif name == "E_attn":
        for sc in ("mobius", "concat"):
            for di in ("mult", "neg", "off"):
                E += _mk(name, f"{sc}_{di}", SEEDS_SCREEN, **_geo("HH", attn_score=sc, attn_dist=di))
    elif name == "E1_main":
        E += _mk(name, "THINK", SEEDS_FINAL, **_geo("HH"))
        E += _mk(name, "THINK_paperNorm", SEEDS_FINAL, **_geo("HH", norm="paper"))
        # Paper/STHAN-SR protocol: full-series-max norm, all 100 epochs, test read every epoch -> test_oracle_sr.
        E += _mk(name, "THINK_paperProtocol", SEEDS_FINAL, **_geo("HH", norm="paper", epochs=100, patience=1000))
        E += _mk(name, "THINK_NASDAQ", SEEDS_FINAL, **_geo("HH", market="NASDAQ", alpha=0.1))
        E += _mk(name, "THINK_NASDAQ_paperNorm", SEEDS_FINAL, **_geo("HH", market="NASDAQ", alpha=0.1, norm="paper"))
        E += _mk(name, "THINK_shuffled", (0, 1, 2), **_geo("HH", shuffle_train_labels=True))
    elif name == "E2_geometry":
        for g in ("HE", "EH", "EE"):
            E += _mk(name, g, SEEDS_FINAL, **_geo(g))
    elif name == "E3_hhn":
        E += _mk(name, "HHN", SEEDS_FINAL, **_geo("HH", attn_dist="off"))
    elif name == "E4_grouping":
        # "both" (industry+wiki) = E1_main/THINK and E2_geometry/EE; "none" = E5_structure/*_none. Not re-run here.
        groups = {"industry": {"sources": ("industry",)}, "wiki": {"sources": ("wiki",)},
                  "random": {"sources": ("industry", "wiki", "random")}, "corr": {"sources": ("corr",)}}
        for g in ("HH", "EE"):
            for gname, kw in groups.items():
                E += _mk(name, f"{g}_{gname}", SEEDS_SWEEP, **_geo(g, **kw))
    elif name == "E5_structure":
        for g in ("HH", "EE"):
            E += _mk(name, f"{g}_clique", SEEDS_FINAL, **_geo(g, structure="clique", **MEMORY_LIGHT))
            E += _mk(name, f"{g}_none", SEEDS_FINAL, **_geo(g, structure="none"))
    elif name == "E6_decompose":
        for g in ("HH", "EE"):
            for s in (15, 9, 5, 3):
                E += _mk(name, f"{g}_large_{s}", SEEDS_SCREEN,
                         **_geo(g, decompose_mode="large_first", decompose_size=s, **MEMORY_LIGHT))
                E += _mk(name, f"{g}_small_{s}", SEEDS_SCREEN,
                         **_geo(g, decompose_mode="small_first", decompose_size=s, **MEMORY_LIGHT))
    elif name == "E7_hubs":
        for g in ("HH", "EE"):
            for th in _hub_thresholds():
                E += _mk(name, f"{g}_hub{th}", SEEDS_SCREEN, **_geo(g, drop_hub_degree=th))
    elif name == "E8_universe":
        for g in ("HH", "EE"):
            for n in (50, 100, 250, 500, 1000):
                for u in (0, 1, 2):
                    E += _mk(name, f"{g}_N{n}_u{u}", SEEDS_SCREEN, **_geo(g, universe_size=n, universe_seed=u))
    elif name == "E9_fresh_daily":
        for g in ("HH", "EE"):
            for st in ("hyper", "clique", "none"):
                E += _mk(name, f"{g}_{st}", SEEDS_SWEEP, **_geo(g, market="FRESH", fresh_name=FRESH_DAILY_NAME,
                                                               sources=("subindustry",), structure=st, alpha=1.0))
    elif name == "E10_hourly":
        # hday: 112 hourly bars = 16 trading days, first kernel 28 bars = 4 days -> 4 steps, exactly like the
        # daily arm (16 days, kernel 4). Same look-back, target, trades and architecture; only granularity differs.
        arms = {"daily": (f"{FRESH_HOURLY_NAME}_daily", 252, {}),
                "hday": (f"{FRESH_HOURLY_NAME}_hday", 252, {"seq": 112, "kernel": 28}),
                "hourly": (f"{FRESH_HOURLY_NAME}_hourly", 1764, {})}
        for arm, (fresh, ppy, extra) in arms.items():
            for g in ("HH", "EE"):
                E += _mk(name, f"{arm}_{g}", SEEDS_SWEEP, **_geo(g, market="FRESH", fresh_name=fresh,
                                                                sources=("subindustry",), periods_per_year=ppy,
                                                                alpha=1.0, **extra))
    else:
        raise KeyError(name)
    return E


FAMILIES = {
    "Q0_normalization_leak": [("E1_main/THINK_paperNorm", "E1_main/THINK")],
    "Q2_hyperbolic": [("E1_main/THINK", "E2_geometry/EE"), ("E1_main/THINK", "E2_geometry/EH"),
                      ("E1_main/THINK", "E2_geometry/HE"), ("E2_geometry/EH", "E2_geometry/EE"),
                      ("E2_geometry/HE", "E2_geometry/EE")],
    "Q_attention": [("E1_main/THINK", "E3_hhn/HHN")],
    "Q3_hyperedges": [("E1_main/THINK", "E5_structure/HH_clique"), ("E1_main/THINK", "E5_structure/HH_none"),
                      ("E2_geometry/EE", "E5_structure/EE_clique"), ("E2_geometry/EE", "E5_structure/EE_none")],
    "Q1_grouping": [(both, f"E4_grouping/{g}_{x}")
                    for g, both in (("HH", "E1_main/THINK"), ("EE", "E2_geometry/EE"))
                    for x in ("industry", "wiki", "random", "corr")],
    "Q6_daily_fresh": [("E9_fresh_daily/HH_hyper", "E9_fresh_daily/EE_hyper"),
                       ("E9_fresh_daily/HH_hyper", "E9_fresh_daily/HH_clique"),
                       ("E9_fresh_daily/HH_hyper", "E9_fresh_daily/HH_none")],
    "Q7_hourly": [("E10_hourly/hday_HH", "E10_hourly/daily_HH"), ("E10_hourly/hday_EE", "E10_hourly/daily_EE"),
                  ("E10_hourly/hday_HH", "E10_hourly/hday_EE"), ("E10_hourly/hourly_HH", "E10_hourly/hourly_EE")],
}
INTERACTIONS = {
    "Q4_interaction": [
        ("E1_main/THINK", "E5_structure/HH_clique", "E2_geometry/EE", "E5_structure/EE_clique"),
        ("E1_main/THINK", "E5_structure/HH_none", "E2_geometry/EE", "E5_structure/EE_none"),
    ],
}
```

`scripts/run_grid.py`:
```python
"""python scripts/run_grid.py E1_main E2_geometry [--labels THINK EE] [--seeds 0-14] [--dry-run] [--set k=v ...]

--labels keeps only those labels (matched before --set is applied); --seeds replaces each label's seed list.
"""
import argparse
import json
from collections import Counter
from dataclasses import replace

from tqdm import tqdm

from hypershift.config import apply_overrides, config_from_dict
from hypershift.experiments.grid import experiment
from hypershift.run import parse_seeds
from hypershift.train.loop import train_one_run

ap = argparse.ArgumentParser()
ap.add_argument("exps", nargs="+")
ap.add_argument("--labels", nargs="*", default=None)
ap.add_argument("--seeds", default=None, help="e.g. 0-14 or 0,3,7; default = the grid's seeds")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()
base = [c for e in args.exps for c in experiment(e) if args.labels is None or c.label in args.labels]
if args.labels:
    unknown = set(args.labels) - {c.label for c in base}
    if unknown:
        raise SystemExit(f"unknown labels for {args.exps}: {sorted(unknown)}")
if args.seeds is not None:
    first = {}
    for c in base:
        first.setdefault((c.exp, c.label), c)
    base = [replace(c, seed=s) for c in first.values() for s in parse_seeds(args.seeds)]
cfgs = [config_from_dict(apply_overrides(c.to_dict(), args.set)) for c in base]
todo = [c for c in cfgs if not (c.run_dir() / "metrics.json").exists()]
print(f"{len(cfgs)} configs, {len(todo)} to run")
if args.dry_run:
    for (exp, label), n in sorted(Counter((c.exp, c.label) for c in cfgs).items()):
        print(f"  {exp}/{label}: {n} seeds")
    raise SystemExit(0)
for cfg in tqdm(todo):
    try:
        train_one_run(cfg)
        (cfg.run_dir() / "failed.json").unlink(missing_ok=True)   # a successful re-run clears an old failure
    except (FloatingPointError, RuntimeError) as err:   # RuntimeError covers CUDA OOM
        cfg.run_dir().mkdir(parents=True, exist_ok=True)
        (cfg.run_dir() / "failed.json").write_text(json.dumps({"error": repr(err)}))
        print("FAILED", cfg.exp, cfg.label, cfg.seed, repr(err))
```

`scripts/time_budget.py`:
```python
"""Decision node D7. Times HH (3 epochs) and HH_clique (1 epoch) on full NYSE, then estimates total GPU hours."""
import json

from hypershift.experiments.grid import EXPERIMENTS, experiment
from hypershift.train.loop import train_one_run

sec = {}
for cfg in experiment("E0_budget"):
    sec[cfg.label] = train_one_run(cfg)["sec_per_epoch"]
print(json.dumps(sec, indent=2))
t_run_min = sec["HH"] * 60 / 60
print(f"t_run ~ {t_run_min:.1f} min (60 epochs incl. early stopping)")
total = 0.0
for e in EXPERIMENTS:
    if e in ("E0_budget", "E7_hubs", "E9_fresh_daily", "E10_hourly"):
        continue
    for c in experiment(e):
        per_epoch = sec["HH_clique"] if c.structure == "clique" else sec["HH"]
        per_epoch *= (c.universe_size / 1737) if c.universe_size else 1.0
        n_epochs = c.epochs if c.patience >= c.epochs else min(c.epochs, 60)   # no-early-stop arms run all epochs
        total += per_epoch * n_epochs
print(f"estimated total for RSR experiments: {total / 3600:.1f} GPU hours")
print("D7: <=6 min/run keep; 6-15 -> configs/global.yaml batch_days: 4; >15 -> batch_days: 8, epochs: 60")
```

- [ ] **Step 4: Run the tests.** Run: `pytest tests/test_grid.py -v`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add src/hypershift/experiments/grid.py scripts/run_grid.py scripts/time_budget.py tests/test_grid.py
git commit -m "feat(experiments): full experiment grid, resumable runner, budget estimator"
```

---

### Task 13: Aggregation, tuning/attention selection, tables and figures

**Files:**
- Create: `scripts/aggregate.py`, `scripts/plots.py`

**Interfaces:**
- Consumes: the run-folder contract (Task 8), `FAMILIES`/`INTERACTIONS` (Task 12), stats (Task 10), `topk_daily_returns_net` and `sharpe` (Task 7), and baseline JSON files (Task 9).
- Produces:
  - `results/tables/summary.csv`: one row per exp/label, with mean and std of `test_sr, test_ndcg5, test_irr, test_mse, test_mdd, val_sr, test_oracle_sr`, plus `n_seeds` and `n_failed`.
  - `results/tables/comparisons.csv` and `results/tables/comparisons.md`: family, A, B, n, meanA, meanB, diff, p, ci_lo, ci_hi, p_holm, p_floor, verdict. Means, CI and p all use only the seeds common to the compared arms.
  - `results/tables/table2_like.md`: our numbers next to the paper's, with both the validation-selected SR and the best-test-epoch (paper-protocol) SR.
  - `results/tables/universe.md` (Q5): per-(N, universe) cells and Spearman trends.
  - `results/tables/costs.md`: E10 net Sharpe at 0/5/10 bps.
  - `--select-tuning` writes `results/tuned.json`. `--select-attn` writes `configs/chosen.yaml`.
  - Figures in `results/figures/`: `geometry.png`, `attention.png`, `fig3a_decompose.png`, `fig3b_hubs.png`, `universe.png`.

- [ ] **Step 1: Implement** `scripts/aggregate.py`

```python
"""python scripts/aggregate.py [--select-tuning] [--select-attn] [--costs]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hypershift.eval.metrics import sharpe, topk_daily_returns_net
from hypershift.eval.stats import (
    holm, sharpe_contrast_ci, sharpe_diff_ci, verdict, wilcoxon_one_sample, wilcoxon_paired,
)
from hypershift.experiments.grid import FAMILIES, INTERACTIONS

R = Path("results")
T = R / "tables"
# NYSE SR from Table II. 1.18 sits only on the paper-protocol row, so it is never read against the honest THINK row.
PAPER = {"E1_main/THINK_paperProtocol": 1.18, "E2_geometry/EH": 1.14}


def load_runs() -> pd.DataFrame:
    rows = []
    for mf in R.glob("*/*/seed_*/metrics.json"):
        m = json.loads(mf.read_text())
        if "val" not in m:                     # e.g. Task 18 classification runs: not a ranking run
            continue
        exp, label, seed = mf.parts[-4], mf.parts[-3], int(mf.parts[-2].split("_")[1])
        rows.append({"exp": exp, "label": label, "key": f"{exp}/{label}", "seed": seed,
                     "val_sr": m["val"]["sr"], "test_sr": m["test"]["sr"], "test_ndcg5": m["test"]["ndcg5"],
                     "test_irr": m["test"]["irr"], "test_mse": m["test"]["mse"], "test_mdd": m["test"]["mdd"],
                     "test_ndcg_sthan": m["test"]["ndcg_sthan"], "test_oracle_sr": m["test_oracle_sr"],
                     "epochs_run": m["epochs_run"], "num_edges": m.get("num_edges", np.nan),
                     "covered_frac": m.get("covered_frac", np.nan), "dir": str(mf.parent)})
    return pd.DataFrame(rows)


def n_failed(key: str) -> int:
    exp, label = key.split("/")
    return len(list((R / exp / label).glob("seed_*/failed.json")))


def summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["test_sr", "test_ndcg5", "test_irr", "test_mse", "test_mdd", "val_sr", "test_oracle_sr"]
    g = df.groupby("key")[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_seeds"] = df.groupby("key").size()
    g["n_failed"] = [n_failed(k) for k in g.index]
    return g.reset_index()


def seed_series(df, key, seeds=None):
    """Per-seed test SR, and the daily top-k return averaged over `seeds` (default: all seeds of `key`).

    Always pass the seeds common to every compared arm: an ensemble over more seeds is smoother, so its
    Sharpe would be mechanically higher and bias the bootstrap CI.
    """
    sub = df[df.key == key].sort_values("seed")
    if seeds is not None:
        sub = sub[sub.seed.isin(list(seeds))]
    daily = np.mean([np.load(Path(d) / "test_daily.npy") for d in sub.dir], axis=0)
    return sub.set_index("seed")["test_sr"], daily


def _common_seeds(df, keys):
    common = None
    for k in keys:
        s = set(df[df.key == k].seed)
        common = s if common is None else common & s
    return sorted(common)


def compare_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, pairs in FAMILIES.items():
        fam_rows = []
        for a, b in pairs:
            if a not in set(df.key) or b not in set(df.key):
                continue
            common = _common_seeds(df, (a, b))
            sa, da = seed_series(df, a, common)
            sb, db = seed_series(df, b, common)
            ci = sharpe_diff_ci(da, db)
            fam_rows.append({"family": fam, "A": a, "B": b, "n": len(common), "meanA": sa.mean(), "meanB": sb.mean(),
                             "diff": sa.mean() - sb.mean(),
                             "p": wilcoxon_paired(sa[common], sb[common]) if len(common) >= 6 else np.nan,
                             "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
        rows += _holm(fam_rows)
    for fam, quads in INTERACTIONS.items():
        fam_rows = []
        for q in quads:
            if not all(k in set(df.key) for k in q):
                continue
            common = _common_seeds(df, q)
            s = [seed_series(df, k, common) for k in q]
            inter = (s[0][0][common] - s[1][0][common]) - (s[2][0][common] - s[3][0][common])
            ci = sharpe_contrast_ci([x[1] for x in s], [1, -1, -1, 1])
            fam_rows.append({"family": fam, "A": f"({q[0]} - {q[1]})", "B": f"({q[2]} - {q[3]})", "n": len(common),
                             "meanA": np.nan, "meanB": np.nan, "diff": inter.mean(),
                             "p": wilcoxon_one_sample(inter) if len(common) >= 6 else np.nan,
                             "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
        rows += _holm(fam_rows)
    return pd.DataFrame(rows)


def _holm(fam_rows):
    ps = {i: r["p"] for i, r in enumerate(fam_rows) if not np.isnan(r["p"])}
    adj = holm(ps) if ps else {}
    m = max(len(ps), 1)
    for i, r in enumerate(fam_rows):
        r["p_holm"] = adj.get(i, np.nan)
        r["p_floor"] = min(1.0, m * 2.0 ** (1 - r["n"]))       # smallest Holm p attainable with n seeds
        if i not in adj or (r["p_floor"] >= 0.01 and r["p_holm"] >= 0.01):
            r["verdict"] = "INSUFFICIENT SEEDS"
        else:
            r["verdict"] = verdict(r["p_holm"], r["ci_lo"], r["ci_hi"])
    return fam_rows


def select_tuning(df):
    """Best mean validation SR per geometry over E_tune labels '<G>_lr<lr>_a<alpha>'."""
    tune = df[df.exp == "E_tune"].groupby("label")["val_sr"].mean()
    out = {}
    for g in ("HH", "HE", "EH", "EE"):
        cand = tune[tune.index.str.startswith(g + "_")]
        if cand.empty:
            raise SystemExit(f"no E_tune runs for {g}; run `python scripts/run_grid.py E_tune` first")
        _, lr, a = cand.idxmax().split("_")
        out[g] = {"lr": float(lr[2:]), "alpha": float(a[1:])}
    (R / "tuned.json").write_text(json.dumps(out, indent=2))
    print(tune.to_string(), "\ntuned:", out)


def select_attn(df):
    att = df[df.exp == "E_attn"].groupby("label")["val_sr"].agg(["mean", "std"])
    att = att[~att.index.str.endswith("_off")].sort_values("mean", ascending=False)
    top = att.index[0]
    choice = top
    if "mobius_mult" in att.index and att.loc[top, "mean"] - att.loc["mobius_mult", "mean"] < att.loc[top, "std"]:
        choice = "mobius_mult"                    # decision node D8: keep the paper's literal form unless clearly beaten
    sc, di = choice.split("_")
    Path("configs/chosen.yaml").write_text(yaml.safe_dump({"attn_score": sc, "attn_dist": di}))
    print(att, "\nchosen:", choice)


def costs_table(df):
    lines = ["| arm | cost bps | SR mean | SR std |", "|---|---|---|---|"]
    for key in sorted(k for k in df.key.unique() if k.startswith("E10_hourly/")):
        ppy = 1764 if "/hourly_" in key else 252
        for bps in (0, 5, 10):
            srs = []
            for d in df[df.key == key].dir:
                p, g, m = (np.load(Path(d) / f) for f in ("test_pred.npy", "test_gt.npy", "test_mask.npy"))
                srs.append(sharpe(topk_daily_returns_net(p, g, m, 5, bps), ppy))
            lines.append(f"| {key} | {bps} | {np.mean(srs):.3f} | {np.std(srs):.3f} |")
    (T / "costs.md").write_text("\n".join(lines) + "\n")


def table2_like(summ):
    """'our test SR' = validation-selected epoch (honest). 'best-test-epoch SR' = the paper/STHAN-SR protocol
    (optimistic; comparable to the paper's 1.18 only on the THINK_paperProtocol row)."""
    lines = ["| model | our test SR (val-selected) | std | best-test-epoch SR | our NDCG@5 | paper SR |",
             "|---|---|---|---|---|---|"]
    paper = PAPER
    for _, r in summ.iterrows():
        if r.key.startswith(("E1_main/", "E2_geometry/", "E3_hhn/", "E5_structure/")):
            lines.append(f"| {r.key} | {r.test_sr_mean:.3f} | {r.test_sr_std:.3f} | {r.test_oracle_sr_mean:.3f} | "
                         f"{r.test_ndcg5_mean:.3f} | {paper.get(r.key, '')} |")
    for f in sorted((R / "baselines").glob("NYSE.json")):
        for k, v in json.loads(f.read_text()).items():
            lines.append(f"| baseline {k} (NYSE) | {v['sr']:.3f} | | | {v.get('ndcg5', float('nan')):.3f} | |")
    (T / "table2_like.md").write_text("\n".join(lines) + "\n")


def universe_table(df):
    """Q5. One row per (N, universe) cell plus the full-universe point; Spearman trends over the cells."""
    from scipy.stats import spearmanr

    hyp_file = R / "hyperbolicity.json"
    hyp = json.loads(hyp_file.read_text()) if hyp_file.exists() else {}
    cells = [(n, u, f"E8_universe/HH_N{n}_u{u}", f"E8_universe/EE_N{n}_u{u}", f"NYSE_N{n}_u{u}")
             for n in (50, 100, 250, 500, 1000) for u in (0, 1, 2)]
    cells.append((1737, 0, "E1_main/THINK", "E2_geometry/EE", "NYSE"))
    rows = []
    for n, u, kh, ke, bname in cells:
        if kh not in set(df.key) or ke not in set(df.key):
            continue
        common = _common_seeds(df, (kh, ke))
        sh, _ = seed_series(df, kh, common)
        se, _ = seed_series(df, ke, common)
        bfile = R / "baselines" / f"{bname}.json"
        rnd = json.loads(bfile.read_text())["random"]["sr"] if bfile.exists() else np.nan
        hrow = df[df.key == kh]
        hkey = "NYSE_full" if n == 1737 else f"NYSE_N{n}_u{u}"      # keys written by scripts/hyperbolicity.py
        rows.append({"N": n, "u": u, "n_seeds": len(common), "HH_sr": sh.mean(), "EE_sr": se.mean(),
                     "gap_HH_minus_EE": (sh - se).mean(), "random5_sr": rnd,
                     "HH_excess": sh.mean() - rnd, "EE_excess": se.mean() - rnd,
                     "num_edges": hrow.num_edges.mean(), "covered_frac": hrow.covered_frac.mean(),
                     "delta_hg": hyp.get(hkey, {}).get("hg", {}).get("delta_max", np.nan)})
    if not rows:
        return
    U = pd.DataFrame(rows)
    trend = []
    for x, y in (("N", "gap_HH_minus_EE"), ("N", "HH_excess"), ("N", "EE_excess"), ("delta_hg", "gap_HH_minus_EE")):
        ok = U[[x, y]].dropna()
        if len(ok) >= 5 and ok[x].nunique() > 1:
            rho, p = spearmanr(ok[x], ok[y])
            trend.append({"x": x, "y": y, "cells": len(ok), "spearman_rho": rho, "p": p,
                          "verdict": "STRONG trend" if p < 0.01 else "NO EVIDENCE of a trend"})
    text = "## Cells (E8: 5 seeds each; N=1737 row = E1/E2 arms)\n\n" + U.to_markdown(index=False, floatfmt=".3f")
    text += "\n\n## Trends\n\n" + (pd.DataFrame(trend).to_markdown(index=False, floatfmt=".4f") if trend else "none")
    text += ("\n\nCaveat: random subsets thin the hypergraph (see num_edges, covered_frac), so N and "
             "hypergraph density are confounded.\n")
    (T / "universe.md").write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-tuning", action="store_true")
    ap.add_argument("--select-attn", action="store_true")
    ap.add_argument("--costs", action="store_true")
    args = ap.parse_args()
    T.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    if args.select_tuning:
        select_tuning(df)
    if args.select_attn:
        select_attn(df)
    summ = summary(df)
    summ.to_csv(T / "summary.csv", index=False)
    comp = compare_all(df)
    comp.to_csv(T / "comparisons.csv", index=False)
    (T / "comparisons.md").write_text(comp.to_markdown(index=False, floatfmt=".4f") if len(comp) else "none\n")
    table2_like(summ)
    universe_table(df)
    if args.costs:
        costs_table(df)
    print(summ.to_string())
    print(comp.to_string())
```

(`DataFrame.to_markdown` needs `tabulate`. Add `"tabulate>=0.9"` to the `pyproject.toml` dependencies in this step, then run `pip install -e ".[dev]"` again.)

- [ ] **Step 2: Implement** `scripts/plots.py`

```python
"""Figures analogous to paper Fig 2, Fig 3a, Fig 3b, plus geometry bars and the universe-size sweep."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = pd.read_csv("results/tables/summary.csv").set_index("key")
F = Path("results/figures")
F.mkdir(parents=True, exist_ok=True)
COL = {"HH": "#1f77b4", "EE": "#d62728"}
FULL = {"HH": "E1_main/THINK", "EE": "E2_geometry/EE"}   # undecomposed, all hyperedges kept


def bar(keys, names, fname, title):
    keys = [k for k in keys if k in S.index]
    plt.figure(figsize=(6, 3.5))
    plt.bar(range(len(keys)), S.loc[keys, "test_sr_mean"], yerr=S.loc[keys, "test_sr_std"], capsize=4, color="#888")
    plt.xticks(range(len(keys)), [names[k] for k in keys], rotation=20)
    plt.ylabel("Test Sharpe (mean ± std over seeds)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(F / fname, dpi=150)
    plt.close()


bar(["E1_main/THINK", "E2_geometry/HE", "E2_geometry/EH", "E2_geometry/EE"],
    {"E1_main/THINK": "HH (THINK)", "E2_geometry/HE": "hyp-T, euc-S", "E2_geometry/EH": "euc-T, hyp-S",
     "E2_geometry/EE": "EE"}, "geometry.png", "Q2: temporal x spatial geometry (NYSE)")
bar(["E1_main/THINK", "E3_hhn/HHN"], {"E1_main/THINK": "THINK", "E3_hhn/HHN": "HHN (no distance attn)"},
    "attention.png", "Fig 2 analogue (NYSE)")


def curve(pattern, xs_label, fname, title, extra_x=None):
    plt.figure(figsize=(6, 3.5))
    for g in ("HH", "EE"):
        pts = []
        for k in S.index:
            m = re.fullmatch(pattern.format(g=g), k)
            if m:
                pts.append((int(m.group(1)), S.loc[k, "test_sr_mean"], S.loc[k, "test_sr_std"]))
        if extra_x is not None and FULL[g] in S.index:
            pts.append((extra_x, S.loc[FULL[g], "test_sr_mean"], S.loc[FULL[g], "test_sr_std"]))
        if not pts:
            continue
        pts.sort(key=lambda t: -t[0])
        x = np.arange(len(pts))
        plt.errorbar(x, [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", color=COL[g], label=g, capsize=3)
        plt.xticks(x, [str(p[0]) for p in pts])
    plt.xlabel(xs_label)
    plt.ylabel("Test Sharpe")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(F / fname, dpi=150)
    plt.close()


curve(r"E6_decompose/{g}_large_(\d+)", "hyperedges larger than x decomposed into pairs", "fig3a_decompose.png",
      "Fig 3a analogue (NYSE)", extra_x=500)
curve(r"E7_hubs/{g}_hub(\d+)", "hyperedges of nodes with degree >= x removed", "fig3b_hubs.png",
      "Fig 3b analogue (NYSE)")

rows = []
for k in S.index:
    m = re.fullmatch(r"E8_universe/(HH|EE)_N(\d+)_u(\d+)", k)
    if m:
        base = json.loads(Path(f"results/baselines/NYSE_N{m.group(2)}_u{m.group(3)}.json").read_text())
        rows.append({"g": m.group(1), "N": int(m.group(2)), "sr": S.loc[k, "test_sr_mean"],
                     "excess": S.loc[k, "test_sr_mean"] - base["random"]["sr"]})
if rows:
    U = pd.DataFrame(rows).groupby(["g", "N"]).agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for g in ("HH", "EE"):
        u = U[U.g == g]
        ax[0].errorbar(u.N, u[("sr", "mean")], yerr=u[("sr", "std")], marker="o", color=COL[g], label=g, capsize=3)
        ax[1].errorbar(u.N, u[("excess", "mean")], yerr=u[("excess", "std")], marker="o", color=COL[g], label=g, capsize=3)
    for a, t in zip(ax, ["Test Sharpe", "Sharpe minus Random-5 (same universe)"]):
        a.set_xscale("log")
        a.set_xlabel("number of stocks N")
        a.set_ylabel(t)
        a.legend()
    plt.tight_layout()
    plt.savefig(F / "universe.png", dpi=150)
    plt.close()
print("figures in", F)
```

- [ ] **Step 3: Smoke-check with fake results.** Run the tiny grid from Task 8's test config against synthetic data. Easiest: `pytest tests/test_loop.py` leaves runs in pytest tmp dirs, so instead run:
  ```bash
  python -m hypershift.run --config configs/think_nyse.yaml --seeds 0-1 --set exp=E1_main epochs=1
  python scripts/aggregate.py
  ```
  Expected: `results/tables/summary.csv` has the row `E1_main/THINK` with `n_seeds = 2`, and `comparisons.md` says `none` or has only rows whose arms exist. Then `rm -rf results/E1_main results/tables` so the smoke runs don't pollute the real study.

- [ ] **Step 4: Commit**

```bash
git add scripts/aggregate.py scripts/plots.py pyproject.toml
git commit -m "feat(report): aggregation with Holm/bootstrap verdicts, selection helpers, figures"
```

---
### Task 14: Run book, RSR study (Phases A–H)

This task writes no new code. It runs the study in a fixed order, and each phase has a gate. **Do not skip a gate.** Log every decision-node outcome in `docs/report.md` under "Run log" (created in Task 17 Step 1; create the file now with just that heading if it doesn't exist yet).

- [ ] **Phase A — Full test suite**

```bash
pytest -v
```
Gate: everything passes, including the `-m data` tests.
- Failure in Tasks 4–6: go to D4.
- Data-test failures: go to D2/D3.

- [ ] **Phase B — Budget (D7)**

```bash
python scripts/time_budget.py
```
- If D7 says to change `batch_days` or `epochs`, write the change to `configs/global.yaml`, for example:
  ```yaml
  batch_days: 4
  ```
  Then delete `results/E0_budget` and re-run `time_budget.py` to confirm.
- Then run `python scripts/run_grid.py E_tune E_attn E1_main E2_geometry E3_hhn E4_grouping E5_structure E6_decompose E7_hubs E8_universe --dry-run` and record the total run count.
- If the estimated hours are more than 72: ask the user (D7).

- [ ] **Phase C — Baselines, hyperbolicity, tuning**

```bash
python scripts/baselines.py
python scripts/hyperbolicity.py
python scripts/run_grid.py E_tune
python scripts/aggregate.py --select-tuning
```
Gates:
- `results/baselines/NYSE.json`: oracle SR > 10, and random `irr` within ±0.15 of market `irr`. Do not compare random SR with market SR, because 5 stocks are much more volatile than the whole market. If a check fails, the metric code is broken; go to D9 step 6 / D10.
- Hyperbolicity: check against the D16 tolerances and record the values.
- `results/tuned.json` exists and has the keys `HH`, `HE`, `EH` and `EE`. Record the printed tuning table in the run log.
- If two lr/alpha cells are within 0.05 validation SR of each other, the choice between them is noise. That is acceptable; do not add seeds.

- [ ] **Phase D — Attention variant (D8)**

```bash
python scripts/run_grid.py E_attn
python scripts/aggregate.py --select-attn
cat configs/chosen.yaml
```
Gate: `configs/chosen.yaml` exists. Record the selection table printed by aggregate.

- [ ] **Phase E — Main result and leakage audit (D9, D10)**

```bash
python scripts/run_grid.py E1_main
python scripts/aggregate.py
```
(If D7 cut the NASDAQ arms, use `python scripts/run_grid.py E1_main --labels THINK THINK_paperNorm THINK_paperProtocol THINK_shuffled` here and every later time.)

Read `results/tables/summary.csv` and `results/tables/table2_like.md`:
1. D9: `P` = `E1_main/THINK_paperProtocol` `test_oracle_sr_mean` vs the paper's 1.18, and `H` = `E1_main/THINK` `test_sr_mean`.
2. D10.1: the `E1_main/THINK_shuffled` `test_sr_mean` against `random.sr + 2·random.sr_std` from `results/baselines/NYSE.json`.
3. D10.2: the gap between `THINK_paperNorm` and `THINK` (`test_sr_mean`).
4. D10.3: the `test_oracle_sr_mean` vs `test_sr_mean` gap.

Gate: D10.1 does not flag a leak. If it does, STOP the study and debug the windows/features (Task 8 tests) before spending more compute.

If D9 walks the checklist, each attempt gets its own `exp` so it does not overwrite E1 results, runs 5 seeds, and touches only `THINK_paperNorm`. Example for attempt 1:
```bash
python scripts/run_grid.py E1_main --labels THINK_paperNorm --seeds 0-4 --set exp=E1_debug_1 target=price
python scripts/aggregate.py        # compare E1_debug_1/THINK_paperNorm val_sr_mean vs E1_main/THINK_paperNorm seeds 0-4
```
Adopting a fix means:
1. Put it in `configs/global.yaml`, so it applies to all arms and overrides `tuned.json`.
2. Delete `results/E_tune results/E_attn results/E1_main` and `results/tuned.json`.
3. Re-run Phases C (tuning only), D and E. The tuning ran under the old setting, so it must be redone.

- [ ] **Phase F — Core ablations (25 seeds)**

```bash
python scripts/run_grid.py E2_geometry E3_hhn E5_structure
python scripts/aggregate.py
```
(Use the `--labels`/`--seeds` flags from D7 for E5 if D7 reduced the clique seeds.)
- Gate: `n_failed == 0` for every label in `summary.csv`.
- Failed seeds: read `failed.json`, then:
  - `OutOfMemoryError`: go to D15. Its step 1 (`micro_batch_days`) is a per-label re-run that needs no deletion. A successful re-run removes the old `failed.json` by itself.
  - `FloatingPointError` (NaN): go to D5. D5 changes go into `configs/global.yaml` and so affect every arm. Delete **all** `results/E*` folders from E_tune onwards (except `E0_budget`) and redo Phases C–F.
  - Anything else: go to D19.

- [ ] **Phase G — Sweeps**

```bash
python scripts/run_grid.py E4_grouping E6_decompose E7_hubs E8_universe
python scripts/aggregate.py
python scripts/plots.py
```
Gate: the figures exist in `results/figures/`.
- E6, E7 and E8 use 5 seeds. They are **descriptive curves**; the report must not attach verdicts to them.
- If a curve shows a striking effect worth claiming, re-run just those labels, and the label they are compared with, on seeds 0-24. The existing seeds are reused. For example:
  ```bash
  python scripts/run_grid.py E6_decompose --labels HH_large_3 EE_large_3 --seeds 0-24
  ```
  Then compare them in the report with an ad-hoc Wilcoxon, labelled "post-hoc".
- `results/tables/universe.md` answers Q5. Read its Trends table using the Q5 rule in Part 0.5.

- [ ] **Phase H — Commit the study artifacts that belong in git**

```bash
git add configs/ docs/report.md
git commit -m "study: RSR reproduction and ablations run"
```
(`results/` stays git-ignored. Copy the final tables into the report in Task 17.)

---

### Task 15: Fresh data pipeline (daily 2015–2026 and hourly)

**Files:**
- Create: `src/hypershift/data/fresh.py`, `scripts/fetch_fresh.py`, `tests/test_fresh.py`

**Interfaces:**
- Consumes: `MarketData` (Task 2); `Hypergraph`, `canonical` (Task 3).
- Produces (in `hypershift.data.fresh`):
  - `load_universe() -> DataFrame[ticker, sector, subindustry]` (tickers use the `-` convention)
  - `fetch_yf(tickers, interval, start=None, end=None, period=None) -> DataFrame[timestamp,ticker,open,high,low,close,volume]`
  - `fetch_alpaca(tickers, start, end, feed="sip") -> DataFrame` (already RTH hourly)
  - `to_rth_hourly(bars) -> DataFrame` (7 bars per day, starting 09:30…15:30 NY)
  - `to_daily(bars) -> DataFrame`
  - `build_panel(bars, train_frac=0.6, val_frac=0.2, max_missing=0.05, min_coverage=0.5, tickers=None, split_at=None) -> MarketData`
  - `daily_horizon(hourly: MarketData) -> MarketData`
  - `save_panel(data, dir, universe=None)`, `load_panel(dir) -> MarketData`
  - `gics_hypergraph(dir, level) -> Hypergraph`
  - Panel folders `data/fresh/<name>/` containing `panel.npz`, `tickers.txt`, `universe.csv`.

- [ ] **Step 1: Write the failing tests** `tests/test_fresh.py`

```python
import numpy as np
import pandas as pd
import pytest
from hypershift.data.fresh import (
    NY, build_panel, daily_horizon, gics_hypergraph, load_panel, save_panel, to_daily, to_rth_hourly,
)


def make_30min_bars(days, tickers=("AAA", "BBB"), drop_ticker=None):
    rows = []
    for di, d in enumerate(days):
        for k, t in enumerate(pd.date_range(f"{d} 09:00", f"{d} 16:30", freq="30min", tz=NY, inclusive="left")):
            for j, tk in enumerate(tickers):
                if tk == drop_ticker and di % 2 == 0:
                    continue
                px = 100 + 10 * j + di + 0.1 * k
                rows.append({"timestamp": t, "ticker": tk, "open": px, "high": px + .05, "low": px - .05,
                             "close": px, "volume": 100})
    return pd.DataFrame(rows)


def test_to_rth_hourly_anchor_and_premarket():
    h = to_rth_hourly(make_30min_bars(["2024-03-04", "2024-03-05"]))
    a = h[h.ticker == "AAA"].sort_values("timestamp")
    assert len(a) == 14
    hhmm = a.timestamp.dt.tz_convert(NY).dt.strftime("%H:%M").unique().tolist()
    assert hhmm == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    assert a.close.iloc[0] == pytest.approx(100.2)      # 09:30 bin closes with the 10:00 half-hour bar
    assert a.close.iloc[6] == pytest.approx(101.3)      # 15:30 bin = last RTH bar; 16:00 bar excluded


def test_to_daily():
    d = to_daily(to_rth_hourly(make_30min_bars(["2024-03-04", "2024-03-05"])))
    a = d[d.ticker == "AAA"].sort_values("timestamp")
    assert len(a) == 2 and a.close.iloc[0] == pytest.approx(101.3)


def _hourly_panel():
    days = [x.strftime("%Y-%m-%d") for x in pd.bdate_range("2024-01-02", periods=30)]
    bars = to_rth_hourly(make_30min_bars(days, tickers=("AAA", "BBB", "CCC"), drop_ticker="CCC"))
    return build_panel(bars)


def test_build_panel_filters_and_normalizes():
    p = _hourly_panel()
    assert p.tickers == ["AAA", "BBB"]                   # CCC missing ~50% -> dropped (D13)
    assert p.features.shape == (2, 210, 5)
    assert (p.valid_index, p.test_index) == (126, 168)
    np.testing.assert_allclose(p.features[:, :126, -1].max(axis=1), 1.0, rtol=1e-6)
    c = p.base_price
    np.testing.assert_allclose(p.gt[:, 5], c[:, 5] / c[:, 4] - 1, atol=1e-6)   # float32 prices -> absolute tol


def test_daily_horizon_targets():
    p = _hourly_panel()
    dh = daily_horizon(p)
    assert dh.eligible_ends.tolist() == [7 * i + 6 for i in range(29)]
    e, e2 = 6, 13
    np.testing.assert_allclose(dh.gt[:, e + 1], p.base_price[:, e2] / p.base_price[:, e] - 1, atol=1e-6)
    assert pd.to_datetime(p.timestamps[0], utc=True).year == 2024     # timestamps stored as ns


def test_save_load_and_gics(tmp_path):
    p = _hourly_panel()
    uni = pd.DataFrame({"ticker": ["AAA", "BBB", "ZZZ"], "sector": ["Tech", "Tech", "Energy"],
                        "subindustry": ["Semis", "Software", "Oil"]})
    save_panel(p, tmp_path / "x", uni)
    q = load_panel(tmp_path / "x")
    np.testing.assert_allclose(q.features, p.features)
    assert q.tickers == p.tickers and q.valid_index == p.valid_index and q.eligible_ends is None
    assert gics_hypergraph(tmp_path / "x", "sector").edges == ((0, 1),)
    assert gics_hypergraph(tmp_path / "x", "subindustry").edges == ()
    save_panel(daily_horizon(p), tmp_path / "y", uni)
    assert load_panel(tmp_path / "y").eligible_ends is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pip install -e ".[dev,fresh]"
pytest tests/test_fresh.py -v
```
Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/data/fresh.py`

```python
"""Fresh S&P 500 panels (yfinance / Alpaca) in RSR-compatible MarketData form. Decision nodes D12, D13, D17."""
from __future__ import annotations

import io
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hypershift.data.hypergraph import Hypergraph, canonical
from hypershift.data.rsr import MarketData

NY = "America/New_York"
UNIVERSE_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
COLS = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]


def load_universe() -> pd.DataFrame:
    try:
        df = pd.read_csv(UNIVERSE_URL)
    except Exception:
        import requests
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        df = pd.read_html(io.StringIO(html))[0]
    df = df.rename(columns={"Symbol": "ticker", "GICS Sector": "sector", "GICS Sub-Industry": "subindustry"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df[["ticker", "sector", "subindustry"]].drop_duplicates("ticker").reset_index(drop=True)


def fetch_yf(tickers, interval, start=None, end=None, period=None) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), 50):
        chunk = list(tickers[i:i + 50])
        raw = yf.download(chunk, interval=interval, start=start, end=end, period=period, group_by="ticker",
                          auto_adjust=True, prepost=False, threads=True, progress=False)
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([chunk, raw.columns])
        for t in chunk:
            if t not in raw.columns.get_level_values(0):
                continue
            d = raw[t].dropna(how="all").copy()
            if d.empty:
                continue
            d.columns = [str(c).lower() for c in d.columns]
            d.index.name = "timestamp"
            d = d.reset_index()
            d["ticker"] = t
            frames.append(d[COLS])
    return pd.concat(frames, ignore_index=True)


def fetch_alpaca(tickers, start, end, feed: str = "sip") -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    frames = []
    for i in range(0, len(tickers), 50):
        chunk = [t.replace("-", ".") for t in tickers[i:i + 50]]
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame(30, TimeFrameUnit.Minute),
                               start=pd.Timestamp(start, tz=NY), end=pd.Timestamp(end, tz=NY),
                               adjustment=Adjustment.ALL, feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
        df = client.get_stock_bars(req).df
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"symbol": "ticker"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        frames.append(to_rth_hourly(df[COLS]))          # resample per chunk to keep memory small
        print(f"alpaca: {i + len(chunk)}/{len(tickers)} tickers")
    return pd.concat(frames, ignore_index=True)


def to_rth_hourly(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(minutes >= 570) & (minutes < 960)]                          # 09:30 <= t < 16:00
    g = df.set_index("timestamp").groupby("ticker").resample("60min", origin="start_day", offset="30min")
    # per-column reducers: pandas 3 turns a dict .agg() on a grouped resample into a cross-product MultiIndex
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    out = out.dropna(subset=["close"]).reset_index()
    return out[COLS]


def to_daily(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df["day"] = df["timestamp"].dt.normalize()
    out = (df.sort_values("timestamp").groupby(["ticker", "day"])
             .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                  volume=("volume", "sum"))
             .reset_index().rename(columns={"day": "timestamp"}))
    return out[COLS]


def build_panel(bars: pd.DataFrame, train_frac=0.6, val_frac=0.2, max_missing=0.05, min_coverage=0.5,
                tickers=None, split_at=None, ma_windows=(5, 10, 20, 30)) -> MarketData:
    df = bars.copy()
    ts_raw = pd.to_datetime(df["timestamp"])
    # yfinance daily bars can be tz-naive dates: treat them as NY dates, not UTC midnight
    df["timestamp"] = ts_raw.dt.tz_localize(NY) if ts_raw.dt.tz is None else ts_raw.dt.tz_convert(NY)
    close = df.pivot_table(index="timestamp", columns="ticker", values="close", aggfunc="last").sort_index()
    close = close[close.notna().mean(axis=1) >= min_coverage]
    if tickers is not None:
        close = close.reindex(columns=list(tickers)).dropna(axis=1, how="all")
    else:
        close = close.loc[:, close.isna().mean(axis=0) <= max_missing]
    mask = close.notna().to_numpy().T.astype(np.float32)
    filled = close.ffill().bfill()
    c = filled.to_numpy().T.astype(np.float64)
    ts = pd.DatetimeIndex(close.index)
    T = c.shape[1]
    if split_at is None:
        vi, ti = int(T * train_frac), int(T * (train_frac + val_frac))
    else:
        vi, ti = (int(ts.searchsorted(pd.Timestamp(x))) for x in split_at)
    mas = [filled.rolling(w, min_periods=1).mean().to_numpy().T for w in ma_windows]
    scale = c[:, :vi].max(axis=1)
    scale = np.where(scale > 0, scale, 1.0)[:, None]
    feats = np.stack(mas + [c], axis=2) / scale[:, :, None]
    gt = np.zeros_like(c)
    gt[:, 1:] = (c[:, 1:] / c[:, :-1] - 1) * mask[:, 1:] * mask[:, :-1]
    return MarketData(list(close.columns), feats.astype(np.float32), mask, gt.astype(np.float32),
                      (c / scale).astype(np.float32), vi, ti, timestamps=ts.as_unit("ns").asi8.copy())


def daily_horizon(h: MarketData) -> MarketData:
    """Hourly inputs, but predict close(next day) / close(today) - 1 from the last bar of each day."""
    day = pd.to_datetime(h.timestamps, utc=True).tz_convert(NY).normalize().as_unit("ns").asi8
    last = np.nonzero(np.r_[day[1:] != day[:-1], True])[0]
    gt = h.gt.copy()
    for e, e2 in zip(last[:-1], last[1:]):
        gt[:, e + 1] = (h.base_price[:, e2] / h.base_price[:, e] - 1) * h.mask[:, e] * h.mask[:, e2]
    return replace(h, gt=gt.astype(np.float32), eligible_ends=last[:-1].copy())


def save_panel(data: MarketData, panel_dir, universe: pd.DataFrame | None = None) -> None:
    panel_dir = Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        panel_dir / "panel.npz", features=data.features, mask=data.mask, gt=data.gt, base_price=data.base_price,
        valid_index=np.array(data.valid_index), test_index=np.array(data.test_index),
        timestamps=data.timestamps if data.timestamps is not None else np.array([], dtype=np.int64),
        has_eligible=np.array(data.eligible_ends is not None),
        eligible_ends=data.eligible_ends if data.eligible_ends is not None else np.array([], dtype=np.int64),
    )
    (panel_dir / "tickers.txt").write_text("\n".join(data.tickers) + "\n")
    if universe is not None:
        universe[universe.ticker.isin(data.tickers)].to_csv(panel_dir / "universe.csv", index=False)


def load_panel(panel_dir) -> MarketData:
    panel_dir = Path(panel_dir)
    z = np.load(panel_dir / "panel.npz")
    return MarketData(
        tickers=(panel_dir / "tickers.txt").read_text().split(),
        features=z["features"], mask=z["mask"], gt=z["gt"], base_price=z["base_price"],
        valid_index=int(z["valid_index"]), test_index=int(z["test_index"]),
        timestamps=z["timestamps"] if len(z["timestamps"]) else None,
        eligible_ends=z["eligible_ends"] if bool(z["has_eligible"]) else None,
    )


def gics_hypergraph(panel_dir, level: str) -> Hypergraph:
    panel_dir = Path(panel_dir)
    tickers = (panel_dir / "tickers.txt").read_text().split()
    uni = pd.read_csv(panel_dir / "universe.csv").set_index("ticker")
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(tickers):
        if t in uni.index:
            groups.setdefault(str(uni.loc[t, level]), []).append(i)
    return Hypergraph(len(tickers), canonical(groups.values()))
```

- [ ] **Step 4: Run the tests.** Run: `pytest tests/test_fresh.py -v`. Expected: PASS.

  The code was verified against pandas 3.0.6. If an older pandas raises `ValueError: cannot insert ticker, already exists` in `to_rth_hourly`, drop the `ticker` column from `out` before `reset_index()`, then re-run the tests.

- [ ] **Step 5: Implement** `scripts/fetch_fresh.py`

```python
"""Examples:
  python scripts/fetch_fresh.py --source yf --kind daily --start 2015-01-01 --end 2026-09-01 --name sp500_daily
  python scripts/fetch_fresh.py --source alpaca --kind hourly --start 2018-01-01 --end 2026-09-01 --name sp500_1h
  python scripts/fetch_fresh.py --source yf --kind hourly --name sp500_1h          (last ~730 days only)
"""
import argparse
from pathlib import Path

import pandas as pd

from hypershift.data.fresh import (
    NY, build_panel, daily_horizon, fetch_alpaca, fetch_yf, load_universe, save_panel, to_daily, to_rth_hourly,
)

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["yf", "alpaca"], required=True)
ap.add_argument("--kind", choices=["daily", "hourly"], required=True)
ap.add_argument("--start", default=None)
ap.add_argument("--end", default=None)
ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
ap.add_argument("--name", required=True)
ap.add_argument("--limit", type=int, default=0, help="first N tickers only (debugging)")
ap.add_argument("--max-missing", type=float, default=0.05)
args = ap.parse_args()

root = Path("data/fresh")
uni = load_universe()
tickers = uni.ticker.tolist()[: args.limit or None]
print(f"universe: {len(tickers)} tickers (current S&P 500 -> survivorship-biased, see D17)")

if args.kind == "daily":
    if args.source != "yf":
        raise SystemExit("daily bars: use --source yf")
    bars = fetch_yf(tickers, "1d", start=args.start, end=args.end)
    (root / args.name).mkdir(parents=True, exist_ok=True)
    bars.to_csv(root / args.name / "bars.csv.gz", index=False)
    panel = build_panel(bars, max_missing=args.max_missing)
    save_panel(panel, root / args.name, uni)
    print(args.name, panel.features.shape, "valid/test index", panel.valid_index, panel.test_index)
else:
    if args.source == "alpaca":
        bars = fetch_alpaca(tickers, args.start, args.end, feed=args.feed)
    else:
        bars = to_rth_hourly(fetch_yf(tickers, "60m", period="730d"))
    raw_dir = root / f"{args.name}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    bars.to_csv(raw_dir / "hourly_bars.csv.gz", index=False)
    h0 = build_panel(bars, max_missing=args.max_missing)
    if len(h0.tickers) < 300:
        print(f"WARNING D13: only {len(h0.tickers)} tickers survive; consider --max-missing 0.10")
    daily = build_panel(to_daily(bars), tickers=h0.tickers)
    ts_d = pd.to_datetime(daily.timestamps, utc=True).tz_convert(NY)
    hourly = build_panel(bars, tickers=daily.tickers, split_at=(ts_d[daily.valid_index], ts_d[daily.test_index]))
    save_panel(hourly, root / f"{args.name}_hourly", uni)
    save_panel(daily, root / f"{args.name}_daily", uni)
    save_panel(daily_horizon(hourly), root / f"{args.name}_hday", uni)
    for suffix, p in (("hourly", hourly), ("daily", daily)):
        print(suffix, p.features.shape, "valid/test", p.valid_index, p.test_index)
```

- [ ] **Step 6: Tiny live check** (network)

```bash
python scripts/fetch_fresh.py --source yf --kind daily --start 2024-01-01 --end 2024-06-30 --name tiny_daily --limit 20
```
Expected: it prints a shape like `(≈20, ≈124, 5)`. Then `rm -rf data/fresh/tiny_daily`.
- yfinance errors or returns empty: `pip install -U yfinance` and retry once. If it still fails, tell the user yfinance is blocked, and skip E9 unless Alpaca daily is added.

- [ ] **Step 7: Commit**

```bash
git add src/hypershift/data/fresh.py scripts/fetch_fresh.py tests/test_fresh.py
git commit -m "feat(data): fresh S&P 500 daily/hourly panels with same-source daily comparison"
```

---

### Task 16: Run book, fresh daily (E9) and hourly (E10)

- [ ] **Step 1: Fetch the daily data (Q6)**

```bash
python scripts/fetch_fresh.py --source yf --kind daily --start 2015-01-01 --end 2026-09-01 --name sp500_daily
```
Gate: N ≥ 400 and T ≥ 2500.
- N too low: re-run with `--max-missing 0.10` and record it.
- Still N < 300: continue with what survives, and note in the report that the fresh universe is smaller.
- T < 2500: the download stopped early. Delete `data/fresh/sp500_daily` and re-run once; if T is still short, continue and record T.

- [ ] **Step 2: Fetch the hourly data (Q7).** Decision node **D12** chooses the command.

With Alpaca keys:
```bash
export ALPACA_API_KEY=...; export ALPACA_SECRET_KEY=...
python scripts/fetch_fresh.py --source alpaca --kind hourly --start 2018-01-01 --end 2026-09-01 --name sp500_1h
```
- A 403 or permission error on SIP: re-run with `--feed iex`. Note in the report that the IEX feed has thin volume.

Without keys:
```bash
python scripts/fetch_fresh.py --source yf --kind hourly --name sp500_1h
```
Gate: `sp500_1h_hourly` has T divisible into about 7 bars per day. Check that `T_hourly / T_daily` is within [6.5, 7.0], since half-days make it slightly under 7. Outside that range, the RTH filter or the resample anchor is wrong; re-run `pytest tests/test_fresh.py`.

- [ ] **Step 3: Run baselines and experiments**

```bash
python scripts/baselines.py --fresh sp500_daily sp500_1h_daily sp500_1h_hday sp500_1h_hourly
python scripts/run_grid.py E9_fresh_daily E10_hourly
python scripts/aggregate.py --costs
python scripts/plots.py
```

- [ ] **Step 4: Decision node D18 (hourly look-back)**
- The `hday` arm already matches the daily arm's look-back: 112 bars = 16 days, first kernel 28 bars = 4 days.
- If both `E10_hourly/hday_*` arms are worse than `daily_*` by more than 0.3 SR, the 28-bar first kernel may squash intraday detail. Run one descriptive variant with a 1-day first kernel (seq 112, kernel 7 → 16 daily steps):
  ```bash
  python scripts/run_grid.py E10_hourly --labels hday_HH hday_EE --set exp=E10_hourly_k7 kernel=7
  ```
- Report the `E10_hourly_k7` rows from `summary.csv` next to the main rows. `FAMILIES` does not include them, so compare them descriptively only.
- If the `hday` panel has fewer than about 250 training days, the 112-bar windows still fit, but say in the report that the hourly study is short (D12).

- [ ] **Step 5: Interpreting E10.** Put this wording in the report.
  - `hday` vs `daily`: same trading frequency (daily), same data source, the same 16-day look-back and the same 4-step spatial layer. The only difference is input granularity. This is **the** answer to "does hourly data help".
  - `hourly` arm: trades every hour, so its Sharpe is annualized with sqrt(1764). Always quote it **net of costs** from `results/tables/costs.md`. At 5 bps per side, hourly turnover typically erases the edge. Say so if the numbers show it.
  - HH vs EE inside each arm answers whether hyperbolic geometry helps at that frequency.

- [ ] **Step 6: Commit** the report updates (Task 17 fills them in).

---

### Task 17: Final report

**Files:**
- Create or modify: `docs/report.md`

- [ ] **Step 1: Write `docs/report.md` from this template.** Replace every `⟨…⟩` with the value from the file named inside it. Keep the headings.

```markdown
# THINK Reproduction & Ablations — Report

## Setup
- Data: RSR NYSE (1737 stocks) / NASDAQ (1026), daily 2013-01-02..2017-12-08; train 756 / val 252 / test ⟨237 NYSE⟩ days.
- Hypergraph: industry + wiki star hyperedges, deduplicated: ⟨#edges, max size, max node degree from test_real_nyse_hypergraph_stats⟩.
- Metric: SR = mean/std(daily top-5 return) * sqrt(252), no rf, no costs. Selection by validation SR.
- Seeds: 25 for answer tables; 15 for E4/E9/E10; 5 for curves (descriptive).
- Tuning: lr × alpha per geometry (HH, HE, EH, EE), 3 seeds each, selected on validation SR. Structure and grouping variants reuse their geometry's setting, which slightly favours the default "hyper" structure.
- Bootstrap CIs are on the Sharpe of the seed-ensemble portfolio (daily returns averaged over the common seeds), not the mean per-seed SR.
- Deviations from paper: ⟨list every D-node that fired, with what changed⟩.

## Run log
⟨one line per decision node outcome, in order⟩

## Q0 — Did we reproduce the paper?
| | paper protocol: norm=paper, 100 epochs, best test epoch (`test_oracle_sr_mean`) | norm=paper, val-selected (`test_sr_mean`) | **honest: norm=train, val-selected** (`test_sr_mean`) | paper |
|---|---|---|---|---|
| THINK NYSE SR (mean ± std, 25 seeds) | ⟨E1_main/THINK_paperProtocol⟩ | ⟨E1_main/THINK_paperNorm⟩ | ⟨E1_main/THINK⟩ | 1.18 ± 0.004 |
| TCONV+DHHAN (= EH) NYSE SR | – | – | ⟨E2_geometry/EH⟩ | 1.14 |
| THINK NYSE NDCG@5 (ours, correct definition) | – | – | ⟨E1_main/THINK test_ndcg5_mean⟩ | 0.86 (buggy evaluator; not comparable) |
| THINK NASDAQ SR | ⟨..._NASDAQ_paperNorm `test_oracle_sr_mean`, early-stopped⟩ | ⟨..._NASDAQ_paperNorm⟩ | ⟨..._NASDAQ⟩ | not reported |
Leakage audit: shuffled-label SR ⟨⟩ vs Random-5 ⟨⟩ ± 2·⟨sr_std⟩; paper-norm gap ⟨⟩; oracle-epoch gap ⟨⟩.
Reference points: Random-5 SR ⟨⟩, market (equal-weight) SR ⟨⟩, Oracle-5 SR ⟨⟩ (results/baselines/NYSE.json).
Verdict: ⟨reproduced under the paper's protocol / not reproduced (gap X)⟩. Then one sentence on how much of the paper's number survives the honest protocol.

## Q1 — How are stocks grouped, and does it matter?
Explain the construction (Part 0.4 of the plan). Table: comparisons.md family Q1_grouping.

## Q2 — Is hyperbolic space actually good?
2×2 table (temporal × spatial) + comparisons family Q2_hyperbolic with verdicts. Figure: results/figures/geometry.png.

## Distance-aware attention (paper Fig 2)
Family Q_attention. Figure: attention.png.

## Q3 — Do hyperedges matter?
Family Q3_hyperedges; Fig 3a/3b analogues (descriptive, 5 seeds).

## Q4 — Hyperedges × hyperbolic interaction
Family Q4_interaction. Interpretation: positive & STRONG ⇒ hyperbolic space helps more when relations are higher-order.

## Q5 — Does the number of stocks matter?
Paste results/tables/universe.md (cells + Spearman trends; verdict words "STRONG trend" / "NO EVIDENCE of a trend"). Figure universe.png (raw SR and SR minus Random-5 for the same universe). State the confound: smaller random universes also have far fewer hyperedges (num_edges, covered_frac columns).

## Hyperbolicity (paper Table I)
| market | δ_hg ours | δ_hg paper | δ_rel ours | δ_rel paper |
|---|---|---|---|---|
| NYSE | ⟨⟩ | 0.5 | ⟨⟩ | 0.087 |
| NASDAQ | ⟨⟩ | 1.0 | ⟨⟩ | 0.107 |

## Q6 — Daily data, out of period (2015–2026, S&P 500, survivorship-biased)
Family Q6_daily_fresh.

## Q7 — Hourly data
Families Q7_hourly + costs.md. State which arm answers what (Task 16 Step 5).

## Bottom line
One sentence per question, using exactly the verdict word the data gave:
- Q1–Q4, Q6, Q7 and attention: STRONG / SEED-ROBUST ONLY / NO EVIDENCE / INSUFFICIENT SEEDS (from comparisons.md).
- Q5: STRONG trend / NO EVIDENCE of a trend (from universe.md).
- Q0: reproduced under the paper's protocol / not reproduced (gap X), plus the honest number.
- E6/E7 curves: "descriptive only".
```

- [ ] **Step 2: Paste the tables** `results/tables/comparisons.md`, `table2_like.md`, `universe.md` and `costs.md` into the matching sections. Embed the figures with relative links such as `../results/figures/geometry.png`, noting that `results/` is local-only.

- [ ] **Step 3: Self-check.** Make sure every question Q0–Q7 has a verdict word, or an explicit "descriptive only" (curves) / "insufficient seeds" statement, and that no number in the report came from test-epoch selection except the clearly labelled `test_oracle_sr` diagnostic.

- [ ] **Step 4: Commit**

```bash
git add docs/report.md
git commit -m "docs: THINK reproduction and ablation report"
```

---

### Task 18 (optional; run only if the user wants the paper's NASDAQ "Clf" F1 column)

**Files:**
- Create: `src/hypershift/train/clf.py`, `scripts/run_clf.py`, `tests/test_clf.py`

**Interfaces:**
- Consumes: `prepare`, `gather_batch`, `window_offsets`, `set_seed` (Task 8); `THINK(out_dim=3)` (Task 6).
- Produces: `tertile_thresholds(data, seq) -> np.ndarray[2]`, `train_clf_run(cfg) -> dict` with keys `best_epoch, val_f1, test_f1`, and run folders `results/E11_clf/<label>/seed_k/metrics.json`.
- Labels: down / neutral / up, split at the training-return tertiles. STHGCN's exact thresholds are not published; say so in the report. Metric: macro-F1 (the paper reports THINK 0.49 and STHGCN 0.40).

- [ ] **Step 1: Write the failing test** `tests/test_clf.py`

```python
from hypershift.config import RunConfig
from hypershift.train.clf import tertile_thresholds, train_clf_run


def test_clf_runs(synthetic_market, synthetic_hypergraph, tmp_path):
    th = tertile_thresholds(synthetic_market, 8)
    assert th[0] < th[1]
    cfg = RunConfig(exp="E11_clf", label="t", seq=8, kernel=2, hidden=8, epochs=2, device="cpu", out_root=str(tmp_path))
    m = train_clf_run(cfg, synthetic_market, synthetic_hypergraph)
    assert 0.0 <= m["test_f1"] <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails.** Run: `pytest tests/test_clf.py -v`. Expected: FAIL with an import error.

- [ ] **Step 3: Implement** `src/hypershift/train/clf.py`

```python
"""Optional: 3-class movement classification (paper Table II 'Clf', NASDAQ), macro-F1."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from hypershift.config import RunConfig, dump_json
from hypershift.models.think import THINK
from hypershift.train.loop import gather_batch, prepare, set_seed, window_offsets


def tertile_thresholds(data, seq) -> np.ndarray:
    _, m, _, g = gather_batch(data, window_offsets(data, seq, "train"), seq)
    return np.quantile(g[m > 0.5], [1 / 3, 2 / 3])


def _f1(model, data, thg, cfg, split, th, device) -> float:
    model.eval()
    ys, ps = [], []
    offs = window_offsets(data, cfg.seq, split)
    with torch.no_grad():
        for i in range(0, len(offs), max(1, cfg.batch_days)):
            x, m, _, g = gather_batch(data, offs[i:i + max(1, cfg.batch_days)], cfg.seq)
            pred = model(torch.as_tensor(x, device=device), thg).argmax(-1).cpu().numpy()
            keep = m > 0.5
            ys.append(np.digitize(g, th)[keep])
            ps.append(pred[keep])
    return float(f1_score(np.concatenate(ys), np.concatenate(ps), average="macro"))


def train_clf_run(cfg: RunConfig, data=None, hg=None) -> dict:
    out = cfg.run_dir()
    if (out / "metrics.json").exists():
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    data, hg = prepare(cfg, data, hg)
    thg = hg.to_torch(device)
    model = THINK(in_dim=data.features.shape[2], hidden=cfg.hidden, seq=cfg.seq, kernel=cfg.kernel,
                  temporal=cfg.temporal, spatial=cfg.spatial, structure=cfg.structure,
                  attn_score=cfg.attn_score, attn_dist=cfg.attn_dist, out_dim=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    th = tertile_thresholds(data, cfg.seq)
    offs = window_offsets(data, cfg.seq, "train")
    rng = np.random.default_rng(cfg.seed)
    best, bad = None, 0
    for epoch in range(cfg.epochs):
        model.train()
        rng.shuffle(offs)
        for i in range(0, len(offs), cfg.batch_days):
            x, m, _, g = gather_batch(data, offs[i:i + cfg.batch_days], cfg.seq)
            logits = model(torch.as_tensor(x, device=device), thg)
            y = torch.as_tensor(np.digitize(g, th), device=device)
            mt = torch.as_tensor(m, device=device).reshape(-1)
            loss = (F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), reduction="none") * mt).sum() / mt.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        v, t = _f1(model, data, thg, cfg, "val", th, device), _f1(model, data, thg, cfg, "test", th, device)
        if best is None or v > best["val_f1"]:
            best, bad = {"best_epoch": epoch, "val_f1": v, "test_f1": t}, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    dump_json({**best, "config": cfg.to_dict()}, out / "metrics.json")
    return best
```

`scripts/run_clf.py`:
```python
"""python scripts/run_clf.py  -> 25 seeds x {HH, EE, EH} on NASDAQ, prints mean±std macro-F1."""
import numpy as np

from hypershift.config import RunConfig
from hypershift.experiments.grid import _geo
from hypershift.train.clf import train_clf_run

for g in ("HH", "EH", "EE"):
    f1s = [train_clf_run(RunConfig(exp="E11_clf", label=g, seed=s, **_geo(g, market="NASDAQ", alpha=0.1)))["test_f1"]
           for s in range(25)]
    print(g, f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
```

- [ ] **Step 4: Run the test.** Run: `pytest tests/test_clf.py -v`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add src/hypershift/train/clf.py scripts/run_clf.py tests/test_clf.py
git commit -m "feat(optional): NASDAQ movement classification (macro-F1)"
```

**Explicitly out of scope (tell the user, don't build):** the paper's graph-RNN baselines (GConvGRU, EGCN-O/H, DCRNN, TGCN, ST-TGCN, DyGrAE, RSR-I). The EE variant (Euclidean temporal conv + Euclidean hypergraph attention, which is STHGCN-like) and the five non-learned baselines serve as references. If the user wants the full baseline table, write a separate plan using `torch-geometric-temporal` on the clique-expanded graph.
