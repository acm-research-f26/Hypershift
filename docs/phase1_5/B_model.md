# Phase 1.5 B: tensor-level model audit vs THINK (ICDM 2022)

Source: `docs/paper/icdm22-think.pdf` pp. 850-851 (eq. 6-17, Fig. 1). Cites are page + equation. Code line numbers are the working tree at commit time. Shapes were measured on the real NYSE graph with the default config (`RunConfig()`: `hidden=32, seq=16, kernel=4, C=5, input_mode=level, batch_days=1`) by forward hooks on CPU, B = 1 day:

- N = 1737 stocks, T = 1245 days.
- **E = 4350 hyperedges** and **M = 10204 incidences** after `prepare` (largest edge: 500 nodes, smallest: 2). Note: `docs/phase1/paper_audit.md` P42 quotes "312 hyperedges"; the measured graph has 4350.
- Total parameters: **1890**.

Verdicts: EXACT = matches the printed equation. DEPARTURE = differs from print (reason given). INFERRED = paper silent, we chose.

## 1. Temporal convolution (eq. 9-12)

| Item | Paper | Our code | Shapes (NYSE, B=1) | Verdict |
|---|---|---|---|---|
| Input map | `X^l = exp_o(X_E)` (p850 Sec. III-A; eq. 17) | `think.py:38` `expmap0(h)` after permute to `[B,T,N,C]` (`think.py:35`) | `[1,1737,16,5]` -> `[1,16,1737,5]`. exp_0 acts on the 5-d feature vector per (day, stock). Level inputs reach ball radius ~0.95 (CLAUDE.md pitfall). | EXACT (map). Input features/scaling INFERRED (not in paper) |
| Kernel K | "kernel size K", "lookback tau = nK" (p850 III-A). K not given. | `kernel=4` (`config.py`) | K = 4 | INFERRED (value) |
| Stride / padding | Not stated. Input `N x nK x C` -> n steps, so receptive fields tile the series. | `_windows` (`layers.py:38-43`): non-overlapping, stride K, no padding; reshape `[B,T/K,K,N,C]` -> `[B,T/K,N,K*C]`, time-major inside a window | | INFERRED (stride = K, pad = 0; implied by tau = nK) |
| Length per stage | tau = nK | conv1 16 -> 4; conv2 4 -> 1 (kernel `k2 = seq/kernel = 4`, `think.py:20,26`) | `[1,16,N,5]` -> `[1,4,N,32]` -> `[1,1,N,1]` | INFERRED (second kernel size; paper only says "another hyperbolic temporal convolution") |
| Eq. 12 pipeline | `F^c(beta-cat({x_is}_{s=1..K}))` | `HypTemporalConv.forward` (`layers.py:53-54`): `fc(expmap0(_windows(logmap0(x) * scale)))` | conv1: `[1,4,N,20]` -> `[1,4,N,32]`; conv2: `[1,4,N,32]` -> window `[1,1,N,128]` -> `[1,1,N,1]` | EXACT structure |
| beta-concat join dim | eq. 11: concatenate K points of dim `n_i` into `n = sum n_i`, tangent-space concat at o | The K points of a window are joined along the **feature** dim (last), `K*C = 20` (conv1) and `K*H = 128` (conv2). Time is folded into the feature axis; N untouched. | | EXACT |
| beta scaling | eq. 11: `beta_n beta_{n_i}^{-1} log_o(x_i)`, `beta_m = B(m/2, 1/2)` | `beta_concat` (`layers.py:16-20`) is exactly this. The conv uses a fused equivalent: `scale = B(K*in/2,1/2)/B(in/2,1/2)` (`layers.py:50,54`), multiply `log0(x)` once, reshape, `exp0`. Same value as `beta_concat` since all `n_i` are equal. conv1 scale = B(10,.5)/B(2.5,.5) = 0.482; conv2 scale = B(64,.5)/B(16,.5) = 0.497 (computed with `beta_fn`) | | EXACT |
| FC eq. 9 | `F^c(x) = w / (1 + sqrt(1 + ||w||^2))`, `w = (sinh(v_k(x)))_k` | `layers.py:34-35`, `project()` after. `v` is clamped to +-15 before `sinh` and the output is projected to radius 1 - 1e-5 | | EXACT up to numerical clamps |
| FC eq. 10 | `v_k = 2||z_k|| asinh(lambda_x <x,z_k> cosh(2 r_k) - (lambda_x - 1) sinh(2 r_k))` (p850) | `layers.py:31-33`: uses `<x, z_k/||z_k||>` (HNN++ [25] form). Bias `r` init 0 | | DEPARTURE (low impact; reparametrises `z_k`; known paper_audit item 8) |
| Non-linearity | None between layers in Fig. 1 / eq. 12 | None in hyp path. (Euclidean ablation conv has ReLU after conv1, `layers.py:66`; not a paper detail) | | EXACT (hyp) |
| Params (hyp) | n/a | conv1 `PoincareLinear(20,32)`: 20*32 + 32 = **672**. conv2 `PoincareLinear(128,1)`: 128 + 1 = **129**. beta scales are constants (0 params). | | n/a |

Whole-model count 1890 = 672 + 129 + DHHAN (`a` 32, unused `gamma` 1, `fc` 32*32 + 32 = 1056).

## 2. DHHAN (eq. 13-16)

| Item | Paper | Our code | Shapes | Verdict |
|---|---|---|---|---|
| Eq. 13 midpoint | `z_i = 1/2 (x) ( sum_k lambda_{u_k} u_k / sum_j (lambda_{u_j} - 1) )` | `gyromidpoint` (`poincare.py:56-62`): `index_add` of `lam*u` and `lam-1` per hyperedge, then `mobius_scalar(0.5, num/den)` (`poincare.py:51-53`, = eq. 8 with scalar `W`) | u `[1,4,1737,32]` -> z `[1,4,4350,32]`, gathers `[1,4,10204,32]` | EXACT. Cosmetic: `den` clamped at 1e-15; the inner Euclidean quotient can leave the ball, which `artanh` clamps handle |
| Eq. 14 `a^T (x) (u_j (+) z_i)` | Mobius mat-vec (eq. 8: `W (x) x = exp_o(W log_o x)`) with `W = a^T` | `attention.py:74` `tanh(log0(mobius_add(uj, zi)) @ a)`. 1-D ball point of `exp_0(a . log_0(.))` is `tanh(a . log_0)`. Order `u_j (+) z_i` as printed. | `a [32]`; scores `[1,4,10204]` | EXACT |
| Eq. 14 `d_B(u_j, z_i)` | eq. 4 | `poincare_dist` (`poincare.py:43-44`) | `[1,4,10204]` | EXACT |
| Eq. 14 `(.)` | Eq. 7: `x (.) y = tan((||xy||/y) arctan(||y||)) ||xy||/||y||` (p850). Ill-formed as printed (tan not tanh, `y` unnormed, `xy` undefined; paper_audit U2). Eq. 8 immediately defines `(x)`, so `(.)` was probably intended as a Mobius-style scalar product. | **Default**: plain product `base * d` (`attention.py:53-54`, `dist="mult"`) | | **DEPARTURE** (printed operator not implementable; plain product is our choice). New switch `attn_odot=mobius` implements a reading, see Sec. 5 |
| Softmax / normalisation | **None printed.** Text only says it "learns the attention coefficient `alpha_ij`" (p851). | `segment_softmax(scores, node_idx, N)` over the hyperedges containing node j (`attention.py:13-19`, called via `_alpha` at `attention.py:79`) | `[1,4,10204]` -> same, rows sum to 1 per node | **DEPARTURE / INFERRED**: standard GAT-style normalisation we added. Switch `attn_norm=none/sum` |
| Other scoring extras | none printed | No LeakyReLU in default `eq14`+`mult` path (LeakyReLU only in `dist=neg/off` variants). `gamma` parameter is unused with `mult`. | | EXACT for default |
| Eq. 15 | `u'_j = exp_o( ReLU( sum_{i: v_j in e_i} alpha_ij log_o(F^c(z_i)) ) )` | `attention.py:85-88`: `fc(z)` is `PoincareLinear(32,32)` (+ `log0`), weighted by `alpha`, `index_add` to nodes, `ReLU`, `expmap0`. FC applied to the hyperedge feature `z_i` as printed. | `fc(z)` `[1,4,4350,32]`; agg and out `[1,4,1737,32]` | EXACT |
| Isolated nodes | Not addressed (sum over empty set gives `exp_o(ReLU(0)) = o`) | `torch.where(hg.has_edge[:,None], out, u)` (`attention.py:89`): nodes in no hyperedge keep their input feature | | DEPARTURE (small: NYSE has a few uncovered stocks; passes `u` through, acts like an identity for them) |
| Residual / skip | none | none | | EXACT |
| Eq. 16 parameter sharing | `HA^tau` "applies the same HA(.) layer to each snapshot" (p851) | Yes: one `HypHypergraphAttention` module; the time axis (4 positions after conv1) is a leading batch dim of `[B,4,N,32]`, so `a`, `fc.z`, `fc.r` are shared. All the index ops use `dim=-2`. | the same 1056 + 33 params serve all 4 positions | EXACT |
| Eq. 16 dynamic graph `G_t` | `{HA(G_t, Q_t)}` time-evolving hypergraph | One **static** graph for all positions (`hg` argument) | | INFERRED (static stock hypergraph; paper_audit P23) |
| Placement | DHHAN sits between the two temporal convolutions (eq. 17, Fig. 1) so it acts on tau = 4 steps | `think.py:39-48` | `[1,4,N,32]` in and out | EXACT |

## 3. Eq. 17 and the output head

Paper: `y^t = log_o( tau-conv( HA^tau( tau-conv( exp_o(X_E) ), G ) ) )`.

Code (`think.py:34-52`): `permute` -> `expmap0` -> `tconv1` -> DHHAN -> `tconv2` -> `logmap0` -> `h[:, 0]` -> `squeeze(-1)`.

- `tconv2` is `PoincareLinear(128, 1)`: the 4 time steps x 32 channels are beta-concatenated and mapped to a **1-D Poincare ball**, i.e. a point in (-1, 1). `logmap0` of that point is `artanh(y)`, unbounded, so the prediction is a real scalar per stock.
- Shapes: `[1,1,1737,1]` -> `[1,1,1737,1]` (log) -> `[1,1737]`.
- **No** pooling, residual, extra MLP head, bias outside `r`, or per-stock embedding. The score is the head output of `tconv2` directly. Verdict: EXACT to eq. 17.
- What the scalar is trained to predict (next-day return, vs price) and the loss (masked MSE + ranking hinge) are not in the paper (INFERRED, `train/loss.py`).
- Known issue worth noting: the tconv2 output norm in a level-input NYSE forward pass is tiny (max ~0.12 in our probe), so the artanh stays in its linear range; scores are compressed, which affects the ranking-hinge scale but not eq. 17 fidelity.

## 4. Departures, ranked by likely impact on results

| Rank | Departure | Where | Why it may matter | Status |
|---|---|---|---|---|
| 1 | `(.)` implemented as plain product `tanh(.) * d_B`; paper's eq. 7 is not implementable as printed | `attention.py:53-54` | Controls whether distance up-weights or only rescales scores; with softmax, multiplying by `d >= 0` sharpens far hyperedges. The paper's distance-aware claim (HHN vs THINK ablation, p852) hinges on it. | `attn_odot=mobius` added |
| 2 | Added per-node softmax over hyperedges (paper prints none) | `attention.py:79` | With softmax, `alpha` weights sum to 1, so messages are a convex combination; raw `alpha` (paper) has arbitrary scale and sign (can be negative) and the node update scales with degree. Changes the effective signal for high-degree hubs (Fig. 3b theme). | `attn_norm=none/sum` added |
| 3 | Static hypergraph for every time slice; wiki edges as stars, graph has E = 4350 | `think.py:39`, `data/hypergraph.py` | Eq. 16 allows `G_t`; paper_audit item 1 says second-order wiki relations should be pairs. | not touched here |
| 4 | Isolated nodes pass through unchanged | `attention.py:89` | Paper's literal sum gives `o` (zero vector) for them; ours keeps `u`. Tiny node set. | unchanged |
| 5 | Eq. 10 uses unit-norm `z_k` inside the inner product (HNN++ form) | `layers.py:32-33` | Reparametrisation of `z_k`; same function class. | unchanged |
| 6 | K = 4, stride K, no padding, second kernel 4, hidden 32, lookback 16 | `config.py`, `layers.py:38-43` | Paper gives none; chosen so tau = nK. | INFERRED |
| 7 | Numerical clamps: `v` +-15, `project` radius 1-1e-5, `artanh` clip 1e-7 | `layers.py:34`, `poincare.py` | Only active near the boundary; relevant for level inputs (radius ~0.95). | unchanged |
| 8 | Unused `gamma` parameter in default config | `attention.py` `_Base` | Dead parameter (counted in the 1890); harmless, weight decay on a zero-gradient param. | unchanged |

## 5. New switches (defaults unchanged: `attn_odot=product`, `attn_norm=softmax`)

Added to `RunConfig` (`config.py`), `THINK.__init__` (`think.py`), both attention classes, and wired in `train/loop.py` and `train/clf.py`. Defaults reproduce existing behaviour bit-for-bit (test `test_odot_norm_defaults_unchanged`).

**`attn_odot=mobius`** (only active when `attn_dist=mult`). Eq. 7 read literally with tan -> tanh and arctan -> artanh:

`x (.) y = tanh( (||xy|| / ||y||) artanh(||y||) ) * ||xy|| / ||y||`

Here x = s = `tanh(a . log_0(u (+) z))` (the eq. 8 scalar) and y = d_B >= 0 are both scalars. Readings we had to choose (INFERRED, not in paper):

- `||xy||` is read as `|x * y|`, so `||xy||/||y|| = |s|`.
- artanh requires `||y|| < 1`, so y is taken as `exp_0(d) = tanh(d)`; then `artanh(||y||) = d`.
- the sign of s is restored.

Result, implemented in `attention.odot_mobius`: **`score = sign(s) |s| tanh(|s| d)`**. Versus the product `s*d` it saturates for large distances and shrinks small ones quadratically.

**`attn_norm`**:

- `softmax` (default): `segment_softmax` per node over its hyperedges.
- `none`: the raw eq. 14 value is used as `alpha` (this is the closest to the print; the paper has no normaliser). Scores can be negative and unbounded.
- `sum`: `alpha = s / sum_{i in node's edges} |s|` (absolute values so a near-zero signed sum cannot blow up). Flagged: the literal `s / sum(s)` is unstable with signed scores.

Both switches apply to the Euclidean attention class too (`odot` uses `d = ||u - z||`). Tests in `tests/test_attention.py` cover the formula (against literal eq. 7 with `y = tanh d`), default preservation, norm behaviour (raw passthrough, softmax rows sum to 1, sum-norm L1 = 1), forward/backward finite for all 5 non-default combinations x {hyp, euc}, and invalid-value rejection.

### CPU smoke test (1 epoch, 150-stock NYSE universe, `device=cpu`, seed 0)

Runs wrote to the scratchpad, not `results/`.

| odot | norm | epochs_run | val SR | test SR | test NDCG@5 |
|---|---|---|---|---|---|
| product (default) | softmax (default) | 1 | 1.473 | 1.528 | 0.574 |
| mobius | softmax | 1 | 1.477 | 1.395 | 0.573 |
| product | none | 1 | 0.736 | 0.554 | 0.567 |
| product | sum | 1 | 0.642 | -0.804 | 0.566 |
| mobius | none | 1 | 0.592 | 0.821 | 0.568 |

All five trained one epoch with finite metrics. These are single-seed, 150-stock, 1-epoch numbers: they show the switches run, not that one is better. Full suite: 256 passed.
