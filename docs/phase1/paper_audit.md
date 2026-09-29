# Paper audit: THINK (ICDM 2022) vs our docs and code

> **U1 resolved (2026-09-29):** the user approved the author-hosted full paper (pp. 849-854, `docs/paper/icdm22-think.pdf`) as the source of truth for every `[A854]` cite.

*Audited 2026-09-29 against `05-Hypershift-OA.pdf` (repo root). Read-only audit: no model code or other doc was changed.*

**Citation key.** `p849`-`p853` = pages of `05-Hypershift-OA.pdf`. `[A854]` = page 854 (references 17-38, Algorithm 1, appendices A and B) of the author-hosted copy `https://tylersnetwork.github.io/papers/icdm22-think.pdf`. I diffed the text of pages 849-853 of both files and they are identical. `[A854]` is not in the repo PDF. Classes: CONFIRMED (cite given), CONTRADICTED (PDF quoted or described), NOT IN PAPER (fine to keep, must be labelled inferred). Eq/Fig/Table numbers are the paper's. Where a row says "code", the file:line is in the working tree.

## Task assumption that failed

The task said the PDF starts at p849 "and includes the appendix". **`05-Hypershift-OA.pdf` has 5 pages (p849-p853) and ends mid-reference-list at [16].** There is no appendix, no Algorithm 1, and no references [17]-[38] in it. Everything marked `[A854]` (hypergraph construction, Sorensen-Dice merge, delta_rel definition, dataset references) was verified only against the author-hosted 6-page copy. See "UNREADABLE - ask user".

## 1. Summary

Counts are per atomic row in the section 2 tables (computed from the tables). Rows labelled "CONFIRMED / NOT IN PAPER" are split into one count each.

**Total 180 rows: CONFIRMED 109, CONTRADICTED 21, NOT IN PAPER 50.**

| Source | CONFIRMED | CONTRADICTED | NOT IN PAPER |
|---|---|---|---|
| 2.1 Plan `docs/superpowers/plans/2026-09-27-think-reproduction.md`, Pa | 35 | 8 | 14 |
| 2.2 `docs/HANDOFF.md` sections 2, 3, 4 (plus paper claims in 5-6) | 17 | 5 | 11 |
| 2.3 `docs/POC_PRESENTATION.md` sections 2, 3, 7 | 13 | 3 | 9 |
| 2.4 `docs/PHASE1_TRACKER.md` | 18 | 1 | 4 |
| 2.5 `docs/phase1/R_feasibility.md` | 18 | 3 | 7 |
| 2.6 `docs/phase1/R2_chickenpox.md` | 3 | 1 | 3 |
| 2.7 `docs/phase1/R8_baselines.md` | 5 | 0 | 2 |

### CONTRADICTED items, ranked by likely impact on results

1. **Wiki hyperedges: the paper says second-order relations are pairwise, our builder makes a star for every wiki channel.** [A854] Sec. B: first-order relation gives "a hyperedge of a source stock and a set of target stocks related to it via the same Wikidata relation"; "The second-order relation is pairwise in nature". `src/hypershift/data/hypergraph.py:73-81` (`wiki_hyperedges`) stars every channel `{i} U {j : rel[i,j,k]=1}`. Plan Part 0.4 says the channel order is unrecoverable (only 29 of 758,189 paths first-order), so nearly all wiki channels are second-order, i.e. should be pairs, not stars. Touches every stock run (312 hyperedges, max node degree 37, clique/decomposition arms, G1/G2/G12). The plan text "second-order relations with a single target become pairs, as the paper describes" is not what [A854] says.
2. **The paper's "Euclidean" ablation is Euclidean temporal conv + hyperbolic hypergraph attention (our EH), not a fully Euclidean model (our EE).** p852 Sec. V.A: "we replace the hyperbolic temporal convolution with a Euclidean temporal convolution [3]"; Table II row "TCONV + DHHAN"; p853 Fig. 3 caption: "Euclidean THINK (Euclidean temporal convolution + hypergraph attention)". The paper reports no EE and no HE arm. Our small-scale "hyperbolic vs Euclidean" contrasts (POC_PRESENTATION, HANDOFF 5b/5d, tracker "tuned hyperbolic vs Euclidean +1.10") are HH vs EE, which the paper never ran. The tracker has no EH arm at small scale (A5 is full NYSE only) and G2/G12 are HH only, so Fig. 3's Euclidean curve is not matched.
3. **Eq. 14 is stated as fact in several docs and all are stale or over-claimed.** p851 eq. 14: `alpha_ij = a^T (x) (u_j (+) z_i) (.) d_B(u_j, z_i)`. HANDOFF sec. 4 says "Symbol lost in the PDF ... literal reading"; Plan Part 0.2 says the glyph "is most likely (+)" and D8 calls `mobius_mult` "the literal reading". Actually `(x)` is the Mobius matrix-vector product of eq. 8 (`W (x) x = exp_o(W log_o x)`), and `(.)` is a separate operator from eq. 7 whose printed formula is not well-formed (see UNREADABLE U2). Code `attention.py:47` (`tanh(a . log0(u (+) z))`) matches eq. 8 for a scalar output; `attention.py:31-33` `_combine` "mult" treats `(.)` as a plain product (inferred); `attention.py:58` adds a segment softmax that eq. 14 does not contain (inferred). POC_PRESENTATION 4d says the exact formula "has since been confirmed from the PDF": the `(x)` part is, the `(.)` part and the softmax are not.
4. **Sharpe ratio definition.** p852 Sec. IV-B: `SR = E[R_a - R_f] / std[R_a - R_f]` with R_f an unspecified risk-free return, "we buy the top-k stocks" (k not given), no annualization. Plan Global Constraints ("No risk-free rate ... (paper-compatible)"), POC_PRESENTATION Sec. 3 (top-5, x sqrt(252) in the "THINK paper" column, "Same?" = Same) state a definition the paper does not give. `metrics.py:34-36` and `config.py:44` implement k=5, sqrt(252), no R_f. Impact: the scale gap to the paper's 1.18 is unknowable (R_f = 0 is our choice).
5. **Evaluation citation.** p852: "Following [1], we adopt a daily-buy-hold trading strategy" and "Following [1], we formulate stock prediction as a ranking problem" ([1] = RSR, Feng et al.). [38] (STHAN-SR) is cited only for hypergraph construction [A854]. HANDOFF Sec. 6 says of the NDCG evaluator inference "the paper says it follows [38]". This weakens the inference that THINK used the STHAN-SR evaluator (the NDCG-bug argument).
6. **Sorensen-Dice merge "garbled".** The SCD sentence in [A854] Sec. B is fully legible: "we merged them until no two pairs had an SCD score lower than a threshold". The threshold is unspecified and the stopping rule reads inverted, but the text is not garbled. R_feasibility (R2 point 2) and R2_chickenpox (inferred choice 3) call it garbled. `src/hypershift/data/pygt.py:44-49` skips the merge entirely: report that as a deviation from a stated step.
7. **R_feasibility R4 says RSR-style industry/Wiki hyperedges "do not exist" for Chinese stocks.** [A854] Sec. B: "Stock Datasets: NYSE, NASDAQ, TSE and CSE ... industry hyperedges and Wiki corporate hyperedges". The paper claims to have built both for CSE and TSE (public availability of the data is a separate question).
8. **Eq. 10 as printed uses the unnormalized `<x, z_k>`.** p850 eq. 10: `v_k(x) = 2||z_k|| sinh^-1(lambda_x <x, z_k> cosh(2 r_k) - (lambda_x - 1) sinh(2 r_k))`. Plan Part 0.2 writes `<x, z_k/||z_k||>` and `layers.py:32-33` divides by `||z_k||` (the HNN++ [25] form). Low impact (reparametrizes `z_k`), but "checked equation by equation" (HANDOFF sec. 3) is not literally true.
9. **Plan Part 0.2 attributes Mobius scalar multiplication to "eq 7's Mobius matrix-vector mult".** The matvec is eq. 8; eq. 7 is `(.)`. No code effect.
10. **Hyperparameters "from the paper".** Plan Global Constraints: "Default hyperparameters (from STHAN-SR/RSR commands and the paper)". The paper (pp849-854) states none (no K, hidden size, lookback, lr, epochs, batch, optimizer, split).
11. **Minor number errors.** R_feasibility R1 says the DTT baselines are "2.03-2.06 for eight of nine": seven are 2.03-2.06, STHGCN is 1.03, RSR-I is "-" (p852 Table II). Tracker R1 "baselines (~2.05)" omits STHGCN 1.03.
12. **Handoff/POC "verified equation by equation" for eq. 9-17** (HANDOFF sec. 3): see items 3 and 8.

Not a contradiction but worth recording: the Fig. 3b x-axis starts at node degree 31, while our NYSE graph has max node degree 37 (Plan 0.4, D3). If 31 is the paper graph's max degree, the graphs differ; the PDF does not say.

## 2. Claim tables

### 2.1 Plan `docs/superpowers/plans/2026-09-27-think-reproduction.md`, Part 0 and Part 1 (D0-D19)

| # | Claim | Class | PDF cite | Note |
|---|---|---|---|---|
| P1 | Curvature c=1, unit ball, conformal factor lambda=2/(1-||x||^2) | CONFIRMED | p850 Sec. II | "manifold B={x: ||x||<1}" |
| P2 | Mobius addition formula | CONFIRMED | p850 eq. 3 | matches `poincare.py:34-40` |
| P3 | d(x,y) = 2 artanh(||-x (+) y||) | CONFIRMED | p850 eq. 4 | printed 2tanh^-1 |
| P4 | exp_0 and log_0 are eq. 5-6 at x=0 | CONFIRMED | p850 eq. 5-6; eq. 11/15/17 use exp_o, log_o | |
| P5 | Mobius scalar mult is "the M=rI case of eq 7's matrix-vector mult" | CONTRADICTED | p850 eq. 8: `W (x) x = exp_o(W log_o(x))` | matvec is eq. 8; eq. 7 is `(.)`. Code `poincare.py:51-53` fine, label wrong |
| P6 | Poincare FC output `w/(1+sqrt(1+||w||^2))`, `w=sinh(v)` | CONFIRMED | p850 eq. 9 | |
| P7 | `v_k` uses `<x, z_k/||z_k||>` | CONTRADICTED | p850 eq. 10 prints `<x, z_k>` | HNN++ [25] normalizes, paper as printed does not; `layers.py:32-33` |
| P8 | `v_k = 2||z_k|| asinh(lambda<.>cosh(2r) - (lambda-1) sinh(2r))` structure | CONFIRMED | p850 eq. 10 | apart from P7 |
| P9 | beta-concat, `beta_m = B(m/2, 1/2)`, scale `beta_n/beta_{n_i}` | CONFIRMED | p850 eq. 11 | `layers.py:16-21` |
| P10 | Temporal conv: length tau=nK, beta-concat K points, then Poincare FC, output length n | CONFIRMED | p850 eq. 12 and text | |
| P11 | Windows are non-overlapping (stride K) | CONFIRMED | p850 Sec. III-A: input `N x nK x C`, output n steps | implied by dimensions; "stride" is never said |
| P12 | Eq. 13 gyromidpoint `1/2 (x) (sum lambda u / sum(lambda-1))` | CONFIRMED | p851 eq. 13 | paper calls it the Einstein midpoint [27]; `poincare.py:56-62` |
| P13 | Eq. 14 is `softmax_i of a^T (u_j (+) z_i) . d(u_j,z_i)` | CONTRADICTED | p851 eq. 14: `alpha_ij = a^T (x) (u_j (+) z_i) (.) d_B(u_j, z_i)` | `(x)` per eq. 8, `(.)` per eq. 7 |
| P14 | "The PDF glyph between u_j and z_i is lost; most likely (+)" | CONTRADICTED | p851 eq. 14 | (+) is inside the bracket; the lost glyphs were `(x)` and `(.)` |
| P15 | Softmax normalization of alpha over `{i : v_j in e_i}` | NOT IN PAPER | p851 Sec. III-B | paper says only "learns the attention coefficient"; `attention.py:58`; label inferred |
| P16 | `(.)` is plain multiplication by d | NOT IN PAPER | p850 eq. 7 unusable as printed | `attention.py:31-33`; label inferred |
| P17 | Eq. 15 `u'_j = exp_0(ReLU(sum alpha log_0(FC(z_i))))` | CONFIRMED | p851 eq. 15 | sum index printed "i such that s_j in e_i" (s_j vs v_j slip) |
| P18 | Eq. 16 same layer per time slice | CONFIRMED | p851 eq. 16 | paper applies `HA(G_t, Q_t)`, a per-snapshot hypergraph |
| P19 | Eq. 17 `y = log_0(TConv(DHHAN(TConv(exp_0(X)))))` | CONFIRMED | p851 eq. 17 | |
| P20 | THINK ranks stocks, buys the top-k at close, sells next close | CONFIRMED | p852 Sec. IV-B | |
| P21 | k = 5 | NOT IN PAPER | p852 says "top-k" | `config.py:44` |
| P22 | `X` is `[N, tau=16, C=5]` | NOT IN PAPER | | lookback, C unspecified |
| P23 | `G` is static | NOT IN PAPER | p851 defines time-evolving `G = {G_i}` | staticness for stocks is our choice |
| P24 | Table II numbers (RSR-I 1.05/0.75/0.99/0.72/0.38; STHGCN 1.10/0.78/1.07/0.74/0.40; TCONV+DHHAN 1.14/0.81/1.11/0.76/0.44; THINK 1.18/0.86/1.19/0.81/0.49) | CONFIRMED | p852 Table II | THINK NYSE SR printed 1.18 +- 4e-3, NDCG 0.86 +- 9e-4 |
| P25 | delta_hg/delta_rel NYSE 0.5/0.087, NASDAQ 1.0/0.107 | CONFIRMED | p850 Table I, p852 Table II | |
| P26 | Paper has no NASDAQ Sharpe, only Clf F1 | CONFIRMED | p852 Table II caption | |
| P27 | Fig. 2 (THINK vs HHN) uses only DTT and CPox | CONFIRMED | p852 Fig. 2 | text says "on all datasets"; the figure shows two |
| P28 | HHN = THINK without distance-aware attention | CONFIRMED | p852 Sec. V.B | |
| P29 | Fig. 3a axis 500,15,9,5,3; SR ~1.2 to ~0.9, Euclid below | CONFIRMED | p853 Fig. 3a | read-off: THINK 1.18 to ~0.955, Euclid ~1.11 to ~0.91 |
| P30 | Fig. 3b axis 31,28,22,16,2; SR ~1.1 to ~0.8 | CONFIRMED | p853 Fig. 3b | approximate: THINK leaves the frame (>1.15) at degree 31, ends ~0.88; Euclid ~1.11 to ~0.85 |
| P31 | Table II is a "25-run mean" | CONFIRMED | p852 caption; Fig. 2/3 "25 runs" | |
| P32 | The +- values are a per-seed spread ("far smaller than realistic") | NOT IN PAPER | p852 | paper never says whether +- is std, s.e. or CI |
| P33 | "Paper protocol follows STHAN-SR: test evaluated each epoch, no checkpointing" | NOT IN PAPER | | statement about code; paper says only "Following [1]" |
| P34 | Paper uses full-series-max feature normalization | NOT IN PAPER | | inferred from authors' earlier code |
| P35 | Grouping follows STHAN-SR [38] and appendix B | CONFIRMED | [A854] Sec. B | |
| P36 | Industry hyperedges: one per industry, all stocks in it | CONFIRMED | [A854] Sec. B | |
| P37 | First-order Wiki: source stock plus targets of the same relation | CONFIRMED | [A854] Sec. B | |
| P38 | Second-order Wiki relations are pairwise | CONFIRMED | [A854] Sec. B ("pairwise in nature", `X -R2-> Z <-R3- Y`) | as a statement of the paper |
| P39 | Our builder stars every wiki channel incl. second-order | CONTRADICTED | [A854] Sec. B | `hypergraph.py:73-81`; the plan acknowledges channels cannot be separated (0.4) |
| P40 | Largest industry (500) matches the 500 on the Fig. 3a axis | CONFIRMED | p853 Fig. 3a axis | consistency check only, not proof the construction "is confirmed" |
| P41 | Dedup identical hyperedges, drop size < 2 | NOT IN PAPER | | inferred |
| P42 | Expected 312 hyperedges, max degree 37, 17 uncovered stocks | NOT IN PAPER | Fig. 3b axis starts at 31 | see note after the contradicted list |
| P43 | Splits `valid_index=756`, `test_index=1008`, T=1245 | NOT IN PAPER | Table I gives T=1245 and 1737 nodes only | split from the RSR code |
| P44 | Default hyperparameters come "from ... the paper" | CONTRADICTED | pp849-854 state none | seq=16, K=4, hidden 32, lr, wd, epochs, patience, alpha, batch |
| P45 | "SR = mean/std x sqrt(252), no risk-free rate ... (paper-compatible)" | CONTRADICTED | p852 SR formula with R_f | R_f unspecified, no annualization |
| P46 | TSE data is Li et al. IJCAI'20 | CONFIRMED | p850 Table I "TSE [21]"; [A854] ref [21] | not-public claim is external |
| P47 | TCONV+DHHAN = Euclidean temporal conv + hyperbolic hypergraph attention | CONFIRMED | p852 Sec. V.A and Table II | row name and text linked by inference, not by an explicit definition |
| P48 | 2x2 temporal x spatial grid (incl. HE) | NOT IN PAPER | | paper has EH only (plus an unspecified "Euclidean THINK" in Fig. 3) |
| P49 | D8: `mobius_mult` is "the literal reading of the paper" | CONTRADICTED | p851 eq. 14 | literal = `(x)` then `(.)`; `eq14` is closer |
| P50 | Decomposition ablation E6, both schedules | CONFIRMED | p853 Sec. V.C vs Fig. 3a | text "in increasing order of hyperedge degree", axis runs 500 to 3; both schedules needed; `hypergraph.py:109-126` docstring says so |
| P51 | Hub removal: sort by degree, remove the hubs' hyperedges | CONFIRMED | p853 Sec. V.C | `hypergraph.py:134-139` |
| P52 | Removing all hyperedges "degenerates THINK to a temporal model" (no-relations arm) | CONFIRMED | p853 Sec. V.C | |
| P53 | D16: delta_hg is a multiple of 0.5 | CONFIRMED | p850 Table I (0.5, 1.0, 1.5) | |
| P54 | D16: delta_rel = 2 delta / diam via Gromov products | CONFIRMED | [A854] App. A, eq. 18-19 | `hyperbolicity.py:45` |
| P55 | Task 18 labels down/neutral/up | CONFIRMED | p851 Sec. IV-A | "going up, going down, or staying neutral" |
| P56 | Task 18 split at training-return tertiles; macro-F1 | NOT IN PAPER | p852 says "F1-score" | thresholds and averaging unspecified (plan itself says thresholds unpublished) |
| P57 | "The EE variant ... is STHGCN-like" | NOT IN PAPER | p852 says STHGCN "use the Euclidean space" | similarity is our inference |

### 2.2 `docs/HANDOFF.md` sections 2, 3, 4 (plus paper claims in 5-6)

| # | Claim | Class | PDF cite | Note |
|---|---|---|---|---|
| H1 | Model `y = log_0(TConv2(DHHAN(TConv1(exp_0(X))), G))` | CONFIRMED | p851 eq. 17 | |
| H2 | Buys the top 5 at close, sells next close | NOT IN PAPER | p852 "top-k" | k=5 from authors' earlier code |
| H3 | Inputs: 16 days x 5 features (MA5/10/20/30, close), each divided by max close | NOT IN PAPER | | features, lookback, scaling unspecified |
| H4 | exp_0 maps inputs onto the ball | CONFIRMED | p851 Fig. 1, eq. 17 | |
| H5 | Temporal conv is non-overlapping | CONFIRMED | p850 (nK to n) | |
| H6 | 4-day kernel, 16 to 4 to 1 | NOT IN PAPER | | K, tau, second conv kernel unspecified |
| H7 | beta-concat then HNN++ Poincare FC | CONFIRMED | p850 eq. 9-11, cites [25] | |
| H8 | DHHAN applied at each time step | CONFIRMED | p851 eq. 16 | |
| H9 | Node to hyperedge = gyromidpoint (eq. 13) | CONFIRMED | p851 eq. 13 | |
| H10 | Attention `alpha = softmax over the node's hyperedges of a^T(u (+) z) . d(u,z)` | CONTRADICTED | p851 eq. 14 | `(x)` and `(.)` not reproduced; softmax not in paper |
| H11 | Hyperedge to node `exp_0(ReLU(sum alpha log_0(FC(z))))` | CONFIRMED | p851 eq. 15 | |
| H12 | Hypergraph "per appendix B": industry + Wikidata "a company plus all companies linked by the same relation" | CONFIRMED | [A854] Sec. B | first-order only; the paper also has pairwise second-order relations (see P39) |
| H13 | Loss = MSE + pairwise ranking; paper doesn't state it | NOT IN PAPER | p852 "Following [1] ... ranking problem" | correctly labelled inferred |
| H14 | Sharpe = mean/std x sqrt(252), no rf, no costs, "per the authors' code" | NOT IN PAPER | p852 formula includes R_f | attributed to code (fine), but the paper's own definition differs |
| H15 | NDCG also reported | CONFIRMED | p852 Sec. IV-B, Table II | |
| H16 | Data: RSR NYSE, 1737 stocks | CONFIRMED | p850 Table I (1,245 timesteps, 1,737 nodes, [1]) | |
| H17 | Train 2013-15 (756), val 2016 (252), test 2017 (237) | NOT IN PAPER | | dates and lengths unspecified |
| H18 | Paper NYSE: THINK 1.18/0.86, TCONV+DHHAN 1.14/0.81, STHGCN 1.10/0.78 | CONFIRMED | p852 Table II | |
| H19 | TCONV+DHHAN = "Euclidean time, hyperbolic relations" | CONFIRMED | p852 Sec. V.A | |
| H20 | "The paper's code repo is empty" | NOT IN PAPER | p852 footnote 1 gives the URL | paper only claims a release; emptiness is our check |
| H21 | Sec. 3: architecture "follows the paper exactly ... checked equation by equation" | CONTRADICTED | p850 eq. 10, p851 eq. 14 | eq. 10 normalization differs; `(.)` and softmax inferred |
| H22 | Sec. 3: paper lists no hyperparameters | CONFIRMED | pp849-854 | |
| H23 | Sec. 4: model selection not stated in the paper | CONFIRMED | pp849-854 | |
| H24 | Sec. 4: paper divides by max close over the whole period | NOT IN PAPER | | from authors' earlier code |
| H25 | Sec. 4 eq. 14 row: "Symbol lost in the PDF"; ours = Mobius add x distance | CONTRADICTED | p851 eq. 14 | stale (code now has the eq. 8 form) |
| H26 | Sec. 4: Euclidean baseline "not fully specified"; ours = EE (linear convs, mean aggregator) | CONTRADICTED | p852 Sec. V.A, p853 Fig. 3 caption | paper's Euclidean arm is Euclidean temporal conv only (EH) |
| H27 | Sec. 4: batch 1 day, 100 epochs | NOT IN PAPER | | earlier code |
| H28 | Sec. 4: TSE and NASDAQ-Clf are reported | CONFIRMED | p852 Table II | |
| H29 | Sec. 6: paper delta_hg/delta_rel values | CONFIRMED | p850 Table I | |
| H30 | Sec. 6: paper does not say which features give delta_rel | CONFIRMED | [A854] App. A: only "the temporal features" | |
| H31 | Sec. 6: paper's delta_hg = 0.5 "consistent with a small sample" | NOT IN PAPER | [A854] Alg. 1 | value of s and any sampling are not stated |
| H32 | Sec. 6: NDCG caveat "the paper says it follows [38]" | CONTRADICTED | p852 "Following [1]" | [38] only for hyperedges [A854] |
| H33 | Sec. 6: the 500-stock "industry" is the n/a bucket | NOT IN PAPER | p853 Fig. 3a shows 500 | identification is ours |

### 2.3 `docs/POC_PRESENTATION.md` sections 2, 3, 7

| # | Claim | Class | PDF cite | Note |
|---|---|---|---|---|
| C1 | Data: RSR NYSE daily prices 2013-2017 | NOT IN PAPER | Table I gives T=1245 only | dates unspecified |
| C2 | Features MA5/10/20/30 + close, / max price, 16 days x 5 | NOT IN PAPER | | |
| C3 | Model predicts the next day's return | CONFIRMED | p852 Sec. IV-B ("predicted return ratio") | |
| C4 | Hypergraph: industry + Wikidata relation hyperedges | CONFIRMED | [A854] Sec. B | |
| C5 | Model eq. 17 | CONFIRMED | p851 eq. 17 | |
| C6 | TConv 16 to 4 to 1 | NOT IN PAPER | | K and tau unspecified |
| C7 | Top-5 rule, Sharpe = mean/std x sqrt(252), no rf, in the "THINK paper" column ("Same") | CONTRADICTED | p852 SR formula with R_f, "top-k" | |
| C8 | Paper NYSE: THINK 1.18, TCONV+DHHAN 1.14, STHGCN 1.10 | CONFIRMED | p852 Table II | |
| C9 | Sec. 3 table "Dataset: RSR NYSE, 1737 stocks" | CONFIRMED | p850 Table I | |
| C10 | Sec. 3 table "Train/val/test 2013-15 / 2016 / 2017" in the THINK column | NOT IN PAPER | | |
| C11 | Price normalization / max over all years (paper column) | NOT IN PAPER | | |
| C12 | Hyperbolic temporal conv eq. 9-12; DHHAN eq. 13-15 | CONFIRMED | p850-851 | |
| C13 | Eq. 14 written `alpha = a^T (x) (u (+) z) (.) d(u,z)` | CONFIRMED | p851 eq. 14 | the printed form |
| C14 | 4d: `a^T (x) (u (+) z) = tanh(a^T log_0(u (+) z))` | CONFIRMED | p850 eq. 8 | for a scalar output |
| C15 | 4d: "the exact formula has since been confirmed from the PDF" / "now corrected to the exact formula" | CONTRADICTED | p851 eq. 14, p850 eq. 7 | `(x)` matches; `(.)` (eq. 7 ill-formed) and softmax are inferred |
| C16 | Loss and hyperparameters "not stated" | NOT IN PAPER | | correctly labelled |
| C17 | Batch 1 day, ~100 epochs (earlier code) | NOT IN PAPER | | correctly labelled |
| C18 | Epoch selection "not stated in the paper" | CONFIRMED | pp849-854 | |
| C19 | Seeds: "25 runs" | CONFIRMED | p852 Table II caption, Fig. 2/3 | |
| C20 | Ablation arms: hyperedges vs pairwise vs none | CONFIRMED | p853 Sec. V.C, Fig. 3a | |
| C21 | Ablation arms: "all three again with Euclidean layers instead of hyperbolic ones" | CONTRADICTED | p852 Sec. V.A | paper's Euclidean variant changes the temporal conv only |
| C22 | Sec. 5: paper reports THINK as 1.18 +- 0.004 over 25 runs | CONFIRMED | p852 Table II | +- is not defined by the paper |
| C23 | Sec. 7: THINK used the STHAN-SR NDCG evaluator (inferred) | NOT IN PAPER | p852 "Following [1]" | labelled inferred; see H32 |
| C24 | Sec. 7: paper's NYSE delta = 0.5 | CONFIRMED | p850 Table I | |
| C25 | Sec. 7: the 500-stock "industry" is the n/a group | NOT IN PAPER | p853 Fig. 3a | |

### 2.4 `docs/PHASE1_TRACKER.md`

| # | Claim | Class | PDF cite | Note |
|---|---|---|---|---|
| T1 | R1 paper MSE 0.58; Table I matches rg17 (120 snapshots, 1000 nodes) | CONFIRMED | p852 Table II; p850 Table I | rg17 identity is ours |
| T2 | R1 "paper's baselines (~2.05)" | CONTRADICTED | p852 Table II | STHGCN DTT is 1.03; RSR-I is "-" |
| T3 | R2 paper 1.09, baselines 1.11-1.14 | CONFIRMED | p852 Table II | |
| T4 | R3 paper 1.05 | CONFIRMED | p852 Table II | |
| T5 | R4 paper 0.32; Table I 85 nodes, 1293 steps | CONFIRMED | p852 Table II; p850 Table I | |
| T6 | R4 "the cited [35] is a US 10-K text paper" | CONFIRMED | p851 (risk forecasting [35]); [A854] ref [35] (Kogan et al. 2009) | |
| T7 | R4 "CSE [22] has 91 stocks over 2 years" | NOT IN PAPER | [A854] ref [22] title only | from the cited paper, not from THINK |
| T8 | R5 paper 1.18 / 0.86 | CONFIRMED | p852 Table II | |
| T9 | R6 paper 1.19 / 0.81 | CONFIRMED | p852 Table II | |
| T10 | R7 paper 0.49 | CONFIRMED | p852 Table II | |
| T11 | R7 thresholds "resolved" (tertiles), macro/micro F1, lookback | NOT IN PAPER | pp851-852 give none | from the STHGCN repo; labelled |
| T12 | R8 baselines STHGCN, RSR-I are in Table II | CONFIRMED | p852 Table II | |
| T13 | R9 delta_hg 0.5, delta_rel 0.087 | CONFIRMED | p850 Table I | |
| T14 | R9 "paper doesn't define its features" | CONFIRMED | [A854] App. A | |
| T15 | Ground rules: paper uses 25 seeds | CONFIRMED | p852 | |
| T16 | A5 TCONV+DHHAN (Euclidean temporal) is a paper ablation | CONFIRMED | p852 Sec. V.A | |
| T17 | A9 2x2 geometry (HH, EH, HE, EE) | NOT IN PAPER | | paper: HH and EH only |
| T18 | A10 attention without distance is a paper ablation (run here on stocks) | CONFIRMED | p852 Sec. V.B, Fig. 2 | paper ran it on DTT and CPox only |
| T19 | G1 hyperedges vs pairwise is a paper ablation | CONFIRMED | p853 Fig. 3a | |
| T20 | G2 decomposition by size (Fig. 3) | CONFIRMED | p853 Fig. 3a | levels 30/15/5 on a 309-stock graph vs paper 500/15/9/5/3 |
| T21 | G12 hub removal (Fig. 3) at degrees 12, 8, 5 | CONFIRMED | p853 Fig. 3b | paper degrees 31/28/22/16/2 on full NYSE |
| T22 | G5 no hyperedges is a paper ablation | CONFIRMED | p853 Sec. V.C | |
| T23 | Ground rules: dates, normalization, splits | NOT IN PAPER | | labelled ours |

### 2.5 `docs/phase1/R_feasibility.md`

| # | Claim | Class | PDF cite | Note |
|---|---|---|---|---|
| F1 | Paper is six pages plus a one-page appendix, pp849-854 | CONFIRMED | [A854] | repo PDF has only pp849-853 |
| F2 | Hyperparameters, splits, lookbacks, scaling are not stated anywhere | CONFIRMED | pp849-854 | |
| F3 | R2 cites [19] and [20] (arXiv numbers) | CONFIRMED | [A854] refs [19], [20] | |
| F4 | Table I: CPox 522 timesteps, 20 nodes, 1.5, 0.190 | CONFIRMED | p850 Table I | |
| F5 | Appendix B quote on the Sorensen-Dice merge | CONFIRMED | [A854] Sec. B | quote accurate |
| F6 | The SCD sentence "is garbled" | CONTRADICTED | [A854] Sec. B | legible; threshold unspecified; rule reads inverted |
| F7 | "Most likely intended: merge while SCD is above a threshold" | NOT IN PAPER | | labelled [I] |
| F8 | tau = nK | CONFIRMED | p850 eq. 12 | |
| F9 | 4 lags means K=2, n=2 | NOT IN PAPER | | K chosen by us, labelled [I] |
| F10 | R3: [20], 319 nodes, 17,472 steps, baselines 1.19-1.38 | CONFIRMED | p850 Table I; p852 Table II | |
| F11 | R3: the paper gives no hyperedge weight cut | CONFIRMED | [A854] Sec. B | only neighbourhood + SCD merge |
| F12 | R1: [18] Beres et al.; Table I 120 / 1000 / 1.0 | CONFIRMED | [A854] ref [18]; p850 Table I | |
| F13 | R1: baselines "2.03-2.06 for eight of nine" | CONTRADICTED | p852 Table II | seven in 2.03-2.06, STHGCN 1.03, RSR-I "-" |
| F14 | R1: the paper used rg17; per-snapshot hypergraph; target log1p | NOT IN PAPER | | labelled [I]; Sec. IV-A/IV-B do not mention DTT at all |
| F15 | R7: NASDAQ 1245 x 1026, cites [2]; Sec. IV-A and IV-B quotes | CONFIRMED | p850 Table I; p851 Sec. IV-A; p852 Sec. IV-B | quotes verbatim |
| F16 | R7: no thresholds or averaging named | CONFIRMED | pp851-852 | |
| F17 | R7: THINK 0.49, TCONV+DHHAN 0.44, STHGCN 0.40, RSR-I 0.38 | CONFIRMED | p852 Table II | |
| F18 | R8: Table II baseline list (9 baselines) | CONFIRMED | p852 Table II | |
| F19 | R8: STHGCN is Sawhney, Agarwal, Wadhwa, Shah, ICDM 2020 | CONFIRMED | p853 ref [2] | |
| F20 | R8: "RSR-I = inner-product variant" | NOT IN PAPER | p852 Table II row "RSR-I [1]" | labelled [I] |
| F21 | R4: CSE row and [22] Huang et al. | CONFIRMED | p850 Table I; [A854] ref [22] | |
| F22 | R4: [22] is 78 CSI100 + 13 HK stocks, 2015-2016 | NOT IN PAPER | | from arXiv summary |
| F23 | R4: "risk forecasting [35]" is Kogan et al. | CONFIRMED | p851; [A854] ref [35] | |
| F24 | R4: risk = log volatility [I] | NOT IN PAPER | | |
| F25 | R4: RSR-style industry/Wiki hyperedges "do not exist" for Chinese stocks | CONTRADICTED | [A854] Sec. B | paper says it built industry + Wiki hyperedges for TSE and CSE |
| F26 | Consolidated list: Dice threshold, per-snapshot graph, F1 thresholds, baseline tuning unspecified | CONFIRMED | pp849-854 | absence confirmed |
| F27 | "the 25-run count is stated" | CONFIRMED | p852 caption | |
| F28 | "+-1e-3 spreads implausibly tight" (meaning of +-) | NOT IN PAPER | p852 | +- is undefined |

### 2.6 `docs/phase1/R2_chickenpox.md`

| # | Claim | Class | PDF cite | Note |
|---|---|---|---|---|
| R2-1 | Paper THINK 1.09, baselines 1.11-1.14 | CONFIRMED | p852 Table II | |
| R2-2 | Sorensen-Dice merge (Appendix B) has an unspecified threshold | CONFIRMED | [A854] Sec. B | |
| R2-3 | ... and is "garbled" | CONTRADICTED | [A854] Sec. B | legible; merge not implemented at `pygt.py:44-49` |
| R2-4 | Lags 4, horizon 1, single channel, K=2, n=2 | NOT IN PAPER | | labelled inferred |
| R2-5 | Split, standardization, hidden size, lr, epochs | NOT IN PAPER | | labelled inferred |
| R2-6 | Table I row (522 x 20) | CONFIRMED | p850 Table I | |
| R2-7 | tconv2 kernel = seq/kernel | NOT IN PAPER | p850 eq. 12 gives no value | |

### 2.7 `docs/phase1/R8_baselines.md`

| # | Claim | Class | PDF cite | Note |
|---|---|---|---|---|
| R8-1 | RSR-I is Feng et al. 2019 | CONFIRMED | p852 Table II "RSR-I [1]"; ref [1] p853 | |
| R8-2 | STHGCN is Sawhney et al. 2020/21 | CONFIRMED | p852 "STHGCN [2]"; ref [2] p853 (2020) | |
| R8-3 | STHGCN is Euclidean | CONFIRMED | p852 Sec. V.A ("STHGCN, EGCN-H ... use the Euclidean space") | |
| R8-4 | STHGCN has no attention | NOT IN PAPER | | from repo layout |
| R8-5 | RSR-I is pairwise (graph, not hypergraph) | CONFIRMED | p852 Sec. V.A ("pairwise edges in ordinary graphs (RSR-I, EGCN-H)") | |
| R8-6 | Baseline widths, lookback, lr, tuning | NOT IN PAPER | | paper gives none; labelled inferred |
| R8-7 | Paper baseline numbers to compare against (RSR-I 1.05/0.75, STHGCN 1.10/0.78) | CONFIRMED | p852 Table II | |

### 2.8 `R1_tennis.md`, `R3_windmill.md`

Neither file exists yet in `docs/phase1/` (only `R_feasibility.md`, `R2_chickenpox.md`, `R8_baselines.md`). Not audited. Facts they must respect: paper numbers DTT 0.58 / WMill 1.05 (p852 Table II); DTT and WMill hyperedges come from a neighbourhood plus SCD merge with an unspecified threshold ([A854] Sec. B); DTT is a "dynamic network" (p850 Table I) with per-snapshot `G_t` allowed by eq. 16 (p851), but its target and window are not stated. `pygt.py:169-177` builds one hypergraph from the union of training-period snapshots, which is our choice.

## 3. Paper facts we had wrong or missed

1. **The repo PDF has no appendix** (pp849-853 only). Appendices, Algorithm 1 and refs 17-38 are only in the author copy [A854].
2. **Eq. 14 is `a^T (x) (u_j (+) z_i) (.) d_B(u_j, z_i)`** (p851). `(x)` = eq. 8 Mobius matvec; `(.)` = eq. 7's operator. Eq. 7 as printed is `tan((||xy||/y) arctan^-1(||y||)) ||xy||/||y||` with `tan`/`arctan` (not `tanh`/`artanh`) and a bare `y` in a denominator, so it cannot be implemented literally. No softmax appears anywhere in Sec. III-B.
3. **Fig. 3's "Euclidean THINK" and Table II's "TCONV + DHHAN" change only the temporal convolution** (p852-853). The paper has no fully-Euclidean model and no HE model.
4. **Sharpe (p852):** `SR = E[R_a - R_f]/std[R_a - R_f]`, top-k (k unspecified), strategy "Following [1]". No sqrt(252), no value of R_f.
5. **Wiki relations ([A854] Sec. B):** first-order = source plus targets of the same Wikidata relation (a hyperedge); second-order = pairwise (`X -R2-> Z <-R3- Y`). Same construction stated for NYSE, NASDAQ, TSE and CSE, following [38].
6. **DTT/CPox/WMill hyperedges ([A854] Sec. B):** for each node v take `N(v)`, form `{(v, N(v))}`, merge pairs by Sorensen-Dice coefficient "until no two pairs had an SCD score lower than a threshold". Threshold not given; legible, not garbled.
7. **Fig. 3a text vs axis:** text "in increasing order of hyperedge degree" (small first) but the axis runs 500 to 3 with SR falling (reads as large first). Running both schedules is right. Fig. 3a THINK ~1.18 at 500 to ~0.955 at 3; Euclid THINK ~1.11 to ~0.91. Fig. 3b (hub removal): x = node degree 31, 28, 22, 16, 2, with vertical bands annotated `delta_hg` = 0.5, 1, 1, 1.5 (hyperbolicity of the remaining hypergraph rises as hubs go); THINK ends ~0.88, Euclid ~0.85. The Fig. 3b axis starts at 31 while our NYSE max node degree is 37.
8. **Fig. 2 (distance ablation) is on DTT and CPox only:** medians roughly DTT HHN 0.667 vs THINK 0.588; CPox HHN 1.127 vs THINK 1.086 (read off the plot). The text claims the drop "on all datasets".
9. **Table II `+-` is never defined** (std, s.e. or CI). Every `+-` is printed as `e-3`/`e-4` (THINK NYSE 1.18 +- 4e-3). One anomaly: EGCN-H Risk `0.39 +- 8e-2` (probably a typo). RSR-I on DTT is "-".
10. **Hyperbolicity definitions:** eq. 2 (p850) says delta_hg is "the minimal value greater than zero"; [A854] eq. 19 says "smallest non-negative value". Algorithm 1 ([A854]) takes `s` as an input and no value of `s` is given. Our `hyperbolicity.py:11-15` defaults to `s=1`. Table I has DTT delta_rel "-".
11. **Dataset-level delta_rel** ([A854] App. A) uses "the temporal features" with Euclidean distance, `delta_rel = 2 delta / diam(W)`; the features are never specified.
12. **No task or metric is given for DTT** in Sec. IV-A/IV-B; DTT appears only in Tables I-II and Fig. 2 (Table II reports MSE).
13. **Reference numbers of the cited RSR-I and STHGCN papers are not in the THINK PDF.** Not verified here; do not quote them from memory. THINK's own numbers for them are in Table II (NYSE SR/NDCG: RSR-I 1.05/0.75, STHGCN 1.10/0.78; TSE 0.99/0.72 and 1.07/0.74; Clf 0.38 and 0.40).
14. **What the paper does state that we use:** 25 runs (Table II, Fig. 2, Fig. 3); Wilcoxon signed-rank p<0.01 marked `*`; NASDAQ labels "going up, going down, or staying neutral" scored with "F1-score" (no averaging named); code URL in footnote 1 (`github.com/shivamag125/ICDM22-THINK`).

## UNREADABLE - ask user

Nothing was blurry at 600 dpi. The items below are missing, ill-formed as printed, or clipped; a clearer scan helps only where noted.

| # | Location | Issue | What is needed |
|---|---|---|---|
| U1 | Whole repo PDF | Ends at p853; p854 (refs 17-38, Algorithm 1, appendices A and B) is absent | Send a copy including p854, or confirm the author-hosted copy is acceptable (used for every `[A854]` cite; pp849-853 match the repo PDF) |
| U2 | p850 eq. 7 | Legible but ill-formed: `tan((||xy||/y) arctan^-1(||y||)) ||xy||/||y||`; the operator `(.)` used in eq. 14 cannot be implemented as printed | A clearer copy will not help; needs the authors' code or an erratum. Until then `(.)` in `attention.py` is inferred |
| U3 | p850 eq. 9 | Overprint artifact: `v_k(x)` renders with a bold `x` over `(x)`; meaning clear | none |
| U4 | p851 eq. 15 | Sum index printed `s_j in e_i` while elsewhere the node is `v_j` | none, notation slip |
| U5 | p850 eq. 2 vs [A854] eq. 19 | "minimal value greater than zero" vs "smallest non-negative" | authors' definition if a tree-like graph (delta=0) matters |
| U6 | p852 Table II, EGCN-H Risk | Printed `0.39 +- 8e-2` (others e-3/e-4) | confirm typo or leave as printed |
| U7 | p853 Fig. 3b | y-axis clipped: THINK's curve leaves the top of the frame near node degree 31 (start value >1.15 not readable) | a copy with the full axis or the underlying numbers |
| U8 | p853 Fig. 3a/3b | 9-10 plotted points but only 5 labelled x ticks; intermediate hyperedge and node degrees unknown | underlying data or the authors' schedule |
| U9 | Table II `+-` | Statistic (std, s.e., CI) undefined | authors |
