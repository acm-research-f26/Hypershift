# W7: co-author and lab-org code search (extends W6)

2026-10-06. CPU only, no `results/` writes. Authors per `docs/paper/icdm22-think.pdf` p.849: Shivam Agarwal, Ramit Sawhney, Megh Thakkar, Preslav Nakov, Jiawei Han, Tyler Derr. W6 repos (sthan-sr-aaai, hyper-stockgat-www, fast-eacl, profit-naacl, finclass-uai, CryptoBubbles-NAACL, sthgcn-icdm, hats, ICDM22-THINK) were not redone.

## 1. Sources searched

| Source | Method | Result |
|---|---|---|
| GitHub users `ramitsawhney27`, `pnakov`, `hanjiawei`, `JiaweiHan`, `tderr`, `derrlab`, `Crisp-Lab`, `midas-iiitd`; `meghthakkar`, `tyler-derr`, `tylerderr` | `gh api users/<u>` | `ramitsawhney27` (1 repo, EMNLP'18 Hinglish tweets), `pnakov` (1 repo `alt_public`, publication list), `hanjiawei` (5 unrelated notes repos; identity unverified). `tderr`, `derrlab`, `Crisp-Lab`, `midas-iiitd` have 0-1 repos, none relevant (identity unverified). `meghthakkar`, `tyler-derr`, `tylerderr`: 404. Nothing relevant. |
| `shivamag125` (16 repos) | `gh api` | `ICDM22-THINK`: size 0, last push 2022-10-10, no branches, no forks. `Temporal_Relational_Stock_Ranking` is a **fork of fulifeng/Temporal_Relational_Stock_Ranking**, last push 2019-09-02, head aa905c1 is a README edit (no THINK or hypergraph change). Other repos unrelated. |
| `midas-research` org (69 repos, `gh api orgs/midas-research/repos`) | listing | Relevant: sthan-sr-aaai, sthgcn-icdm, hyper-stockgat-www, fast-eacl, profit-naacl, finclass-uai (W6), plus new `hyperbolic-tlstm-sigir`, `man-sf-emnlp`, `multimodal-financial-forecasting` (cloned, section 2a). No repo named THINK, hypergraph-hyperbolic, or hyperbolicity. |
| Other orgs: `mbzuai-nlp` (Nakov), `gtfintechlab`, `mbzuai-oryx` | `gh api` existence only | Exist; the searches below surfaced no THINK-related repo. Not crawled repo by repo. |
| GitHub repo search ("THINK hypergraph hyperbolic", "hyperbolic hypergraph attention", "hyperbolic hypergraph stock", "DHHAN", "s-walk hyperbolicity hypergraph", "gyromidpoint hypergraph", "ICDM22-THINK", "Temporal Hypergraph Hyperbolic Network", "hyperbolic hypergraph neural network time series") | `gh search repos`, REST search | Only the empty `shivamag125/ICDM22-THINK`; paper lists (naganandy/graph-based-deep-learning-literature, gzcsudo/Awesome-Hypergraph-Network) that list THINK without code; `cdk925/H2AML` (blockchain AML, unrelated authors, not examined). |
| GitHub code search ("hypergraph_nyse.npy", "hypergraph_nasdaq", "DHHAN", "gyromidpoint hypergraph") | `gh search code` | `hypergraph_nyse.npy` only in sthan-sr-aaai `training/train_nyse.py` (W6) and a copy of it (YuTian315/GNN-for-different-fields). No builder anywhere. "DHHAN", gyromidpoint: nothing relevant. |
| Homepages: Sawhney (`sites.google.com/iiitd.ac.in/ramitsawhney/publications`), Derr (`tylersnetwork.github.io`, hosts `papers/icdm22-think.pdf`), Agarwal (W6), `par.nsf.gov/biblio/10633923` (THINK record, NSF award 2019897) | WebFetch | No code links for THINK; no post-2022 THINK follow-up on Sawhney's page. Derr page fetch gave no publication or code list. `ramitsawhney.com` does not resolve. Thakkar and Han homepages not found by search. ideals.illinois.edu/items/131426, dl.acm.org and dblp.org returned 403 (not read). |
| Citing works | Semantic Scholar API (12 citations) | Surveys (TKDE, IEEE Access, TMLR), epidemic papers (arXiv 2503.20114: reference list only, no THINK numbers or code link; checked), "Hyperbolic simplicial convolutional network" (ESWA 2026), "Spatiotemporal Hypergraph Learning for Stock Return Sequence Prediction" (ICBAR 2024), adaptive hybrid hypergraph conv (Physica A 2025), others. Only the two stock/hyperbolic ones could report THINK numbers; neither was readable, so UNKNOWN. |

## 2. Findings per item

(a) **THINK-part code.** None released. The authors' hyperbolic code in other repos is the HGCN lineage (Chami et al.) or geoopt, a different design from paper eq. 6-17:
- `external/hyper-stockgat-www` (cbd6f40) `training/layers/hyp_layers.py` L117-142 `HypAgg`: `logmap0`, then (attention-weighted) adjacency matmul in tangent space, then `expmap0` and `proj`. This is tangent-space aggregation, not a gyromidpoint. Attention is `DenseAtt` (`training/layers/att_layers.py` L9-27): `sigmoid(Linear(cat(x_i,x_j)))` times adjacency, with no hyperbolic distance and no softmax.
- `external/hyperbolic-tlstm-sigir` (b6d6f2a) `code/hyrnn/nets.py` L9-30 `mobius_linear`: geoopt `mobius_matvec` plus `mobius_add` bias, with Mobius GRU cells. Not HNN++ Poincare FC, and no beta-concatenation (a grep for `concat|beta|gyro|midpoint` over both repos finds only GAT head concat and Riemannian-Adam betas).
- `man-sf-emnlp` (393fcd9) and `multimodal-financial-forecasting` (e4960a1): no hyperbolic or hypergraph code.

Consequence: nothing here settles eq. 7 (the odot operator), eq. 10 (normalisation) or the attention softmax. Our `src/hypershift/models/*.py` choices stay INFERRED (not in paper). A weak hint only: the authors' earlier attention used a sigmoid score, not a softmax (a different model; not evidence for THINK).

(b) **Hypergraph builder.** Not found. sthan-sr-aaai only loads `hypergraph_nyse.npy` (W6). No repo, code-search hit or homepage links a builder. Our cache v2 (4350 hyperedges, max degree 114, App. B p.854) cannot be checked against edge counts. Unchanged.

(c) **delta_hg / delta_rel code.** Only the HyperStock-GAT `hyperbolicity.py` sampler (W6). No s-walk-distance (Algorithm 1, p.854) or delta_rel code anywhere. Nothing new to run beyond W5/W6.

(d) **Protocol, seeds, 25-run mean+-std.** No new code. Siblings remain print-only, fixed seed 123456789, no aggregation (W6). Nothing explains the paper's +-1e-3.

(e) **Other Table II datasets and follow-ups.** No code for DTT, CPox, WMill, Risk, Clf. No later paper by these authors reusing THINK with code was found. 12 citing works exist, but none located reports THINK NYSE/NASDAQ numbers. Two stock-related ones (ACM 10.1145/3718751.3718862, and 10.1145/3768292.3770389 from a search hit, not confirmed to cite THINK) returned HTTP 403.

## 3. What we can take (actionable)

1. Nothing changes the model or the tables.
2. Optional, about 1 h plus user help (ACM blocks the bot): read the two stock-related citing papers manually and note whether they re-run THINK. An independent THINK Sharpe/NDCG on RSR NYSE/NASDAQ under a stated protocol would be a third-party datapoint for the closure report.
3. Optional, low expected value: email the first author or Derr for the THINK code or logs (the only thing that can move the study, per W6 section 6). The repo being empty since 2022-10-10 suggests a reply is unlikely.
4. Optional ablation, only on a decision (1 smoke plus 1-3 h GPU; the GPU queue owns the GPU): an HGCN-style arm (tangent-space aggregation plus sigmoid `DenseAtt`) to test whether the authors' earlier attention style behaves differently from our eq. 14 reading. No evidence it was used in THINK.

## 4. Effect on REPORT_CLOSURE.md

None. No code or numbers were found that reproduce or contradict the closure conclusions. The co-author, lab-org and homepage routes are now searched and empty, which reinforces "THINK's own code is unavailable".

## 5. Engineering log
Shallow clones added to `external/` (git-ignored): hyperbolic-tlstm-sigir, man-sf-emnlp, multimodal-financial-forecasting. `gh` authenticated; about 25 API calls, no rate-limit issues. Temp file `/tmp/ho.pdf` only.
