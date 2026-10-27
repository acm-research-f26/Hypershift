# Saved-rank variation, persistence and blend boundaries

2026-10-27. Frozen protocol5d2a487, original saved-export ranks, no neural fitting. Four annual folds, three fixed seed groups, B(parent)/D(dynamic)/E(rankblend); top5/top10. All36variation cells,3546adjacent-origin rows and121998eligible stock/boundary rows retained. Four focused tests pass, including weighted ANOVA with unbalanced eligibility, constant scores, fold boundaries and fractional top-k ties.

## Rankings are predominantly persistent within each annual fit

Between-stock fraction of total eligible score variation, ten-seed ensembles:

| Annual fit | Parent B | Dynamic D | Blend E |
|---|---:|---:|---:|
|2022|94.25%|93.92%|96.17%|
|2023|99.93%|87.93%|92.97%|
|2024|90.47%|74.61%|87.65%|
|2025|89.07%|86.67%|90.33%|

The weighted sum-of-squares decomposition respects how often each ID is eligible. All total sums equal between-ID plus within-ID sums. Annual fits stay separate. These fractions describe held-out forecast behavior, not fractions of return explained, causal importance or an implementable test-mean control. Persistent means can result from identity weights, stable features or repeatedly similar changing signals.

Across adjacent five-session origins, E's eligible-common-ID rank correlation averages.931–.973per year. Top10target-weight overlap averages73.0–79.4%; top5overlap64.5–78.4%. Parent2023is especially static: correlation.9993 and top10overlap99.18%. High full-universe persistence does not imply zero top-tail trading: small score changes near a cutoff can replace holdings. The dynamic model varies more within IDs than the parent, particularly in2024; whether this variation adds causal forecast information remains unisolated.

## The blend creates a new selection

Average E-selected target weight by whether each standalone account selected the same stock at the same k, ten seeds/all201origins:

| Category | Top5 | Top10 |
|---|---:|---:|
| Both standalone models select |21.79%|29.30%|
| Parent only |37.91%|33.76%|
| Dynamic only |24.18%|28.30%|
| Neither standalone model selects |16.12%|8.63%|

The neither category is possible because averaging percentile ranks can promote names moderately high in both families over names extreme in only one. It is direct evidence that rank blending does more than combine standalone holdings.

Observed-label-weighted five-session returns for top10categories are both+1.044%,parent-only+.250%,dynamic-only+.565%,neither−.144%. These are retrospective frictionless entry-to-exit origin outcomes weighted by target allocation. They omit share drift, caps, execution fees, settlement and terminal-account handling; do not add them as canonical P&L or infer causality from the best category. All missing-label weight and every annual/group category are retained in BLEND_SELECTION_CATEGORIES.csv. Labels were used only for outcome summaries, never inference eligibility or selections.

## Mechanism implication and remaining limits

Combined with reconciled persistent NVIDIA exposure and large issuer concentration, this makes a stable upper-tail selection mechanism a concrete hypothesis. The evidence does not yet show whether learned security-indexed filters, AH/PM features, correlation groups or learned hyperedges produce those persistent rankings. Hpred/changing-score controls remain behind the strict checkpoint gate; Hlabel equivalence is unproven. Exact cost-aware category attribution, causal style matching and the training-only surrogate remain open. Boundary stock rows preserve ex-ante scores/weights and labels for later audited analysis.
