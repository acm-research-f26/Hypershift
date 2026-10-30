# Initialization robustness, retraining and the SR>1.5 target

**The target was not achieved.** The fixed 120-fit retraining study and the
separately frozen 198-account factor-control arm produced no recipe with net
Sharpe above 1.5 in every predefined ensemble. The two arms retain all **774
accounts**, including all individual seeds, untrained controls, cost stresses
and failed methods. The result is improved initialization consistency at lower
selected performance, not a robust replication of 1.7 Sharpe.

## What explained the original discrepancy

In 2023, each old THINK model's stock forecasts tracked its stock-specific mixer
biases almost perfectly: mean daily rank correlation was approximately 0.995.
The trained biases retained approximately 0.999 rank correlation with values
reconstructed from the seeded initialization. The original account also averaged
32.6% of NAV in GOOG and GOOGL combined. These are specific initialization and
concentration mechanisms associated with the high historical result; they do
not prove that the whole network or every year was untrained.

Buying the retrospectively identified winning names would introduce lookahead.
Instead, the new experiment removed random stock-ID offsets at initialization,
used a date-centered Huber/pairwise ranking objective, split prior-year validation
into checkpoint-selection and diagnostic halves, and retained a third predefined
five-seed group. It also tested issuer consolidation and causal factor proxies.

[Detailed attribution and parameter integrity](../ndx_seed_attribution_20261029/REPORT.md)

## A more balanced historical result

All values below use the existing 20% basket-volatility target, continuous QQQ
short cap of 75%, base transaction costs and 3% annual short borrowing. The
retrained rows use loss-selected checkpoints of the rank-aligned training path.

| Ensemble | Old selected retention strategy | Retrained, five-security retention | Retrained, five issuers |
|---|---:|---:|---:|
| Original five | 1.731 | 1.168 | 1.171 |
| New five | 0.648 | 1.269 | 1.147 |
| Third predefined five | Not fitted in old model study | 1.345 | 1.200 |
| Combined ten | 1.381 | 0.837 | 1.023 |
| Combined fifteen | Not fitted in old model study | 1.057 | 1.235 |

![Historical ensemble comparison](balanced_ensembles.png)

The five-issuer recipe has the highest observed minimum Sharpe across all five
ensembles in this retraining grid. Its three disjoint five-seed groups differ
by only about 0.053 SR, compared with the old original/new gap of 1.084. That
improvement comes with lower original-five and combined-ten performance. Its
original/new/third annualized returns are 22.02% / 21.44% / 22.35%; combined ten
and fifteen return 19.04% / 22.61%.

At 2x transaction costs plus 6% borrowing, the same recipe's original/new/third/
ten/fifteen Sharpes are **0.983 / 0.969 / 1.013 / 0.848 / 1.062**. It therefore
does not establish a robust SR>1.5 result under the cost assumptions either.

The five-security retention recipe has the highest weaker-disjoint-group SR,
but the pooled ten/fifteen versions are worse. Averaging seed rankings changes
which stocks are selected, retention state and risk sizing; it is not an average
of the seed-group accounts. More seeds need not monotonically raise Sharpe.

## What did not work

- Ranking-selected checkpoints did not outperform the loss-selected path in
  these financial comparisons. The 18 trained recipes all miss the declared
  five-ensemble performance target.
- Three-snapshot selection did not recover the old 1.7 headline consistently.
- The separately specified momentum/risk/liquidity controls and fixed neural
  blends also failed. Their strongest weaker-group hybrid has original/new/
  third/ten/fifteen SR of 0.751 / 0.837 / 0.911 / 0.762 / 0.891. This is an
  adaptation-specific finding, not a general rejection of momentum research.
- The deterministic training-mean-rank control achieved SR 1.123 with retention
  and 0.902 with issuer consolidation. Neural incremental performance depends
  on the allocation and ensemble; not every neural variant beats that control.

[Every retraining recipe](REPORT.md) · [Every factor/control recipe](../ndx_factor_controls_20261029/REPORT.md)

## Paper interpretation

Do not cherry-pick the 1.731 cohort and present it as representative of the
model. It can be labeled a selected historical best case, with the weaker
cohort, combined ensemble and search history disclosed. The corrected study
supports a narrower claim: neutral initialization and ranking-aligned training
can reduce observed seed-cohort disparity, but this experiment did not retain
the headline SR or establish reliable SR>1.5.

Initialization, objective, training duration and validation design changed
together; this is not a factorial causal estimate of each intervention. The
issuer result also changes the portfolio rule. All 2022–2025 history has been
repeatedly inspected. Disjoint initialization groups share the same market
history, while pooled groups overlap their constituent models. No untouched
future confirmation or multiple-search-adjusted significance is claimed.

[Suggested reporting and original-result wording](../ndx_seed_attribution_20261029/PAPER_REPORTING.md)

## Verification and implementation history

The 576 retraining accounts and 198 factor-arm accounts match independent
dollar-position accounting within 1e-12 daily return. The original three
accounts' stock attribution reproduces their paths within 6.67e-16. Forty old
THINK checkpoint files match their published hashes. The focused model/account/
workspace tests (47) and causal-factor tests (3) passed.

After 54 fits, a scheduling-only amendment moved independent family/year jobs
into two processes. Both THINK/AdamW and dynamic/Adam resumed-epoch checks
reproduced saved parameters and validation forecasts exactly. The scientific
code, budget, preserved fit prefixes and seed lists remained fixed.

The factor runner completed all 198 accounts before a pandas `style` attribute
collision stopped summary generation. The separate finalizer repaired explicit
column access, recovered paths from saved NAV ledgers, reconstructed targets and
verified independent replay. All original metrics and ledgers remained
byte-identical. No strategy or outcome was retuned during this repair.

[Main audit](AUDIT.json) · [Scheduling amendment](EXECUTION_AMENDMENT.json) ·
[THINK resume check](RESUME_CHECK.json) · [Dynamic resume check](RESUME_DYNAMIC_CHECK.json) ·
[Factor aggregation repair](../ndx_factor_controls_20261029/AGGREGATION_AMENDMENT.json)
