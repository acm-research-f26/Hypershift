# Industry-preference sensitivity to expanded filing coverage

Protocol ../SEC_COMBINED_PREFERENCE_PROTOCOL.md (8f18b91) and implementation 456dc2a were committed before calculating this comparison. We preserved the first-signal, training-label Hlabel ranks and the earlier SEC matches, then added only previously unmatched names with independently checked dated SEC cover/header evidence. No industry was imputed for the remaining unmatched names.

| Fold | Earlier covered | Expanded covered | Earlier between-industry share of covered rank variation | Expanded share |
|---|---:|---:|---:|---:|
| 2024 | 81 / 101 | 98 / 101 | 13.37% | 10.22% |
| 2025 | 79 / 101 | 100 / 101 | 13.84% | 9.66% |

The additional 17 and 21 names have mean Hlabel ranks 0.383 and 0.470, compared with 0.534 and 0.518 among the original covered names. Five and eight additional names respectively use the original fewer-than-30-training-label median fallback. The three remaining unmatched 2024 names and one 2025 name are kept separate in SOURCE_GROUPS.csv. The original covered-only fractions reproduce exactly; all 202 joined rows, source-group means/counts and both expanded variance partitions were independently recalculated. Seven focused tests pass.

The decrease shows sensitivity to coverage; it does not estimate the proportion of strategy returns caused by industry or establish what either neural model learned. These are first-origin snapshots of a training-label-mean control, not independent market histories or neural predictions. The supplemental candidates were discovered from a current-vintage ticker list but independently matched to earlier SEC cover/header evidence. Neither evidence set certifies unchanged identity after filing, the NDX price-series entity, current SIC or as-published membership. SIC/FF12 is not GICS. Small and singleton categories remain in INDUSTRY_GROUPS.csv. Detailed joins, counts and hashes are alongside this report; earlier results are preserved. No financial account, protected-window score, model training or replay gate changed.
