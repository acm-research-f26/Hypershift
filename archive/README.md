# Archived research

The active project starts in [latest](../latest/README.md), with shared
historical-monthly NASDAQ-100 infrastructure in [research](../research/README.md).
Current results remain in the [master charts](../charts.md) and
[comparison workbook](../docs/NASDAQ100_strategy_comparison.xlsx).

| Folder | Contents |
| --- | --- |
| [studies_20261025](studies_20261025/README.md) | Unrelated cohort/NYSE experiments, retired-universe reports, old protocols, decks and document artifacts |
| `local_20261025/` | Ignored old datasets, run directories and scratch files; preserved locally by same-volume directory renames |
| [legacy](legacy/README.md) | Original tutorials, reconstruction and early GNN/hourly experiments archived October 5 |

[study_layout_20261025.json](../config/study_layout_20261025.json) records the new
moves and hashes. The earlier `cleanup_manifest.json` records the October 5
archive. Archived code is excluded from normal discovery; opt in with
`python -m hypershift --archived list` for the October 25 study archive.
The earlier tutorial tests retain their separate command:

```powershell
python -m unittest discover -s archive/legacy -p "test_*.py"
```

Nothing was deleted or retrained. Shared helpers required by active NASDAQ models
remain outside the archive. Historical datasets/runs are not copied back merely
by selecting archived source code; see the study archive guide before a rerun.
