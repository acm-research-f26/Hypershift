# Latest NASDAQ-100 study: AH+PM and dynamic rank blends

This is the entry point for the latest completed model study. It compares the
THINK+mixer after-hours/premarket parent with complementary MASTER, restricted
dynamic-hypergraph and PRISM-VQ transfers, alongside MSE/Huber/pseudo-Huber and
optimizer controls. All development results use the previously inspected
2022–2025 historical-monthly NASDAQ-100 universe.

| Folder | Contents |
| --- | --- |
| [models](models/README.md) | Reference-network adapters, losses and optimizer recipes |
| [training](training/README.md) | Annual fits, reference workers and protocol freezes |
| [data](data/README.md) | Pinned author implementations and factor acquisition |
| [evaluation](evaluation/README.md) | Source, checkpoint and publication seal checks |
| [reporting](reporting/README.md) | Study reports and latest master-comparison integration |
| [tests](tests/README.md) | Adapter parity, losses, selection and rank blending |
| [config](config/README.md) | Exact frozen protocols and the unactivated confirmation gate |
| [results](results/README.md) | Portable, hash-verified tables and audit evidence |
| [docs](docs/README.md) | Model explanation, interpretation and reproduction boundaries |

The strongest observed five-holding blend has **0.896 Sharpe / 31.87% CAGR across
ten seeds**, and **0.919 / 32.94% in the original five-seed half**. These are
historical candidates, not untouched-period confirmations. The blend still uses
THINK: it averages the two model families' ensemble percentile ranks 50:50,
then chooses one equal-weight portfolio.

The [root master charts](../charts.md) and
[comparison workbook](../docs/NASDAQ100_strategy_comparison.xlsx) retain their
canonical locations and full cross-study coverage. Shared data, accounting and
parent-model code remains under [research](../research/README.md). Earlier active
NASDAQ-100 studies remain indexed in [docs](../docs/README.md); unrelated cohort
and NYSE work is in [archive](../archive/README.md).

Run commands from the repository root:

```powershell
python -m hypershift list latest
python -m hypershift run run_ndx_reference_networks --help
python -m hypershift test test_ndx_reference_models test_ndx_reference_evaluation -v
python -m hypershift run seal_ndx_comparison ndx_reference_networks_20261024
```

The launcher temporarily restores original source and protocol paths, preserving
real source hashes. Edit the organized files. Local market histories, trained
weights and large ledgers retain their original `data/` and `runs/` paths because
completed experiments hash those locations. No models were retrained or scores
changed by this layout cleanup.
