# Latest study: reporting

Metrics, uncertainty and integration into the master comparison.

| Source | Purpose |
| --- | --- |
| [report_ndx_loss_optimizer.py](report_ndx_loss_optimizer.py) | Evaluate and independently audit the frozen training comparison; no documents. |
| [report_ndx_reference_networks.py](report_ndx_reference_networks.py) | Identical portfolios, uncertainty and complementarity for reference transfers. |
| [update_ndx_new_findings_20261024.py](update_ndx_new_findings_20261024.py) | Publish the sealed loss/optimizer and reference-network findings to the catalog. |

Run from the repository root using `python -m hypershift run <module> ...`
(or `python -m hypershift test <test_module> -v` for tests).
Use `python -m hypershift list latest` to discover sources. Do not execute
these nested historical files directly: the launcher supplies their frozen
original module/config paths. See the [study guide](../README.md) and
[shared infrastructure](../../research/README.md).
