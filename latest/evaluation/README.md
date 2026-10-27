# Latest study: evaluation

Frozen source, fit and publication-seal verification.

| Source | Purpose |
| --- | --- |
| [seal_ndx_comparison.py](seal_ndx_comparison.py) | Seal audited local experiment artifacts without producing documents. |

Run from the repository root using `python -m hypershift run <module> ...`
(or `python -m hypershift test <test_module> -v` for tests).
Use `python -m hypershift list latest` to discover sources. Do not execute
these nested historical files directly: the launcher supplies their frozen
original module/config paths. See the [study guide](../README.md) and
[shared infrastructure](../../research/README.md).
