# Latest study: data

Pinned reference source and factor acquisition.

| Source | Purpose |
| --- | --- |
| [fetch_ndx_reference_assets.py](fetch_ndx_reference_assets.py) | Restore hash-pinned public research assets; never execute downloaded code. |

Run from the repository root using `python -m hypershift run <module> ...`
(or `python -m hypershift test <test_module> -v` for tests).
Use `python -m hypershift list latest` to discover sources. Do not execute
these nested historical files directly: the launcher supplies their frozen
original module/config paths. See the [study guide](../README.md) and
[shared infrastructure](../../research/README.md).
