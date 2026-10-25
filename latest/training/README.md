# Latest study: training

Protocol freezes, annual training and reference-network workers.

| Source | Purpose |
| --- | --- |
| [ndx_reference_training.py](ndx_reference_training.py) | Causal common feature adapter and finite reference-network training. |
| [run_ndx_loss_optimizer.py](run_ndx_loss_optimizer.py) | Finite, frozen loss/optimizer training experiment; no outcome-based extension. |
| [run_ndx_reference_networks.py](run_ndx_reference_networks.py) | Freeze and run the finite author-network transfer comparison. |
| [run_ndx_reference_parallel.py](run_ndx_reference_parallel.py) | Execution-only resume: parallel independent missing seeds, frozen fit math. |

Run from the repository root using `python -m hypershift run <module> ...`
(or `python -m hypershift test <test_module> -v` for tests).
Use `python -m hypershift list latest` to discover sources. Do not execute
these nested historical files directly: the launcher supplies their frozen
original module/config paths. See the [study guide](../README.md) and
[shared infrastructure](../../research/README.md).
