# Latest study: tests

Latest-study parity, loss, selection and rank-blend checks.

| Source | Purpose |
| --- | --- |
| [test_ndx_loss_optimizer.py](test_ndx_loss_optimizer.py) | Objective-scale and optimizer checks for the controlled training extension. |
| [test_ndx_loss_optimizer_evaluation.py](test_ndx_loss_optimizer_evaluation.py) | Latest-study unit tests. |
| [test_ndx_reference_evaluation.py](test_ndx_reference_evaluation.py) | Latest-study unit tests. |
| [test_ndx_reference_models.py](test_ndx_reference_models.py) | Latest-study unit tests. |

Run from the repository root using `python -m hypershift run <module> ...`
(or `python -m hypershift test <test_module> -v` for tests).
Use `python -m hypershift list latest` to discover sources. Do not execute
these nested historical files directly: the launcher supplies their frozen
original module/config paths. See the [study guide](../README.md) and
[shared infrastructure](../../research/README.md).
