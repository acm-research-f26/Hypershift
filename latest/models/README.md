# Latest study: models

Prediction adapters, robust losses and optimizer recipes.

| Source | Purpose |
| --- | --- |
| [ndx_loss_optimizer.py](ndx_loss_optimizer.py) | Controlled losses and optimizers for the unchanged extended-hours network. |
| [ndx_reference_models.py](ndx_reference_models.py) | Reviewed author neural networks with explicit NASDAQ input/membership adapters. |

Run from the repository root using `python -m hypershift run <module> ...`
(or `python -m hypershift test <test_module> -v` for tests).
Use `python -m hypershift list latest` to discover sources. Do not execute
these nested historical files directly: the launcher supplies their frozen
original module/config paths. See the [study guide](../README.md) and
[shared infrastructure](../../research/README.md).
