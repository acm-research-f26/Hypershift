# Shared research and portfolio machinery

These modules implement panel construction, causal masking, forecast aggregation,
portfolio policies and economic accounting. They are imported by experiment runners.

| Module | Responsibility |
| --- | --- |
| [pit_research_engine.py](pit_research_engine.py) | Historical membership, chronological panels and long-only accounting |
| [ndx_policy_engine.py](ndx_policy_engine.py) | NASDAQ strategy and portfolio policy support |
| [long_short_pit.py](long_short_pit.py) | Signed portfolios, cash and borrow-cost accounting |
| [research_engine.py](research_engine.py), [robust_engine.py](robust_engine.py) | Shared and earlier research-engine utilities |
| [training_device.py](training_device.py) | CPU/CUDA selection and reproducibility settings |

See [training](../training/README.md) for entry points and
[tests](../../tests/README.md) for timing, masking, cost and accounting checks.
The [hardware guide](../../docs/TRAINING_HARDWARE.md) describes backend constraints.
Run historical imports through the root launcher, as described in the
[layout guide](../../docs/PROJECT_LAYOUT.md).
