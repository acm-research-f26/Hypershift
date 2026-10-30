# Model definitions

This directory defines prediction networks, losses and reference adapters. It
does not select winners, execute portfolios or publish performance tables.

| Sources | Model family |
| --- | --- |
| [ndx_shared_think.py](ndx_shared_think.py) | Controlled removal of THINK's stock-ID dense residual mixer; unchanged tensors retain their exact matched seed initialization |
| [think_model.py](think_model.py), [research_models.py](research_models.py), [robust_models.py](robust_models.py) | THINK, mixers and earlier architecture/loss comparisons |
| [ndx_extended_hours.py](ndx_extended_hours.py) | Daily core with bounded after-hours and premarket corrections |
| [ndx_loss_optimizer.py](../../latest/models/ndx_loss_optimizer.py), [pseudohuber_think_mixer.py](pseudohuber_think_mixer.py), [ndx_ranking_objectives.py](ndx_ranking_objectives.py) | Robust losses, optimizer recipes and ranking objectives |
| [ndx_reference_models.py](../../latest/models/ndx_reference_models.py) | MASTER, restricted dynamic hypergraph and PRISM-VQ transfer adapters |
| [ndx_alpaca_models.py](ndx_alpaca_models.py), [ndx_broad_models.py](ndx_broad_models.py), [ndx_corrections.py](ndx_corrections.py) | Matched input, broader feature and frozen-core correction experiments |

Training and hyperparameters are owned by [training](../training/README.md) and
[dated protocols](../../config/README.md). The best observed dynamic blend combines
THINK+mixer AH+PM and a separately trained restricted dynamic network by averaging
their ensemble percentile ranks. It is a strategy built from two model families;
its top-five and top-ten results are different portfolios.

Use the [master charts](../../charts.md) and
[workbook](../../docs/NASDAQ100_strategy_comparison.xlsx) for current comparisons.
Reference adapters have declared NASDAQ-specific adaptations and do not reproduce
the papers' published performance claims.
