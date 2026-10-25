# Research and launcher tests

These tests cover causal feature/label timing, historical membership masks,
loss gradients, reference-model parity, validation selection, portfolio costs,
Sharpe inference and workspace compatibility. They use synthetic fixtures and
mocked HTTP responses where applicable.

Run from the repository root:

```powershell
python -m hypershift test -v
python -m hypershift test test_ndx_reference_models test_ndx_loss_optimizer -v
python -m hypershift test test_workspace_layout -v
```

The launcher supplies the original flat module names and keeps test temporary
directories under `.cache/test_tmp`. The current full suite contains 173 tests,
including 11 ownership, cleanup and compatibility checks. Reference parity tests
also require the pinned author sources/dependencies fetched for the experiment;
see [data acquisition](../research/data/README.md).

Unit tests do not replace [saved-checkpoint and portfolio audits](../research/evaluation/README.md)
or establish out-of-sample profitability. Archived tutorial tests remain separate
under [archive/legacy](../archive/README.md).
