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
directories under `.cache/test_tmp`. The current active suite contains 171 tests,
including 15 ownership, cleanup and compatibility checks. Latest-study tests live
in [latest/tests](../latest/tests/README.md) and participate in normal discovery.
Reference parity tests
also require the pinned author sources/dependencies fetched for the experiment;
see [data acquisition](../research/data/README.md).

Unit tests do not replace [saved-checkpoint and portfolio audits](../research/evaluation/README.md)
or establish out-of-sample profitability. Archived tutorial tests remain separate
under [archive/legacy](../archive/README.md).
The two archived cohort test modules can be included with
`python -m hypershift --archived test -v`; their source imports and old protocol
aliases are restored only for that command. That combined suite has 177 tests.
