# Latest frozen experiment protocols

| File | Scope |
| --- | --- |
| [ndx_reference_networks_20261024.json](ndx_reference_networks_20261024.json) | MASTER, restricted dynamic hypergraph, PRISM-VQ and fixed rank blends |
| [ndx_reference_assets_20261024.json](ndx_reference_assets_20261024.json) | Pinned source implementations and declared factor inputs |
| [ndx_loss_optimizer_20261024.json](ndx_loss_optimizer_20261024.json) | Matched robust-loss and optimizer controls |
| [ndx_confirmation_gate_20261024.json](ndx_confirmation_gate_20261024.json) | Requirements for future confirmation; currently not activated |

These files retain their original bytes. The launcher temporarily provides the
original `config/` filenames while commands run, so completed source/input locks
still check the actual bytes. They are removed again after the child exits.
Ordinary changes still invalidate a frozen experiment; use new protocols/runs
for new research. [Earlier NASDAQ protocols](../../config/README.md) stay separate.
