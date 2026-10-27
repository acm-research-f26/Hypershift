# Latest study results and evidence

These are compact byte-identical exports from the two sealed completed studies:

- [Reference networks](reference_networks/metrics.csv): standalone transfers,
  parent controls and fixed rank blends, including the dynamic-hypergraph family.
- [Loss/optimizer controls](loss_optimizer/metrics.csv): MSE, Huber, pseudo-Huber,
  AdamW, RAdam and momentum-SGD comparisons.
- [Export manifest](MANIFEST.json): original locations and SHA256 hashes for all
  45 copied summaries and metadata files.

Each study folder includes annual/monthly summaries, prediction/ranking quality,
complementarity, absolute Sharpe and paired intervals, protocol and audit records.
`FINAL_PUBLICATION.json` is a copy of the original seal; it describes the full
original run, not a claim that this compact export contains every sealed artifact.

The large daily curves, trade ledgers, checkpoints and source snapshots remain
under the original ignored `runs/` locations. The unchanged master comparison
is in [charts.md](../../charts.md) and
[docs/NASDAQ100_strategy_comparison.xlsx](../../docs/NASDAQ100_strategy_comparison.xlsx).
No metric or model was changed for this export.
