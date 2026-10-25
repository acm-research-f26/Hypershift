# Archived cohort, NYSE and retired-universe studies

This archive preserves work unrelated to the active historical-monthly NASDAQ-100
study. It contains earlier cohort acquisition/analysis/publication sources,
retired 2026-selected universe reports, pooled results, presentations and document
build artifacts. The original relative directory structure is retained inside
this folder, and every moved file's bytes were verified.

- [Earlier robust-loss protocol](ROBUST_RESEARCH.md),
  [architecture study](ARCHITECTURE_RESEARCH.md), and
  [StockMixer paper review](STOCKMIXER_PAPER_REVIEW.md).
- [Published cohort documents](docs/README.md) and [source map](research/README.md).
- `config/`: old fixed-cohort, robust, random-trial and presentation protocols.
- `presentations/` and `output/`: older published decks and document artifacts.
- `tests/`: tests specific to the older Huber/ranking and retired-universe studies.

From the repository root, archived source imports and synthetic tests are opt-in:

```powershell
python -m hypershift --archived list archive
python -m hypershift --archived test test_huber_ranking test_largecap_retraining -v
```

The launcher restores archived module names, old protocols and root document
aliases for that command. It does not restore the old local datasets/runs or all
published output paths. Full historical reruns require explicitly restoring those
inputs from the [move manifest](../../config/study_layout_20261025.json); retired
universe training remains blocked by its original runner.

Frozen reports keep their original bytes, including old relative links and
machine-local paths. This index is the navigation entry point for the archive.
Original paths can be looked up in the manifest rather than rewriting audited
historical evidence. Current work starts in [latest](../../latest/README.md).
