# Hypershift

The 250-stock forecasting experiment is implemented in `experiments/market_ablation`.
Run `uv run python -m experiments.market_ablation --help` for its phase commands.
Use a fresh `--output local-experiments/runs/<name>` directory for each new run.
The compact [main ablation notebook](notebooks/main_ablation.ipynb) reads saved
results without repeating training.

Local experiment code, prompts, diagnostic reports, forecasts and checkpoints
belong in `local-experiments/`, which is excluded from Git. Shared source data
stays in `local-data/`. Keep reusable library code and tests in the source tree.
