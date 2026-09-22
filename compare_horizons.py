"""Compare daily, weekly, and optionally monthly skill on matching forecast dates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from report_results import markdown_table, plt
from stock_gnn import metrics


def evaluate(frame, dates, tickers, label, subset):
    selected = frame.set_index(["as_of", "ticker"]).loc[
        pd.MultiIndex.from_product([dates, tickers], names=["as_of", "ticker"])]
    actual = selected.actual_return_pp.to_numpy().reshape(len(dates), len(tickers))
    zero_error = np.sqrt(np.mean(actual ** 2))
    rows = []
    for model in ("zero", "train_mean", "ridge", "mlp", "gnn"):
        prediction = selected[f"{model}_return_pp"].to_numpy().reshape(actual.shape)
        result = metrics(actual, prediction)
        rows.append({"horizon": label, "subset": subset, "model": model, "origins": len(dates),
                     "rmse_pp": result["rmse_pp"], "mae_pp": result["mae_pp"],
                     "rmse_improvement_pct": 100 * (1 - result["rmse_pp"] / zero_error),
                     "sign_accuracy": result["sign_accuracy"],
                     "mean_rank_ic": result["mean_daily_rank_ic"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=Path("runs/historical"))
    parser.add_argument("--weekly", type=Path, default=Path("runs/weekly"))
    parser.add_argument("--monthly", type=Path, help="Optional 21-session run to include.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.out = args.out or Path("runs/monthly_horizon_comparison" if args.monthly else "runs/horizon_comparison")
    runs = [args.daily, args.weekly] + ([args.monthly] if args.monthly else [])
    labels = ["Daily (1 session)", "Weekly (5 sessions)"] + (["Monthly (21 sessions)"] if args.monthly else [])
    horizons = [1, 5] + ([21] if args.monthly else [])
    stride = horizons[-1]
    longest = "monthly" if args.monthly else "weekly"
    reports = [json.loads((run / "metrics.json").read_text(encoding="utf-8"))
               for run in runs]
    if [r.get("horizon_sessions", 1) for r in reports] != horizons:
        parser.error("Expected daily=1, weekly=5, and monthly=21 sessions when supplied.")
    if len({r["prices_sha256"] for r in reports}) != 1:
        parser.error("Price snapshots must match for this controlled comparison.")
    for key in ("lookback", "neighbors", "hidden", "epochs", "patience", "batch_size", "lr", "seed"):
        if len({r["arguments"][key] for r in reports}) != 1:
            parser.error(f"Training setting {key} differs between runs.")
    frames = [pd.read_csv(run / "test_predictions.csv", parse_dates=["as_of", "target_date"])
              for run in runs]
    if any(frame.duplicated(["as_of", "ticker"]).any() for frame in frames):
        parser.error("Duplicate stock-date forecasts found.")
    tickers = sorted(frames[0].ticker.unique())
    if any(tickers != sorted(frame.ticker.unique()) for frame in frames[1:]):
        parser.error("Stock universes must match.")
    dates = pd.DatetimeIndex(sorted(set.intersection(*(set(frame.as_of) for frame in frames))))
    if len(dates) < stride:
        parser.error("Not enough common forecast origins.")
    rows = []
    for label, frame in zip(labels, frames):
        rows += evaluate(frame, dates, tickers, label, "all_common_origins")
        rows += evaluate(frame, dates[::stride], tickers, label, "nonoverlap_common_origins")
    table = pd.DataFrame(rows)
    full = table[table.subset == "all_common_origins"].drop(columns="subset")
    spaced = table[table.subset == "nonoverlap_common_origins"].drop(columns="subset")
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "comparison.csv", index=False)
    # Show sensitivity to the starting date, without choosing a phase by its score.
    phase_rows = []
    for offset in range(stride):
        results = evaluate(frames[-1], dates[offset::stride], tickers, labels[-1], f"offset_{offset}")
        gnn = next(row for row in results if row["model"] == "gnn")
        phase_rows.append({"start_offset_sessions": offset, "periods": len(dates[offset::stride]),
                           "gnn_rmse_improvement_pct": gnn["rmse_improvement_pct"]})
    phases = pd.DataFrame(phase_rows)
    phase_filename = f"{longest}_nonoverlap_offsets.csv"
    phases.to_csv(args.out / phase_filename, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, subset, title in zip(axes, (full, spaced),
                                (f"All {len(dates)} common forecast dates", f"Every {stride} sessions: {len(dates[::stride])} periods")):
        models = ["train_mean", "ridge", "mlp", "gnn"]
        width = 0.8 / len(labels)
        for i, (label, color) in enumerate(zip(labels, ["#8196ad", "#d9822b", "#247b68"])):
            values = subset[subset.horizon == label].set_index("model").loc[models, "rmse_improvement_pct"]
            bars = ax.bar(np.arange(len(models)) + (i - (len(labels) - 1) / 2) * width,
                          values, width, label=label, color=color)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.axhline(0, color="#444", linewidth=0.9)
        ax.set_xticks(np.arange(len(models)), ["Train mean", "Ridge", "No graph", "GNN"])
        ax.set(title=title, ylabel="RMSE reduction vs zero-return baseline (%)")
        ax.margins(y=0.25)
        ax.grid(axis="y", alpha=0.15)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Forecast horizons: above zero beats the no-change baseline", fontsize=14)
    fig.savefig(args.out / "horizon_comparison.png", dpi=160)
    plt.close(fig)

    summary = full[full.model == "gnn"]
    summary_text = "; ".join(f"{row.horizon}: **{row.rmse_improvement_pct:.3f}%**"
                             for row in summary.itertuples())
    original_daily = reports[0]["metrics"]["test"]["gnn"]["rmse_pp"]
    original_zero = reports[0]["metrics"]["test"]["zero"]["rmse_pp"]
    text = f"""# Comparison of forecast horizons

Models predict **total compounded returns over {', '.join(map(str, horizons))} trading sessions**, respectively. Twenty-one sessions approximate one month, not a calendar-month-end interval. All models use the same cached 2018–2025 data, 12 stocks, daily features, architecture, seed, optimizer settings, and validation-based stopping rule. No hyperparameters were selected from the new horizon's test results.

On matching forecast dates, GNN RMSE improvement over predicting zero is {summary_text}. Positive is better, negative is worse. This comparison is relative to each horizon's own baseline; raw errors at different horizons are on different scales.

## Comparable dates and leakage prevention

We compare forecasts issued on the same **{len(dates)} dates**, from **{dates[0].date()}** through **{dates[-1].date()}**, for **{len(tickers)} stocks**. The longest-horizon outcome dates run from **{frames[-1].target_date.min().date()}** through **{frames[-1].target_date.max().date()}**. Later origins without outcomes for every horizon are excluded from all models in this comparison. The original daily results remain unchanged: GNN RMSE {original_daily:.4f}, zero-return RMSE {original_zero:.4f}, over {frames[0].as_of.nunique()} dates. Earlier saved reports use their own larger date sets and therefore have slightly different daily/weekly metrics.

The validation and test start dates are anchored to the original daily experiment. We purge h forecast origins before each later block for an h-session target. Every training label is known before validation begins; every validation label is known before testing begins. The shorter eligible training feature period means graph/scaler estimates are refitted on that eligible period. Features still describe 20 past daily returns plus trend and volatility summaries. Only the prediction horizon and necessary sample eligibility change.

## All common forecast dates

{markdown_table(full)}

RMSE and MAE are in percentage points of the relevant total return. `rmse_improvement_pct = 100 × (1 - model_RMSE / zero_RMSE)`. Direction accuracy is a fraction, not a percentage; an always-up/train-mean benchmark is more relevant than zero's neutral sign. Rank IC measures ordering of stocks within each forecast date. Consecutive multi-session forecasts overlap and must not be treated as independent observations.

![Relative forecast error across horizons](horizon_comparison.png)

## Forecasts spaced {stride} sessions apart

To check the result without overlapping outcome intervals, take every {stride}-th origin starting from the first common test origin, fixed before looking at scores. Use those same dates for all models. This leaves **{len(dates[::stride])} periods**; the stock returns within a period remain related, and nonoverlap does not remove all serial dependence. A small number of periods limits how much confidence we can place in differences between models.

{markdown_table(spaced)}

Starting on a different day changes which nonoverlapping periods are included. Here are all {stride} starting offsets for the longest horizon; none is selected as the winning result. These alternative subsets overlap with each other, so they are not independent replications:

{markdown_table(phases)}

## How to interpret this experiment

Switching horizons is useful only if performance improves relative to appropriate baselines for that horizon. Beating zero alone would not establish that the graph adds value: compare with the training-mean and no-graph models too. Small differences in one period do not establish a durable edge. Earlier test results motivated these follow-ups using the same market history, so this is an exploratory comparison, not an untouched independent confirmation. No statistical significance or trading-profit claim is made. New walk-forward periods would be needed for stronger evidence.

The original survivorship, adjusted-price revision, and execution limitations still apply. These are after-close forecasts of adjusted-price returns, not a tradable portfolio simulation.

## Artifacts

- [Comparison numbers](comparison.csv)
- [Longest-horizon nonoverlapping phase check]({phase_filename})
- Weekly run: `{args.weekly.as_posix()}` (REPORT.md, test_predictions.csv, checkpoint, graph, and histories).
- Original daily run: `{args.daily.as_posix()}` (preserved).
- Monthly run when supplied: `{args.monthly.as_posix() if args.monthly else 'not supplied'}`.
"""
    (args.out / "REPORT.md").write_text(text, encoding="utf-8")
    print(full.to_string(index=False))
    print("\nLongest-horizon nonoverlap phase checks:\n" + phases.to_string(index=False))
    print(f"\nSaved comparison to {args.out.resolve()}")


if __name__ == "__main__":
    main()
