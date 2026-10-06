"""Create readable tables and plots from an already completed experiment."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_gnn import metrics


def markdown_table(frame: pd.DataFrame) -> str:
    rows = [[str(v) for v in frame.columns]]
    rows += [["undefined" if pd.isna(v) else f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in frame.itertuples(index=False, name=None)]
    rows.insert(1, ["---"] * len(frame.columns))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/monthly"))
    parser.add_argument("--ticker", default="AAPL", help="Stock shown in the time series and sample table.")
    args = parser.parse_args()
    report = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    horizon = report.get("horizon_sessions", 1)
    predictions = pd.read_csv(args.run / "test_predictions.csv", parse_dates=["as_of", "target_date"])
    # Accept original daily artifacts as well as new horizon-explicit columns.
    predictions = predictions.rename(columns={"actual_next_adjusted_close": "actual_target_adjusted_close",
        "gnn_predicted_next_adjusted_close": "gnn_predicted_target_adjusted_close"})
    if args.ticker not in predictions.ticker.unique():
        parser.error(f"Ticker {args.ticker} not in this experiment.")
    table = pd.DataFrame(report["metrics"]["test"]).T.rename_axis("model").reset_index()
    table["rank_ic_days"] = table["rank_ic_days"].astype(int)
    table.to_csv(args.run / "comparison.csv", index=False)
    individual = []
    for ticker, rows in predictions.groupby("ticker", sort=False):
        # Each stock's error metrics over time; cross-sectional rank IC is not applicable here.
        actual, pred = rows.actual_return_pp.to_numpy(), rows.gnn_return_pp.to_numpy()
        result = metrics(actual[:, None], pred[:, None])
        individual.append({"ticker": ticker, "rmse_pp": result["rmse_pp"], "mae_pp": result["mae_pp"],
                           "r2_vs_zero": result["r2_vs_zero"], "sign_accuracy": result["sign_accuracy"]})
    per_stock = pd.DataFrame(individual)
    per_stock.to_csv(args.run / "per_stock_metrics.csv", index=False)
    sample = predictions[predictions.ticker == args.ticker].tail(10).copy()
    sample["error_pp"] = sample.gnn_return_pp - sample.actual_return_pp
    columns = ["as_of", "target_date", "gnn_return_pp", "actual_return_pp", "error_pp",
               "gnn_predicted_target_adjusted_close", "actual_target_adjusted_close"]
    sample["as_of"] = sample.as_of.dt.strftime("%Y-%m-%d")
    sample["target_date"] = sample.target_date.dt.strftime("%Y-%m-%d")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax = axes[0, 0]
    colors = ["#d9822b" if m == "gnn" else "#8196ad" for m in table.model]
    ax.bar(table.model, table.rmse_pp, color=colors)
    for i, value in enumerate(table.rmse_pp):
        ax.text(i, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    ax.set(title="Held-out forecast error: lower is better", ylabel="RMSE (percentage points)")
    ax.set_ylim(0, table.rmse_pp.max() * 1.2)
    ax = axes[0, 1]
    ax.scatter(predictions.actual_return_pp, predictions.gnn_return_pp,
               s=6, alpha=0.16, color="#285a83", rasterized=True)
    limit = float(np.max(np.abs(predictions[["actual_return_pp", "gnn_return_pp"]].to_numpy()))) * 1.05
    ax.plot([-limit, limit], [-limit, limit], color="#d9822b", linestyle="--", linewidth=1, label="Perfect forecast")
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="Actual return (%)",
           ylabel="GNN forecast (%)", title="All test stocks and dates; full range")
    ax.legend(frameon=False)
    ax = axes[1, 0]
    recent = predictions[predictions.ticker == args.ticker].tail(60)
    ax.plot(recent.target_date, recent.actual_return_pp, label="Actual", color="#285a83", linewidth=1.2)
    ax.plot(recent.target_date, recent.gnn_return_pp, label="GNN", color="#d9822b", linewidth=1.5)
    ax.axhline(0, color="#bbb", linewidth=0.7)
    ax.set(title=f"{args.ticker}: final 60 target dates", ylabel=f"{horizon}-session total return (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    ax = axes[1, 1]
    history = pd.read_csv(args.run / "gnn_history.csv")
    ax.plot(history.epoch, history.train_mse_pp2, label="Training", color="#285a83")
    ax.plot(history.epoch, history.validation_mse_pp2, label="Validation", color="#d9822b")
    ax.axvline(report["best_epochs"]["gnn"], color="#555", linestyle=":", label="Saved epoch")
    ax.set(title="Learning curve; test data excluded", xlabel="Epoch", ylabel="MSE (percentage points squared)")
    ax.legend(frameon=False)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.15)
    first, last = predictions.target_date.min().date(), predictions.target_date.max().date()
    fig.suptitle(f"{horizon}-session stock-return forecasts | {first} to {last}", fontsize=16, fontweight="bold")
    fig.savefig(args.run / "comparison.png", dpi=160)
    fig.savefig(args.run / "comparison.pdf")
    plt.close(fig)

    gnn = report["metrics"]["test"]["gnn"]["rmse_pp"]
    zero = report["metrics"]["test"]["zero"]["rmse_pp"]
    difference = (gnn / zero - 1) * 100
    verdict = f"The GNN's RMSE is {abs(difference):.2f}% {'higher (worse)' if difference >= 0 else 'lower (better)'} than the zero-return baseline."
    behavior = {
        "gnn_positive_forecast_fraction": float((predictions.gnn_return_pp > 0).mean()),
        "always_up_sign_accuracy": float((predictions.actual_return_pp > 0).mean()),
        "gnn_sign_accuracy": report["metrics"]["test"]["gnn"]["sign_accuracy"],
        "gnn_min_forecast_pp": float(predictions.gnn_return_pp.min()),
        "gnn_max_forecast_pp": float(predictions.gnn_return_pp.max()),
    }
    (args.run / "forecast_behavior.json").write_text(json.dumps(behavior, indent=2), encoding="utf-8")
    nonoverlap_text = ""
    if "test_nonoverlap" in report["metrics"]:
        nonoverlap_table = pd.DataFrame(report["metrics"]["test_nonoverlap"]).T.rename_axis("model").reset_index()
        nonoverlap_table["rank_ic_days"] = nonoverlap_table["rank_ic_days"].astype(int)
        nonoverlap_text = (f"## Nonoverlapping target intervals\n\n"
            f"Take every {horizon}-th forecast origin starting at the first test origin, selected before inspecting results. "
            f"This gives {report['nonoverlap_origin_count']} periods. Shared endpoint prices are allowed; "
            "no daily return is counted in two target intervals. This removes overlap, not all market dependence.\n\n"
            + markdown_table(nonoverlap_table) + "\n\n[These forecasts and actuals](test_predictions_nonoverlap.csv)\n")
    text = f"""# Historical prediction results

Data source label: **{report['source']}**. Forecasts estimate the **total compounded return over the next {horizon} trading sessions**, starting at each observed close. The graph, scaler, and network weights remain frozen throughout the test block. The daily input features and architecture are unchanged.

{verdict} This single experiment does not establish a reliable trading advantage.

The GNN predicts positive returns in **{100 * behavior['gnn_positive_forecast_fraction']:.2f}%** of cases. Its direction accuracy is **{100 * behavior['gnn_sign_accuracy']:.2f}%**, compared with **{100 * behavior['always_up_sign_accuracy']:.2f}%** for always predicting an increase. A high direction score alone can reflect a generally rising sample rather than successful identification of future declines. [Forecast behavior diagnostics](forecast_behavior.json).

## Dates and sample counts

{markdown_table(pd.DataFrame(report['splits']).T.rename_axis('split').reset_index())}

The evaluated target dates run from **{first}** through **{last}**, covering **{predictions.target_date.nunique()} target dates**, **{predictions.ticker.nunique()} stocks**, and **{len(predictions):,} stock-date forecasts**. All stocks on a date belong to the same split. A {horizon}-origin gap separates adjacent blocks so every earlier label ends strictly before the next block starts. For horizons above one session, adjacent daily forecasts have overlapping outcome windows; they are not independent evidence.

## Comparison with simpler models

{markdown_table(table)}

RMSE and MAE use percentage points of the **{horizon}-session total return**. Smaller is better. `r2_vs_zero` is improvement in squared error over predicting zero, not conventional centered R-squared. A negative value loses to zero. Sign accuracy treats a zero prediction as neutral (it is only correct when actual return is zero). Daily rank IC is the mean cross-sectional Spearman correlation calculated per forecast date, even for a multi-session target; undefined dates are omitted. `undefined` means undefined, not zero. Compare horizons by their improvement over their own baseline, not their raw error magnitudes.

{nonoverlap_text}

![Forecast comparisons](comparison.png)

## Actual numbers: {args.ticker}, final 10 test sessions

Rows are selected by date, regardless of performance. All return and error columns are in percentage points; price columns use the provider's adjusted-close scale in USD. Forecast error is predicted minus actual return.

{markdown_table(sample[columns])}

The predicted target adjusted close is today's adjusted close multiplied by `1 + predicted_return_pp / 100`. It is a direct {horizon}-session forecast restarted from each day's observed close, not a recursive simulation of intermediate prices. Adjusted closes are historical accounting series and can differ from the prices actually quoted on those dates.

## GNN results by stock

{markdown_table(per_stock)}

## Files and reproducibility

- [Every prediction and actual value](test_predictions.csv)
- [Model comparison CSV](comparison.csv)
- [Exact settings, split dates, software versions, and input checksum](metrics.json)
- [Input price snapshot](prices.csv)
- [Graph edge weights](graph_weights.csv)
- [Learning history](gnn_history.csv)
- [Exportable chart PDF](comparison.pdf)

The stock list contains companies surviving at selection time; this limits the historical inference. Provider revisions and corporate-action adjustments are not point-in-time data. The sample is 12 chosen US stocks, not the entire stock market. No commissions, spreads, execution delays, or portfolio rules are modeled. The target starts at an already observed close, so this error comparison is not an executable trading backtest. Hyperparameters were fixed before inspecting the results for this horizon. Follow-up horizon experiments reuse the historical period already inspected for earlier horizons, so they are exploratory comparisons, not fresh independent confirmation. Further tuning should use validation or new walk-forward folds, with another untouched final test period. Twenty-one sessions approximate a trading month; they are not calendar-month-end intervals.
"""
    (args.run / "REPORT.md").write_text(text, encoding="utf-8")
    print(verdict)
    print(f"Created report and charts in {args.run.resolve()}")


if __name__ == "__main__":
    main()
