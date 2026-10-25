# Understanding the latest study

The latest candidate combines two separately trained forecasts: the original
THINK+mixer daily core with bounded AH+PM correction, and a restricted dynamic
hypergraph network. Each model family's seed forecasts become eligible-stock
percentile ranks. Their ensemble ranks are averaged 50:50 before selecting the
five or ten highest-ranked holdings. This is one ranked portfolio, rather than
a split of capital between two independently traded portfolios.

Both branches use the declared daily/AH/PM features. Premarket bars end by 09:15
Eastern, the decision is at 09:20 and execution is at the regular-session close.
Rebalancing occurs every five exchange sessions. The transferred dynamic network
uses a train-only correlation snapshot rather than the full author sector and
lead/lag input graph. MASTER and PRISM-VQ are declared NASDAQ transfers too.

The [master charts](../../charts.md) describe seed sensitivity, complementary
errors, weaker models and paired uncertainty. They distinguish the original-five
0.919 Sharpe from the ten-seed 0.896 result at five holdings. Top ten was primary;
top five was a declared secondary portfolio test. The repeated development period
does not supply an untouched confirmation sample.

Use [protocols](../config/README.md) for exact epochs, learning rates, losses,
optimizers, seeds and validation rules. Use [portable results](../results/README.md)
for metrics, annual/monthly summaries, complementarity and uncertainty. The
[workbook](../../docs/NASDAQ100_strategy_comparison.xlsx) retains all comparison
families and diagnostics; its Method tab defines the reported statistics.

Full checkpoints and original seals remain locally under
`runs/ndx_reference_networks_20261024` and `runs/ndx_loss_optimizer_20261024`.
Market inputs remain under `data/nasdaq_monthly_pit/`, including Alpaca bars and
versioned features. Pinned author sources and reference dependencies remain in
`.cache/`. A clone contains the exported summaries, not those large local assets.
The [layout guide](../../docs/PROJECT_LAYOUT.md) explains compatibility and the
archive boundary.
