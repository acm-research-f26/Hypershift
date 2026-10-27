# Monthly stock forecasting: what we built and what it learned

The monthly model forecasts the **total adjusted-price return over the next 21
trading sessions**. This approximates a month; it is not a calendar-month-end
forecast. A +2% prediction at an adjusted price of $100 implies an adjusted price
of $102 at the forecast endpoint. It does not predict the intervening path.

The [monthly report](runs/monthly/REPORT.md) contains every baseline and sample
actual-versus-predicted values. The [horizon comparison](runs/monthly_horizon_comparison/REPORT.md)
compares daily, weekly, and monthly forecasts issued on matching dates.

## What information the GNN receives

The model has **23 input numbers per stock per date**, plus a graph describing
which stocks exchange information. These are inputs, not proven causal drivers
or a ranking of learned feature importance.

| Input | Plain-language meaning | What the model could potentially use it for |
| --- | --- | --- |
| Previous 20 individual daily returns | The stock's recent sequence of gains and losses, oldest first | Patterns of continuation, reversal, or changing behavior |
| Average daily return over the last 5 sessions | A short trend summary | Distinguish a recently rising stock from a falling one |
| Average daily return over the last 20 sessions | A longer trend summary | Compare recent movement with the broader recent trend |
| Standard deviation of the last 20 daily returns | How variable recent daily changes have been | Learn whether a relationship differs in calm versus volatile conditions |
| Neighbors' corresponding inputs | Recent behavior of stocks connected to this one | Share information that might reflect common business or market influences |

The trend summaries are arithmetic averages of daily returns, not compounded
returns. They overlap with information already present in the 20 individual
returns. The volatility number describes past variability; it is not a confidence
interval around the forecast.

The graph is estimated from **daily-return correlations in eligible training
history**. Each stock selects up to three positive-correlation neighbors, and we
make the connections symmetric. Negative correlations are omitted. A self-loop
retains a route for the stock's own information. The resulting 18 undirected
edges in this run form these three groups:

| Connected group | Stocks |
| --- | --- |
| Technology-related | AAPL, MSFT, GOOGL, NVDA |
| Financial companies | JPM, BAC, GS, MS |
| Energy-related | XOM, CVX, COP, SLB |

These groups emerged from the correlations; industry labels were not supplied
as model inputs. There are no edges between these groups in this fitted graph.
The graph says that companies tended to move together, not that one company's
movement caused or reliably preceded another's.

This version receives **no trading volume, earnings, valuation ratios, news,
interest rates, inflation, or explicit economic indicators**. Price history
might indirectly reflect some of these influences, but the model cannot
identify them separately from the supplied data. Tickers label nodes; they are
not company descriptions or learned ticker embeddings.

## What training actually does

For each historical date, the model takes the information available at that
close and produces 12 return forecasts. The training answer is the observed
change from that close to 21 sessions later:

```text
target = 100 × (adjusted price 21 sessions later / today's adjusted price - 1)
```

The neural network shares information between connected stocks twice. Between
these steps, it learns combinations of the inputs using a hidden layer with 32
values per stock. PyTorch adjusts 801 weights and biases to reduce squared
forecast errors. The weights are shared across stocks. The graph itself stays
fixed; this is not a model that learns new edges during training.

The learning rate, hidden size, features, seed, and stopping rule are unchanged
from the previous experiments. Only the target horizon and the associated
eligible sample dates change. This isolates the horizon question rather than
combining it with an untested architectural improvement.

Training examples end at September 28, 2022, with outcomes known by October 27.
Validation starts October 28, 2022, with its final outcomes known by May 30,
2024. Test forecasts start May 31, 2024. We exclude 21 forecast origins at each
boundary so every earlier outcome is known before the later block starts.
The model and preprocessing are frozen during testing. Validation selected
epoch 17 for the GNN; training stopped after epoch 37 without further improvement.

Historical examples overlap heavily. We have 1,174 training origins, but that
does not mean 1,174 independent months. Twelve related stocks also do not turn
each market-wide event into twelve independent events.

## Results, including an important limitation

Test results over 377 daily forecast origins, or 4,524 stock-date predictions:

| Model | Monthly return RMSE, percentage points (lower is better) |
| --- | ---: |
| Predict no change | 7.9074 |
| Each stock's average monthly training return | 7.7603 |
| Ridge regression | 7.7630 |
| Neural network without graph | 7.7084 |
| GNN | 7.6995 |

The GNN reduces RMSE by **2.63% versus no change**, but only **0.116% versus the
network without a graph**. That small second number is the relevant evidence
about whether graph connections add value in this setup.

The GNN predicts an increase in **every test case**. Its direction accuracy of
61.47% therefore matches simply always predicting an increase. It varies the
forecast size but has not demonstrated useful detection of down months in this
test. Its average stock-ranking correlation over all forecast dates is also
negative (-0.0577), so lower aggregate squared error does not translate into
evidence that it can rank stocks well.

To avoid counting overlapping monthly outcome intervals, we also use every
21st origin beginning with the first test date. There are only **18 such
periods**. The GNN improves on no change by **2.30%**, but the no-graph network
does slightly better than the GNN on that subset (RMSE 7.4864 versus 7.4905).
The report also shows every alternative starting offset; these subsets are
not independent replications because they overlap with one another.

These results suggest that the monthly target is more promising than the daily
target for this experiment, but **the benefit specifically attributable to
the graph is not convincing yet**. We have revisited the same market period
after seeing earlier results, so this is exploratory evidence. It is not an
untouched final test, a significance claim, or a trading backtest.

## Improvements worth testing, in order

1. **Test stability across more time periods.** Use expanding training windows
   and several later validation/test blocks, refitting the graph and scaler only
   on each training window. Retain a final period that has not influenced design
   decisions. Compare several random seeds using a rule set before viewing test
   results. The idea of chronological evaluation with a gap is described in
   [scikit-learn's time-series split documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
   A gap must reflect the forecast horizon; a generic splitter alone does not
   guarantee that feature timestamps and labels are safe.
2. **Give the model information suited to a monthly question.** Test 3-, 6-, and
   12-month price trends, market-relative returns, volume changes, and longer
   volatility summaries. The present model sees only about one month of recent
   inputs. Add timestamped earnings or fundamentals only when their historical
   release dates and revisions are handled correctly. More inputs can add noise
   as easily as signal, so introduce groups through validation-based comparisons.
3. **Keep each company's own information distinct from its neighbors.** Add a
   separate own-stock branch and combine it with aggregated neighbor information.
   Our two rounds of mixing produce very similar forecasts within each group.
   A residual or GraphSAGE-style design could preserve company differences.
   Neighborhood aggregation is described in the
   [GraphSAGE paper](https://arxiv.org/abs/1706.02216); it is not evidence that this
   proposed change will improve financial predictions.
4. **Improve the relationships in the graph.** Test relationships known at the
   time, such as sectors or supplier/customer links, or lagged relationships
   where one series precedes another. A rolling graph can adapt to changing
   relationships but also introduces noisy estimates and a more complicated
   evaluation procedure. Correlation edges alone are not lead-lag evidence.
5. **Predict uncertainty as well as a central estimate.** Quantile forecasts can
   distinguish a modest expected gain from a wide range of possible outcomes.
   Evaluate whether intervals attain their claimed coverage on future dates;
   changing the output alone does not make uncertainty reliable.

Do not judge these modifications by repeatedly selecting whichever does best
on the present test dates. That would gradually turn the test set into training
feedback, even if no gradient updates explicitly use its rows.

## Alternative models and tradeoffs

These are proposed next experiments, not additional models secretly trained
or demonstrated to outperform our current run.

| Alternative | Why try it? | Main downside |
| --- | --- | --- |
| Historical mean / ridge regression | Cheap, understandable reference models; ridge can show whether richer features help before adding complexity | Constant or linear predictions cannot capture every interaction |
| Gradient-boosted decision trees | Learn nonlinear interactions in a table of trends, volume, fundamentals, and market context | Can fit noise; temporal history and relationships must be represented explicitly as features |
| Small temporal convolutional network or LSTM/GRU | Models the evolution of an ordered history, potentially using a longer lookback | More fitting choices and overfitting risk; does not automatically model company relationships |
| Graph attention network | Learns how much to weight connected neighbors from their features | More parameters and unstable neighbor weights; attention is not a causal explanation |
| Temporal model plus GNN | Separates learning within each stock's history from sharing information across stocks | Most complex option; harder to attribute gains and justify with our small number of independent periods |

For the mechanics behind these alternatives, see the official
[gradient-boosting documentation](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting),
the [temporal convolution/recurrent comparison](https://arxiv.org/abs/1803.01271),
and the [graph attention paper](https://arxiv.org/abs/1710.10903).
Their general capabilities do not establish that they will forecast this stock
universe better. Chronological validation and the same baselines remain necessary.

## When a GNN is a good choice—and when it is not

Its potential advantage is explicit information sharing across an irregular
network of relationships. Shared parameters make it possible to learn a common
prediction rule rather than a separate model for every company. A meaningful
relationship graph can provide a useful modeling assumption, especially if the
input information from related companies is predictive.

Its liabilities include choosing an unhelpful graph, propagating noise, and
blurring differences between companies through repeated averaging. This latter
problem is related to the smoothing behavior studied in
[research on graph convolutions](https://arxiv.org/abs/1801.07606). The graph also
adds data-maintenance work: company relationships can change, timestamps matter,
and dense operations become costly for large universes.

For this project, I would prioritize **better chronological evaluation and
richer, correctly timestamped inputs**, comparing ridge and boosted trees,
before building a larger GNN. A graph is useful if its relationships improve
unseen forecasts; complexity by itself is not a reason to keep it.

## Run it

```powershell
.\.venv\Scripts\python.exe stock_gnn.py --csv data\prices.csv --horizon 21 --out runs\my_monthly_run
.\.venv\Scripts\python.exe report_results.py --run runs\my_monthly_run
.\.venv\Scripts\python.exe compare_horizons.py --monthly runs\my_monthly_run --out runs\my_monthly_comparison
.\.venv\Scripts\python.exe predict.py --checkpoint runs\my_monthly_run\gnn.pt --csv data\prices.csv
```

The final command forecasts from the last date in the supplied file, not the
current live market. The cached snapshot ends on December 31, 2025; its latest
forecast therefore has no observed outcome inside this snapshot. Use
`test_predictions.csv` for forecasts with known historical outcomes. The
existing `runs/monthly` checkpoint and all previous runs are preserved.
