# From price forecasts to direction, hyperedges, and hyperbolic geometry

Start with [the actual experiment results](runs/direction/REPORT.md), then use this guide to understand the code. The new experiment estimates whether each stock will rise over **21 trading sessions**. It uses the same real 2018–2025 adjusted-close dataset as our earlier models. It is a research foundation: the results currently do **not** establish an advantage for a trading bot.

## 1. What question are we asking?

Imagine stopping the historical tape after the market closes on a particular day. We give the model only information available by that close. It produces one number per stock, such as 0.58. This is its estimated probability that the adjusted close 21 sessions later will exceed today's adjusted close.

The historical answer is simple: `up = future_adjusted_close > current_adjusted_close`. Up is 1; down or exactly unchanged is 0. This label concerns the two endpoint prices, not whether the stock rises at any moment during the month. We do not discard small returns. We do not introduce a neutral class in this experiment. Adjusted prices incorporate corporate-action adjustments; these are not forecasts of the raw quoted dollar price.

A probability becomes a class using a threshold. With a 0.55 threshold, a score of 0.58 is called up and a score of 0.52 is called down/flat. Our threshold is selected on validation data, separately for each trained model. It is not necessarily 0.50. We save results at 0.50 too, so you can distinguish model quality from that decision rule. Calling a score a probability does not guarantee that its probabilities are well calibrated.

**Direction is only one ingredient of a trading decision.** A hypothetical trade with a 60% chance of gaining 1% and a 40% chance of losing 2% has expected return `0.6*1% - 0.4*2% = -0.2%`, before costs. A trading system also needs estimated payoff sizes, execution prices, position sizing, and rules for staying out.

## 2. What does this model know?

For each stock it receives 23 numbers: the last 20 daily percentage returns in chronological order, the mean return over the last five sessions, the mean over 20 sessions, and the 20-session return standard deviation. These summarize recent movement and volatility. The lag positions preserve order, but the small encoder is a fully connected layer, not a recurrent network or temporal convolution.

The inputs exclude news, earnings, volume, intraday prices, interest rates, fundamentals, and actual supply-chain records. The model cannot learn information that is absent from its inputs. Relationship models additionally receive a training-derived graph or hypergraph. No model is given the future prices when making its test predictions.

Standardization subtracts a training-set mean and divides by a training-set standard deviation for each feature. This puts differently sized inputs on comparable scales. Those statistics are then frozen for validation, testing and saved-model inference.

## 3. What is a hyperedge?

An ordinary graph edge connects two stocks. A hyperedge connects a **group** in one relationship. For example, a hypothetical group might contain several semiconductor companies. A company can belong to several groups: an industry group, a supplier group, and a geographical group.

Consider three stocks A, B and C. One hyperedge `{A,B,C}` says there is one shared group. Three ordinary edges `{A,B}`, `{A,C}`, `{B,C}` give the same pairwise connections, but do not retain the identity of that shared group. A hypergraph model can build a group summary and then decide how much each member should use it. This is the higher-order representation idea behind [hypergraph neural networks](https://arxiv.org/abs/1809.09401).

We represent membership using an **incidence matrix H**. Rows are stocks, columns are groups, and a 1 means membership. It is not a matrix of stock-to-stock correlations. In this run H has 12 rows and six columns.

For the runnable example, we build each candidate group from a stock and its five strongest positively correlated neighbors using training-period returns only. Identical groups are merged. One resulting group is AAPL, MSFT, GOOGL, NVDA, GS and MS. This is a co-movement group, not a verified industry or corporate relationship. The surprising cross-industry members illustrate why those two meanings must not be confused. All six groups are saved in `runs/direction/geometry.json`.

Group membership is frozen after training. Group summaries and attention weights change with each day's inputs. Thus dynamic attention here does not mean that we have implemented a changing hypergraph topology.

## 4. What does Gromov hyperbolicity measure?

It asks about the shape of a **distance system**: how close is it, in a specific metric sense, to the behavior of distances on a tree? It does not ask whether stock prices are predictable.

Start with a root w and two points x and y. The Gromov product is

`(x,y)_w = [d(w,x) + d(w,y) - d(x,y)] / 2`.

On a tree, this is the length of the shared part of the paths from w toward x and y. Thinking about shared paths gives some intuition for why the quantity can describe branching structure. Away from a tree, the corresponding consistency conditions can fail; delta measures their worst failure.

The code uses an equivalent **four-point test**. For every four stocks A, B, C and D, calculate:

1. `d(A,B) + d(C,D)`
2. `d(A,C) + d(B,D)`
3. `d(A,D) + d(B,C)`

Sort these sums. Half the difference between the largest and second-largest is that quadruple's contribution. Delta is the maximum over quadruples. We compute all 495 distinct quadruples for 12 stocks, so this calculation is exact for our chosen finite distance matrix. It is not a sampled estimate. The exhaustive method scales poorly to thousands of stocks.

For a four-node cycle with unit sides, the sums are 2, 4 and 2, giving delta 1. For a tree, the two largest sums agree, giving zero. **A complete graph also has zero vertex-metric delta**: all distinct-node distances are 1, so every sum is 2. This counterexample matters: low delta alone does not prove that stocks have a meaningful economic hierarchy. See the [SageMath definition and examples](https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/hyperbolicity.html).

### Which distances did we use?

For the hypergraph diagnostic, two stocks are adjacent when they share at least `s` distinct hyperedges. Distance is the shortest number of these steps. With s=1, one shared group is enough. With s=2, two shared groups are required. Because we merge duplicate groups, duplicate columns do not artificially increase this count.

Our s=1 graph has delta 0.5 and diameter 2. Its normalized value `2*delta/diameter` is 0.5. Increasing s to 2 disconnects it into five components. There is then no finite global shortest-path metric, so we report component diagnostics and `null` for global delta. Substituting zero for infinite distances would be wrong.

We separately measure Euclidean distances between stocks' standardized **training-return trajectories**. Each stock becomes a vector of its training-period daily returns, standardized within that stock and divided by sqrt(number of observations). For that particular representation, normalized delta is approximately 0.1308. This is different from the shortest-path diagnostic. Changing the features, scaling or time interval can change it.

These diagnostic numbers do not determine our model's curvature or update its weights. They describe a possible reason to investigate a geometry. The decisive experiment is still whether that geometry improves predictions on later data.

## 5. What is hyperbolic geometry doing inside the network?

The hyperbolic model stores intermediate representations inside a Poincare ball. You can imagine the two-dimensional version as the inside of a disk. Near the boundary, a small movement in ordinary coordinates corresponds to a large hyperbolic distance. This geometry can give branching structures more representational room in a small number of dimensions. It does not add information about the future.

Our `hyperbolic` model performs these steps:

1. A shared linear layer reads each stock's 23 inputs. A bounded transformation turns them into a 24-number description of recent behavior.
2. `exp0` maps that description into the ball. We keep points safely inside its boundary to avoid numerical problems.
3. Stocks send their descriptions to their groups. `einstein_midpoint` computes a geometry-aware group average by converting to Klein coordinates, averaging with Lorentz weights, and converting back.
4. Each stock measures hyperbolic distances to the summaries of groups it belongs to. Softmax converts negative distances into attention weights. A learned temperature controls how strongly distance affects attention. Nonmember groups get zero attention.
5. The model combines the relevant group summaries, maps the result back with `log0`, and combines it with the stock's own description.
6. A final layer outputs a logit. Applying sigmoid converts that logit into a number between zero and one.

These operations are in `hypergraph.py`. The geometry-aware averaging follows standard Poincare/Klein formulas; see [Hyperbolic Image Embeddings and its linked implementation](https://arxiv.org/abs/1904.02239). Our curvature is fixed at -1. We do not learn it. The learned parameters remain in ordinary tangent coordinates, so the optimizer is standard AdamW rather than a Riemannian optimizer.

The `hypergraph` comparison uses the same memberships, encoder, decoder and number of parameters, but ordinary means and Euclidean distances. Comparing these two models isolates the geometry more cleanly than comparing unrelated large and small networks. The ordinary GNN uses a separate pairwise, positive-correlation top-three graph. The graph-free neural network uses only its own stock's description. Logistic regression is the simpler linear baseline.

## 6. How this relates to THINK

[THINK](https://tylersnetwork.github.io/papers/icdm22-think.pdf) combines hyperbolic temporal processing with distance-aware hypergraph attention. Its stock hyperedges use industries and Wikidata relations. NASDAQ direction classification has up/down/neutral labels and uses F1; NYSE/TSE ranking uses NDCG and Sharpe. It reports averages over 25 runs.

Our model uses binary monthly labels, correlation groups, a flat lag encoder, simplified attention and five seeds. This is therefore an inspired implementation, not a reproduction. The paper does not fully specify the neutral band, F1 averaging or NDCG convention. Our choices are explicit below; our numerical scores are not directly comparable with its tables.

The [authors' linked repository](https://github.com/shivamag125/ICDM22-THINK) was empty when inspected on 2026-09-28. A faithful reproduction would require resolving these protocol details and obtaining the same datasets, relation timestamps, architecture and evaluation setup.

## 7. PyTorch, from the fundamentals

A **tensor** is an array of numbers. Our training input has shape `[dates, stocks, features]`, here `[1174, 12, 23]`. A batch might contain 128 dates. It keeps all stocks for each date together so a relationship layer can exchange information within that date.

An `nn.Module` is a function with stored state. Some state consists of adjustable parameters, such as the weights of `nn.Linear`. Other state, registered as a buffer, includes fixed group memberships and graph adjacency. Buffers are saved with the model but are not optimized.

A linear layer computes weighted sums plus biases. Initially its weights are random; there is no financial expertise hidden in them. `forward` defines how inputs travel through the layers to make a prediction. Calling `model(x)` runs that function. It does not train the model by itself.

The logit is an unrestricted score: zero becomes probability 0.50 after sigmoid, and a positive logit becomes a probability above 0.50. A **loss** grades the prediction against the historical label. Binary cross-entropy penalizes confident wrong predictions strongly. We use `BCEWithLogitsLoss`, which combines sigmoid and cross-entropy in a numerically stable calculation. We therefore pass it logits, not already-sigmoided probabilities. See [PyTorch's loss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html).

This is the essential training loop, with batching and early stopping omitted:

```python
optimizer.zero_grad(set_to_none=True)  # Clear the previous batch's gradients.
logits = model(x)                      # Make predictions with current weights.
loss = loss_function(logits, labels)   # Compare them with historical outcomes.
loss.backward()                       # Calculate how weights affected this loss.
optimizer.step()                      # Adjust weights using those gradients.
```

`backward()` uses automatic differentiation: PyTorch remembers the operations that produced the loss and applies the chain rule to calculate gradients. A gradient describes how a small parameter change would change the loss locally. AdamW uses those gradients, along with running estimates of their behavior, to update parameters. It does not reason about companies; it adjusts numbers to reduce historical prediction error. The official [autograd introduction](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) explains this machinery.

An **epoch** is one pass through the training examples. We allow up to 150 epochs, use batches of 128 dates and learning rate 0.003, clip unusually large gradients, and stop after 20 epochs without improved validation loss. We restore the weights with the lowest validation loss. Dropout temporarily hides some intermediate activations during training; `model.eval()` turns that randomness off. `torch.no_grad()` skips gradient bookkeeping for inference.

The `state_dict` checkpoint stores the learned weights and buffers. Our checkpoint also saves stock order, scaling, horizon and threshold so future inputs receive the same transformations. A checkpoint is not a copy of all training data or a set of future prices.

## 8. How do we prevent the model from seeing the answer?

Time order determines the split. Training origins run from January 2018 through September 2022; validation origins run from October 2022 through April 2024; test origins run from May 2024 through December 2025. The final test outcome is December 31, 2025. Exact dates are in `experiment.json`.

A monthly label uses prices after its forecast date. We therefore remove origins near split boundaries until every earlier block's last target is strictly before the next block's first forecast. This is why merely splitting rows by date is insufficient for multi-session targets. Scaling and relationship construction use only training observations. Weight updates use only training labels. Checkpoint choice and threshold choice use only validation labels. Test labels are used for evaluation.

Shuffling batches *within* the training block is fine for this independent-window model: it does not move validation or test examples into training. The model has no persistent hidden state that carries information between shuffled batches.

The test window was already examined in earlier versions of this project. Consequently, this is an exploratory historical comparison, even though the code keeps test labels out of optimization. A fresh final holdout must remain unexamined while choosing future improvements.

## 9. How do we judge it?

| Metric | Plain-language meaning | Important qualification |
|---|---|---|
| Accuracy | Fraction of correct up/down calls | Always-up gets 61.47% here because most labels are up. |
| Macro F1 | Compute an F1 score for each class, then average | Balances attention to the two classes; not a percentage of profitable trades. |
| Balanced accuracy | Average the fraction of actual ups caught and actual downs caught | An always-up classifier scores 0.50. |
| AUC | How often an actual-up example scores above an actual-down example | 0.50 means no ranking discrimination; this pooled metric differs from within-date stock ranking. |
| Brier score | Mean squared probability error | Lower is better; compare with the training prevalence baseline. |
| NDCG@3 | How useful the three highest-ranked stocks were relative to the best possible ranking | Our explicit relevance is max(actual percentage return, 0); this rewards return magnitude. |
| Sharpe | Mean excess portfolio return relative to its variability | Depends on strategy, costs, timing, risk-free assumption and annualization. |

F1 combines **precision** (how many predicted ups were really up) and **recall** (how many actual ups were caught), using their harmonic mean. We also compute F1 treating down as the positive class, then average the two. An always-up classifier has zero down-class F1. This explains why another model can improve macro F1 while still providing weak information. Our full confusion matrices, up/down F1, MCC and log loss are saved too.

For NDCG, we use logarithmic position discounts and linear positive-return gains; all-negative dates are excluded and counted. Tied predicted scores receive expected DCG over their possible orders. That prevents a constant baseline from gaining an arbitrary advantage from ticker ordering.

For trading, we rank the probabilities and form a long-only top-three basket, irrespective of the classification threshold. Signals generated after close t enter at the next session's close. We hold 21 sessions, liquidate, and leave one cash session before the next basket. We charge 10 basis points per side. Exact ties split allocation across tied stocks. Constant scores consequently give an equal-weight basket of all 12 stocks.

The trading return has a different endpoint from the classification label because execution is delayed. That is deliberate and reported explicitly; it prevents pretending that today's completed close was available before a trade at that same close. A future deployment-focused version should train directly on the chosen executable holding interval.

We calculate both raw cycle Sharpe and annualized Sharpe with zero risk-free return. Annualization is `sqrt(252/22)` for these cycles, not `sqrt(252)` for monthly observations. Daily closing valuations determine drawdown. There are only 18 completed baskets: a good-looking Sharpe from so few periods would still be weak evidence.

## 10. What did we learn, and what should change next?

Across five seeds, the hyperbolic model's mean macro F1 is 0.450, AUC 0.441, and annualized net Sharpe 0.383. The equal-weight basket's net Sharpe is 1.101. Its cumulative net return is 38.60%, compared with 11.30% averaged across hyperbolic runs. Seed variation is not a confidence interval, and the hyperbolic and Euclidean variants are very close. See the report for all models, first-date actual outcomes and the full limitations.

The main benefit of the GNN idea is shared information: another company's behavior may help describe this company's situation. Hyperedges let that sharing reflect groups. Geometry offers a possible way to represent relationships efficiently. The costs are more assumptions, more complex debugging, sensitivity to incorrect or stale relationships, and possible smoothing away of useful stock-specific signals. No component earns its place merely by being sophisticated.

My suggested next experiments, in priority order:

1. **Improve evaluation before increasing model size.** Use rolling training windows, fit all preprocessing within each window, and reserve a new final period. Build a point-in-time universe including delisted stocks. This costs data and engineering effort but addresses the biggest weaknesses of the current evidence.
2. **Add information with known availability times.** Test volume, market/sector returns, longer momentum and volatility windows, and correctly timestamped earnings or fundamentals. These are hypotheses to test, not promised improvements. Revised macro data and current corporate relationships can leak future information into a backtest.
3. **Compare simple alternatives on the same splits.** Logistic regression is cheap and inspectable but linear. A [boosted-tree model such as XGBoost](https://arxiv.org/abs/1603.02754) can test nonlinear interactions in tabular features, at the cost of extra tuning and overfitting risk. A small [temporal convolutional network](https://arxiv.org/abs/1803.01271) can learn patterns across longer ordered histories without a graph, but adds parameters and does not guarantee a financial edge. These sources establish model families, not stock-market profitability.
4. **Improve relationships only with a testable hypothesis.** Compare point-in-time industry and supply-chain groups, trailing-window correlations, random groups of matching sizes, and no groups. A random-group control helps tell whether genuine relationships add information or aggregation merely regularizes the model. Changing memberships requires rebuilding them using only information available at each forecast.
5. **Align the objective with the intended action.** Try an execution-aligned label and a cost-aware neutral band chosen before testing. Estimate payoff sizes as well as direction, validate probability calibration, and test a no-trade rule. A ranking loss might suit stock selection better than binary cross-entropy, but should be compared under exactly the same trading assumptions.
6. **Revisit hyperbolic complexity after the simpler tests.** Larger justified universes, better temporal encoders, and validation-selected curvature are reasonable research directions. Additional flexibility increases the need for fresh data and controlled comparisons. Low delta is a reason to investigate, not a reason to skip these controls.

## 11. Run and inspect it

From the project folder in PowerShell:

```powershell
# Train every comparison model across five fixed seeds; use a new output folder.
.\.venv\Scripts\python.exe train_direction.py --csv data\prices.csv --horizon 21 --out runs\my_direction

# Build the readable results report and plots.
.\.venv\Scripts\python.exe report_direction.py --run runs\my_direction

# Restore a checkpoint and score the last date available in the CSV.
.\.venv\Scripts\python.exe predict_direction.py --checkpoint runs\direction\hyperbolic_7.pt --csv data\prices.csv

# Run the forecasting and direction/geometry checks.
.\.venv\Scripts\python.exe -m unittest -v test_stock_gnn test_direction
```

The last-date prediction is dated December 31, 2025 for this snapshot. It is not a live September 2026 forecast, and its future outcome is absent. `predictions.csv` contains the historical forecasts whose outcomes are known. Changing `--horizon` to 1 or 5 retrains the classifier for daily or weekly endpoints; it does not reproduce a paper benchmark or turn an existing monthly checkpoint into a daily model.

Code reading order: `train_direction.py` for the experiment, `hypergraph.py` for model operations, `direction_metrics.py` for evaluation, and `test_direction.py` for concrete examples of correctness. The original regression model remains available in `stock_gnn.py` for comparison.
