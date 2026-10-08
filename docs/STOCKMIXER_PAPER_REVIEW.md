# What the StockMixer paper establishes

Source: Jinyong Fan and Yanyan Shen, **StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting**, AAAI 2024. Reviewed the user-supplied `09728-AAAI24.FanJ.pdf`, especially the experimental setup on printed page 8393 and Tables 2-3 on page 8394. [Publisher PDF](https://ojs.aaai.org/index.php/AAAI/article/download/28681/29322); [official code](https://github.com/SJTU-DMTai/StockMixer).

## Did it beat SPY?

**The reported comparison does not establish that.** Table 2 reports StockMixer Sharpe ratios of 1.465 for NASDAQ, 1.454 for NYSE and 1.586 for an S&P500 stock dataset. It compares StockMixer with other prediction models, including LSTM, GAT, RSR-I, STHAN-SR, ESTIMATE and Linear. There is no buy-and-hold S&P 500 or SPY row. An S&P500 dataset is the stock universe, not an ETF performance benchmark.

The S&P500 dataset contains 474 stocks from January 2016 through May 2022, with 1,006 training days, 253 validation days and 352 test days. The exchange datasets contain 1,026 NASDAQ and 1,737 NYSE names and end in December 2017. These are different universes, trading horizons and periods from our small fixed-cohort experiment. The main paper does not supply the directly comparable, matched-date, net-of-trading-cost SPY backtest needed to answer whether it beat an investable index fund.

The paper averages three repeats and reports significance against model baselines. That is not evidence of persistent annual outperformance against SPY, nor does it validate our local implementation.

## What is relevant to our experiment?

- The paper forecasts one-session return ratios. Our previous test forecast five-session returns, so daily forecasting is a relevant additional experiment. Our daily execution still waits until the following close, avoiding use of the signal close as an available fill.
- Its training objective combines MSE and a pairwise ranking penalty. Huber is a proposed local ablation, not the paper's reported loss.
- The paper uses multi-scale temporal patches with scales 1, 2 and 4 for a 16-session lookback, causal temporal mixing, and a bottleneck for stock-to-market-to-stock mixing.
- Our original `stockmixer` is explicitly a simplified inspiration. Our `think_mix` is a different hybrid: THINK plus a score-level stock mixer. Its favorable 2017 result must not be attributed to the paper's complete StockMixer architecture.
- The new `multiscale_mse`/`multiscale_huber` implement those three multi-scale temporal paths and an eight-state market bottleneck in our common data pipeline. They remain local adaptations rather than certified paper reproductions.

## Huber hypothesis

MSE squares residuals, allowing extreme returns or bad observations to dominate the gradient. Huber is quadratic near zero and linear in the tails. The new experiment uses targets in percentage points and **twice Huber with delta = 1 percentage point**: this preserves MSE curvature near zero and limits tail gradients. Existing ranking penalties are retained where applicable.

This could improve typical cross-sectional ordering, but it might also underweight genuinely informative large winners or losers. Every Huber variant therefore has an MSE comparator with the same architecture, initialization seeds, data and optimizer. All variants retain validation-MSE early stopping, so the comparison isolates training loss; robust early stopping is not tested in this batch.

## Interpretation rule

The objective is a repeatable improvement, not finding one attractive historical line. The complete batch and stop rule are saved in [the experiment plan](../config/robust_experiment_plan.json). Model and portfolio choices use only prior validation periods. Fixed-policy ablations separate loss/architecture effects from policy changes. All outcomes, including failures, are reported. Previously inspected 2024-2025 windows are exploratory, and the retrospectively chosen longer period is not a pristine confirmation test.
