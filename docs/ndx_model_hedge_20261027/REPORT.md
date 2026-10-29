# Own-model hedge signals — October 27, 2026

The own-model signals did not improve Sharpe. Raw forecasts never triggered. Prior-year bias correction triggered hedging throughout 2023, reducing gains in that rebound. The strongest observed Sharpe came from the continuous partial-hedge control: **1.454 original-five / 1.325 ten-seed**. This is a control outcome, not successful model timing.

|Strategy|Seeds|Sharpe|CAGR|Max drawdown|95% SR interval|
|---|---|---:|---:|---:|---|
|Continuous / hedge up to 50%|Original five|1.454|27.79%|-14.56%|[0.463, 2.441]|
|QQQ MA 200 / hedge up to 50%|Original five|1.315|28.10%|-17.05%|[0.325, 2.309]|
|Calibrated family agreement / hedge up to 50%|Original five|1.111|26.27%|-20.56%|[0.154, 2.078]|
|Five issuers + 20% volatility target|Original five|1.178|28.57%|-20.56%|[0.223, 2.137]|
|SPY|Benchmark|0.498|10.85%|-24.55%|[-0.367, 1.419]|
|QQQ|Benchmark|0.469|11.93%|-34.87%|[-0.408, 1.394]|
|Continuous / hedge up to 50%|Ten seeds|1.325|24.83%|-14.78%|[0.388, 2.238]|
|QQQ MA 200 / hedge up to 50%|Ten seeds|1.234|26.24%|-14.82%|[0.300, 2.178]|
|Calibrated family agreement / hedge up to 25%|Ten seeds|1.045|24.64%|-21.69%|[0.130, 1.984]|
|Five issuers + 20% volatility target|Ten seeds|1.112|26.81%|-21.69%|[0.190, 2.047]|

![Comparison](high_sharpe_comparison.png)

## Signal design and timing

The fixed grid uses eligible-market mean<0, proposed five-issuer basket return<0, at least 60% negative forecasts, or both model-family means<0. Each family averages its raw percentage-point predictions across its fixed seeds; the two family predictions receive equal 50:50 weight. These raw averages are separate from the percentile-rank blend used to select holdings. Four additional rules subtract each family/group/year's mean prior-year validation forecast error. Errors average equally by date. No slope or threshold was optimized on test returns.

Calibration uses saved validation predictions from the same annual fit. Those labels precede the first test signal. Checkpoints were selected on that validation fold, so it is not an independent calibration sample. No new neural-network fit was made. Features end 09:15 ET, decisions form 09:20 ET, shares use preceding-close NAV/marks/ADV, and fills occur at that day's regular close. Rebalances remain every five exchange sessions. Every neural account holds five issuers, apart from boundary ties/trading effects, and optionally shorts QQQ.

Raw signals stayed off on all 201 test dates across all three groups. The calibrated gates all produced the same 50 weak dates, all in 2023. Those four nominal model signals therefore give identical paths within each action and seed group. The annual intercept absorbed the prior-year return regime and did not provide useful within-year market discrimination here. This finding concerns these fixed tests; it does not rule out every possible market-timing model.

## Best model signals that actually triggered

- Original five: `cal_mean/hedge50`, SR 1.111, CAGR 26.27%, drawdown -20.56%.
- Ten seeds: `cal_basket/hedge25`, SR 1.045, CAGR 24.64%, drawdown -21.69%.
- New five: `cal_mean/hedge25`, SR 0.596, CAGR 13.73%, drawdown -17.99%.

The highest own-model Sharpe if no-op rules are included is just the unhedged baseline: 1.178 original-five / 1.112 ten-seed / 0.677 new-five. The new-five continuous hedge gives 0.811 Sharpe, materially below the original-five leader. Different seed groups share the same market and the ten-seed ensemble includes both halves.

## Hedge mechanics and stress

Cash exits remove longs when weak. Hedge 25/50 keeps the baseline 20% basket-volatility-target stock sleeve and shorts QQQ within min(25/50% NAV,unused long budget,20%/QQQ 20-session annualized volatility). Switch 50 exits stocks and uses a similarly volatility-capped 50% short. The always-on control uses hedge 50 at every rebalance. Model triggers do not use QQQ price trend; QQQ trailing volatility sizes every short for comparability. All targets have gross exposure<=100%, with possible realized drift. The signed engine retains 1% ADV caps, cancellations, adjusted economic-price dividend/settlement approximations, calendar-day borrowing and the fee reserve.

37 recipes x 3 seed groups x 3 trading costs=333 accounts. 27 actual short recipes x 3 groups x 2 extra borrow assumptions=162 more, **495 total**. Base borrow is assumed 3% annual; stress rates 1%/10% at base trading costs. Trading costs are also tested separately at 2x/4x. These are assumptions, not historical broker quotes. Idle cash and short-proceeds rebate are zero. Actual locates, recalls, margin enforcement, taxes and full market impact are not modeled. All [accounts](metrics.csv), [annual results](annual.csv), [targets](targets.csv), [calibration estimates](calibration.csv) and [signal decisions](signals.csv) remain available.

## Statistical interpretation

There are 120 fixed contrasts: 32 model rules versus matched baseline plus 8 switches versus matched cash, each in three seed groups. Exactly 60 contrasts have identical paths. The inherited bootstrap rejected them as zero variance after every account had been saved. A timestamped [statistical amendment](STATISTICAL_AMENDMENT.json) retains the frozen source/grid/accounts, assigns deterministic zero differences and zero-width intervals to identical paths, and bootstraps the other 60 jointly after deduplicating identical columns. No epsilon was introduced and no trading outcome changed.

10,000 joint circular-block draws use 5/20/60 sessions. Simultaneous annualized arithmetic active-mean intervals cover the 120-comparison family. At primary 20-session blocks, 0 have a positive lower bound. SR/CAGR-difference intervals and absolute SR intervals are pointwise and conditional on fits. The new continuous/QQQ controls are descriptive controls outside that own-model 120-contrast family. Displayed leader intervals do not adjust for their retrospective selection or earlier model/risk research. All 2022–2025 data are repeatedly inspected development history.

## Verification

Eleven meaningful unit tests passed across model signals, deterministic-noop inference and signed accounting. The independent auditor reconstructed all 4,824 signal flags, 24 calibration estimates and 111 target schedules and replayed 42 accounts using dollar-position accounting. Maximum NAV discrepancy was $0.000000011. Every account metric was remeasured and shared QQQ/baseline paths reproduce the earlier study. This verifies the declared calculations, not vendor or broker execution accuracy.

No 2026 data, neural retraining, trades, automations or prospective-policy changes were made. The current prospective long/cash candidate remains unchanged. The workbook retains its prior ten sheets and adds Model hedges, Hedge accounts and Hedge paths. The defensive table adds 108 base-cost nonbaseline accounts. Charts at the top of the master file explicitly separate successful continuous hedging from unsuccessful own-model timing.

Reproduce: `python -m hypershift run run_ndx_model_hedge --stage freeze`, then `--stage run`. The preserved original runner stops on deterministic no-op uncertainty; `python -m hypershift run finalize_ndx_model_hedge` completes the amended inference from its saved account paths. Audit with `python -m hypershift run audit_ndx_model_hedge`. Completed runs refuse overwrite.

[Protocol](PROTOCOL.json) · [Freeze](FREEZE.json) · [Amendment](STATISTICAL_AMENDMENT.json) · [Independent audit](AUDIT.json) · [All495accounts](metrics.csv) · [All120contrasts](uncertainty.csv)

Secondary descriptive paired comparisons of the continuous hedge versus the unhedged and QQQ-timed controls are in [control_uncertainty.csv](control_uncertainty.csv). These six post hoc contrasts have their own simultaneous intervals and do not extend the frozen own-model family.
