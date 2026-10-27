"""Evaluate every frozen fit and policy in the declared Alpaca comparison."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ndx_alpaca_models import inputs, scale
from pit_research_engine import load_data, fold, continuous_backtest, etf_backtest
from research_engine import performance, ranking_metrics
from run_robust_research import ensemble_predictions, model_data
from run_ndx_alpaca_training import RUN, FEATURE_ROOT, feature_data, verify, sha, write


def check_complete():
    verify()
    tests = json.loads((RUN/'UNIT_TESTS.json').read_text())
    if tests['status']!='passed':
        raise ValueError('Consolidated unit tests must pass before evaluation')
    for name, expected in tests['source_hashes'].items():
        if sha(Path(__file__).resolve().parent/name)!=expected:
            raise ValueError('Unit-test source changed after verification')
    plan = json.loads((RUN/'PROTOCOL.json').read_text())
    lock = json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())
    if lock['neural_fits'] != plan['planned_neural_fits']:
        raise ValueError('Financial evaluation requires all declared fits')
    for name, expected in lock['hashes'].items():
        if sha(RUN/name) != expected:
            raise ValueError('Fit artifact changed')
    for name, expected in json.loads((RUN/'INTRADAY_DATA_LOCK.json').read_text()).items():
        if sha(name) != expected:
            raise ValueError('Intraday input changed')
    return plan


def load_scores(data, plan):
    ids = np.concatenate([fold(data, year)[2] for year in plan['test_years']])
    split_rows = []
    for year in plan['test_years']:
        train, val, test = fold(data,year)
        dates = data['panel'].dates
        split_rows.append(dict(test_year=year,training_signal_dates=len(train),validation_signal_dates=len(val),
            test_forecasts=len(test),train_first_signal=str(dates[data['origin'][train[0]]].date()),
            train_last_label=str(dates[data['exit'][train[-1]]].date()),
            validation_first_signal=str(dates[data['origin'][val[0]]].date()),
            validation_last_label=str(dates[data['exit'][val[-1]]].date()),
            first_test_fill=str(dates[data['entry'][test[0]]].date()),
            unobserved_training_targets=int((data['mask'][train]&~data['labelmask'][train]).sum()),
            unobserved_validation_targets=int((data['mask'][val]&~data['labelmask'][val]).sum())))
    pd.DataFrame(split_rows).to_csv(RUN/'folds.csv',index=False)
    forecasts, records, accuracies = {}, [], []
    feature = feature_data()[0]
    for view in plan['views']:
        for kind in plan['models']:
            seeds = []
            for seed in plan['seeds']:
                forecasts_by_year = []
                for year in plan['test_years']:
                    folder = RUN/view/str(year)
                    with np.load(folder/f'{kind}_{seed}_predictions.npz') as saved:
                        forecasts_by_year.append(saved['test'].copy())
                    records.append(json.loads((folder/f'{kind}_{seed}_record.json').read_text()))
                prediction = np.concatenate(forecasts_by_year)
                seeds.append(prediction)
                error = (prediction-data['y'][ids]*100)[data['labelmask'][ids]]
                accuracies.append(dict(view=view, model=kind, forecast=str(seed),
                    mse_pp2=float(np.mean(error**2)), mae_pp=float(np.mean(np.abs(error))), labels=len(error)))
            forecasts[view, kind] = seeds
            error = (np.mean(seeds, axis=0)-data['y'][ids]*100)[data['labelmask'][ids]]
            accuracies.append(dict(view=view, model=kind, forecast='mean_return_forecast_not_rank_portfolio_score',
                mse_pp2=float(np.mean(error**2)), mae_pp=float(np.mean(np.abs(error))), labels=len(error)))
        raw = inputs(data, view, feature if view=='intraday_augmented' else None)
        ridge_test = []
        for year in plan['test_years']:
            train, val, test = fold(data, year)
            x, _, _ = scale(raw, data['mask'], train)
            design = x.transpose(0, 2, 1, 3).reshape(len(x), len(data['panel'].tickers), -1)
            design = np.concatenate([design, np.ones((*design.shape[:2], 1))], axis=-1)
            a = design[train][data['labelmask'][train]]
            y = (data['y'][train]*100)[data['labelmask'][train]]
            penalty = np.eye(a.shape[-1])*10.
            penalty[-1, -1] = 0.
            coefficients = np.linalg.solve(a.T@a+penalty, a.T@y)
            prediction = design[test]@coefficients
            folder = RUN/'ridge'/view
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/f'{year}.npz', coefficients=coefficients, test=prediction,
                                 validation=design[val]@coefficients)
            ridge_test.append(prediction)
        forecasts[view, 'ridge'] = [np.concatenate(ridge_test)]
        error = (forecasts[view,'ridge'][0]-data['y'][ids]*100)[data['labelmask'][ids]]
        accuracies.append(dict(view=view,model='ridge',forecast='shared_ridge_return_forecast',
            mse_pp2=float(np.mean(error**2)),mae_pp=float(np.mean(np.abs(error))),labels=len(error)))
    for kind, field in [('momentum126', 'momentum'), ('reversal5', 'reversal')]:
        forecasts['control', kind] = [data[field][ids]]
    pd.DataFrame(records).to_csv(RUN/'training_records.csv', index=False)
    pd.DataFrame(accuracies).to_csv(RUN/'prediction_accuracy.csv', index=False)
    return ids, forecasts


def paired_intervals(primary, plan, repeats=10000):
    contrasts = []
    for kind in plan['models']:
        contrasts.append((kind, 'intraday_minus_daily', ('intraday_augmented', kind), ('daily_control', kind)))
        for benchmark in ['SPY', 'QQQ']:
            contrasts.append((kind, benchmark, ('intraday_augmented', kind), ('control', benchmark)))
    a = np.column_stack([primary[left] for _, _, left, _ in contrasts])
    b = np.column_stack([primary[right] for _, _, _, right in contrasts])
    n = len(a)
    observed = (a-b).mean(0)*252
    hurdle = np.expm1(np.log1p(.03)/252)
    sharpe_observed = np.sqrt(252)*((a.mean(0)-hurdle)/a.std(0, ddof=1)-(b.mean(0)-hurdle)/b.std(0, ddof=1))
    rows = []
    for block in [5, 20, 60]:
        rng = np.random.default_rng(20261019+block)
        means, sharpes = [], []
        for start in range(0, repeats, 100):
            size = min(100, repeats-start)
            starts = rng.integers(0, n, (size, int(np.ceil(n/block))))
            resample = ((starts[..., None]+np.arange(block))%n).reshape(size, -1)[:, :n]
            aa, bb = a[resample], b[resample]
            means.append((aa-bb).mean(1)*252)
            sharpes.append(np.sqrt(252)*((aa.mean(1)-hurdle)/aa.std(1, ddof=1)-
                                        (bb.mean(1)-hurdle)/bb.std(1, ddof=1)))
        means, sharpes = np.concatenate(means), np.concatenate(sharpes)
        standard_error = np.maximum(means.std(0, ddof=1), 1e-12)
        critical = float(np.quantile(np.max(np.abs((means-observed)/standard_error), axis=1), .95))
        for j, (kind, comparison, _, _) in enumerate(contrasts):
            rows.append(dict(model=kind, comparison=comparison, block_sessions=block, replicates=repeats,
                active_mean_annual=observed[j], active_low=np.quantile(means[:, j], .025),
                active_high=np.quantile(means[:, j], .975),
                simultaneous_low=observed[j]-critical*standard_error[j],
                simultaneous_high=observed[j]+critical*standard_error[j],
                sharpe_difference=sharpe_observed[j], sharpe_low=np.quantile(sharpes[:, j], .025),
                sharpe_high=np.quantile(sharpes[:, j], .975), family_comparisons=len(contrasts),
                max_statistic_critical=critical,
                warning='Fixed-path exploratory inference; no training/research-search uncertainty; stationary circular-block approximation.'))
    frame = pd.DataFrame(rows)
    frame.to_csv(RUN/'paired_intervals.csv', index=False)
    return frame


def exposure_diagnostics(primary):
    rows = []
    hurdle = np.expm1(np.log1p(.03)/252)
    for (view, kind), returns in primary.items():
        if kind in ['SPY','QQQ']:
            continue
        for benchmark in ['SPY','QQQ']:
            control = primary['control',benchmark]
            x = np.column_stack([np.ones(len(control)),control-hurdle])
            y = returns-hurdle
            coefficients = np.linalg.lstsq(x,y,rcond=None)[0]
            residual = y-x@coefficients
            score = x*residual[:,None]
            meat = score.T@score
            for lag in range(1,21):
                cross = score[lag:].T@score[:-lag]
                meat += (1-lag/21)*(cross+cross.T)
            bread = np.linalg.inv(x.T@x)
            covariance = bread@meat@bread*len(x)/(len(x)-2)
            se = float(np.sqrt(max(covariance[0,0],0.))*252)
            alpha = float(coefficients[0]*252)
            rows.append(dict(view=view,model=kind,benchmark=benchmark,beta=float(coefficients[1]),
                alpha_annual=alpha,alpha_hac_low=alpha-1.96*se,alpha_hac_high=alpha+1.96*se,
                tracking_error_annual=float(np.std(returns-control,ddof=1)*np.sqrt(252)),hac_lags=20,
                warning='Supplemental single-factor descriptive fit; fixed 3% hurdle; unadjusted normal HAC interval; no selection or full factor adjustment.'))
    pd.DataFrame(rows).to_csv(RUN/'exposure_diagnostics.csv',index=False)


def evaluate():
    plan = check_complete()
    if (RUN/'completion.json').exists():
        raise ValueError('Completed evaluation is immutable; use saved outputs')
    write(RUN/'EVALUATION_CODE_LOCK.json', {str(Path(__file__).resolve()): sha(Path(__file__))})
    data = load_data()
    ids, forecasts = load_scores(data, plan)
    metrics, curves, ledgers, ranking, rank_periods, score_archive, primary = [], [], [], [], [], {}, {}
    def record(view, kind, seed, k, mult, metric, daily, account=None):
        tags = dict(view=view, model=kind, seed=str(seed), k=k, cost_multiplier=mult)
        metrics.append(dict(**tags, **metric))
        curves.append(daily.assign(**tags))
        if account is not None and seed=='ensemble' and k==10 and mult==1:
            ledgers.append(account.assign(**tags))
        if seed=='ensemble' and k==10 and mult==1:
            primary[view, kind] = daily.net.to_numpy()
    for (view, kind), available in forecasts.items():
        adjusted = model_data(data, kind)
        score = ensemble_predictions(available, adjusted['mask'][ids]) if len(available)>1 else available[0]
        score_archive[view+'__'+kind] = score
        summary, periods = ranking_metrics(score, adjusted['y'][ids], adjusted['mask'][ids], adjusted['labelmask'][ids], [5, 10, 20])
        ranking.append(dict(view=view, model=kind, **summary))
        periods['entry_date'] = data['panel'].dates[data['entry'][ids]].strftime('%Y-%m-%d')
        rank_periods.append(periods.assign(view=view, model=kind))
        for k in plan['sensitivities']['k']:
            for mult in plan['sensitivities']['cost_multipliers']:
                metric, daily, account = continuous_backtest(adjusted, ids, score, k, mult)
                record(view, kind, 'ensemble', k, mult, metric, daily, account)
        if len(available)>1:
            for seed, single in zip(plan['seeds'], available):
                metric, daily, account = continuous_backtest(adjusted, ids, single, 10, 1.)
                record(view, kind, seed, 10, 1., metric, daily, account)
        print(f'Evaluated all declared portfolios: {view} {kind}', flush=True)
    for mult in plan['sensitivities']['cost_multipliers']:
        metric, daily, account = continuous_backtest(data, ids, np.zeros_like(data['y'][ids]), 10, mult, basket=True)
        record('control', 'monthly_basket', 'ensemble', 10, mult, metric, daily, account)
        for kind in ['SPY', 'QQQ']:
            metric, daily = etf_backtest(data, kind, mult)
            record('control', kind, 'ensemble', 10, mult, metric, daily)
    metric_frame = pd.DataFrame(metrics)
    curve_frame = pd.concat(curves, ignore_index=True)
    metric_frame.to_csv(RUN/'metrics.csv', index=False)
    curve_frame.to_csv(RUN/'curves.csv', index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(RUN/'primary_account_ledger.csv', index=False)
    pd.DataFrame(ranking).to_csv(RUN/'ranking.csv', index=False)
    pd.concat(rank_periods, ignore_index=True).to_csv(RUN/'ranking_periods.csv', index=False)
    np.savez_compressed(RUN/'portfolio_scores.npz', ids=ids, **score_archive)
    annual, monthly, leave_year = [], [], []
    columns = ['view', 'model', 'seed', 'k', 'cost_multiplier']
    for key, piece in curve_frame.groupby(columns, sort=False):
        tags = dict(zip(columns, key))
        piece = piece.assign(year=pd.to_datetime(piece.date).dt.year, month=pd.to_datetime(piece.date).dt.to_period('M').astype(str))
        for year, subset in piece.groupby('year'):
            annual.append(dict(**tags, year=int(year), **performance(subset.net)))
        for month, subset in piece.groupby('month'):
            monthly.append(dict(**tags, month=month, net_return=float(np.prod(1+subset.net)-1)))
        if tags['seed']=='ensemble' and tags['k']==10 and tags['cost_multiplier']==1:
            for year in plan['test_years']:
                leave_year.append(dict(**tags, omitted_year=year, **performance(piece.loc[piece.year!=year, 'net'])))
    pd.DataFrame(annual).to_csv(RUN/'annual.csv', index=False)
    pd.DataFrame(monthly).to_csv(RUN/'monthly.csv', index=False)
    pd.DataFrame(leave_year).to_csv(RUN/'leave_one_year_out.csv', index=False)
    seeds = []
    qqq = float(metric_frame[(metric_frame.model=='QQQ')&(metric_frame.cost_multiplier==1)].annualized_return.iloc[0])
    for (view, kind), group in metric_frame[(metric_frame.seed!='ensemble')].groupby(['view', 'model']):
        seeds.append(dict(view=view, model=kind, minimum_cagr=group.annualized_return.min(),
            median_cagr=group.annualized_return.median(), maximum_cagr=group.annualized_return.max(),
            seeds_beating_qqq=int((group.annualized_return>qqq).sum()), seeds=len(group)))
    pd.DataFrame(seeds).to_csv(RUN/'seed_summary.csv', index=False)
    paired_intervals(primary, plan)
    exposure_diagnostics(primary)
    if len(metrics) != 241:
        raise ValueError('Unexpected portfolio inventory')
    write(RUN/'completion.json', dict(status='completed_exploratory_evaluation_audit_pending',
        neural_fits=280, continuous_paths=len(metrics), annual_resets=0,
        first_date='2022-01-03', last_date='2025-12-31',
        primary='Fixed equal top 10; all five seeds; no test-selected model or K',
        interpretation='Intraday-informed five-session forecasts; not a 15-minute trading backtest.'))
    print('All 241 accounts evaluated. Independent audit required before publication.', flush=True)


if __name__ == '__main__':
    evaluate()
