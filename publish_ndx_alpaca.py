"""Publish the complete independently audited NASDAQ-100 Alpaca experiment."""
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_ndx_alpaca_training import RUN, ROOT, FEATURE_ROOT, sha, write

DOCS = ROOT/'docs/nasdaq_alpaca_20261019'
LABELS = {'think':'THINK', 'think_mix':'THINK+mixer', 'think_huber':'THINK Huber',
          'think_mix_huber':'THINK+mixer Huber', 'think_mix_pseudohuber':'THINK+mixer pseudo-Huber',
          'mlp':'MLP', 'stockmixer':'StockMixer', 'ridge':'Ridge', 'monthly_basket':'Monthly basket',
          'SPY':'SPY', 'QQQ':'QQQ', 'momentum126':'Momentum 126', 'reversal5':'Reversal 5'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def pct(value):
    return f'{value*100:.1f}%'


def main():
    audit = json.loads((RUN/'AUDIT.json').read_text())
    unit_tests = json.loads((RUN/'UNIT_TESTS.json').read_text())
    if audit['status'] != 'passed':
        raise ValueError('Independent audit must pass before publication')
    plan = json.loads((RUN/'PROTOCOL.json').read_text())
    review = json.loads((FEATURE_ROOT/'REVIEW.json').read_text())
    raw = json.loads((ROOT/'data/nasdaq_monthly_pit/alpaca_15min/sip_raw_20180101_20260101/manifest.json').read_text())
    metrics = pd.read_csv(RUN/'metrics.csv', dtype={'seed':str})
    curves = pd.read_csv(RUN/'curves.csv', dtype={'seed':str})
    primary = metrics[(metrics.seed=='ensemble')&(metrics.k==10)&(metrics.cost_multiplier==1)].set_index(['view','model'])
    yearly = pd.read_csv(RUN/'annual.csv', dtype={'seed':str})
    monthly = pd.read_csv(RUN/'monthly.csv', dtype={'seed':str})
    intervals = pd.read_csv(RUN/'paired_intervals.csv')
    exposures = pd.read_csv(RUN/'exposure_diagnostics.csv').set_index(['view','model','benchmark'])
    folds = pd.read_csv(RUN/'folds.csv')
    coverage = pd.read_csv(FEATURE_ROOT/'coverage_by_security_year.csv')
    seeds = pd.read_csv(RUN/'seed_summary.csv').set_index(['view','model'])
    accuracy = pd.read_csv(RUN/'prediction_accuracy.csv')
    accuracy = accuracy[accuracy.forecast.isin(['mean_return_forecast_not_rank_portfolio_score','shared_ridge_return_forecast'])].set_index(['view','model'])
    DOCS.mkdir(parents=True, exist_ok=True)
    figures = DOCS/'figures'
    figures.mkdir(exist_ok=True)
    selected = curves[(curves.seed=='ensemble')&(curves.k==10)&(curves.cost_multiplier==1)]
    colors = dict(zip(plan['models']+['ridge'], plt.get_cmap('tab10').colors))
    upper = max(float(np.cumprod(1+p.net.to_numpy()).max()) for _,p in selected.groupby(['view','model']))*1.04
    for view in plan['views']:
        fig, ax = plt.subplots(figsize=(13,8))
        for kind in plan['models']+['ridge']:
            piece = selected[(selected.view==view)&(selected.model==kind)]
            metric = primary.loc[(view,kind)]
            ax.plot(pd.to_datetime(piece.date), np.cumprod(1+piece.net), color=colors[kind], linewidth=1.7,
                    label=f'{LABELS[kind]}: CAGR {pct(metric.annualized_return)}, SR {metric.sharpe:.3f}')
        for kind, color, style in [('SPY','black','--'),('QQQ','#333399','--'),('monthly_basket','#888888',':'),
                                   ('momentum126','#666633','-.'),('reversal5','#996666','-.')]:
            piece = selected[(selected.view=='control')&(selected.model==kind)]
            metric = primary.loc[('control',kind)]
            ax.plot(pd.to_datetime(piece.date), np.cumprod(1+piece.net), color=color, linestyle=style,
                    linewidth=2., label=f'{LABELS[kind]}: CAGR {pct(metric.annualized_return)}, SR {metric.sharpe:.3f}')
        ax.set(title=('Daily-only matched control' if view=='daily_control' else '15-minute-informed inputs')+
               ' — fixed equal top 10, 2022–2025', ylabel='Net wealth per initial $1', ylim=(0,upper))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='upper left')
        fig.tight_layout()
        fig.savefig(figures/f'{view}_wealth.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12,6))
    for j, kind in enumerate(plan['models']):
        daily = metrics[(metrics.view=='daily_control')&(metrics.model==kind)&(metrics.k==10)&
                        (metrics.cost_multiplier==1)&(metrics.seed!='ensemble')].set_index('seed')
        intraday = metrics[(metrics.view=='intraday_augmented')&(metrics.model==kind)&(metrics.k==10)&
                        (metrics.cost_multiplier==1)&(metrics.seed!='ensemble')].set_index('seed')
        changes = (intraday.annualized_return-daily.annualized_return)*100
        ax.scatter(changes, np.full(len(changes),j)+np.linspace(-.12,.12,len(changes)), color=colors[kind], s=35)
        difference = (primary.loc[('intraday_augmented',kind),'annualized_return']-
                      primary.loc[('daily_control',kind),'annualized_return'])*100
        ax.scatter([difference],[j],color='black',marker='D',s=45)
    ax.axvline(0,color='black',linewidth=1)
    ax.set(yticks=np.arange(len(plan['models'])), yticklabels=[LABELS[k] for k in plan['models']],
           xlabel='Intraday-informed minus daily-only CAGR (percentage points)',
           title='Matched-seed changes; black diamonds are rank-ensemble changes')
    ax.grid(axis='x',alpha=.2)
    fig.tight_layout(); fig.savefig(figures/'paired_seed_changes.png',dpi=160); plt.close(fig)
    subset = intervals[(intervals.block_sessions==20)&(intervals.comparison=='QQQ')].set_index('model')
    fig, ax = plt.subplots(figsize=(12,6))
    for j, kind in enumerate(plan['models']):
        row = subset.loc[kind]
        ax.plot([row.simultaneous_low*100,row.simultaneous_high*100],[j,j],color='#bbbbbb',linewidth=5)
        ax.plot([row.active_low*100,row.active_high*100],[j,j],color=colors[kind],linewidth=2)
        ax.scatter([row.active_mean_annual*100],[j],color=colors[kind],zorder=3)
    ax.axvline(0,color='black',linewidth=1)
    ax.set(yticks=np.arange(len(plan['models'])),yticklabels=[LABELS[k] for k in plan['models']],
           xlabel='Annualized arithmetic active return versus QQQ (percentage points)',
           title='Conditional 95% intervals: colored paired percentiles; gray simultaneous max-statistic diagnostic')
    ax.grid(axis='x',alpha=.2)
    fig.tight_layout(); fig.savefig(figures/'active_intervals_qqq.png',dpi=160); plt.close(fig)
    matched_rows, seed_rows, annual_rows, monthly_rows, cost_rows = [], [], [], [], []
    primary_monthly = monthly[(monthly.seed=='ensemble')&(monthly.k==10)&(monthly.cost_multiplier==1)]
    qqq_months = primary_monthly[(primary_monthly.view=='control')&(primary_monthly.model=='QQQ')].set_index('month').net_return
    spy_months = primary_monthly[(primary_monthly.view=='control')&(primary_monthly.model=='SPY')].set_index('month').net_return
    for kind in plan['models']+['ridge']:
        daily, intra = primary.loc[('daily_control',kind)], primary.loc[('intraday_augmented',kind)]
        mse_daily, mse_intra = accuracy.loc[('daily_control',kind),'mse_pp2'], accuracy.loc[('intraday_augmented',kind),'mse_pp2']
        matched_rows.append([LABELS[kind],pct(daily.annualized_return),pct(intra.annualized_return),
            f'{(intra.annualized_return-daily.annualized_return)*100:+.2f} pp',f'{intra.sharpe:.3f}',pct(intra.maximum_drawdown),
            f'{mse_daily:.3f} → {mse_intra:.3f}'])
        if kind in plan['models']:
            seed = seeds.loc[('intraday_augmented',kind)]
            seed_rows.append([LABELS[kind],pct(seed.minimum_cagr),pct(seed.median_cagr),pct(seed.maximum_cagr),
                              f'{int(seed.seeds_beating_qqq)}/{int(seed.seeds)}'])
        annual = yearly[(yearly.view=='intraday_augmented')&(yearly.model==kind)&(yearly.seed=='ensemble')&
                        (yearly.k==10)&(yearly.cost_multiplier==1)].set_index('year')
        annual_rows.append([LABELS[kind]]+[pct(annual.loc[y,'cumulative_return']) for y in plan['test_years']])
        months = primary_monthly[(primary_monthly.view=='intraday_augmented')&(primary_monthly.model==kind)].set_index('month').net_return
        monthly_rows.append([LABELS[kind],f'{int((months>qqq_months).sum())}/48',f'{int((months>spy_months).sum())}/48',
                             pct((months-qqq_months).std())])
        costs = metrics[(metrics.view=='intraday_augmented')&(metrics.model==kind)&(metrics.seed=='ensemble')&
                        (metrics.k==10)].set_index('cost_multiplier')
        cost_rows.append([LABELS[kind]]+[pct(costs.loc[m,'annualized_return']) for m in [1,2,4]]+
                         [f'{costs.loc[1,"turnover_annual"]:.1f}×'])
    for kind in ['monthly_basket','momentum126','reversal5','SPY','QQQ']:
        annual = yearly[(yearly.view=='control')&(yearly.model==kind)&(yearly.seed=='ensemble')&
                        (yearly.k==10)&(yearly.cost_multiplier==1)].set_index('year')
        annual_rows.append([LABELS[kind]]+[pct(annual.loc[y,'cumulative_return']) for y in plan['test_years']])
    exposure_rows = []
    risk_rows = []
    for kind in plan['models']:
        q = exposures.loc[('intraday_augmented',kind,'QQQ')]
        s = exposures.loc[('intraday_augmented',kind,'SPY')]
        d = exposures.loc[('daily_control',kind,'QQQ')]
        risk_rows.append([LABELS[kind],pct(d.tracking_error_annual),pct(q.tracking_error_annual),
                          f'{q.tracking_error_annual/d.tracking_error_annual:.2f}×'])
        exposure_rows.append([LABELS[kind],f'{q.beta:.2f}',f'{s.beta:.2f}',pct(q.tracking_error_annual),
            f'{q.alpha_annual*100:+.2f} pp',f'[{q.alpha_hac_low*100:+.2f}, {q.alpha_hac_high*100:+.2f}] pp'])
    ci_rows = []
    for row in intervals[(intervals.block_sessions==20)].itertuples(index=False):
        ci_rows.append([LABELS[row.model],row.comparison,f'{row.active_mean_annual*100:+.2f} pp',
            f'[{row.active_low*100:+.2f}, {row.active_high*100:+.2f}] pp',
            f'[{row.simultaneous_low*100:+.2f}, {row.simultaneous_high*100:+.2f}] pp',
            f'[{row.sharpe_low:+.3f}, {row.sharpe_high:+.3f}]'])
    qqq = primary.loc[('control','QQQ')]
    spy = primary.loc[('control','SPY')]
    winner = primary.loc['intraday_augmented'].loc[plan['models']].annualized_return.idxmax()
    best = primary.loc[('intraday_augmented',winner)]
    gains = sum(primary.loc[('intraday_augmented',k),'annualized_return']>
                primary.loc[('daily_control',k),'annualized_return'] for k in plan['models'])
    family_positive = intervals[(intervals.block_sessions==20)&(intervals.comparison.isin(['SPY','QQQ']))&
                               (intervals.simultaneous_low>0)]
    size_tables = []
    for view in plan['views']:
        rows = []
        for kind in plan['models']+['ridge']:
            sizes = metrics[(metrics.view==view)&(metrics.model==kind)&(metrics.seed=='ensemble')&
                            (metrics.cost_multiplier==1)].set_index('k')
            rows.append([LABELS[kind]]+[f'{pct(sizes.loc[k,"annualized_return"])} / '
                f'{sizes.loc[k,"sharpe"]:.3f} / {pct(sizes.loc[k,"maximum_drawdown"])}' for k in [5,10,20]])
        size_tables.extend([('Daily-only controls' if view=='daily_control' else 'Intraday-informed inputs'),'',
                            table(['Model','Top 5: CAGR / Sharpe / drawdown','Top 10','Top 20'],rows),''])
    lines = [
        '# NASDAQ-100: Alpaca 15-minute-informed model comparison',
        '',
        '**Complete: 280 neural fits, eight ridge fits, 241 continuous portfolios; independent audit passed.**',
        '',
        f'The download contains **{raw["rows"]:,} 15-minute SIP bars** in {raw["completed_chunks"]:,} '
        f'symbol/year chunks, covering 2018–2025. The regular-session review found '
        f'**{review["complete_member_session_fraction"]:.2%} complete sessions** across historical member/tradable days.',
        '',
        f'The identity review quarantined **{review["rejected_symbol_sessions"]:,} source sessions** outside verified '
        'issuer/ticker validity dates. In particular, META referred to a Roundhill ETF before Facebook adopted '
        'that ticker on June 9, 2022. FB and META histories are joined only within their verified date windows; '
        'the other rename chains have explicit boundaries too. Rejected records remain in the raw download '
        'and in rejected_symbol_sessions.csv, and never enter model features. '
        '[Meta SEC announcement](https://www.sec.gov/Archives/edgar/data/1326801/000132680122000070/may312022-exhibit991.htm) '
        'and [Roundhill issuer record](https://www.roundhillinvestments.com/etf/metv/) support this correction.',
        '',
        f'Intraday-informed inputs improve ensemble CAGR for **{gains}/7** neural variants relative to matched '
        f'daily controls. The highest observed primary neural intraday CAGR belongs to **{LABELS[winner]}**, '
        f'with {pct(best.annualized_return)} CAGR, {best.sharpe:.3f} Sharpe and '
        f'{pct(best.maximum_drawdown)} maximum drawdown. This is a descriptive leader on inspected dates, '
        'not an independently selected future strategy.',
        '',
        f'QQQ: {pct(qqq.annualized_return)} CAGR, {qqq.sharpe:.3f} Sharpe, '
        f'{pct(qqq.maximum_drawdown)} drawdown. SPY: {pct(spy.annualized_return)} CAGR, '
        f'{spy.sharpe:.3f} Sharpe, {pct(spy.maximum_drawdown)} drawdown.',
        '',
        f'{len(family_positive)}/14 augmented-versus-ETF contrasts have a positive lower bound in the '
        'declared 21-comparison, 20-session max-statistic diagnostic. These conditional intervals do not '
        'account for the prior research search or training uncertainty, and the inspected years remain exploratory.',
        '',
        '## What was tested',
        '',
        'Monthly historical NASDAQ-100 eligibility is unchanged: 100 companies can produce 101–104 '
        'included share classes per month, with 156 stable security IDs across the full panel. '
        'Models still forecast five-session returns '
        'and use next-close execution. This is a test of information derived from 15-minute bars; it is **not** '
        'a 15-minute trading strategy or a replacement of the reviewed daily economic return labels.',
        '',
        'Both input views use 13-feature models with identical starting parameters for matched model/seed pairs. '
        'Daily controls retain the five daily features and pad intraday channels. Augmented inputs add seven '
        'regular-session summaries plus observed coverage: session return/range, within-bar realized volatility, '
        'first/last-hour returns, last-hour volume share, and close-to-VWAP distance. Both views use 16-session '
        'windows, prior-year validation-MSE stopping, and five seeds. They are new capacity-matched controls, '
        'not exact reruns of the old five-feature checkpoints.',
        '',
        table(['Test year','Training: first signal → last label','Validation: first signal → last label','Train / validation / test origins'],
              [[int(r.test_year),r.train_first_signal+' → '+r.train_last_label,
                r.validation_first_signal+' → '+r.validation_last_label,
                f'{int(r.training_signal_dates)} / {int(r.validation_signal_dates)} / {int(r.test_forecasts)}']
               for r in folds.itertuples(index=False)]),
        '',
        'Earlier 2018 data supplies warmup. Outcomes crossing split boundaries are purged. '
        'There are 200 complete five-session test periods plus one final partial holding period, '
        'which is valued and liquidated but excluded from forecast-error/ranking labels.',
        '',
        'Every account starts with $1 million and carries cash and holdings continuously across model years. '
        'Base per-trade costs are 7 basis points plus 3 basis points times the square root of participation '
        'divided by 1%; normal orders are capped at 1% of lagged 20-session average dollar volume. '
        'The cost multipliers scale both terms. ETF controls buy at the same initial close and liquidate '
        'at the common terminal close with the same cost formula. No interest is credited to cash.',
        '',
        'Calendar opens/closes include early closes. A bar is usable only at its end. Incomplete sessions have '
        'missing numeric intraday features, imputed with training-only means; their observed coverage remains '
        'a feature. No historical member is removed for missing Alpaca data. Corporate-action settlement '
        'and portfolio returns retain the reviewed daily engine.',
        '',
        'Coverage is not uniform. The lowest completeness security/year slices are disclosed below; '
        'these securities retain the same daily eligibility and labels, with explicit missing intraday '
        'features on incomplete days. The missing bar inventory alone does not identify its market-data cause.',
        '',
        table(['Year','Security','Member/tradable sessions','Complete sessions','Complete fraction'],
              [[int(r.year),r.security_id,int(r.member_tradable_days),int(r.complete_days),pct(r.complete_fraction)]
               for r in coverage.nsmallest(8,'complete_fraction').itertuples(index=False)]),
        '',
        '## Primary matched comparison',
        '',
        table(['Model','Daily CAGR','Intraday CAGR','CAGR change','Intraday Sharpe','Intraday drawdown','Return-forecast MSE (pp²)'],matched_rows),
        '',
        'MSE uses the mean of the five neural return forecasts; Ridge uses its single shared return forecast. '
        'Neural portfolio scores instead average eligible-stock ranks. '
        'A smaller MSE therefore need not imply better portfolio returns. CAGR changes are differences of '
        'standalone CAGRs, unlike the arithmetic active-return means in the interval tables.',
        '',
        '![Daily matched controls](figures/daily_control_wealth.png)',
        '',
        '![Intraday-informed models](figures/intraday_augmented_wealth.png)',
        '',
        '## Annual and monthly consistency',
        '',
        table(['Intraday model / control','2022','2023','2024','2025'],annual_rows),
        '',
        'Annual rows are net calendar-year returns from the continuous accounts; accounts do not reset.',
        '',
        table(['Intraday model','Months above QQQ','Months above SPY','Monthly QQQ excess SD'],monthly_rows),
        '',
        '## Training-seed sensitivity',
        '',
        table(['Intraday model','Minimum CAGR','Median CAGR','Maximum CAGR','Seeds above QQQ'],seed_rows),
        '',
        'The five seeds share the same market history. Rank ensembling creates a different portfolio, so '
        'ensemble CAGR need not equal the average seed CAGR.',
        '',
        '![Matched-seed input changes](figures/paired_seed_changes.png)',
        '',
        '## Fixed size sensitivity',
        '',
        'Top five and top twenty are secondary fixed sensitivities. They do not change the trained forecasts. '
        'No best portfolio size is selected from this table. Cells show base-cost CAGR / Sharpe / maximum drawdown.',
        '',
        *size_tables,
        '## Costs and uncertainty',
        '',
        table(['Intraday model','1× cost CAGR','2× cost CAGR','4× cost CAGR','Base annual traded-notional/NAV'],cost_rows),
        '',
        table(['Intraday model','QQQ beta','SPY beta','QQQ tracking error','QQQ OLS alpha/year','Unadjusted 95% HAC alpha interval'],exposure_rows),
        '',
        'Exposure regressions are supplemental single-factor descriptions, with a fixed 3% annual hurdle '
        'and 20-lag normal HAC intervals. They do not control all sector/style exposures, training uncertainty '
        'or research selection, and they are not allocation signals or additional confirmation tests.',
        '',
        '![Active-return intervals](figures/active_intervals_qqq.png)',
        '',
        table(['Intraday model','Comparison','Active mean/year','Paired 95% interval','Simultaneous diagnostic','Sharpe-difference interval'],ci_rows),
        '',
        'Intervals use 10,000 paired circular block samples. The table uses 20-session blocks; every declared '
        '5/20/60-session result is retained in paired_intervals.csv. The simultaneous diagnostic applies '
        'one joint max-statistic family to seven augmented-minus-daily and fourteen augmented-minus-ETF '
        'active-mean contrasts. It is a fixed-path dependence approximation, not a correction for all earlier '
        'research, a training-refit interval, or a probability of future outperformance. Rolling or annual '
        'subsamples are dependent. More bars do not create more independent market years.',
        '',
        '## Independent verification and remaining limits',
        '',
        f'The consolidated acquisition, timing, masking, feature, inference, and accounting unit suite passed '
        f'**{unit_tests["tests"]} tests**.',
        '',
        f'The audit reproduced {audit["checkpoints_reproduced"]} checkpoints and all validation/test exports, '
        f'refitted eight ridge controls, independently reconstructed ranking ensembles and metrics, and replayed '
        f'{audit["independent_accounts"]} continuous accounts. Maximum NAV discrepancy was '
        f'${audit["largest_nav_difference_dollars"]:.8f}. It rehashed all raw chunks and independently '
        f'recomputed {audit["independent_intraday_session_samples"]} sampled regular-session summaries.',
        '',
        f'The maximum independently recomputed ranking-metric difference was '
        f'{audit["maximum_ranking_metric_difference"]:.2e}. Pearson IC uses a 1e-8 absolute audit tolerance '
        'because the production calculation centers float32 targets while independent NumPy correlation '
        'promotes them to float64; RankIC and NDCG retain the tighter 1e-10 tolerance.',
        '',
        'This verifies the specified mathematics and timing. Historical ticker identities and corporate actions '
        'retain the prior research-grade limitations; every vendor row is not independently certified. Adjusted '
        'daily accounts approximate total returns and fills. New-feature missingness, ticker-specific mixer '
        'cold starts, concentration, execution assumptions, seed sensitivity and already-inspected 2022–2025 '
        'dates limit investment conclusions. Vendor revisions, as-published bar vintages and real-time '
        'feed latency are not independently reconstructed. No adaptive feature/K/seed search or trading orders were performed.',
        '',
        '## What this adds, and the next improvements',
        '',
        'The project now has consolidated intraday inputs, date-bounded ticker identities, matched-capacity '
        'daily controls, a shared ridge baseline, checkpoint-consistent validation exports, complete seed/cost '
        'disclosure and independent account replay. These strengthen the research process even when an '
        'augmented model fails to improve returns.',
        '',
        'The main model gaps remain short effective time history, dense ticker-specific mixer cold starts, '
        'raw-return MSE versus a benchmark-relative ranking objective, and abrupt concentrated top-K orders. '
        'The new intraday inputs summarize sessions; they do not model the full sequence of 15-minute bars '
        'or identify which bar patterns are predictive.',
        '',
        table(['Model','Daily QQQ tracking error','Intraday QQQ tracking error','Intraday / daily'],risk_rows),
        '',
        'Tracking error measures benchmark-relative volatility, not skill. At unchanged long-run dependence '
        'and history length it is a rough driver of mean-active-return interval width. The ratios above '
        'are descriptive; they are not confidence-interval ratios or promotion criteria.',
        '',
        '1. Freeze a small candidate family and the complete refit/allocation rule before evaluating genuinely '
        'untouched dates. Use nested chronological development and confirmation, including every attempt in '
        'the research ledger. More 15-minute rows do not replace independent years or erase prior inspection.',
        '2. In one declared ablation, compare the current mixer with a shared permutation-equivariant '
        'aggregator or a residual gated off for securities without training support. Report newly admitted '
        'member cohorts separately. Compare benchmark-residual targets and one ranking-aware objective '
        'against the current MSE criterion, retaining simple regularized baselines.',
        '3. Test a fixed issuer/sector-aware active-risk budget, turnover penalty and retention buffer through '
        'full-account simulation. Reduce uncompensated benchmark-relative risk without selecting the most '
        'attractive backtested K. Scaling the active sleeve alone shrinks both its return and volatility and '
        'does not create statistical evidence of an edge.',
        '4. Extend reliable point-in-time history and independently check corporate actions and execution. '
        'Retain paired dependence-aware inference, separate training uncertainty from market-path uncertainty, '
        'and declare the benchmark, economic hurdle and evaluation schedule before confirmation. For a '
        'fixed stationary process, roughly four times the effective history halves standard-error width; '
        'additional resampling iterations merely reduce simulation noise.',
        '',
        '## Complete artifacts',
        '',
        '[Protocol](PROTOCOL.json) · [Audit](AUDIT.json) · [All metrics](metrics.csv) · '
        '[Every daily path](curves.csv) · [Annual returns](annual.csv) · [Monthly returns](monthly.csv) · '
        '[Seed summaries](seed_summary.csv) · [Ranking](ranking.csv) · [Forecast accuracy](prediction_accuracy.csv) · '
        '[Exposure diagnostics](exposure_diagnostics.csv) · '
        '[Exact folds](folds.csv) · '
        '[All paired intervals](paired_intervals.csv) · [Leave-one-year-out diagnostic](leave_one_year_out.csv) · '
        '[Input review](INPUT_REVIEW.json) · [Coverage by security/year](coverage_by_security_year.csv) · '
        '[Identity rules and sources](alpaca_symbol_identity_20261019.json) · [Quarantined sessions](rejected_symbol_sessions.csv)',
        '',
        'Leave-one-year-out statistics omit fixed saved daily returns for influence diagnosis; they are not '
        'counterfactual accounts retraded with that year absent. Full checkpoints, forecasts, preprocessing, '
        'histories and integrity locks remain in runs/nasdaq100_alpaca_20261019.',
        ''
    ]
    (DOCS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    names = ['PROTOCOL.json','AUDIT.json','AUDIT_CODE_LOCK.json','EVALUATION_CODE_LOCK.json','UNIT_TESTS.json','metrics.csv','curves.csv','annual.csv','monthly.csv',
             'seed_summary.csv','ranking.csv','prediction_accuracy.csv','paired_intervals.csv',
             'leave_one_year_out.csv','training_records.csv','independent_intraday_samples.csv',
             'independent_ranking_precision.csv','exposure_diagnostics.csv','folds.csv']
    for name in names:
        shutil.copyfile(RUN/name,DOCS/name)
    shutil.copyfile(FEATURE_ROOT/'REVIEW.json',DOCS/'INPUT_REVIEW.json')
    shutil.copyfile(FEATURE_ROOT/'coverage_by_security_year.csv',DOCS/'coverage_by_security_year.csv')
    shutil.copyfile(FEATURE_ROOT/'rejected_symbol_sessions.csv',DOCS/'rejected_symbol_sessions.csv')
    shutil.copyfile(ROOT/'config/alpaca_symbol_identity_20261019.json',DOCS/'alpaca_symbol_identity_20261019.json')
    snapshot = RUN/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ['fetch_alpaca_15min.py','fetch_alpaca_batch.py','prepare_ndx_alpaca.py',
                 'report_ndx_alpaca.py','audit_ndx_alpaca.py','audit_historical_ndx.py',
                 'audit_historical_ndx_ranking.py','publish_ndx_alpaca.py',
                 'run_ndx_alpaca_pipeline.py','test_alpaca_15min.py','test_ndx_alpaca.py',
                 'test_ndx_alpaca_reporting.py','requirements.txt','requirements-data.txt',
                 'requirements-cuda.txt','requirements-tested.txt']:
        (snapshot/name).write_bytes((ROOT/name).read_bytes())
    write(RUN/'FINAL_PUBLICATION.json',dict(status='completed_audited_exploratory_study',neural_fits=280,
        continuous_paths=241,raw_bars=raw['rows'],report=str(DOCS/'REPORT.md'),
        intraday_models_with_higher_cagr_than_matched_daily=int(gains),
        descriptive_primary_leader=winner,publish_code_sha256=sha(Path(__file__)),
        artifact_hashes={str(p.relative_to(DOCS)):sha(p) for p in DOCS.rglob('*') if p.is_file()}))
    print(json.dumps(json.loads((RUN/'FINAL_PUBLICATION.json').read_text()),indent=2),flush=True)


if __name__=='__main__':
    main()
