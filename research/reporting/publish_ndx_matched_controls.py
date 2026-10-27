"""Publish audited matched benchmarks and the inactive prospective policy."""
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

import run_ndx_matched_controls as study

LABELS = {
    study.PRIMARY: 'Neural: five issuers + 20% target',
    study.SECONDARY: 'Neural: five issuers + 25% target + trend',
    'neural/ten_seed/equal5/full': 'Neural: prior full-exposure blend',
    'neural/ten_seed/equal5/vol20': 'Neural: original selections + 20% target',
    'qqq_own/full': 'QQQ: weekly full exposure',
    'qqq_own/vol20': 'QQQ: own 20% target',
    'qqq_own/vol25trend': 'QQQ: own 25% target + trend',
    'qqq_replay/ten_seed/issuer5/vol20': 'QQQ: neural 20% exposure targets',
    'qqq_replay/ten_seed/issuer5/vol25trend': 'QQQ: neural 25% + trend exposure targets',
    'momentum/issuer5/full': '12-minus-1 momentum: full exposure',
    'momentum/issuer5/vol20': '12-minus-1 momentum: 20% target',
    'momentum/issuer5/vol25trend': '12-minus-1 momentum: 25% target + trend',
    'buy_hold/SPY': 'SPY: buy and hold', 'buy_hold/QQQ': 'QQQ: buy and hold'}


def plot(metrics, curves):
    colors = ['#086A9F','#B36A16','#087D85','#639C2F','#C44832','#7544A0']
    groups = [[study.PRIMARY,'qqq_own/vol20','qqq_replay/ten_seed/issuer5/vol20',
               'momentum/issuer5/vol20','buy_hold/SPY','buy_hold/QQQ'],
              [study.SECONDARY,'qqq_own/vol25trend','qqq_replay/ten_seed/issuer5/vol25trend',
               'momentum/issuer5/vol25trend','buy_hold/SPY','buy_hold/QQQ']]
    fig = plt.figure(figsize=(19,11))
    grid = fig.add_gridspec(3,2,height_ratios=[3,1.65,1.65],hspace=.20,wspace=.14)
    axes = [[fig.add_subplot(grid[row,col]) for col in range(2)] for row in range(2)]
    dates = pd.DatetimeIndex(pd.to_datetime(curves.date)).insert(0,pd.Timestamp('2021-12-31'))
    limits = []
    for col,keys in enumerate(groups):
        for j,key in enumerate(keys):
            row = metrics.loc[key]; wealth = 100*np.r_[1,np.cumprod(1+curves[key].to_numpy())]
            limits.extend([wealth.min(),wealth.max()])
            style = dict(color=colors[j],lw=2.4 if j==0 else 1.7,ls='--' if j>=4 else '-')
            axes[0][col].plot(dates,wealth,label=LABELS[key]+f'\nSR {row.sharpe:.3f} | CAGR {row.annualized_return:.2%}',**style)
            axes[1][col].plot(dates,wealth/np.maximum.accumulate(wealth)-1,**style)
        axes[0][col].set_title('Primary: 20% volatility target' if col==0 else 'Secondary: 25% target + trend',weight='bold',fontsize=13)
        axes[0][col].tick_params(labelbottom=False)
        axes[1][col].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        for ax in [axes[0][col],axes[1][col]]:
            ax.grid(alpha=.18); ax.set_xlim(dates[0],dates[-1]); ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        legend_ax = fig.add_subplot(grid[2,col]); legend_ax.axis('off')
        handles,labels = axes[0][col].get_legend_handles_labels()
        legend = legend_ax.legend(handles,labels,loc='upper left',ncol=2,fontsize=9.4,frameon=False,labelspacing=1.1)
        for i,text in enumerate(legend.get_texts()):
            if i in [0,4,5]:text.set_weight('bold')
    for ax in axes[0]:ax.set_ylim(min(limits)*.95,max(limits)*1.05)
    bottom = min(ax.get_ylim()[0] for ax in axes[1])
    for ax in axes[1]:ax.set_ylim(bottom,.01)
    axes[0][0].set_ylabel('Growth of $100 after modeled costs')
    axes[1][0].set_ylabel('Drawdown from previous peak')
    fig.suptitle('Does stock selection add to exposure management? Matched controls',fontsize=18,weight='bold',y=.97)
    fig.text(.5,.022,'2022-2025 exploratory history | All ten neural seeds | Same five-session decisions / regular-close execution | Cash earns zero | No new model fits',ha='center',fontsize=10,color='#555555')
    fig.subplots_adjust(left=.065,right=.985,top=.91,bottom=.07)
    fig.savefig(study.DOC/'matched_controls.png',dpi=160); plt.close(fig)


def main():
    study.verify_plan()
    complete = json.loads((study.RUN/'COMPLETE.json').read_text())
    audit = json.loads((study.RUN/'AUDIT.json').read_text())
    if audit['status'] != 'passed':raise ValueError('Audit not passed')
    for name,h in complete['output_sha256'].items():
        if study.parent.sha(study.RUN/name)!=h:raise ValueError('Output changed after completion')
    study.DOC.mkdir(parents=True,exist_ok=True)
    for name in ['PROTOCOL.json','FREEZE.json','COMPLETE.json','AUDIT.json','metrics.csv','annual.csv',
                 'uncertainty.csv','comparison_curves.csv','momentum_support.csv','target_exposures.csv']:
        shutil.copyfile(study.RUN/name,study.DOC/name)
    shutil.copyfile(study.POLICY,study.DOC/'PROSPECTIVE_POLICY.json')
    metrics = pd.read_csv(study.RUN/'metrics.csv')
    base = metrics[metrics.cost_multiplier==1].set_index('strategy')
    curves = pd.read_csv(study.RUN/'comparison_curves.csv')
    uncertainty = pd.read_csv(study.RUN/'uncertainty.csv')
    support = pd.read_csv(study.RUN/'momentum_support.csv')
    plot(base,curves)
    policy = json.loads(study.POLICY.read_text())
    readiness = dict(status='policy_frozen_activation_prerequisites_pending', activated=False,
                     frozen_timestamp_utc=policy['frozen_timestamp_utc'], policy_sha256=study.parent.sha(study.POLICY),
                     candidate=study.PRIMARY, candidate_changed_after_controls=False,
                     historical_daily_data_last_date=str(curves.date.iloc[-1]), prospective_predictions=0,
                     prerequisites=[dict(requirement=requirement,verified=False) for requirement in policy['activation_requirements']],
                     old_confirmation_gate_unchanged=True,
                     explanation='Local reviewed inputs and trained exports used here stop in 2025. A future-date data/inference/timestamp review and actual activation must precede a genuinely prospective signal. No 2026 observations were inspected by this study.')
    study.parent.write(study.DOC/'READINESS.json',readiness)
    headline = [study.PRIMARY,study.SECONDARY,'neural/ten_seed/equal5/full',
                'qqq_own/full','qqq_own/vol20','qqq_own/vol25trend',
                'qqq_replay/ten_seed/issuer5/vol20','qqq_replay/ten_seed/issuer5/vol25trend',
                'momentum/issuer5/full','momentum/issuer5/vol20','momentum/issuer5/vol25trend','buy_hold/SPY','buy_hold/QQQ']
    primary = base.loc[study.PRIMARY]
    qqq = base.loc['qqq_own/vol20']; momentum = base.loc['momentum/issuer5/vol20']
    lines = ['# Matched benchmark controls and prospective candidate freeze','',
             'October 27, 2026. Same 1,003 exchange sessions in 2022–2025; repeatedly inspected development history. Exactly 29 predeclared strategies and 87 cost accounts. No network retrained.','',
             '![Matched neural, QQQ and momentum returns with SR/CAGR legends and drawdown panels](matched_controls.png)','',
             f'The fixed primary ten-seed candidate has **{primary.sharpe:.3f} Sharpe / {primary.annualized_return:.2%} CAGR / {primary.maximum_drawdown:.2%} drawdown**. QQQ with its own 20% volatility target has **{qqq.sharpe:.3f} / {qqq.annualized_return:.2%} / {qqq.maximum_drawdown:.2%}**; the matched momentum control has **{momentum.sharpe:.3f} / {momentum.annualized_return:.2%} / {momentum.maximum_drawdown:.2%}**. These are historical comparisons, not confirmation.','',
             '| Strategy | Sharpe | Net CAGR | Max drawdown | Mean stock exposure |',
             '| --- | ---: | ---: | ---: | ---: |']
    for key in headline:
        row = base.loc[key]; exposure = row.get('average_equity_exposure',np.nan)
        lines.append(f'| {LABELS[key]} | {row.sharpe:.3f} | {row.annualized_return:.2%} | {row.maximum_drawdown:.2%} | '+(f'{exposure:.1%}' if np.isfinite(exposure) else 'Buy and hold')+' |')
    lines += ['', 'All neural rows above use all ten seeds. Original-five and new-five accounts remain in the full metrics file; they are diagnostics, not independent market replications. Exposure averages exclude the final liquidation session.','',
              '## What each comparison isolates','',
              '- QQQ own-risk controls use QQQ’s trailing 20-session volatility under the same target/cap and five-session execution calendar. This tests a simple passive exposure-management alternative.',
              '- QQQ replay controls apply the neural basket-derived target exposure exactly. This holds intended stock/cash timing fixed while changing the assets. Realized exposure drift, beta, correlations and costs still differ, so it is not a proof of factor-adjusted alpha.',
              '- Momentum uses the prior signal close’s price 21 sessions earlier divided by its price 252 sessions earlier, minus one. It selects five issuers with the same tie/share-class handling and applies each rule to its own selected basket. No lookback or weighting search was conducted.',
              '- The weekly full-exposure QQQ control uses lagged fixed-share sizing like the model. Buy-and-hold SPY/QQQ anchors retain the parent’s first-close available-dollar investment. Both use the same first fill date, costs and terminal liquidation; their sizing conventions are explicitly different.', '',
              f'Momentum lacked endpoint history for a mean of **{support.missing_history.mean():.1f}** otherwise neural-eligible securities per decision (maximum **{support.missing_history.max()}**). Those names are excluded only from momentum; the neural universe is unchanged. [Per-date support](momentum_support.csv).','',
              '## Paired uncertainty: primary 20-session blocks','',
              'Eight comparisons were declared before control results. The active-mean intervals below are simultaneous across those eight comparisons. Sharpe-difference intervals are pointwise. Neither adjusts the extensive earlier model/risk-rule search, retraining uncertainty, or input-vintage uncertainty. Active mean is an annualized arithmetic daily return difference, not a CAGR difference.','',
              '| Candidate | Comparator | Annual active mean | Simultaneous 95% active-mean interval | Paired 95% SR-difference interval |',
              '| --- | --- | ---: | --- | --- |']
    for row in uncertainty[uncertainty.block_sessions==20].itertuples():
        lines.append(f'| {"Primary" if row.strategy==study.PRIMARY else "Secondary"} | {LABELS[row.comparator]} | {row.annual_active_mean:.2%} | [{row.simultaneous_active_low:.2%}, {row.simultaneous_active_high:.2%}] | [{row.paired_sr_low:.3f}, {row.paired_sr_high:.3f}] |')
    lines += ['', 'Positive point estimates do not establish superiority when the corresponding interval includes zero. Five/60-session block sensitivity and paired CAGR intervals are retained in [uncertainty.csv](uncertainty.csv).','',
              '## Trading-cost stress','',
              '| Strategy | Cost multiple | Sharpe | Net CAGR | Max drawdown |',
              '| --- | ---: | ---: | ---: | ---: |']
    for key in [study.PRIMARY,'qqq_own/vol20','qqq_replay/ten_seed/issuer5/vol20','momentum/issuer5/vol20']:
        for row in metrics[(metrics.strategy==key)].itertuples():
            lines.append(f'| {LABELS[key]} | {row.cost_multiplier}x | {row.sharpe:.3f} | {row.annualized_return:.2%} | {row.maximum_drawdown:.2%} |')
    lines += ['', 'Cost multipliers are applied to each account’s own trades, including ETF controls. Cash earns zero; taxes and a full auction/market-impact model remain excluded. Higher modeled fees do not constitute a live-capacity test.','',
              '## Frozen prospective policy','',
              f'The primary ten-seed five-issuer/20% candidate was frozen at **{policy["frozen_timestamp_utc"]}**, before this control study’s metrics. Its choice is retained regardless of the results here. The return-preserving 25%/trend variant is secondary. [Complete policy](PROSPECTIVE_POLICY.json).','',
              'The policy fixes the architectures, ten seeds, 50:50 rank blend, issuer/tie rules, 20% exposure target, annual chronological refit/checkpoint rules, feature cutoffs, share sizing, costs and execution calendar. Unsupported eligible stock IDs stop collection rather than receiving random mixer weights or being silently dropped.','',
              'The first signal must follow actual activation and the freeze, no earlier than October 28, 2026. One formal evaluation occurs after 252 completed exchange sessions. The primary endpoint is net arithmetic active return versus QQQ with its own 20% volatility target; paired Sharpe/CAGR, other controls, seed diagnostics and block-length checks are secondary. No interim return-based stopping or tuning is permitted. A year may remain statistically underpowered.','',
              '**Prepared, not activated:** reviewed local observations and model exports used here stop in 2025. Fresh point-in-time membership/actions and AH/PM inputs, supported stock-ID handling, current-year model refits, a future-date inference audit, a verified exchange calendar, and append-only predictions with a pre-outcome timestamp anchor are still required. No prospective performance has been collected. The older confirmation gate is unchanged. [Activation readiness](READINESS.json).','',
              '## Verification and limitations','',
              f'The independent audit recalculated all **{audit["independently_recomputed_metrics"]}** account metrics and replayed all **{audit["independent_account_replays"]}** accounts; maximum NAV discrepancy was **${audit["maximum_nav_difference_dollars"]:.10f}**. All 12 neural base-cost paths and both buy-and-hold ETF paths reproduced the parent records. Momentum, issuer weights and exposure rules were independently reconstructed. Future-price perturbation did not affect earlier momentum scores.','',
              'The inherited adjusted-unit dividend/acquisition approximations, reconstructed monthly membership, sparse extended-hours observations, missing publication-vintage guarantees, concentrated top-five risk and daily-close drawdown resolution remain. Audit success verifies the declared simulation; it does not independently validate vendor prices or establish executable/live returns.','',
              '[Protocol](PROTOCOL.json) · [Freeze](FREEZE.json) · [Audit](AUDIT.json) · [All 87 accounts](metrics.csv) · [Annual returns](annual.csv) · [Daily curves](comparison_curves.csv) · [Exact exposure targets](target_exposures.csv)','',
              'Reproduce: `python -m hypershift run run_ndx_matched_controls --stage freeze`, then `--stage run`; audit with `python -m hypershift run audit_ndx_matched_controls`; publish with `python -m hypershift run publish_ndx_matched_controls`. Completed studies refuse recomputation or changed seals.']
    (study.DOC/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    study.parent.write(study.DOC/'PUBLICATION.json',dict(status='published_audited_exploratory_controls',
                       strategies=29,accounts=87,prospective_activated=False,
                       policy_sha256=study.parent.sha(study.POLICY),
                       artifact_sha256={p.name:study.parent.sha(p) for p in study.DOC.iterdir() if p.is_file() and p.name!='PUBLICATION.json'}))
    print('Published audited controls, wealth/drawdown figure and frozen prospective policy:',study.DOC,flush=True)


if __name__ == '__main__':
    main()
