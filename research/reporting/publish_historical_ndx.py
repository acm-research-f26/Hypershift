"""Publish every declared result only after the independent audit passes."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from audit_historical_ndx_ranking import main as audit_rankings
from publish_ndx_topk_extension import build_section as build_extension_section
from publish_ndx_long_short import build_section as build_short_section
from chart_performance_labels import performance_label, LEGEND_TITLE
from publish_ndx_rank_depth import build_section as build_rank_depth_section

RUN=Path('runs/nasdaq100_historical_monthly_20261011')
OUT=Path('docs/nasdaq_monthly_pit')
NAMES={'think':'THINK','think_mix':'THINK + mixer','think_huber':'THINK Huber',
       'think_mix_huber':'THINK + mixer Huber','mlp':'MLP','stockmixer':'StockMixer',
       'ridge':'Ridge','momentum126':'Momentum 126','reversal5':'Reversal 5',
       'monthly_basket':'Monthly equal basket','SPY':'SPY','QQQ':'QQQ'}
NEURAL=list(NAMES)[:6]
ORDER=list(NAMES)
COLORS=dict(think='#0072b2',think_mix='#d55e00',think_huber='#56b4e9',think_mix_huber='#cc79a7',
            mlp='#7f7f7f',stockmixer='#8c564b',ridge='#6e4c9e',momentum126='#e6ab02',
            reversal5='#17becf',monthly_basket='#228b22',SPY='#111111',QQQ='#8844aa')


def pct(value):return f'{100*value:.1f}%'
def num(value):return f'{value:.3f}'
def table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])
def primary(frame):
    return frame[(frame.seed=='ensemble')&(frame.k==10)&(frame.cost_multiplier==1)].copy()
def save(fig,name):
    fig.savefig(OUT/'figures'/name,dpi=165,bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    completed=json.loads((RUN/'completion.json').read_text())
    audit=json.loads((RUN/'AUDIT.json').read_text())
    assert completed['neural_fits']==120 and completed['continuous_portfolio_paths']==120
    assert audit['status']=='passed' and audit['independent_share_cash_paths']==120
    audit_rankings()
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    read=lambda f:pd.read_csv(RUN/f,dtype={'seed':str})
    metrics=read('metrics.csv');curves=read('curves.csv');annual=read('annual.csv');monthly=read('monthly.csv')
    paired=read('paired_bootstrap.csv');fits=read('training_records.csv')
    ranking=read('ranking.csv').set_index('model')
    ranking_periods=pd.read_csv(OUT/'ranking_periods.csv')
    p=primary(metrics).set_index('model');a=primary(annual);m=primary(monthly);c=primary(curves)
    assert set(p.index)==set(ORDER)
    for f in ['metrics.csv','annual.csv','monthly.csv','training_records.csv','paired_bootstrap.csv','ranking.csv',
              'AUDIT.json','PROTOCOL.json','INPUT_CODE_LOCK.json','DEPENDENCY_CODE_LOCK.json','FITS_COMPLETE_LOCK.json','completion.json']:
        shutil.copyfile(RUN/f,OUT/f)
    c.to_csv(OUT/'primary_curves.csv',index=False)
    shutil.copyfile('data/nasdaq_monthly_pit/assembled/fold_preflight.json',OUT/'fold_preflight.json')
    # Mirror the earlier five-session charts: one linear wealth axis, all
    # declared models, and a right-hand legend with matched daily Sharpe.
    fig,ax=plt.subplots(figsize=(16,8))
    for model in ORDER:
        sub=c[c.model==model];dates=pd.to_datetime(sub.date)
        wealth=np.cumprod(1+sub.net.to_numpy())
        np.testing.assert_allclose(wealth[-1],1+p.loc[model,'cumulative_return'],atol=1e-12)
        ax.plot(dates,wealth,label=performance_label(NAMES[model],p.loc[model]),
                color=COLORS[model],lw=2.3 if model in ['QQQ','SPY','think_mix'] else 1.6,
                linestyle='--' if 'mix' in model or model=='QQQ' else '-')
    for year in [2023,2024,2025]:
        ax.axvline(pd.Timestamp(f'{year}-01-01'),color='#666666',lw=.7,ls=':',alpha=.4)
    ax.set(title='Historical monthly NASDAQ-100 | 5-session forecasts | all declared models',
           ylabel='Net wealth per initial $1')
    ax.set_xlim(pd.Timestamp('2022-01-01'),pd.Timestamp('2025-12-31'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(alpha=.2)
    ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=10,
              title=LEGEND_TITLE,title_fontsize=10)
    fig.subplots_adjust(left=.07,right=.70,bottom=.16,top=.90)
    fig.text(.07,.08,'Fixed equal top 10; neural models use all five seeds in a rank ensemble. All curves include modeled costs.',fontsize=10)
    fig.text(.07,.045,'Continuous 2022–2025 accounts; no annual reset. Historical monthly membership; public-data limitations remain.',fontsize=10)
    save(fig,'continuous_accounts.png')

    # Calendar-year views rebase the same continuous account for display only.
    dated=c.assign(year=pd.to_datetime(c.date).dt.year)
    year_paths={}
    for year in [2022,2023,2024,2025]:
        for model in ORDER:
            sub=dated[(dated.year==year)&(dated.model==model)]
            wealth=np.r_[1.,np.cumprod(1+sub.net.to_numpy())]
            row=a[(a.year==year)&(a.model==model)].iloc[0]
            np.testing.assert_allclose(wealth[-1],1+row.cumulative_return,atol=1e-12)
            year_paths[year,model]=(sub,wealth,row)
    low=min(path[1].min() for path in year_paths.values())
    high=max(path[1].max() for path in year_paths.values())
    pad=(high-low)*.05
    for model in ORDER:
        np.testing.assert_allclose(np.prod([year_paths[year,model][1][-1] for year in [2022,2023,2024,2025]]),
                                   1+p.loc[model,'cumulative_return'],atol=1e-12)
    yearly_sections=[]
    for year in [2022,2023,2024,2025]:
        fig,ax=plt.subplots(figsize=(16,8))
        for model in ORDER:
            sub,wealth,row=year_paths[year,model]
            dates=pd.DatetimeIndex([pd.Timestamp(year,1,1),*pd.to_datetime(sub.date)])
            ax.plot(dates,wealth,label=performance_label(NAMES[model],row),
                    color=COLORS[model],lw=2.3 if model in ['QQQ','SPY','think_mix'] else 1.6,
                    linestyle='--' if 'mix' in model or model=='QQQ' else '-')
        ax.set(title=f'Historical monthly NASDAQ-100 | {year} | 5-session forecasts | all declared models',
               ylabel='Net wealth per $1 at year start',ylim=(low-pad,high+pad))
        ax.set_xlim(pd.Timestamp(year,1,1),pd.Timestamp(year,12,31))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,3,5,7,9,11]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.axhline(1,color='#666666',lw=.8,ls=':',alpha=.6);ax.grid(alpha=.2)
        ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=10,
                  title=f'{year} annualized CAGR | annualized SR\n(3% annual hurdle)',title_fontsize=10)
        fig.subplots_adjust(left=.07,right=.70,bottom=.16,top=.90)
        fig.text(.07,.08,'Fixed equal top 10; all five seeds in a rank ensemble. Same vertical scale across all four years.',fontsize=10)
        fig.text(.07,.045,'Rebased to $1 for display; the continuous account carries holdings and cash across years. Modeled costs included.',fontsize=10)
        name=f'accounts_{year}.png';save(fig,name)
        yearly_sections.append(f'### {year}\n\n![{year}: all models, net wealth per $1 at year start](figures/{name})')

    seed=metrics[(metrics.seed!='ensemble')&(metrics.k==10)&(metrics.cost_multiplier==1)]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,field,label in [(axes[0],'annualized_return','Net CAGR (%)'),(axes[1],'sharpe','Daily Sharpe (3% annual hurdle)')]:
        factor=100 if field=='annualized_return' else 1
        for i,model in enumerate(NEURAL):
            sub=seed[seed.model==model].sort_values('seed');values=sub[field].to_numpy()*factor
            ax.plot([values.min(),values.max()],[i,i],color=COLORS[model],lw=3,alpha=.4)
            ax.scatter(values,np.linspace(i-.11,i+.11,len(values)),color=COLORS[model],s=32)
            ax.scatter(p.loc[model,field]*factor,i,marker='D',facecolors='white',edgecolors=COLORS[model],s=70,zorder=5)
        for model in ['SPY','QQQ']:
            ax.axvline(p.loc[model,field]*factor,color=COLORS[model],ls='--',label=performance_label(NAMES[model],p.loc[model]))
        ax.set_yticks(range(6),[NAMES[x] for x in NEURAL]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.set_xlabel(label);ax.legend(fontsize=9)
    fig.suptitle('Training-seed stability | five individual seeds; hollow diamond = rank ensemble')
    fig.tight_layout();save(fig,'seed_stability.png')

    av=a.pivot(index='model',columns='year',values='cumulative_return').reindex(ORDER)
    fig,ax=plt.subplots(figsize=(9,7));values=av.to_numpy()*100
    limit=max(10,float(np.nanmax(np.abs(values))))
    im=ax.imshow(values,cmap='RdYlGn',vmin=-limit,vmax=limit,aspect='auto')
    for i in range(len(ORDER)):
        for j in range(4):ax.text(j,i,f'{values[i,j]:.1f}%',ha='center',va='center',fontsize=10)
    ax.set_xticks(range(4),av.columns);ax.set_yticks(range(len(ORDER)),[NAMES[x] for x in ORDER])
    ax.set_title('Calendar-year net return | the same continuous account')
    fig.colorbar(im,ax=ax,label='Net return (%)');fig.tight_layout();save(fig,'annual_returns.png')

    mv=m.pivot(index='model',columns='month',values='net_return').reindex(ORDER)
    excess=100*mv.subtract(mv.loc['QQQ'],axis=1)
    fig,ax=plt.subplots(figsize=(14,7));values=excess.to_numpy()
    limit=max(1,float(np.nanmax(np.abs(values))))
    im=ax.imshow(values,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(range(len(ORDER)),[NAMES[x] for x in ORDER]);ax.set_xticks(range(0,48,3),mv.columns[::3],rotation=60)
    ax.set_title('Monthly net excess return over QQQ | every month shown; blue = ahead')
    fig.colorbar(im,ax=ax,label='Percentage points vs QQQ');fig.tight_layout();save(fig,'monthly_excess_qqq.png')

    fig,axes=plt.subplots(1,2,figsize=(12,6))
    variants=ORDER[:9]
    for ax,column,filters,title in [(axes[0],'k',metrics.cost_multiplier==1,'Allocation sensitivity (base costs)'),
                                    (axes[1],'cost_multiplier',metrics.k==10,'Cost stress (fixed top 10)')]:
        sub=metrics[(metrics.seed=='ensemble')&filters&metrics.model.isin(variants)]
        pivot=sub.pivot(index='model',columns=column,values='annualized_return').reindex(variants)*100
        values=pivot.to_numpy();limit=max(10,float(np.nanmax(np.abs(values))))
        im=ax.imshow(values,cmap='RdYlGn',vmin=-limit,vmax=limit,aspect='auto')
        for i in range(9):
            for j in range(3):ax.text(j,i,f'{values[i,j]:.1f}%',ha='center',va='center',fontsize=10)
        ax.set_yticks(range(9),[NAMES[x] for x in variants])
        ax.set_xticks(range(3),[f'Top {int(x)}' if column=='k' else f'{int(x)}× cost' for x in pivot.columns])
        ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7,label='Net CAGR (%)')
    fig.suptitle('All declared sensitivity results | these do not select a winning policy')
    fig.tight_layout();save(fig,'sensitivity.png')

    mainrows=[];consistency=[]
    for model in ORDER:
        row=p.loc[model];yr=av.loc[model];months=mv.loc[model]
        mainrows.append([NAMES[model],pct(row.annualized_return),num(row.sharpe),pct(row.maximum_drawdown),
                         pct(yr.min()),pct(row.cumulative_return)])
        consistency.append([NAMES[model],f'{int((months>mv.loc["QQQ"]).sum())}/48',
                            f'{int((yr>av.loc["QQQ"]).sum())}/4',f'{int((months>mv.loc["SPY"]).sum())}/48',
                            f'{int((yr>av.loc["SPY"]).sum())}/4',f'{int((yr>0).sum())}/4',
                            pct((months-mv.loc['QQQ']).std())])
    seedrows=[]
    for model in NEURAL:
        sub=seed[seed.model==model]
        seed_years=annual[(annual.model==model)&(annual.seed!='ensemble')&(annual.k==10)&(annual.cost_multiplier==1)]
        excess=seed_years.cumulative_return-seed_years.year.map(av.loc['QQQ'].to_dict())
        seedrows.append([NAMES[model],f'{pct(sub.annualized_return.min())} / {pct(sub.annualized_return.median())} / {pct(sub.annualized_return.max())}',
                         f'{num(sub.sharpe.min())} / {num(sub.sharpe.median())} / {num(sub.sharpe.max())}',
                         f'{int((sub.annualized_return>p.loc["QQQ","annualized_return"]).sum())}/5',
                         pct(seed_years.cumulative_return.min()),f'{int((excess>0).sum())}/20'])
    uncertainty=[]
    for r in paired.itertuples():
        if r.comparison!='QQQ' and r.comparison not in ['think','think_huber']:continue
        uncertainty.append([NAMES[r.model],NAMES[r.comparison],num(r.observed),f'[{num(r.low)}, {num(r.high)}]',
                            'includes zero' if r.low<=0<=r.high else 'excludes zero'])
    fpre=json.loads((OUT/'fold_preflight.json').read_text())
    folds=table(['Test year','Train forecasts','Validation forecasts','Test forecasts','Unobserved train / validation targets'],
                [[r['year'],r['training_dates'],r['validation_dates'],r['test_forecasts'],
                  f"{r['unobserved_training_labels']} / {r['unobserved_validation_labels']}"] for r in fpre])
    annualtable=table(['Model']+[str(y) for y in av.columns],[[NAMES[x]]+[pct(v) for v in av.loc[x]] for x in ORDER])
    mixer=[]
    for mixed,base in [('think_mix','think'),('think_mix_huber','think_huber')]:
        r=paired[(paired.model==mixed)&(paired.comparison==base)].iloc[0]
        mixer.append(f'{NAMES[mixed]} versus {NAMES[base]} has a Sharpe difference of {num(r.observed)} '
                     f'with exploratory 95% interval [{num(r.low)}, {num(r.high)}]. '
                     +('The interval includes zero, so this study does not establish a consistent mixer advantage.'
                       if r.low<=0<=r.high else 'The interval excludes zero in this sample; it remains exploratory and unadjusted for the many comparisons.'))
    ahead=[NAMES[x] for x in NEURAL if (av.loc[x]>av.loc['QQQ']).all()]
    allseeds=[NAMES[x] for x in NEURAL if (seed[seed.model==x].annualized_return>p.loc['QQQ','annualized_return']).all()]
    assert p.annualized_return.idxmax()=='think_mix' and p.sharpe.idxmax()=='think_mix'
    lead=paired[(paired.model=='think_mix')&(paired.comparison=='think')].iloc[0]
    winner=(f'THINK + mixer has the highest primary net CAGR ({pct(p.loc["think_mix","annualized_return"])}) '
            f'and Sharpe ({num(p.loc["think_mix","sharpe"])}), but is **not a definitive best model**. '
            f'Its maximum drawdown is {pct(p.loc["think_mix","maximum_drawdown"])}. '
            f'Its Sharpe lead over THINK has an exploratory 95% interval [{num(lead.low)}, {num(lead.high)}], '
            'which includes zero; only three of five individual seeds beat QQQ, and the ensemble trails QQQ in 2024. '
            'Higher return in this inspected sample does not establish reliable future superiority.')
    rankingrows=[[NAMES[x]]+[f'{ranking.loc[x,field]:.4f}' for field in
                 ['IC','RankIC','ICIR','RankICIR','NDCG@5','NDCG@10','NDCG@20']] for x in ORDER[:9]]
    random10=ranking_periods.groupby('model')['Random NDCG@10'].mean()
    rankinterpretation=(f'THINK + mixer: IC {ranking.loc["think_mix","IC"]:.4f}, '
                        f'RankIC {ranking.loc["think_mix","RankIC"]:.4f}, NDCG@10 {ranking.loc["think_mix","NDCG@10"]:.4f}; '
                        f'its matched random-ranking expectation is {random10["think_mix"]:.4f}. '
                        f'Regular THINK has higher IC ({ranking.loc["think","IC"]:.4f}) and '
                        f'RankIC ({ranking.loc["think","RankIC"]:.4f}); THINK + mixer has the highest mean '
                        'NDCG@10 among these nine signals. Its ranking advantage is small in absolute terms and '
                        'has not been established as statistically significant. These metrics do not identify a universal winner.')
    headline=(f'QQQ returned {pct(p.loc["QQQ","annualized_return"])} CAGR (Sharpe {p.loc["QQQ","sharpe"]:.3f}); SPY returned '
              f'{pct(p.loc["SPY","annualized_return"])} CAGR (Sharpe {p.loc["SPY","sharpe"]:.3f}). '
              +(f'Neural ensembles ahead of QQQ in every calendar year: {", ".join(ahead)}. ' if ahead else
                'No neural ensemble beat QQQ in every calendar year. ')
              +(f'Variants whose five individual seeds all exceeded QQQ’s full-period CAGR: {", ".join(allseeds)}.' if allseeds else
                'No neural variant had all five individual seeds exceed QQQ’s full-period CAGR.'))
    extension_section=build_extension_section(NAMES,COLORS)
    short_section=build_short_section(NAMES,COLORS)
    rank_depth_section=build_rank_depth_section()
    text=f'''# Historical monthly NASDAQ-100 charts, 2022–2025

**Original batch — Completed: 120 neural fits, 120 continuous portfolio paths; independent audit passed.**
{headline}

Original top-10 comparison: {winner}

This uses the user-authorized **official NASDAQ-100 fallback**, selected at each
historical month start, rather than the unavailable custom 100-largest-NASDAQ-company
ranking. Financial companies are excluded. Every included share class is retained
(101–104 securities); the 2026-selected NASDAQ models are retired and deleted.

{rank_depth_section}

{short_section}

{extension_section}

## NASDAQ-100 historical monthly universe, 5-session forecasts

![All models: net wealth per initial $1](figures/continuous_accounts.png)

All six neural variants, three simple signals, the monthly basket, SPY and QQQ
share one linear wealth axis. Net wealth per initial $1 is the cumulative product
of one plus each daily net return, including entry and exit fees. The legend
shows net CAGR and annualized net Sharpe (SR) together for each curve, including
QQQ and SPY. Sharpe is calculated from daily returns and uses a 3% annual hurdle.

The primary portfolio is fixed equal top 10, long only, rebalanced every five
sessions. Signals freeze at the prior close; share orders fill at the next close.
Accounts start with $1 million on 2022-01-03 and end with liquidation on
2025-12-31. Holdings and cash carry across annual model changes. ETF controls use
the same dates and entry/exit cost schedule. Neural primary results average
eligible-stock ranks from all five seeds; no best seed or best portfolio size is chosen.

**Fixed equal top 10** means the portfolio targets the ten highest-scoring
eligible securities at each rebalance, with 10% of signal-time account value
allocated to each, long only and without leverage. For a $1 million account,
that is a $100,000 target per security before fees and execution constraints.
"Fixed" refers to the predeclared count of ten, not ten permanently held names;
the selected names can change every five trading sessions. "Equal" means score
size does not change the target allocation. The strategy still buys the highest
scores even when all predicted returns are negative. Ties at the tenth-place
boundary split the remaining allocation fairly and can produce more than ten
positions. Prices move between the signal and fill; fees, liquidity caps and
cash limits can leave actual weights unequal or leave residual cash. Weights
also drift between rebalances. Top 5 and top 20 are sensitivity checks, not
choices made after seeing which returned the most.

{table(['Model','Net CAGR','Sharpe','Max drawdown','Worst year','Total return'],mainrows)}

Sharpe uses daily net returns, 252 sessions/year and a fixed 3% annual hurdle;
CAGR uses compounded net returns. Drawdowns include initial cash. These are
research backtests, not verified executable returns or a future-return forecast.

## Forecast ranking: IC, RankIC and NDCG

{table(['Model','IC','RankIC','ICIR','RankICIR','NDCG@5','NDCG@10','NDCG@20'],rankingrows)}

These are equally weighted averages across **200 complete five-session forecast
periods** in 2022–2025, calculated across signal-time eligible stocks with
observed next-close-to-five-sessions-later returns. They are gross forecast
diagnostics; transaction costs enter the wealth chart, not these metrics.
Neural rows use the same five-seed average rank scores that drive the primary
portfolio, so their IC measures ensemble score/return correlation rather than
calibration of a predicted return magnitude. The final partial holding period
has no complete target and is excluded; 101 eligible stock/period targets in
total are unobserved, including that final period. SPY, QQQ and the monthly
equal basket do not produce cross-sectional stock forecasts, so these metrics
are not applicable to them.

- **IC (information coefficient):** Pearson correlation between the model's
  scores and subsequent realized returns within each forecast period.
- **RankIC:** Spearman correlation, using average ranks for ties. Both
  correlations range from -1 to +1; zero indicates no linear or rank association
  respectively. They describe the whole eligible stock ranking.
- **ICIR / RankICIR:** mean period correlation divided by its sample standard
  deviation. These are not annualized and are not significance tests.
- **NDCG@k:** normalized discounted cumulative gain for the first k predicted
  positions. Realized-return percentile ranks supply nonnegative linear gains;
  positions receive a 1/log2(position + 1) discount, with positions numbered
  from one. Prediction ties average their possible position discounts. One
  equals the hindsight ideal ranking for those outcomes. **0.539 does not mean
  53.9% correct predictions**: random orderings also receive positive gain.
  Even an all-negative-return period can have high NDCG because this metric
  judges relative ordering, not whether buying those stocks makes money.

{rankinterpretation}

The ranking summaries were independently recomputed from frozen predictions
and ridge coefficients using separate rank, correlation and discount
calculations. [Ranking audit](RANKING_AUDIT.json) ·
[Pooled ranking metrics](ranking.csv) · [Ranking metrics by year](ranking_by_year.csv) ·
[Every forecast period and matched random NDCG expectation](ranking_periods.csv).

## Year-by-year wealth charts

Each chart shows all twelve primary model and control paths for one calendar
year, with the same colors, line styles and vertical scale. Curves are rebased
to $1 at the start of each year so that within-year gains and losses can be
compared; this is display normalization of the existing continuous account.
Holdings and cash still carry across years. Each legend reports that year's
annualized CAGR and annualized net Sharpe, and each curve's final value minus one equals the net annual
return in the table below. The four annual growth factors compound to the
full-period return shown above.

{chr(10).join(yearly_sections)}

## Consistency by year, month and training seed

![Calendar-year returns](figures/annual_returns.png)

{annualtable}

![Every month versus QQQ](figures/monthly_excess_qqq.png)

{table(['Model','Months > QQQ','Years > QQQ','Months > SPY','Years > SPY','Positive years','Monthly QQQ excess SD'],consistency)}

Equal returns count as ties rather than wins. Monthly excess SD is measured in
percentage points; a low spread alone is not success if returns remain poor.
The basket is equal weighted at the first scheduled fill of each month; QQQ
uses index weights and actual fund returns, so they are different controls.

![All five individual training seeds](figures/seed_stability.png)

{table(['Model','Seed CAGR: min / median / max','Seed Sharpe: min / median / max','Seeds CAGR > QQQ','Worst seed/year','Seed/year wins vs QQQ'],seedrows)}

Seeds 7, 19, 42, 61 and 97 vary initialization within the same dated universe.
They are not five independent market histories. The earlier three random draws
tested cohort sensitivity separately and had substantial dispersion; see
[all earlier random-sample charts](../random_trials_20261010/CHARTS.md).
Those draws requested 100 names but recovered only 54–70 price histories, so
they are not complete historical random samples of 100. Neither averaging
seeds nor averaging draws removes a biased universe. Historical
membership addresses that selection problem; unseen forward data remain necessary.

## Fixed sensitivity checks and uncertainty

![All portfolio sizes and cost multipliers](figures/sensitivity.png)

Base costs are 7 basis points plus 3 basis points times the square root of
participation divided by 1%. Trades are capped at 1% of signal-time 20-session
average dollar volume. Cash-short buy orders resize proportionally; missing
fills cancel without reranking. Both costs and impact are multiplied in the
2×/4× stress paths. This remains a fractional-unit, adjusted-total-return account
approximation; auction fill quality, integer lots, taxes and cash interest are
not modeled. No alternative size or cost scenario becomes the primary result.

{table(['Model','Comparison','Sharpe difference','Exploratory 95% interval','Zero check'],uncertainty)}

The intervals use 1,000 paired circular 20-session block resamples of daily
returns. They preserve same-date pairing but are not adjusted for multiple
comparisons, data revisions or repeated use of these test years.

{' '.join(mixer)}

Adding a mixer changes the learned cross-stock interactions and adds parameters;
it need not improve returns. Training minimizes forecast loss, while the traded
portfolio depends on the ranking near the top-10 boundary. Better forecast loss
does not automatically mean a better portfolio. The earlier NASDAQ/NYSE
difference cannot establish an exchange-specific advantage from one sample.

## What the models do

**126-session momentum** scores a stock as adjusted close today divided by
adjusted close 126 trading sessions ago, minus one: roughly six months of
total-return momentum. It has no fitted coefficients and does not skip the
latest month. Here it buys the ten highest available scores every five sessions.
Large gains can occur when recent winners keep winning; concentrated exposure
also loses when leadership reverses. The earlier spectacular result used
2026-selected survivors and a different annual-reset experiment, so it is not
directly comparable to this historical, continuous account.

**Ridge** is one shared linear return forecast fitted across the eligible
historical stocks and dates. Each stock contributes 16 sessions of five
features (80 numbers), plus an intercept. It forecasts the next five-session
return with squared-error loss and an L2 coefficient penalty of 10; the
intercept is unpenalized. Shrinkage controls large fitted weights. It does not
directly optimize Sharpe, CAGR or portfolio returns.

**THINK** is this repository's local hyperbolic temporal/hypergraph adaptation.
Its graph links stocks using positive correlations computed only on training
history and historical membership. **THINK + mixer** adds a learned cross-stock
residual to its scores. The **Huber** variants cap the influence of large
forecast errors beyond one percentage point; they still select epochs using
prior-year validation MSE. **MLP** applies a small shared nonlinear network to
each stock's feature window. **StockMixer** combines temporal, feature and
cross-stock mixing and includes a pairwise ranking penalty. These implementations
are controlled local comparisons, not certified reproductions of published papers.
**Reversal 5** ranks by negative recent five-session return and is unfitted.

All learned forecasts use daily return, distance from 5/20-session moving
averages, 20-session return volatility and dollar-volume relative to its
20-session average. Training-only normalization, monthly masks and inactive-node
sanitization prevent future members from entering active graph/mixer calculations.
Targets are next-close to five sessions later; outcomes crossing train/validation
boundaries are purged. Architecture and seed comparisons were declared before
these replacement test metrics; no live candidate is selected from this table.

## Data and audit limits

Training begins July 2019, a declared free-source amendment made before these
metrics. The four folds have 1.5, 2.5, 3.5 and four training years, respectively,
and the preceding calendar year for validation. The first fold has materially
less training history. Earlier prices only warm up features.

{folds}

Four unique supervised targets spanning older unpriced mergers are unobserved;
fold rows overlap, so those counts should not be summed into distinct targets.
They are excluded from loss without removing their signal-time members. The
last partial test holding period is valued and liquidated but has no full
five-session supervised target.

Membership is a pinned historical event reconstruction with annual closure
checks and Nasdaq annual-announcement cross-checks. It is not a fully licensed
as-published constituent database. The exact 2025 SOLS off-cycle dates conflict
across sources; both produce the same November monthly snapshot, and IPO warmup
keeps SOLS outside the forecasting portfolios. Monthly membership is intentionally
frozen even when the actual index changes within the month.

Tiingo supplies recovered former-member histories; Yahoo supplies continuing
prices. Reused/unidentified histories were rejected. No unexplained selected-stock
trading-price gaps remain. Vendor rows after acquisition were trimmed. XLNX,
SGEN, SPLK and ANSS use verified cash/share acquisition consideration; old
holdings become cash rather than being carried indefinitely at their last quote.
Stock consideration is sold at completion close with fees and cash is available
then: actual broker settlement can be later, and no settlement-delay stress was
run. Retrospectively adjusted public prices can still contain revisions/errors.

The audit reproduced **{audit['neural_checkpoints_reproduced']} checkpoints**,
training-only scaling/graphs and ridge coefficients, independently replayed
**{audit['independent_share_cash_paths']} share/cash paths** and recomputed their
metrics. The largest NAV difference was **${audit['largest_nav_difference_dollars']:.8f}**.
Accounting reproduction checks arithmetic and timing; it does not certify every
provider fact. The 2022–2025 years were already inspected in earlier experiments,
so these results remain exploratory. Consistently good future performance still
requires an unseen period or paper trading with the frozen strategy.

[Data sources and acquisition terms](DATA_REQUIREMENTS.md) ·
[Frozen protocol](PROTOCOL.json) · [Independent audit](AUDIT.json) ·
[120-fit record](training_records.csv) · [All metrics/sensitivities](metrics.csv) ·
[Annual returns](annual.csv) · [Monthly returns](monthly.csv) ·
[Primary daily curves](primary_curves.csv) · [Paired intervals](paired_bootstrap.csv) ·
[Input/code fingerprints](INPUT_CODE_LOCK.json) · [Imported dependency fingerprints](DEPENDENCY_CODE_LOCK.json) ·
[2026 NASDAQ retirement record](../audit/nasdaq_2026_retirement.json)
'''
    (OUT/'CHARTS.md').write_text(text,encoding='utf-8')
    print('Published audited charts and every declared metric:',OUT/'CHARTS.md')


if __name__=='__main__':main()
