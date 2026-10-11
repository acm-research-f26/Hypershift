"""Publish all fixed long/short allocations after independent replay passes."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from run_ndx_long_short import RUN, SOURCE, verify_lock
from chart_performance_labels import performance_label, LEGEND_TITLE

OUT=Path('docs/nasdaq_monthly_pit/long_short')
POLICIES={'long100':'100% long', 'long75_cash25':'75% long / 25% cash',
    'long50_cash50':'50% long / 50% cash','long75_short25':'75% long / 25% short',
    'long50_short50':'50% long / 50% short'}
PCOLORS={'long100':'#0072b2','long75_cash25':'#56b4e9','long50_cash50':'#009e73',
    'long75_short25':'#d55e00','long50_short50':'#cc79a7'}


def pct(x): return f'{100*x:.1f}%'
def table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def build_section(original_names,original_colors):
    if not (RUN/'AUDIT.json').exists(): return ''
    verify_lock()
    audit=json.loads((RUN/'AUDIT.json').read_text())
    assert audit['status']=='passed' and audit['independent_signed_cash_paths']==570
    OUT.mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
    for name in ['PROTOCOL.json','INPUT_CODE_LOCK.json','AUDIT.json','completion.json','metrics.csv',
        'primary_metrics.csv','primary_curves.csv','annual.csv','monthly.csv','seed_summary.csv','paired_bootstrap.csv']:
        shutil.copyfile(RUN/name,OUT/name)
    names={**original_names,'think_mix_pseudohuber':'THINK + mixer pseudo-Huber'}
    colors={**original_colors,'think_mix_pseudohuber':'#e7298a'}
    models=list(original_names)[:9]+['think_mix_pseudohuber']
    read=lambda name:pd.read_csv(RUN/name,dtype={'seed':str})
    rows=read('metrics.csv'); primary=read('primary_metrics.csv').set_index(['model','k','policy'])
    curves=read('primary_curves.csv'); annual=read('annual.csv'); seeds=read('seed_summary.csv')
    paired=read('paired_bootstrap.csv')
    parent=pd.read_csv(SOURCE/'curves.csv',dtype={'seed':str})
    parent_metrics=pd.read_csv(SOURCE/'metrics.csv',dtype={'seed':str})
    control=lambda model:parent[(parent.model==model)&(parent.seed=='ensemble')&(parent.k==10)&(parent.cost_multiplier==1)]
    control_row=lambda model:parent_metrics[(parent_metrics.model==model)&(parent_metrics.seed=='ensemble')&
        (parent_metrics.k==10)&(parent_metrics.cost_multiplier==1)].iloc[0]
    metric=lambda model,k,policy:primary.loc[(model,k,policy)]
    def path(model,k,policy):
        row=metric(model,k,policy)
        return curves[curves.path_id==row.path_id]
    def save(fig,name):
        fig.savefig(OUT/'figures'/name,dpi=160,bbox_inches='tight',facecolor='white'); plt.close(fig)
    def format_axis(ax,year=None):
        ax.grid(alpha=.2); ax.set_ylabel('Net wealth per initial $1')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7] if year is None else [1,4,7,10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m' if year is None else '%b'))
        ax.set_xlim(pd.Timestamp('2022-01-01' if year is None else f'{year}-01-01'),
            pd.Timestamp('2025-12-31' if year is None else f'{year}-12-31'))
    def wealth_line(ax,sub,label,color,style='-',year=None):
        if year is not None:
            sub=sub[pd.to_datetime(sub.date).dt.year==year]
        dates=pd.to_datetime(sub.date)
        ax.plot(pd.DatetimeIndex([dates.iloc[0]-pd.Timedelta(days=1),*dates]),
            np.r_[1.,np.cumprod(1+sub.net)],label=label,color=color,ls=style,lw=1.7)
    low=high=1.
    for sub in [part for _,part in curves.groupby('path_id')]+[control('SPY'),control('QQQ')]:
        wealth=np.r_[1.,np.cumprod(1+sub.net)]
        low=min(low,wealth.min()); high=max(high,wealth.max())
    pad=(high-low)*.04
    chart_text=[]; allocation_text=[]; year_text=[]; metric_text=[]; seed_text=[]; stress_text=[]
    for k in [5,10,20]:
        for policy in ['long75_short25','long50_short50']:
            fig,ax=plt.subplots(figsize=(16,8))
            for model in models:
                row=metric(model,k,policy)
                wealth_line(ax,path(model,k,policy),performance_label(names[model],row),colors[model],
                    '-.' if model=='think_mix_pseudohuber' else '--' if 'mix' in model else '-')
            for model in ['SPY','QQQ']:
                wealth_line(ax,control(model),performance_label(f'{model} buy-and-hold',control_row(model)),colors[model],'--')
            format_axis(ax); ax.set_ylim(low-pad,high+pad)
            ax.set_title(f'Historical monthly NASDAQ-100 | top {k} long / bottom {k} short | {POLICIES[policy]}')
            ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=9,
                title=LEGEND_TITLE)
            fig.subplots_adjust(left=.07,right=.70,bottom=.17,top=.91)
            fig.text(.07,.085,'Fixed forecasts and allocations; next-close fills; rebalance every five sessions; 3% annual assumed stock borrow.',fontsize=10)
            fig.text(.07,.045,'Includes modeled trading/borrow costs. Short availability and broker margin execution were not established.',fontsize=10)
            filename=f'wealth_{policy}_k{k}.png'; save(fig,filename)
            chart_text.append(f'### K = {k}: {POLICIES[policy]}\n\n![All models, {POLICIES[policy]}, K={k}](long_short/figures/{filename})')
            # Calendar views are display rebasing of continuous accounts, not annual resets.
            fig,axes=plt.subplots(2,2,figsize=(16,10)); ymin=ymax=1.
            for year,ax in zip([2022,2023,2024,2025],axes.flat):
                for model in models+['SPY','QQQ']:
                    sub=control(model) if model in ['SPY','QQQ'] else path(model,k,policy)
                    row=control_row(model) if model in ['SPY','QQQ'] else metric(model,k,policy)
                    wealth_line(ax,sub,performance_label(names[model],row),colors[model], '--' if model in ['SPY','QQQ'] else '-',year)
                    part=sub[pd.to_datetime(sub.date).dt.year==year]
                    w=np.r_[1.,np.cumprod(1+part.net)]; ymin=min(ymin,w.min()); ymax=max(ymax,w.max())
                format_axis(ax,year); ax.set_title(str(year))
            ypad=(ymax-ymin)*.04
            for ax in axes.flat: ax.set_ylim(ymin-ypad,ymax+ypad)
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='center left',bbox_to_anchor=(.80,.5),frameon=False,fontsize=8.5,
                title='2022–2025 net CAGR | annualized SR')
            fig.suptitle(f'{POLICIES[policy]} | top/bottom {k} | calendar-year net wealth',fontsize=15)
            fig.subplots_adjust(left=.07,right=.79,bottom=.10,top=.91,hspace=.32,wspace=.18)
            fig.text(.07,.035,'Year panels are rebased to $1; legend CAGR/SR describe the full 2022–2025 account. Costs and assumed borrow included.',fontsize=10)
            filename=f'years_{policy}_k{k}.png'; save(fig,filename)
            year_text.append(f'### K = {k}: {POLICIES[policy]} by year\n\n![Yearly breakdown, {POLICIES[policy]}, K={k}](long_short/figures/{filename})')
        # Fixed focus models span the original model, mixer, new loss and mixer baseline.
        fig,axes=plt.subplots(2,2,figsize=(16,10))
        for model,ax in zip(['think','think_mix','think_mix_pseudohuber','stockmixer'],axes.flat):
            for policy in POLICIES:
                row=metric(model,k,policy)
                wealth_line(ax,path(model,k,policy),performance_label(POLICIES[policy],row,multiline=False),PCOLORS[policy],
                    '--' if 'cash' in policy else '-')
            for benchmark,color,style in [('QQQ','#111111',':'),('SPY','#777777','--')]:
                wealth_line(ax,control(benchmark),performance_label(f'{benchmark} buy-and-hold',control_row(benchmark),multiline=False),color,style)
            format_axis(ax); ax.set_title(names[model]); ax.legend(fontsize=7.5,frameon=False)
        fig.suptitle(f'Fixed allocations | top/bottom {k} | net wealth per $1',fontsize=15)
        fig.subplots_adjust(left=.07,right=.97,bottom=.11,top=.90,hspace=.32,wspace=.19)
        fig.text(.07,.035,'Cash controls earn zero interest. Short allocations use 3% assumed annual borrow; all execution costs included.',fontsize=10)
        filename=f'allocations_k{k}.png'; save(fig,filename)
        allocation_text.append(f'### Allocation comparison, K = {k}\n\n![Fixed allocation comparisons K={k}](long_short/figures/{filename})')
        compact=[]
        for model in models:
            compact.append([names[model]]+[f'{pct(metric(model,k,p).annualized_return)} / {pct(metric(model,k,p).maximum_drawdown)}' for p in POLICIES])
        metric_text.append(f'### K = {k}: CAGR / maximum drawdown\n\n'+table(['Model']+list(POLICIES.values()),compact))
        seed_rows=[]
        for model in models:
            for policy in ['long75_short25','long50_short50']:
                sub=seeds[(seeds.model==model)&(seeds.k==k)&(seeds.policy==policy)]
                if len(sub):
                    row=sub.iloc[0]
                    seed_rows.append([names[model],POLICIES[policy],pct(row.median_CAGR),
                        f'{pct(row.min_CAGR)} to {pct(row.max_CAGR)}',f'{row.positive_seeds}/5',f'{row.seeds_beating_QQQ}/5'])
        seed_text.append(f'### K = {k}: individual training seeds\n\n'+table(['Model','Allocation','Median CAGR','Seed range','Positive CAGR','Beat QQQ'],seed_rows))
        stress_rows=[]
        for model in models:
            for policy in ['long75_short25','long50_short50']:
                sub=rows[(rows.model==model)&(rows.seed=='ensemble')&(rows.k==k)&(rows.policy==policy)]
                def stress(mult,borrow):
                    return pct(sub[np.isclose(sub.cost_multiplier,mult)&np.isclose(sub.borrow_rate,borrow)].iloc[0].annualized_return)
                stress_rows.append([names[model],POLICIES[policy],stress(1,.03),stress(1,.05),stress(1,.10),stress(2,.03),stress(4,.03)])
        stress_text.append(f'### K = {k}: cost and borrow sensitivity (net CAGR)\n\n'+table(
            ['Model','Allocation','Base 3% borrow','5% borrow','10% borrow','2× trading costs','4× trading costs'],stress_rows))
    diagnostic=[]; yearly=[]
    for model in models:
        for policy in ['long75_short25','long50_short50']:
            row=metric(model,10,policy); years=annual[annual.path_id==row.path_id].set_index('year')
            diagnostic.append([names[model],POLICIES[policy],f'{row.sharpe:.3f}',f'{row.sharpe_rf0:.3f}',
                pct(row.annualized_volatility),f'{row.beta_to_QQQ:.2f}',f'{row.correlation_to_QQQ:.2f}',
                pct(row.mean_net_exposure),pct(years.cumulative_return.min())])
            yearly.append([names[model],POLICIES[policy]]+[pct(years.loc[y,'cumulative_return']) for y in [2022,2023,2024,2025]])
    short=primary.reset_index(); short=short[short.policy.isin(['long75_short25','long50_short50'])]
    positive=int((short.annualized_return>0).sum()); improved=beat_matched=0
    for row in short.itertuples():
        full=metric(row.model,row.k,'long100')
        matched=metric(row.model,row.k,'long75_cash25' if row.policy=='long75_short25' else 'long50_cash50')
        improved+=int(row.maximum_drawdown>full.maximum_drawdown)
        beat_matched+=int(row.annualized_return>matched.annualized_return)
    qqq_cagr=control_row('QQQ').annualized_return
    beat_qqq=int((short.annualized_return>qqq_cagr).sum())
    focus=[]
    for model,k in [('think_mix',10),('think_mix_pseudohuber',5),('stockmixer',5)]:
        for policy in POLICIES:
            row=metric(model,k,policy)
            focus.append([names[model],k,POLICIES[policy],pct(row.annualized_return),pct(row.maximum_drawdown),f'{row.sharpe:.3f}'])
    neutral=short[short.policy=='long50_short50']
    stock=metric('stockmixer',5,'long75_short25')
    stock_cash=metric('stockmixer',5,'long75_cash25')
    stock_seeds=seeds[(seeds.model=='stockmixer')&(seeds.k==5)&(seeds.policy=='long75_short25')].iloc[0]
    stock_interval=paired[(paired.model=='stockmixer')&(paired.k==5)&
        (paired.policy=='long75_short25')&(paired.comparison=='long75_cash25')].iloc[0]
    text=f'''## Long highest / short lowest predicted returns

**Completed: no retraining; 600 disclosed portfolio paths, including 570 new signed accounts independently replayed and 30 unchanged long-only ensemble paths.**
All allocations were fixed before this extension's returns were calculated.
Forecasts, membership, annual model folds, all five seeds and K = 5/10/20 are
unchanged. This tests the portfolio rule, not a new predictor.

Across the 60 base-cost ensemble short portfolios, {positive}/60 have positive
net CAGR, {beat_qqq}/60 beat QQQ's {pct(qqq_cagr)} CAGR, and {improved}/60 have a
smaller maximum drawdown than the corresponding 100% long portfolio.
Only {beat_matched}/60 exceed their **same-long-allocation plus cash** control's
CAGR. These counts span related models and overlapping stock portfolios;
they are not 60 independent market trials.

A favorable example is StockMixer at K = 5: 75% long / 25% short has
{pct(stock.annualized_return)} CAGR and {pct(stock.maximum_drawdown)} maximum drawdown,
versus {pct(stock_cash.annualized_return)} and {pct(stock_cash.maximum_drawdown)} for
75% long / 25% cash. However, its five individual seeds have median CAGR
{pct(stock_seeds.median_CAGR)} (range {pct(stock_seeds.min_CAGR)} to {pct(stock_seeds.max_CAGR)}).
All five are positive but none beat QQQ. Its Sharpe difference versus the cash
control has an exploratory 95% interval [{stock_interval.low:.3f}, {stock_interval.high:.3f}],
which includes zero. This example was identified after viewing all fixed
results; it is not a validated allocation recommendation.

The 50% long / 50% short ensembles have measured daily QQQ betas ranging from
{neutral.beta_to_QQQ.min():.2f} to {neutral.beta_to_QQQ.max():.2f}. Equal long/short dollars
do not enforce zero beta or equal sector exposure, and positions drift between
five-session rebalances. A lowest-ranked stock can still rise; a short then loses.

### What the allocations mean

| Policy | Long target | Short target | Gross stock exposure | Net stock exposure |
| --- | --- | --- | --- | --- |
| 100% long | 100% | 0% | 100% | 100% |
| 75% long / 25% cash | 75% | 0% | 75% | 75% |
| 50% long / 50% cash | 50% | 0% | 50% | 50% |
| 75% long / 25% short | 75% | 25% | 100% | 50% |
| 50% long / 50% short | 50% | 50% | 100% | 0% |

At K = 10, the 75/25 short policy targets 7.5% of NAV in each top-ranked stock
and shorts 2.5% in each bottom-ranked stock. The 50/50 policy targets 5% each
on both sides. Orders use only signal-close information and fill the next close.
The same K applies to each side. Fractional boundary ties cancel overlapping
long/short weights; under 2K eligible stocks the short portfolio stays in cash.
Short proceeds remain in the cash/collateral account; they do not increase the
declared investment budgets. Actual cash balances differ from net exposure.

### Previously discussed model comparisons

All full-period wealth legends show net CAGR alongside Sharpe (SR), including
QQQ and SPY. Shared legends on four-panel yearly figures describe the complete
2022–2025 account; each panel is rebased for that calendar year. Sharpe is
annualized from daily returns and uses a 3% annual hurdle.

{table(['Model','K','Allocation','Net CAGR','Max drawdown','Sharpe'],focus)}

{'\n\n'.join(allocation_text)}

### Every model and portfolio size

Each cell reports **annualized net return / worst peak-to-trough loss**.
All results include trading fees; short portfolios additionally use assumed
3% annual borrowing fees. Long-only controls have no borrow charge.

{'\n\n'.join(metric_text)}

{'\n\n'.join(chart_text)}

### Top/bottom 10 risk and market exposure

Sharpe uses the same 3% annual hurdle as the earlier charts. Sharpe at 0%
is shown too, because these research cash accounts earn zero interest.
Beta and correlation are measured against matched daily QQQ returns after costs;
they describe this sample, not a beta constraint imposed on trading.

{table(['Model','Allocation','Sharpe 3%','Sharpe 0%','Volatility','QQQ beta','QQQ corr.','Mean net exposure','Worst year'],diagnostic)}

### Calendar-year returns, K = 10

{table(['Model','Allocation','2022','2023','2024','2025'],yearly)}

{'\n\n'.join(year_text)}

### Training-seed consistency

Ensembles average ranks across all five seeds; their returns are not the mean
individual-seed returns. Seed ranges measure training sensitivity on the same
historical market path, not independent economic evidence. The non-neural
baselines have no training-seed variation.

{'\n\n'.join(seed_text)}

### Borrowing and execution sensitivity

The fee rates below are **assumptions**, not observed historical lending quotes.
Trading-cost stresses and borrow-fee stresses are separate scenarios.
Fees accrue on prior-close short notional using calendar days / 365, including
weekends; no cash interest or short-proceeds rebate is credited.

{'\n\n'.join(stress_text)}

### What was verified and what remains approximate

An independent implementation reproduced all 570 new daily signed cash accounts,
fees and borrow debits; largest ledger difference was
${audit['largest_ledger_difference_dollars']:.8f}. The signed engine also reproduced all
30 original long-only ensemble paths. A future-price/liquidity/score perturbation
left {audit['future_perturbation_unchanged_daily_marks']} earlier daily account marks unchanged.
Return metrics and calendar-year compounding were checked independently.

The models and IC/RankIC/NDCG are unchanged: shorting changes which positions
are held, not the existing forecasts or their ranking accuracy. Forecast audits
from the parent and pseudo-Huber experiments still apply. Paired Sharpe
intervals compare every short portfolio with its matched cash control, full-long
portfolio and QQQ using 1,000 resamples of 20-session blocks. They remain
exploratory, overlapping and unadjusted for multiple comparisons.

These signed holdings use the same **adjusted economic-price total-return proxy**
as the parent study. Dividend effects enter with the position's sign; adding a
second dividend debit would double-count them. Acquisition consideration is
settled at the reviewed completion close with an absolute positive cost for
stock consideration, including when settling a short obligation. This retains
the parent's settlement-timing approximation and is not a raw-share broker replay.

Historical stock locates, recalls, stock-specific changing borrowing fees,
broker margin liquidations, settlement financing and short-sale execution
restrictions were not established. No borrowing of cash is simulated; short
sale proceeds and idle cash earn zero. Actual shorting costs depend on stock
borrowing fees and proceeds interest ([IBKR explanation](https://www.interactivebrokers.com/en/pricing/short-sale-cost.php?menu=A));
short sellers owe dividend-equivalent payments ([SEC explanation](https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work/stock-purchases-and-sales-long-and)).
Therefore the charts estimate the signed strategy under explicit assumptions;
they do not establish that every historical short was executable. 2022–2025
has already been inspected extensively; unseen dates remain necessary.

[All 600 path metrics](long_short/metrics.csv) · [150 primary ensemble curves](long_short/primary_curves.csv) ·
[All calendar years](long_short/annual.csv) · [Monthly returns](long_short/monthly.csv) ·
[Seed summaries](long_short/seed_summary.csv) · [All paired intervals](long_short/paired_bootstrap.csv) ·
[Frozen protocol](long_short/PROTOCOL.json) · [Independent audit](long_short/AUDIT.json).
'''
    return text
