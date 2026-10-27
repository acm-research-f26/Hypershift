"""Explain rank-depth sensitivity without attributing it to untested causes."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from analyze_ndx_rank_depth import RUN, MODELS, LABELS, digest, SOURCE
from chart_performance_labels import performance_label
from publish_ndx_loss_similarity import build_section as build_loss_similarity_section

OUT=Path('docs/nasdaq_monthly_pit/rank_depth')


def pct(x): return f'{100*x:.1f}%'
def table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def build_section():
    if not (RUN/'AUDIT.json').exists(): return ''
    audit=json.loads((RUN/'AUDIT.json').read_text())
    assert audit['status']=='passed' and audit['original_paths_unchanged']==6
    for name,expected in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        assert digest(name)==expected,name
    OUT.mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
    for name in ['AUDIT.json','INPUT_CODE_LOCK.json','metrics.csv','curves.csv','stock_attribution.csv',
        'year_stock_attribution.csv','forecast_buckets.csv','paired_bootstrap.csv','seed_pairs.csv','year_concentration.csv']:
        shutil.copyfile(RUN/name,OUT/name)
    metrics=pd.read_csv(RUN/'metrics.csv').set_index(['model','policy','cost_multiplier'])
    curves=pd.read_csv(RUN/'curves.csv'); seed=pd.read_csv(RUN/'seed_pairs.csv')
    stock=pd.read_csv(RUN/'year_stock_attribution.csv'); buckets=pd.read_csv(RUN/'forecast_buckets.csv')
    paired=pd.read_csv(RUN/'paired_bootstrap.csv').set_index('model')
    original=pd.read_csv(SOURCE/'metrics.csv',dtype={'seed':str})
    original_curves=pd.read_csv(SOURCE/'curves.csv',dtype={'seed':str})
    benchmarks=original[(original.model.isin(['QQQ','SPY']))&(original.seed=='ensemble')&
        (original.k==10)&(original.cost_multiplier==1)].set_index('model')
    ranking=pd.read_csv(SOURCE/'ranking.csv').set_index('model')
    periods=pd.read_csv(SOURCE/'ranking_periods.csv')
    def save(fig,name):
        fig.savefig(OUT/'figures'/name,dpi=165,bbox_inches='tight',facecolor='white'); plt.close(fig)
    colors={'top5':'#0072b2','top10':'#d55e00','ranks6_10':'#777777','QQQ':'#8844aa','SPY':'#111111'}
    policy_names={'top5':'Ranks 1–5','top10':'Ranks 1–10','ranks6_10':'Ranks 6–10 diagnostic'}
    fig,axes=plt.subplots(1,3,figsize=(17,6),sharey=True)
    for ax,model in zip(axes,MODELS):
        for policy in policy_names:
            sub=curves[(curves.model==model)&(curves.policy==policy)&(curves.cost_multiplier==1)]
            dates=pd.DatetimeIndex([pd.Timestamp('2022-01-01'),*pd.to_datetime(sub.date)])
            ax.plot(dates,np.r_[1.,np.cumprod(1+sub.net)],color=colors[policy],lw=1.8,
                label=performance_label(policy_names[policy],metrics.loc[(model,policy,1)]))
        for model_control in ['QQQ','SPY']:
            sub=original_curves[(original_curves.model==model_control)&(original_curves.seed=='ensemble')&
                (original_curves.k==10)&(original_curves.cost_multiplier==1)]
            dates=pd.DatetimeIndex([pd.Timestamp('2022-01-01'),*pd.to_datetime(sub.date)])
            ax.plot(dates,np.r_[1.,np.cumprod(1+sub.net)],color=colors[model_control],ls='--',lw=1.4,
                label=performance_label(model_control,benchmarks.loc[model_control]))
        ax.set_title(LABELS[model],fontsize=12); ax.grid(alpha=.2)
        ax.set_xlim(pd.Timestamp('2022-01-01'),pd.Timestamp('2025-12-31'))
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.legend(fontsize=8,frameon=False,loc='upper left')
    axes[0].set_ylabel('Net wealth per initial $1')
    fig.suptitle('What happens when ranks six through ten are added? | existing five-seed rank ensembles',fontsize=14)
    fig.subplots_adjust(left=.06,right=.98,top=.86,bottom=.17,wspace=.12)
    fig.text(.06,.07,'Ranks 6–10 are a standalone diagnostic using the same frozen signals, next-close orders, costs and continuous account.',fontsize=10)
    fig.text(.06,.025,'CAGRs cannot be averaged to reconstruct top ten. Sharpe is annualized from daily returns with a 3% annual hurdle.',fontsize=10)
    save(fig,'rank_depth_wealth.png')
    # Exact additive account attribution for the year that dominates StockMixer's gap.
    st=stock[(stock.model=='stockmixer')&(stock.year==2024)].pivot(index='ticker',columns='policy',values='year_return_contribution')
    selected=['SMCI','NVDA','TSLA']; names=selected+['All other stocks and their fees']
    top5=[st.loc[t,'top5']*100 for t in selected]+[(st.top5.sum()-st.loc[selected,'top5'].sum())*100]
    top10=[st.loc[t,'top10']*100 for t in selected]+[(st.top10.sum()-st.loc[selected,'top10'].sum())*100]
    fig,ax=plt.subplots(figsize=(12,6)); y=np.arange(4)
    ax.barh(y-.18,top5,height=.34,color=colors['top5'],label='Top 5')
    ax.barh(y+.18,top10,height=.34,color=colors['top10'],label='Top 10')
    for offset,values in [(-.18,top5),(.18,top10)]:
        for i,value in enumerate(values): ax.text(value+(.35 if value>=0 else -.35),i+offset,f'{value:+.1f} pp',
            va='center',ha='left' if value>=0 else 'right',fontsize=10)
    ax.set_yticks(y,names); ax.invert_yaxis(); ax.axvline(0,color='#444444',lw=.8)
    ax.set_xlabel('Contribution to 2024 portfolio return (percentage points)'); ax.grid(axis='x',alpha=.2)
    ax.set_title(f'StockMixer 2024 | top 5 {pct(st.top5.sum())} versus top 10 {pct(st.top10.sum())}')
    ax.legend(frameon=False); ax.set_xlim(min(top5+top10)-5,max(top5+top10)+5)
    fig.subplots_adjust(left=.28,right=.95,top=.88,bottom=.18)
    fig.text(.04,.045,'Contributions sum to the actual net calendar-year return. These are portfolio P&L contributions, not each stock’s own return.',fontsize=10)
    save(fig,'stockmixer_2024_attribution.png')
    # A paired seed plot makes ensemble-specific sensitivity visible.
    fig,axes=plt.subplots(1,3,figsize=(16,5),sharex=True,sharey=True)
    gaps=[]
    for ax,model in zip(axes,MODELS):
        sub=seed[seed.model==model].set_index('seed').loc[[7,19,42,61,97]]
        values=sub.gap.to_numpy()*100
        ensemble_gap=(metrics.loc[(model,'top5',1),'annualized_return']-metrics.loc[(model,'top10',1),'annualized_return'])*100
        gaps.extend([*values,ensemble_gap]); ax.scatter(values,np.arange(5),color='#0072b2',s=50)
        ax.scatter(ensemble_gap,5,color='#e7298a',marker='D',s=65)
        for i,value in enumerate([*values,ensemble_gap]): ax.annotate(f'{value:+.1f}',(value,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=9)
        ax.axvline(0,color='#444444',ls='--',lw=.8); ax.grid(axis='x',alpha=.2)
        ax.set_title(LABELS[model],fontsize=12); ax.set_xlabel('Top-5 minus top-10 CAGR (percentage points)')
    axes[0].set_yticks(range(6),['Seed 7','Seed 19','Seed 42','Seed 61','Seed 97','All-five-seed ensemble'])
    axes[0].invert_yaxis()
    for ax in axes: ax.set_xlim(min(gaps)-3,max(gaps)+4)
    fig.suptitle('Does the top-five advantage repeat across training seeds?',fontsize=14)
    fig.subplots_adjust(left=.14,right=.98,bottom=.19,top=.82,wspace=.15)
    fig.text(.06,.04,'Blue = individual trained seed; pink = equal-rank ensemble. Same market history; neither seeds nor models are independent market trials.',fontsize=10)
    save(fig,'seed_depth_sensitivity.png')
    main_rows=[]; yearly_rows=[]; depth_rows=[]; cost_rows=[]; seed_rows=[]; ranking_rows=[]; intervals=[]
    bucket_means=buckets.groupby('model')[['top5_observed_return','ranks6_10_observed_return','top10_observed_return']].mean()
    for model in MODELS:
        main_rows.append([LABELS[model]]+[pct(metrics.loc[(model,p,1),'annualized_return']) for p in ['top5','top10']])
        depth_rows.append([LABELS[model]]+[pct(metrics.loc[(model,p,1),'annualized_return']) for p in ['top5','ranks6_10','top10']]+
            [f'{100*bucket_means.loc[model,c]:.3f}%' for c in ['top5_observed_return','ranks6_10_observed_return']])
        for policy in ['top5','top10','ranks6_10']:
            annual=stock[(stock.model==model)&(stock.policy==policy)].groupby('year').year_return_contribution.sum()
            yearly_rows.append([LABELS[model],policy_names[policy]]+[pct(annual.loc[y]) for y in [2022,2023,2024,2025]])
        cost_rows.append([LABELS[model]]+[pct(metrics.loc[(model,p,c),'annualized_return']) for c in [1,0] for p in ['top5','top10']])
        s=seed[seed.model==model]
        seed_rows.append([LABELS[model],pct(s.top5_CAGR.median()),pct(s.top10_CAGR.median()),
            f'{int((s.gap>0).sum())}/5',f'{100*s.gap.median():+.2f} pp',
            f'{100*(metrics.loc[(model,"top5",1),"annualized_return"]-metrics.loc[(model,"top10",1),"annualized_return"]):+.2f} pp'])
        ranking_rows.append([LABELS[model]]+[f'{ranking.loc[model,c]:.4f}' for c in ['IC','RankIC','NDCG@5','NDCG@10']])
        row=paired.loc[model]; intervals.append([LABELS[model],f'{row.observed:+.3f}',f'[{row.low:+.3f}, {row.high:+.3f}]'])
    for control in ['QQQ','SPY']: main_rows.append([control,pct(benchmarks.loc[control,'annualized_return']),pct(benchmarks.loc[control,'annualized_return'])])
    random=periods[periods.model=='stockmixer']['Random NDCG@10'].mean()
    smci_total=st.loc[selected,'top5'].sum(); share=smci_total/st.top5.sum()
    attribution_rows=[[ticker,f'{100*st.loc[ticker,"top5"]:+.1f} pp',f'{100*st.loc[ticker,"top10"]:+.1f} pp',
        f'{100*(st.loc[ticker,"top5"]-st.loc[ticker,"top10"]):+.1f} pp'] for ticker in selected]
    attribution_rows.append(['All other stocks and their fees',f'{top5[-1]:+.1f} pp',f'{top10[-1]:+.1f} pp',f'{top5[-1]-top10[-1]:+.1f} pp'])
    text=f'''## Why top five and top ten differ

**Diagnostic analysis only: existing models and forecasts; no new training.**
The relevant robust variant is **THINK + mixer pseudo-Huber**, whose top-ten
CAGR is {pct(metrics.loc[('think_mix_pseudohuber','top10',1),'annualized_return'])}.
Ordinary THINK + mixer Huber is a separate model and returned 6.0% at top ten.

{table(['Model / benchmark','Top-5 net CAGR','Top-10 net CAGR'],main_rows)}

### Adding five stocks also halves the weight in the first five

Top five targets roughly 20% per selected stock; top ten targets roughly 10%.
For the same frozen ranking, the top-ten **signal weights** equal half the
top-five weights plus half the ranks-six-to-ten weights, including fractional
boundary ties. We replayed ranks six through ten as a fully invested diagnostic
basket using the same next-close orders, liquidity limits, costs and settlements.
Its returns expose what the additional selections contributed. Different account
values, compounding and execution mean its CAGR cannot simply be averaged with
top-five CAGR to reconstruct top ten.

{table(['Model','Top-5 CAGR','Ranks 6–10 CAGR','Top-10 CAGR','Mean 5-session return: ranks 1–5','Mean: ranks 6–10'],depth_rows)}

The last two columns are arithmetic averages of observed five-session labels,
before execution costs; the final partial holding period has no full label and
is excluded. They are diagnostics, not annualized or executable portfolio returns.

![Rank-depth portfolios with benchmark CAGR and Sharpe](rank_depth/figures/rank_depth_wealth.png)

**StockMixer:** ranks six through ten are weak overall and especially poor in
2022 and 2024. Expanding to ten both adds those positions and reduces exposure
to the profitable first five. Its top-five/top-ten gap is predominantly a
2024 concentration effect, not a uniformly repeated advantage in every year.

**THINK + mixer:** ranks six through ten are strong in this sample, especially
in 2023. Expanding to ten improves the portfolio; its first five alone are not
the strongest part of its ranking.

**THINK + mixer pseudo-Huber:** it has the same qualitative dilution problem
as StockMixer. Its ranks-six-to-ten basket loses money over the full period;
top ten remains near the ETFs because its profitable first five offset those
weaker additions. It falls from 22.4% to 11.4% CAGR, a substantial drop even
though the final level is close to SPY/QQQ.

### The calendar-year evidence

{table(['Model','Diagnostic portfolio','2022 net return','2023','2024','2025'],yearly_rows)}

StockMixer top five returned {pct(st.top5.sum())} in 2024 versus {pct(st.top10.sum())}
for top ten. SMCI, NVIDIA and Tesla together contributed
{100*smci_total:.1f} percentage points, about {100*share:.0f}% of the top-five account's
net 2024 gain. The contributions below sum to each account's calendar-year
return; they are **portfolio P&L contributions**, not each stock's own return
and not additive contributions to annualized CAGR.

{table(['Stock / remainder','Top-5 contribution','Top-10 contribution','Contribution to the gap'],attribution_rows)}

![StockMixer 2024 stock-level return attribution](rank_depth/figures/stockmixer_2024_attribution.png)

SMCI's largest selected holding windows included 2024-11-13 to 2024-11-20
(+26.9% adjusted-price return) and 2024-11-20 to 2024-11-27 (+35.9%). StockMixer
targeted 20% at top five. These were rebounds after its historical index entry,
not a claim that the strategy captured its early-2024 rally. The reviewed monthly
membership first permits SMCI in August 2024, following its July 22 index addition
([Nasdaq announcement](https://ir.nasdaq.com/node/107766/pdf)). Its October 1
10-for-1 split was already reflected in the adjusted series
([Nasdaq corporate-action notice](https://classic.nasdaqtrader.com/TraderNews.aspx?id=ECA2024-451)).

### This does not establish a stable top-five specialization

{table(['Model','Individual-seed median CAGR: top 5','Top 10','Seeds where top 5 wins','Median paired seed gap','Ensemble gap'],seed_rows)}

StockMixer's individual seeds have a much smaller median paired advantage than
the ensemble, and two of five favor top ten. Pseudo-Huber favors top five in
only two of five individual seeds. Regular THINK + mixer's ensemble favors ten,
while four of five individual seeds favor five. Averaging ranks creates a new
selected portfolio; it does not average individual-seed returns. These findings
make the exact optimal K sensitive to the particular ensemble and market path.

![Individual-seed and ensemble top-five/top-ten differences](rank_depth/figures/seed_depth_sensitivity.png)

### Neither transaction costs nor a ranking metric alone explains the result

{table(['Model','Top 5: base costs','Top 10: base costs','Top 5: zero costs','Top 10: zero costs'],cost_rows)}

The zero-cost counterfactual retains the same signals and liquidity caps. The
ordering survives with trading costs removed. StockMixer actually turns over
more at top five (55.6 times NAV annually) than top ten (48.4); costs narrow
its top-five advantage rather than create it.

{table(['Model','IC','RankIC','NDCG@5','NDCG@10'],ranking_rows)}

StockMixer has higher overall IC/RankIC than these THINK+mixer variants, while
its NDCG@10 ({ranking.loc['stockmixer','NDCG@10']:.4f}) is only slightly above the
matched random-ranking expectation ({random:.4f}). Neither fact establishes
reliable ordering specifically around the fifth/tenth cutoff. This study's NDCG
uses **return percentiles**, so an extreme 35.9% winner need not contribute more
than a modest winner with the same realized percentile. Portfolio profits depend
on return magnitudes, weights, timing and costs. Architecture and loss alone
cannot causally explain this particular sample's portfolio differences.

Paired top-five minus top-ten Sharpe intervals, using 1,000 circular resamples
of 20-session blocks, include zero for all three models:

{table(['Model','Observed Sharpe difference','Exploratory 95% interval'],intervals)}

These are Sharpe intervals, not CAGR intervals, and are not adjusted for the
many inspected models and sizes. The descriptive accounting explanation is
well established here; a permanent top-five advantage remains unproven.
Repeated selection among historical configurations can inflate observed performance
([Bailey and López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)).

### Verification and reproducibility

All 18 diagnostic account paths match the frozen production accounting engine.
An additional independent replay checks the 12 top-five/top-ten base- and
zero-cost paths. All six previously published base-cost paths remain unchanged.
Every stock-level dollar contribution reconciles to cumulative account return;
yearly contributions reconcile to calendar-year returns. No model was retrained
and no source forecast, membership, price or selected checkpoint was changed.

[Stock-level attribution](rank_depth/stock_attribution.csv) · [Attribution by year](rank_depth/year_stock_attribution.csv) ·
[Diagnostic curves](rank_depth/curves.csv) · [Diagnostic metrics](rank_depth/metrics.csv) ·
[Forecast buckets](rank_depth/forecast_buckets.csv) · [Paired seed results](rank_depth/seed_pairs.csv) ·
[Sharpe intervals](rank_depth/paired_bootstrap.csv) · [Reconciliation audit](rank_depth/AUDIT.json).
'''
    return text+'\n'+build_loss_similarity_section()


def verify(text):
    audit=json.loads((RUN/'AUDIT.json').read_text())
    assert audit['status']=='passed' and audit['diagnostic_paths_matched_frozen_production_engine']==18
    verified=[]
    for name,expected in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        assert digest(name)==expected,name
    for name in ['AUDIT.json','INPUT_CODE_LOCK.json','metrics.csv','curves.csv','stock_attribution.csv',
        'year_stock_attribution.csv','forecast_buckets.csv','paired_bootstrap.csv','seed_pairs.csv','year_concentration.csv']:
        assert (RUN/name).read_bytes()==(OUT/name).read_bytes(),name
        verified.append(OUT/name)
    metrics=pd.read_csv(OUT/'metrics.csv').set_index(['model','policy','cost_multiplier'])
    assert len(metrics)==18 and metrics.index.is_unique
    curves=pd.read_csv(OUT/'curves.csv')
    for key,sub in curves.groupby(['model','policy','cost_multiplier']):
        assert len(sub)==1003 and sub.date.is_unique and sub.date.is_monotonic_increasing
        from research_engine import performance
        result=performance(sub.net)
        for field in ['annualized_return','sharpe','maximum_drawdown','cumulative_return']:
            np.testing.assert_allclose(result[field],metrics.loc[key,field],atol=1e-10)
    attribution=pd.read_csv(OUT/'stock_attribution.csv')
    for (model,policy),sub in attribution.groupby(['model','policy']):
        np.testing.assert_allclose(sub.cumulative_return_contribution.sum(),metrics.loc[(model,policy,1),'cumulative_return'],atol=1e-10)
    for name in ['rank_depth_wealth.png','stockmixer_2024_attribution.png','seed_depth_sensitivity.png']:
        image=OUT/'figures'/name
        assert image.exists() and f'rank_depth/figures/{name}' in text
        verified.append(image)
    for phrase in ['Why top five and top ten differ','56.8 percentage points','ranks-six-to-ten basket loses money']:
        assert phrase in text,phrase
    print('Verified rank-depth diagnostics: 18 reconciled paths and three explanatory charts')
    return verified
