"""Render the audited top-K/pseudo-Huber follow-up into the same charts page."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from run_ndx_topk_pseudohuber import RUN
from pseudohuber_think_mixer import KIND
from chart_performance_labels import performance_label, LEGEND_TITLE

OUT=Path('docs/nasdaq_monthly_pit/topk_pseudohuber')


def table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def pct(x):
    return f'{100*x:.1f}%'


def build_section(original_names,original_colors):
    if not (RUN/'AUDIT.json').exists():
        return ''
    audit=json.loads((RUN/'AUDIT.json').read_text());assert audit['status']=='passed'
    assert audit['new_checkpoints_reproduced']==20 and audit['independent_share_cash_paths']==204
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    for filename in ['metrics.csv','curves.csv','annual.csv','monthly.csv','ranking.csv','ranking_periods.csv',
        'ranking_by_year.csv','paired_bootstrap.csv','training_records.csv','PROTOCOL.json','completion.json',
        'AUDIT.json','INPUT_CODE_LOCK.json','SOURCE_ARTIFACT_LOCK.json','FITS_COMPLETE_LOCK.json','VALIDATION_EXPORT_CORRECTION.json']:
        shutil.copyfile(RUN/filename,OUT/filename)
    names={**original_names,KIND:'THINK + mixer pseudo-Huber'}
    colors={**original_colors,KIND:'#e7298a'}
    order=list(original_names)[:9]+[KIND]+list(original_names)[9:]
    neural=list(original_names)[:6]+[KIND]
    read=lambda file:pd.read_csv(RUN/file,dtype={'seed':str})
    metrics=read('metrics.csv');curves=read('curves.csv');annual=read('annual.csv')
    ranking=read('ranking.csv').set_index('model');paired=read('paired_bootstrap.csv')
    periods=read('ranking_periods.csv')
    primary=metrics[(metrics.seed=='ensemble')&(metrics.cost_multiplier==1)].set_index(['model','k'])
    data=curves[(curves.seed=='ensemble')&(curves.cost_multiplier==1)].copy()
    data['year']=pd.to_datetime(data.date).dt.year
    years=annual[(annual.seed=='ensemble')&(annual.cost_multiplier==1)].set_index(['model','k','year'])
    metric=lambda model,k:primary.loc[(model,10 if model in ['SPY','QQQ','monthly_basket'] else k)]
    path=lambda model,k:data[(data.model==model)&(data.k==(10 if model in ['SPY','QQQ','monthly_basket'] else k))]
    year_metric=lambda model,k,year:years.loc[(model,10 if model in ['SPY','QQQ','monthly_basket'] else k,year)]
    def save(fig,filename):
        fig.savefig(OUT/'figures'/filename,dpi=165,bbox_inches='tight',facecolor='white');plt.close(fig)
    def line(ax,dates,wealth,model,label=None):
        return ax.plot(dates,wealth,color=colors[model],label=label,lw=2.3 if model in [KIND,'think_mix','SPY','QQQ'] else 1.6,
                       ls='-.' if model==KIND else '--' if 'mix' in model or model=='QQQ' else '-')
    ks=[5,10,20]
    wealth={(model,k):np.r_[1.,np.cumprod(1+path(model,k).net.to_numpy())] for k in ks for model in order}
    lo=min(x.min() for x in wealth.values());hi=max(x.max() for x in wealth.values());pad=(hi-lo)*.04
    year_paths={}
    for model in order:
        for k in ks:
            np.testing.assert_allclose(wealth[model,k][-1],1+metric(model,k).cumulative_return,atol=1e-12)
            for year in [2022,2023,2024,2025]:
                sub=path(model,k);sub=sub[sub.year==year]
                w=np.r_[1.,np.cumprod(1+sub.net.to_numpy())]
                np.testing.assert_allclose(w[-1],1+year_metric(model,k,year).cumulative_return,atol=1e-12)
                year_paths[model,k,year]=(sub,w)
            np.testing.assert_allclose(np.prod([year_paths[model,k,y][1][-1] for y in [2022,2023,2024,2025]]),wealth[model,k][-1],atol=1e-12)
    ylo=min(x[1].min() for x in year_paths.values());yhi=max(x[1].max() for x in year_paths.values());ypad=(yhi-ylo)*.04
    charts=[]
    for k in ks:
        fig,ax=plt.subplots(figsize=(16,8))
        for model in order:
            sub=path(model,k);dates=pd.DatetimeIndex([pd.Timestamp('2022-01-01'),*pd.to_datetime(sub.date)])
            line(ax,dates,wealth[model,k],model,performance_label(names[model],metric(model,k)))
        for year in [2023,2024,2025]:
            ax.axvline(pd.Timestamp(year,1,1),color='#666666',lw=.7,ls=':',alpha=.4)
        ax.set(title=f'Historical monthly NASDAQ-100 | equal top {k} | 5-session forecasts',
            ylabel='Net wealth per initial $1',ylim=(lo-pad,hi+pad),xlim=(pd.Timestamp('2022-01-01'),pd.Timestamp('2025-12-31')))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(alpha=.2);ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=9.5,
                                  title=LEGEND_TITLE,title_fontsize=10)
        fig.subplots_adjust(left=.07,right=.68,bottom=.16,top=.90)
        fig.text(.07,.08,f'Equal top {k}: {100/k:g}% target per selected security; five-seed rank ensembles; rebalance every five sessions.',fontsize=10)
        fig.text(.07,.045,'Continuous 2022–2025 accounts and modeled costs. SPY, QQQ and the monthly basket keep their control policies.',fontsize=10)
        save(fig,f'wealth_top{k}.png')
        fig,axes=plt.subplots(2,2,figsize=(16,9))
        handles=[]
        for ax,year in zip(axes.flat,[2022,2023,2024,2025]):
            for model in order:
                sub,w=year_paths[model,k,year];dates=pd.DatetimeIndex([pd.Timestamp(year,1,1),*pd.to_datetime(sub.date)])
                drawn=line(ax,dates,w,model,names[model])
                if year==2022:
                    handles.extend(drawn)
            ax.set(title=str(year),ylim=(ylo-ypad,yhi+ypad),xlim=(pd.Timestamp(year,1,1),pd.Timestamp(year,12,31)))
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.axhline(1,color='#666666',lw=.7,ls=':',alpha=.6);ax.grid(alpha=.2)
        axes[0,0].set_ylabel('Net wealth per $1 at year start');axes[1,0].set_ylabel('Net wealth per $1 at year start')
        fig.legend(handles,[performance_label(names[x],metric(x,k)) for x in order],loc='center left',bbox_to_anchor=(.71,.5),frameon=False,fontsize=9,
                   title='2022–2025 net CAGR | annualized SR')
        fig.suptitle(f'Equal top {k} | all models by calendar year | same vertical scale across every K and year',fontsize=14)
        fig.subplots_adjust(left=.07,right=.70,bottom=.10,top=.90,hspace=.24,wspace=.17)
        fig.text(.07,.035,'Year panels are rebased to $1; legend CAGR/SR describe the full 2022–2025 account. Modeled costs included.',fontsize=10)
        save(fig,f'years_top{k}.png')
        rows=[[names[model]]+[pct(year_metric(model,k,y).cumulative_return) for y in [2022,2023,2024,2025]] for model in order]
        charts.append(f'### Equal top {k}\n\n![All models: equal top {k}](topk_pseudohuber/figures/wealth_top{k}.png)\n\n'
                      f'![Year-by-year: equal top {k}](topk_pseudohuber/figures/years_top{k}.png)\n\n'
                      +table(['Model','2022 net return','2023','2024','2025'],rows))
    seeds=metrics[(metrics.seed!='ensemble')&(metrics.cost_multiplier==1)]
    fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True,sharex=True)
    seedrows=[]
    for ax,k in zip(axes,ks):
        for i,model in enumerate(neural):
            sub=seeds[(seeds.model==model)&(seeds.k==k)];values=sub.annualized_return.to_numpy()*100
            ax.plot([values.min(),values.max()],[i,i],color=colors[model],lw=3,alpha=.4)
            ax.scatter(values,np.linspace(i-.1,i+.1,len(values)),color=colors[model],s=30)
            ax.scatter(metric(model,k).annualized_return*100,i,marker='D',facecolors='white',edgecolors=colors[model],s=60)
            seedrows.append([names[model],k,pct(sub.annualized_return.min()),pct(sub.annualized_return.median()),
                pct(sub.annualized_return.max()),f'{int((sub.annualized_return>metric("QQQ",k).annualized_return).sum())}/5'])
        for model in ['SPY','QQQ']:
            ax.axvline(metric(model,k).annualized_return*100,color=colors[model],ls='--',label=performance_label(model,metric(model,k)))
        ax.set(title=f'Equal top {k}',xlabel='Net CAGR (%)');ax.grid(axis='x',alpha=.2);ax.legend(fontsize=9)
    axes[0].set_yticks(range(len(neural)),[names[x] for x in neural]);axes[0].invert_yaxis()
    fig.suptitle('Every training seed | hollow diamond = all-five-seed rank ensemble');fig.tight_layout()
    save(fig,'seed_stability.png')
    summaryrows=[[names[model]]+[f'{pct(metric(model,k).annualized_return)} / {metric(model,k).sharpe:.3f} / {pct(metric(model,k).maximum_drawdown)}' for k in ks] for model in order]
    uncertaintyrows=[[int(r.k),names[r.comparison],f'{r.observed:.3f}',f'[{r.low:.3f}, {r.high:.3f}]'] for r in paired.itertuples()]
    rankrows=[[names[model]]+[f'{ranking.loc[model,field]:.4f}' for field in ['IC','RankIC','ICIR','RankICIR','NDCG@5','NDCG@10','NDCG@20']]
              for model in ['think_mix','think_mix_huber',KIND]]
    r=ranking.loc[KIND];chance=periods[periods.model==KIND]['Random NDCG@10'].mean()
    php=metric(KIND,10);mse=metric('think_mix',10);huber=metric('think_mix_huber',10)
    ph5=metric(KIND,5);seed5=seeds[(seeds.model==KIND)&(seeds.k==5)]
    lead5=paired[(paired.k==5)&(paired.comparison=='QQQ')].iloc[0]
    concentration=(f'Pseudo-Huber top 5 has {pct(ph5.annualized_return)} CAGR but a {pct(ph5.maximum_drawdown)} '
        f'maximum drawdown. Individual seed CAGRs range from {pct(seed5.annualized_return.min())} to '
        f'{pct(seed5.annualized_return.max())}, with median {pct(seed5.annualized_return.median())}; '
        f'{int((seed5.annualized_return>metric("QQQ",5).annualized_return).sum())}/5 seeds beat QQQ. '
        f'Its Sharpe advantage over QQQ has exploratory 95% interval [{lead5.low:.3f}, {lead5.high:.3f}]. '
        'An ensemble combines rankings and its return need not equal the mean or median individual-seed return. '
        'The larger ensemble return is not evidence of reliable performance across training seeds.')
    comparison=(f'At the unchanged top-10 policy, pseudo-Huber has net CAGR {pct(php.annualized_return)}, '
                f'Sharpe {php.sharpe:.3f}, and maximum drawdown {pct(php.maximum_drawdown)}. '
                f'MSE THINK + mixer has {pct(mse.annualized_return)} CAGR and {mse.sharpe:.3f} Sharpe; '
                f'Huber THINK + mixer has {pct(huber.annualized_return)} CAGR and {huber.sharpe:.3f} Sharpe. '
                'Changing K changes concentration and turnover, not the forecasts or fitted coefficients. '
                'All three sizes are reported; none is selected as a new primary strategy from these results.')
    leaders=[]
    for k in ks:
        best=max(order,key=lambda model:metric(model,k).annualized_return)
        leaders.append(f'top {k}: {names[best]} ({pct(metric(best,k).annualized_return)} CAGR)')
    leader_text='Highest observed CAGR at each size: '+ '; '.join(leaders)+'. These are descriptive leaders, not statistically established winners.'
    costrows=[]
    for k in ks:
        sub=metrics[(metrics.model==KIND)&(metrics.seed=='ensemble')&(metrics.k==k)].set_index('cost_multiplier')
        costrows.append([k]+[pct(sub.loc[mult,'annualized_return']) for mult in [1,2,4]])
    return f'''
## Top-5/top-10/top-20 and pseudo-Huber follow-up

**Completed: 20 additional pseudo-Huber fits; 204 compared portfolio paths independently audited.**
The user substituted the already-available top 20 for top 15. All 120 parent
paths are unchanged; 84 additional paths cover the new loss and the old neural
models' individual seeds at top 5 and top 20. No old model was refitted.

The new **THINK + mixer pseudo-Huber** uses exactly the existing mixer
architecture, features, purged train/validation folds, five seeds and training
settings. Pseudo-Huber is a smooth robust loss: near small errors it behaves
like squared error; large errors have bounded gradient influence. Its fixed
scale is **one percentage point** of return, matching the previous Huber scale.
We use twice the usual pseudo-Huber loss to match MSE curvature near zero:
`2 × δ² × (sqrt(1 + (error/δ)²) − 1)`. Epoch selection still uses prior-year
validation MSE. All 20 fits were locked before these new financial metrics.

Top 5 targets about **20% per name**, top 10 **10%**, and top 20 **5%**. Rebalancing
remains every **five sessions**, with frozen signal-time orders filling at the
next close, the same cash account, costs and liquidity limits. ETF and monthly
basket control policies stay the same at every K.

{comparison}

{leader_text}

{concentration}

Cells below show **net CAGR / Sharpe / maximum drawdown**:

{table(['Model','Top 5','Top 10','Top 20'],summaryrows)}

All-model wealth charts share one linear vertical scale across K. The annual
charts rebase the continuous accounts to $1 for display, with one vertical
scale across all years and sizes. They do not restart trading accounts.

{(chr(10)*2).join(charts)}

### Training-seed stability across portfolio sizes

![Every seed at every size](topk_pseudohuber/figures/seed_stability.png)

{table(['Model','K','Seed CAGR min','Median','Max','Seeds CAGR > QQQ'],seedrows)}

Five seeds share the same market history; their range measures sensitivity to
training initialization, not five independent market trials. Full 2×/4× cost
stress results remain visible in the linked metrics file.

Pseudo-Huber ensemble net CAGR under the declared cost stresses:

{table(['K','Base costs','2× costs','4× costs'],costrows)}

### Pseudo-Huber forecast quality and uncertainty

{table(['Model','IC','RankIC','ICIR','RankICIR','NDCG@5','NDCG@10','NDCG@20'],rankrows)}

The same forecasts drive every K, so IC and RankIC do not change merely because
we buy a different number of names. These diagnostics average 200 full
five-session periods. Pseudo-Huber's NDCG@10 is {r['NDCG@10']:.4f}, versus
{chance:.4f} expected for random ordering of the same observed stocks. NDCG is
relative ranking quality, not percentage accuracy or net portfolio return.

Exploratory **pseudo-Huber minus comparator** Sharpe differences:

{table(['K','Comparator','Sharpe difference','95% paired interval'],uncertaintyrows)}

Intervals use the same 1,000 paired circular 20-session block resamples as the
parent study and are not adjusted for multiple comparisons. An interval that
includes zero does not establish an advantage. The 2022–2025 years were already
inspected before this user-requested follow-up: this is exploratory research,
not fresh confirmation. Original public-data, membership and settlement limits
remain. No delta, epoch, seed or K was selected by these test returns.

The extension audit reproduced all 20 checkpoints and validation epochs,
independently checked ranking metrics for ten signals, and replayed all 204
share/cash paths. It confirmed all 120 parent paths unchanged; the largest NAV
difference was ${audit['largest_nav_difference_dollars']:.8f}.

The audit caught a validation-export issue: the originally saved validation
arrays were from the final epoch, while the checkpoint and test forecasts were
from the selected best-validation epoch. Selected-checkpoint validation arrays
were regenerated separately, matched to the recorded best validation MSE,
and all test forecasts reproduced unchanged. Original fit files and hashes
are preserved. No model was retrained or selected by test returns to make this
correction. [Validation export correction](topk_pseudohuber/VALIDATION_EXPORT_CORRECTION.json).

[Extension protocol](topk_pseudohuber/PROTOCOL.json) · [Independent audit](topk_pseudohuber/AUDIT.json) ·
[All sizes, seeds and cost stresses](topk_pseudohuber/metrics.csv) · [All dated paths](topk_pseudohuber/curves.csv) ·
[Annual returns](topk_pseudohuber/annual.csv) · [Monthly returns](topk_pseudohuber/monthly.csv) ·
[All forecast diagnostics](topk_pseudohuber/ranking.csv) · [Forecast diagnostics by year](topk_pseudohuber/ranking_by_year.csv) ·
[Every forecast period](topk_pseudohuber/ranking_periods.csv) · [Paired intervals](topk_pseudohuber/paired_bootstrap.csv) ·
[Twenty new fits](topk_pseudohuber/training_records.csv) · [Input/code lock](topk_pseudohuber/INPUT_CODE_LOCK.json) ·
[Parent artifact lock](topk_pseudohuber/SOURCE_ARTIFACT_LOCK.json) · [Fit-completion lock](topk_pseudohuber/FITS_COMPLETE_LOCK.json) ·
[Completion record](topk_pseudohuber/completion.json)

'''
