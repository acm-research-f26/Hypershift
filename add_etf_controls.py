"""Add matched SPY/QQQ controls without changing any prediction or selection.

QQQ was explicitly requested after training began. This supplemental control is
not eligible for model or portfolio selection. Earlier locked analyses remain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research_engine import performance
from report_robust_research import LABELS, stratified_interval
from report_architecture_research import table
from run_architecture_research import write_json


def etf_path(prices, dates, symbol):
    gross=prices[symbol].pct_change(fill_method=None).reindex(pd.to_datetime(dates)).to_numpy()
    assert np.isfinite(gross).all()
    net=gross.copy()
    net[0]=(1+net[0])*(1-.0007)-1
    net[-1]=(1+net[-1])*(1-.0007)-1
    return pd.DataFrame(dict(date=list(dates),gross=gross,net=net))


def plot_panel(ax,paths,title,colors):
    for kind,path in paths.items():
        ax.plot(pd.to_datetime(path.date),np.cumprod(1+path.net),
                label=kind if kind in ['SPY','QQQ'] else LABELS[kind],
                color=colors[kind],linestyle='--' if kind=='QQQ' or 'huber' in kind else '-',
                linewidth=2.4 if kind in ['SPY','QQQ'] else 1.1)
    ax.set_title(title);ax.grid(alpha=.2);ax.set_ylabel('Net wealth per $1')
    ax.tick_params(axis='x',rotation=30,labelsize=8)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('runs/robust_comparison'))
    args=parser.parse_args();root=args.run
    prices=pd.read_csv('data/recent_cohorts/adj_close.csv',index_col=0,parse_dates=True)
    annual=pd.read_csv(root/'annual_comparison.csv')
    pooled=pd.read_csv(root/'pooled_comparison.csv')
    selected=pd.read_csv(root/'validation_selected_by_year.csv')
    joint=pd.read_csv(root/'combined_validation_selected.csv')
    lock=json.loads((root/'COMBINED_SELECTION_LOCK.json').read_text())
    roots={int(c['horizon']):Path(c['run']) for group in lock['candidates'].values() for c in group}
    out=root/'benchmarks';out.mkdir(exist_ok=True)
    palette=list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('Dark2').colors)
    colors={kind:palette[i] for i,kind in enumerate(LABELS)}
    colors.update(SPY='#111111',QQQ='#7513a3',buy_hold='#777777',equal_universe='#217a45')
    controls={};control_rows=[];model_rows=[];pooled_rows=[];series_inventory=[]
    for (h,d,y),group in annual.groupby(['horizon','dataset','year'],sort=False):
        folder=roots[h]/f'{d}_{y}'
        saved_spy=pd.read_csv(folder/'market_proxy_daily.csv')
        for symbol in ['SPY','QQQ']:
            path=etf_path(prices,saved_spy.date,symbol)
            if symbol=='SPY':
                np.testing.assert_allclose(path.net,saved_spy.net,rtol=1e-10,atol=1e-12)
            controls[(h,d,y,symbol)]=path
            path.to_csv(out/f'{symbol}_h{h}_{d}_{y}.csv',index=False)
            metrics={f'net_{k}':v for k,v in performance(path.net).items()}
            metrics.update({f'gross_{k}':v for k,v in performance(path.gross).items()})
            control_rows.append(dict(horizon=h,dataset=d,year=y,symbol=symbol,start=path.date.iloc[0],end=path.date.iloc[-1],
                                     turnover_annual=2*252/len(path),**metrics))
        qq_sr=performance(controls[(h,d,y,'QQQ')].net)['sharpe']
        for _,row in group.iterrows():
            model_rows.append(dict(row,qqq_sharpe=qq_sr,delta_qqq=row.net_sharpe-qq_sr,
                                   beats_both=bool(row.net_sharpe>max(qq_sr,row.spy_sharpe))))
    etfs=pd.DataFrame(control_rows);compare=pd.DataFrame(model_rows)
    etfs.to_csv(root/'etf_controls_by_year.csv',index=False)
    compare.to_csv(root/'model_vs_spy_qqq_by_year.csv',index=False)
    for _,row in pooled.iterrows():
        h,d,kind=row.horizon,row.dataset,row.model
        years=sorted(annual[(annual.horizon==h)&(annual.dataset==d)].year.unique())
        a=[pd.read_csv(roots[h]/f'{d}_{y}'/f'{kind}_daily.csv').net.to_numpy() for y in years]
        b=[controls[(h,d,y,'QQQ')].net.to_numpy() for y in years]
        qq=performance(np.concatenate(b));interval=stratified_interval(a,b)
        annual_sub=compare[(compare.horizon==h)&(compare.dataset==d)&(compare.model==kind)]
        pooled_rows.append(dict(row,qqq_sharpe=qq['sharpe'],delta_qqq=row.net_sharpe-qq['sharpe'],
                                years_beating_both=int(annual_sub.beats_both.sum()),qqq_delta_low=interval['low'],qqq_delta_high=interval['high']))
    pooled_compare=pd.DataFrame(pooled_rows)
    pooled_compare.to_csv(root/'pooled_model_vs_spy_qqq.csv',index=False)
    procedure_rows=[]
    for procedure,h in [('Five-session',5),('Daily',1),('Joint validation choice',None)]:
        frame=joint if h is None else selected[selected.horizon==h]
        for d,group in frame.groupby('dataset',sort=False):
            a=[];b=[];spy=[];wins=0
            for _,row in group.sort_values('year').iterrows():
                h0,y=int(row.horizon),int(row.year)
                a.append(pd.read_csv(roots[h0]/f'{d}_{y}'/f'{row.model}_daily.csv').net.to_numpy())
                b.append(controls[(h0,d,y,'QQQ')].net.to_numpy());spy.append(controls[(h0,d,y,'SPY')].net.to_numpy())
                wins+=int(performance(a[-1])['sharpe']>max(performance(b[-1])['sharpe'],performance(spy[-1])['sharpe']))
            am,bm,sm=performance(np.concatenate(a)),performance(np.concatenate(b)),performance(np.concatenate(spy))
            interval=stratified_interval(a,b)
            procedure_rows.append(dict(procedure=procedure,dataset=d,net_sharpe=am['sharpe'],spy_sharpe=sm['sharpe'],qqq_sharpe=bm['sharpe'],
                                      delta_spy=am['sharpe']-sm['sharpe'],delta_qqq=am['sharpe']-bm['sharpe'],years_beating_both=wins,
                                      qqq_delta_low=interval['low'],qqq_delta_high=interval['high']))
    procedures=pd.DataFrame(procedure_rows);procedures.to_csv(root/'selection_procedures_vs_spy_qqq.csv',index=False)
    # Regenerate every all-model overview/full-size panel with both ETFs.
    for h,run in roots.items():
        protocol=json.loads((run/'PROTOCOL.json').read_text())
        kinds=protocol['arguments']['variants']+['ridge','momentum126','reversal5','buy_hold','equal_universe']
        for d in annual.dataset.unique():
            years=sorted(annual.year.unique())
            fig,axes=plt.subplots(1,len(years),figsize=(22,8),sharey=True)
            for ax,y in zip(axes,years):
                paths={kind:pd.read_csv(run/f'{d}_{y}'/f'{kind}_daily.csv') for kind in kinds}
                paths.update({symbol:controls[(h,d,y,symbol)] for symbol in ['SPY','QQQ']})
                plot_panel(ax,paths,str(y),colors)
                series_inventory.extend(dict(horizon=h,dataset=d,year=int(y),series=kind) for kind in paths)
                panel,pax=plt.subplots(figsize=(15,7))
                plot_panel(pax,paths,f'{d.replace("_recent", "")} {y} | {h}-session | SPY and QQQ controls',colors)
                pax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=8.5,frameon=False)
                panel.text(.02,.015,'Original predictions and validation-selected policies unchanged. Black: SPY; purple dashed: QQQ. All curves include modeled costs.',fontsize=9)
                panel.tight_layout(rect=(0,.05,1,1));panel.savefig(root/'figures'/f'panel_h{h}_{d}_{y}.png',dpi=150);plt.close(panel)
            handles,labels=axes[0].get_legend_handles_labels()
            fig.suptitle(f'{d.replace("_recent", " cohort")} | {h}-session | all models plus SPY and QQQ',fontsize=18,y=.98)
            fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=9,bbox_to_anchor=(.5,.035),frameon=False)
            fig.subplots_adjust(left=.05,right=.99,top=.89,bottom=.30,wspace=.10)
            fig.text(.02,.015,'Matched dates and estimated costs. Separate annual accounts; public-data coverage bias remains. Black: SPY; purple dashed: QQQ.',fontsize=9)
            fig.savefig(root/'figures'/f'all_models_h{h}_{d}.png',dpi=160);plt.close(fig)
    # Dedicated procedure chart includes both controls, avoiding hindsight model choice.
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    for i,h in enumerate([5,1]):
        for j,d in enumerate(annual.dataset.unique()):
            group=selected[(selected.horizon==h)&(selected.dataset==d)].sort_values('year')
            ax=axes[i,j]
            for label,color,style in [('Validation-selected','#256ea0','-'),('SPY','#111111','-'),('QQQ','#7513a3','--')]:
                pieces=[]
                for _,row in group.iterrows():
                    pieces.append(pd.read_csv(roots[h]/f'{d}_{row.year}'/f'{row.model}_daily.csv') if label=='Validation-selected' else controls[(h,d,int(row.year),label)])
                p=pd.concat(pieces,ignore_index=True)
                ax.plot(pd.to_datetime(p.date),np.cumprod(1+p.net),label=label,color=color,linestyle=style)
            ax.set_title(f'{d.replace("_recent", "")} | {h}-session');ax.set_ylabel('Net fold-compounded wealth');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Validation-selected procedures versus SPY and QQQ, 2022-2025',fontsize=17)
    fig.text(.02,.015,'Annual accounts reset; untested boundary sessions omitted. QQQ is an added control and never influenced training or selection.',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(root/'figures'/'selected_procedures.png',dpi=160);plt.close(fig)
    sections=['## SPY and QQQ controls','',
              'QQQ was added at the user\'s request after the training batch began. It is a comparison control, never a model-selection candidate. '
              'Both ETFs use frozen adjusted prices, the exact model evaluation dates, and 7bp entry plus 7bp exit costs. '
              'ETF expenses are already reflected in market prices. Sharpe uses the same fixed 3% hurdle.','',
              '### Four-year selection procedures','',
              table(procedures,[('procedure','Procedure'),('dataset','Cohort'),('net_sharpe','Model net SR'),('spy_sharpe','SPY SR'),('qqq_sharpe','QQQ SR'),
                                ('delta_spy','Minus SPY'),('delta_qqq','Minus QQQ'),('years_beating_both','Years > both')]),'',
              '### ETF controls for every annual window','',
              table(etfs,[('horizon','Horizon'),('dataset','Cohort'),('year','Year'),('symbol','ETF'),('net_sharpe','Net SR'),
                          ('net_cumulative_return','Net return'),('net_annualized_volatility','Ann. vol'),('net_maximum_drawdown','Max DD')],
                         ['net_cumulative_return','net_annualized_volatility','net_maximum_drawdown']),'',
              '[Every model/year versus both ETFs](model_vs_spy_qqq_by_year.csv) | '
              '[Every pooled model versus both ETFs, including QQQ intervals](pooled_model_vs_spy_qqq.csv) | '
              '[Selection procedures](selection_procedures_vs_spy_qqq.csv)','',
              'QQQ tracks a different, more concentrated investment universe than SPY. Neither ETF is a matched-security benchmark for our small historical cohorts. '
              'Pooled intervals are exploratory and unadjusted for repeated research.','']
    text='\n'.join(sections);(root/'ETF_CONTROLS.md').write_text(text,encoding='utf-8')
    report=(root/'REPORT.md').read_text(encoding='utf-8')
    marker='## SPY and QQQ controls'
    if marker in report: report=report[:report.index(marker)].rstrip()+'\n\n'
    (root/'REPORT.md').write_text(report+'\n'+text,encoding='utf-8')
    charts=(root/'CHARTS.md').read_text(encoding='utf-8')
    note='Both SPY (black) and QQQ (purple dashed) are now included in every all-model overview/full-size panel and in the selected-procedure chart. '
    if note not in charts:
        charts=charts.replace('\n\n','\n\n'+note+'[Benchmark tables](ETF_CONTROLS.md). The focused loss-ablation charts retain SPY as their original reference.\n\n',1)
    (root/'CHARTS.md').write_text(charts,encoding='utf-8')
    write_json(root/'etf_control_inventory.json',series_inventory)
    write_json(root/'etf_control_provenance.json',dict(source='data/recent_cohorts/adj_close.csv',sha256=hashlib.sha256(Path('data/recent_cohorts/adj_close.csv').read_bytes()).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reason='User requested both ETFs after training began; no selection changes.'))
    update_historical_charts(colors,prices)
    print(procedures.round(3).to_string(index=False))


def update_historical_charts(colors,modern_prices):
    """Keep the requested 2017 | 2024 | 2025 comparison, now with both ETFs."""
    root=Path('runs/recent_2024_2025')
    old=Path('runs/architecture_research')
    historical_prices=pd.read_csv('data/research_cohorts/benchmark.csv',index_col=0,parse_dates=True)
    kinds=list(LABELS)[:14]  # All original kinds except market_proxy, replaced with explicit ETFs.
    assert 'market_proxy' not in kinds and len(kinds)==14
    records=[]
    for cohort in ['NYSE','NASDAQ']:
        fig,axes=plt.subplots(1,3,figsize=(20,8),sharey=True)
        for ax,year in zip(axes,[2017,2024,2025]):
            source=old if year==2017 else root;dataset=cohort if year==2017 else cohort+'_recent'
            folder=source/f'{dataset}_{year}'
            paths={kind:pd.read_csv(folder/f'{kind}_daily.csv') for kind in kinds}
            dates=paths['think'].date
            for symbol in ['SPY','QQQ']:
                paths[symbol]=etf_path(historical_prices if year==2017 else modern_prices,dates,symbol)
                records.append(dict(dataset=dataset,year=year,symbol=symbol,**performance(paths[symbol].net)))
            saved=pd.read_csv(folder/'market_proxy_daily.csv')
            original_symbol='QQQ' if cohort=='NASDAQ' and year==2017 else 'SPY'
            np.testing.assert_allclose(paths[original_symbol].net,saved.net,atol=1e-12,rtol=1e-10)
            plot_panel(ax,paths,str(year),colors)
            ax.set_xlabel(f'{dates.iloc[0]} to {dates.iloc[-1]}')
        fig.suptitle(f'{cohort}: saved 2017 | 2024 | 2025 with SPY and QQQ',fontsize=18,y=.96)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9,bbox_to_anchor=(.5,.045),frameon=False)
        fig.subplots_adjust(left=.06,right=.985,top=.86,bottom=.31,wspace=.08)
        fig.text(.03,.02,'Original model curves unchanged. Black: SPY; purple dashed: QQQ. Costs included; annual accounts reset.\n'
                 'Old stock dividend conventions and modern public coverage differ; this is not a controlled year-only comparison.',fontsize=9)
        fig.savefig(root/'figures'/f'{cohort}_2017_vs_2024_2025.png',dpi=170);plt.close(fig)
    pd.DataFrame(records).to_csv(root/'historical_spy_qqq_controls.csv',index=False)
    path=root/'CHARTS.md';text=path.read_text(encoding='utf-8')
    note='The 2017 | 2024 | 2025 panels now show **both SPY and QQQ**, alongside all original model/strategy curves. '
    if note not in text:
        text=text.replace('## Saved 2017 results beside the modern tests','## Saved 2017 results beside the modern tests\n\n'+note+
                          'SPY is black and QQQ purple dashed. The original benchmark choice described below remains the reference for the saved metrics. '
                          '[New 2022-2025 daily and five-session experiments](../robust_comparison/CHARTS.md).')
    path.write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
