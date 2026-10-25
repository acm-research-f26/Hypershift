"""Supplement the complete model charts with loss, portfolio and frequency diagnostics."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report_robust_research import LABELS, draw_heatmap


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('runs/robust_comparison'))
    root=parser.parse_args().run
    pooled=pd.read_csv(root/'pooled_comparison.csv')
    annual=pd.read_csv(root/'annual_comparison.csv')
    all_metrics=pd.read_csv(root/'all_metrics.csv')
    out=root/'figures'
    entries=[]
    for h in [5,1]:
        sub=pooled[pooled.horizon==h]
        datasets=list(sub.dataset.unique())
        kinds=list(sub.model.unique())
        rankers=[k for k in kinds if k not in ['buy_hold','equal_universe','market_proxy']]
        for suffix,keys,titles,rows in [
            ('ranking',['IC','RankIC','NDCG@10'],['Full-universe IC','Full-universe RankIC','NDCG@10'],rankers),
            ('risk',['net_annualized_volatility','net_maximum_drawdown','turnover_annual'],['Annual volatility','Maximum drawdown','Annual turnover (x NAV)'],kinds),
            ('costs',['gross_sharpe','net_sharpe','delta_spy'],['Gross Sharpe','Net Sharpe','Net Sharpe minus SPY'],kinds)]:
            fig,axes=plt.subplots(1,3,figsize=(19,max(8,len(rows)*.40)))
            for ax,key,title in zip(axes,keys,titles):
                values=sub.pivot(index='model',columns='dataset',values=key).reindex(index=rows,columns=datasets).to_numpy()
                draw_heatmap(ax,values,[LABELS[k] for k in rows] if ax is axes[0] else ['']*len(rows),
                             [d.replace('_recent','') for d in datasets],title)
            fig.suptitle(f'{h}-session forecasts: four annual folds, 2022-2025',fontsize=16)
            fig.text(.02,.015,'Validation-selected policies; returns include modeled costs. Pooled folds omit annual gaps and reset capital. Public-data coverage bias remains.',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,.96));filename=f'{suffix}_h{h}.png'
            fig.savefig(out/filename,dpi=150);plt.close(fig)
            entries.append((f'{h}-session: {suffix}',filename))
        # Show every K and weighting for all rankers, rather than only winners.
        fixed=all_metrics[(all_metrics.horizon==h)&(all_metrics.seed=='ensemble')&(all_metrics.cost_multiplier==1)&
                          all_metrics.method.isin(['equal','confidence','inverse_vol','buffer_equal','buffer_inverse_vol'])&
                          all_metrics.k.isin([5,10,20])]
        for dataset in datasets:
            fig,axes=plt.subplots(1,5,figsize=(23,max(8,len(rankers)*.42)),sharey=True)
            for ax,method in zip(axes,['equal','confidence','inverse_vol','buffer_equal','buffer_inverse_vol']):
                values=fixed[(fixed.dataset==dataset)&(fixed.method==method)].groupby(['model','k']).net_sharpe.median().unstack().reindex(index=rankers,columns=[5,10,20]).to_numpy()
                draw_heatmap(ax,values,[LABELS[k] for k in rankers],[5,10,20],method)
                ax.set_xlabel('K')
            fig.suptitle(f'{dataset.replace("_recent", "")} | {h}-session | median annual net Sharpe across K and weights',fontsize=16)
            fig.text(.02,.015,'Sensitivity diagnostic, not a test-based selection menu. All settings were declared before evaluation. Colors are scaled within each weighting panel; compare numbers.',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,.95));filename=f'k_sensitivity_h{h}_{dataset}.png'
            fig.savefig(out/filename,dpi=145);plt.close(fig)
            entries.append((f'{dataset}, {h}-session: K and portfolio sensitivity',filename))
    # Fixed K10 loss/frequency comparisons plotted per year, preserving failed windows.
    pairs=pd.read_csv(root/'horizon_comparison.csv')
    for dataset in pairs.dataset.unique():
        sub=pairs[pairs.dataset==dataset]
        kinds=list(sub.model.unique())
        values=sub.pivot(index='model',columns='year',values='delta_sharpe_daily_minus_five').reindex(index=kinds,columns=[2022,2023,2024,2025]).to_numpy()
        fig,ax=plt.subplots(figsize=(12,8))
        draw_heatmap(ax,values,[LABELS[k] for k in kinds],[2022,2023,2024,2025],f'{dataset.replace("_recent", "")}: daily minus five-session net Sharpe')
        fig.text(.02,.015,'Fixed equal K10. Forecast horizon, rebalance interval and training batch differ. Own calendars; see shared-date diagnostic CSV.',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,1));filename=f'frequency_effect_{dataset}.png'
        fig.savefig(out/filename,dpi=160);plt.close(fig)
        entries.append((f'{dataset}: daily versus five-session',filename))
    # Stress costs by re-simulating original selected portfolios, not arithmetic subtraction.
    stresses=all_metrics[(all_metrics.seed=='ensemble')&all_metrics.selected_policy&all_metrics.cost_multiplier.isin([0,1,2,4])]
    fig,axes=plt.subplots(2,2,figsize=(17,12))
    for i,h in enumerate([5,1]):
        for j,dataset in enumerate(annual.dataset.unique()):
            ax=axes[i,j]
            sub=stresses[(stresses.horizon==h)&(stresses.dataset==dataset)]
            kinds=[k for k in sub.model.unique() if sub[sub.model==k].cost_multiplier.nunique()==4]
            for kind in kinds:
                g=sub[sub.model==kind].groupby('cost_multiplier').net_sharpe.median()
                ax.plot(g.index,g.values,marker='.',label=LABELS[kind],linewidth=1.2)
            spy=annual[(annual.horizon==h)&(annual.dataset==dataset)&(annual.model=='market_proxy')].net_sharpe.median()
            ax.axhline(spy,color='black',label='SPY median (base costs)',linewidth=2)
            ax.set_title(f'{dataset.replace("_recent", "")} | {h}-session');ax.set_xticks([0,1,2,4]);ax.grid(alpha=.2)
            ax.set_xlabel('Trading cost multiplier');ax.set_ylabel('Median annual net Sharpe')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.5,.01))
    fig.suptitle('Transaction-cost sensitivity: all ranked strategies',fontsize=17)
    fig.tight_layout(rect=(0,.19,1,.95));fig.savefig(out/'cost_stress.png',dpi=160);plt.close(fig)
    entries.append(('Transaction-cost stress','cost_stress.png'))
    charts_path=root/'CHARTS.md'
    content=charts_path.read_text(encoding='utf-8')
    marker='## Additional metric and sensitivity graphs'
    content=content.split(marker)[0].rstrip()+'\n\n'+marker+'\n\n'
    for title,filename in entries:
        content+=f'### {title}\n\n![{title}](figures/{filename})\n\n'
    charts_path.write_text(content,encoding='utf-8')
    images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',content)
    assert all((root/path).is_file() for path in images)
    inventory=json.loads((root/'line_chart_inventory.json').read_text())
    for (h,d,y),group in pd.DataFrame(inventory).groupby(['horizon','dataset','year']):
        expected=set(annual[(annual.horizon==h)&(annual.dataset==d)&(annual.year==y)].model)
        assert set(group.model)==expected
    (root/'graph_audit.json').write_text(json.dumps(dict(embedded_images=len(images),all_primary_models_plotted=True,
                             audited_primary_curves=len(inventory),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(f'Added {len(entries)} diagnostics; verified {len(images)} images and {len(inventory)} primary curves.')


if __name__=='__main__':
    main()
