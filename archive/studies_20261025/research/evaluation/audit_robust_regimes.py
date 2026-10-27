"""Signal-time regime diagnostics and data/execution-quality disclosure.

Descriptive checks only: never change a model, portfolio or selection lock.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_engine import load_panel, prepare_panel, fold_indices, performance
from report_architecture_research import table


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('runs/robust_comparison'))
    root=parser.parse_args().run
    annual=pd.read_csv(root/'annual_comparison.csv')
    choices=json.loads((root/'COMBINED_SELECTION_LOCK.json').read_text())['candidates']
    roots={int(c['horizon']):Path(c['run']) for group in choices.values() for c in group}
    records=[]
    for (h,dataset),group in annual.groupby(['horizon','dataset'],sort=False):
        data=prepare_panel(load_panel(dataset),horizon=int(h))
        p=data['panel']
        b=pd.Series(p.benchmark,index=p.dates)
        risk=b.pct_change(fill_method=None).rolling(20).std().to_numpy()
        trend=(b/b.shift(126)-1).to_numpy()
        for year in sorted(group.year.unique()):
            train,_,test=fold_indices(data,int(year))
            threshold=np.nanmedian(risk[data['origin'][train]])
            tags=np.array([('rising' if trend[t]>=0 else 'falling')+'_'+('high_vol' if risk[t]>=threshold else 'low_vol')
                           for t in data['origin'][test]])
            folder=roots[h]/f'{dataset}_{year}'
            spy=pd.read_csv(folder/'market_proxy_daily.csv').net.to_numpy()
            qqq=pd.read_csv(root/'benchmarks'/f'QQQ_h{h}_{dataset}_{year}.csv').net.to_numpy()
            for _,row in group[group.year==year].iterrows():
                values=pd.read_csv(folder/f'{row.model}_daily.csv').net.to_numpy()
                ranking=folder/f'{row.model}_ranking.csv'
                rank=pd.read_csv(ranking) if ranking.exists() else None
                for tag in sorted(set(tags)):
                    period_mask=tags==tag;mask=np.repeat(period_mask,int(h))
                    if mask.sum()<2:continue
                    metrics=performance(values[mask])
                    records.append(dict(horizon=h,dataset=dataset,year=year,model=row.model,regime=tag,
                                        periods=int(period_mask.sum()),sessions=int(mask.sum()),conditional_sharpe=metrics['sharpe'],
                                        spy_conditional_sharpe=performance(spy[mask])['sharpe'],qqq_conditional_sharpe=performance(qqq[mask])['sharpe'],
                                        IC=rank.IC[period_mask].mean() if rank is not None else None,
                                        RankIC=rank.RankIC[period_mask].mean() if rank is not None else None))
    regimes=pd.DataFrame(records);regimes.to_csv(root/'regime_diagnostics.csv',index=False)
    # Counts refer to each account independently and must not be summed as unique events.
    quality=annual[annual.model!='market_proxy'].groupby(['horizon','dataset','year']).agg(
        models=('model','size'),max_stale_position_marks=('stale_position_marks','max'),
        max_terminal_liquidation_over_cap=('terminal_liquidation_over_cap','max'),
        max_constrained_orders=('constrained_orders','max'),max_unobserved_labels=('unobserved_labels','max'),
        eligible_min=('eligible_min','min'),eligible_max=('eligible_max','max')).reset_index()
    quality.to_csv(root/'execution_quality_by_year.csv',index=False)
    source=Path(__file__)
    (root/'regime_diagnostic_provenance.json').write_text(json.dumps(dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         definition='126-session SPY trend sign, 20-session SPY volatility relative to training-period median; signal-time labels; no model selection'),indent=2))
    text=['## Regime and execution-quality checks','',
          'Regimes use information available at each signal: the sign of SPY\'s trailing 126-session return and its trailing '
          '20-session volatility relative to a training-only median. [Detailed results](regime_diagnostics.csv) compare every model '
          'with both ETFs within each regime. Conditional Sharpe values use noncontiguous subsets and small samples; they are not '
          'tradable regime-switching portfolios or additional model-selection evidence.','',
          'The table records worst-case counts across the separately tested model accounts, not unique market events. '
          'Stale position marks indicate missing contemporaneous prices for held securities. Terminal liquidation is costed but '
          'may exceed the daily participation cap; that is a simulator limitation. Unobserved labels are excluded only from '
          'ranking-metric calculation, not retrospectively from signal-time eligibility.','',
          table(quality,[('horizon','Horizon'),('dataset','Cohort'),('year','Year'),('eligible_min','Min eligible'),('eligible_max','Max eligible'),
                         ('max_stale_position_marks','Max stale marks'),('max_unobserved_labels','Missing labels'),
                         ('max_terminal_liquidation_over_cap','Terminal orders > cap'),('max_constrained_orders','Constrained orders')]),'']
    if quality.max_stale_position_marks.max()>0:
        text.extend(['**Some accounts contain stale held-price marks. Their returns are conditional on carried-forward valuation and cannot be treated as fully verified realizable returns.**',''])
    report=root/'REPORT.md';content=report.read_text(encoding='utf-8')
    marker='## Regime and execution-quality checks'
    if marker in content:content=content[:content.index(marker)].rstrip()
    report.write_text(content+'\n\n'+'\n'.join(text),encoding='utf-8')
    print(f'Saved {len(regimes)} regime rows; max stale marks {quality.max_stale_position_marks.max():.0f}; max missing labels {quality.max_unobserved_labels.max():.0f}')


if __name__=='__main__':
    main()
