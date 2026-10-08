"""Summarize locked outcomes; identify exploratory candidates, never retune them."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_engine import performance
from report_robust_research import LABELS, stratified_interval
from report_architecture_research import table
from run_architecture_research import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('runs/robust_comparison'))
    args=parser.parse_args();root=args.run
    annual=pd.read_csv(root/'annual_comparison.csv')
    pooled=pd.read_csv(root/'pooled_comparison.csv')
    selected=pd.read_csv(root/'validation_selected_by_year.csv')
    joint=pd.read_csv(root/'combined_validation_selected.csv')
    meta=json.loads((root/'COMBINED_SELECTION_LOCK.json').read_text())['selected']
    roots={int(choice['horizon']):Path(choice['run']) for choices in
           json.loads((root/'COMBINED_SELECTION_LOCK.json').read_text())['candidates'].values() for choice in choices}
    rows=[]
    for procedure,h in [('Five-session',5),('Daily',1),('Joint validation choice',None)]:
        chosen=joint if h is None else selected[selected.horizon==h]
        for dataset,group in chosen.groupby('dataset',sort=False):
            a,b=[],[];gross=[]
            for _,row in group.sort_values('year').iterrows():
                folder=roots[int(row.horizon)]/f'{dataset}_{row.year}'
                path=pd.read_csv(folder/f'{row.model}_daily.csv')
                spy=pd.read_csv(folder/'market_proxy_daily.csv')
                assert path.date.equals(spy.date)
                a.append(path.net.to_numpy());b.append(spy.net.to_numpy());gross.extend(path.gross)
            metrics=performance(np.concatenate(a));spy_metrics=performance(np.concatenate(b))
            interval=stratified_interval(a,b)
            rows.append(dict(procedure=procedure,dataset=dataset,**metrics,gross_sharpe=performance(gross)['sharpe'],
                             spy_sharpe=spy_metrics['sharpe'],delta_spy=metrics['sharpe']-spy_metrics['sharpe'],
                             years_beating_spy=int((group.delta_spy>0).sum()),turnover_annual=np.average(group.turnover_annual,weights=group.net_days),
                             **interval))
    procedures=pd.DataFrame(rows)
    procedures.to_csv(root/'pooled_selection_procedures.csv',index=False)
    # This ordering is explicitly hindsight diagnosis, not an executable selection.
    ranked=annual[~annual.model.isin(['market_proxy','buy_hold','equal_universe'])].groupby(['horizon','model']).agg(
        median_delta_spy=('delta_spy','median'),worst_delta_spy=('delta_spy','min'),
        years_cohorts_beating_spy=('beats_spy','sum'),windows=('year','size'),
        median_IC=('IC','median'),median_RankIC=('RankIC','median')).reset_index().sort_values('median_delta_spy',ascending=False)
    ranked.to_csv(root/'exploratory_stability_order.csv',index=False)
    ab=pd.read_csv(root/'fixed_k10_ablations.csv')
    loss=ab[ab.after.str.endswith('_huber') & ~ab.before.str.endswith('_huber')].copy()
    summary=loss.groupby(['horizon','before','after']).agg(median_SR_change=('delta_sharpe','median'),
                  positive=('delta_sharpe',lambda v:int((v>0).sum())),windows=('year','size')).reset_index()
    h=pd.read_csv(root/'horizon_comparison.csv')
    p=pd.read_csv(root/'fixed_k10_portfolio_effects.csv')
    cost=pd.read_csv(root/'all_metrics.csv')
    cost=cost[(cost.seed=='ensemble')&cost.selected_policy&(cost.cost_multiplier==1)&
              ~cost.model.isin(['market_proxy','buy_hold','equal_universe'])]
    cost['sharpe_drag']=cost.gross_sharpe-cost.net_sharpe
    sensitivity=cost.groupby('horizon').agg(median_sharpe_drag=('sharpe_drag','median'),median_turnover=('turnover_annual','median')).reset_index()
    all_success=ranked[ranked.years_cohorts_beating_spy==ranked.windows]
    lines=['## Findings from the completed batch','',
           f'**No model beat SPY in all eight cohort/year windows.**' if all_success.empty else
           '**Models beating SPY in all eight windows:** '+', '.join(f"{r.model} ({r.horizon}-session)" for _,r in all_success.iterrows()),'',
           'That statement uses each model\'s validation-selected portfolio and is conditional on this dataset. '
           'The individual model comparison below includes hindsight discoveries; the selection procedures here are the implementable historical rules.','',
           '### Four-year selection-procedure results','',
           table(procedures,[('procedure','Procedure'),('dataset','Cohort'),('gross_sharpe','Gross SR'),('sharpe','Net SR'),('spy_sharpe','SPY SR'),
                             ('delta_spy','Difference'),('years_beating_spy','Years > SPY'),('annualized_return','Ann. net return'),
                             ('maximum_drawdown','Max DD'),('low','Delta SR CI low'),('high','Delta SR CI high')],
                            ['annualized_return','maximum_drawdown']),'',
           'Pooled statistics compound separate annual accounts, omit boundary gaps and use a 3% annual hurdle. '
           'Each SPY series uses exactly the procedure\'s dates. Bootstrap intervals are exploratory, not multiplicity-adjusted.','',
           '### Did Huber help?','',
           table(summary,[('horizon','Horizon'),('before','MSE comparator'),('after','Huber variant'),('median_SR_change','Median SR change'),
                          ('positive','Positive windows'),('windows','Windows')]),'',
           'These comparisons fix equal weights and K=10. A lower training loss is not counted as success; the table measures realized net Sharpe. '
           'Huber and MSE use identical early-stopping criteria.','',
           '### Did daily trading help?','',
           f'At fixed equal K=10, daily forecasts improved net Sharpe in **{int((h.delta_sharpe_daily_minus_five>0).sum())} of {len(h)}** '
           f'matched model/cohort/year comparisons. The median change was **{h.delta_sharpe_daily_minus_five.median():+.3f}**. '
           'This includes the simple ranking strategies as well as neural models; it is not a count of independent trials.','',
           table(sensitivity,[('horizon','Horizon'),('median_sharpe_drag','Median gross-minus-net SR'),('median_turnover','Median annual turnover')]),'',
           'Changing the frequency also changes target horizon and batch size. A daily gain should therefore be attributed to the whole tested pipeline.','',
           '### Exploratory candidates, not newly selected winners','',
           'The following order uses median annual Sharpe difference from SPY across both cohorts. It describes stability after seeing the tests; '
           'it must not replace the validation-selected choices above.','',
           table(ranked.head(8),[('horizon','Horizon'),('model','Model'),('median_delta_spy','Median SR-SPY'),('worst_delta_spy','Worst SR-SPY'),
                                ('years_cohorts_beating_spy','Wins / 8'),('median_IC','Median IC'),('median_RankIC','Median RankIC')]),'',
           'Even a positive four-year Sharpe difference or confidence interval would remain conditional on fixed survivor cohorts, '
           'retrospective period choice and the full history of experiments. No further variants were added after this batch\'s results.','',
           '### Paper and historical comparison','',
           'StockMixer\'s AAAI paper reports a Sharpe of 1.586 on its S&P500 stock dataset, but provides no matched SPY/index buy-and-hold row. '
           'It therefore does not demonstrate outperformance of the S&P 500 itself. See the [paper review](../../STOCKMIXER_PAPER_REVIEW.md). '
           'Our older THINK+mixer NASDAQ 2017 result was net Sharpe 1.600 versus QQQ 2.188; that hybrid is not the complete paper architecture.','']
    findings='\n'.join(lines)
    (root/'FINDINGS.md').write_text(findings,encoding='utf-8')
    report=(root/'REPORT.md').read_text(encoding='utf-8')
    if '## Findings from the completed batch' in report:
        a=report.index('## Findings from the completed batch');b=report.index('## What was tested',a)
        report=report[:a]+report[b:]
    marker='## What was tested'
    report=report.replace(marker,findings+'\n'+marker,1)
    (root/'REPORT.md').write_text(report,encoding='utf-8')
    print(procedures[['procedure','dataset','sharpe','spy_sharpe','delta_spy']].round(3).to_string(index=False))
    print(summary.round(3).to_string(index=False))


if __name__=='__main__':
    main()
