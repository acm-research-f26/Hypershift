"""Initial capital split with actual sleeve sizing, fees and participation caps."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import continuous_backtest,load_data
from research_engine import performance


def pooled_returns(left_nav,right_nav,initial):
    nav=np.asarray(left_nav,float)+np.asarray(right_nav,float)
    if nav.ndim!=1 or len(nav)==0 or np.any(~np.isfinite(nav)) or np.any(nav<=0) or initial<=0:
        raise ValueError('Invalid pooled NAV')
    return nav/np.r_[initial,nav[:-1]]-1


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch4_capital_v2')
    large=Path('results/RISHI_MECHANISM_CAPITAL_20261027_V2')
    for path in (out,large):
        if path.exists():raise FileExistsError(path)
        path.mkdir(parents=True)
    source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
    old_metrics=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch2_exports/ACCOUNT_METRICS.csv')
    data=load_data();summary=[];curves=[]
    for group in ['original_five','new_five','ten_seed']:
        b=np.load(source/f'scores_B_{group}.npz');d=np.load(source/f'scores_D_{group}.npz')
        np.testing.assert_array_equal(b['ids'],d['ids'])
        for k in (5,10):
            bm,bc,bl=continuous_backtest(data,b['ids'],b['scores'],k,aum=500000.)
            dm,dc,dl=continuous_backtest(data,d['ids'],d['scores'],k,aum=500000.)
            if bc.date.tolist()!=dc.date.tolist():raise AssertionError('Dates')
            net=pooled_returns(bl.nav,dl.nav,1000000.);m=performance(net)
            fullb=pd.read_csv(source/f'ledger_B_{group}_{k}.csv')
            fulld=pd.read_csv(source/f'ledger_D_{group}_{k}.csv')
            if fullb.date.tolist()!=bc.date.tolist() or fulld.date.tolist()!=dc.date.tolist():
                raise AssertionError('Full-capital comparator dates')
            algebra=pooled_returns(fullb.nav/2,fulld.nav/2,1000000.)
            old=performance(algebra)
            e=old_metrics[(old_metrics.candidate=='E')&(old_metrics.group==group)&(old_metrics.k==k)].iloc[0]
            summary.append(dict(group=group,k=k,**m,cost_dollars=bm['cost_dollars']+dm['cost_dollars'],
                algebraic_cagr=old['annualized_return'],algebraic_sharpe=old['sharpe'],rank_blend_cagr=float(e.annualized_return),
                rank_blend_sharpe=float(e.sharpe),sizing_cagr_increment=m['annualized_return']-old['annualized_return']))
            curves.append(pd.DataFrame(dict(date=bc.date,net=net,nav=bl.nav+dl.nav,cash=bl.cash+dl.cash,group=group,k=k)))
            bl.to_csv(large/f'ledger_B_{group}_{k}_500k.csv',index=False)
            dl.to_csv(large/f'ledger_D_{group}_{k}_500k.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'CAPITAL_SPLIT_METRICS.csv',index=False)
    pd.concat(curves).to_csv(large/'CURVES.csv',index=False)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=6,sleeves=12,initial_total=1000000.,initial_each=500000.,
        policy='Independent initial capital split; no transfers or order netting'),indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
