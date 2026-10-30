"""Verify fixed-preference account ledgers, exposures and paired mean intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data
from research_engine import performance
from rishi_mechanism_batch1 import stationary_indices


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch49_validation_account');root=Path('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029')
    paths=np.load(root/'PATHS.npz',allow_pickle=False);targets=np.load(root/'TARGETS.npz',allow_pickle=False)
    metrics=pd.read_csv(out/'METRICS.csv');contrasts=pd.read_csv(out/'CONTRASTS.csv');data=load_data()
    ids=targets['ids'];origins=data['origin'][ids];days=data['panel'].dates.get_indexer(pd.DatetimeIndex(paths['dates']))
    maximum=0.;accounts=0
    for row in metrics.itertuples():
        key=f'{row.group}__{row.arm}__{row.policy}';net=paths[key];ledger=pd.read_csv(root/f'{key}_ledger.csv');w=targets[f'{row.group}__{row.arm}']
        assert ledger.date.tolist()==paths['dates'].tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
        maximum=max(maximum,float(np.max(abs(ledger.nav/1e6-np.cumprod(1+net)))))
        m=performance(net)
        maximum=max(maximum,*[abs(m[x]-getattr(row,x)) for x in ('sharpe','annualized_return','maximum_drawdown')])
        exposure=[]
        for day in days:
            j=int(np.searchsorted(origins,day-1,side='right')-1);assert j>=0
            if row.policy=='full': exposure.append(1.);continue
            origin=origins[j];active=w[j]>0;hist=np.nan_to_num(data['returns'][origin-19:origin+1,active],nan=0.)@w[j,active]
            exposure.append(min(1.,.2/max(float(hist.std(ddof=1)*np.sqrt(252)),.05)))
        maximum=max(maximum,float(np.max(abs(np.asarray(exposure)-ledger.target_exposure))))
        accounts+=1
    for block in (5,21,63):
        draw=stationary_indices(len(days),10000,block,27102029)
        for row in contrasts[contrasts.block==block].itertuples():
            a=paths[f'{row.group}__original__{row.policy}'];b=paths[f'{row.group}__fixed__{row.policy}'];delta=a-b
            expected=np.r_[delta.mean()*252,np.quantile(delta[draw].mean(1)*252,[.025,.975])]
            maximum=max(maximum,float(np.max(abs(expected-[row.mean_difference,row.mean_low,row.mean_high]))))
    assert accounts==12 and len(contrasts)==18 and maximum<1e-10
    print(f'{accounts} accounts, 18 mean intervals and target exposures verified; max error {maximum:.3g}')


if __name__=='__main__':main()
