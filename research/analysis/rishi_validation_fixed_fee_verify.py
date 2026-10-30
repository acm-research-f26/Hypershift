"""Recompute all fixed-preference fee components from ledgers and paths."""
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_mechanism_batch1 import stationary_indices


def main():
    root=Path('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029');out=Path('docs/rishi_mechanism_20261027/execution_batch50_validation_fee')
    p=np.load(root/'PATHS.npz',allow_pickle=False);rows=pd.read_csv(out/'COMPONENTS.csv');annual=pd.read_csv(out/'CALENDAR.csv')
    dates=p['dates'];years=pd.DatetimeIndex(dates).year;draw=stationary_indices(len(dates),10000,21,27102030);maxerr=0.;checked=0
    for group in ('original_five','new_five','ten_seed'):
        for policy in ('full','vol20'):
            series={}
            for arm in ('original','fixed'):
                key=f'{group}__{arm}__{policy}';ledger=pd.read_csv(root/f'{key}_ledger.csv');nav=ledger.nav.to_numpy();prev=np.r_[1e6,nav[:-1]]
                assert ledger.date.tolist()==dates.tolist();net=nav/prev-1;fee=ledger.fee.to_numpy()/prev
                maxerr=max(maxerr,float(np.max(abs(net-p[key]))));series[arm]={'net':net,'fee':fee,'gross':net+fee}
            for component in ('net','gross','fee'):
                a=series['original'][component];b=series['fixed'][component];delta=a-b
                expected=np.r_[252*a.mean(),252*b.mean(),252*delta.mean(),np.quantile(252*delta[draw].mean(1),[.025,.975])]
                r=rows[(rows.group==group)&(rows.policy==policy)&(rows.component==component)].iloc[0]
                maxerr=max(maxerr,float(np.max(abs(expected-r[['original_annual','fixed_annual','difference','low','high']].to_numpy(dtype=float)))))
                for year in (2022,2023,2024,2025):
                    r2=annual[(annual.group==group)&(annual.policy==policy)&(annual.component==component)&(annual.year==year)].iloc[0]
                    maxerr=max(maxerr,abs(float(r2.difference)-252*delta[years==year].mean()))
                checked+=1
    assert checked==18 and maxerr<1e-10;print(f'{checked} component intervals and 72 calendar partitions verified; max error {maxerr:.3g}')


if __name__=='__main__':main()
