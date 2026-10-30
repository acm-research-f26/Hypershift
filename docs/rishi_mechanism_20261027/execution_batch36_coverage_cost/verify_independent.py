"""Rebuild metrics, NAV and paired mean intervals from independent formulas."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('results/RISHI_COVERAGE_COST_20261028');out=Path('docs/rishi_mechanism_20261027/execution_batch36_coverage_cost')
paths=np.load(root/'PATHS.npz',allow_pickle=False);metrics=pd.read_csv(out/'METRICS.csv');contrasts=pd.read_csv(out/'CONTRASTS.csv')
assert len(metrics)==16 and len(contrasts)==24
max_metric=0.;max_nav=0.;max_interval=0.
for row in metrics.itertuples():
    key=f'{row.model}_{row.arm}_{row.mode}';r=paths[key];ledger=pd.read_csv(root/f'{key}_ledger.csv')
    assert ledger.date.tolist()==paths['dates'].tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
    wealth=np.r_[1.,np.cumprod(1+r)];rf=np.expm1(np.log1p(.03)/252)
    sr=(r.mean()-rf)*np.sqrt(252)/r.std(ddof=1)
    cagr=wealth[-1]**(252/len(r))-1
    drawdown=np.min(wealth/np.maximum.accumulate(wealth)-1)
    max_metric=max(max_metric,abs(sr-row.sharpe),abs(cagr-row.annualized_return),abs(drawdown-row.maximum_drawdown))
    max_nav=max(max_nav,float(np.max(abs(ledger.nav.to_numpy()-wealth[1:]*1e6))/1e6))
for block in (5,21,63):
    n=len(paths['dates']);rng=np.random.default_rng(27102028);ids=np.empty((10000,n),dtype=np.int32);ids[:,0]=rng.integers(n,size=10000)
    for j in range(1,n):
        restart=rng.random(10000)<1/block;start=rng.integers(n,size=10000)
        ids[:,j]=np.where(restart,start,(ids[:,j-1]+1)%n)
    for model in ('E','E_Hlabel'):
        for mode in ('modeled1','zero','modeled2','modeled4'):
            delta=paths[f'{model}_original_{mode}']-paths[f'{model}_coverage6_{mode}']
            values=np.r_[252*delta.mean(),np.quantile(252*delta[ids].mean(1),[.025,.975])]
            row=contrasts[(contrasts.model==model)&(contrasts['mode']==mode)&(contrasts.block==block)].iloc[0]
            max_interval=max(max_interval,float(np.max(abs(values-row[['mean_difference','mean_low','mean_high']].to_numpy(float)))))
assert max_metric<1e-11 and max_nav<1e-8 and max_interval<1e-12
print('16 accounts, 24 paired mean intervals; max metric error',max_metric,'NAV relative error',max_nav,'interval error',max_interval)
