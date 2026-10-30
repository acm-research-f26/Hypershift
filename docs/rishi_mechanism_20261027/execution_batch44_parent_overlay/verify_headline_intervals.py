"""Independent stationary bootstrap SR/CAGR intervals for headline comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('results/RISHI_PARENT_OVERLAY_20261028');out=Path('docs/rishi_mechanism_20261027/execution_batch44_parent_overlay')
paths=np.load(root/'PATHS.npz',allow_pickle=False);result=pd.read_csv(out/'CONTRASTS.csv');n=len(paths['dates'])
rng=np.random.default_rng(27102028);draw=np.empty((10000,n),dtype=np.int32);draw[:,0]=rng.integers(n,size=10000)
for j in range(1,n):
    restart=rng.random(10000)<1/21;start=rng.integers(n,size=10000)
    draw[:,j]=np.where(restart,start,(draw[:,j-1]+1)%n)
rf=np.expm1(np.log1p(.03)/252);maximum=0.;checked=0
for group in ('original_five','ten_seed'):
    for allocation in ('equal5','issuer5'):
        a=paths[f'after_hours_premarket__{group}__{allocation}__vol20'];b=paths[f'daily__{group}__{allocation}__vol20']
        aa=a[draw];bb=b[draw]
        sr=lambda x:(x.mean(axis=-1)-rf)/x.std(axis=-1,ddof=1)*np.sqrt(252)
        cagr=lambda x:np.expm1(np.log1p(x).mean(axis=-1)*252)
        row=result[(result['group']==group)&(result.allocation==allocation)&(result.policy=='vol20')&(result.comparator=='daily')&(result.block==21)].iloc[0]
        for label,values in [('sr',sr(aa)-sr(bb)),('cagr',cagr(aa)-cagr(bb))]:
            expected=np.quantile(values,[.025,.975]);observed=row[[label+'_low',label+'_high']].to_numpy(float)
            maximum=max(maximum,float(np.max(abs(expected-observed))))
        checked+=1
assert checked==4 and maximum<1e-12
print('Four headline SR and CAGR interval pairs rebuilt; max error',maximum)
