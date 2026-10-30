"""Reconstruct factor regressions with independent parser and two-sided HAC sum."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('data/factors')
def parse(name,columns):
    rows=[s.split(',') for s in (root/name).read_text(encoding='utf-8-sig').splitlines() if len(s.split(',')[0].strip())==8 and s.split(',')[0].strip().isdigit()]
    f=pd.DataFrame(rows,columns=['date']+columns);f.date=pd.to_datetime(f.date,format='%Y%m%d')
    for c in columns:f[c]=pd.to_numeric(f[c])/100
    return f.set_index('date')
f=parse('F-F_Research_Data_5_Factors_2x3_daily.csv',['market','smb','hml','rmw','cma','rf'])
f=f.join(parse('F-F_Momentum_Factor_daily.csv',['mom']),how='inner').join(parse('F-F_ST_Reversal_Factor_daily.csv',['rev']),how='inner')
z=np.load('results/RISHI_VALIDATION_TOP10_20261029/PATHS.npz',allow_pickle=False);dates=pd.DatetimeIndex(z['dates']);f=f.loc[dates]
q=pd.read_csv('data/nasdaq_monthly_pit/assembled/etf_adj_close.csv',index_col=0,parse_dates=True).QQQ.pct_change(fill_method=None).loc[dates]
actual=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch56_validation_top10_factor/COEFFICIENTS.csv');maximum=0.;count=0
for group in ('original_five','new_five','ten_seed'):
    a=z[f'{group}__original__c1'];b=z[f'{group}__fixed__c1'];ys={'original':a-f.rf,'fixed':b-f.rf,'difference':a-b}
    for design in ('FF7','FF7_QQQ'):
        x=np.column_stack([np.ones(len(dates)),f[['market','smb','hml','rmw','cma','mom','rev']].to_numpy()]+([q.to_numpy()-f.rf.to_numpy()] if design=='FF7_QQQ' else []));inv=np.linalg.inv(x.T@x)
        for response,y in ys.items():
            y=np.asarray(y);beta=inv@x.T@y;score=x*(y-x@beta)[:,None]
            for lag in (5,20):
                meat=np.zeros((x.shape[1],x.shape[1]))
                for off in range(-lag,lag+1):
                    left=score[max(off,0):len(score)+min(off,0)];right=score[max(-off,0):len(score)-max(off,0)]
                    meat+=(1-abs(off)/(lag+1))*left.T@right
                se=np.sqrt(np.diag(inv@meat@inv)*len(y)/(len(y)-x.shape[1]));scale=np.ones(len(beta));scale[0]=252
                r=actual[(actual.group==group)&(actual.design==design)&(actual.response==response)&(actual.lag==lag)]
                assert len(r)==len(beta);maximum=max(maximum,float(np.max(abs(r.coefficient.to_numpy()-beta*scale))),float(np.max(abs(r.se.to_numpy()-se*scale))));count+=len(r)
assert len(dates)==1003 and count==306 and maximum<1e-10
print(f'36 regressions, {count} coefficients and standard errors verified; max error {maximum:.3g}')
