"""Independent reconstruction of coefficients and Bartlett covariance."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('data/factors')
def parse(name,columns):
    lines=(root/name).read_text().splitlines()
    rows=[s.split(',') for s in lines if len(s.split(',')[0].strip())==8 and s.split(',')[0].strip().isdigit()]
    frame=pd.DataFrame(rows,columns=['date']+columns)
    frame.date=pd.to_datetime(frame.date,format='%Y%m%d')
    for column in columns:frame[column]=pd.to_numeric(frame[column])/100
    return frame.set_index('date')
fs=parse('F-F_Research_Data_5_Factors_2x3_daily.csv',['market','smb','hml','rmw','cma','rf'])
fs=fs.join(parse('F-F_Momentum_Factor_daily.csv',['mom']),how='inner').join(parse('F-F_ST_Reversal_Factor_daily.csv',['rev']),how='inner')
accounts=pd.read_csv('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/ACCOUNT_CURVES.csv',parse_dates=['date'])
paths=accounts.pivot(index='date',columns='id',values='net')
ids=['E/ten_seed/10','B/ten_seed/10','E_Hlabel/ten_seed/10']
dates=paths[ids].dropna().index
fs=fs.loc[dates]
qqq=pd.read_csv('data/nasdaq_monthly_pit/assembled/etf_adj_close.csv',index_col=0,parse_dates=True).QQQ.pct_change(fill_method=None).loc[dates]
e,b,h=[paths[n].loc[dates].to_numpy() for n in ids]
ys={'E':e-fs.rf,'B':b-fs.rf,'E_Hlabel':h-fs.rf,'E_minus_B':e-b,'E_minus_E_Hlabel':e-h}
actual=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch28_multifactor/COEFFICIENTS.csv')
max_beta=max_se=0.
for design in ['FF7','FF7_QQQ']:
    x=np.column_stack([np.ones(len(dates)),fs[['market','smb','hml','rmw','cma','mom','rev']].to_numpy()]+([qqq.to_numpy()-fs.rf.to_numpy()] if design=='FF7_QQQ' else []))
    inv=np.linalg.inv(x.T@x)
    for name,y in ys.items():
        y=np.asarray(y); beta=inv@x.T@y; residual=y-x@beta; scores=x*residual[:,None]
        for lag in [5,20]:
            meat=np.zeros((x.shape[1],x.shape[1]))
            for offset in range(-lag,lag+1):
                left=scores[max(offset,0):len(scores)+min(offset,0)]
                right=scores[max(-offset,0):len(scores)-max(offset,0)]
                meat+=(1-abs(offset)/(lag+1))*left.T@right
            se=np.sqrt(np.diag(inv@meat@inv)*len(y)/(len(y)-x.shape[1])); scale=np.ones(len(beta));scale[0]=252
            rows=actual[(actual.design==design)&(actual.response==name)&(actual.lag==lag)]
            max_beta=max(max_beta,float(np.max(np.abs(rows.coefficient.to_numpy()-beta*scale))))
            max_se=max(max_se,float(np.max(np.abs(rows.se.to_numpy()-se*scale))))
assert len(dates)==1003 and len(actual)==170 and max_beta<1e-10 and max_se<1e-10
print(f'1003 dates, 20 regressions, 170 coefficients; maximum coefficient error {max_beta:.3g}; maximum SE error {max_se:.3g}')
