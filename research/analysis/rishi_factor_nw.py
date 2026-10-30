"""Supportive fixed exposure/spanning OLS with Newey-West Bartlett uncertainty."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def fit_hac(y,x,lag):
    y=np.asarray(y,float);x=np.asarray(x,float)
    if y.ndim!=1 or x.ndim!=2 or len(y)!=len(x) or not np.isfinite(y).all() or not np.isfinite(x).all():raise ValueError('Finite aligned design required')
    n,p=x.shape
    if not 0<=lag<n or n<=p or np.linalg.matrix_rank(x)!=p:raise ValueError('Invalid lag or singular design')
    beta=np.linalg.lstsq(x,y,rcond=None)[0];residual=y-x@beta;scores=x*residual[:,None]
    meat=scores.T@scores
    for j in range(1,lag+1):
        cross=scores[j:].T@scores[:-j];meat+=(1-j/(lag+1))*(cross+cross.T)
    bread=np.linalg.inv(x.T@x);covariance=bread@meat@bread*n/(n-p)
    denominator=float(np.sum((y-y.mean())**2))
    return dict(beta=beta,covariance=covariance,r2=1-float(residual@residual)/denominator if denominator>0 else np.nan,n=n)


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch13_factor'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/ACCOUNT_CURVES.csv')
    risk_path=Path('results/RISHI_MECHANISM_RISK_20261027/PATHS_PORTABLE.npz')
    q_path=Path('data/nasdaq_monthly_pit/assembled/etf_adj_close.csv')
    frame=pd.read_csv(source);risk=np.load(risk_path,allow_pickle=False);dates=risk['dates'].tolist()
    def account(name):
        row=frame[frame.id==name]
        if row.date.tolist()!=dates:raise ValueError('Daily account dates')
        return row.net.to_numpy()
    def path(name):return risk['returns'][:,np.flatnonzero(risk['ids']==name)[0]]
    q=pd.read_csv(q_path,index_col=0,parse_dates=True)['QQQ'].pct_change(fill_method=None).reindex(pd.DatetimeIndex(dates)).to_numpy()
    rf=1.03**(1/252)-1;e=account('E/ten_seed/10');b=account('B/ten_seed/10');h=account('E_Hlabel/ten_seed/10')
    regressions=[('E10_on_QQQ',e-rf,q-rf),('E_Hlabel10_on_QQQ',h-rf,q-rf),
        ('E10_minus_B10_on_QQQ',e-b,q-rf),('E10_minus_E_Hlabel10_on_QQQ',e-h,q-rf)]
    for candidate,group,allocation in [('E','ten_seed','equal5'),('E','ten_seed','issuer5'),('QQQ','benchmark','one_etf')]:
        base=f'{candidate}/{group}/{allocation}'
        regressions.append((f'{candidate}_{allocation}_vol20_on_full',path(base+'/vol20/modeled1')-rf,path(base+'/full/modeled1')-rf))
    rows=[]
    for name,y,factor in regressions:
        design=np.column_stack([np.ones(len(y)),factor])
        for lag in [5,20]:
            result=fit_hac(y,design,lag);beta=result['beta'];se=np.sqrt(np.maximum(np.diag(result['covariance']),0))
            z=beta[0]/se[0] if se[0]>0 else np.nan
            rows.append(dict(regression=name,lag=lag,n=result['n'],alpha_daily=float(beta[0]),
                alpha_annual=float(beta[0]*252),alpha_annual_se=float(se[0]*252),
                alpha_annual_low=float((beta[0]-1.95996398454*se[0])*252),
                alpha_annual_high=float((beta[0]+1.95996398454*se[0])*252),
                alpha_p_two_sided=float(math.erfc(abs(z)/np.sqrt(2))),factor_beta=float(beta[1]),factor_beta_se=float(se[1]),r2=result['r2']))
    pd.DataFrame(rows).to_csv(out/'FACTOR_ATTRIBUTION.csv',index=False)
    inputs=[source,risk_path,q_path,Path(__file__),doc/'FACTOR_NW_PROTOCOL.md']
    (out/'RESULTS.json').write_text(json.dumps(dict(regressions=7,rows=len(rows),daily_sessions=len(dates),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        scope='Selected-history explanatory regressions only; not traded alpha, comprehensive factors or a primary decision family'),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
