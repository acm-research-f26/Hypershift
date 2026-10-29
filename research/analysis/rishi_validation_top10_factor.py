"""Retrospective factors for saved model and fixed-validation accounts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_multifactor import read_factors
from rishi_factor_nw import fit_hac


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch56_validation_top10_factor');out.mkdir(parents=True,exist_ok=False)
    root=Path('data/factors');files=[root/'F-F_Research_Data_5_Factors_2x3_daily.csv',root/'F-F_Momentum_Factor_daily.csv',root/'F-F_ST_Reversal_Factor_daily.csv']
    source=Path('results/RISHI_VALIDATION_TOP10_20261029/PATHS.npz');qqq_path=Path('data/nasdaq_monthly_pit/assembled/etf_adj_close.csv')
    z=np.load(source,allow_pickle=False);dates=pd.DatetimeIndex(z['dates']);assert len(dates)==1003 and not dates.has_duplicates and dates.is_monotonic_increasing
    f=pd.concat([read_factors(files[0],['Mkt-RF','SMB','HML','RMW','CMA','RF']),read_factors(files[1],['MOM']),read_factors(files[2],['ST_REV'])],axis=1).reindex(dates)
    q=pd.read_csv(qqq_path,index_col=0,parse_dates=True)['QQQ'].pct_change(fill_method=None).reindex(dates)
    assert np.isfinite(f.to_numpy()).all() and (f.to_numpy()>-.99).all() and np.isfinite(q).all()
    names=['Mkt-RF','SMB','HML','RMW','CMA','MOM','ST_REV'];rf=f.RF.to_numpy();rows=[]
    for group in ('original_five','new_five','ten_seed'):
        a=z[f'{group}__original__c1'];b=z[f'{group}__fixed__c1'];responses={'original':a-rf,'fixed':b-rf,'difference':a-b}
        for design in ('FF7','FF7_QQQ'):
            terms=['intercept']+names+(['QQQ_excess'] if design=='FF7_QQQ' else [])
            x=np.column_stack([np.ones(len(dates)),f[names].to_numpy()]+([q.to_numpy()-rf] if design=='FF7_QQQ' else []))
            for response,y in responses.items():
                for lag in (5,20):
                    fit=fit_hac(y,x,lag);se=np.sqrt(np.maximum(np.diag(fit['covariance']),0))
                    for j,term in enumerate(terms):
                        scale=252 if j==0 else 1;beta=float(fit['beta'][j]*scale);sigma=float(se[j]*scale)
                        rows.append(dict(group=group,design=design,response=response,lag=lag,term=term,n=fit['n'],coefficient=beta,se=sigma,low=beta-1.95996398454*sigma,high=beta+1.95996398454*sigma,r2=fit['r2']))
    assert len(rows)==3*3*2*(8+9) # FF7 8 terms, FF7_QQQ 9 terms
    frame=pd.DataFrame(rows);frame.to_csv(out/'COEFFICIENTS.csv',index=False)
    inputs=files+[source,qqq_path,Path(__file__),Path('research/analysis/rishi_factor_nw.py'),Path('research/analysis/rishi_multifactor.py'),Path('docs/rishi_mechanism_20261027/VALIDATION_TOP10_FACTOR_PROTOCOL_20261029.md')]
    (out/'INPUT_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2),encoding='utf-8')
    print(frame[(frame.group=='ten_seed')&(frame.response=='difference')&(frame.term=='intercept')][['design','lag','coefficient','low','high']].to_string(index=False))


if __name__=='__main__':main()
