"""Fixed retrospective factor attribution for matched coverage controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_multifactor import read_factors
from rishi_factor_nw import fit_hac

DOC=Path('docs/rishi_mechanism_20261027');ROOT=Path('data/factors')
PATHS=Path('results/RISHI_INDUSTRY_COVERAGE_20261028/PATHS.npz')
QQQ=Path('data/nasdaq_monthly_pit/assembled/etf_adj_close.csv')


def main():
    out=DOC/'execution_batch38_coverage_factors';out.mkdir(parents=True,exist_ok=False)
    files=[ROOT/'F-F_Research_Data_5_Factors_2x3_daily.csv',ROOT/'F-F_Momentum_Factor_daily.csv',ROOT/'F-F_ST_Reversal_Factor_daily.csv']
    saved=np.load(PATHS,allow_pickle=False);dates=pd.DatetimeIndex(saved['dates'])
    if len(dates)!=1003 or dates.has_duplicates or not dates.is_monotonic_increasing:raise ValueError('Bad dates')
    factors=pd.concat([read_factors(files[0],['Mkt-RF','SMB','HML','RMW','CMA','RF']),
        read_factors(files[1],['MOM']),read_factors(files[2],['ST_REV'])],axis=1).reindex(dates)
    qqq=pd.read_csv(QQQ,index_col=0,parse_dates=True)['QQQ'].pct_change(fill_method=None).reindex(dates)
    if not np.isfinite(factors.to_numpy()).all() or (factors.to_numpy()<=-.99).any() or not np.isfinite(qqq).all():raise ValueError('Bad factor coverage')
    names=['Mkt-RF','SMB','HML','RMW','CMA','MOM','ST_REV'];rf=factors.RF.to_numpy()
    e,h,ec,hc=(saved[k] for k in ('E_original','E_Hlabel_original','E_coverage6','E_Hlabel_coverage6'))
    responses={'E':e-rf,'E_Hlabel':h-rf,'E_coverage6':ec-rf,'E_Hlabel_coverage6':hc-rf,
               'E_minus_coverage6':e-ec,'E_Hlabel_minus_coverage6':h-hc}
    prior=pd.read_csv(DOC/'execution_batch28_multifactor/COEFFICIENTS.csv')
    rows=[];max_prior=0.
    for design in ('FF7','FF7_QQQ'):
        columns=['intercept']+names+(['QQQ_excess'] if design=='FF7_QQQ' else [])
        parts=[np.ones(len(dates)),factors[names].to_numpy()]
        if design=='FF7_QQQ':parts.append(qqq.to_numpy()-rf)
        x=np.column_stack(parts)
        for response,y in responses.items():
            for lag in (5,20):
                fit=fit_hac(y,x,lag);se=np.sqrt(np.maximum(np.diag(fit['covariance']),0))
                for i,term in enumerate(columns):
                    scale=252 if i==0 else 1;beta=float(fit['beta'][i]*scale);sigma=float(se[i]*scale)
                    rows.append(dict(design=design,response=response,lag=lag,term=term,n=fit['n'],coefficient=beta,se=sigma,
                        low=beta-1.95996398454*sigma,high=beta+1.95996398454*sigma,r2=fit['r2']))
                    if response in ('E','E_Hlabel'):
                        ref=prior[(prior.design==design)&(prior.response==response)&(prior.lag==lag)&(prior.term==term)]
                        if len(ref)!=1:raise ValueError('Missing prior coefficient')
                        max_prior=max(max_prior,abs(beta-float(ref.coefficient.iloc[0])),abs(sigma-float(ref.se.iloc[0])))
    if max_prior>1e-10:raise ValueError('Previous original regression mismatch')
    result=pd.DataFrame(rows);result.to_csv(out/'COEFFICIENTS.csv',index=False)
    paths=[*files,PATHS,QQQ,DOC/'execution_batch28_multifactor/COEFFICIENTS.csv',Path(__file__),
        Path('research/analysis/rishi_multifactor.py'),Path('research/analysis/rishi_factor_nw.py'),DOC/'COVERAGE_MULTIFACTOR_PROTOCOL.md']
    (out/'RESULTS.json').write_text(json.dumps(dict(regressions=24,sessions=len(dates),max_prior_error=max_prior,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limits='Exploratory selected-history attribution, not causal or confirmed alpha'),indent=2)+'\n')
    print(result[result.term=='intercept'][['design','response','lag','coefficient','low','high','r2']].to_string(index=False))


if __name__=='__main__':main()
