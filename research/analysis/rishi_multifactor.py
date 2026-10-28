"""Frozen retrospective multifactor attribution of existing account paths."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_factor_nw import fit_hac

DOC = Path('docs/rishi_mechanism_20261027')
FACTORS = Path('data/factors')


def read_factors(path, names):
    rows = []
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if row and len(row[0].strip()) == 8 and row[0].strip().isdigit():
                if len(row) != len(names)+1:
                    raise ValueError(f'Malformed factor row: {path}')
                rows.append([pd.to_datetime(row[0].strip(), format='%Y%m%d'), *[float(v.strip()) for v in row[1:]]])
    result = pd.DataFrame(rows, columns=['date', *names]).set_index('date')
    if result.index.has_duplicates or result.empty:
        raise ValueError(f'Duplicate or missing factor dates: {path}')
    return result / 100.


def main():
    out = DOC/'execution_batch28_multifactor'
    out.mkdir(parents=True, exist_ok=False)
    files = [FACTORS/'F-F_Research_Data_5_Factors_2x3_daily.csv',
             FACTORS/'F-F_Momentum_Factor_daily.csv', FACTORS/'F-F_ST_Reversal_Factor_daily.csv']
    source = Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/ACCOUNT_CURVES.csv')
    qpath = Path('data/nasdaq_monthly_pit/assembled/etf_adj_close.csv')
    frame = pd.read_csv(source)
    accounts = {}
    for name in ('E/ten_seed/10', 'B/ten_seed/10', 'E_Hlabel/ten_seed/10'):
        part = frame.loc[frame.id == name, ['date', 'net']].copy()
        part.date = pd.to_datetime(part.date)
        if part.date.duplicated().any() or not part.date.is_monotonic_increasing:
            raise ValueError('Invalid account dates')
        accounts[name] = part.set_index('date').net
    index = accounts['E/ten_seed/10'].index
    if len(index) != 1003 or any(not s.index.equals(index) for s in accounts.values()):
        raise ValueError('Account calendar mismatch')
    factors = pd.concat([read_factors(files[0], ['Mkt-RF','SMB','HML','RMW','CMA','RF']),
                         read_factors(files[1], ['MOM']), read_factors(files[2], ['ST_REV'])], axis=1).reindex(index)
    qqq = pd.read_csv(qpath, index_col=0, parse_dates=True)['QQQ'].pct_change(fill_method=None).reindex(index)
    values = factors.to_numpy()
    if not np.isfinite(values).all() or (values <= -.99).any() or not np.isfinite(qqq).all():
        raise ValueError('Missing, nonfinite or sentinel factor or QQQ value')
    e,b,h = (accounts[n].to_numpy() for n in ('E/ten_seed/10','B/ten_seed/10','E_Hlabel/ten_seed/10'))
    responses = {'E':e-factors.RF.to_numpy(), 'B':b-factors.RF.to_numpy(),
                 'E_Hlabel':h-factors.RF.to_numpy(), 'E_minus_B':e-b, 'E_minus_E_Hlabel':e-h}
    names = ['Mkt-RF','SMB','HML','RMW','CMA','MOM','ST_REV']
    rows = []
    for design in ('FF7','FF7_QQQ'):
        columns = ['intercept']+names+(['QQQ_excess'] if design == 'FF7_QQQ' else [])
        components = [np.ones(len(index)), factors[names].to_numpy()]
        if design == 'FF7_QQQ':
            components.append((qqq-factors.RF).to_numpy())
        x = np.column_stack(components)
        for response, y in responses.items():
            for lag in (5,20):
                fit = fit_hac(y,x,lag)
                for j, term in enumerate(columns):
                    beta = float(fit['beta'][j]); se = float(np.sqrt(max(fit['covariance'][j,j],0)))
                    scale = 252 if j == 0 else 1
                    rows.append(dict(design=design,response=response,lag=lag,term=term,n=fit['n'],
                        coefficient=beta*scale,se=se*scale,low=(beta-1.95996398454*se)*scale,
                        high=(beta+1.95996398454*se)*scale,
                        p_two_sided=math.erfc(abs(beta/se)/np.sqrt(2)) if se else None,r2=fit['r2']))
    result = pd.DataFrame(rows)
    result.to_csv(out/'COEFFICIENTS.csv',index=False)
    inputs = files+[source,qpath,Path('research/analysis/rishi_factor_nw.py'),Path(__file__),DOC/'MULTIFACTOR_PROTOCOL.md']
    meta = {'sessions':len(index),'first':str(index[0].date()),'last':str(index[-1].date()),
            'regressions':20,'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (out/'RESULTS.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(result.loc[(result.term=='intercept'),['design','response','lag','coefficient','low','high','r2']].to_string(index=False))


if __name__ == '__main__':
    main()
