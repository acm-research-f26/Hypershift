"""Fixed historical blend grid; source baseline parity precedes search."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from pit_research_engine import load_data
import ndx_drawdown_engine as engine
from rishi_mechanism_risk import POLICIES

SOURCE=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
GROUPS=('original_five','new_five','ten_seed')

def blend(b,d,mask,weight):
    b=np.asarray(b);d=np.asarray(d);mask=np.asarray(mask,dtype=bool)
    if b.shape!=d.shape or b.shape!=mask.shape or not 0<=weight<=1 or not np.isfinite(b[mask]).all() or not np.isfinite(d[mask]).all(): raise ValueError('invalid blend inputs')
    out=np.full(b.shape,np.nan);out[mask]=(1-weight)*b[mask]+weight*d[mask];return out

def main():
    out=Path('docs/historical_sharpe_20261029/blend_batch1');dest=Path('results/HISTORICAL_SHARPE_BLEND_20261029')
    for p in (out,dest):
        if p.exists(): raise FileExistsError(p)
    out.mkdir(parents=True);dest.mkdir(parents=True)
    data=load_data();market=engine.market_inputs(data);source=np.load('runs/ndx_drawdown_20261026/batch1/paths.npz',allow_pickle=False);top=np.load('results/RISHI_VALIDATION_TOP10_20261029/PATHS.npz',allow_pickle=False)
    curves={};metrics=[];annual=[];parity=[];inputs=[]
    for group in GROUPS:
        scores={}
        for model in ('B','D','E'):
            p=SOURCE/f'scores_{model}_{group}.npz';inputs.append(p);z=np.load(p,allow_pickle=False);scores[model]=z['scores'];
            if model=='B':ids=z['ids']
            else:np.testing.assert_array_equal(ids,z['ids'])
        mask=data['mask'][ids];np.testing.assert_allclose(blend(scores['B'],scores['D'],mask,.5)[mask],scores['E'][mask],rtol=0,atol=1e-12)
        dates=data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy();np.testing.assert_array_equal(dates,top['dates']);np.testing.assert_array_equal(dates,source['dates'])
        for weight in (.5,0,.25,.75,1):
            score=blend(scores['B'],scores['D'],mask,weight)
            for kind in ('issuer5','equal10'):
                w=engine.allocations(data,ids,score,kind);risk=engine.risk_inputs(data,engine.daily_allocation(data,ids,w));exposure=engine.exposure_series(POLICIES['vol20' if kind=='issuer5' else 'full'],market,risk)
                m,path,ledger=engine.simulate(data,ids,w,exposure,'weekly',multiplier=1,detailed=True)
                assert ledger.date.tolist()==dates.tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
                if weight==.5:
                    if kind=='issuer5':
                        key=f'issuer5/basketvol20_target20_weekly/{group}';ref=source['returns'][:,np.flatnonzero(source['ids']==key)[0]]
                    else:ref=top[f'{group}__original__c1']
                    err=float(np.max(abs(path-ref)));assert err<1e-10;parity.append(dict(group=group,kind=kind,max_error=err))
                key=f'{group}__{kind}__w{int(weight*100):03d}';curves[key]=path;ledger.to_csv(dest/f'{key}_ledger.csv',index=False)
                metrics.append(dict(group=group,kind=kind,dynamic_weight=weight,**m,mean_exposure=float(ledger.equity_exposure.mean())))
                for year in (2022,2023,2024,2025):
                    y=pd.DatetimeIndex(dates).year==year;annual.append(dict(id=key,year=year,calendar_return=float(np.prod(1+path[y])-1)))
        print(group,flush=True)
    assert len(metrics)==30 and len(parity)==6
    pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(annual).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(parity).to_csv(out/'PARITY.csv',index=False);np.savez_compressed(dest/'PATHS.npz',dates=dates,**curves)
    inputs.extend([Path(__file__),Path('docs/historical_sharpe_20261029/PLAN.md')]);(out/'HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).sort_values('sharpe',ascending=False)[['group','kind','dynamic_weight','sharpe','annualized_return','maximum_drawdown']].to_string(index=False))

if __name__=='__main__':main()
