"""Re-execute fixed validation and changing-score targets at fixed cost multiples."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data
import ndx_drawdown_engine as engine
from rishi_mechanism_risk import POLICIES
from rishi_mechanism_batch1 import stationary_indices,paired_contrast


def main():
    source=Path('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029');out=Path('docs/rishi_mechanism_20261027/execution_batch51_validation_cost');large=Path('results/RISHI_VALIDATION_FIXED_COST_20261029')
    for p in (out,large):
        if p.exists():raise FileExistsError(p)
        p.mkdir(parents=True)
    data=load_data();market=engine.market_inputs(data);targets=np.load(source/'TARGETS.npz',allow_pickle=False);old=np.load(source/'PATHS.npz',allow_pickle=False)
    ids=targets['ids'];dates=old['dates'];np.testing.assert_array_equal(targets['security_ids'],data['panel'].tickers)
    paths={};metrics=[];calendar=[];parity=[]
    for group in ('original_five','new_five','ten_seed'):
        for arm in ('original','fixed'):
            weights=targets[f'{group}__{arm}'];risk=engine.risk_inputs(data,engine.daily_allocation(data,ids,weights))
            for policy in ('full','vol20'):
                exposure=engine.exposure_series(POLICIES[policy],market,risk)
                for cost in (0,1,2,4):
                    m,net,ledger=engine.simulate(data,ids,weights,exposure,'weekly',multiplier=cost,detailed=True)
                    assert ledger.date.tolist()==dates.tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
                    base=f'{group}__{arm}__{policy}';key=f'{base}__c{cost}'
                    if cost==1:
                        ref=pd.read_csv(source/f'{base}_ledger.csv');error=float(np.max(abs(net-old[base])));nav=float(np.max(abs(ledger.nav-ref.nav))/1e6)
                        assert error<1e-10 and nav<1e-8;parity.append(dict(group=group,arm=arm,policy=policy,daily_error=error,nav_error=nav))
                    paths[key]=net;ledger.to_csv(large/f'{key}_ledger.csv',index=False)
                    metrics.append(dict(group=group,arm=arm,policy=policy,cost=cost,**m,mean_exposure=float(ledger.equity_exposure.mean())))
                    for year in (2022,2023,2024,2025):
                        msk=pd.DatetimeIndex(dates).year==year;calendar.append(dict(group=group,arm=arm,policy=policy,cost=cost,year=year,calendar_return=float(np.prod(1+net[msk])-1)))
                print(group,arm,policy,flush=True)
    assert len(metrics)==48 and len(parity)==12
    contrasts=[]
    for block in (5,21,63):
        draws=stationary_indices(len(dates),10000,block,27102031)
        for group in ('original_five','new_five','ten_seed'):
            for policy in ('full','vol20'):
                for cost in (0,2,4):
                    a=paths[f'{group}__original__{policy}__c{cost}'];b=paths[f'{group}__fixed__{policy}__c{cost}']
                    contrasts.append(dict(group=group,policy=policy,cost=cost,block=block,**paired_contrast(a,b,draws,252)))
    assert len(contrasts)==54
    pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(parity).to_csv(out/'BASE_REPLAY.csv',index=False)
    pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'CONTRASTS.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',dates=dates,**paths)
    inputs=[source/'TARGETS.npz',source/'PATHS.npz',Path('docs/rishi_mechanism_20261027/VALIDATION_FIXED_COST_PROTOCOL_20261029.md'),Path(__file__)]
    (out/'INPUT_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2),encoding='utf-8')
    print(pd.DataFrame(contrasts).query("group=='ten_seed' and policy=='vol20' and block==21").to_string(index=False))


if __name__=='__main__':main()
