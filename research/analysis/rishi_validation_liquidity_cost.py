"""Replay ten-seed validation versus liquidity under fixed cost scenarios."""
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
    source=Path('results/RISHI_VALIDATION_STYLE_ACCOUNTS_20261029');out=Path('docs/rishi_mechanism_20261027/execution_batch63_liquidity_cost');large=Path('results/RISHI_VALIDATION_LIQUIDITY_COST_20261029')
    for p in (out,large):
        if p.exists():raise FileExistsError(p)
        p.mkdir(parents=True)
    data=load_data();market=engine.market_inputs(data);t=np.load(source/'TARGETS.npz',allow_pickle=False);old=np.load(source/'PATHS.npz',allow_pickle=False);ids=t['ids'];dates=old['dates'];np.testing.assert_array_equal(t['security_ids'],data['panel'].tickers)
    paths={};metrics=[];calendar=[];parity=[]
    for kind in ('issuer5','equal10'):
        for arm in ('fixed','liquidity'):
            base=f'ten_seed__{kind}__{arm}';w=t[base];risk=engine.risk_inputs(data,engine.daily_allocation(data,ids,w));exposure=engine.exposure_series(POLICIES['vol20' if kind=='issuer5' else 'full'],market,risk)
            for cost in (0,1,2,4):
                m,net,ledger=engine.simulate(data,ids,w,exposure,'weekly',multiplier=cost,detailed=True);assert ledger.date.tolist()==dates.tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
                if cost==1:
                    ref=pd.read_csv(source/f'{base}_ledger.csv');error=float(np.max(abs(net-old[base])));nav=float(np.max(abs(ledger.nav-ref.nav))/1e6);assert error<=1e-10 and nav<=1e-8;parity.append(dict(kind=kind,arm=arm,daily_error=error,nav_error=nav))
                key=f'{kind}__{arm}__c{cost}';paths[key]=net;ledger.to_csv(large/f'{key}_ledger.csv',index=False);metrics.append(dict(kind=kind,arm=arm,cost=cost,**m,mean_exposure=float(ledger.equity_exposure.mean())))
                for year in (2022,2023,2024,2025):
                    mask=pd.DatetimeIndex(dates).year==year;calendar.append(dict(kind=kind,arm=arm,cost=cost,year=year,calendar_return=float(np.prod(1+net[mask])-1)))
            print(kind,arm,flush=True)
    contrasts=[]
    for block in (5,21,63):
        draw=stationary_indices(len(dates),10000,block,29102027)
        for kind in ('issuer5','equal10'):
            for cost in (0,2,4):contrasts.append(dict(kind=kind,cost=cost,block=block,**paired_contrast(paths[f'{kind}__fixed__c{cost}'],paths[f'{kind}__liquidity__c{cost}'],draw,252)))
    assert len(metrics)==16 and len(parity)==4 and len(contrasts)==18
    pd.DataFrame(metrics).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(parity).to_csv(out/'REPLAY.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'CONTRASTS.csv',index=False);np.savez_compressed(large/'PATHS.npz',dates=dates,**paths)
    inputs=[source/'TARGETS.npz',source/'PATHS.npz',Path(__file__),Path('docs/rishi_mechanism_20261027/VALIDATION_LIQUIDITY_COST_PROTOCOL_20261029.md')];(out/'INPUT_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2),encoding='utf-8')
    print(pd.DataFrame(contrasts).query('block==21')[['kind','cost','mean_difference','mean_low','mean_high']].to_string(index=False))


if __name__=='__main__':main()
