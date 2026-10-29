"""Fixed parent-view blends under previously selected exposure policies."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from rishi_mechanism_risk import POLICIES
from rishi_mechanism_batch1 import stationary_indices,paired_contrast

DOC=Path('docs/rishi_mechanism_20261027');SCORE=Path('results/RISHI_EXTENDED_PARENT_BLEND_20261028/SCORES.npz')
RISK=Path('results/RISHI_MECHANISM_RISK_20261027')
VIEWS=('after_hours_premarket','daily','after_hours');GROUPS=('original_five','new_five','ten_seed')


def main():
    out=DOC/'execution_batch44_parent_overlay';large=Path('results/RISHI_PARENT_OVERLAY_20261028')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();market=engine.market_inputs(data);scores=np.load(SCORE,allow_pickle=False);ids=scores['ids']
    prior=np.load(RISK/'PATHS_PORTABLE.npz',allow_pickle=False);dates=prior['dates'];paths={};targets={};rows=[];replays=[];calendar=[];overlap=[]
    inputs=[SCORE,RISK/'PATHS_PORTABLE.npz',Path(__file__),DOC/'EXTENDED_PARENT_OVERLAY_PROTOCOL.md',
        Path('research/analysis/rishi_mechanism_risk.py'),Path('research/analysis/rishi_mechanism_batch1.py'),
        Path('research/core/ndx_drawdown_engine.py'),Path('research/core/pit_research_engine.py'),Path('research/core/research_engine.py')]
    base=Path('data/nasdaq_monthly_pit/assembled');inputs.extend(base/f for f in ('manifest.json','economic_close.csv','dollar_volume.csv','tradable.csv','membership.csv','etf_adj_close.csv'))
    for view in VIEWS:
        for group in GROUPS:
            score=scores[f'{view}__{group}']
            for allocation in ('equal5','issuer5'):
                weights=engine.allocations(data,ids,score,allocation);targets[f'{view}__{group}__{allocation}']=weights
                risk=engine.risk_inputs(data,engine.daily_allocation(data,ids,weights))
                if view!='after_hours_premarket':
                    reference=targets[f'after_hours_premarket__{group}__{allocation}']
                    for i,origin in enumerate(ids):overlap.append(dict(view=view,group=group,allocation=allocation,origin=int(origin),
                        weight_overlap=float(np.minimum(reference[i],weights[i]).sum()),shared_names=int(((reference[i]>0)&(weights[i]>0)).sum())))
                for policy in ('vol20','vol25trend'):
                    exposure=engine.exposure_series(POLICIES[policy],market,risk)
                    metrics,net,ledger=engine.simulate(data,ids,weights,exposure,'weekly',multiplier=1.,detailed=True)
                    if ledger.date.tolist()!=dates.tolist() or ledger.cash.min() < -1e-5 or ledger.equity_exposure.iloc[-1]!=0:raise ValueError('Dates/cash/liquidation')
                    if view=='after_hours_premarket':
                        refkey=f'E/{group}/{allocation}/{policy}/modeled1';position=np.flatnonzero(prior['ids']==refkey)
                        if len(position)!=1:raise ValueError('Missing source risk path')
                        source_ledger=RISK/(refkey.replace('/','__')+'.csv');inputs.append(source_ledger);old=pd.read_csv(source_ledger)
                        if old.date.tolist()!=dates.tolist():raise ValueError('Source ledger dates')
                        err=float(np.max(abs(net-prior['returns'][:,position[0]])));nav=float(np.max(abs(ledger.nav-old.nav))/1e6)
                        replays.append(dict(group=group,allocation=allocation,policy=policy,daily_error=err,nav_error=nav))
                        if err>1e-10 or nav>1e-8:raise ValueError('Original overlay replay failed')
                    key=f'{view}__{group}__{allocation}__{policy}';paths[key]=net
                    ledger.to_csv(large/f'{key}_ledger.csv',index=False)
                    rows.append(dict(view=view,group=group,allocation=allocation,policy=policy,**metrics,
                        mean_exposure=float(ledger.equity_exposure.mean()),mean_target_exposure=float(ledger.target_exposure.mean()),min_cash=float(ledger.cash.min())))
                    years=pd.DatetimeIndex(dates).year
                    for year in (2022,2023,2024,2025):calendar.append(dict(view=view,group=group,allocation=allocation,policy=policy,year=year,calendar_return=float(np.prod(1+net[years==year])-1)))
                    print(key,flush=True)
    results=[]
    for block in (5,21,63):
        draws=stationary_indices(len(dates),10000,block,27102028)
        for group in GROUPS:
            for allocation in ('equal5','issuer5'):
                for policy in ('vol20','vol25trend'):
                    a=paths[f'after_hours_premarket__{group}__{allocation}__{policy}']
                    for view in ('daily','after_hours'):
                        b=paths[f'{view}__{group}__{allocation}__{policy}']
                        results.append(dict(group=group,allocation=allocation,policy=policy,comparator=view,block=block,**paired_contrast(a,b,draws,252)))
    pd.DataFrame(rows).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(replays).to_csv(out/'BASE_REPLAY.csv',index=False)
    pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(overlap).to_csv(out/'OVERLAP.csv',index=False)
    pd.DataFrame(results).to_csv(out/'CONTRASTS.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',dates=dates,**paths);np.savez_compressed(large/'TARGETS.npz',ids=ids,security_ids=np.asarray(data['panel'].tickers,dtype='U'),**targets)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=len(rows),base_replays=len(replays),paired_intervals=len(results),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limit='Fixed selected-history parent package and risk-overlay comparison, not causal or prospective'),indent=2)+'\n')
    print(pd.DataFrame(results).query("group=='ten_seed' and block==21")[[ 'allocation','policy','comparator','mean_difference','mean_low','mean_high']].to_string(index=False))


if __name__=='__main__':main()
