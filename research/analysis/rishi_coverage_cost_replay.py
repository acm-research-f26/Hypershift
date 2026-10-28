"""Fixed source-engine cost scenarios on saved industry-matched targets."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from rishi_mechanism_risk import fee_view
from rishi_mechanism_batch1 import stationary_indices,paired_contrast

DOC=Path('docs/rishi_mechanism_20261027')
SOURCE=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
MATCHED=Path('results/RISHI_INDUSTRY_COVERAGE_20261028')


def main():
    out=DOC/'execution_batch36_coverage_cost';large=Path('results/RISHI_COVERAGE_COST_20261028')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();old=np.load(MATCHED/'PATHS.npz',allow_pickle=False)
    prior=pd.read_csv(DOC/'execution_batch34_industry_coverage/PAIRED_CONTRASTS.csv')
    inputs=[MATCHED/'PATHS.npz',DOC/'execution_batch34_industry_coverage/PAIRED_CONTRASTS.csv',Path(__file__),
        DOC/'COVERAGE_COST_REPLAY_PROTOCOL.md',Path('research/analysis/rishi_mechanism_risk.py'),
        Path('research/analysis/rishi_mechanism_batch1.py'),Path('research/core/ndx_drawdown_engine.py'),
        Path('research/core/pit_research_engine.py'),Path('research/core/research_engine.py')]
    base=Path('data/nasdaq_monthly_pit/assembled')
    inputs.extend(base/f for f in ('manifest.json','economic_close.csv','dollar_volume.csv','tradable.csv','membership.csv','etf_adj_close.csv'))
    books=[]
    for model in ('E','E_Hlabel'):
        score=SOURCE/f'scores_{model}_ten_seed.npz';weight=MATCHED/f'{model}_coverage6_weights.npz';inputs.extend([score,weight])
        with np.load(score,allow_pickle=False) as z:ids=z['ids'].copy();original=engine.allocations(data,ids,z['scores'],'equal10')
        with np.load(weight,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['security_ids'],data['panel'].tickers);control=z['weights'].copy()
        books.extend([(model,'original',ids,original),(model,'coverage6',ids,control)])
    records=[];calendar=[];all_paths={};replay=[]
    for mode in ('modeled1','zero','modeled2','modeled4'):
        for model,arm,ids,weights in books:
            with fee_view(mode) as multiplier:
                metrics,returns,ledger=engine.simulate(data,ids,weights,np.ones(len(data['panel'].dates)),multiplier=multiplier,detailed=True)
            if ledger.date.tolist()!=old['dates'].tolist() or ledger.cash.min() < -1e-5 or ledger.equity_exposure.iloc[-1]!=0:raise ValueError('Account dates/cash/liquidation')
            if mode=='modeled1':
                old_ledger=pd.read_csv(MATCHED/f'{model}_{arm}_ledger.csv');err=float(np.max(np.abs(returns-old[f'{model}_{arm}'])));naverr=float(np.max(np.abs(ledger.nav-old_ledger.nav))/1e6)
                replay.append(dict(model=model,arm=arm,daily_error=err,relative_nav_error=naverr))
                if err>1e-10 or naverr>1e-8:raise ValueError('Base replay failure; other modes not scored')
            key=f'{model}_{arm}_{mode}';all_paths[key]=returns
            ledger.to_csv(large/f'{key}_ledger.csv',index=False)
            records.append(dict(model=model,arm=arm,mode=mode,**metrics,mean_exposure=float(ledger.equity_exposure.mean()),min_cash=float(ledger.cash.min())))
            years=pd.DatetimeIndex(old['dates']).year
            for year in (2022,2023,2024,2025):calendar.append(dict(model=model,arm=arm,mode=mode,year=year,calendar_return=float(np.prod(1+returns[years==year])-1)))
            print(key,flush=True)
    contrasts=[]
    for model in ('E','E_Hlabel'):
        for mode in ('modeled1','zero','modeled2','modeled4'):
            a=all_paths[f'{model}_original_{mode}'];b=all_paths[f'{model}_coverage6_{mode}']
            for block in (5,21,63):
                result=paired_contrast(a,b,stationary_indices(len(a),10000,block,27102028),252)
                if mode=='modeled1':
                    ref=prior[(prior.model==model)&(prior.block==block)].iloc[0]
                    np.testing.assert_allclose([result['mean_difference'],result['mean_low'],result['mean_high']],[ref.mean_difference,ref.mean_low,ref.mean_high],rtol=0,atol=1e-12)
                contrasts.append(dict(model=model,mode=mode,block=block,**result))
    pd.DataFrame(records).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'CONTRASTS.csv',index=False)
    pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(replay).to_csv(out/'BASE_REPLAY.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',dates=old['dates'],**all_paths)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=len(records),days=len(old['dates']),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limits='Fixed historical cost scenarios with source execution; zero cost is optimistic'),indent=2)+'\n')
    print(pd.DataFrame(contrasts).query('block==21')[['model','mode','mean_difference','mean_low','mean_high']].to_string(index=False))


if __name__=='__main__':main()
