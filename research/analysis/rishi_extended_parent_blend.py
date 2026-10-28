"""Fixed saved-parent-view blends with the same saved dynamic forecasts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data,fold
from run_robust_research import ensemble_predictions
from rishi_mechanism_batch1 import stationary_indices,paired_contrast

DOC=Path('docs/rishi_mechanism_20261027');RUN=Path('runs/ndx_extended_hours_20261024')
SOURCE=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
SEEDS=[7,19,42,61,97,1828801906,1868046758,239154507,287727767,981756941]
GROUPS={'original_five':SEEDS[:5],'new_five':SEEDS[5:],'ten_seed':SEEDS}
VIEWS=('daily','after_hours','after_hours_premarket')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=DOC/'execution_batch42_extended_parent';large=Path('results/RISHI_EXTENDED_PARENT_BLEND_20261028')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();ids=np.concatenate([fold(data,y)[2] for y in (2022,2023,2024,2025)])
    if len(ids)!=201:raise ValueError('Unexpected origin count')
    mask=data['mask'][ids];locked=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())['hashes']
    old=np.load(RUN/'scores.npz',allow_pickle=False);np.testing.assert_array_equal(old['ids'],ids)
    inputs=[RUN/'FITS_COMPLETE_LOCK.json',RUN/'scores.npz',SOURCE/'ACCOUNT_CURVES.csv',Path(__file__),DOC/'EXTENDED_PARENT_BLEND_PROTOCOL.md',
        Path('research/training/run_robust_research.py'),Path('research/core/ndx_drawdown_engine.py'),Path('research/core/pit_research_engine.py'),
        Path('research/core/research_engine.py'),Path('research/analysis/rishi_mechanism_batch1.py')]
    base=Path('data/nasdaq_monthly_pit/assembled');inputs.extend(base/f for f in ('manifest.json','economic_close.csv','dollar_volume.csv','tradable.csv','membership.csv','etf_adj_close.csv'))
    scores={};max_score_error=0.;verified=0
    for view in VIEWS:
        raw={}
        for seed in SEEDS:
            pieces=[]
            for year in (2022,2023,2024,2025):
                path=RUN/view/str(year)/f'{seed}_predictions.npz';rel=f'{view}\\{year}\\{seed}_predictions.npz'
                if sha(path)!=locked[rel]:raise ValueError('Parent forecast hash mismatch')
                inputs.append(path);verified+=1
                with np.load(path,allow_pickle=False) as z:
                    test=z['test'].copy();count=len(fold(data,year)[2])
                    if test.shape!=(count,len(data['panel'].tickers)):raise ValueError('Parent forecast shape')
                    pieces.append(test)
            raw[seed]=np.concatenate(pieces)
        for group,seeds in GROUPS.items():
            parent=ensemble_predictions([raw[s] for s in seeds],mask)
            if not np.array_equal(parent,old[view+'__'+group]):raise ValueError('Parent published score mismatch')
            dynamic_path=SOURCE/f'scores_D_{group}.npz'
            with np.load(dynamic_path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids);dynamic=z['scores'].copy()
            blend=.5*parent+.5*dynamic;scores[view,group]=blend
            if view=='after_hours_premarket':
                for name,value in [('B',parent),('E',blend)]:
                    path=SOURCE/f'scores_{name}_{group}.npz';inputs.append(path)
                    with np.load(path,allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],ids)
                        err=float(np.max(abs(value-z['scores'])));max_score_error=max(max_score_error,err)
                        if err>1e-14:raise ValueError('Previously verified source score mismatch')
            inputs.append(dynamic_path)
    if verified!=120:raise ValueError('Incomplete source forecast inventory')
    original=pd.read_csv(SOURCE/'ACCOUNT_CURVES.csv');results=[];replays=[];calendar=[];overlap=[];paths={};weights_all={};dates=None
    for view in ('after_hours_premarket','daily','after_hours'):
        for group in GROUPS:
            for k in (5,10):
                score=scores[view,group];weights=engine.allocations(data,ids,score,f'equal{k}')
                metrics,net,ledger=engine.simulate(data,ids,weights,np.ones(len(data['panel'].dates)),detailed=True)
                if dates is None:dates=ledger.date.to_numpy(dtype='U')
                if ledger.date.tolist()!=dates.tolist() or ledger.cash.min() < -1e-5 or ledger.equity_exposure.iloc[-1]!=0:raise ValueError('Feasibility/calendar')
                if view=='after_hours_premarket':
                    prior=original[original.id==f'E/{group}/{k}'];ledger_path=SOURCE/f'ledger_E_{group}_{k}.csv';inputs.append(ledger_path)
                    ref=pd.read_csv(ledger_path)
                    if prior.date.tolist()!=dates.tolist() or ref.date.tolist()!=dates.tolist():raise ValueError('Base date mismatch')
                    err=float(np.max(abs(net-prior.net.to_numpy())));naverr=float(np.max(abs(ledger.nav-ref.nav))/1e6)
                    replays.append(dict(group=group,k=k,daily_error=err,nav_error=naverr))
                    if err>1e-10 or naverr>1e-8:raise ValueError('Base replay failure; stop')
                key=f'{view}__{group}__k{k}';paths[key]=net;weights_all[key]=weights
                ledger.to_csv(large/f'{key}_ledger.csv',index=False)
                results.append(dict(view=view,group=group,k=k,**metrics,mean_exposure=float(ledger.equity_exposure.mean()),min_cash=float(ledger.cash.min())))
                years=pd.DatetimeIndex(dates).year
                for year in (2022,2023,2024,2025):calendar.append(dict(view=view,group=group,k=k,year=year,calendar_return=float(np.prod(1+net[years==year])-1)))
                if view!='after_hours_premarket':
                    basew=weights_all[f'after_hours_premarket__{group}__k{k}']
                    for i,origin in enumerate(ids):overlap.append(dict(view=view,group=group,k=k,origin=int(origin),
                        shared_names=int(((weights[i]>0)&(basew[i]>0)).sum()),weight_overlap=float(np.minimum(weights[i],basew[i]).sum())))
                print(key,flush=True)
    contrasts=[]
    for group in GROUPS:
        for k in (5,10):
            a=paths[f'after_hours_premarket__{group}__k{k}']
            for view in ('daily','after_hours'):
                b=paths[f'{view}__{group}__k{k}']
                for block in (5,21,63):
                    contrasts.append(dict(group=group,k=k,comparator=view,block=block,**paired_contrast(a,b,stationary_indices(len(dates),10000,block,27102028),252)))
    pd.DataFrame(results).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(replays).to_csv(out/'BASE_REPLAY.csv',index=False)
    pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False);pd.DataFrame(overlap).to_csv(out/'OVERLAP.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'CONTRASTS.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',dates=dates,**paths)
    np.savez_compressed(large/'TARGETS.npz',ids=ids,security_ids=np.asarray(data['panel'].tickers,dtype='U'),**weights_all)
    np.savez_compressed(large/'SCORES.npz',ids=ids,**{f'{v}__{g}':a for (v,g),a in scores.items()})
    (out/'RESULTS.json').write_text(json.dumps(dict(parent_predictions_verified=verified,accounts=len(results),paired_intervals=len(contrasts),max_prior_score_error=max_score_error,
        input_sha256={str(p):sha(p) for p in inputs},limits='Parent-view swap in same saved-dynamic blend; selected history'),indent=2)+'\n')
    print(pd.DataFrame(contrasts).query("group=='ten_seed' and k==10 and block==21")[[ 'comparator','mean_difference','mean_low','mean_high']].to_string(index=False))


if __name__=='__main__':main()
