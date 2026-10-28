"""Post-score issuer exclusions selected solely from training-label means."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data,fold
from rishi_mechanism_batch1 import stationary_indices,paired_contrast

DOC=Path('docs/rishi_mechanism_20261027');SOURCE=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')


def selected_issuers(tickers,issuers,scores,eligible,counts):
    if not all(len(x)==len(tickers) for x in (issuers,scores,eligible,counts)):raise ValueError('Axes mismatch')
    records={}
    for ticker,issuer,score,ok,count in zip(tickers,issuers,scores,eligible,counts):
        if ok and count>=30:
            if not np.isfinite(score):raise ValueError('Missing qualifying score')
            if issuer not in records or score>records[issuer][0] or (score==records[issuer][0] and ticker<records[issuer][1]):
                records[issuer]=(float(score),str(ticker),int(count))
    ordered=sorted(records,key=lambda issuer:(-records[issuer][0],issuer))
    if len(ordered)<3:raise ValueError('Insufficient qualifying issuers')
    return [(issuer,*records[issuer]) for issuer in ordered[:3]]


def main():
    out=DOC/'execution_batch39_training_exclusion';large=Path('results/RISHI_TRAINING_EXCLUSION_20261028')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();tickers=np.asarray(data['panel'].tickers);issuers=np.asarray([engine.ISSUERS.get(t,t) for t in tickers])
    support=pd.read_csv(DOC/'execution_batch2_exports/TRAINING_SUPPORT.csv');support=support[support.family=='Hlabel']
    if support.duplicated(['year','ticker']).any():raise ValueError('Duplicate support IDs')
    arrays={};ids=None;files=[]
    for name in ('E','E_Hlabel','Hlabel'):
        path=SOURCE/f'scores_{name}_ten_seed.npz';files.append(path)
        with np.load(path,allow_pickle=False) as z:
            if ids is None:ids=z['ids'].copy()
            else:np.testing.assert_array_equal(ids,z['ids'])
            arrays[name]=z['scores'].copy()
    if len(ids)!=201 or any(a.shape!=(len(ids),len(tickers)) for a in arrays.values()):raise ValueError('Score axes')
    years=data['panel'].dates[data['entry'][ids]].year.to_numpy();chosen={};selection=[]
    for year in (2022,2023,2024,2025):
        position=int(np.flatnonzero(years==year)[0]);origin=ids[position];train,_,_=fold(data,year)
        counts=data['labelmask'][train].sum(0)
        prior=support[support.year==year].set_index('ticker')['count'].reindex(tickers)
        if prior.isna().any():raise ValueError('Missing training support')
        np.testing.assert_array_equal(counts,prior.to_numpy())
        choices=selected_issuers(tickers,issuers,arrays['Hlabel'][position],data['mask'][origin],counts)
        chosen[year]=[x[0] for x in choices]
        for place,(issuer,score,ticker,count) in enumerate(choices,1):
            selection.append(dict(year=year,origin=int(origin),decision_date=str(data['panel'].dates[data['entry'][origin]].date()),
                order=place,issuer=issuer,representative=ticker,training_labels=count,hlabel_score=score))
    pd.DataFrame(selection).to_csv(out/'TRAINING_SELECTION.csv',index=False)
    masks={0:data['mask'][ids].copy()};mask_rows=[]
    for n in (1,3):
        masks[n]=data['mask'][ids].copy()
        for i,year in enumerate(years):
            removed=np.isin(issuers,chosen[year][:n]);masks[n][i]&=~removed
            mask_rows.append(dict(origin=int(ids[i]),year=int(year),excluded=n,removed_eligible=int((data['mask'][ids[i]]&removed).sum()),remaining=int(masks[n][i].sum())))
        if (masks[n].sum(1)<10).any():raise ValueError('Insufficient eligible stocks')
    pd.DataFrame(mask_rows).to_csv(out/'MASKS.csv',index=False)
    curves=pd.read_csv(SOURCE/'ACCOUNT_CURVES.csv');results=[];paths={};replays=[];calendar=[];dates=None;baseline_weights={};removed_targets=[]
    for n in (0,1,3):
        selection_data=data.copy();selection_data['mask']=data['mask'].copy();selection_data['mask'][ids]=masks[n]
        for model in ('E','E_Hlabel'):
            if not np.isfinite(arrays[model][masks[n]]).all():raise ValueError('Invalid eligible scores')
            for k in (5,10):
                weights=engine.allocations(selection_data,ids,arrays[model],f'equal{k}')
                if (weights[~masks[n]]!=0).any() or not np.allclose(weights.sum(1),1,rtol=0,atol=1e-12):raise ValueError('Invalid weights')
                if n==0:baseline_weights[model,k]=weights.copy()
                else:
                    prior_weights=baseline_weights[model,k]
                    for i,origin in enumerate(ids):
                        removed_targets.append(dict(model=model,k=k,exclude=n,origin=int(origin),
                            removed_original_target_mass=float(prior_weights[i,~masks[n][i]].sum()),
                            new_target_overlap=float(np.minimum(prior_weights[i],weights[i]).sum())))
                metrics,net,ledger=engine.simulate(data,ids,weights,np.ones(len(data['panel'].dates)),detailed=True)
                if ledger.cash.min() < -1e-5 or ledger.equity_exposure.iloc[-1]!=0:raise ValueError('Cash or terminal exposure')
                if dates is None:dates=ledger.date.to_numpy(dtype='U')
                if ledger.date.tolist()!=dates.tolist():raise ValueError('Calendar mismatch')
                if n==0:
                    old=curves[curves.id==f'{model}/ten_seed/{k}'];ref=pd.read_csv(SOURCE/f'ledger_{model}_ten_seed_{k}.csv')
                    if old.date.tolist()!=dates.tolist() or ref.date.tolist()!=dates.tolist():raise ValueError('Base date mismatch')
                    err=float(np.max(abs(net-old.net.to_numpy())));naverr=float(np.max(abs(ledger.nav-ref.nav))/1e6)
                    replays.append(dict(model=model,k=k,daily_error=err,nav_error=naverr))
                    if err>1e-10 or naverr>1e-8:raise ValueError('Base replay failed')
                key=f'{model}_k{k}_exclude{n}';paths[key]=net
                np.savez_compressed(large/f'{key}_weights.npz',ids=ids,weights=weights,security_ids=tickers.astype('U'))
                ledger.to_csv(large/f'{key}_ledger.csv',index=False)
                results.append(dict(model=model,k=k,exclude=n,**metrics,mean_exposure=float(ledger.equity_exposure.mean()),min_cash=float(ledger.cash.min())))
                for year in (2022,2023,2024,2025):
                    a=net[pd.DatetimeIndex(dates).year==year];calendar.append(dict(model=model,k=k,exclude=n,year=year,calendar_return=float(np.prod(1+a)-1)))
                print(key,flush=True)
    contrast=[]
    for n in (1,3):
        for k in (5,10):
            pairs=[(f'{model}_k{k}_exclude{n}',f'{model}_k{k}_exclude0','screen_minus_original') for model in ('E','E_Hlabel')]
            pairs.append((f'E_k{k}_exclude{n}',f'E_Hlabel_k{k}_exclude{n}','E_minus_E_Hlabel'))
            for left,right,kind in pairs:
                for block in (5,21,63):
                    result=paired_contrast(paths[left],paths[right],stationary_indices(len(dates),10000,block,27102028),252)
                    contrast.append(dict(left=left,right=right,kind=kind,k=k,exclude=n,block=block,**result))
    pd.DataFrame(results).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(contrast).to_csv(out/'CONTRASTS.csv',index=False)
    pd.DataFrame(replays).to_csv(out/'BASE_REPLAY.csv',index=False);pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False)
    pd.DataFrame(removed_targets).to_csv(out/'REMOVED_TARGETS.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',dates=dates,**paths)
    base=Path('data/nasdaq_monthly_pit/assembled');input_paths=files+[SOURCE/'ACCOUNT_CURVES.csv',DOC/'execution_batch2_exports/TRAINING_SUPPORT.csv',
        Path(__file__),DOC/'TRAINING_ISSUER_EXCLUSION_PROTOCOL.md',Path('research/core/ndx_drawdown_engine.py'),Path('research/core/pit_research_engine.py'),
        Path('research/core/research_engine.py'),Path('research/analysis/rishi_mechanism_batch1.py'),
        *(base/f for f in ('manifest.json','economic_close.csv','dollar_volume.csv','tradable.csv','membership.csv','etf_adj_close.csv'))]
    input_paths.extend(SOURCE/f'ledger_{m}_ten_seed_{k}.csv' for m in ('E','E_Hlabel') for k in (5,10))
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=len(results),contrasts=len(contrast),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
        limit='Post-score annual training-label issuer exclusions, not re-inference or causal removal'),indent=2)+'\n')
    print(pd.DataFrame(contrast).query('block==21')[['left','right','mean_difference','mean_low','mean_high']].to_string(index=False))


if __name__=='__main__':main()
