"""Fixed sparse industry-matched controls using previously exported signals."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from rishi_industry_sparse import priorities
from rishi_mechanism_batch1 import stationary_indices,paired_contrast


def signal_control(original,eligible,industry,score,priority):
    w=np.asarray(original,float);eligible=np.asarray(eligible,bool);industry=np.asarray(industry,float);score=np.asarray(score,float);priority=np.asarray(priority,float)
    if any(x.shape!=w.shape for x in (eligible,industry,score,priority)) or not np.isfinite(w).all() or not np.isfinite(score[eligible]).all() or not np.isfinite(priority).all() or np.any(w<0) or np.any(w[~eligible]!=0) or not np.isclose(w.sum(),1,rtol=0,atol=1e-12):raise ValueError('Invalid control inputs')
    groups=np.where(np.isfinite(industry),industry,-1);result=np.zeros_like(w)
    for label in np.unique(groups[eligible]):
        members=np.flatnonzero(eligible&(groups==label));weights=np.sort(w[members][w[members]>0])[::-1]
        chosen=members[np.lexsort((priority[members],-score[members]))[:len(weights)]];result[chosen]=weights
        if not np.array_equal(np.sort(result[members][result[members]>0]),np.sort(weights)):raise ValueError('Group multiset mismatch')
    if np.any(result[~eligible]!=0) or not np.isclose(result.sum(),1,rtol=0,atol=1e-12):raise ValueError('Invalid matched weights')
    return result


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch26_industry_simple';large=Path('results/RISHI_INDUSTRY_SIMPLE_20261028')
    for p in (out,large):
        if p.exists():raise FileExistsError(p)
        p.mkdir(parents=True)
    root=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS');detail=doc/'execution_batch21_saved_industry/DETAIL.csv';summary=doc/'execution_batch21_saved_industry/SUMMARY.csv'
    classes=pd.read_csv(detail,float_precision='round_trip');classes=classes[classes.model=='E'];cut=pd.read_csv(summary);cut=cut[cut.model=='E'].sort_values('year');cutoffs=np.array(cut.signal_date,dtype='datetime64[D]')
    data=load_data();tickers=data['panel'].tickers;curves=pd.read_csv(root/'ACCOUNT_CURVES.csv');fallback=doc/'execution_batch2_exports/COMPARATOR_FALLBACKS.csv'
    inputs=[detail,summary,root/'ACCOUNT_CURVES.csv',fallback,Path(__file__),doc/'INDUSTRY_SIMPLE_SIGNAL_PROTOCOL.md',Path('research/core/ndx_drawdown_engine.py'),Path('research/analysis/rishi_industry_sparse.py')]
    expected={};all_ids=None
    for model in ('E','E_Hlabel'):
        score=root/f'scores_{model}_ten_seed.npz';ledger_path=root/f'ledger_{model}_ten_seed_10.csv';inputs.extend([score,ledger_path])
        with np.load(score,allow_pickle=False) as z:ids=z['ids'].copy();raw=z['scores'].copy()
        if all_ids is None:all_ids=ids
        elif not np.array_equal(ids,all_ids):raise ValueError('Original score axes mismatch')
        weekly=engine.allocations(data,ids,raw,'equal10');metric,r,ledger=engine.simulate(data,ids,weekly,np.ones(len(data['panel'].dates)),detailed=True)
        prior=curves[curves.id==f'{model}/ten_seed/10'];old=pd.read_csv(ledger_path)
        if prior.date.tolist()!=ledger.date.tolist() or old.date.tolist()!=ledger.date.tolist():raise ValueError('Original replay calendar')
        err=float(np.max(abs(r-prior.net.to_numpy())));nav=float(np.max(abs(ledger.nav-old.nav))/1_000_000.)
        if err>1e-10 or nav>1e-8:
            (out/'REPLAY_FAILED.json').write_text(json.dumps({'model':model,'daily_error':err,'nav_error':nav},indent=2)+'\n');raise ValueError('Original replay failed; no control scoring')
        expected[model]=(weekly,metric,r,ledger,dict(model=model,daily_error=err,relative_nav_error=nav))
    signals={}
    for signal in ('Hlabel','REV5','MOM126','ridge5'):
        path=root/f'scores_{signal}_ten_seed.npz';inputs.append(path)
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['ids'],all_ids) or z['scores'].shape!=(len(all_ids),len(tickers)):raise ValueError('Signal axes mismatch')
            signals[signal]=z['scores'].copy()
    rows=[];constraints=[];metrics=[];contrasts=[];calendar=[];paths={};replay=[];dates=None
    for model,(weekly,original_metrics,original_returns,original_ledger,receipt) in expected.items():
        replay.append(receipt);paths[model+'_original']=original_returns;dates=original_ledger.date.tolist();original_ledger.to_csv(large/f'{model}_original_ledger.csv',index=False)
        for signal,score in signals.items():
            weights=[]
            for row,origin in enumerate(all_ids):
                day=np.datetime64(data['panel'].dates[data['origin'][origin]].date(),'D');idx=int(np.searchsorted(cutoffs,day,side='right')-1)
                if idx<0:raise ValueError('No dated SIC snapshot')
                year=int(cut.iloc[idx].year);industry=classes[classes.year==year].set_index('security_id').industry.reindex(tickers).to_numpy(float)
                eligible=data['mask'][origin];w=signal_control(weekly[row],eligible,industry,score[row],priorities(year,0,len(tickers)));weights.append(w)
                rows.append(dict(model=model,signal=signal,origin=int(origin),shared_names=int(((weekly[row]>0)&(w>0)).sum()),
                    target_weight_overlap=float(np.minimum(weekly[row],w).sum()),original_names=int((weekly[row]>0).sum()),alternative_names=int((w>0).sum()),
                    unknown_target=float(weekly[row,eligible&~np.isfinite(industry)].sum())))
                groups=np.where(np.isfinite(industry),industry,-1)
                for group in np.unique(groups[eligible]):
                    mask=eligible&(groups==group);a=weekly[row,mask];b=w[mask]
                    constraints.append(dict(model=model,signal=signal,origin=int(origin),industry='unclassified' if group==-1 else str(int(group)),
                        group_error=float(abs(a.sum()-b.sum())),original_count=int((a>0).sum()),alternative_count=int((b>0).sum())))
            weights=np.stack(weights);np.savez_compressed(large/f'{model}_{signal}_weights.npz',ids=all_ids,weights=weights,security_ids=np.asarray(tickers,dtype='U'))
            metric,r,ledger=engine.simulate(data,all_ids,weights,np.ones(len(data['panel'].dates)),detailed=True)
            if ledger.date.tolist()!=dates or ledger.cash.min()<-1e-5 or ledger.equity_exposure.iloc[-1]!=0:raise ValueError('Control feasibility')
            ledger.to_csv(large/f'{model}_{signal}_ledger.csv',index=False);paths[f'{model}_{signal}']=r
            metrics.append(dict(model=model,signal=signal,**metric,mean_exposure=float(ledger.equity_exposure.mean()),min_cash=float(ledger.cash.min())))
            for block in (5,21,63):
                draws=stationary_indices(len(r),10000,block,27102028);contrasts.append(dict(model=model,signal=signal,block=block,**paired_contrast(original_returns,r,draws,252)))
            for year,g in pd.DataFrame({'date':pd.to_datetime(dates),'net':r}).groupby(lambda i:pd.Timestamp(dates[i]).year):calendar.append(dict(model=model,signal=signal,year=int(year),calendar_return=float(np.prod(1+g.net)-1)))
            print(model,signal,'complete',flush=True)
    cons=pd.DataFrame(constraints)
    if cons.group_error.max()>1e-12 or not (cons.original_count==cons.alternative_count).all():raise ValueError('Constraint failure')
    pd.DataFrame(rows).to_csv(out/'OVERLAP.csv',index=False);cons.to_csv(out/'GROUP_CONSTRAINTS.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'CONTROL_METRICS.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'PAIRED_CONTRASTS.csv',index=False)
    pd.DataFrame(calendar).to_csv(out/'CONTROL_CALENDAR.csv',index=False);pd.DataFrame(replay).to_csv(out/'ORIGINAL_REPLAY.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',dates=np.asarray(dates,dtype='U'),**paths)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=8,group_constraints=len(cons),max_group_error=float(cons.group_error.max()),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limits='Exploratory within-industry simple-signal controls, not neural causation or original mechanism gate'),indent=2)+'\n')
    print(pd.DataFrame(contrasts).query('block == 21')[['model','signal','mean_difference','mean_low','mean_high']].to_string(index=False))


if __name__=='__main__':main()
