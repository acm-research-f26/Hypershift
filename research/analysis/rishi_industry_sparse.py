"""Fixed same-group sparse random-name controls; no model retraining."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from rishi_mechanism_batch1 import stationary_indices,paired_contrast


def priorities(year,index,n):return np.random.default_rng(np.random.SeedSequence([27102028,int(year),int(index)])).random(n)


def sparse_control(original,eligible,industry,priority):
    w=np.asarray(original,float);eligible=np.asarray(eligible,bool);industry=np.asarray(industry,float);priority=np.asarray(priority,float)
    if any(x.shape!=w.shape for x in (eligible,industry,priority)) or not np.isfinite(w).all() or not np.isfinite(priority).all() or np.any(w<0) or np.any(w[~eligible]!=0) or not np.isclose(w.sum(),1,rtol=0,atol=1e-12):raise ValueError('Invalid input')
    group=np.where(np.isfinite(industry),industry,-1);result=np.zeros_like(w)
    for label in np.unique(group[eligible]):
        members=np.flatnonzero(eligible&(group==label));weights=np.sort(w[members][w[members]>0])[::-1]
        chosen=members[np.argsort(priority[members],kind='stable')[:len(weights)]];result[chosen]=weights
        if not np.array_equal(np.sort(result[members][result[members]>0]),np.sort(weights)):raise ValueError('Group multiset mismatch')
    if np.any(result[~eligible]!=0) or not np.isclose(result.sum(),1,rtol=0,atol=1e-12):raise ValueError('Invalid control')
    return result


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch25_industry_sparse';large=Path('results/RISHI_INDUSTRY_SPARSE_20261028')
    for p in (out,large):
        if p.exists():raise FileExistsError(p)
        p.mkdir(parents=True)
    source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS');detail=doc/'execution_batch21_saved_industry/DETAIL.csv';summary=doc/'execution_batch21_saved_industry/SUMMARY.csv'
    c=pd.read_csv(detail,float_precision='round_trip');c=c[c.model=='E'];cut=pd.read_csv(summary);cut=cut[cut.model=='E'].sort_values('year');cutoffs=np.array(cut.signal_date,dtype='datetime64[D]')
    data=load_data();tickers=data['panel'].tickers;curves=pd.read_csv(source/'ACCOUNT_CURVES.csv');expected={};inputs=[detail,summary,source/'ACCOUNT_CURVES.csv',Path(__file__),doc/'INDUSTRY_SPARSE_RANDOM_PROTOCOL.md',Path('research/core/ndx_drawdown_engine.py')]
    for model in ('E','E_Hlabel'):
        score_path=source/f'scores_{model}_ten_seed.npz';ledger_path=source/f'ledger_{model}_ten_seed_10.csv';inputs.extend([score_path,ledger_path])
        with np.load(score_path,allow_pickle=False) as z:ids=z['ids'].copy();scores=z['scores'].copy()
        weekly=engine.allocations(data,ids,scores,'equal10');metrics,returns,ledger=engine.simulate(data,ids,weekly,np.ones(len(data['panel'].dates)),detailed=True)
        prior=curves[curves.id==f'{model}/ten_seed/10'];old=pd.read_csv(ledger_path)
        if ledger.date.tolist()!=prior.date.tolist() or ledger.date.tolist()!=old.date.tolist():raise ValueError('Replay date mismatch')
        err=float(np.max(abs(returns-prior.net.to_numpy())));nav=float(np.max(abs(ledger.nav.to_numpy()-old.nav.to_numpy()))/1_000_000.)
        if err>1e-10 or nav>1e-8:
            (out/'REPLAY_FAILED.json').write_text(json.dumps(dict(model=model,daily_error=err,relative_nav_error=nav),indent=2)+'\n')
            raise ValueError('Original replay failed before control scoring')
        expected[model]=(ids,weekly,metrics,returns,ledger,dict(model=model,daily_error=err,relative_nav_error=nav))
    if not np.array_equal(expected['E'][0],expected['E_Hlabel'][0]):raise ValueError('Model origin mismatch')
    dates=expected['E'][4].date.tolist();metrics_rows=[];paired=[];calendar=[];paths={};replay=[];constraints=[]
    for model,(ids,weekly,original_metrics,original_returns,original_ledger,rep) in expected.items():
        replay.append(rep);paths[f'{model}_original']=original_returns;ledger_path=large/f'{model}_original_ledger.csv';original_ledger.to_csv(ledger_path,index=False)
        controls=[]
        for control in range(20):
            w=[]
            for row,origin in enumerate(ids):
                day=np.datetime64(data['panel'].dates[data['origin'][origin]].date(),'D');idx=int(np.searchsorted(cutoffs,day,side='right')-1)
                if idx<0:raise ValueError('No available SIC snapshot')
                year=int(cut.iloc[idx].year);industry=c[c.year==year].set_index('security_id').industry.reindex(tickers).to_numpy(float)
                eligible=data['mask'][origin];w.append(sparse_control(weekly[row],eligible,industry,priorities(year,control,len(tickers))))
                groups=np.where(np.isfinite(industry),industry,-1)
                for group in np.unique(groups[eligible]):
                    active=eligible&(groups==group);a=weekly[row,active];b=w[-1][active]
                    constraints.append(dict(model=model,control=control,origin=int(origin),industry='unclassified' if group==-1 else str(int(group)),
                        group_error=float(abs(a.sum()-b.sum())),original_count=int((a>0).sum()),control_count=int((b>0).sum())))
            w=np.stack(w);np.savez_compressed(large/f'{model}_{control:02d}_weights.npz',ids=ids,weights=w,security_ids=np.asarray(tickers,dtype='U'))
            metric,r,ledger=engine.simulate(data,ids,w,np.ones(len(data['panel'].dates)),detailed=True)
            if ledger.date.tolist()!=dates or ledger.cash.min()<-1e-5 or ledger.equity_exposure.iloc[-1]!=0:raise ValueError('Account feasibility')
            paths[f'{model}_{control:02d}']=r;controls.append(r);ledger.to_csv(large/f'{model}_{control:02d}_ledger.csv',index=False)
            metrics_rows.append(dict(model=model,control=control,**metric,mean_exposure=float(ledger.equity_exposure.mean()),
                min_cash=float(ledger.cash.min()),original_minus_control_mean=float((original_returns-r).mean()*252)))
            for year,g in pd.DataFrame({'date':pd.to_datetime(dates),'net':r}).groupby(lambda i:pd.Timestamp(dates[i]).year):
                calendar.append(dict(model=model,control=control,year=int(year),calendar_return=float(np.prod(1+g.net)-1)))
            print(model,'control',control,'complete',flush=True)
        average=np.mean(np.stack(controls),axis=0);paths[f'{model}_diagnostic_average']=average
        for block in (5,21,63):
            draw=stationary_indices(len(average),10000,block,27102028);paired.append(dict(model=model,block=block,**paired_contrast(original_returns,average,draw,252)))
    constraints=pd.DataFrame(constraints)
    if constraints.group_error.max()>1e-12 or not (constraints.original_count==constraints.control_count).all():raise ValueError('Group constraints failed')
    np.savez_compressed(large/'PATHS.npz',dates=np.asarray(dates,dtype='U'),**paths)
    pd.DataFrame(metrics_rows).to_csv(out/'CONTROL_METRICS.csv',index=False);pd.DataFrame(paired).to_csv(out/'PAIRED_AVERAGES.csv',index=False)
    pd.DataFrame(calendar).to_csv(out/'CONTROL_CALENDAR.csv',index=False);pd.DataFrame(replay).to_csv(out/'ORIGINAL_REPLAY.csv',index=False)
    constraints.to_csv(out/'GROUP_CONSTRAINTS.csv',index=False)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=40,controls_per_model=20,group_constraints=len(constraints),
        max_group_error=float(constraints.group_error.max()),input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limits='Average of twenty full-size accounts is not an executable pooled $1m portfolio; no original gate change'),indent=2)+'\n')
    print(pd.DataFrame(paired)[['model','block','mean_difference','mean_low','mean_high']].to_string(index=False))


if __name__=='__main__':main()
