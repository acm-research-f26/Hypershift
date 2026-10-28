"""Fixed same-industry-target equal-weight controls on original saved scores."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data
from rishi_mechanism_batch1 import stationary_indices,paired_contrast


def matched_control(original,eligible,industry):
    w=np.asarray(original,float);eligible=np.asarray(eligible,bool);industry=np.asarray(industry,float)
    if w.shape!=eligible.shape or w.shape!=industry.shape or not np.isfinite(w).all() or np.any(w<0) or np.any(w[~eligible]!=0) or not np.isclose(w.sum(),1,atol=1e-12):raise ValueError('Invalid original eligible weights')
    groups=np.where(np.isfinite(industry),industry,-1);result=np.zeros_like(w)
    for group in np.unique(groups[eligible]):
        members=eligible&(groups==group);mass=w[members].sum();result[members]=mass/members.sum()
        if not np.isclose(result[members].sum(),mass,atol=1e-12):raise ValueError('Group mismatch')
    if not np.isclose(result.sum(),1,atol=1e-12) or np.any(result[~eligible]!=0):raise ValueError('Invalid matched weights')
    return result


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch24_industry_matched';large=Path('results/RISHI_INDUSTRY_MATCHED_20261028')
    for p in (out,large):
        if p.exists():raise FileExistsError(p)
        p.mkdir(parents=True)
    source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS');detail=doc/'execution_batch21_saved_industry/DETAIL.csv';summary=doc/'execution_batch21_saved_industry/SUMMARY.csv'
    classes=pd.read_csv(detail,float_precision='round_trip');classes=classes[classes.model=='E'];cut=pd.read_csv(summary);cut=cut[cut.model=='E'].sort_values('year')
    dates=np.array(cut.signal_date,dtype='datetime64[D]');data=load_data();tickers=data['panel'].tickers;curves=pd.read_csv(source/'ACCOUNT_CURVES.csv')
    all_inputs=[detail,summary,source/'ACCOUNT_CURVES.csv',Path(__file__),doc/'INDUSTRY_MATCHED_CONTROL_PROTOCOL.md',Path('research/core/ndx_drawdown_engine.py')]
    expected={};saved_ids=None
    for name in ('E','E_Hlabel'):
        path=source/f'scores_{name}_ten_seed.npz';all_inputs.append(path)
        with np.load(path,allow_pickle=False) as z:ids=z['ids'].copy();scores=z['scores'].copy()
        if saved_ids is None:saved_ids=ids
        elif not np.array_equal(saved_ids,ids):raise ValueError('Forecast origin mismatch')
        weekly=engine.allocations(data,ids,scores,'equal10');exposure=np.ones(len(data['panel'].dates))
        metrics,returns,ledger=engine.simulate(data,ids,weekly,exposure,detailed=True)
        prior=curves[curves.id==f'{name}/ten_seed/10'];previous_ledger=source/f'ledger_{name}_ten_seed_10.csv';all_inputs.append(previous_ledger)
        old=pd.read_csv(previous_ledger)
        if prior.date.tolist()!=ledger.date.tolist() or old.date.tolist()!=ledger.date.tolist():raise ValueError('Original replay calendar mismatch')
        err=float(np.max(np.abs(returns-prior.net.to_numpy())));nav=float(np.max(np.abs(ledger.nav.to_numpy()-old.nav.to_numpy()))/1_000_000.)
        if err>1e-10 or nav>1e-8:
            (out/'REPLAY_FAILED.json').write_text(json.dumps(dict(model=name,daily_error=err,relative_nav_error=nav),indent=2)+'\n')
            raise ValueError('Original replay gate failed; controls not scored')
        expected[name]=(ids,weekly,metrics,returns,ledger,dict(model=name,daily_error=err,relative_nav_error=nav))
    # Both original accounts must pass before any control is scored.
    control_records=[];weights_records=[];metrics_rows=[];contrasts=[];all_returns={};all_dates=None;replays=[]
    for name,(ids,weekly,original_metrics,original_returns,original_ledger,receipt) in expected.items():
        replays.append(receipt);controls=[]
        for row,origin in enumerate(ids):
            day=np.datetime64(data['panel'].dates[data['origin'][origin]].date(),'D');idx=int(np.searchsorted(dates,day,side='right')-1)
            if idx<0:raise ValueError('No available snapshot')
            known=classes[classes.year==int(cut.iloc[idx].year)].set_index('security_id').industry.reindex(tickers).to_numpy(float)
            eligible=data['mask'][origin];control=matched_control(weekly[row],eligible,known);controls.append(control)
            group=np.where(np.isfinite(known),known,-1)
            for label in np.unique(group[eligible]):
                m=eligible&(group==label);weights_records.append(dict(model=name,origin=int(origin),signal_date=str(day),industry='unclassified' if label==-1 else str(int(label)),
                    original_mass=float(weekly[row,m].sum()),control_mass=float(control[m].sum()),eligible_count=int(m.sum())))
        controls=np.stack(controls);np.savez_compressed(large/f'{name}_weights.npz',ids=ids,weights=controls,security_ids=np.asarray(tickers,dtype='U'))
        metrics,returns,ledger=engine.simulate(data,ids,controls,np.ones(len(data['panel'].dates)),detailed=True)
        if ledger.date.tolist()!=original_ledger.date.tolist() or ledger.cash.min() < -1e-5 or ledger.equity_exposure.iloc[-1]!=0:raise ValueError('Control feasibility/axes failure')
        for label,m,r,l in [('original',original_metrics,original_returns,original_ledger),('matched',metrics,returns,ledger)]:
            metrics_rows.append(dict(model=name,arm=label,**m,mean_exposure=float(l.equity_exposure.mean()),min_cash=float(l.cash.min()),
                max_issuer=float(l.largest_issuer_weight.max())))
            all_returns[f'{name}_{label}']=r;all_dates=l.date.tolist();l.to_csv(large/f'{name}_{label}_ledger.csv',index=False)
        for block in (5,21,63):
            draws=stationary_indices(len(returns),10000,block,27102028)
            contrast=paired_contrast(original_returns,returns,draws,252)
            contrasts.append(dict(model=name,block=block,**contrast))
        for label,r in [('original',original_returns),('matched',returns)]:
            f=pd.DataFrame({'date':pd.to_datetime(all_dates),'net':r})
            for year,g in f.groupby(f.date.dt.year):control_records.append(dict(model=name,arm=label,year=int(year),sessions=len(g),calendar_return=float(np.prod(1+g.net)-1)))
    pd.DataFrame(replays).to_csv(out/'ORIGINAL_REPLAY.csv',index=False);pd.DataFrame(weights_records).to_csv(out/'GROUP_TARGETS.csv',index=False)
    pd.DataFrame(metrics_rows).to_csv(out/'METRICS.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'PAIRED_CONTRASTS.csv',index=False)
    pd.DataFrame(control_records).to_csv(out/'CALENDAR.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',dates=np.asarray(all_dates,dtype='U'),**all_returns)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=4,origins=len(saved_ids),replay=replays,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in all_inputs},
        limits='Matched decision-time targets, not realized exposure or causal industry effect; selected development history'),indent=2)+'\n')
    print(pd.DataFrame(metrics_rows)[['model','arm','sharpe','annualized_return','maximum_drawdown','turnover_annual','mean_exposure']].to_string(index=False))
    print(pd.DataFrame(contrasts)[['model','block','mean_difference','mean_low','mean_high']].to_string(index=False))


if __name__=='__main__':main()
