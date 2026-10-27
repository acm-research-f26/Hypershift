"""Matched saved-ranking allocation/exposure/cost accounts; frozen protocol14a910f."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import ndx_drawdown_engine as engine
from pit_research_engine import load_data,fold
from rishi_mechanism_batch1 import stationary_indices,paired_contrast,holm

DOC=Path('docs/rishi_mechanism_20261027')
SOURCE=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
GROUPS=['original_five','new_five','ten_seed']
POLICIES={
    'full':{},'constant75':{'constant':.75},'vol20':{'vol':'basket','window':20,'target':.2},
    'ma200':{'ma':200,'off':.5},'vol25trend':{'vol':'basket','window':20,'target':.25,'ma':200,'off':.5}}
SOURCE_NAMES={'full':'full','constant75':'constant_75','vol20':'basketvol20_target20_weekly',
              'ma200':'ma200_off50_weekly','vol25trend':'ma200off50_basketvol20_target25_weekly'}
COSTS=['modeled1','modeled2','modeled4','flat10','flat20','flat30','zero']


def calibrate_fixed_exposure(returns,training_origins,weights):
    returns=np.asarray(returns,float);weights=np.asarray(weights,float);active=weights>0;values=[]
    if not active.any() or not np.isclose(weights.sum(),1):raise ValueError('Normalized basket required')
    for day in training_origins:
        if day<19:continue
        history=returns[day-19:day+1,active];valid=np.isfinite(history).all(1)
        if valid.sum()>=15:values.append(float(np.std(history[valid]@weights[active],ddof=1)*np.sqrt(252)))
    fallback=len(values)<30;risk=.30 if fallback else max(float(np.median(values)),.05)
    return dict(risk=risk,exposure=min(1.,.2/risk),training_windows=len(values),fallback=fallback)


@contextmanager
def fee_view(mode):
    original=engine.cost_rate
    if mode.startswith('modeled'):
        yield float(mode.removeprefix('modeled'));return
    if mode!='zero' and mode not in ['flat10','flat20','flat30']:raise ValueError('Unknown cost view')
    rate=0. if mode=='zero' else int(mode.removeprefix('flat'))/10000
    def fixed(dollars,adv,multiplier=1.):return np.full(np.broadcast_shapes(np.shape(dollars),np.shape(adv)),rate,dtype=float)
    try:
        engine.cost_rate=fixed;yield 1.
    finally:engine.cost_rate=original


def calendar_attribution(path,dates):
    frame=pd.DataFrame(dict(net=np.asarray(path,float),year=pd.DatetimeIndex(dates).year))
    return pd.DataFrame([dict(year=int(y),sessions=len(s),calendar_return=float(np.prod(1+s.net)-1),
        log_growth=float(np.log1p(s.net).sum())) for y,s in frame.groupby('year')])


def qqq_data(data):
    root=Path(data['manifest']['root']);dates=data['panel'].dates
    close=pd.read_csv(root/'etf_adj_close.csv',index_col=0,parse_dates=True)['QQQ']
    volume=pd.read_csv(root/'etf_dollar_volume.csv',index_col=0,parse_dates=True)['QQQ']
    if not close.index.equals(dates) or not volume.index.equals(dates):raise ValueError('ETF date axes')
    result=data.copy();result.update(panel=SimpleNamespace(dates=dates,tickers=['QQQ']),marks=close.to_numpy()[:,None],
        returns=close.pct_change(fill_method=None).to_numpy()[:,None],
        adv=volume.rolling(20,min_periods=20).mean().to_numpy()[:,None],
        tradable=np.isfinite(close.to_numpy())[:,None],events=[])
    return result


def fixed_series(data,ids,weekly):
    exposure=np.ones(len(data['panel'].dates));rows=[]
    years=data['panel'].dates[data['entry'][ids]].year.to_numpy()
    for year in [2022,2023,2024,2025]:
        positions=np.flatnonzero(years==year);first=positions[0]
        training,_,_=fold(data,year)
        value=calibrate_fixed_exposure(data['returns'],data['origin'][training],weekly[first])
        start=int(data['origin'][ids[first]])
        stop=int(data['origin'][ids[positions[-1]+1]]) if positions[-1]+1<len(ids) else len(exposure)
        exposure[start:stop]=value['exposure']
        rows.append(dict(year=year,first_origin_id=int(ids[first]),first_signal_date=str(data['panel'].dates[start].date()),**value))
    return exposure,rows


def sr_null_p(left,right,draws):
    rf=1.03**(1/252)-1
    def sr(x):return (x.mean(-1)-rf)*np.sqrt(252)/x.std(-1,ddof=1)
    difference=float(sr(left)-sr(right));bootstrap=sr(left[draws])-sr(right[draws])
    return float((1+np.sum(bootstrap-difference>=difference))/(len(bootstrap)+1))


def main():
    out=DOC/'execution_batch8_risk';large=Path('results/RISHI_MECHANISM_RISK_20261027')
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    data=load_data();market=engine.market_inputs(data);qq=qqq_data(data);books=[];allids=None
    for candidate in ['E','B','D','REV5','Hlabel']:
        for group in GROUPS:
            with np.load(SOURCE/f'scores_{candidate}_{group}.npz') as saved:
                ids=saved['ids'].copy();score=saved['scores'].copy()
            if allids is None:allids=ids
            else:np.testing.assert_array_equal(ids,allids)
            for allocation in (['equal5','issuer5'] if candidate=='E' else ['equal5']):
                weights=engine.allocations(data,ids,score,allocation)
                policies=list(POLICIES) if candidate=='E' else ['full','vol20','fixed']
                if candidate=='E' and allocation=='equal5':policies.append('fixed')
                books.append((candidate,group,allocation,data,weights,policies))
    eligible=data['mask'][allids].astype(float);ew=eligible/eligible.sum(1,keepdims=True)
    books.extend([('EW','benchmark','eligible_weekly',data,ew,['full','vol20','fixed']),
                  ('QQQ','benchmark','one_etf',qq,np.ones((len(allids),1)),['full','vol20','fixed'])])
    dates=data['panel'].dates[int(data['entry'][allids[0]]):]
    source_metrics=pd.read_csv('docs/ndx_drawdown_20261026/metrics.csv')
    source_paths=np.load('runs/ndx_drawdown_20261026/batch1/paths.npz',allow_pickle=False)
    if source_paths['dates'].tolist()!=dates.strftime('%Y-%m-%d').tolist():raise ValueError('Source path calendar')
    old=pd.read_csv(SOURCE/'ACCOUNT_CURVES.csv');metrics=[];annual=[];calibration=[];parity=[];paths={};details={}
    for candidate,group,allocation,inputs,weights,policies in books:
        risk=engine.risk_inputs(inputs,engine.daily_allocation(inputs,allids,weights))
        fixed,cal=fixed_series(inputs,allids,weights) if 'fixed' in policies else (None,[])
        calibration.extend([dict(candidate=candidate,group=group,allocation=allocation,**r) for r in cal])
        for policy in policies:
            exposure=fixed if policy=='fixed' else engine.exposure_series(POLICIES[policy],market,risk)
            for mode in COSTS:
                key=f'{candidate}/{group}/{allocation}/{policy}/{mode}'
                with fee_view(mode) as multiplier:
                    m,path,detail=engine.simulate(inputs,allids,weights,exposure,'weekly',multiplier=multiplier,detailed=True)
                np.testing.assert_allclose(1e6*np.cumprod(1+path),detail.nav,rtol=1e-10,atol=1e-6)
                if detail.date.tolist()!=dates.strftime('%Y-%m-%d').tolist():raise ValueError('Output dates')
                if detail.cash.min() < -1e-5 or detail.equity_exposure.iloc[-1]!=0:raise ValueError('Cash/terminal feasibility')
                target=market['qqq'][int(data['entry'][allids[0]]):]/market['qqq'][int(data['entry'][allids[0]])-1:-1]-1
                qcenter=target-target.mean();rcenter=path-path.mean()
                previous=np.r_[1e6,detail.nav.to_numpy()[:-1]]
                extra=dict(mean_actual_exposure=float(detail.equity_exposure.mean()),
                    mean_target_exposure=float(detail.target_exposure.mean()),
                    qqq_beta=float(np.cov(path,target,ddof=1)[0,1]/target.var(ddof=1)),
                    downside_covariance_annual=float((rcenter[qcenter<0]*qcenter[qcenter<0]).mean()*252),
                    annual_arithmetic_mean=float(path.mean()*252),annual_fee_drag=float((detail.fee.to_numpy()/previous).mean()*252))
                metrics.append(dict(id=key,candidate=candidate,group=group,allocation=allocation,policy=policy,cost=mode,**m,**extra))
                annual.extend([dict(id=key,**r) for r in calendar_attribution(path,dates).to_dict('records')])
                paths[key]=path;details[key]=detail
                detail.assign(net=path).to_csv(large/(key.replace('/','__')+'.csv'),index=False)
                if mode=='modeled1' and candidate=='E' and policy in SOURCE_NAMES:
                    source_key=f'{allocation}/{SOURCE_NAMES[policy]}/{group}'
                    match=np.flatnonzero(source_paths['ids']==source_key)
                    if len(match)!=1:raise ValueError('Source account ID')
                    error=float(np.max(np.abs(path-source_paths['returns'][:,match[0]])))
                    sm=source_metrics[(source_metrics.id==source_key)&(source_metrics.cost_multiplier==1)].iloc[0]
                    for field in m:np.testing.assert_allclose(m[field],float(sm[field]),rtol=1e-10,atol=1e-6 if field in ['cost_dollars','terminal_nav'] else 1e-10)
                    if error>1e-10:raise AssertionError('Source path mismatch')
                    parity.append(dict(id=key,source_id=source_key,daily_error=error))
                if mode=='modeled1' and allocation=='equal5' and policy=='full':
                    expected=old[old.id==f'{candidate}/{group}/5']
                    if expected.date.tolist()!=detail.date.tolist():raise ValueError('Earlier control calendar')
                    np.testing.assert_allclose(path,expected.net,rtol=0,atol=1e-10)
        print(f'Completed risk book {candidate}/{group}/{allocation}',flush=True)
    pd.DataFrame(metrics).to_csv(out/'ACCOUNT_METRICS.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'CALENDAR_ATTRIBUTION.csv',index=False)
    pd.DataFrame(calibration).to_csv(out/'FIXED_CALIBRATION.csv',index=False)
    pd.DataFrame(parity).to_csv(out/'SOURCE_REPLAY.csv',index=False)
    np.savez_compressed(large/'PATHS.npz',ids=np.array(list(paths)),dates=dates.strftime('%Y-%m-%d').to_numpy(),returns=np.column_stack(list(paths.values())))
    covariance=[]
    for candidate in ['E','B','D','REV5','Hlabel','EW','QQQ']:
        group='benchmark' if candidate in ['EW','QQQ'] else 'ten_seed'
        allocation={'EW':'eligible_weekly','QQQ':'one_etf'}.get(candidate,'equal5')
        base=f'{candidate}/{group}/{allocation}';full=paths[base+'/full/zero'];managed=paths[base+'/vol20/zero']
        exposure=np.r_[0.,details[base+'/vol20/zero'].equity_exposure.to_numpy()[:-1]]
        scaled=exposure*full;cov=float(np.mean((exposure-exposure.mean())*(full-full.mean())))
        np.testing.assert_allclose(scaled.mean(),exposure.mean()*full.mean()+cov,atol=1e-12)
        covariance.append(dict(candidate=candidate,mean_exposure=float(exposure.mean()),full_mean_annual=float(full.mean()*252),
            mean_exposure_times_mean_return_annual=float(exposure.mean()*full.mean()*252),covariance_annual=cov*252,
            scaled_full_mean_annual=float(scaled.mean()*252),managed_mean_annual=float(managed.mean()*252),
            account_residual_annual=float((managed-scaled).mean()*252)))
    pd.DataFrame(covariance).to_csv(out/'FRICTIONLESS_COVARIANCE.csv',index=False)
    contrasts=[]
    for block in [21,5,63]:
        draws=stationary_indices(len(dates),10000,block,27102032)
        block_rows=[]
        for candidate in ['E','B','D','REV5','Hlabel','EW','QQQ']:
            group='benchmark' if candidate in ['EW','QQQ'] else 'ten_seed'
            allocation={'EW':'eligible_weekly','QQQ':'one_etf'}.get(candidate,'equal5');base=f'{candidate}/{group}/{allocation}'
            left=paths[base+'/vol20/modeled1']
            for comparator in ['full','fixed']:
                right=paths[base+'/'+comparator+'/modeled1']
                block_rows.append(dict(candidate=candidate,comparator=comparator,block=block,
                    **paired_contrast(left,right,draws,252),sr_p_one_sided=sr_null_p(left,right,draws)))
        adjusted=holm([r['sr_p_one_sided'] for r in block_rows])
        for row,p in zip(block_rows,adjusted):row['sr_p_holm']=float(p)
        contrasts.extend(block_rows);print(f'Completed risk contrasts block{block}',flush=True)
    pd.DataFrame(contrasts).to_csv(out/'PAIRED_RISK_CONTRASTS.csv',index=False)
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),DOC/'RISK_EXECUTION_PROTOCOL.md',Path('research/core/ndx_drawdown_engine.py')]}
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=len(metrics),source_accounts=len(parity),calendar_rows=len(annual),
        calibration_rows=len(calibration),contrasts=len(contrasts),daily_sessions=len(dates),source_hashes=hashes,
        status='completed_saved_export_risk_execution_checkpoint_gate_unchanged'),indent=2)+'\n')


if __name__=='__main__':main()
