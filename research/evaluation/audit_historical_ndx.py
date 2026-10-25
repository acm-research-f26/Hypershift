"""Independent share/cash replay and checkpoint reproduction for the new study."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from robust_models import make_model
from pit_research_engine import load_data,fold,historical_groups
from run_robust_research import scale_inputs,predict,ensemble_predictions,model_data
from training_device import configure_device
from research_engine import performance

RUN=Path('runs/nasdaq100_historical_monthly_20261011')


def independent_metrics(returns):
    r=np.asarray(returns,dtype=float)
    account=np.r_[1.,np.cumprod(1+r)]
    deviation=np.std(r,ddof=1)*np.sqrt(252.)
    hurdle=np.expm1(np.log1p(.03)/252.)
    return dict(sharpe=(r.mean()-hurdle)*252./deviation,
                annualized_return=account[-1]**(252./len(r))-1.,
                cumulative_return=account[-1]-1.,
                maximum_drawdown=np.min(account/np.maximum.accumulate(account)-1.))


def check_future_invariance(data,root):
    """Perturb raw future inputs in memory, never the frozen source files."""
    cutoff=pd.Timestamp('2021-12-30');read_csv=pd.read_csv
    def altered(path,*args,**kwargs):
        frame=read_csv(path,*args,**kwargs)
        if isinstance(frame.index,pd.DatetimeIndex):
            later=frame.index>cutoff
            if Path(path).stem in ['economic_close','dollar_volume']:
                frame.loc[later]*=np.linspace(2.,20.,frame.shape[1])
        return frame
    with patch('pit_research_engine.pd.read_csv',side_effect=altered):
        changed=load_data(root)
    past=np.flatnonzero(data['panel'].dates[data['entry']]<=cutoff)
    for field in ['x','mask','momentum','reversal']:
        np.testing.assert_array_equal(data[field][past],changed[field][past])
    matured=np.flatnonzero(data['panel'].dates[data['exit']]<=cutoff)
    np.testing.assert_array_equal(data['y'][matured],changed['y'][matured])
    tr,_,_=fold(data,2022)
    _,mean,std=scale_inputs(data,tr);_,other_mean,other_std=scale_inputs(changed,tr)
    np.testing.assert_array_equal(mean,other_mean);np.testing.assert_array_equal(std,other_std)
    assert historical_groups(data,tr)==historical_groups(changed,tr)
    return len(past)


def replay(data,ids,scores,k,mult,basket=False):
    """Independent daily unit ledger: no production allocation/backtest calls."""
    names=data['panel'].tickers;dates=data['panel'].dates;n=len(names)
    shares=np.zeros(n);bank=1e6;previous=1e6;values=[];new_order=None
    quotes=data['marks'];first=int(data['entry'][ids[0]]);last=len(dates)-1
    forecasts={int(data['origin'][i]):(i,np.asarray(scores[j])) for j,i in enumerate(ids)}
    actions={str(e['date']):e for e in data['events']};last_month=None
    def fee(trades,adv):
        rate=.0007+.0003*np.sqrt(np.abs(trades)/np.maximum(np.nan_to_num(adv,nan=1.),1.)/.01)
        return float((np.abs(trades)*rate).sum()*mult)
    def freeze(day,i,s):
        nonlocal last_month
        weights=np.zeros(n)
        if basket:
            month=str(dates[data['entry'][i]].to_period('M'))
            if month==last_month:return None
            last_month=month
            selected=data['membership'][data['entry'][i]] & data['tradable'][day]
            weights[selected]=1/max(int(selected.sum()),1)
        else:
            selected=data['mask'][i];available=np.flatnonzero(selected)
            if not len(available):return None
            cutoff=np.sort(s[available])[max(0,len(available)-k)]
            strict=selected&(s>cutoff);tie=selected&(s==cutoff)
            weights[strict]=1.;weights[tie]=(min(k,len(available))-strict.sum())/max(tie.sum(),1)
            weights/=weights.sum()
        target=np.zeros(n);cap=np.zeros(n)
        for j in range(n):
            if np.isfinite(quotes[day,j]) and quotes[day,j]>0:
                target[j]=weights[j]*previous/quotes[day,j]
                cap[j]=.01*np.nan_to_num(data['adv'][day,j])/quotes[day,j]
        return np.maximum(np.minimum(target-shares,cap),-cap),data['adv'][day].copy()
    i,s=forecasts[first-1];new_order=freeze(first-1,i,s)
    for day in range(first,last+1):
        action=actions.get(str(dates[day].date()))
        if action:
            node=names.index(action['security_id']);value=shares[node]*quotes[day,node]
            stock=value*action['stock_fraction_at_completion']
            bank+=value-fee(np.array([stock]),np.array([data['adv'][day-1,node]]));shares[node]=0.
        if day==last:new_order=None
        if new_order is not None:
            quantities,adv=new_order
            quantities=np.where(data['tradable'][day],quantities,0.)
            quantities=np.maximum(quantities,-shares)
            trades=np.nan_to_num(quantities*quotes[day])
            sells=np.minimum(trades,0.);buys=np.maximum(trades,0.)
            sellfee=fee(sells,adv);buyfee=fee(buys,adv)
            available=bank-sells.sum()-sellfee
            if buys.sum()+buyfee>available and buys.sum()+buyfee>0:
                quantities[trades>0]*=max(0.,available/(buys.sum()+buyfee))
                trades=np.nan_to_num(quantities*quotes[day])
            bank-=trades.sum()+fee(trades,adv);shares+=quantities;new_order=None
        if day==last:
            trades=np.nan_to_num(shares*quotes[day]);bank+=trades.sum()-fee(trades,data['adv'][day-1]);shares[:]=0.
        worth=bank+float(np.nan_to_num(shares*quotes[day]).sum())
        values.append(worth);previous=worth
        if day in forecasts:
            i,s=forecasts[day];new_order=freeze(day,i,s)
    return np.asarray(values)


def main():
    assert (RUN/'FITS_COMPLETE_LOCK.json').exists(),'No audit before every declared fit is locked'
    assert (RUN/'completion.json').exists(),'Evaluation must be complete'
    root=Path('data/nasdaq_monthly_pit/assembled');data=load_data(root)
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    locks=json.loads((RUN/'INPUT_CODE_LOCK.json').read_text())
    locks.update(json.loads((RUN/'DEPENDENCY_CODE_LOCK.json').read_text()))
    for path,expected in locks.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    fitlock=json.loads((RUN/'FITS_COMPLETE_LOCK.json').read_text())
    for path,expected in fitlock['checkpoint_hashes'].items():
        assert hashlib.sha256((RUN/path).read_bytes()).hexdigest()==expected,path
    unchanged_forecasts=check_future_invariance(data,root)
    device=configure_device('cuda');reproduced=0;predictions={};ids=[]
    for year in plan['test_years']:
        train,val,test=fold(data,year);ids.extend(test);x,mean,std=scale_inputs(data,train)
        saved_pre=np.load(RUN/str(year)/'preprocessing.npz')
        for field,current in [('train',train),('val',val),('test',test),('mean',mean),('std',std)]:
            np.testing.assert_array_equal(saved_pre[field],current)
        groups=[sorted(g) for g in historical_groups(data,train)]
        for kind in plan['models']:
            paths=[]
            for seed in plan['training_seeds']:
                path=RUN/str(year)/f'{kind}_{seed}.pt'
                # These files were created by our frozen local fit routine and
                # verified against the fit lock above. Their group lists contain
                # NumPy scalar metadata that the restricted tensor loader rejects.
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                assert checkpoint['groups']==groups
                assert checkpoint['seed']==seed and checkpoint['kind']==kind
                record=json.loads((RUN/str(year)/f'{kind}_{seed}_record.json').read_text())
                history=json.loads((RUN/str(year)/f'{kind}_{seed}_history.json').read_text())
                best=np.inf;chosen=0
                for item in history:
                    if item['validation_mse_pp2']<best-1e-7:
                        best=item['validation_mse_pp2'];chosen=item['epoch']
                assert chosen==checkpoint['epoch']==record['chosen_epoch']
                assert len(history)==record['epochs_run']
                model=make_model(kind,len(data['panel'].tickers),checkpoint['groups'],causal_membership=True).to(device)
                model.load_state_dict(checkpoint['state_dict'])
                saved=np.load(RUN/str(year)/f'{kind}_{seed}_predictions.npz')
                vp=predict(model,x,data['mask'],val,8);tp=predict(model,x,data['mask'],test,8)
                np.testing.assert_allclose(vp,saved['validation'],rtol=0,atol=1e-6)
                np.testing.assert_allclose(tp,saved['test'],rtol=0,atol=1e-6)
                vmse=float(((vp-data['y'][val]*100)**2)[data['labelmask'][val]].mean())
                np.testing.assert_allclose(vmse,best,rtol=1e-6,atol=1e-6)
                paths.append(tp);reproduced+=1
            predictions[(year,kind)]=paths
        design=x.transpose(0,2,1,3).reshape(len(x),len(data['panel'].tickers),-1)
        design=np.concatenate([design,np.ones((*design.shape[:2],1))],axis=-1)
        coef=np.load(RUN/str(year)/'ridge_coef.npy')
        a=design[train][data['labelmask'][train]]
        target=(data['y'][train]*100)[data['labelmask'][train]]
        penalty=np.eye(a.shape[-1])*10.;penalty[-1,-1]=0.
        recomputed_coef=np.linalg.solve(a.T@a+penalty,a.T@target)
        np.testing.assert_allclose(coef,recomputed_coef,rtol=1e-10,atol=1e-10)
        predictions[(year,'ridge')]=[design[test]@coef]
        predictions[(year,'momentum126')]=[data['momentum'][test]]
        predictions[(year,'reversal5')]=[data['reversal'][test]]
        print(f'{year}: checkpoints, validation epochs, scaler, graph and ridge reproduced',flush=True)
    ids=np.asarray(ids);curves=pd.read_csv(RUN/'curves.csv',dtype={'seed':str})
    replayed=0;metric_checks=0;largest_error=0.
    metrics=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str})
    for key,piece in curves.groupby(['model','seed','k','cost_multiplier'],sort=False):
        kind,seed,k,mult=key
        assert piece.date.is_unique and piece.date.is_monotonic_increasing
        source=metrics[(metrics.model==kind)&(metrics.seed==seed)&(metrics.k==k)&(metrics.cost_multiplier==mult)]
        assert len(source)==1
        p=independent_metrics(piece.net)
        for field in ['sharpe','annualized_return','cumulative_return','maximum_drawdown']:
            np.testing.assert_allclose(p[field],source[field].iloc[0],atol=1e-10)
        metric_checks+=1
        if kind in ['SPY','QQQ']:
            # Independent passive account using one initial fractional purchase.
            prices=pd.read_csv(root/'etf_adj_close.csv',index_col=0,parse_dates=True)[kind]
            dollar=pd.read_csv(root/'etf_dollar_volume.csv',index_col=0,parse_dates=True)[kind]
            first=data['panel'].dates.get_loc(pd.Timestamp(piece.date.iloc[0]));last=len(prices)-1
            start_rate=mult*(.0007+.0003*np.sqrt(1e6/dollar.iloc[first-20:first].mean()/.01))
            units=1e6/(1+start_rate)/prices.iloc[first]
            independent=(units*prices.iloc[first:]).to_numpy().copy()
            end_rate=mult*(.0007+.0003*np.sqrt(independent[-1]/dollar.iloc[last-20:last].mean()/.01))
            independent[-1]*=1-end_rate
        else:
            d=model_data(data,kind)
            if kind=='monthly_basket':score=np.zeros_like(data['y'][ids])
            else:
                available=[np.concatenate([predictions[(year,kind)][j] for year in plan['test_years']])
                           for j in range(len(predictions[(2022,kind)]))]
                score=ensemble_predictions(available,d['mask'][ids]) if seed=='ensemble' and len(available)>1 else available[0] if seed=='ensemble' else available[plan['training_seeds'].index(int(seed))]
            independent=replay(d,ids,score,k,mult,basket=kind=='monthly_basket')
        published=1e6*np.cumprod(1+piece.net.to_numpy())
        error=float(np.max(np.abs(independent-published)))
        largest_error=max(largest_error,error)
        np.testing.assert_allclose(independent,published,rtol=1e-10,atol=1e-5)
        replayed+=1
        if replayed%20==0:print(f'{replayed}/120 independent accounts matched',flush=True)
    accounts=pd.read_csv(RUN/'account_ledger.csv')
    np.testing.assert_allclose(accounts.nav,accounts.cash+accounts.holdings,rtol=1e-12,atol=1e-7)
    assert accounts.cash.min()>=-1e-5
    result=dict(status='passed',neural_checkpoints_reproduced=reproduced,independent_share_cash_paths=replayed,
                recomputed_metrics=metric_checks,largest_nav_difference_dollars=largest_error,
                annual_resets=0,held_missing_prices_without_actions=0,
                future_input_perturbation_past_forecasts_unchanged=unchanged_forecasts,
                note='Arithmetic/timing reproduction does not certify vendor or reconstructed-index data; limitations remain disclosed.')
    (RUN/'AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
