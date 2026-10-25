"""Reproduce extension checkpoints, ranking diagnostics and every cash account."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from robust_models import make_model
from run_robust_research import scale_inputs, predict, model_data
from pit_research_engine import fold, historical_groups
from training_device import configure_device
from audit_historical_ndx import independent_metrics, replay, check_future_invariance
from audit_historical_ndx_ranking import average_ranks, diagnostics, FIELDS
from run_ndx_topk_pseudohuber import RUN, PARENT, load_forecasts, verify_lock, write, digest
from pseudohuber_think_mixer import KIND


def independent_ensemble(available, mask):
    if len(available)==1:
        return available[0]
    ranked=[]
    for prediction in available:
        values=np.zeros_like(prediction,dtype=float)
        for i,eligible in enumerate(mask):
            values[i,eligible]=average_ranks(prediction[i,eligible])/eligible.sum()
        ranked.append(values)
    return np.mean(ranked,axis=0)


def main():
    assert (RUN/'completion.json').exists()
    verify_lock();data,ids,forecasts=load_forecasts()
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    unchanged=check_future_invariance(data,Path('data/nasdaq_monthly_pit/assembled'))
    device=configure_device('cuda');reproduced=0;validation_exports=[]
    for year in plan['test_years']:
        train,val,test=fold(data,year);x,mean,std=scale_inputs(data,train)
        folder=RUN/str(year);groups=[sorted(g) for g in historical_groups(data,train)]
        with np.load(folder/'preprocessing.npz') as preprocessing:
            for field,value in [('train',train),('val',val),('test',test),('mean',mean),('std',std)]:
                np.testing.assert_array_equal(preprocessing[field],value)
        for seed in plan['training_seeds']:
            # Self-created local checkpoints; their hashes are checked by load_forecasts.
            checkpoint=torch.load(folder/f'{KIND}_{seed}.pt',map_location='cpu',weights_only=False)
            assert checkpoint['kind']==KIND and checkpoint['seed']==seed
            assert checkpoint['groups']==groups and checkpoint['delta_pp']==plan['delta_pp']==1.
            model=make_model('think_mix',len(data['panel'].tickers),groups,causal_membership=True).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            vp=predict(model,x,data['mask'],val,plan['batch']);tp=predict(model,x,data['mask'],test,plan['batch'])
            with np.load(folder/f'{KIND}_{seed}_predictions.npz') as saved:
                np.testing.assert_allclose(tp,saved['test'],rtol=0,atol=1e-6)
                cached_validation=saved['validation'].copy()
            history=json.loads((folder/f'{KIND}_{seed}_history.json').read_text())
            best=np.inf;chosen=0
            for row in history:
                if row['validation_mse_pp2']<best-1e-7:
                    best=row['validation_mse_pp2'];chosen=row['epoch']
            assert chosen==checkpoint['epoch']
            vmse=float(((vp-data['y'][val]*100)**2)[data['labelmask'][val]].mean())
            np.testing.assert_allclose(vmse,best,atol=1e-6,rtol=1e-6)
            # The fit routine returned its last-epoch validation array while
            # correctly saving/predicting the selected checkpoint. Preserve that
            # locked cache, verify what it represents, and export selected-epoch
            # validation separately. This array was never used for portfolio
            # allocation, early stopping, or the saved test forecasts.
            cached_mse=float(((cached_validation-data['y'][val]*100)**2)[data['labelmask'][val]].mean())
            np.testing.assert_allclose(cached_mse,history[-1]['validation_mse_pp2'],atol=1e-6,rtol=1e-6)
            corrected_folder=RUN/'selected_validation'/str(year);corrected_folder.mkdir(parents=True,exist_ok=True)
            corrected=corrected_folder/f'{KIND}_{seed}.npz'
            np.savez_compressed(corrected,validation=vp,validation_ids=val)
            validation_exports.append(dict(year=year,seed=seed,selected_epoch=chosen,
                original_export_epoch=history[-1]['epoch'],original_mse=cached_mse,selected_mse=vmse,
                corrected_file=str(corrected),sha256=digest(corrected),test_forecasts_unchanged=True))
            assert np.isfinite(tp[data['mask'][test]]).all()
            reproduced+=1
        print(f'{year}: all five pseudo-Huber checkpoints and validation epochs reproduced',flush=True)
    write(RUN/'VALIDATION_EXPORT_CORRECTION.json',dict(status='corrected_without_retraining',
        issue='Original frozen prediction files contain last-epoch validation arrays, but best-epoch test forecasts',
        correction='Selected-checkpoint validation arrays exported separately; original fit files and hashes preserved',
        selection='Checkpoint epochs verified against prior-validation MSE histories; no test return used',
        training_and_test_predictions_changed=False,exports=validation_exports))
    rank_rows=[];ensembles={};ranking=pd.read_csv(RUN/'ranking.csv').set_index('model')
    for kind,available in forecasts.items():
        d=model_data(data,kind);ensembles[kind]=independent_ensemble(available,d['mask'][ids])
        for j,i in enumerate(ids):
            observed=d['labelmask'][i];eligible=d['mask'][i]
            row=dict(model=kind,year=int(data['panel'].dates[data['entry'][i]].year),
                entry_date=str(data['panel'].dates[data['entry'][i]].date()),
                observed=int(observed.sum()),unobserved=int((eligible & ~observed).sum()),
                **diagnostics(ensembles[kind][j,observed],d['y'][i,observed]))
            rank_rows.append(row)
        sub=pd.DataFrame([row for row in rank_rows if row['model']==kind])
        for field in FIELDS:
            np.testing.assert_allclose(sub[field].mean(),ranking.loc[kind,field],atol=1e-10)
        for field in ['IC','RankIC']:
            np.testing.assert_allclose(sub[field].mean()/sub[field].std(),ranking.loc[kind,field+'IR'],atol=1e-10)
            assert sub[field].count()==ranking.loc[kind,field+'_periods']==200
        assert sub.unobserved.sum()==ranking.loc[kind,'unobserved_labels']
    ranked=pd.DataFrame(rank_rows);ranked.to_csv(RUN/'ranking_periods.csv',index=False)
    ranked.groupby(['model','year'])[FIELDS+['Random NDCG@10']].mean().to_csv(RUN/'ranking_by_year.csv')
    curves=pd.read_csv(RUN/'curves.csv',dtype={'seed':str})
    metrics=pd.read_csv(RUN/'metrics.csv',dtype={'seed':str})
    old_curves=pd.read_csv(PARENT/'curves.csv',dtype={'seed':str})
    old_metrics=pd.read_csv(PARENT/'metrics.csv',dtype={'seed':str})
    matched=0;replayed=0;largest=0.;neural=list(json.loads((PARENT/'PROTOCOL.json').read_text())['models'])+[KIND]
    keys=['model','seed','k','cost_multiplier']
    old_groups=dict(tuple(old_curves.groupby(keys,sort=False)))
    for key,piece in curves.groupby(keys,sort=False):
        kind,seed,k,mult=key
        assert piece.date.is_unique and piece.date.is_monotonic_increasing and len(piece)==1003
        assert piece.date.iloc[0]=='2022-01-03' and piece.date.iloc[-1]=='2025-12-31'
        selected=metrics
        for field,value in zip(keys,key):
            selected=selected[selected[field]==value]
        assert len(selected)==1
        independent=independent_metrics(piece.net)
        for field,value in independent.items():
            np.testing.assert_allclose(value,selected[field].iloc[0],atol=1e-10)
        if key in old_groups:
            prior=old_groups[key]
            assert list(piece.date)==list(prior.date)
            np.testing.assert_allclose(piece.net,prior.net,rtol=0,atol=2e-15)
            matched+=1
        if kind in ['SPY','QQQ']:
            root=Path(data['manifest']['root'])
            prices=pd.read_csv(root/'etf_adj_close.csv',index_col=0,parse_dates=True)[kind]
            dollar=pd.read_csv(root/'etf_dollar_volume.csv',index_col=0,parse_dates=True)[kind]
            first=data['panel'].dates.get_loc(pd.Timestamp('2022-01-03'));last=len(prices)-1
            rate=mult*(.0007+.0003*np.sqrt(1e6/dollar.iloc[first-20:first].mean()/.01))
            units=1e6/(1+rate)/prices.iloc[first]
            nav=(units*prices.iloc[first:]).to_numpy().copy()
            rate=mult*(.0007+.0003*np.sqrt(nav[-1]/dollar.iloc[last-20:last].mean()/.01))
            nav[-1]*=1-rate
        else:
            d=model_data(data,kind)
            score=(np.zeros_like(data['y'][ids]) if kind=='monthly_basket' else
                ensembles[kind] if seed=='ensemble' else forecasts[kind][plan['training_seeds'].index(int(seed))])
            nav=replay(d,ids,score,k,mult,basket=kind=='monthly_basket')
        published=1e6*np.cumprod(1+piece.net.to_numpy())
        largest=max(largest,float(np.max(np.abs(nav-published))))
        np.testing.assert_allclose(nav,published,rtol=1e-10,atol=1e-5)
        replayed+=1
        if replayed%40==0:
            print(f'{replayed}/204 independently replayed accounts matched',flush=True)
    accounts=pd.read_csv(RUN/'new_account_ledger.csv')
    np.testing.assert_allclose(accounts.nav,accounts.cash+accounts.holdings,atol=1e-7,rtol=1e-12)
    assert accounts.cash.min()>=-1e-5
    assert reproduced==20 and replayed==204 and matched==120 and len(ranking)==10
    result=dict(status='passed',new_checkpoints_reproduced=reproduced,independent_share_cash_paths=replayed,
        parent_paths_unchanged=matched,new_paths=84,ranking_models_independently_checked=10,
        largest_nav_difference_dollars=largest,annual_resets=0,
        future_input_perturbation_past_forecasts_unchanged=unchanged,
        validation_export_correction='VALIDATION_EXPORT_CORRECTION.json; original caches preserved; selected-checkpoint validation and unchanged test forecasts verified',
        limitation='Exploratory follow-up on already-inspected years; original public-data and corporate-action limitations remain')
    write(RUN/'AUDIT.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
