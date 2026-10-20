"""Inspect matched MSE/pseudo-Huber fits and selections without fitting anything."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from robust_models import make_model
from pit_research_engine import fold, continuous_backtest
from research_engine import target_weights, performance
from run_robust_research import ensemble_predictions
from run_ndx_topk_pseudohuber import load_forecasts, PARENT, RUN as PH_RUN
from analyze_ndx_rank_depth import trace, digest

RUN=Path('runs/nasdaq_loss_similarity_20261011')
KINDS=['think_mix','think_mix_pseudohuber']
SEEDS=[7,19,42,61,97]


def ranks(score,mask):
    return pd.Series(score[mask]).rank(method='average').to_numpy()


def corr(a,b): return float(np.corrcoef(a,b)[0,1]) if min(np.std(a),np.std(b))>1e-12 else np.nan


def weights(score,mask):
    first=target_weights(score,mask,5,'equal'); ten=target_weights(score,mask,10,'equal')
    return {'top5':first,'top10':ten,'ranks6_10':2*ten-first}


def main():
    data,ids,forecasts=load_forecasts()
    RUN.mkdir(exist_ok=True)
    lock=json.loads((PH_RUN/'INPUT_CODE_LOCK.json').read_text())
    lock.update(json.loads((PH_RUN/'SOURCE_ARTIFACT_LOCK.json').read_text()))
    for name,expected in json.loads((PH_RUN/'FITS_COMPLETE_LOCK.json').read_text())['hashes'].items():
        lock[str(PH_RUN/name)]=expected
    for name in ['AUDIT.json','VALIDATION_EXPORT_CORRECTION.json']:
        lock[str(PH_RUN/name)]=digest(PH_RUN/name)
    for year in [2022,2023,2024,2025]:
        for seed in SEEDS:
            path=PH_RUN/f'selected_validation/{year}/think_mix_pseudohuber_{seed}.npz'
            lock[str(path)]=digest(path)
    for name in [Path(__file__).name,'analyze_ndx_rank_depth.py']:
        lock[name]=digest(name)
    for name,expected in lock.items(): assert digest(name)==expected,name
    if (RUN/'INPUT_CODE_LOCK.json').exists(): assert json.loads((RUN/'INPUT_CODE_LOCK.json').read_text())==lock
    else: (RUN/'INPUT_CODE_LOCK.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    parent_plan=json.loads((PARENT/'PROTOCOL.json').read_text())
    ph_plan=json.loads((PH_RUN/'PROTOCOL.json').read_text())
    for field in ['training_seeds','test_years','epochs','patience','lr']:
        assert parent_plan[field]==ph_plan[field],field
    assert ph_plan['batch']==8 and ph_plan['delta_pp']==1.
    matched=[]; residuals=[]
    for year in [2022,2023,2024,2025]:
        train,val,test=fold(data,year)
        with np.load(PARENT/str(year)/'preprocessing.npz') as a,np.load(PH_RUN/str(year)/'preprocessing.npz') as b:
            for field in ['train','val','test','mean','std']: np.testing.assert_array_equal(a[field],b[field])
        observed=data['labelmask'][val]; labels=data['y'][val]*100
        for seed in SEEDS:
            previous=torch.load(PARENT/str(year)/f'think_mix_{seed}.pt',map_location='cpu',weights_only=False)
            robust=torch.load(PH_RUN/str(year)/f'think_mix_pseudohuber_{seed}.pt',map_location='cpu',weights_only=False)
            assert previous['groups']==robust['groups']
            assert previous['training_device']==robust['training_device']
            assert set(previous['state_dict'])==set(robust['state_dict'])
            assert all(previous['state_dict'][key].shape==robust['state_dict'][key].shape for key in previous['state_dict'])
            # Recreate initialization only; no backward pass or optimizer step.
            torch.manual_seed(seed); initial_mse=make_model('think_mix',len(data['panel'].tickers),previous['groups'],causal_membership=True)
            torch.manual_seed(seed); initial_ph=make_model('think_mix',len(data['panel'].tickers),robust['groups'],causal_membership=True)
            assert all(torch.equal(a,b) for a,b in zip(initial_mse.parameters(),initial_ph.parameters()))
            with np.load(PARENT/str(year)/f'think_mix_{seed}_predictions.npz') as cached:
                mse_val=cached['validation']
            with np.load(PH_RUN/f'selected_validation/{year}/think_mix_pseudohuber_{seed}.npz') as selected:
                np.testing.assert_array_equal(selected['validation_ids'],val); ph_val=selected['validation']
            record=dict(year=year,seed=seed,MSE_epoch=previous['epoch'],PH_epoch=robust['epoch'],
                parameter_count=sum(v.numel() for v in previous['state_dict'].values()),
                same_groups=True,same_device=True,same_shapes=True,recreated_initial_parameters_identical=True,
                final_parameters_identical=all(torch.equal(previous['state_dict'][k],robust['state_dict'][k]) for k in previous['state_dict']))
            for kind,pred in zip(KINDS,[mse_val,ph_val]):
                error=np.asarray(pred-labels)[observed]
                record[('MSE' if kind=='think_mix' else 'PH')+'_validation_MSE']=float(np.mean(error**2))
                residuals.extend([dict(year=year,seed=seed,model=kind,
                    median_abs_error_pp=float(np.median(np.abs(error))),
                    fraction_abs_error_above_1pp=float(np.mean(np.abs(error)>1)),
                    fraction_abs_error_above_5pp=float(np.mean(np.abs(error)>5)),
                    mean_PH_to_MSE_gradient_ratio=float(np.mean(1/np.sqrt(1+error**2))))])
            matched.append(record)
    agreement=[]; portfolios=[]; curve_rows=[]
    scores={kind:{str(seed):pred for seed,pred in zip(SEEDS,forecasts[kind])} for kind in KINDS}
    for kind in KINDS: scores[kind]['ensemble']=ensemble_predictions(forecasts[kind],data['mask'][ids])
    for seed in [*map(str,SEEDS),'ensemble']:
        for j,i in enumerate(ids):
            mask=data['mask'][i]; a=scores[KINDS[0]][seed][j]; b=scores[KINDS[1]][seed][j]
            wa=weights(a,mask); wb=weights(b,mask)
            row=dict(seed=seed,year=int(data['panel'].dates[data['entry'][i]].year),
                entry_date=str(data['panel'].dates[data['entry'][i]].date()),
                score_correlation=corr(a[mask],b[mask]),rank_correlation=corr(ranks(a,mask),ranks(b,mask)),
                mse_score_std=float(np.std(a[mask])),PH_score_std=float(np.std(b[mask])))
            for policy in wa: row[policy+'_common_weight']=float(np.minimum(wa[policy],wb[policy]).sum())
            agreement.append(row)
        for kind in KINDS:
            band=np.stack([weights(s,m)['ranks6_10'] for s,m in zip(scores[kind][seed],data['mask'][ids])])
            daily,pnl,fees,exposure=trace(data,ids,band)
            with patch('pit_research_engine.target_weights',side_effect=iter(band)):
                _,reference,_=continuous_backtest(data,ids,scores[kind][seed],k=5)
            np.testing.assert_allclose(daily.net,reference.net,atol=1e-12,rtol=1e-10)
            result=performance(daily.net)
            portfolios.append(dict(model=kind,seed=seed,policy='ranks6_10',**result))
            curve_rows.append(daily.assign(model=kind,seed=seed))
    matched=pd.DataFrame(matched); agreement=pd.DataFrame(agreement); portfolios=pd.DataFrame(portfolios)
    matched.to_csv(RUN/'matched_fits.csv',index=False)
    agreement.to_csv(RUN/'agreement_by_period.csv',index=False)
    agreement.groupby('seed').mean(numeric_only=True).drop(columns=['year']).reset_index().to_csv(RUN/'agreement_summary.csv',index=False)
    pd.DataFrame(residuals).to_csv(RUN/'validation_residuals.csv',index=False)
    portfolios.to_csv(RUN/'marginal_seed_metrics.csv',index=False)
    pd.concat(curve_rows,ignore_index=True).to_csv(RUN/'marginal_seed_curves.csv',index=False)
    derivative=[]
    for error in [.1,1.,5.,10.,20.]:
        derivative.append(dict(abs_error_pp=error,MSE_gradient=2*error,
            PH_gradient=2*error/np.sqrt(1+error**2),gradient_ratio=1/np.sqrt(1+error**2)))
    pd.DataFrame(derivative).to_csv(RUN/'gradient_examples.csv',index=False)
    year_summary=agreement[agreement.seed=='ensemble'].groupby('year').mean(numeric_only=True).reset_index()
    year_summary.to_csv(RUN/'ensemble_agreement_by_year.csv',index=False)
    audit=dict(status='passed',new_model_fits=0,matched_checkpoint_pairs=20,
        same_architecture_groups_preprocessing_seeds_optimizer_and_runtime=True,
        recreated_initial_parameters_identical=True,
        final_parameters_identical_pairs=int(matched.final_parameters_identical.sum()),
        different_selected_epochs=int((matched.MSE_epoch!=matched.PH_epoch).sum()),
        independent_marginal_seed_accounts=12,
        validation_exports='Selected-checkpoint sidecars for pseudo-Huber; stale final-epoch exports not used',
        limitations=['No retraining; loss comparison includes the intended resulting changes in fitted weights and validation-selected epochs',
            'Initialization recreation verifies the code/seed rule, not a stored initial-state artifact',
            'Gradient examples concern unnormalized per-observation loss derivatives; AdamW, batching and global norm clipping affect actual updates',
            'Validation-error summaries describe selected checkpoints, not every training-step residual',
            'Observed performance gaps do not establish persistent loss superiority'])
    (RUN/'AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
    print(agreement.groupby('seed')[['rank_correlation','top5_common_weight','top10_common_weight','ranks6_10_common_weight']].mean().to_string())
    print(portfolios.pivot(index='seed',columns='model',values='annualized_return').to_string())


if __name__=='__main__': main()
