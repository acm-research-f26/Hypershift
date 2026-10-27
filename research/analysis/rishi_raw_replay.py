"""Original selected-checkpoint replay and fixed stock-signal controls; no fitting neural models."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from ndx_reference_training import prepare,predictions
from ndx_reference_models import make_reference
from ndx_extended_hours import ExtendedModel,predict
from pit_research_engine import load_data,fold,continuous_backtest
from run_robust_research import ensemble_predictions
from ndx_ranking_objectives import rank_ic
from rishi_artifact_verify import verify_contract,write_receipt
from rishi_mechanism_batch1 import stationary_indices,paired_contrast,holm
from research_engine import target_weights

ROOT=Path.cwd();DOC=ROOT/'docs/rishi_mechanism_20261027';OUT=DOC/'execution_batch2'
LARGE=ROOT/'results/RISHI_MECHANISM_REPLAY_20261027'
SEEDS=[7,19,42,61,97,1828801906,1868046758,239154507,287727767,981756941]
GROUPS={'original_five':SEEDS[:5],'new_five':SEEDS[5:],'ten_seed':SEEDS}
REFERENCE=ROOT/'runs/ndx_reference_networks_20261024'
PARENT=ROOT/'runs/ndx_extended_hours_20261024/after_hours_premarket'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def persistent_mean(values,mask,minimum=30):
    values=np.asarray(values,float);good=np.asarray(mask,bool)&np.isfinite(values)
    count=good.sum(0);total=np.where(good,values,0).sum(0)
    means=np.divide(total,count,out=np.full(values.shape[1],np.nan),where=count>0)
    usable=count>=minimum
    if not usable.any():raise ValueError('No qualifying training IDs')
    return np.where(usable,means,np.median(means[usable])),count


def neutralize_missing(values,mask):
    result=np.asarray(values,float).copy();missing=0
    for row,eligible in zip(result,mask):
        good=eligible&np.isfinite(row);bad=eligible&~np.isfinite(row)
        missing+=int(bad.sum());row[bad]=np.median(row[good]) if good.any() else 0.
        row[~eligible]=0.
    return result,missing


def fixed_ridge(x,y,mask):
    x=np.asarray(x,float);y=np.asarray(y,float)
    good=np.asarray(mask,bool)&np.isfinite(x).all(-1)&np.isfinite(y)
    design=np.column_stack([np.ones(int(good.sum())),x[good]])
    penalty=np.eye(design.shape[1]);penalty[0,0]=0.
    return np.linalg.solve(design.T@design+penalty,design.T@y[good])


def forecast_diagnostics(actual,expected,mask):
    correlations=[];changes={5:0,10:0}
    for a,b,e in zip(actual,expected,mask):
        ar=pd.Series(a[e]).rank().to_numpy();br=pd.Series(b[e]).rank().to_numpy()
        if ar.std()>0 and br.std()>0:correlations.append(float(np.corrcoef(ar,br)[0,1]))
        elif np.array_equal(ar,br):correlations.append(1.)
        for k in changes:
            if not np.allclose(target_weights(a,e,k,'equal'),target_weights(b,e,k,'equal'),atol=1e-14,rtol=0):changes[k]+=1
    return dict(mean_rank_correlation=float(np.mean(correlations)) if correlations else None,
                top5_changed_origins=changes[5],top10_changed_origins=changes[10])


def check_replayed_account(metrics,curve,ledger,source_metrics,source_curve,source_ledger,aum=1_000_000.):
    assert list(curve.date)==list(source_curve.date)==list(ledger.date)==list(source_ledger.date)
    daily_error=float(np.max(abs(curve.net.to_numpy()-source_curve.net.to_numpy())))
    nav_error=float(np.max(abs(ledger.nav.to_numpy()-source_ledger.nav.to_numpy()))/aum)
    assert daily_error<=1e-10 and nav_error<=1e-8
    columns=[c for c in ['nav','cash','holdings','fee','traded_dollars','settled_dollars'] if c in ledger and c in source_ledger]
    np.testing.assert_allclose(ledger[columns],source_ledger[columns],rtol=1e-10,atol=1e-6)
    for key,value in metrics.items():np.testing.assert_allclose(value,source_metrics[key],rtol=1e-10,atol=1e-10)
    return dict(max_daily_error=daily_error,max_relative_nav_error=nav_error,ledger_columns=len(columns),metric_fields=len(metrics))


def save_json(path,value):write_receipt(path,value)


def load_selected_checkpoint(path):
    allowed_names={'numpy.dtype','numpy._core.multiarray.scalar'}
    unexpected=set(torch.serialization.get_unsafe_globals_in_checkpoint(path))-allowed_names
    if unexpected:raise ValueError('Unexpected checkpoint metadata globals: '+str(sorted(unexpected)))
    with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,np.dtypes.Int64DType]):
        return torch.load(path,map_location='cpu',weights_only=True)


def reforecast(p,family,year,seed,device='cpu'):
    folder=(PARENT if family=='B' else REFERENCE/'dynamic_restricted')/str(year)
    cp=load_selected_checkpoint(folder/f'{seed}.pt')
    if family=='B':
        model=ExtendedModel(len(p['data']['panel'].tickers),[set(g) for g in cp['groups']],'after_hours_premarket')
    else:model=make_reference('dynamic_restricted',len(p['data']['panel'].tickers),p['snapshot'])
    model.load_state_dict(cp['state_dict']);model.to(device);model.eval();output={}
    if family=='B':
        x=p['features'][...,:5].transpose(0,2,1,3).copy()
        extra=p['features'][...,5:].reshape(len(p['features']),p['features'].shape[1],320).copy()
    with torch.no_grad():
        for split in ['train','validation','test']:
            ids=p[split]
            output[split]=predict(model,x,extra,p['data']['mask'],ids,device,8) if family=='B' else predictions(model,p,ids,'dynamic_restricted',device)
    with np.load(folder/f'{seed}_predictions.npz',allow_pickle=False) as saved:
        errors={split:float(np.max(np.abs(output[split]-saved[split]))) for split in ['validation','test']}
        exported={split:saved[split].copy() for split in ['validation','test']}
    row=dict(year=year,seed=seed,family=family,inference_device=str(device),checkpoint_sha256=sha(folder/f'{seed}.pt'),
        max_validation_error=errors['validation'],max_test_error=errors['test'],passed=max(errors.values())<=1e-6,
        selected_epoch=cp.get('epoch',cp.get('epochs')),
        diagnostics={s:forecast_diagnostics(output[s],exported[s],p['data']['mask'][p[s]]) for s in ['validation','test']})
    file=LARGE/f'{family}_{year}_{seed}.npz'
    if file.exists():raise FileExistsError(file)
    np.savez_compressed(file,**output,train_ids=p['train'],validation_ids=p['validation'],test_ids=p['test'])
    save_json(OUT/f'checkpoint_{family}_{year}_{seed}.json',row)
    print(json.dumps(row),flush=True)
    if not row['passed']:raise ValueError('Frozen forecast tolerance violated; inspect before continuing')
    return output['train'],exported['test'],row


def main():
    global OUT,LARGE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--large',type=Path,default=LARGE)
    parser.add_argument('--saved-only',action='store_true',help='Use verified exports; leave training-prediction controls and checkpoint gate incomplete')
    args=parser.parse_args();OUT=args.out;LARGE=args.large
    for dest in [OUT,LARGE]:
        if dest.exists() and any(dest.iterdir()):raise FileExistsError('Preserve existing replay output '+str(dest))
        dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);started=time.monotonic()
    manifest=pd.read_csv(ROOT/'docs/ndx_artifacts_20261027/manifest.csv')
    verification=verify_contract({'files':manifest[['path','sha256']].to_dict('records')},ROOT)
    if verification['matched']!=len(manifest):raise ValueError('Publication hash mismatch')
    assert all((ROOT/r.path).stat().st_size==r.bytes for r in manifest.itertuples())
    save_json(OUT/'PUBLICATION_VERIFICATION.json',dict(files=len(manifest),matched=verification['matched'],bytes=int(manifest.bytes.sum())))
    timing=pd.read_csv(ROOT/'runs/ndx_extended_hours_20261024/TIMING.csv')
    cutoff=pd.to_datetime(timing.feature_cutoff_utc,utc=True);decision=pd.to_datetime(timing.decision_utc,utc=True);fills=pd.to_datetime(timing.fill_utc,utc=True)
    assert ((cutoff<decision)&(decision<fills)).all()
    all_scores={};ids_all=[];checkpoints=[];support=[];fallbacks=[]
    for year in [2022,2023,2024,2025]:
        p=prepare(year);d=p['data'];tr=p['train'];te=p['test'];ids_all.extend(te.tolist())
        np.testing.assert_array_equal(timing.origin,d['origin']);np.testing.assert_array_equal(timing.entry,d['entry'])
        eligibility=d['mask'][te];labelmask=d['labelmask'][tr]
        hlabel,counts=persistent_mean(d['y'][tr],labelmask)
        for j,name in enumerate(d['panel'].tickers):support.append(dict(year=year,family='Hlabel',seed='none',ticker=name,count=int(counts[j])))
        raw={'B':{},'D':{}};hpred={'B':{},'D':{}};changing={'B':{},'D':{}}
        for family in ['B','D']:
            for seed in SEEDS:
                if args.saved_only:
                    folder=(PARENT if family=='B' else REFERENCE/'dynamic_restricted')/str(year)
                    with np.load(folder/f'{seed}_predictions.npz',allow_pickle=False) as saved:test=saved['test'].copy()
                    raw[family][seed]=test
                else:
                    train,test,row=reforecast(p,family,year,seed);checkpoints.append(row)
                    mean,count=persistent_mean(train,d['mask'][tr]);raw[family][seed]=test
                    hpred[family][seed]=np.broadcast_to(mean,test.shape).copy();changing[family][seed]=test-mean
                    for j,name in enumerate(d['panel'].tickers):support.append(dict(year=year,family=family,seed=seed,ticker=name,count=int(count[j])))
        simple={'Hlabel':np.broadcast_to(hlabel,eligibility.shape).copy(),'REV5':d['reversal'][te],'MOM126':d['momentum'][te]}
        for name,channels in [('ridge5',list(range(5))),('coverage6',[12,13,14,22,23,24])]:
            features=p['features'][:,:,-1,:][...,channels]
            beta=fixed_ridge(features[tr].reshape(-1,len(channels)),(d['y'][tr]*100).reshape(-1),labelmask.reshape(-1))
            simple[name]=beta[0]+features[te]@beta[1:]
        for group,seeds in GROUPS.items():
            values={family:ensemble_predictions([raw[family][s] for s in seeds],eligibility) for family in ['B','D']}
            values['E']=.5*values['B']+.5*values['D']
            for family in ['B','D']:
                if not args.saved_only:
                    values['Hpred'+family]=ensemble_predictions([hpred[family][s] for s in seeds],eligibility)
                    values[family+'change']=ensemble_predictions([changing[family][s] for s in seeds],eligibility)
            for name,score in simple.items():
                all_missing=np.flatnonzero((eligibility&np.isfinite(score)).sum(1)==0)
                score,missing=neutralize_missing(score,eligibility)
                values[name]=ensemble_predictions([score],eligibility)
                fallbacks.append(dict(year=year,group=group,candidate=name,missing_scores=missing,
                    all_missing_origin_ids=json.dumps(d['origin'][te[all_missing]].tolist())))
            values['E_Hlabel']=.5*values['B']+.5*values['Hlabel']
            if not args.saved_only:values['E_HpredD']=.5*values['B']+.5*values['HpredD']
            for name,score in values.items():all_scores.setdefault((name,group),[]).append(score)
        print(('Prepared saved exports year ' if args.saved_only else 'Prepared and reproduced year ')+str(year),flush=True)
    ids=np.asarray(ids_all);data=load_data();expected=REFERENCE/'curves.csv'
    saved_curves=pd.read_csv(expected);saved_metrics=pd.read_csv(REFERENCE/'metrics.csv');saved_ledgers=pd.read_csv(REFERENCE/'ledgers.csv')
    metrics=[];curves=[];replay_errors=[];ranking=[]
    pd.DataFrame(checkpoints).to_csv(OUT/'CHECKPOINT_REPLAY.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'TRAINING_SUPPORT.csv',index=False)
    pd.DataFrame(fallbacks).to_csv(OUT/'COMPARATOR_FALLBACKS.csv',index=False)
    for (name,group),pieces in all_scores.items():
        scores=np.concatenate(pieces);ranking.append(dict(candidate=name,group=group,rankic=float(np.nanmean(rank_ic(scores,data['y'][ids],data['labelmask'][ids])))))
        np.savez_compressed(LARGE/f'scores_{name}_{group}.npz',scores=scores,ids=ids)
        for k in [5,10]:
            m,curve,ledger=continuous_backtest(data,ids,scores,k,1)
            account=f'{name}/{group}/{k}'
            metrics.append(dict(id=account,candidate=name,group=group,k=k,**m));curves.append(curve.assign(id=account))
            if name in ['B','D','E']:
                method={'B':'baseline','D':'dynamic_restricted','E':'dynamic_restricted_blend'}[name]
                key=f'{method}/{group}/ensemble/{k}/1';source=saved_curves[saved_curves.id==key]
                check=check_replayed_account(m,curve,ledger,saved_metrics[saved_metrics.id==key].iloc[0],source,saved_ledgers[saved_ledgers.id==key])
                replay_errors.append(dict(id=account,**check))
            ledger.to_csv(LARGE/f'ledger_{name}_{group}_{k}.csv',index=False)
        print('Account reconstruction '+name+'/'+group,flush=True)
    mf=pd.DataFrame(metrics);cf=pd.concat(curves,ignore_index=True)
    mf.to_csv(OUT/'ACCOUNT_METRICS.csv',index=False);cf.to_csv(LARGE/'ACCOUNT_CURVES.csv',index=False)
    pd.DataFrame(replay_errors).to_csv(OUT/'ORIGINAL_ACCOUNT_REPLAY.csv',index=False)
    pd.DataFrame(ranking).to_csv(OUT/'RANKING.csv',index=False)
    contrasts=[]
    for block in [21,5,63]:
        draws=stationary_indices(1003,10000,block,27102030)
        for right in ['B','E_Hlabel']:
            left=cf[cf.id=='E/ten_seed/10'].net.to_numpy();other=cf[cf.id==right+'/ten_seed/10'].net.to_numpy()
            contrasts.append(dict(right=right,block_sessions=block,**paired_contrast(left,other,draws,252)))
    ct=pd.DataFrame(contrasts)
    for block in [21,5,63]:
        select=ct.block_sessions==block;ct.loc[select,'mean_p_holm']=holm(ct.loc[select,'mean_p_one_sided'])
    ct.to_csv(OUT/'PRIMARY_CONTRASTS.csv',index=False)
    save_json(OUT/'RESULTS.json',dict(status='completed_saved_export_accounts_checkpoint_gate_unresolved' if args.saved_only else 'completed_checkpoint_saved_account_replay_and_initial_identity_controls',
        publication_files=len(manifest),checkpoints=len(checkpoints),accounts=len(metrics),original_accounts=len(replay_errors),
        max_forecast_error=max(max(r['max_validation_error'],r['max_test_error']) for r in checkpoints) if checkpoints else None,
        max_daily_replay_error=max(r['max_daily_error'] for r in replay_errors),seconds=time.monotonic()-started,
        limits=['No original per-stock order/position snapshots were saved; instrumented replay still needed',
                'Current control accounts are full-exposure five-session original execution, not overlay or future confirmation']))


if __name__=='__main__':main()
