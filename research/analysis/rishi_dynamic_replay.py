"""Source-CUDA dynamic-only replay; parent checkpoint gate is kept separate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rishi_raw_replay as rr
from ndx_reference_training import prepare
from training_device import configure_device,device_metadata
from run_robust_research import ensemble_predictions
from pit_research_engine import continuous_backtest,load_data
from ndx_ranking_objectives import rank_ic


def main():
    rr.OUT=rr.DOC/'execution_batch2_dynamic_cuda'
    rr.LARGE=rr.ROOT/'results/RISHI_MECHANISM_REPLAY_20261027_DYNAMIC_CUDA'
    for p in [rr.OUT,rr.LARGE]:
        if p.exists() and any(p.iterdir()):raise FileExistsError('Preserve previous attempt')
        p.mkdir(parents=True,exist_ok=True)
    device=configure_device('cuda');rr.save_json(rr.OUT/'ENVIRONMENT.json',device_metadata(device))
    scores={};all_ids=[];rows=[];support=[]
    for year in [2022,2023,2024,2025]:
        p=prepare(year);d=p['data'];te=p['test'];mask=d['mask'][te];all_ids.extend(te.tolist())
        constant={};change={};parent={}
        for seed in rr.SEEDS:
            train,test,row=rr.reforecast(p,'D',year,seed,device);rows.append(row)
            mean,count=rr.persistent_mean(train,d['mask'][p['train']])
            constant[seed]=np.broadcast_to(mean,test.shape);change[seed]=test-mean
            with np.load(rr.PARENT/str(year)/f'{seed}_predictions.npz') as saved:parent[seed]=saved['test'].copy()
            for j,ticker in enumerate(d['panel'].tickers):support.append(dict(year=year,seed=seed,ticker=ticker,count=int(count[j]),mean=float(mean[j])))
        for group,seeds in rr.GROUPS.items():
            b=ensemble_predictions([parent[s] for s in seeds],mask)
            h=ensemble_predictions([constant[s] for s in seeds],mask)
            c=ensemble_predictions([change[s] for s in seeds],mask)
            for name,x in [('HpredD',h),('Dchange',c),('E_HpredD',.5*b+.5*h)]:scores.setdefault((name,group),[]).append(x)
    ids=np.asarray(all_ids);data=load_data();metrics=[];curves=[];ranking=[]
    for (name,group),parts in scores.items():
        s=np.concatenate(parts);np.savez_compressed(rr.LARGE/f'scores_{name}_{group}.npz',scores=s,ids=ids)
        ranking.append(dict(candidate=name,group=group,rankic=float(np.nanmean(rank_ic(s,data['y'][ids],data['labelmask'][ids])))))
        for k in [5,10]:
            m,c,l=continuous_backtest(data,ids,s,k,1)
            metrics.append(dict(candidate=name,group=group,k=k,**m));curves.append(c.assign(id=f'{name}/{group}/{k}'))
    pd.DataFrame(metrics).to_csv(rr.OUT/'ACCOUNT_METRICS.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(rr.LARGE/'ACCOUNT_CURVES.csv',index=False)
    pd.DataFrame(rows).to_csv(rr.OUT/'CHECKPOINT_REPLAY.csv',index=False)
    pd.DataFrame(support).to_csv(rr.OUT/'TRAINING_PREDICTION_SUPPORT.csv',index=False)
    pd.DataFrame(ranking).to_csv(rr.OUT/'RANKING.csv',index=False)
    rr.save_json(rr.OUT/'RESULTS.json',dict(status='dynamic_checkpoints_reproduced_parent_gate_unresolved',checkpoints=len(rows),accounts=len(metrics),
        max_forecast_error=max(max(r['max_validation_error'],r['max_test_error']) for r in rows)))


if __name__=='__main__':main()
