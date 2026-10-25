"""Verify checkpoint inference, validation epoch choices and baseline stresses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from research_engine import load_panel, prepare_panel, fold_indices, training_groups, performance
from run_robust_research import scale_inputs, predict, model_data, ensemble_predictions
from robust_models import make_model
from robust_engine import backtest

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows=[];checks=[]
    for study,datasets,roots in [('original',['NYSE_recent','NASDAQ_recent'],{5:'runs/robust_2022_2025_final',1:'runs/daily_2022_2025_final'}),
                                ('top100',['NYSE_top100_20261010','NASDAQ100_20261010'],{5:'runs/largecap_20261010/five_session',1:'runs/largecap_20261010/daily'})]:
      for h,rootstr in roots.items():
        root=Path(rootstr); protocol=json.loads((root/'PROTOCOL.json').read_text());batch=protocol['arguments']['batch']
        device=protocol.get('training_backend',{}).get('device','cpu')
        for d in datasets:
          data=prepare_panel(load_panel(d),horizon=h);stress={}
          for year in [2022,2023,2024,2025]:
            folder=root/f'{d}_{year}';train,val,test=fold_indices(data,year);x,_,_=scale_inputs(data,train);groups=training_groups(data,train)
            for kind in ['think','think_mix','think_huber','think_mix_huber']:
                cached=np.load(folder/f'{kind}_predictions.npz')
                for j,seed in enumerate([7,19,42]):
                    checkpoint=torch.load(folder/f'{kind}_{seed}.pt',weights_only=False,map_location='cpu')
                    assert checkpoint['groups']==[sorted(g) for g in groups]
                    history=json.loads((folder/f'{kind}_{seed}_history.json').read_text());best=np.inf;epoch=None
                    for r in history:
                        if r['validation_mse_pp2']<best-1e-7:best=r['validation_mse_pp2'];epoch=r['epoch']
                    assert checkpoint['epoch']==epoch
                    model=make_model(kind,len(data['panel'].tickers),groups).to(device);model.load_state_dict(checkpoint['state_dict'])
                    for split,ids in [('test',test),('validation',val)]:
                        reproduced=predict(model,x,data['mask'],ids,batch)
                        np.testing.assert_allclose(reproduced,cached[split][j],atol=2e-5,rtol=2e-5)
                    checks.append(dict(study=study,dataset=d,horizon=h,year=year,model=kind,seed=seed,epoch=epoch,status='checkpoint forecasts and validation stop reproduced'))
            if h==5:
              for kind in ['ridge','momentum126']:
                cache=np.load(folder/f'{kind}_predictions.npz');dd=model_data(data,kind);score=cache['test'][0]
                selected=json.loads((folder/f'{kind}_policy.json').read_text())['selected']
                for mult in [1,2,4]:
                    _,path,_=backtest(dd,test,score,selected['k'],selected['method'],cost_multiplier=mult)
                    stress.setdefault((kind,f'selected_cost_x{mult}'),[]).append(path)
                _,path,_=backtest(dd,test,-score,10,'equal')
                stress.setdefault((kind,'reversed_score_fixed_k10'),[]).append(path)
              for seed in range(20):
                score=np.random.default_rng(20261010+seed+year).normal(size=data['y'][test].shape)
                _,path,_=backtest(data,test,score,10,'equal')
                stress.setdefault(('random',str(seed)),[]).append(path)
          for (kind,scenario),pieces in stress.items():
              rows.append(dict(study=study,dataset=d,horizon=h,model=kind,scenario=scenario,**performance(pd.concat(pieces).net)))
          print('Verified checkpoints:',study,h,d,flush=True)
    out=Path('docs/audit/backtests_20261010');pd.DataFrame(rows).to_csv(out/'stress_and_random.csv',index=False)
    (out/'checkpoint_audit.json').write_text(json.dumps(dict(status='passed',checkpoints=len(checks),checks=checks),indent=2))

if __name__=='__main__':main()
