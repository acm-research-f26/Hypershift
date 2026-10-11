"""Complete new-universe reporting only after all declared fits are finished."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from research_engine import performance

ROOT = Path('runs/largecap_20261010')
REPORT = ROOT/'comparison'

def run(script,*args):
    command=[sys.executable,'-u',script,*map(str,args)]
    print('Reporting:',script,flush=True)
    subprocess.run(command,check=True)

def main():
    for directory in ['five_session','daily']:
        assert (ROOT/directory/'completion.json').is_file(), f'Incomplete training: {directory}'
    run('report_robust_research.py','--runs',ROOT/'five_session',ROOT/'daily','--out',REPORT,'--data-root','data/largecap_20261010')
    run('summarize_robust_findings.py','--run',REPORT)
    run('add_etf_controls.py','--run',REPORT,'--prices','data/largecap_20261010/adj_close.csv','--skip-historical')
    run('plot_combined_research.py','--run',REPORT)
    run('report_pooled_uncertainty.py','--run',REPORT)
    run('plot_combined_research.py','--run',REPORT)
    run('audit_pooled_chronology.py','--runs',ROOT/'five_session',ROOT/'daily','--out',REPORT)
    run('plot_robust_diagnostics.py','--run',REPORT)
    old_all=pd.read_csv('docs/all_metrics.csv')
    plan=json.loads(Path('config/largecap_retraining_20261010.json').read_text())
    fixed=[]
    fixed_yearly=[]
    changes=[]
    for horizon,folder,models in [(5,'five_session',plan['five_session_models']),(1,'daily',plan['daily_models'])]:
        for dataset in plan['datasets']:
            old_dataset='NASDAQ_recent' if dataset.startswith('NASDAQ') else 'NYSE_recent'
            old_folder=Path('runs/robust_2022_2025_final' if horizon==5 else 'runs/daily_2022_2025_final')
            for model in models+['ridge','momentum126','reversal5']:
                pieces=[pd.read_csv(ROOT/folder/f'{dataset}_{year}'/f'{model}_equal_k10_daily.csv') for year in plan['test_years']]
                for year,piece in zip(plan['test_years'],pieces):
                    fixed_yearly.append(dict(horizon=horizon,dataset=dataset,model=model,year=year,
                                             **{f'net_{key}':value for key,value in performance(piece.net).items()}))
                path=pd.concat(pieces,ignore_index=True)
                stats={f'net_{k}':v for k,v in performance(path.net).items()}
                row=dict(horizon=horizon,dataset=dataset,model=model,**stats)
                fixed.append(row)
                old_paths=[old_folder/f'{old_dataset}_{year}'/f'{model}_equal_k10_daily.csv' for year in plan['test_years']]
                if not all(file.exists() for file in old_paths):
                    continue
                previous=pd.concat([pd.read_csv(file) for file in old_paths],ignore_index=True)
                np.testing.assert_array_equal(path.date,previous.date)
                old_stat=performance(previous.net)
                changes.append(dict(horizon=horizon,dataset=dataset,old_dataset=old_dataset,model=model,
                                    old_net_sharpe=old_stat['sharpe'],new_net_sharpe=stats['net_sharpe'],
                                    delta_sharpe=stats['net_sharpe']-old_stat['sharpe'],
                                    old_cumulative_return=old_stat['cumulative_return'],new_cumulative_return=stats['net_cumulative_return']))
    pd.DataFrame(fixed).to_csv(REPORT/'fixed_k10_pooled.csv',index=False)
    yearly=pd.DataFrame(fixed_yearly)
    yearly.to_csv(REPORT/'fixed_k10_annual.csv',index=False)
    pairs=yearly[yearly.horizon==1].drop(columns='horizon').merge(
        yearly[yearly.horizon==5].drop(columns='horizon'),on=['dataset','model','year'],
        suffixes=('_daily','_five_session'),validate='one_to_one')
    assert len(pairs)==88
    pairs['daily_minus_five_session_sharpe']=pairs.net_sharpe_daily-pairs.net_sharpe_five_session
    pairs.to_csv(REPORT/'fixed_k10_horizon_pairs.csv',index=False)
    pd.DataFrame(changes).to_csv(REPORT/'old_new_fixed_k10_comparison.csv',index=False)
    completion=[]
    for folder in ['five_session','daily']:
        records=pd.read_csv(ROOT/folder/'training_records.csv')
        complete=json.loads((ROOT/folder/'completion.json').read_text())
        assert len(records)==complete['neural_fits']
        completion.append(dict(horizon=5 if folder=='five_session' else 1,neural_fits=len(records),
                               epoch_sum=int(records.epochs_run.sum()),elapsed_seconds=complete['elapsed_seconds']))
    assert sum(row['neural_fits'] for row in completion)==648
    (REPORT/'largecap_reporting_complete.json').write_text(json.dumps({'status':'complete','training':completion,
        'original_results_preserved':True,'current_membership_retrospective':True},indent=2))
    print('New-universe report tables, charts, confidence intervals and chronology checks complete.',flush=True)

if __name__=='__main__':
    main()
