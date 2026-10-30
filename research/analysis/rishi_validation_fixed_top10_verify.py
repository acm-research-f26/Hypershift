"""Independently verify fixed top-ten targets, accounts and paired means."""
from pathlib import Path
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as study
from research_engine import performance
from rishi_mechanism_batch1 import stationary_indices


def top(score,mask):
    active=np.flatnonzero(mask);cut=sorted(score[active],reverse=True)[9];higher=active[score[active]>cut];ties=active[score[active]==cut]
    w=np.zeros(len(score));w[higher]=.1;w[ties]=(10-len(higher))/(10*len(ties));return w


def main():
    root=Path('results/RISHI_VALIDATION_TOP10_20261029');out=Path('docs/rishi_mechanism_20261027/execution_batch54_validation_top10');z=np.load(root/'PATHS.npz',allow_pickle=False);t=np.load(root/'TARGETS.npz',allow_pickle=False)
    data=study.load_data();metrics=pd.read_csv(out/'METRICS.csv');contrasts=pd.read_csv(out/'CONTRASTS.csv');scores=np.load('results/RISHI_EXTENDED_PARENT_BLEND_20261028/SCORES.npz',allow_pickle=False)
    roots=[Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),Path('runs/ndx_reference_networks_20261024/dynamic_restricted')];maximum=0.;targets=0
    for group,seeds in study.GROUPS.items():
        cursor=0
        for year in (2022,2023,2024,2025):
            _,val,test=study.fold(data,year);families=[]
            for r in roots:
                ranks=[]
                for seed in seeds:
                    with np.load(r/str(year)/f'{seed}_predictions.npz',allow_pickle=False) as f:
                        x=pd.DataFrame(f['validation']).where(data['mask'][val]);ranks.append(x.rank(axis=1,pct=True))
                families.append(sum(ranks)/len(ranks))
            blend=(families[0]+families[1])/2;mean=blend.mean();support=blend.count()>=5;fixed=mean.where(support,mean[support].median()).to_numpy()
            for j,i in enumerate(test):
                for arm,score in [('fixed',fixed),('original',scores[f'after_hours_premarket__{group}'][cursor+j])]:
                    maximum=max(maximum,float(np.max(abs(top(score,data['mask'][i])-t[f'{group}__{arm}'][cursor+j]))));targets+=1
            cursor+=len(test)
    for row in metrics.itertuples():
        key=f'{row.group}__{row.arm}__c{row.cost}';r=z[key];ledger=pd.read_csv(root/f'{key}_ledger.csv');assert ledger.date.tolist()==z['dates'].tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
        maximum=max(maximum,float(np.max(abs(ledger.nav/1e6-np.cumprod(1+r)))))
        p=performance(r);maximum=max(maximum,*[abs(p[k]-getattr(row,k)) for k in ('sharpe','annualized_return','maximum_drawdown')])
    for block in (5,21,63):
        draw=stationary_indices(len(z['dates']),10000,block,27102032)
        for row in contrasts[contrasts.block==block].itertuples():
            delta=z[f'{row.group}__original__c{row.cost}']-z[f'{row.group}__fixed__c{row.cost}'];expected=np.r_[252*delta.mean(),np.quantile(252*delta[draw].mean(1),[.025,.975])]
            maximum=max(maximum,float(np.max(abs(expected-[row.mean_difference,row.mean_low,row.mean_high]))))
    assert targets==1206 and len(metrics)==12 and len(contrasts)==18 and maximum<1e-10
    print(f'{targets} target rows, 12 NAV/metrics and 18 paired mean intervals verified; max error {maximum:.3g}')


if __name__=='__main__':main()
