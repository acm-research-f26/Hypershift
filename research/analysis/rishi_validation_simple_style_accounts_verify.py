"""Independently verify style targets, account paths and paired mean intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pit_research_engine import load_data,fold
from research_engine import performance
from rishi_mechanism_batch1 import stationary_indices
from rishi_validation_simple_style_verify import target


def main():
    root=Path('results/RISHI_VALIDATION_STYLE_ACCOUNTS_20261029');out=Path('docs/rishi_mechanism_20261027/execution_batch61_validation_style_accounts')
    p=np.load(root/'PATHS.npz',allow_pickle=False);t=np.load(root/'TARGETS.npz',allow_pickle=False);d=load_data();names=np.asarray(d['panel'].tickers);ids=t['ids'];np.testing.assert_array_equal(names,t['security_ids'])
    metrics=pd.read_csv(out/'METRICS.csv');contrasts=pd.read_csv(out/'CONTRASTS.csv');common=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch58_validation_style_partial/COMMON_STOCKS.csv');origins=d['origin'][ids];days=d['panel'].dates.get_indexer(pd.DatetimeIndex(p['dates']));maximum=0.;checked=0
    for group in ('original_five','new_five','ten_seed'):
        cursor=0
        for year in (2022,2023,2024,2025):
            _,_,test=fold(d,year);sub=common[(common.group==group)&(common.year==year)].set_index('security_id');index={name:j for j,name in enumerate(names)};support=np.isin(names,sub.index);scores={}
            for label,col in [('beta','beta252'),('liquidity','log10adv20')]:
                a=np.full(len(names),np.nan)
                for name,value in zip(sub.index,rankdata(sub[col])/len(sub)):a[index[name]]=value
                scores[label]=a
            scores['equal_blend']=(scores['beta']+scores['liquidity'])/2
            for label,raw in scores.items():
                score=raw.copy();score[~support]=np.median(score[support])
                for kind in ('issuer5','equal10'):
                    for j,i in enumerate(test):
                        expected=target(score,d['mask'][i],names,kind);actual=t[f'{group}__{kind}__{label}'][cursor+j];maximum=max(maximum,float(np.max(abs(expected-actual))));checked+=1
            cursor+=len(test)
    for row in metrics.itertuples():
        key=f'{row.group}__{row.kind}__{row.arm}';r=p[key];ledger=pd.read_csv(root/f'{key}_ledger.csv');w=t[key]
        assert ledger.date.tolist()==p['dates'].tolist() and ledger.cash.min()>=-1e-5 and ledger.equity_exposure.iloc[-1]==0
        maximum=max(maximum,float(np.max(abs(ledger.nav/1e6-np.cumprod(1+r)))))
        m=performance(r);maximum=max(maximum,*[abs(m[k]-getattr(row,k)) for k in ('sharpe','annualized_return','maximum_drawdown')])
        exposure=[]
        for day in days:
            j=int(np.searchsorted(origins,day-1,side='right')-1);assert j>=0
            if row.kind=='equal10':exposure.append(1.);continue
            origin=origins[j];active=w[j]>0;hist=np.nan_to_num(d['returns'][origin-19:origin+1,active],nan=0.)@w[j,active]
            exposure.append(min(1.,.2/max(float(hist.std(ddof=1)*np.sqrt(252)),.05)))
        maximum=max(maximum,float(np.max(abs(np.asarray(exposure)-ledger.target_exposure))))
    for block in (5,21,63):
        draw=stationary_indices(len(days),10000,block,29102026)
        for row in contrasts[contrasts.block==block].itertuples():
            delta=p[f'{row.group}__{row.kind}__{row.arm}']-p[f'{row.group}__{row.kind}__{row.control}'];expected=np.r_[252*delta.mean(),np.quantile(252*delta[draw].mean(1),[.025,.975])]
            maximum=max(maximum,float(np.max(abs(expected-[row.mean_difference,row.mean_low,row.mean_high]))))
    assert checked==3618 and len(metrics)==30 and len(contrasts)==108 and maximum<1e-10
    print(f'{checked} style targets, 30 accounts/exposures and 108 paired means verified; max error {maximum:.3g}')


if __name__=='__main__':main()
