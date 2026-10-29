"""Relate validation rank preferences to pretest stock characteristics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import run_ndx_drawdown_research as study
from rishi_validation_stable_rank import ranks
from rishi_mechanism_selection import asof_style


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch57_validation_style');out.mkdir(parents=True,exist_ok=False)
    data=study.load_data();qpath=Path('data/nasdaq_monthly_pit/assembled/etf_adj_close.csv');q=pd.read_csv(qpath,index_col=0,parse_dates=True).QQQ.pct_change(fill_method=None).to_numpy()
    roots=[Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),Path('runs/ndx_reference_networks_20261024/dynamic_restricted')];names=np.asarray(data['panel'].tickers)
    hashes={};rows=[];stocks=[]
    for year in (2022,2023,2024,2025):
        _,val,test=study.fold(data,year);i=int(test[0]);day=int(data['origin'][i]);assert data['exit'][val].max()<day
        style=asof_style(data,day,q)
        for group,seeds in study.GROUPS.items():
            families=[]
            for root in roots:
                values=[]
                for seed in seeds:
                    path=root/str(year)/f'{seed}_predictions.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path,allow_pickle=False) as z:values.append(ranks(z['validation'],data['mask'][val]))
                families.append(np.mean(values,axis=0))
            a=(families[0]+families[1])/2;n=np.isfinite(a).sum(0);pref=np.divide(np.nansum(a,axis=0),n,out=np.full(len(n),np.nan),where=n>0)
            for k,feature in enumerate(('beta252','momentum126','volatility20','log10adv20')):
                good=data['mask'][i]&(n>=5)&np.isfinite(pref)&np.isfinite(style[:,k]);x=pref[good];y=style[good,k]
                rho=float(spearmanr(x,y).statistic) if len(x)>=3 and np.ptp(x)>0 and np.ptp(y)>0 else np.nan
                rows.append(dict(year=year,group=group,origin=str(data['panel'].dates[day].date()),feature=feature,pairs=len(x),spearman=rho))
                for j in np.flatnonzero(good):stocks.append(dict(year=year,group=group,feature=feature,security_id=names[j],preference=float(pref[j]),value=float(style[j,k])))
    assert len(rows)==48
    pd.DataFrame(rows).to_csv(out/'CORRELATIONS.csv',index=False);pd.DataFrame(stocks).to_csv(out/'STOCKS.csv',index=False)
    for p in (qpath,Path(__file__),Path('research/analysis/rishi_mechanism_selection.py'),Path('docs/rishi_mechanism_20261027/VALIDATION_FIXED_STYLE_PROTOCOL_20261029.md')):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).query("group=='ten_seed'").to_string(index=False));print('stock rows',len(stocks))


if __name__=='__main__':main()
