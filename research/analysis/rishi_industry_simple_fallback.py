"""Audit already-saved control target mass assigned to source fallback scores."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch26_industry_simple';large=Path('results/RISHI_INDUSTRY_SIMPLE_20261028')
    data=load_data();tickers=data['panel'].tickers;support_path=doc/'execution_batch2_exports/TRAINING_SUPPORT.csv';fallback_path=doc/'execution_batch2_exports/COMPARATOR_FALLBACKS.csv'
    support=pd.read_csv(support_path);support=support[support.family=='Hlabel'];fallback=pd.read_csv(fallback_path);fallback=fallback[fallback.group=='ten_seed']
    cut=pd.read_csv(doc/'execution_batch21_saved_industry/SUMMARY.csv');cut=cut[cut.model=='E'].sort_values('year');cutoffs=np.array(cut.signal_date,dtype='datetime64[D]')
    rows=[]
    for model in ('E','E_Hlabel'):
        for signal in ('Hlabel','REV5','MOM126','ridge5'):
            z=np.load(large/f'{model}_{signal}_weights.npz',allow_pickle=False);np.testing.assert_array_equal(z['security_ids'],tickers)
            for i,origin in enumerate(z['ids']):
                day=np.datetime64(data['panel'].dates[data['origin'][origin]].date(),'D');year=int(cut.iloc[int(np.searchsorted(cutoffs,day,side='right')-1)].year);eligible=data['mask'][origin]
                if signal=='Hlabel':
                    counts=support[support.year==year].set_index('ticker')['count'].reindex(tickers)
                    if counts.isna().any():raise ValueError('Missing Hlabel support')
                    use=counts.to_numpy()<30
                elif signal=='REV5':use=~np.isfinite(data['reversal'][origin])
                elif signal=='MOM126':use=~np.isfinite(data['momentum'][origin])
                else:
                    rec=fallback[(fallback.year==year)&(fallback.candidate==signal)]
                    if len(rec)!=1 or int(rec.iloc[0].missing_scores)!=0:raise ValueError('Ridge fallback needs stock-level evidence')
                    use=np.zeros(len(tickers),bool)
                rows.append(dict(model=model,signal=signal,year=year,origin=int(origin),eligible_fallback=int((eligible&use).sum()),
                    selected_fallback=int(((z['weights'][i]>0)&use).sum()),fallback_target_weight=float(z['weights'][i,use].sum())))
    frame=pd.DataFrame(rows)
    for year in (2022,2023,2024,2025):
        for signal in ('REV5','MOM126'):
            observed=int(frame[(frame.model=='E')&(frame.signal==signal)&(frame.year==year)].eligible_fallback.sum())
            expected=int(fallback[(fallback.year==year)&(fallback.candidate==signal)].missing_scores.iloc[0])
            if observed!=expected:raise ValueError('Aggregate neutral fallback counts do not match original export')
    frame.to_csv(out/'FALLBACK_BY_ORIGIN.csv',index=False)
    grouped=frame.groupby(['model','signal','year']).agg(origins=('origin','size'),eligible_fallbacks=('eligible_fallback','sum'),
        selected_fallbacks=('selected_fallback','sum'),mean_fallback_target=('fallback_target_weight','mean'),max_fallback_target=('fallback_target_weight','max')).reset_index()
    grouped.to_csv(out/'FALLBACK_ANNUAL.csv',index=False)
    paths=[support_path,fallback_path,Path(__file__),Path('research/core/pit_research_engine.py')]
    (out/'FALLBACK_AUDIT.json').write_text(json.dumps({'rows':len(frame),'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'definition':'Hlabel below 30 training labels; REV5/MOM126 unavailable raw scores; ridge source reports zero'},indent=2)+'\n')
    print(grouped.to_string(index=False))


if __name__=='__main__':main()
