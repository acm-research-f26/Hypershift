"""Fold-specific saved-rank variation, persistence and blend boundary diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data
from research_engine import target_weights


def rank_anova(scores,mask):
    scores=np.asarray(scores,float);mask=np.asarray(mask,bool)
    if scores.shape!=mask.shape or scores.ndim!=2 or not mask.any():raise ValueError('Score/mask axes')
    if not np.isfinite(scores[mask]).all():raise ValueError('Nonfinite active score')
    count=mask.sum(0);grand=float(scores[mask].mean())
    means=np.divide(np.where(mask,scores,0).sum(0),count,out=np.zeros(scores.shape[1]),where=count>0)
    total=float(((scores[mask]-grand)**2).sum())
    between=float((count*(means-grand)**2).sum())
    within=float(np.where(mask,(scores-means)**2,0).sum())
    if not np.isclose(total,between+within,rtol=1e-10,atol=1e-10):raise AssertionError('ANOVA identity')
    return dict(total_ss=total,between_id_ss=between,within_id_ss=within,
                between_id_fraction=between/total if total>0 else None,
                eligible_observations=int(mask.sum()),observed_ids=int((count>0).sum()))


def selection_categories(parent_weights,dynamic_weights):
    b=np.asarray(parent_weights)>0;d=np.asarray(dynamic_weights)>0
    return np.where(b&d,0,np.where(b,1,np.where(d,2,3)))


def adjacent_behavior(scores,mask,folds,k):
    rows=[]
    for i in range(1,len(scores)):
        if folds[i]!=folds[i-1]:continue
        common=mask[i]&mask[i-1]
        a=pd.Series(scores[i-1,common]).rank().to_numpy();b=pd.Series(scores[i,common]).rank().to_numpy()
        correlation=float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>0 and b.std()>0 else None
        wa=target_weights(scores[i-1],mask[i-1],k,'equal');wb=target_weights(scores[i],mask[i],k,'equal')
        rows.append(dict(row=i,rank_correlation=correlation,common_ids=int(common.sum()),
            membership_entries=int((mask[i]&~mask[i-1]).sum()),membership_exits=int((mask[i-1]&~mask[i]).sum()),
            selected_weight_overlap=float(np.minimum(wa,wb).sum())))
    return rows


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch5_behavior')
    large=Path('results/RISHI_MECHANISM_BEHAVIOR_20261027')
    for path in (out,large):
        if path.exists():raise FileExistsError(path)
        path.mkdir(parents=True)
    source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
    data=load_data();anova=[];persistence=[];categories=[];boundaries=[]
    names=['both','parent_only','dynamic_only','neither']
    for group in ['original_five','new_five','ten_seed']:
        saved={m:np.load(source/f'scores_{m}_{group}.npz') for m in ['B','D','E']}
        ids=saved['E']['ids'];scores={m:saved[m]['scores'] for m in saved}
        for m in saved:np.testing.assert_array_equal(ids,saved[m]['ids'])
        mask=data['mask'][ids];years=data['panel'].dates[data['entry'][ids]].year.to_numpy()
        for year in [2022,2023,2024,2025]:
            use=years==year
            for m in scores:
                anova.append(dict(candidate=m,group=group,year=year,**rank_anova(scores[m][use],mask[use])))
                for k in (5,10):
                    for row in adjacent_behavior(scores[m][use],mask[use],years[use],k):
                        persistence.append(dict(candidate=m,group=group,year=year,k=k,**row))
        for k in (5,10):
            for r,i in enumerate(ids):
                weights={m:target_weights(scores[m][r],mask[r],k,'equal') for m in scores}
                category=selection_categories(weights['B'],weights['D']);we=weights['E']
                observed=data['labelmask'][i]&np.isfinite(data['y'][i])
                for j in np.flatnonzero(mask[r]):
                    boundaries.append(dict(group=group,k=k,origin_id=int(i),year=int(years[r]),
                        signal_date=str(data['panel'].dates[data['origin'][i]].date()),
                        entry_date=str(data['panel'].dates[data['entry'][i]].date()),security_id=data['panel'].tickers[j],
                        B_score=float(scores['B'][r,j]),D_score=float(scores['D'][r,j]),E_score=float(scores['E'][r,j]),
                        B_weight=float(weights['B'][j]),D_weight=float(weights['D'][j]),E_weight=float(we[j]),
                        category=names[category[j]],label_observed=bool(observed[j]),
                        label_return=float(data['y'][i,j]) if observed[j] else None))
                for z,name in enumerate(names):
                    w=np.where(category==z,we,0.);valid=w*observed
                    categories.append(dict(group=group,k=k,year=int(years[r]),origin_id=int(i),category=name,
                        selected_weight=float(w.sum()),observed_weight=float(valid.sum()),
                        weighted_label_return=float((valid*np.nan_to_num(data['y'][i])).sum())))
    af=pd.DataFrame(anova);pf=pd.DataFrame(persistence);cf=pd.DataFrame(categories)
    af.to_csv(out/'FOLD_VARIATION.csv',index=False);pf.to_csv(large/'ADJACENT_ORIGINS.csv',index=False)
    pf.groupby(['candidate','group','year','k'],as_index=False).agg(
        adjacent_pairs=('row','size'),mean_rank_correlation=('rank_correlation','mean'),
        mean_selected_weight_overlap=('selected_weight_overlap','mean'),
        membership_entries=('membership_entries','sum'),membership_exits=('membership_exits','sum')).to_csv(out/'FOLD_PERSISTENCE.csv',index=False)
    cf.to_csv(large/'ORIGIN_CATEGORIES.csv',index=False)
    grouped=cf.groupby(['group','k','year','category'],as_index=False).agg(
        origins=('origin_id','size'),selected_weight_sum=('selected_weight','sum'),
        observed_weight_sum=('observed_weight','sum'),weighted_label_return_sum=('weighted_label_return','sum'))
    grouped['mean_selected_weight']=grouped.selected_weight_sum/grouped.origins
    grouped['observed_weight_fraction']=grouped.observed_weight_sum/grouped.selected_weight_sum.replace(0,np.nan)
    grouped['observed_weighted_label_return']=grouped.weighted_label_return_sum/grouped.observed_weight_sum.replace(0,np.nan)
    grouped.to_csv(out/'BLEND_SELECTION_CATEGORIES.csv',index=False)
    pd.DataFrame(boundaries).to_csv(large/'BOUNDARY_STOCK_ROWS.csv',index=False)
    (out/'RESULTS.json').write_text(json.dumps(dict(annual_variation_cells=len(af),adjacent_rows=len(pf),
        boundary_rows=len(boundaries),origins=len(ids),interpretation='Descriptive held-out ranks and frictionless origin outcomes; no test-mean trading control'),indent=2)+'\n')
    print(af[af.group=='ten_seed'].to_string(index=False))


if __name__=='__main__':main()
