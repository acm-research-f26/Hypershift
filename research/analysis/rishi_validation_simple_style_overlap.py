"""Compare fixed pretest style-only targets to saved validation targets."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as study
from ndx_drawdown_engine import allocations


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch60_validation_style_overlap';out.mkdir(parents=True,exist_ok=False)
    source=doc/'execution_batch58_validation_style_partial/COMMON_STOCKS.csv';f=pd.read_csv(source);data=study.load_data();names=np.asarray(data['panel'].tickers)
    roots={'issuer5':Path('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029/TARGETS.npz'),'equal10':Path('results/RISHI_VALIDATION_TOP10_20261029/TARGETS.npz')}
    saved={k:np.load(p,allow_pickle=False) for k,p in roots.items()};ids=np.concatenate([study.fold(data,y)[2] for y in (2022,2023,2024,2025)])
    for z in saved.values():np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['security_ids'],names)
    rows=[]
    for group in ('original_five','new_five','ten_seed'):
        cursor=0
        for year in (2022,2023,2024,2025):
            _,_,test=study.fold(data,year);sub=f[(f.group==group)&(f.year==year)].set_index('security_id');assert sub.index.is_unique
            assert set(sub.index)<=set(names);support=np.isin(names,sub.index);scores={}
            for label,column in [('beta','beta252'),('liquidity','log10adv20')]:
                ranks=sub[column].rank(method='average',pct=True);a=np.full(len(names),np.nan);idx={name:i for i,name in enumerate(names)}
                for name,value in ranks.items():a[idx[name]]=value
                scores[label]=a
            scores['equal_blend']=(scores['beta']+scores['liquidity'])/2
            for label,score in scores.items():
                score=score.copy();score[~support]=np.median(score[support]);matrix=np.broadcast_to(score,(len(test),len(names)))
                for kind,z in saved.items():
                    control=allocations(data,test,matrix,kind);reference=z[f'{group}__fixed'][cursor:cursor+len(test)]
                    for j,i in enumerate(test):
                        w=control[j];ref=reference[j];assert not np.any((ref>0)&~data['mask'][i]);np.testing.assert_allclose(ref.sum(),1,atol=1e-12)
                        rows.append(dict(group=group,year=year,origin=int(i),fill_date=str(data['panel'].dates[data['entry'][i]].date()),control=label,allocation=kind,
                            shared_names=int(np.count_nonzero((w>0)&(ref>0))),weight_overlap=float(np.minimum(w,ref).sum()),
                            fallback_eligible=int(np.count_nonzero(data['mask'][i]&~support)),fallback_selected=int(np.count_nonzero((w>0)&~support))))
            cursor+=len(test)
    frame=pd.DataFrame(rows);assert len(frame)==3618
    frame.to_csv(out/'ORIGINS.csv',index=False);frame.groupby(['group','year','control','allocation'],as_index=False)[['shared_names','weight_overlap','fallback_eligible','fallback_selected']].mean().to_csv(out/'ANNUAL.csv',index=False)
    inputs=[source,*roots.values(),Path(__file__),doc/'VALIDATION_SIMPLE_STYLE_OVERLAP_PROTOCOL_20261029.md'];(out/'INPUT_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2),encoding='utf-8')
    print(frame.query("group=='ten_seed'").groupby(['year','control','allocation'])[['weight_overlap','shared_names','fallback_selected']].mean().to_string())


if __name__=='__main__':main()
