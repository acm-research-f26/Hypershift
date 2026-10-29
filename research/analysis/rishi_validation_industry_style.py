"""Describe fixed validation preference after style and industry adjustment."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'execution_batch59_validation_industry';out.mkdir(parents=True,exist_ok=False)
    source=doc/'execution_batch58_validation_style_partial/COMMON_STOCKS.csv';classfile=doc/'execution_batch21_saved_industry/DETAIL.csv'
    data=pd.read_csv(source);cl=pd.read_csv(classfile);cl=cl[cl.model=='E'][['year','security_id','industry']];assert not cl.duplicated(['year','security_id']).any()
    joined=data.merge(cl,on=['year','security_id'],how='left',validate='many_to_one');features=['beta252','momentum126','volatility20','log10adv20'];rows=[]
    for group in ('original_five','new_five','ten_seed'):
        for year in (2022,2023,2024,2025):
            allrows=joined[(joined.group==group)&(joined.year==year)];f=allrows.dropna(subset=['industry']).sort_values('security_id').copy()
            ranks=f[['preference']+features].rank(method='average');cats=sorted(f.industry.unique());ind=np.column_stack([(f.industry.to_numpy()==c).astype(float) for c in cats[1:]]) if len(cats)>1 else np.empty((len(f),0))
            for feature in features:
                x=ranks[feature].to_numpy();y=ranks.preference.to_numpy();other=ranks[[c for c in features if c!=feature]].to_numpy();design=np.column_stack([np.ones(len(f)),other,ind])
                assert np.linalg.matrix_rank(design)==design.shape[1]
                rx=x-design@np.linalg.lstsq(design,x,rcond=None)[0];ry=y-design@np.linalg.lstsq(design,y,rcond=None)[0];assert np.std(rx)>1e-12 and np.std(ry)>1e-12
                rows.append(dict(group=group,year=year,feature=feature,classified=len(f),unclassified=len(allrows)-len(f),industries=len(cats),raw=float(np.corrcoef(x,y)[0,1]),adjusted=float(np.corrcoef(rx,ry)[0,1])))
    assert len(rows)==48
    joined.to_csv(out/'SUPPORT.csv',index=False);pd.DataFrame(rows).to_csv(out/'CORRELATIONS.csv',index=False)
    paths=[source,classfile,Path(__file__),doc/'VALIDATION_INDUSTRY_STYLE_PROTOCOL_20261029.md'];(out/'INPUT_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).query("group=='ten_seed'").to_string(index=False))


if __name__=='__main__':main()
