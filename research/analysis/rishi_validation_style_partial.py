"""Partial rank associations on common pretest characteristic support."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    source=Path('docs/rishi_mechanism_20261027/execution_batch57_validation_style/STOCKS.csv');out=Path('docs/rishi_mechanism_20261027/execution_batch58_validation_style_partial');out.mkdir(parents=True,exist_ok=False)
    stocks=pd.read_csv(source);features=['beta252','momentum126','volatility20','log10adv20'];rows=[];common_rows=[]
    for group in ('original_five','new_five','ten_seed'):
        for year in (2022,2023,2024,2025):
            f=stocks[(stocks.group==group)&(stocks.year==year)];assert not f.duplicated(['security_id','feature']).any()
            pref=f.groupby('security_id').preference.agg(['min','max']);assert np.max(abs(pref['max']-pref['min']))<1e-12
            x=f.pivot(index='security_id',columns='feature',values='value').reindex(columns=features).dropna();pref=pref.loc[x.index,'min']
            ranks=x.rank(method='average');response=pref.rank(method='average').to_numpy()
            for name in features:
                y=ranks[name].to_numpy();others=ranks.drop(columns=name).to_numpy();design=np.column_stack([np.ones(len(x)),others])
                assert np.linalg.matrix_rank(design)==design.shape[1]
                ry=response-design@np.linalg.lstsq(design,response,rcond=None)[0];rx=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
                assert np.std(rx)>1e-12 and np.std(ry)>1e-12
                rows.append(dict(group=group,year=year,feature=name,pairs=len(x),raw=float(np.corrcoef(response,y)[0,1]),partial=float(np.corrcoef(ry,rx)[0,1])))
            for name in x.index:
                common_rows.append(dict(group=group,year=year,security_id=name,preference=float(pref[name]),**{k:float(x.loc[name,k]) for k in features}))
    assert len(rows)==48
    pd.DataFrame(rows).to_csv(out/'CORRELATIONS.csv',index=False);pd.DataFrame(common_rows).to_csv(out/'COMMON_STOCKS.csv',index=False)
    inputs=[source,Path(__file__),Path('docs/rishi_mechanism_20261027/VALIDATION_FIXED_STYLE_PARTIAL_PROTOCOL_20261029.md')]
    (out/'INPUT_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).query("group=='ten_seed'").to_string(index=False));print('common rows',len(common_rows))


if __name__=='__main__':main()
