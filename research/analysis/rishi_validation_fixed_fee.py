"""Account-level fee attribution for existing fixed validation comparisons."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rishi_mechanism_batch1 import stationary_indices


def main():
    source=Path('results/RISHI_VALIDATION_FIXED_ACCOUNT_20261029');out=Path('docs/rishi_mechanism_20261027/execution_batch50_validation_fee')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    paths=np.load(source/'PATHS.npz',allow_pickle=False);dates=paths['dates'];years=pd.DatetimeIndex(dates).year
    draws=stationary_indices(len(dates),10000,21,27102030);rows=[];annual=[];inputs={}
    for group in ('original_five','new_five','ten_seed'):
        for policy in ('full','vol20'):
            values={}
            for arm in ('original','fixed'):
                key=f'{group}__{arm}__{policy}';p=source/f'{key}_ledger.csv';inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                ledger=pd.read_csv(p);net=paths[key];assert ledger.date.tolist()==dates.tolist()
                nav=ledger.nav.to_numpy();prev=np.r_[1e6,nav[:-1]]
                np.testing.assert_allclose(nav/prev-1,net,rtol=0,atol=1e-12)
                fee=ledger.fee.to_numpy()/prev;gross=net+fee
                values[arm]={'net':net,'gross':gross,'fee':fee}
            for component in ('net','gross','fee'):
                diff=values['original'][component]-values['fixed'][component]
                low,high=np.quantile(diff[draws].mean(1)*252,[.025,.975])
                rows.append(dict(group=group,policy=policy,component=component,original_annual=252*values['original'][component].mean(),
                    fixed_annual=252*values['fixed'][component].mean(),difference=252*diff.mean(),low=low,high=high))
                for year in (2022,2023,2024,2025):
                    m=years==year;annual.append(dict(group=group,policy=policy,component=component,year=year,difference=252*diff[m].mean()))
            np.testing.assert_allclose(values['original']['gross']-values['fixed']['gross']-(values['original']['fee']-values['fixed']['fee']),values['original']['net']-values['fixed']['net'],atol=1e-15,rtol=0)
    assert len(rows)==18
    pd.DataFrame(rows).to_csv(out/'COMPONENTS.csv',index=False);pd.DataFrame(annual).to_csv(out/'CALENDAR.csv',index=False)
    inputs[str(source/'PATHS.npz')]=hashlib.sha256((source/'PATHS.npz').read_bytes()).hexdigest();(out/'INPUT_HASHES.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).query("group=='ten_seed'").to_string(index=False))


if __name__=='__main__':main()
