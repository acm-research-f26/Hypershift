from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from research_engine import load_panel,prepare_panel,fold_indices
root = Path('data/largecap_20261010')
manifest = json.loads((root/'manifest.json').read_text())
for path,expected in manifest['hashes'].items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected
rows=[]
for dataset,names in manifest['selection'].items():
    panel=load_panel(dataset)
    assert panel.tickers == names
    assert len(names)==(101 if dataset.startswith('NASDAQ') else 100)
    for horizon in [1,5]:
        data=prepare_panel(panel,horizon)
        for year in [2022,2023,2024,2025]:
            train,val,test=fold_indices(data,year)
            counts=data['mask'][test].sum(1)
            assert counts.min()>=20
            assert data['exit'][train[-1]]<data['origin'][val[0]]
            assert data['exit'][val[-1]]<data['origin'][test[0]]
            assert np.isfinite(data['x'][test]).all()
            rows.append(dict(dataset=dataset,horizon=horizon,year=year,eligible_min=int(counts.min()),eligible_max=int(counts.max()),
                             eligible_mean=float(counts.mean()),test_signals=len(test),
                             missing_future_labels=int((data['mask'][test]&~data['labelmask'][test]).sum())))
frame=pd.DataFrame(rows)
frame.to_csv(root/'eligibility_preflight.csv',index=False)
print(frame.to_string(index=False),flush=True)
