"""Independently recalculate rank correlations from retained stock rows."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('docs/rishi_mechanism_20261027/execution_batch57_validation_style')
stocks=pd.read_csv(root/'STOCKS.csv');results=pd.read_csv(root/'CORRELATIONS.csv');maximum=0.
for row in results.itertuples():
    f=stocks[(stocks.year==row.year)&(stocks.group==row.group)&(stocks.feature==row.feature)]
    assert len(f)==row.pairs and not f.security_id.duplicated().any()
    corr=f.preference.rank(method='average').corr(f.value.rank(method='average'))
    maximum=max(maximum,abs(corr-row.spearman))
assert len(results)==48 and len(stocks)==4560 and maximum<1e-12
print(f'48 correlations and 4560 stock rows verified; max error {maximum:.3g}')
