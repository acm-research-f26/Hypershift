"""Independently spot-check stored rank data against raw cube and scores."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path('docs/rishi_mechanism_20261027/execution_batch30_extended_association')
members=pd.read_csv(root/'RANKS.csv');timing=pd.read_csv('runs/ndx_extended_hours_20261024/TIMING.csv')
cube=np.load('data/nasdaq_monthly_pit/extended_hours_features_20261024/features.npz',allow_pickle=False)
arrays={s:cube[s] for s in ('after_hours','premarket')};tickers=cube['tickers'].tolist();names=cube['feature_names'].tolist()
scores={m:np.load(f'results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/scores_{m}_ten_seed.npz',allow_pickle=False) for m in ('B','D','E','Hlabel')}
groups=members.groupby(['year','origin','model','feature'],sort=False);checked=0
for position,((year,origin,model,feature),g) in enumerate(groups):
    if position%67:continue
    segment,field={'AH_return':('after_hours','return'),'AH_coverage':('after_hours','coverage'),
                   'PM_return':('premarket','return'),'PM_coverage':('premarket','coverage')}[feature]
    day=int(timing.iloc[origin]['origin' if segment=='after_hours' else 'entry'])
    indices=[tickers.index(t) for t in g.security_id]
    z=scores[model];score=z['scores'][np.flatnonzero(z['ids']==origin)[0],indices]
    feature_values=arrays[segment][day,indices,names.index(field)]
    sr=pd.Series(score).rank(method='average',pct=True).to_numpy()
    fr=pd.Series(feature_values).rank(method='average',pct=True).to_numpy()
    assert np.allclose(sr,g.score_rank,rtol=0,atol=1e-12)
    assert np.allclose(fr,g.feature_rank,rtol=0,atol=1e-12)
    checked+=1
print('Raw feature/score rank spot-checks:',checked)
