from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import numpy as np,pandas as pd,torch
from pit_research_engine import load_data,fold,historical_groups
from ndx_extended_hours import ExtendedModel
from research_engine import ranks,corr
D=load_data();_,_,_=fold(D,2023);tr=fold(D,2023)[0];groups=historical_groups(D,tr);rows=[]
for seed in [7,19,42,61,97,1828801906,1868046758,239154507,287727767,981756941]:
    torch.manual_seed(seed);initial=ExtendedModel(156,groups,'after_hours_premarket').core.market[2].bias.detach().numpy()
    record=torch.load(Path('runs/ndx_extended_hours_20261024/after_hours_premarket/2023')/f'{seed}.pt',map_location='cpu',weights_only=False)
    final=record['state_dict']['core.market.2.bias'].numpy()
    rows.append(dict(seed=seed,initial_final_bias_rank_correlation=corr(ranks(initial),ranks(final)),initial_final_bias_mae=float(abs(initial-final).mean())))
r=pd.DataFrame(rows);r.to_csv('docs/ndx_seed_attribution_20261029/INITIALIZATION_BIAS_CHECK.csv',index=False);print(r.to_string(index=False))
