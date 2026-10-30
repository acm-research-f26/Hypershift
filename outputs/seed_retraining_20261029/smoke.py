from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()))
import numpy as np,torch
from ndx_extended_hours import ExtendedModel
from ndx_reference_models import make_reference,wavelet_snapshot
from ndx_rank_retraining import neutralize_stock_initialization,rank_objective
from training_device import configure_device
device=configure_device('cuda');groups=[{i,(i+1)%156} for i in range(156)];result={}
for family in ['think','dynamic']:
    torch.manual_seed(0)
    model=(ExtendedModel(156,groups,'after_hours_premarket') if family=='think' else make_reference('dynamic_restricted',156,wavelet_snapshot(groups,156))).to(device)
    neutralize_stock_initialization(model,family)
    if family=='think':
        y=torch.randn(8,156,device=device);mask=torch.ones_like(y,dtype=torch.bool)
        score=model(torch.randn(8,16,156,5,device=device),torch.randn(8,156,320,device=device),mask.float())[0]
    else:
        y=torch.randn(156,device=device);mask=torch.ones_like(y,dtype=torch.bool)
        score=model(torch.randn(156,16,25,device=device),None,mask)[0]
    loss=rank_objective(score,y,mask);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result[family]={'status':'passed','loss':float(loss.detach())}
Path('outputs/seed_retraining_20261029/RANK_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
