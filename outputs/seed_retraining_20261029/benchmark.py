from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path.cwd()))
import numpy as np,torch
from training_device import configure_device,device_metadata
from ndx_extended_hours import ExtendedModel
from ndx_reference_models import make_reference,wavelet_snapshot
d=configure_device('cuda');groups=[{i,(i+1)%156} for i in range(156)];result={}
for name in ['think','dynamic']:
    torch.manual_seed(0)
    if name=='think':
        model=ExtendedModel(156,groups,'after_hours_premarket').to(d)
        x=torch.randn(8,16,156,5,device=d);extra=torch.randn(8,156,320,device=d);mask=torch.ones(8,156,device=d);y=torch.randn(8,156,device=d)
        call=lambda:model(x,extra,mask)[0]
    else:
        model=make_reference('dynamic_restricted',156,wavelet_snapshot(groups,156)).to(d)
        x=torch.randn(156,16,25,device=d);mask=torch.ones(156,dtype=torch.bool,device=d);y=torch.randn(156,device=d)
        call=lambda:model(x,torch.zeros(1,16,10,device=d),mask)[0]
    opt=torch.optim.Adam(model.parameters(),lr=.0001)
    for _ in range(2):
        opt.zero_grad();loss=(call()-y).square().mean();loss.backward();opt.step()
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(5):
        opt.zero_grad();loss=(call()-y).square().mean();loss.backward();opt.step()
    torch.cuda.synchronize();result[name]=dict(seconds_per_train_step=(time.perf_counter()-start)/5,parameters=sum(p.numel() for p in model.parameters()))
    del model,opt,x,y;torch.cuda.empty_cache()
result['environment']=device_metadata(d)
Path('outputs/seed_retraining_20261029/SYNTHETIC_BENCHMARK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
