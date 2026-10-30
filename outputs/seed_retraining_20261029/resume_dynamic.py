from pathlib import Path
import sys,shutil,json
sys.path.insert(0,str(Path.cwd()))
import numpy as np,torch
import run_ndx_rank_retraining as t
from training_device import configure_device
run=Path.cwd()/'runs/ndx_rank_retraining_20261029';temporary=run/'resume_dynamic_check';source=run/'fits/dynamic/2022/7';dest=temporary/'fits/dynamic/2022/7';dest.mkdir(parents=True,exist_ok=False)
for name in ['epoch_00.pt','epoch_01.pt']:shutil.copyfile(source/name,dest/name)
d=configure_device('cuda');p=t.prepare(2022,d);t.fit(temporary,p,'dynamic',2022,7,dict(epochs=2),d)
a=torch.load(source/'epoch_02.pt',map_location='cpu',weights_only=False)['state_dict'];b=torch.load(dest/'epoch_02.pt',map_location='cpu',weights_only=False)['state_dict']
error=max(float((a[k].double()-b[k].double()).abs().max()) for k in a)
with np.load(source/'validation_02.npz') as a,np.load(dest/'validation_02.npz') as b:pred=float(abs(a['validation']-b['validation']).max())
assert error<=1e-7 and pred<=1e-7
(run/'RESUME_DYNAMIC_CHECK.json').write_text(json.dumps(dict(status='passed',model_max_abs_error=error,validation_max_abs_error=pred,scope='Additional dynamic/Adam resume check; scientific model code unchanged'),indent=2)+'\n');print('DYNAMIC RESUME',error,pred)
