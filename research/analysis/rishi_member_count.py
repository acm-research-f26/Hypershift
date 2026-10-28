"""Fixed random-initialization member-count/masking probes, no market data."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rishi_synthetic_gate import make_model
from rishi_synthetic_robustness import forecast


def toy_input(case,count,population,inactive):
    if case not in (0,1,2) or count not in (2,4,6,8) or population not in ('diverse','replicated_peer') or inactive not in ('zero','nan','garbage'):raise ValueError('Invalid toy case')
    rng=np.random.default_rng(np.random.SeedSequence([27102030,case]));x=rng.normal(0,.2,(8,16,25)).astype('float32');eligible=np.arange(8)<count
    if population=='replicated_peer':x[2:count]=x[1]
    x[~eligible]={'zero':0.,'nan':np.nan,'garbage':12345.}[inactive]
    return x,eligible


def evaluate(model,raw,mask,arm):
    x=torch.tensor(raw,requires_grad=True);eligible=torch.tensor(mask);model.zero_grad(set_to_none=True)
    score=forecast(model,x,eligible,arm);score[eligible].sum().backward()
    return score.detach().numpy(),x.grad.detach().numpy()


def main():
    doc=Path('docs/rishi_mechanism_20261027');out=doc/'synthetic_member_count_20261028'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);rows=[]
    for seed in (7,19,42):
        model=make_model(seed).eval()
        for case in (0,1,2):
            for population in ('diverse','replicated_peer'):
                baseline={}
                for count in (2,4,6,8):
                    for arm in ('full','temporal_only'):
                        outputs={};grads={};mask=None
                        for variant in (('zero','nan','garbage') if count<8 else ('zero',)):
                            raw,mask=toy_input(case,count,population,variant);outputs[variant],grads[variant]=evaluate(model,raw,mask,arm)
                        x=outputs['zero'];g=grads['zero'];finite=bool(np.isfinite(x[mask]).all() and np.isfinite(g).all())
                        invariant=None if count==8 else all(np.array_equal(outputs[v][mask],x[mask]) and np.array_equal(grads[v],g) for v in ('nan','garbage'))
                        inactive_gradient=None if count==8 else float(np.max(abs(g[~mask])))
                        if count==2:baseline[arm]=float(x[0])
                        rows.append(dict(seed=seed,case=case,population=population,active_count=count,arm=arm,focal=float(x[0]),
                            focal_change_from_two=float(x[0]-baseline[arm]),mean_active_abs=float(np.mean(abs(x[mask]))),
                            finite=finite,inactive_value_invariant=invariant,inactive_gradient_max=inactive_gradient))
    frame=pd.DataFrame(rows);frame.to_csv(out/'CELLS.csv',index=False)
    paths=[Path(__file__),doc/'MEMBER_COUNT_TOY_PROTOCOL.md',Path('research/analysis/rishi_synthetic_gate.py'),
        Path('research/analysis/rishi_synthetic_robustness.py'),doc/'source_snapshot/model.py',doc/'source_snapshot/layer.py',Path('latest/models/ndx_reference_models.py')]
    (out/'RESULTS.json').write_text(json.dumps({'cells':len(rows),'non_applicable':int(frame.inactive_value_invariant.isna().sum()),
        'finite_failures':int((~frame.finite).sum()),'invariance_failures':int(frame.inactive_value_invariant.eq(False).sum()),
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'limit':'Fixed random-initialization synthetic probes, no learned financial signal'},indent=2)+'\n')
    print(frame.groupby(['population','arm','active_count']).focal_change_from_two.agg(['mean','min','max']).to_string())
    print('finite failures',int((~frame.finite).sum()),'invariance failures',int(frame.inactive_value_invariant.eq(False).sum()))


if __name__=='__main__':main()
