"""Disaggregate fixed random-model synthetic membership sensitivity."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rishi_member_count import toy_input
from rishi_synthetic_gate import make_model


def components(model, raw, eligible):
    m = model.network
    mask = eligible[:, None]
    clean = torch.where(eligible[:, None, None], raw, torch.zeros_like(raw))
    encoded = m.positional_encoding(m.feature_embedding(clean))
    encoded = m.individual_attn(encoded)
    encoded = m.mixing_layer(torch.cat([encoded, m.intensity_func(encoded)], dim=-1))
    encoded, _ = m.temporal_attn(encoded)
    encoded = torch.where(mask, encoded, torch.zeros_like(encoded))
    pair = mask & mask.T
    snapshot = [dict(wavelets=model.wavelets*pair, wavelets_inv=model.wavelets_inv*pair)]
    h1 = F.leaky_relu(m.whconv1(encoded, snapshot, 0), .1)
    h1 = m.dropout(h1)*mask
    wavelet = F.leaky_relu(m.whconv2(h1, snapshot, 0), .1)*mask*m.params_of_snapshot[0]
    dynamic, incidence = m.dynamic_graph_layer(encoded)
    layer = m.dynamic_graph_layer
    edges = incidence.T @ encoded
    edges = layer.activation(layer.U.weight @ edges) + edges
    reconstructed = layer.norm(layer.activation(incidence @ edges) + encoded)
    weights = torch.softmax(m.beta, dim=0)
    fused = weights[0]*encoded + weights[1]*dynamic + weights[2]*wavelet
    score = torch.where(eligible, m.mlp(fused), torch.zeros(len(encoded)))
    return dict(encoded=encoded, dynamic=dynamic, wavelet=wavelet, fused=fused, score=score,
                incidence=incidence, reconstructed=reconstructed)


def main():
    doc = Path('docs/rishi_mechanism_20261027')
    out = doc/'synthetic_branch_count_20261028'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rows=[]
    with torch.no_grad():
        for seed in (7,19,42):
            model=make_model(seed).eval()
            for case in (0,1,2):
                for population in ('diverse','replicated_peer'):
                    baseline=None
                    for count in (2,4,6,8):
                        raw,mask=toy_input(case,count,population,'zero')
                        eligible=torch.tensor(mask)
                        output=components(model,torch.tensor(raw),eligible)
                        actual=model(torch.tensor(raw),None,eligible)[0]
                        if baseline is None:
                            baseline={k:output[k][0].clone() for k in ('encoded','dynamic','wavelet','fused','score')}
                        row=dict(seed=seed,case=case,population=population,active_count=count,
                            adapter_exact=bool(torch.equal(actual[eligible],output['score'][eligible])),
                            dynamic_exact=bool(torch.equal(output['dynamic'],output['reconstructed'])),
                            finite=all(bool(torch.isfinite(v).all()) for v in output.values()),
                            incidence_sum_error=float(torch.max(torch.abs(output['incidence'].sum(-1)-1))))
                        for key in ('encoded','dynamic','wavelet','fused'):
                            row[key+'_change_l2']=float(torch.linalg.vector_norm(output[key][0]-baseline[key]))
                        row['score_change']=float(output['score'][0]-baseline['score'])
                        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'CELLS.csv',index=False)
    paths=[Path(__file__),doc/'BRANCH_COUNT_PROTOCOL.md',doc/'MEMBER_COUNT_TOY_PROTOCOL.md',
           Path('research/analysis/rishi_member_count.py'),Path('research/analysis/rishi_synthetic_gate.py'),
           doc/'source_snapshot/model.py',doc/'source_snapshot/layer.py',Path('latest/models/ndx_reference_models.py')]
    metadata=dict(cells=len(frame), adapter_failures=int((~frame.adapter_exact).sum()),
        dynamic_failures=int((~frame.dynamic_exact).sum()), finite_failures=int((~frame.finite).sum()),
        incidence_max_error=float(frame.incidence_sum_error.max()),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'RESULTS.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k!='input_sha256'},indent=2))
    print(frame[frame.active_count==8].groupby('population')[[f'{k}_change_l2' for k in ('encoded','dynamic','wavelet','fused')]+['score_change']].agg(['min','max']).to_string())


if __name__=='__main__':
    main()
