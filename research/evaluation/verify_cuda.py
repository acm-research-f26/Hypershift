"""Verify GPU execution with synthetic inputs; does not run a research backtest."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import numpy as np
import torch
from training_device import configure_device, device_metadata
from robust_models import EXPERIMENTS, make_model, robust_objective
from run_robust_research import fit_model as robust_fit, predict as robust_predict, scale_inputs
from run_architecture_research import fit_model as original_fit, predict as original_predict
from research_engine import prepare_panel, fold_indices, training_groups
from test_research import fixture


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/audit/cuda_verification.json'))
    args=parser.parse_args()
    device=configure_device('cuda')
    records=[]
    torch.manual_seed(20261008)
    nodes=100
    groups=[{i,(i+1)%nodes,(i+2)%nodes} for i in range(nodes)]
    x=torch.randn(8,16,nodes,5)*.05
    mask=torch.ones(8,nodes);mask[:,-4:]=0
    y=torch.randn(8,nodes)
    for kind in EXPERIMENTS:
        model=make_model(kind,nodes,groups).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            score,aux=model(x.to(device),mask.to(device))
            loss=robust_objective(score,y.to(device),mask.to(device),kind,aux,torch.ones_like(y).to(device))
            assert score.is_cuda and loss.is_cuda
            assert torch.isfinite(loss) and torch.isfinite(score).all()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        model.eval()
        with torch.no_grad():
            gpu_score=model(x.to(device),mask.to(device))[0].cpu()
            cpu_model=make_model(kind,nodes,groups)
            cpu_model.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()})
            cpu_model.eval()
            cpu_score=cpu_model(x,mask)[0]
        torch.testing.assert_close(gpu_score,cpu_score,rtol=.005,atol=.0005)
        records.append(dict(model=kind,device=str(device),optimizer_steps=3,
                            finite_outputs_and_gradients=True,elapsed_seconds=elapsed,
                            cpu_gpu_max_abs_prediction_difference=float((gpu_score-cpu_score).abs().max())))
        print(f'CUDA verified: {kind}',flush=True)
    # Exercise the actual fit/validation/checkpoint/predict code paths on a
    # synthetic chronological panel, with no real-stock/test performance study.
    data=prepare_panel(fixture())
    train,val,test=fold_indices(data,2017)
    scaled,_,_=scale_inputs(data,train)
    groups=training_groups(data,train)
    smoke=SimpleNamespace(device='cuda',lr=.001,epochs=1,patience=1,batch=40)
    runner_records=[]
    with tempfile.TemporaryDirectory(prefix='cuda-smoke-',dir='.cache') as temp:
        folder=Path(temp)
        for name,fit,predict,kind in [('robust',robust_fit,robust_predict,'think_mix_huber_rank_03'),
                                     ('original',original_fit,original_predict,'think_mix')]:
            model,chosen,completed=fit(data,scaled,train,val,kind,7,groups,smoke,folder)
            assert next(model.parameters()).is_cuda
            output=predict(model,scaled,data['mask'],test[:4],40)
            assert isinstance(output,np.ndarray) and np.isfinite(output).all()
            checkpoint=torch.load(folder/f'{kind}_7.pt',map_location='cpu',weights_only=False)
            assert all(t.device.type=='cpu' for t in checkpoint['state_dict'].values())
            assert checkpoint['training_device']['device']=='cuda'
            runner_records.append(dict(runner=name,synthetic_epoch_completed=True,numpy_predictions_verified=True,
                                       portable_cpu_checkpoint_verified=True))
    result=dict(status='passed',checked_on='2026-10-08',hardware=device_metadata(device),
                synthetic_architecture_checks=records,training_runner_checks=runner_records,
                peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                scope='Synthetic GPU smoke verification only. No historical models retrained and no backtest results changed.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'PASS: {len(records)} CUDA architectures and both training runners. Receipt: {args.output}',flush=True)


if __name__=='__main__':
    main()
