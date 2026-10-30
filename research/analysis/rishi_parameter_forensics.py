"""CPU-only original tensor/snapshot inspection; no model forward or optimizer."""
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pit_research_engine import load_data,fold
from ndx_extended_hours import ExtendedModel
from ndx_reference_models import make_reference
from rishi_raw_replay import load_selected_checkpoint,SEEDS,PARENT,REFERENCE


def indexed_vector_update(trained,initial,axis):
    return np.linalg.norm(np.asarray(trained,float)-np.asarray(initial,float),axis=axis)


def delta_stats(trained,initial):
    trained=np.asarray(trained,float);initial=np.asarray(initial,float)
    if trained.shape!=initial.shape or not trained.size or not np.isfinite(trained).all() or not np.isfinite(initial).all():raise ValueError('Finite aligned tensors required')
    change=trained-initial;a=float(np.linalg.norm(initial));b=float(np.linalg.norm(change))
    return dict(elements=int(trained.size),initial_l2=a,trained_l2=float(np.linalg.norm(trained)),
        update_l2=b,update_rms=float(np.sqrt(np.mean(change**2))),relative_update=b/a if a>0 else None,
        exact_unchanged_fraction=float(np.mean(change==0)))


def id_categories(eligible,labels,test):
    if np.any(labels>eligible) or np.any(eligible<0) or np.any(labels<0) or np.any(test<0):raise ValueError('Invalid supervision counts')
    return np.where(labels>=30,'supported',np.where(eligible>0,'sparse',np.where(test>0,'newly_eligible','unsupported')))


def operator_stats(w,inverse):
    w=np.asarray(w,float);inverse=np.asarray(inverse,float)
    if w.shape!=inverse.shape or w.ndim!=2 or w.shape[0]!=w.shape[1]:raise ValueError('Square original operators required')
    values=np.linalg.svd(w,compute_uv=False);eigen=np.linalg.eigvalsh((w+w.T)/2)
    inv_values=np.linalg.svd(inverse,compute_uv=False);inv_eigen=np.linalg.eigvalsh((inverse+inverse.T)/2)
    product=w@inverse;offdiag=product-np.diag(np.diag(product))
    return dict(nodes=len(w),rank=int(np.linalg.matrix_rank(w)),density=float(np.mean(w!=0)),inverse_density=float(np.mean(inverse!=0)),
        symmetry_max=float(np.max(np.abs(w-w.T))),singular_min=float(values[-1]),singular_max=float(values[0]),
        symmetric_eigen_min=float(eigen[0]),symmetric_eigen_max=float(eigen[-1]),
        condition=float(values[0]/values[-1]) if values[-1]>1e-15*values[0] else None,
        inverse_rank=int(np.linalg.matrix_rank(inverse)),inverse_symmetry_max=float(np.max(np.abs(inverse-inverse.T))),
        inverse_singular_min=float(inv_values[-1]),inverse_singular_max=float(inv_values[0]),
        inverse_symmetric_eigen_min=float(inv_eigen[0]),inverse_symmetric_eigen_max=float(inv_eigen[-1]),
        inverse_condition=float(inv_values[0]/inv_values[-1]) if inv_values[-1]>1e-15*inv_values[0] else None,
        inverse_defect_fro=float(np.linalg.norm(product-np.eye(len(w)))),uniform_filter_offdiag_fro=float(np.linalg.norm(offdiag)),
        uniform_filter_diagonal_mean=float(np.diag(product).mean()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='docs/rishi_mechanism_20261027/execution_batch14_parameters')
    parser.add_argument('--large',default='results/RISHI_MECHANISM_PARAMETERS_20261027')
    args=parser.parse_args()
    doc=Path('docs/rishi_mechanism_20261027');out=Path(args.out)
    large=Path(args.large)
    for dest in (out,large):
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
    torch.set_num_threads(2);data=load_data();n=len(data['panel'].tickers)
    tensors=[];nodes=[];fits=[];snapshots=[];snapshot_seen={};inputs=[];controls=[];cohorts=[]
    for year in [2022,2023,2024,2025]:
        train,_,test=fold(data,year);elig=data['mask'][train].sum(0);labels=data['labelmask'][train].sum(0);tested=data['mask'][test].sum(0)
        category=id_categories(elig,labels,tested)
        for family in ['B','D']:
            root=PARENT if family=='B' else REFERENCE/'dynamic_restricted'
            for seed in SEEDS:
                path=root/str(year)/f'{seed}.pt';inputs.append(path);cp=load_selected_checkpoint(path);state=cp['state_dict']
                if family=='D':snapshot={key:state[key].clone() for key in ['wavelets','wavelets_inv']}
                torch.manual_seed(seed)
                model=ExtendedModel(n,[set(g) for g in cp['groups']],'after_hours_premarket') if family=='B' else make_reference('dynamic_restricted',n,snapshot)
                initial={key:value.detach().cpu().numpy().copy() for key,value in model.state_dict().items()}
                if set(initial)!=set(state) or any(initial[key].shape!=tuple(state[key].shape) for key in initial):raise ValueError('Source initialization state shape/key mismatch')
                trained={key:value.detach().cpu().numpy().copy() for key,value in state.items()}
                epoch=cp['epoch'] if family=='B' else cp['epochs']['prediction']
                if family=='D':
                    for key in ['network.whconv1.par','network.whconv2.par']:
                        error=float(np.max(np.abs(trained[key]-initial[key])))
                        controls.append(dict(year=year,seed=seed,parameter=key,max_error=error,exact_equal=bool(np.array_equal(trained[key],initial[key]))))
                        if error!=0:
                            (out/'FAILED_INITIALIZATION_CONTROL.json').write_text(json.dumps(controls[-1],indent=2)+'\n')
                            raise ValueError('Unused random-vector reference failed; stop delta interpretation')
                    w=trained['wavelets'];inverse=trained['wavelets_inv'];digest=hashlib.sha256(w.tobytes()+inverse.tobytes()).hexdigest()
                    if (year,digest) not in snapshot_seen:
                        snapshots.append(dict(year=year,snapshot_sha256=digest,first_seed=seed,**operator_stats(w,inverse)))
                        snapshot_seen[year,digest]=True
                    beta=torch.softmax(state['network.beta'],dim=0).cpu().numpy()
                    fits.append(dict(family=family,year=year,seed=seed,selected_epoch=int(epoch),snapshot_sha256=digest,
                        own_fusion=float(beta[0]),dynamic_fusion=float(beta[1]),static_fusion=float(beta[2]),
                        snapshot_multiplier=float(state['network.params_of_snapshot'][0]),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    blocks={key:(trained[key],initial[key]) for key in ['network.whconv1.diagonal_weight_filter','network.whconv2.diagonal_weight_filter']}
                else:
                    fits.append(dict(family=family,year=year,seed=seed,selected_epoch=int(epoch),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    blocks={'mixer_incoming_l2':(np.linalg.norm(trained['core.market.0.weight'],axis=0),np.linalg.norm(initial['core.market.0.weight'],axis=0)),
                        'mixer_outgoing_l2':(np.linalg.norm(trained['core.market.2.weight'],axis=1),np.linalg.norm(initial['core.market.2.weight'],axis=1)),
                        'mixer_output_bias':(trained['core.market.2.bias'],initial['core.market.2.bias'])}
                    blocks['mixer_incoming_vector_update_l2']=(indexed_vector_update(trained['core.market.0.weight'],initial['core.market.0.weight'],0),np.zeros(n))
                    blocks['mixer_outgoing_vector_update_l2']=(indexed_vector_update(trained['core.market.2.weight'],initial['core.market.2.weight'],1),np.zeros(n))
                for key,_ in model.named_parameters():tensors.append(dict(family=family,year=year,seed=seed,parameter=key,selected_epoch=int(epoch),**delta_stats(trained[key],initial[key])))
                for key,(value,origin) in blocks.items():
                    for j,name in enumerate(data['panel'].tickers):
                        nodes.append(dict(family=family,year=year,seed=seed,parameter=key,security_id=name,cohort=category[j],
                            train_eligible=int(elig[j]),train_labels=int(labels[j]),test_eligible=int(tested[j]),
                            initial=float(origin[j]),trained=float(value[j]),change=float(value[j]-origin[j])))
                    for label in ['supported','sparse','newly_eligible','unsupported']:
                        use=category==label
                        if use.any():cohorts.append(dict(family=family,year=year,seed=seed,parameter=key,cohort=label,ids=int(use.sum()),
                            initial_mean=float(origin[use].mean()),trained_mean=float(value[use].mean()),
                            mean_abs_change=float(np.abs(value[use]-origin[use]).mean()),update_rms=float(np.sqrt(np.mean((value[use]-origin[use])**2))),
                            exact_unchanged_fraction=float(np.mean(value[use]==origin[use]))))
                del model
        print('Inspected original tensor year '+str(year),flush=True)
    pd.DataFrame(tensors).to_csv(out/'TENSOR_CHANGES.csv',index=False)
    pd.DataFrame(fits).to_csv(out/'CHECKPOINT_INVENTORY.csv',index=False)
    pd.DataFrame(controls).to_csv(out/'UNUSED_INITIALIZATION_CONTROLS.csv',index=False)
    pd.DataFrame(snapshots).to_csv(out/'SNAPSHOT_OPERATORS.csv',index=False)
    pd.DataFrame(cohorts).to_csv(out/'INDEXED_COHORTS.csv',index=False)
    pd.DataFrame(nodes).to_csv(large/'INDEXED_ID_CHANGES.csv',index=False)
    inputs.extend([Path(__file__),doc/'PARAMETER_FORENSICS_PROTOCOL.md',Path('research/models/ndx_extended_hours.py'),Path('latest/models/ndx_reference_models.py')])
    (out/'RESULTS.json').write_text(json.dumps(dict(checkpoints=len(fits),tensor_rows=len(tensors),indexed_id_rows=len(nodes),
        snapshots=len(snapshots),unused_controls=len(controls),all_unused_controls_exact=True,torch=str(torch.__version__),device='cpu',
        forward_calls=0,optimizer_steps=0,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limits='Parameter-only evidence; no branch contribution, forward replay acceptance or Hpred/perturbation gate claim'),indent=2)+'\n')


if __name__=='__main__':main()
