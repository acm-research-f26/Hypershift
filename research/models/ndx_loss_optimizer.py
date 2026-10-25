"""Controlled losses and optimizers for the unchanged extended-hours network."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from ndx_extended_hours import ExtendedModel, prepared, raw_auxiliary, scale_auxiliary, predict
from pit_research_engine import fold, historical_groups
from pseudohuber_think_mixer import pseudo_huber_loss
from run_robust_research import scale_inputs
from training_device import configure_device, tensor_on
from run_ndx_alpaca_training import ROOT, sha, write

RUN = ROOT/'runs/ndx_loss_optimizer_20261024'
PARENT = ROOT/'runs/ndx_extended_hours_20261024'
CONFIG = ROOT/'config/ndx_loss_optimizer_20261024.json'
METHODS = ['fixed_adamw_mse','fixed_adamw_huber','fixed_adamw_pseudohuber',
           'tuned_adamw_mse','tuned_radam_mse','tuned_sgd_mse']


def objective(score, y, mask, loss):
    if loss == 'mse':
        # Preserve parent arithmetic exactly for the production finite inputs.
        return ((score-y).square()*mask).sum()/mask.sum().clamp_min(1)
    if loss == 'pseudohuber': return pseudo_huber_loss(score, y, mask, 1.)
    if loss != 'huber': raise ValueError(loss)
    observed = mask.bool()
    point = 2*F.huber_loss(torch.where(observed,score,torch.zeros_like(score)),
                         torch.where(observed,y,torch.zeros_like(y)), reduction='none', delta=1.)
    return (point*mask).sum()/mask.sum().clamp_min(1)


def optimizer_for(model, optimizer, lr):
    if optimizer == 'adamw': return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
    if optimizer == 'radam': return torch.optim.RAdam(model.parameters(),lr=lr,weight_decay=.001,decoupled_weight_decay=True)
    if optimizer == 'sgd': return torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=.001)
    raise ValueError(optimizer)


def reference(loss, optimizer, lr): return loss == 'mse' and optimizer == 'adamw' and lr == .001
def folder_for(loss, optimizer, lr, year): return RUN/'fits'/loss/optimizer/f'{lr:.4g}'/str(year)


def paths(loss, optimizer, lr, year, seed):
    folder = PARENT/'after_hours_premarket'/str(year) if reference(loss,optimizer,lr) else folder_for(loss,optimizer,lr,year)
    return folder/f'{seed}.pt',folder/f'{seed}_predictions.npz',folder/f'{seed}_history.json',folder/f'{seed}_record.json'


def checked_record(loss, optimizer, lr, year, seed):
    cp,pred,hist,record = paths(loss,optimizer,lr,year,seed)
    value = json.loads(record.read_text())
    for name,h in value['hashes'].items():
        if sha(record.parent/name) != h: raise ValueError('Changed fit artifact: '+name)
    return value


def fit_one(data,x,extra,tr,va,te,groups,device,loss,optimizer,lr,year,seed,folder,plan):
    folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed)
    model = ExtendedModel(len(data['panel'].tickers),groups,'after_hours_premarket').to(device)
    optim = optimizer_for(model,optimizer,lr);target = data['y']*100
    best = np.inf;state = None;stale = 0;chosen = 0;history = [];termination = 'epoch_budget'
    for epoch in range(1,plan['epochs']+1):
        model.train();total = 0.;count = 0;failed = False
        for start in range(0,len(tr),plan['batch']):
            sub = tr[start:start+plan['batch']];optim.zero_grad(set_to_none=True)
            score,_ = model(tensor_on(x[sub],device),tensor_on(extra[sub],device),tensor_on(data['mask'][sub].astype('float32'),device))
            value = objective(score,tensor_on(target[sub],device),tensor_on(data['labelmask'][sub].astype('float32'),device),loss)
            if not torch.isfinite(value): failed = True;termination = 'nonfinite_loss';break
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_norm_cap'])
            if not torch.isfinite(norm): failed = True;termination = 'nonfinite_gradient';break
            optim.step();total += float(value.detach())*len(sub);count += len(sub)
        if failed: break
        vp = predict(model,x,extra,data['mask'],va,device,plan['batch'])
        mse = float(((vp-target[va])**2)[data['labelmask'][va]].mean())
        history.append(dict(epoch=epoch,training_loss=total/count,validation_mse_pp2=mse))
        if not np.isfinite(mse): termination = 'nonfinite_validation';break
        if mse < best-plan['minimum_improvement']:
            best = mse;chosen = epoch;stale = 0;state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale += 1
        if stale >= plan['patience']: termination = 'validation_patience';break
    record = folder/f'{seed}_record.json'
    if state is None:
        write(record,dict(status='failed_no_finite_checkpoint',loss=loss,optimizer=optimizer,lr=lr,year=year,seed=seed,termination=termination,hashes={}))
        raise RuntimeError('No finite checkpoint: '+str(record))
    model.load_state_dict(state)
    vp,tp = [predict(model,x,extra,data['mask'],sub,device,plan['batch']) for sub in [va,te]]
    if not np.isfinite(tp[data['mask'][te]]).all(): raise ValueError('Nonfinite selected test forecast')
    cp=folder/f'{seed}.pt';pred=folder/f'{seed}_predictions.npz';hp=folder/f'{seed}_history.json'
    torch.save(dict(state_dict=state,loss=loss,optimizer=optimizer,lr=lr,year=year,seed=seed,
        groups=[sorted(g) for g in groups],epoch=chosen),cp)
    np.savez_compressed(pred,validation=vp,test=tp);write(hp,history)
    value = dict(status='finite_checkpoint',loss=loss,optimizer=optimizer,lr=lr,year=year,seed=seed,
        chosen_epoch=chosen,validation_mse=best,epochs_run=len(history),termination=termination,
        hashes={p.name:sha(p) for p in [cp,pred,hp]})
    write(record,value)
    return value


def prepared_fold(year):
    data,cubes = prepared();tr,va,te = fold(data,year);x,mean,std = scale_inputs(data,tr)
    extra,amean,astd = scale_auxiliary(raw_auxiliary(data,cubes,'after_hours_premarket'),data['mask'],tr)
    groups = historical_groups(data,tr)
    with np.load(PARENT/'after_hours_premarket'/str(year)/'preprocessing.npz') as saved:
        for name,value in dict(train=tr,val=va,test=te,mean=mean,std=std,aux_mean=amean,aux_std=astd).items():np.testing.assert_array_equal(saved[name],value)
    return data,x,extra,tr,va,te,groups


def candidate_specs(plan,selection=None):
    specs = set()
    for year in plan['years']:
        for seed in plan['seeds']:
            for loss in ['mse','huber','pseudohuber']: specs.add((loss,'adamw',.001,year,seed))
        for optimizer,rates in plan['learning_rates'].items():
            for lr in rates:
                for seed in plan['pilot_seeds']:specs.add(('mse',optimizer,lr,year,seed))
        if selection is not None:
            for optimizer in plan['learning_rates']:
                lr = selection[str(year)][optimizer]['lr']
                for seed in plan['seeds']:specs.add(('mse',optimizer,lr,year,seed))
    return sorted(specs)


def select_learning_rates(plan):
    result = {}
    for year in plan['years']:
        result[str(year)] = {}
        for optimizer,rates in plan['learning_rates'].items():
            choices = []
            for lr in rates:
                records = [checked_record('mse',optimizer,lr,year,seed) for seed in plan['pilot_seeds']]
                choices.append(dict(lr=lr,mean_validation_mse=float(np.mean([r['validation_mse'] for r in records]))))
            best = min(c['mean_validation_mse'] for c in choices)
            selected = min([c for c in choices if c['mean_validation_mse'] <= best+1e-7],key=lambda c:c['lr'])
            result[str(year)][optimizer] = dict(**selected,candidates=choices)
    return result


def method_spec(method,year,selection):
    family,optimizer,loss = method.split('_',2)
    return loss,optimizer,.001 if family == 'fixed' else selection[str(year)][optimizer]['lr']
