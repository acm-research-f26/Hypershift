"""Matched daily / intraday-informed models; frozen parent modules stay intact."""
import json
from pathlib import Path

import numpy as np
import torch

from research_models import RankModel, objective
from robust_models import robust_objective
from pseudohuber_think_mixer import pseudo_huber_loss
from run_robust_research import predict
from training_device import configure_device, device_metadata, tensor_on

FEATURES = 13


def make(kind, nodes, groups):
    base = 'think_mix' if kind in ('think_mix_huber', 'think_mix_pseudohuber') else (
        'think' if kind == 'think_huber' else kind)
    return RankModel(base, nodes, groups, features=FEATURES, lookback=16,
                     hidden=16, causal_membership=True)


def inputs(data, view, daily_features=None):
    old = data['x']
    extra = np.full((*old.shape[:-1], 8), np.nan, dtype='float32')
    extra[..., -1] = 0.
    if view == 'intraday_augmented':
        if daily_features is None:
            raise ValueError('Reviewed intraday features are required.')
        extra = np.stack([daily_features[t-15:t+1] for t in data['origin']])
    elif view != 'daily_control':
        raise ValueError('Unknown input view.')
    return np.concatenate([old, extra], axis=-1)


def scale(raw, mask, train):
    values = raw[train].transpose(0, 2, 1, 3)[mask[train]]
    finite = np.isfinite(values)
    count = finite.sum((0, 1), keepdims=True)
    mean = np.divide(np.where(finite, values, 0.).sum((0, 1), keepdims=True),
                     count, out=np.zeros((1, 1, FEATURES)), where=count > 0)
    square = np.where(finite, (values - mean)**2, 0.).sum((0, 1), keepdims=True)
    std = np.sqrt(np.divide(square, count, out=np.ones_like(mean), where=count > 0))
    std = np.maximum(std, 1e-5)
    x = np.nan_to_num(np.clip((raw - mean) / std, -8, 8), nan=0.) * .2
    return x.astype('float32'), mean, std


def loss_for(score, y, mask, kind):
    if kind == 'think_mix_pseudohuber':
        return pseudo_huber_loss(score, y, mask, 1.)
    return robust_objective(score, y, mask, kind)


def fit(data, x, train, val, test, groups, kind, seed, plan, folder):
    torch.manual_seed(seed)
    device = configure_device('cuda')
    model = make(kind, len(data['panel'].tickers), groups).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
    target = data['y'] * 100
    best, state, stale, chosen, history = np.inf, None, 0, 0, []
    for epoch in range(1, plan['epochs'] + 1):
        model.train()
        total, count = 0., 0
        for begin in range(0, len(train), plan['batch']):
            ids = train[begin:begin + plan['batch']]
            optimizer.zero_grad(set_to_none=True)
            score, _ = model(tensor_on(x[ids], device), tensor_on(data['mask'][ids].astype('float32'), device))
            loss = loss_for(score, tensor_on(target[ids], device),
                            tensor_on(data['labelmask'][ids].astype('float32'), device), kind)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), plan['gradient_norm_cap'])
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            count += len(ids)
        vp = predict(model, x, data['mask'], val, plan['batch'])
        mse = float(((vp - target[val])**2)[data['labelmask'][val]].mean())
        history.append(dict(epoch=epoch, training_loss=total/count, validation_mse=mse))
        if mse < best - 1e-7:
            best, chosen, stale = mse, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= plan['patience']:
            break
    if state is None:
        raise RuntimeError('No finite validation checkpoint')
    model.load_state_dict(state)
    vp = predict(model, x, data['mask'], val, plan['batch'])
    tp = predict(model, x, data['mask'], test, plan['batch'])
    if not np.isfinite(tp[data['mask'][test]]).all():
        raise RuntimeError('Nonfinite test forecast')
    np.testing.assert_allclose(((vp-target[val])**2)[data['labelmask'][val]].mean(), best, rtol=1e-6, atol=1e-6)
    torch.save(dict(state_dict=state, kind=kind, seed=seed, features=FEATURES,
                    epoch=chosen, groups=[sorted(g) for g in groups],
                    device=device_metadata(device)), folder/f'{kind}_{seed}.pt')
    (folder/f'{kind}_{seed}_history.json').write_text(json.dumps(history, indent=2))
    return vp, tp, dict(model=kind, seed=seed, chosen_epoch=chosen, epochs_run=len(history),
                         parameters=sum(p.numel() for p in model.parameters()), validation_mse=best)
