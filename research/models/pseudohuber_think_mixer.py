"""Controlled pseudo-Huber extension without editing frozen training modules."""
import json
import numpy as np
import torch
from robust_models import make_model
from run_robust_research import predict, write_json
from training_device import configure_device, device_metadata, tensor_on

KIND = 'think_mix_pseudohuber'


def pseudo_huber_loss(score, target, mask, delta=1.):
    """Twice pseudo-Huber: MSE curvature near zero, bounded tail influence.

    Scores and targets are percentage points. Rationalizing the subtraction
    avoids cancellation for small errors. Mask before arithmetic for NaN safety.
    """
    if delta <= 0:
        raise ValueError('delta must be positive')
    observed = mask.bool()
    safe_score = torch.where(observed, score, torch.zeros_like(score))
    safe_target = torch.where(observed, target, torch.zeros_like(target))
    error = safe_score - safe_target
    point = 2 * delta * error.square() / (torch.sqrt(error.square() + delta**2) + delta)
    return (point * mask).sum() / mask.sum().clamp_min(1)


def fit_seed(data, x, train, val, test, seed, groups, plan, folder):
    """Match the frozen THINK+mixer fit; change only the training loss."""
    torch.manual_seed(seed)
    device = configure_device('cuda')
    model = make_model('think_mix', len(data['panel'].tickers), groups,
                       causal_membership=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'], weight_decay=.001)
    y = data['y'] * 100
    best, state, stale, history, chosen = np.inf, None, 0, [], 0
    for epoch in range(1, plan['epochs'] + 1):
        model.train()
        total, count = 0., 0
        for start in range(0, len(train), plan['batch']):
            sub = train[start:start + plan['batch']]
            optimizer.zero_grad(set_to_none=True)
            score, _ = model(tensor_on(x[sub], device), tensor_on(data['mask'][sub].astype('float32'), device))
            loss = pseudo_huber_loss(score, tensor_on(y[sub], device),
                tensor_on(data['labelmask'][sub].astype('float32'), device), plan['delta_pp'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite pseudo-Huber loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(sub)
            count += len(sub)
        vp = predict(model, x, data['mask'], val, plan['batch'])
        vmse = float(((vp-y[val])**2)[data['labelmask'][val]].mean())
        history.append(dict(epoch=epoch, train_loss=total/count, validation_mse_pp2=vmse))
        if vmse < best - 1e-7:
            best, state, stale, chosen = vmse, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0, epoch
        else:
            stale += 1
        if stale >= plan['patience']:
            break
    model.load_state_dict(state)
    torch.save(dict(state_dict=state, kind=KIND, seed=seed, hidden=16, lookback=16,
        groups=[sorted(g) for g in groups], features=5, epoch=chosen, delta_pp=plan['delta_pp'],
        loss='2 * delta^2 * (sqrt(1 + (error/delta)^2) - 1)',
        training_device=device_metadata(device)), folder/f'{KIND}_{seed}.pt')
    write_json(folder/f'{KIND}_{seed}_history.json', history)
    return (vp, predict(model, x, data['mask'], test, plan['batch']), chosen, len(history),
            sum(p.numel() for p in model.parameters()))
