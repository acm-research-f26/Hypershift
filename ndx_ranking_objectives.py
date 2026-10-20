"""Causal benchmark residuals and a fixed, ranking-aware training comparison."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ndx_alpaca_models import make
from training_device import configure_device, tensor_on, device_metadata
from run_robust_research import predict


def lagged_betas(stock_returns, benchmark_returns, window=126, minimum=60):
    """OLS on paired past observations, including the known signal-close return."""
    x = pd.DataFrame(stock_returns, dtype=float)
    q = pd.DataFrame(np.broadcast_to(np.asarray(benchmark_returns)[:, None], x.shape)).where(x.notna())
    valid = np.isfinite(x) & np.isfinite(q)
    x, q = x.where(valid), q.where(valid)
    rolling = lambda z: z.rolling(window, min_periods=minimum).mean()
    covariance = rolling(x*q)-rolling(x)*rolling(q)
    variance = rolling(q*q)-rolling(q)**2
    slope = covariance/variance.where(variance>1e-14)
    available = np.isfinite(slope.to_numpy())
    beta = (.5+.5*slope).clip(0, 2).fillna(1.).to_numpy()
    return beta, available


def target_data(data):
    root = Path(data['manifest']['root'])
    frame = pd.read_csv(root/'etf_adj_close.csv', index_col=0)
    np.testing.assert_array_equal(frame.index.to_numpy(),data['panel'].dates.strftime('%Y-%m-%d').to_numpy())
    q = frame['QQQ'].to_numpy(float)
    if not np.isfinite(q).all() or (q<=0).any():
        raise ValueError('Invalid benchmark prices')
    qr = np.r_[np.nan, q[1:]/q[:-1]-1]
    beta, support = lagged_betas(data['returns'], qr)
    at_signal = beta[data['origin']]
    benchmark = q[data['exit']]/q[data['entry']]-1
    residual = (data['y'].astype(float)-at_signal*benchmark[:, None])*100
    return dict(raw=data['y'].astype(float)*100, residual=residual,
                beta=at_signal, beta_supported=support[data['origin']], benchmark=benchmark)


def rank_values(values):
    return pd.Series(np.asarray(values, dtype=float)).rank(method='average').to_numpy()


def rank_ic(prediction, target, observed):
    result = []
    for p, y, m in zip(prediction, target, observed):
        if m.sum()<2:
            result.append(np.nan)
            continue
        if not np.isfinite(p[m]).all() or not np.isfinite(y[m]).all():
            raise ValueError('Nonfinite observed ranking inputs')
        a, b = rank_values(p[m]), rank_values(y[m])
        a, b = a-a.mean(), b-b.mean()
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        result.append(float(a@b/denominator) if denominator>1e-14 else 0.)
    return np.asarray(result)


def training_loss(score, target, observed, kind, weight=1., temperature=1.):
    selected = observed.bool()
    error = torch.where(selected, score-target, torch.zeros_like(score))
    mse = (error.square().sum(1)/selected.sum(1).clamp_min(1)).mean()
    if kind!='residual_rank':
        return mse
    # One orientation of each unequal pair, with equal weight to each date.
    winners = selected[:, :, None]&selected[:, None, :]&(target[:, :, None]>target[:, None, :])
    difference = (score[:, :, None]-score[:, None, :])/temperature
    loss = torch.where(winners, F.softplus(-difference), torch.zeros_like(difference))
    pair = (loss.sum((1,2))/winners.sum((1,2)).clamp_min(1)).mean()
    return mse+weight*pair


def fit(data, x, targets, train, val, test, groups, kind, seed, plan, folder):
    torch.manual_seed(seed)
    device = configure_device('cuda')
    model = make('think_mix',len(data['panel'].tickers),groups).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    y = targets['raw' if kind=='raw_mse' else 'residual'].astype('float32')
    best, stale, state, chosen, history = -np.inf, 0, None, 0, []
    for epoch in range(1,plan['epochs']+1):
        model.train()
        losses = []
        for start in range(0,len(train),plan['batch']):
            ids = train[start:start+plan['batch']]
            optimizer.zero_grad(set_to_none=True)
            score,_ = model(tensor_on(x[ids],device),tensor_on(data['mask'][ids].astype('float32'),device))
            loss = training_loss(score,tensor_on(y[ids],device),tensor_on(data['labelmask'][ids],device),kind,
                                 plan['ranking_weight'],plan['ranking_temperature_pp'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_norm_cap'])
            optimizer.step()
            losses.append(float(loss.detach()))
        vp = predict(model,x,data['mask'],val,plan['batch'])
        metric = float(np.nanmean(rank_ic(vp,targets['residual'][val],data['labelmask'][val])))
        history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_residual_rankic=metric,
                            validation_target_mse=float(((vp-y[val])**2)[data['labelmask'][val]].mean())))
        if metric>best+1e-7:
            best, chosen, stale = metric, epoch, 0
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
        if stale>=plan['patience']:
            break
    if state is None:
        raise ValueError('No finite ranking validation checkpoint')
    model.load_state_dict(state)
    vp, tp = [predict(model,x,data['mask'],ids,plan['batch']) for ids in [val,test]]
    metric = float(np.nanmean(rank_ic(vp,targets['residual'][val],data['labelmask'][val])))
    np.testing.assert_allclose(metric,best,rtol=0,atol=1e-12)
    if not np.isfinite(tp[data['mask'][test]]).all():
        raise ValueError('Nonfinite eligible test predictions')
    torch.save(dict(state_dict=state,kind=kind,seed=seed,epoch=chosen,features=13,
                    groups=[sorted(g) for g in groups],device=device_metadata(device)),folder/f'{kind}_{seed}.pt')
    (folder/f'{kind}_{seed}_history.json').write_text(json.dumps(history,indent=2))
    return vp,tp,dict(objective=kind,seed=seed,chosen_epoch=chosen,epochs_run=len(history),
                      best_validation_residual_rankic=best,parameters=sum(p.numel() for p in model.parameters()))
