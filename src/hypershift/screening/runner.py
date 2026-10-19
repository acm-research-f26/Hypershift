"""Resumable matched experiments. Saved forecasts remain distinct from smoke tests."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge

from .data import load_panel, sha256
from .metrics import rank_metrics, trading
from .models import ScreenModel, objective

ARMS = {
    'static': ('baseline', 'static', 'raw', 'mse', 'static'),
    'hyp': ('baseline', 'hyp', 'raw', 'mse', 'static'),
    'none': ('F', 'none', 'raw', 'mse', 'static'),
    'mlp': ('baseline', 'mlp', 'raw', 'mse', 'static'),
    'soft': ('A', 'soft', 'raw', 'mse', 'static'),
    'residual_soft': ('A', 'residual_soft', 'raw', 'mse', 'static'),
    'rolling': ('A', 'rolling', 'raw', 'mse', 'static'),
    'random_graph': ('A', 'random_graph', 'raw', 'mse', 'static'),
    'market_target': ('B', 'static', 'market', 'mse', 'static'),
    'cluster_target': ('B', 'static', 'cluster', 'mse', 'static'),
    'pairwise': ('C', 'static', 'raw', 'pairwise', 'static'),
    'listwise': ('C', 'static', 'raw', 'listwise', 'static'),
    'hybrid': ('C', 'static', 'raw', 'hybrid', 'static'),
    'multiscale': ('D', 'multiscale', 'raw', 'mse', 'static'),
    'mix': ('E', 'mix', 'raw', 'mse', 'static'),
    'gated': ('E', 'gated', 'raw', 'mse', 'mix'),
    'lowrank': ('E', 'lowrank', 'raw', 'mse', 'static'),
    'two': ('F', 'two', 'raw', 'mse', 'static'),
    'two_residual': ('F', 'two_residual', 'raw', 'mse', 'static'),
    'long': ('G', 'long', 'raw', 'mse', 'static'),
    'summary': ('G', 'summary', 'raw', 'mse', 'static'),
    'pretrain': ('H', 'pretrain', 'raw', 'mse', 'static'),
    'adaptive': ('I', 'adaptive', 'raw', 'mse', 'static'),
    'frozen': ('I', 'static', 'raw', 'mse', 'static'),
    'master': ('extra', 'master', 'raw', 'mse', 'static'),
    'ridge': ('baseline', None, 'raw', 'mse', 'static'),
    'momentum': ('baseline', None, 'raw', 'mse', 'static'),
    'reversal': ('baseline', None, 'raw', 'mse', 'static'),
    'random': ('baseline', None, 'raw', 'mse', 'static'),
}


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def source_manifest():
    root = Path(__file__).resolve().parents[3]
    files = [Path(__file__), Path(__file__).with_name('models.py'), Path(__file__).with_name('data.py'),
             Path(__file__).with_name('metrics.py'), root/'scripts'/'research_screen.py']
    for name in ('models', 'geometry', 'data'):
        files += sorted((root/'src'/'hypershift'/name).rglob('*.py'))
    return {str(p.relative_to(root)): sha256(p) for p in files}


def tensors(panel, signals, device):
    x, e = panel.batch(signals)
    return torch.as_tensor(x, device=device), torch.as_tensor(e, device=device)


def predict(model, panel, signals, batch, device):
    out = []
    model.eval()
    with torch.no_grad():
        for j in range(0, len(signals), batch):
            x, e = tensors(panel, signals[j:j+batch], device)
            out.append(model(x, e).cpu().numpy()*panel.yscale)
    return np.concatenate(out)


def summaries(panel, signals):
    pieces = []
    for j in range(0, len(signals), 16):
        x, _ = panel.batch(signals[j:j+16])
        pieces.append(np.concatenate([x[:, :, -s:].mean(2) for s in (4, 16, 64)]+[x[:, :, -1]], -1))
    return np.concatenate(pieces)


def statistical(arm, panel, seed, signals):
    if arm == 'random':
        return np.random.default_rng(seed).standard_normal((len(signals), len(panel.tickers))).astype(np.float32)*panel.yscale
    if arm in ('momentum', 'reversal'):
        pred = (panel.price[:, signals]/np.maximum(panel.price[:, signals-16], 1e-12)-1).T
        return pred if arm == 'momentum' else -pred
    features = summaries(panel, panel.train)
    y, m = panel.target(panel.train)
    ridge = Ridge(alpha=100.)
    ridge.fit(features[m], y[m])
    x = summaries(panel, signals)
    return ridge.predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])*panel.yscale


def fit(arm, panel, seed, path, cfg):
    kind, target_kind, loss_kind = ARMS[arm][1:4]
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = ScreenModel(kind, len(panel.tickers), cfg['hidden'], panel.groups).to(cfg['device'])
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    best, best_state, bad, start_epoch = float('inf'), None, 0, 0
    history = []
    if (path/'last.pt').exists():
        state = torch.load(path/'last.pt', map_location=cfg['device'], weights_only=False)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        best, best_state, bad, start_epoch = state['best'], state['best_state'], state['bad'], state['epoch']+1
        history = state['history']; rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'].cpu())
    elif kind == 'pretrain':
        # Mask reconstruction only on TRAIN input windows. No validation/test pretraining.
        for ep in range(5):
            losses = []
            for j in range(0, len(panel.train), cfg['batch']):
                x, e = tensors(panel, panel.train[j:j+cfg['batch']], cfg['device'])
                hide = (torch.rand_like(x[:, :, -16:]) < .2) & e[:, :, None, None]
                corrupted = x.clone(); corrupted[:, :, -16:] = torch.where(hide, 0., x[:, :, -16:])
                opt.zero_grad(); rec = model.reconstruct(corrupted, e)
                loss = ((rec-x[:, :, -16:])[hide]**2).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
                losses.append(loss.item())
            print('pretrain', ep, float(np.mean(losses)), flush=True)
        # Reset optimizer so supervised optimization has the same fresh moments.
        opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    # Fixed non-overlapping validation signal schedule for checkpoint selection;
    # evaluation forecasts remain daily. Shared by every neural arm.
    validation_signals = panel.valid[::5]
    val_y, val_m = panel.target(validation_signals, target_kind)
    for epoch in range(start_epoch, cfg['epochs']):
        if bad >= cfg['patience']:
            break
        model.train(); losses, grads = [], []
        indices = rng.permutation(panel.train)
        for j in range(0, len(indices), cfg['batch']):
            ts = indices[j:j+cfg['batch']]
            x, e = tensors(panel, ts, cfg['device'])
            y, m = panel.target(ts, target_kind)
            y = torch.as_tensor(y, device=cfg['device']); m = torch.as_tensor(m, device=cfg['device'])
            opt.zero_grad(); pred = model(x, e)
            loss = objective(pred, y, m, loss_kind)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            loss.backward(); g = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(g):
                raise ValueError('nonfinite gradient')
            opt.step(); losses.append(float(loss.detach())); grads.append(float(g))
        pv = predict(model, panel, validation_signals, cfg['batch'], cfg['device'])/panel.yscale
        vmse = float(np.mean((pv[val_m]-val_y[val_m])**2))
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation_MSE': vmse,
               'gradient_norm': float(np.mean(grads)), 'prediction_std': float(pv[val_m].std()),
               'embedding_variance': getattr(model, 'embedding_variance', None),
               'representation_cosine': getattr(model, 'representation_cosine', None),
               'assignment_entropy': getattr(model, 'assignment_entropy', None)}
        history.append(row)
        if vmse < best-1e-6:
            best = vmse; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
        state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'best': best,
                 'best_state': best_state, 'bad': bad, 'epoch': epoch, 'history': history,
                 'rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state()}
        torch.save(state, path/'last.tmp'); os.replace(path/'last.tmp', path/'last.pt')
        dump(path/'history.json', history)
        print(arm, seed, 'epoch', epoch, 'val', round(vmse, 6), 'bad', bad, flush=True)
    if best_state is None:
        raise ValueError('no valid best checkpoint')
    model.load_state_dict(best_state)
    torch.save(best_state, path/'best_state.pt')
    return (predict(model, panel, panel.evaluate, cfg['batch'], cfg['device']),
            {'epochs': len(history), 'best_validation_MSE': best,
             'parameters': sum(p.numel() for p in model.parameters()), 'history': history})


def run(args):
    torch.set_num_threads(args.threads)
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('requested GPU unavailable')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    source = source_manifest()
    for year in args.years:
        print('loading fold', year, flush=True)
        panel = load_panel(args.panel, year)
        if args.smoke:
            # Functionality validation only, retained separately from empirical records.
            panel.train = panel.train[:4]; panel.valid = panel.valid[:4]; panel.evaluate = panel.evaluate[:4]
        audit = {'dataset_sha256': panel.digest, 'stocks': len(panel.tickers),
                 'dates': [str(panel.dates[0]), str(panel.dates[-1])],
                 'train_signals': len(panel.train), 'validation_signals': len(panel.valid),
                 'evaluation_signals': len(panel.evaluate), 'train_last_maturity': str(panel.dates[panel.train[-1]+5]),
                 'validation_last_maturity': str(panel.dates[panel.valid[-1]+5]),
                 'scale': panel.scale.tolist(), 'target_scale': panel.yscale,
                 'eligible_counts': {k: [int(panel.eligible[:, ts].sum(0).min()), int(panel.eligible[:, ts].sum(0).max())]
                                     for k, ts in [('train', panel.train), ('validation', panel.valid), ('evaluation', panel.evaluate)]},
                 'notes': 'Exploratory availability-conditioned panel; no delisting payouts; later years not scored.',
                 'panel_path': str(args.panel)}
        dump(output/f'data_audit_{year}.json', audit)
        for arm in args.arms:
            reference_year = 2015 if year < 2020 else 2020
            if arm == 'frozen' and year == reference_year:
                continue  # Identical to static on this fold; do not duplicate it.
            arm_panel = panel
            if arm == 'frozen':
                old = load_panel(args.panel, reference_year)
                arm_panel = replace(panel, train=old.train, valid=old.valid, groups=old.groups,
                                    scale=old.scale, yscale=old.yscale)
            for seed in args.seeds:
                if arm in ('ridge', 'momentum', 'reversal') and seed != args.seeds[0]:
                    continue  # Deterministic baselines are not independent seeded fits.
                path = output/f'{year}'/arm/f'seed_{seed}'; path.mkdir(parents=True, exist_ok=True)
                cfg = {'arm': arm, 'year': year, 'seed': seed, 'dataset_sha256': panel.digest,
                       'source': source, 'epochs': 1 if args.smoke else args.epochs,
                       'patience': args.patience, 'batch': args.batch, 'hidden': 16,
                       'device': args.device, 'smoke': args.smoke, 'threads': args.threads,
                       'protocol': 'broad-screen-v2-self-edges', 'train_stride': 5, 'validation_stride': 5, 'horizon': 5}
                cfg['preprocessing'] = {'scale': arm_panel.scale.tolist(), 'target_scale': arm_panel.yscale,
                                        'groups_sha256': hashlib.sha256(arm_panel.groups.tobytes()).hexdigest(),
                                        'train_signals': arm_panel.train.tolist(), 'validation_signals': arm_panel.valid.tolist()}
                if (path/'config.json').exists():
                    if json.loads((path/'config.json').read_text()) != cfg:
                        raise ValueError(f'refusing incompatible resume: {path}')
                else:
                    dump(path/'config.json', cfg)
                if (path/'metrics.json').exists():
                    print('reuse', path, flush=True); continue
                start = time.monotonic()
                try:
                    if ARMS[arm][1] is None:
                        pred = statistical(arm, arm_panel, seed, panel.evaluate); training = {'epochs': 0, 'parameters': None}
                    else:
                        pred, training = fit(arm, arm_panel, seed, path, cfg)
                    raw, mask = panel.target(panel.evaluate)
                    raw *= panel.yscale
                    result, daily = rank_metrics(pred, raw, mask)
                    neutral, nm = panel.target(panel.evaluate, 'cluster')
                    nr, nd = rank_metrics(pred, neutral*panel.yscale, nm)
                    result['neutral_RankIC'] = nr['RankIC']
                    result['neutral_spread10'] = nr['spread10']
                    portfolio, pnl = trading(pred, panel.evaluate, panel)
                    result.update(portfolio)
                    result.update({'runtime_s': time.monotonic()-start, 'training': training,
                                   'year': year, 'seed': seed, 'arm': arm, 'experiment': ARMS[arm][0],
                                   'parent': ARMS[arm][4], 'smoke': args.smoke, 'status': 'complete'})
                    np.savez_compressed(path/'predictions.npz', prediction=pred, target=raw, mask=mask,
                                        signals=panel.evaluate, dates=panel.dates[panel.evaluate], tickers=panel.tickers)
                    np.savez_compressed(path/'daily.npz', **daily, neutral_RankIC=nd['RankIC'], portfolio=pnl)
                    result['artifacts'] = {name: sha256(path/name) for name in ('config.json', 'predictions.npz', 'daily.npz')}
                    dump(path/'metrics.json', result)
                    print('COMPLETE', arm, year, seed, 'RankIC', round(result['RankIC'], 6), 'seconds', round(result['runtime_s'], 1), flush=True)
                except Exception as ex:
                    dump(path/'failed.json', {'error': repr(ex), 'runtime_s': time.monotonic()-start})
                    raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', default='data/raw/rsr/data/alpaca_panel_2016_2023.npz')
    p.add_argument('--output', default='results/SCREEN_20261019')
    p.add_argument('--years', nargs='+', type=int, default=[2020, 2022])
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 19, 42])
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke and 'smoke' not in args.output.lower():
        p.error('smoke output must contain smoke to prevent accidental empirical reuse')
    if any(y not in (2015, 2016, 2020, 2022) for y in args.years):
        p.error('only predeclared exploratory folds 2015/2016 or 2020/2022 allowed')
    run(args)
