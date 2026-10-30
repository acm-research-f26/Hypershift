"""Bounded neutral-initialization/rank-objective retraining, all seeds retained.

Use ``python -m hypershift run run_ndx_rank_retraining --stage freeze|train``.
Evaluation and publication use report_ndx_rank_retraining after all120fits lock.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from ndx_loss_optimizer import prepared_fold
from ndx_extended_hours import ExtendedModel
from ndx_reference_models import make_reference, wavelet_snapshot
from ndx_rank_retraining import (neutralize_stock_initialization, rank_objective,
    split_validation, rank_statistics, percentile_predictions, epoch_choices)
from training_device import configure_device, device_metadata
from run_ndx_retention_replication import ROOT, sha, write
from run_ndx_rank_stability import verify as verify_prior

CONFIG = ROOT/'config/ndx_rank_retraining_20261029.json'
DEFAULT = 'ndx_rank_retraining_20261029'


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    write(temporary, value); temporary.replace(path)


def freeze(run):
    prior = verify_prior(ROOT/'runs/ndx_rank_stability_20261029_v2')
    inputs = dict(prior['input_sha256']); code = dict(prior['source_sha256'])
    paths = [CONFIG, ROOT/'config/ndx_seed_expansion_20261020.json',
             ROOT/'outputs/seed_retraining_20261029/SYNTHETIC_BENCHMARK.json']
    for name in ['features.npz', 'REVIEW.json']:
        paths.append(ROOT/'data/nasdaq_monthly_pit/extended_hours_features_20261024'/name)
    paths += list((ROOT/'docs/ndx_seed_attribution_20261029').glob('*.csv'))
    paths += list((ROOT/'.cache/architecture_references_20261024/dynamic_hypergraph/src/framework').glob('*.py'))
    paths += [ROOT/f'runs/ndx_extended_hours_20261024/after_hours_premarket/{year}/preprocessing.npz'
              for year in (2022, 2023, 2024, 2025)]
    for path in paths: inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    names = ['research/training/run_ndx_rank_retraining.py',
        'research/reporting/report_ndx_rank_retraining.py',
        'research/core/ndx_rank_retraining.py', 'research/core/ndx_seed_attribution.py',
        'research/core/training_device.py', 'research/training/run_ndx_alpaca_training.py',
        'research/training/run_robust_research.py', 'latest/models/ndx_loss_optimizer.py',
        'latest/models/ndx_reference_models.py', 'tests/test_ndx_rank_retraining.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'research/models').glob('*.py')]
    for name in names: code[name] = sha(ROOT/name)
    plan = json.loads(CONFIG.read_text())
    old = json.loads((ROOT/'config/ndx_seed_expansion_20261020.json').read_text())
    assert plan['groups']['fresh_five'] == old['new_seeds'][5:10]
    assert len(plan['groups']['fifteen_seed']) == len(set(plan['groups']['fifteen_seed'])) == 15
    run.mkdir(parents=True, exist_ok=False)
    write(run/'PROTOCOL.json', dict(config=plan, source_sha256=code, input_sha256=inputs))
    write(run/'FREEZE.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(run/'PROTOCOL.json'), fits_absent=True,
        scope='Exploratory development;120fixedfits;576fixedaccounts; no seed selection'))
    snapshot = run/'source_snapshot'
    for name in code:
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    write(run/'ENVIRONMENT.json', device_metadata(configure_device('cuda')))
    print('Frozen120fits,12epochs,15fixedseeds,576accounts before new fitting', flush=True)


def verify(run):
    protocol = json.loads((run/'PROTOCOL.json').read_text())
    assert sha(run/'PROTOCOL.json') == json.loads((run/'FREEZE.json').read_text())['protocol_sha256']
    for section in ('source_sha256', 'input_sha256'):
        for name, expected in protocol[section].items():
            if sha(ROOT/name) != expected: raise ValueError('Frozen source/input changed: '+name)
    return protocol['config']


def prepare(year, device):
    data, x, extra, train, validation, test, groups = prepared_fold(year)
    selection, audit = split_validation(data, validation)
    assert data['exit'][audit].max() < data['origin'][test].min()
    assert data['exit'][train].max() < data['origin'][selection].min()
    features = np.concatenate([x.transpose(0, 2, 1, 3), extra.reshape(len(extra), extra.shape[1], 16, 20)], axis=-1)
    assert np.isfinite(features).all()
    return dict(data=data, groups=groups, train=train, validation=validation, test=test,
        selection=selection, audit=audit, selection_rows=np.searchsorted(validation, selection),
        audit_rows=np.searchsorted(validation, audit), snapshot=wavelet_snapshot(groups, len(data['panel'].tickers)),
        x=torch.from_numpy(x).to(device), extra=torch.from_numpy(extra).to(device),
        features=torch.from_numpy(features).to(device),
        eligible=torch.from_numpy(data['mask']).to(device),
        observed=torch.from_numpy(data['labelmask']).to(device),
        target=torch.from_numpy(data['y']*100).to(device))


def forward(model, prepared, ids, family):
    if family == 'think':
        return model(prepared['x'][ids], prepared['extra'][ids], prepared['eligible'][ids].float())[0]
    return model(prepared['features'][ids], None, prepared['eligible'][ids])[0]


def predict(model, prepared, ids, family):
    model.eval(); output = []
    with torch.no_grad():
        if family == 'think':
            for start in range(0, len(ids), 8):
                output.append(forward(model, prepared, ids[start:start+8], family).cpu().numpy())
            return np.concatenate(output)
        for i in ids: output.append(forward(model, prepared, int(i), family).cpu().numpy())
    return np.stack(output)


def save_checkpoint(path, model, optimizer, history, epoch, family, seed):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(epoch=epoch, family=family, seed=seed,
        state_dict={key:value.detach().cpu().clone() for key, value in model.state_dict().items()},
        optimizer=optimizer.state_dict(), history=history,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), temporary)
    temporary.replace(path)


def fit(run, prepared, family, year, seed, plan, device):
    folder = run/'fits'/family/str(year)/str(seed); record = folder/'RECORD.json'
    if record.exists():
        saved = json.loads(record.read_text())
        for name, expected in saved['hashes'].items(): assert sha(folder/name) == expected
        return
    folder.mkdir(parents=True, exist_ok=True); torch.manual_seed(seed)
    model = (ExtendedModel(len(prepared['data']['panel'].tickers), prepared['groups'], 'after_hours_premarket')
             if family == 'think' else make_reference('dynamic_restricted', len(prepared['data']['panel'].tickers), prepared['snapshot'])).to(device)
    neutralize_stock_initialization(model, family)
    optimizer = (torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
                 if family == 'think' else torch.optim.Adam(model.parameters(), lr=.0001))
    history = []; start = 1; checkpoints = sorted(folder.glob('epoch_*.pt'))
    if checkpoints:
        # Checkpoint contains its own complete history and random state, so an
        # interrupted history-JSON write cannot change the resumed trajectory.
        saved = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        history = saved['history']; start = saved['epoch']+1
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        save_checkpoint(folder/'epoch_00.pt', model, optimizer, [], 0, family, seed)
    begin = time.monotonic(); validation = prepared['validation']
    selrows, auditrows = prepared['selection_rows'], prepared['audit_rows']
    actual = prepared['data']['y'][validation]*100; observed = prepared['data']['labelmask'][validation]
    for epoch in range(start, plan['epochs']+1):
        model.train(); losses = []; batch = 8 if family == 'think' else 1
        for offset in range(0, len(prepared['train']), batch):
            sub = prepared['train'][offset:offset+batch]
            index = sub if family == 'think' else int(sub[0])
            optimizer.zero_grad(set_to_none=True)
            loss = rank_objective(forward(model, prepared, index, family),
                                  prepared['target'][index], prepared['observed'][index])
            if not torch.isfinite(loss): raise ValueError('Nonfinite rank loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm): raise ValueError('Nonfinite rank gradient')
            optimizer.step(); losses.append(float(loss.detach()))
        prediction = predict(model, prepared, validation, family)
        selected = rank_statistics(prediction[selrows], actual[selrows], observed[selrows])
        audit = rank_statistics(prediction[auditrows], actual[auditrows], observed[auditrows])
        with torch.no_grad():
            value = float(rank_objective(torch.from_numpy(prediction[selrows]),
                torch.from_numpy(actual[selrows]), torch.from_numpy(observed[selrows])))
        history.append(dict(epoch=epoch, training_loss=float(np.mean(losses)),
            selection_loss=value, selection_criterion=selected['criterion'],
            selection_rankic=selected['rankic'], selection_ndcg10=selected['ndcg10'],
            audit_rankic=audit['rankic'], audit_ndcg10=audit['ndcg10']))
        np.savez_compressed(folder/f'validation_{epoch:02d}.npz', validation=prediction)
        save_checkpoint(folder/f'epoch_{epoch:02d}.pt', model, optimizer, history, epoch, family, seed)
        atomic_json(folder/'HISTORY.json', history)
        atomic_json(run/'PROGRESS.json', dict(stage='training', year=year, family=family, seed=seed,
            epoch=epoch, total_epochs=plan['epochs'], completed_fits=len(list((run/'fits').rglob('RECORD.json'))),
            current_fit_seconds=time.monotonic()-begin, test_accounts_scored=0))
    choices = epoch_choices(history); diagnostics = []
    for method, epochs in choices.items():
        vp = []; tp = []
        for epoch in epochs:
            checkpoint = torch.load(folder/f'epoch_{epoch:02d}.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            v = predict(model, prepared, validation, family)
            t = predict(model, prepared, prepared['test'], family)
            if method == 'snapshots':
                v = percentile_predictions(v, prepared['data']['mask'][validation])
                t = percentile_predictions(t, prepared['data']['mask'][prepared['test']])
            vp.append(v); tp.append(t)
        v, t = np.mean(vp, axis=0), np.mean(tp, axis=0)
        if not np.isfinite(v).all() or not np.isfinite(t).all(): raise ValueError('Nonfinite forecast')
        path = folder/(method+'_predictions.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                np.testing.assert_array_equal(existing['validation'], v); np.testing.assert_array_equal(existing['test'], t)
        else: np.savez_compressed(path, validation=v, test=t)
        diagnostics.append(dict(method=method, epochs=epochs,
            selection=rank_statistics(v[selrows], actual[selrows], observed[selrows]),
            audit=rank_statistics(v[auditrows], actual[auditrows], observed[auditrows])))
    write(record, dict(family=family, year=year, seed=seed, selected=choices, diagnostics=diagnostics,
        selection_ids=prepared['selection'].tolist(), audit_ids=prepared['audit'].tolist(),
        hashes={path.name:sha(path) for path in folder.iterdir() if path.is_file() and path.name != 'RECORD.json'}))
    print(f'Fit {family}/{year}/{seed}: {choices}; {time.monotonic()-begin:.0f}s', flush=True)
    del model, optimizer; torch.cuda.empty_cache()


def train(run):
    plan = verify(run)
    if (run/'FITS_COMPLETE.json').exists(): raise FileExistsError('All fits already complete')
    device = configure_device('cuda')
    for year in plan['years']:
        prepared = prepare(year, device)
        write(run/f'FOLD_{year}.json', dict(train=prepared['train'].tolist(),
            validation=prepared['validation'].tolist(), selection=prepared['selection'].tolist(),
            audit=prepared['audit'].tolist(), test=prepared['test'].tolist()))
        for family in plan['families']:
            for seed in plan['groups']['fifteen_seed']:
                fit(run, prepared, family, year, seed, plan, device)
        del prepared; torch.cuda.empty_cache()
    records = sorted((run/'fits').rglob('RECORD.json')); assert len(records) == plan['fits'] == 120
    hashes = {}
    for record in records:
        for name, expected in json.loads(record.read_text())['hashes'].items():
            path = record.parent/name; assert sha(path) == expected
            hashes[path.relative_to(run).as_posix()] = expected
        hashes[record.relative_to(run).as_posix()] = sha(record)
    verify(run)
    write(run/'FITS_COMPLETE.json', dict(fits=120, hashes=hashes,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), test_account_metrics_absent=True))
    atomic_json(run/'PROGRESS.json', dict(stage='fits_complete', completed_fits=120, test_accounts_scored=0))
    print('All120fits locked before scoring any new financial accounts', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze','train'], required=True)
    parser.add_argument('--name', default=DEFAULT); args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name): parser.error('Invalid run name')
    run = ROOT/'runs'/args.name
    freeze(run) if args.stage == 'freeze' else train(run)


if __name__ == '__main__': main()
