"""Frozen, matched three-epoch input-dropout/consistency continuation study."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

import run_ndx_rank_retraining as trainer
import run_ndx_beta_robustness as parent
from ndx_consistency_training import symmetric_rank_kl, stochastic_forward, ParameterEMA
from ndx_rank_retraining import rank_objective, rank_statistics, percentile_predictions
from training_device import configure_device, device_metadata
from run_ndx_retention_replication import sha, write
from ndx_quants_overlays import portfolio_targets
from ndx_drawdown_engine import market_inputs
from ndx_portfolio_robustness import average_targets, simulate_banded
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq
from audit_ndx_weak_market_short import value_replay
from research_engine import performance

ROOT = parent.ROOT
RUN = ROOT/'runs/ndx_consistency_retraining_20261031'
DOC = ROOT/'docs/ndx_consistency_retraining_20261031'
CODE = 'research/training/run_ndx_consistency_retraining.py'
CORE = 'research/core/ndx_consistency_training.py'
TEST = 'tests/test_ndx_consistency_training.py'
ARMS = ['dropout_control', 'consistency']
METHODS = ['original', 'dropout_control_raw', 'dropout_control_ema', 'consistency_raw', 'consistency_ema']


def source_folder(family, year, seed):
    reserve = json.loads((parent.reserve.CONF/'PROTOCOL.json').read_text())
    base = parent.reserve.CONF/'base' if seed in reserve['seeds'] else parent.prior.parent.SOURCE
    return base/'fits'/family/str(year)/str(seed)


def source_checkpoint(family, year, seed):
    folder = source_folder(family, year, seed)
    record = json.loads((folder/'RECORD.json').read_text())
    epoch = record['selected']['loss'][0]
    path = folder/f'epoch_{epoch:02d}.pt'
    assert sha(path) == record['hashes'][path.name], path
    return path


def make_model(prepared, family, device):
    nodes = len(prepared['data']['panel'].tickers)
    model = (trainer.ExtendedModel(nodes, prepared['groups'], 'after_hours_premarket') if family == 'think'
             else trainer.make_reference('dynamic_restricted', nodes, prepared['snapshot']))
    return model.to(device)


def atomic_checkpoint(path, state):
    temporary = path.with_suffix('.tmp'); torch.save(state, temporary)
    for attempt in range(50):
        try: temporary.replace(path); return
        except PermissionError:
            if attempt == 49: raise
            time.sleep(.1)


def preflight():
    if (RUN/'PROTOCOL.json').exists(): raise FileExistsError('Preserve the frozen preflight input')
    device = configure_device('cuda'); prepared = trainer.prepare(2025, device)
    result = dict(environment=device_metadata(device), families={})
    for family in ('think', 'dynamic'):
        torch.manual_seed(7); model = make_model(prepared, family, device)
        path = source_checkpoint(family, 2025, 7)
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['state_dict'])
        optim = torch.optim.AdamW(model.parameters(), lr=.00025) if family == 'think' else torch.optim.Adam(model.parameters(), lr=.000025)
        batch = 8 if family == 'think' else 1
        losses, kls = [], []; torch.cuda.synchronize(); begin = time.monotonic()
        model.train()
        for offset in range(0, 4*batch, batch):
            ids = prepared['train'][offset:offset+batch]
            index = ids if family == 'think' else int(ids[0])
            optim.zero_grad(set_to_none=True)
            a, b = [stochastic_forward(model, prepared, index, family) for _ in range(2)]
            kl = symmetric_rank_kl(a, b, prepared['eligible'][index])
            loss = .5*(rank_objective(a, prepared['target'][index], prepared['observed'][index])+
                       rank_objective(b, prepared['target'][index], prepared['observed'][index]))+kl
            assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step(); losses.append(float(loss.detach())); kls.append(float(kl.detach()))
        torch.cuda.synchronize()
        result['families'][family] = dict(seconds_per_step=(time.monotonic()-begin)/4,
            steps_per_latest_epoch=int(np.ceil(len(prepared['train'])/batch)), mean_kl=float(np.mean(kls)), finite=True)
        del model, optim; torch.cuda.empty_cache()
    out = ROOT/'outputs/consistency_retraining_20261031/PREFLIGHT.json'
    out.parent.mkdir(parents=True, exist_ok=True); write(out, result)
    print(json.dumps(result, indent=2), flush=True)


def freeze():
    plan = parent.verify(); upstream = json.loads((parent.RUN/'PROTOCOL.json').read_text())
    inputs, code = dict(upstream['input_sha256']), dict(upstream['source_sha256'])
    for name in (CODE, CORE, TEST, 'research/core/ndx_quants_overlays.py'):
        code[name] = sha(ROOT/name)
    for family in ('think', 'dynamic'):
        for year in plan['years']:
            for seed in plan['seed_groups']['expanded_twenty']:
                path = source_checkpoint(family, year, seed)
                for p in (path, path.parent/'RECORD.json', path.parent/'loss_predictions.npz'):
                    inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    pre = ROOT/'outputs/consistency_retraining_20261031/PREFLIGHT.json'
    inputs[pre.relative_to(ROOT).as_posix()] = sha(pre)
    plan = dict(years=plan['years'], groups=plan['groups'], seed_groups=plan['seed_groups'],
        seeds=plan['seed_groups']['expanded_twenty'], families=['think', 'dynamic'], arms=ARMS, methods=METHODS,
        epochs=3, fits=320, workers=4, dropout_probability=.1, kl_weight=1., temperature=.5, ema_decay=.99,
        cost_regimes=[dict(cost=1, borrow=.03), dict(cost=2, borrow=.06)],
        mixtures=['both', 'think', 'dynamic'], accounts=810,
        scope='A bounded continuation screen, not a full from-scratch architecture retrain.320fits=2arms*2families*4years*20seeds,3additionalepochs each. Existing checkpoints and all20seeds are spent development. No fresh market/seed holdout.',
        initialization='Start every matched pair from the same existing first-half-loss-selected checkpoint. Reset optimizer to AdamWlr0.00025wd0.001THINK or Adam lr0.000025dynamic;3epochs, batch8THINK/1dynamic, chronological order, clip1, deterministicFP32CUDA. No pretrained source is mutated.',
        matched_control='Both arms use two independent10%input-dropout passes and average the original supervised rank losses. THINK had no existingdropout;dynamic retains existinginternaldropout. Control reports KL but gives it weight0;consistency adds1*symKL between eligible cross-sectionalsoftmax(score/0.5) distributions. Same seed and RNG draws across paired arms.',
        averaging='Track parameter EMA decay0.99 after every optimizer step, initialized to source weights; copy fixed buffers. Evaluate raw/EMA trajectories separately. This is EMA,not an originalSWA reproduction andnot earlier averaging of selected predictions.',
        selection='For raw andEMA independently choose minimum original supervisedrank loss on prior-year firsthalf validation over epochs0,1,2,3; earliest ties. Epoch0 exactly preserves source. Secondhalf auditdiagnostics never choosecheckpoints. Allfitartifacts lock before any new financialaccount scores.',
        portfolio='Unchanged26week training prior50%+neuralranks50%. both uses25%THINK+25%dynamic;single-family controls use50%thatfamily. Select5issuers per fixed5seedcohort,20%basketvol20,continuousQQQshort<=75%/remaininggross/QQQvol;average cohorttargets. Same next-close,ADV,fees,calendarborrow,zero cashyield.',
        primary='Both-family prior ensemble: every7group baseSR improves original by>=0.02, fourdisjointspread<=0.25, no7group SR loss under2xfees/6%borrow. Consistency must also have positive worst7group mean delta versus its matched raw/EMA dropoutcontrol. Separateall7SR>1.5 flag. Family-only andsingle-seedresults arediagnostics;no seed filtering.',
        continuation='Only a passing consistency/EMA candidate justifies scaling this training recipe to a longer from-scratch budget. Architecture/prior-decomposition/multihorizon/portfolio-objective changes require separate frozen experiments; they are not silently mixed into this isolation study.',
        reporting='All810accounts,allindividualseeds,perfamily validation/audit,chosenepochs,trainingKL,base/stress,baselineexactpathparity,independentdollarpositionreplay. Initializationconsistencyisnot unseen-marketgeneralization.',
        sources=['https://arxiv.org/abs/2106.14448','https://proceedings.mlr.press/r16/izmailov18a.html'])
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN/'PROTOCOL.json', dict(config=plan, source_sha256=code, input_sha256=inputs))
    write(RUN/'FREEZE.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(RUN/'PROTOCOL.json'), new_fits_absent=True, new_accounts_absent=True))
    for name in (CODE, CORE, TEST):
        dest = RUN/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dest)
    write(RUN/'ENVIRONMENT.json', device_metadata(configure_device('cuda')))
    print('Frozen320three-epoch continuation fits and810accounts;20fixedseeds', flush=True)


def verify(full=True):
    p = json.loads((RUN/'PROTOCOL.json').read_text())
    assert sha(RUN/'PROTOCOL.json') == json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    for section in ('source_sha256', 'input_sha256') if full else ('source_sha256',):
        for name, expected in p[section].items(): assert sha(ROOT/name) == expected, name
    return p['config']


def fit(prepared, family, year, seed, arm, plan, device):
    dest = RUN/'fits'/arm/family/str(year)/str(seed); record_path = dest/'RECORD.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        for name, expected in record['hashes'].items(): assert sha(dest/name) == expected, name
        return
    dest.mkdir(parents=True, exist_ok=True); torch.manual_seed(seed)
    model = make_model(prepared, family, device)
    checkpoint = source_checkpoint(family, year, seed)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'])
    optimizer = (torch.optim.AdamW(model.parameters(), lr=.00025, weight_decay=.001) if family == 'think'
                 else torch.optim.Adam(model.parameters(), lr=.000025))
    ema = ParameterEMA(model, plan['ema_decay']); history = []; start = 0
    checkpoints = sorted(dest.glob('epoch_*.pt'))
    if checkpoints:
        saved = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        ema.load_state_dict(saved['ema'], device); history = saved['history']; start = saved['epoch']+1
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        torch.manual_seed(seed ^ 20261031)
    selection, audit = prepared['selection_rows'], prepared['audit_rows']
    actual = prepared['data']['y'][prepared['validation']]*100
    observed = prepared['data']['labelmask'][prepared['validation']]
    begin = time.monotonic()
    for epoch in range(start, plan['epochs']+1):
        supervised, kls = [], []
        if epoch:
            model.train(); batch = 8 if family == 'think' else 1
            for offset in range(0, len(prepared['train']), batch):
                sub = prepared['train'][offset:offset+batch]; index = sub if family == 'think' else int(sub[0])
                optimizer.zero_grad(set_to_none=True)
                a, b = [stochastic_forward(model, prepared, index, family, plan['dropout_probability']) for _ in range(2)]
                sup = .5*(rank_objective(a, prepared['target'][index], prepared['observed'][index])+
                          rank_objective(b, prepared['target'][index], prepared['observed'][index]))
                kl = symmetric_rank_kl(a, b, prepared['eligible'][index], plan['temperature'])
                loss = sup+(plan['kl_weight'] if arm == 'consistency' else 0.)*kl
                if not torch.isfinite(loss): raise ValueError('Nonfinite training objective')
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm): raise ValueError('Nonfinite gradient')
                optimizer.step(); ema.update(model)
                supervised.append(float(sup.detach())); kls.append(float(kl.detach()))
        raw_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        predictions = {'raw':trainer.predict(model, prepared, prepared['validation'], family)}
        ema_state = ema.averaged_state(model); model.load_state_dict(ema_state)
        predictions['ema'] = trainer.predict(model, prepared, prepared['validation'], family)
        model.load_state_dict(raw_state)
        if epoch == 0:
            with np.load(checkpoint.parent/'loss_predictions.npz', allow_pickle=False) as p:
                initial_error = float(np.max(np.abs(predictions['raw']-p['validation'])))
            assert initial_error <= 1e-7, (family, year, seed, initial_error)
            write(dest/'INITIAL_PARITY.json', dict(validation_max_abs_error=initial_error, source_checkpoint_sha256=sha(checkpoint)))
        row = dict(epoch=epoch, mean_training_loss=None if not supervised else float(np.mean(supervised)),
                   mean_training_kl=None if not kls else float(np.mean(kls)), ema_updates=ema.updates)
        for method, prediction in predictions.items():
            row[method+'_selection_loss'] = float(rank_objective(torch.from_numpy(prediction[selection]),
                torch.from_numpy(actual[selection]), torch.from_numpy(observed[selection])))
            row[method+'_selection'] = rank_statistics(prediction[selection], actual[selection], observed[selection])
            row[method+'_audit'] = rank_statistics(prediction[audit], actual[audit], observed[audit])
        history.append(row)
        atomic_checkpoint(dest/f'epoch_{epoch:02d}.pt', dict(epoch=epoch, state_dict=raw_state, ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), history=history, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
        np.savez_compressed(dest/f'validation_{epoch:02d}.npz', **predictions)
        write(dest/'HISTORY.json', history)
        write(RUN/'worker_progress'/f'{family}_{year}.json', dict(arm=arm, family=family, year=year, seed=seed,
            epoch=epoch, seconds=time.monotonic()-begin, timestamp_utc=datetime.now(timezone.utc).isoformat()))
    choices = {}
    for method in ('raw', 'ema'):
        best = min(history, key=lambda r:(r[method+'_selection_loss'], r['epoch']))
        epoch = best['epoch']; choices[method] = epoch
        saved = torch.load(dest/f'epoch_{epoch:02d}.pt', map_location='cpu', weights_only=False)
        state = saved['state_dict']
        if method == 'ema': state = dict(state, **saved['ema']['shadow'])
        model.load_state_dict(state)
        v = trainer.predict(model, prepared, prepared['validation'], family)
        t = trainer.predict(model, prepared, prepared['test'], family)
        assert np.isfinite(v).all() and np.isfinite(t).all()
        np.savez_compressed(dest/f'{method}_predictions.npz', validation=v, test=t)
    write(record_path, dict(arm=arm, family=family, year=year, seed=seed, selected=choices,
        seconds=time.monotonic()-begin, source_checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        source_sha256=sha(checkpoint), hashes={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file()}))
    print(f'Fit {arm}/{family}/{year}/{seed}: {choices}; {time.monotonic()-begin:.1f}s', flush=True)
    del model, optimizer, ema; torch.cuda.empty_cache()


def worker(family, year):
    plan = verify(False); device = configure_device('cuda'); prepared = trainer.prepare(year, device)
    write(RUN/f'FOLD_{family}_{year}.json', {name:prepared[name].tolist() for name in ('train', 'validation', 'selection', 'audit', 'test')})
    for seed in plan['seeds']:
        for arm in plan['arms']: fit(prepared, family, year, seed, arm, plan, device)
    return family, year


def train():
    plan = verify()
    if (RUN/'FITS_COMPLETE.json').exists(): raise FileExistsError('All fits already sealed')
    (RUN/'worker_progress').mkdir(exist_ok=True)
    jobs = [(f, y) for y in reversed(plan['years']) for f in reversed(plan['families'])]
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        futures = [pool.submit(worker, family, year) for family, year in jobs]
        for future in as_completed(futures): print('Worker complete', future.result(), flush=True)
    records = sorted((RUN/'fits').rglob('RECORD.json')); assert len(records) == plan['fits']
    hashes = {}
    for path in records:
        record = json.loads(path.read_text())
        for name, expected in record['hashes'].items():
            p = path.parent/name; assert sha(p) == expected; hashes[p.relative_to(RUN).as_posix()] = expected
        hashes[path.relative_to(RUN).as_posix()] = sha(path)
    verify()
    write(RUN/'FITS_COMPLETE.json', dict(fits=len(records), hashes=hashes, new_financial_accounts_absent=True,
        timestamp_utc=datetime.now(timezone.utc).isoformat()))


def evaluate():
    plan = verify(); lock = json.loads((RUN/'FITS_COMPLETE.json').read_text())
    assert lock['fits'] == plan['fits']
    for name, expected in lock['hashes'].items(): assert sha(RUN/name) == expected, name
    data = parent.prior.parent.parent.load_data()
    ids = np.concatenate([parent.prior.parent.parent.fold(data, y)[2] for y in plan['years']])
    original = parent.reserve.scores(parent.reserve.verify(), data, ids, 'original')
    neural = {'original':original}; validation = []
    for arm in plan['arms']:
        for method in ('raw', 'ema'):
            families = []
            for family in plan['families']:
                seeds = []
                for seed in plan['seeds']:
                    annual = []
                    for year in plan['years']:
                        folder = RUN/'fits'/arm/family/str(year)/str(seed)
                        with np.load(folder/f'{method}_predictions.npz', allow_pickle=False) as p: annual.append(p['test'].copy())
                        record = json.loads((folder/'RECORD.json').read_text()); epoch = record['selected'][method]
                        h = json.loads((folder/'HISTORY.json').read_text())[epoch]
                        validation.append(dict(arm=arm, method=method, family=family, seed=seed, year=year,
                            epoch=epoch, **{'selection_'+k:v for k,v in h[method+'_selection'].items()},
                            **{'audit_'+k:v for k,v in h[method+'_audit'].items()}))
                    seeds.append(percentile_predictions(np.concatenate(annual), data['mask'][ids]))
                families.append(np.asarray(seeds))
            neural[arm+'_'+method] = np.asarray(families)
    _, rank, _ = parent.prior.priors(plan, data, ids, 26)
    market = market_inputs(data); combined = append_qqq(data, etf_data(data, 'QQQ'))
    eligible = np.column_stack((data['mask'][ids], np.ones(len(ids), bool)))
    dates = data['panel'].dates[int(data['entry'][ids[0]]):].strftime('%Y-%m-%d').to_numpy(dtype=str)
    samples = [(g, [plan['seeds'].index(s) for s in members]) for g, members in plan['seed_groups'].items()]
    samples += [(f'seed_{s}', [j]) for j, s in enumerate(plan['seeds'])]
    source = pd.read_csv(parent.RUN/'METRICS.csv'); source_paths = np.load(parent.RUN/'PATHS.npz', allow_pickle=False)
    np.testing.assert_array_equal(dates, source_paths['dates'])
    (RUN/'ledgers').mkdir(exist_ok=False)
    rows, periods, checks = [], [], []; paths, targets = {}, {}
    for mixture in plan['mixtures']:
        for method in plan['methods']:
            cache = {}
            def target(index):
                key = tuple(index)
                if key not in cache:
                    values = neural[method]
                    score = .5*rank
                    if mixture == 'both': score = score+.25*values[0, index].mean(0)+.25*values[1, index].mean(0)
                    else: score = score+.5*values[int(mixture == 'dynamic'), index].mean(0)
                    cache[key] = portfolio_targets(data, ids, score, market, {})
                return cache[key]
            for group, members in samples:
                chunks = [members] if len(members) == 1 else [members[k:k+5] for k in range(0, len(members), 5)]
                desired = average_targets([target(chunk) for chunk in chunks])
                for regime in plan['cost_regimes']:
                    cost, borrow = regime['cost'], regime['borrow']
                    key = f'{mixture}__{method}__{group}__c{cost}__b{int(borrow*100):02d}'
                    measured, path, ledger, effective, _ = simulate_banded(combined, ids, desired, eligible, 0., borrow, cost)
                    error = float(np.max(np.abs(path-value_replay(combined, ids, effective, borrow, cost))))
                    assert error < 1e-12
                    parity = None
                    if mixture == 'both' and method == 'original' and group in plan['groups']:
                        match = source[(source.model == 'prior26_issuer') & (source.policy == 'original20') &
                            (source.group == group) & (source.cost == cost) & (source.borrow == borrow)].iloc[0]
                        parity = float(np.max(np.abs(path-source_paths[match.id]))); assert parity < 1e-12
                    rows.append(dict(id=key, mixture=mixture, method=method, group=group, cost=cost, borrow=borrow,
                        sample_type='single_seed' if group.startswith('seed_') else 'ensemble', independent_error=error, **measured))
                    paths[key] = path; targets[key] = effective
                    checks.append(dict(id=key, independent_error=error, baseline_parity_error=parity))
                    ledger.to_csv(RUN/'ledgers'/(key+'.csv.gz'), index=False, compression='gzip')
                    years = pd.to_datetime(dates).year.to_numpy()
                    for year in plan['years']:
                        periods.append(dict(id=key, mixture=mixture, method=method, group=group, cost=cost, borrow=borrow,
                                            year=year, **performance(path[years == year])))
                print('Accounts', mixture, method, group, len(rows), '/', plan['accounts'], flush=True)
            pd.DataFrame(rows).to_csv(RUN/'PROGRESS_METRICS.csv', index=False)
    source_paths.close(); assert len(rows) == plan['accounts']
    pd.DataFrame(rows).to_csv(RUN/'METRICS.csv', index=False)
    pd.DataFrame(periods).to_csv(RUN/'PERIOD_METRICS.csv', index=False)
    pd.DataFrame(checks).to_csv(RUN/'ACCOUNT_CHECKS.csv', index=False)
    pd.DataFrame(validation).to_csv(RUN/'VALIDATION.csv', index=False)
    np.savez_compressed(RUN/'PATHS.npz', dates=dates, **paths)
    np.savez_compressed(RUN/'TARGETS.npz', ids=ids, **targets)
    verify(); write(RUN/'ACCOUNTS_COMPLETE.json', dict(accounts=len(rows), max_independent_error=max(r['independent_error'] for r in rows),
        output_sha256={p.relative_to(RUN).as_posix():sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}))


def summarize():
    plan = verify(); lock = json.loads((RUN/'ACCOUNTS_COMPLETE.json').read_text())
    for name, expected in lock['output_sha256'].items(): assert sha(RUN/name) == expected, name
    frame = pd.read_csv(RUN/'METRICS.csv'); rows = []
    for mixture in plan['mixtures']:
        ensembles = frame[(frame.mixture == mixture) & (frame.sample_type == 'ensemble')]
        original = ensembles[(ensembles.method == 'original') & (ensembles.cost == 1)].set_index('group').loc[plan['groups']]
        original_stress = ensembles[(ensembles.method == 'original') & (ensembles.cost == 2)].set_index('group').loc[plan['groups']]
        for method in plan['methods']:
            b = ensembles[(ensembles.method == method) & (ensembles.cost == 1)].set_index('group').loc[plan['groups']]
            s = ensembles[(ensembles.method == method) & (ensembles.cost == 2)].set_index('group').loc[plan['groups']]
            delta = b.sharpe-original.sharpe; stress_delta = s.sharpe-original_stress.sharpe
            matched = method.replace('consistency', 'dropout_control')
            control = ensembles[(ensembles.method == matched) & (ensembles.cost == 1)].set_index('group').loc[plan['groups']]
            paired = b.sharpe-control.sharpe
            singleton = frame[(frame.mixture == mixture) & (frame.method == method) & (frame.cost == 1) & (frame.sample_type == 'single_seed')]
            sr = b.sharpe.to_numpy()
            row = dict(mixture=mixture, method=method, minimum_sr=float(sr.min()), mean_sr=float(sr.mean()),
                disjoint_spread=float(np.ptp(sr[:4])), min_delta=float(delta.min()), groups_improved=int((delta >= .02).sum()),
                minimum_stress_delta=float(stress_delta.min()), matched_mean_delta=float(paired.mean()), matched_min_delta=float(paired.min()),
                singleton_min=float(singleton.sharpe.min()), singleton_median=float(singleton.sharpe.median()), singleton_std=float(singleton.sharpe.std()),
                goal_passed=bool(mixture == 'both' and method != 'original' and delta.min() >= .02 and np.ptp(sr[:4]) <= .25 and stress_delta.min() >= 0 and
                                 (not method.startswith('consistency') or paired.mean() > 0)), all_above_1_5=bool(sr.min() > 1.5))
            row.update({g:float(b.loc[g].sharpe) for g in plan['groups']}); rows.append(row)
    summary = pd.DataFrame(rows); summary.to_csv(RUN/'CONSISTENCY.csv', index=False)
    passed = summary[summary.goal_passed].sort_values(['minimum_sr', 'mean_sr', 'method'], ascending=[False, False, True])
    write(RUN/'SELECTION.json', dict(passes=len(passed), selected=None if passed.empty else passed.iloc[0].to_dict(),
        scope='Three-epoch fine-tuning screen only; no full retraining or architecture results', untouched_confirmation=False))
    publish(plan, frame, summary, lock)
    print(summary.to_string(index=False), flush=True)


def publish(plan, frame, summary, lock):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    DOC.mkdir(parents=True, exist_ok=True)
    primary = summary[summary.mixture == 'both']; fig, ax = plt.subplots(figsize=(11, 5))
    for row in primary.itertuples():
        ax.plot(range(7), [getattr(row, g) for g in plan['groups']], marker='o', label=row.method)
    ax.set_xticks(range(7), ['Original5','Second5','Third5','Fourth5','Combined10','Combined15','Combined20'])
    ax.axhline(1.5, color='gray', ls='--'); ax.set_ylabel('Net Sharpe, 3% risk-free hurdle')
    ax.set_title('Matched consistency / EMA continuation: spent 2022–2025 development')
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(DOC/'ensemble_sharpe.png', dpi=160); plt.close(fig)
    for name in ('PROTOCOL.json','FREEZE.json','ENVIRONMENT.json','METRICS.csv','PERIOD_METRICS.csv','VALIDATION.csv','CONSISTENCY.csv','ACCOUNT_CHECKS.csv','SELECTION.json'):
        shutil.copyfile(RUN/name, DOC/name)
    lines = ['# Consistency regularization and EMA continuation — October 31, 2026', '',
        f"**Completed {plan['fits']} three-epoch continuation fits and {lock['accounts']} account backtests. "
        f"{int(primary.goal_passed.sum())} candidates pass the primary improvement gate.**", '',
        '![Fixed seed ensembles](ensemble_sharpe.png)', '',
        'This is a short fine-tuning screen of two training approaches, not a full from-scratch retraining or a test of the proposed architecture, multi-horizon or portfolio-loss changes. '
        'Every existing twenty-seed THINK/dynamic pair is retained. Each matched pair starts from the same saved validation-loss-selected checkpoint, '
        'resets its optimizer at one-quarter of the original learning rate, and trains three more epochs on the same training fold. '
        'All four annual folds and all 1,003 sessions in 2022–2025 are previously inspected development data; no untouched confirmation is claimed.', '',
        'THINK has no native dropout. Both arms therefore use 10% input dropout and two supervised passes; dynamic retains its internal dropout too. '
        'The consistency arm additionally penalizes symmetric KL between eligible cross-sectional softmax(score/0.5) distributions. '
        'Both arms use the same seeded dropout draws. EMA decay 0.99 tracks parameters after each update; fixed buffers are copied. '
        'This is an R-Drop-inspired stock-ranking adaptation and EMA, not an exact reproduction of either cited paper.', '',
        'For raw and EMA separately, first-half prior-year supervised validation loss selects among epochs 0–3 with earliest ties. '
        'Epoch zero exactly preserves the source. The second validation half is diagnostic only. Every fit and checkpoint locks before new financial account scoring. '
        'The unchanged primary portfolio uses 50% training prior, 25% THINK ranks and 25% dynamic ranks, five issuers per cohort, '
        'the existing volatility/QQQ hedge rules and equal averaging of cohort targets. Single-family controls retain the same 50% prior, replacing the other 50% with that family.', '',
        '## Both-family primary comparison', '',
        '| Method | Minimum ensemble SR | Four-cohort spread | Combined20 SR | Worst SR change | Worst stress SR change | Pass |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in primary.itertuples():
        lines.append(f'| {r.method} | {r.minimum_sr:.3f} | {r.disjoint_spread:.3f} | {r.expanded_twenty:.3f} | {r.min_delta:+.3f} | {r.minimum_stress_delta:+.3f} | {r.goal_passed} |')
    lines += ['', '## All family and singleton diagnostics', '',
        '| Mixture | Method | Min ensemble SR | Singleton min / median SR | Singleton SR SD | Mean SR change versus matched dropout control |',
        '|---|---|---:|---|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.mixture} | {r.method} | {r.minimum_sr:.3f} | {r.singleton_min:.3f} / {r.singleton_median:.3f} | {r.singleton_std:.3f} | {r.matched_mean_delta:+.3f} |')
    records = [json.loads(p.read_text()) for p in (RUN/'fits').rglob('RECORD.json')]
    selected = pd.read_csv(RUN/'VALIDATION.csv'); changes = int((selected.epoch > 0).sum())
    lines += ['', f'{changes}/{len(selected)} raw/EMA fit selections choose an epoch after zero; zero selections remain in every account. '
        'Selecting zero is a failure to validate additional training, not evidence of a newly learned improvement.', '',
        'The primary gate requires >=+0.02 net Sharpe versus the original in every seven-group account, four-cohort spread <=0.25, '
        'no group worse at double transaction fees and 6% borrowing, and positive mean matched-control improvement for consistency. '
        'SR>1.5 across all groups is reported separately. Individual seeds and overlapping pooled ensembles share market dates and are not independent evidence.', '',
        f"All {lock['accounts']} accounts pass independent dollar-position accounting within {lock['max_independent_error']:.3g} daily-return error. "
        'Every selected source checkpoint is hash-pinned, initial validation predictions reproduce within 1e-7, and all fourteen current-strategy reference paths reproduce within 1e-12 daily returns. '
        'Raw/EMA state, optimizer, CPU/CUDA RNG, every epoch validation path, selected forecasts, compressed ledgers, daily account paths and effective targets are retained in the dated run.', '',
        'Tests cover masking/symmetry/gradients, EMA arithmetic and buffer handling, and exact optimizer/RNG/EMA continuation on a stochastic network. '
        'That synthetic resume check does not certify bitwise regeneration of every production fit. All source files and input/checkpoint hashes are checked before and after fitting.', '',
        '## Interpretation and next steps', '',
        'Passing would justify a separately frozen longer training experiment; this short continuation cannot rule out the approaches under other schedules. '
        'Failure does not establish that an architecture or new information source cannot help. Architecture/prior decomposition, auxiliary horizons and execution-aware objectives remain separate untested proposals in this study. '
        'No promising seed is excluded, no failed method is hidden, and no production strategy is replaced automatically.', '',
        '[All account results](METRICS.csv) · [Annual results](PERIOD_METRICS.csv) · [Validation and chosen epochs](VALIDATION.csv) · [Protocol](PROTOCOL.json)', '',
        '[R-Drop](https://arxiv.org/abs/2106.14448) motivates consistency regularization. '
        '[Weight averaging research](https://proceedings.mlr.press/r16/izmailov18a.html) motivates averaging, but this EMA experiment uses a different optimizer and schedule.', '',
        '```powershell', r'.\.venv\Scripts\python.exe -m hypershift test test_ndx_consistency_training -v',
        r'.\.venv\Scripts\python.exe -m hypershift run run_ndx_consistency_retraining --stage preflight',
        r'.\.venv\Scripts\python.exe -m hypershift run run_ndx_consistency_retraining --stage freeze',
        r'.\.venv\Scripts\python.exe -m hypershift run run_ndx_consistency_retraining --stage train',
        r'.\.venv\Scripts\python.exe -m hypershift run run_ndx_consistency_retraining --stage evaluate',
        r'.\.venv\Scripts\python.exe -m hypershift run run_ndx_consistency_retraining --stage summarize', '```', '',
        'Completed artifacts are immutable. Training can resume from the same frozen epoch states; a new scientific setup needs a separate run.', '']
    (DOC/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')
    write(DOC/'MANIFEST.json', dict(run=RUN.relative_to(ROOT).as_posix(), fits=plan['fits'], accounts=lock['accounts'],
        protocol_sha256=sha(RUN/'PROTOCOL.json'), fit_lock_sha256=sha(RUN/'FITS_COMPLETE.json'),
        account_lock_sha256=sha(RUN/'ACCOUNTS_COMPLETE.json'), output_sha256={p.name:sha(p) for p in sorted(DOC.iterdir()) if p.name != 'MANIFEST.json'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['preflight','freeze','train','evaluate','summarize'])
    args = parser.parse_args()
    dict(preflight=preflight, freeze=freeze, train=train, evaluate=evaluate, summarize=summarize)[args.stage]()
