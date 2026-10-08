"""Predeclared, chronological multi-universe ablations. Never tunes on test metrics.

python run_robust_research.py --out runs/robust_2022_2025
Every output directory is immutable: use a fresh directory for a new experiment.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from robust_models import make_model, EXPERIMENTS, robust_objective as objective
from robust_engine import backtest, METHODS
VARIANTS = list(EXPERIMENTS)
from research_engine import (load_panel, prepare_panel, fold_indices, training_groups,
                             ranking_metrics, performance, ranks)


def clean(value):
    if isinstance(value, dict):
        return {k:clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, data):
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False), encoding='utf-8')


def robust_score(values):
    a = np.array([v for v in values if v is not None and np.isfinite(v)])
    return float(np.median(a)-.25*np.std(a)) if len(a) else -1e6


def scale_inputs(data, train):
    a = data['x'][train].transpose(0, 2, 1, 3)[data['mask'][train]]
    mean = a.mean((0, 1), keepdims=True)
    std = np.maximum(a.std((0, 1), keepdims=True), 1e-5)
    x = np.clip((data['x']-mean)/std, -8, 8)*.2
    return x.astype('float32'), mean, std


def predict(model, x, mask, ids, batch):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[sub]), torch.from_numpy(mask[sub].astype('float32')))[0].numpy()
                               for sub in np.array_split(ids, max(1, int(np.ceil(len(ids)/batch))))])


def fit_model(data, x, train, val, kind, seed, groups, args, folder):
    torch.manual_seed(seed)
    model = make_model(kind, len(data['panel'].tickers), groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001)
    y = data['y']*100
    risk = data['risk']*100
    best, state, stale, history, chosen = np.inf, None, 0, [], 0
    for epoch in range(1, args.epochs+1):
        model.train()
        total, count = 0., 0
        # No temporal shuffling, including within a training epoch.
        for start in range(0, len(train), args.batch):
            sub = train[start:start+args.batch]
            optimizer.zero_grad(set_to_none=True)
            score, aux = model(torch.from_numpy(x[sub]), torch.from_numpy(data['mask'][sub].astype('float32')))
            loss = objective(score, torch.from_numpy(y[sub]),
                             torch.from_numpy(data['labelmask'][sub].astype('float32')), kind,
                             aux, torch.from_numpy(risk[sub]))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {kind}/{seed}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(sub)
            count += len(sub)
        vp = predict(model, x, data['mask'], val, args.batch)
        m = data['labelmask'][val]
        vmse = float(((vp-y[val])**2)[m].mean())
        history.append(dict(epoch=epoch, train_loss=total/count, validation_mse_pp2=vmse))
        if vmse < best-1e-7:
            best, state, stale, chosen = vmse, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(state)
    torch.save(dict(state_dict=state, kind=kind, seed=seed, hidden=16, lookback=16,
                    groups=[sorted(g) for g in groups], features=5, epoch=chosen), folder/f'{kind}_{seed}.pt')
    write_json(folder/f'{kind}_{seed}_history.json', history)
    return model, chosen, len(history)


def fit_seed_job(data, x, train, val, test, kind, seed, groups, args, folder):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, chosen, completed = fit_model(data, x, train, val, kind, seed, groups, args, folder)
    return (predict(model, x, data['mask'], val, args.batch),
            predict(model, x, data['mask'], test, args.batch), chosen, completed,
            sum(p.numel() for p in model.parameters()))


def ensemble_predictions(predictions, eligible=None):
    # Equal-weight rank ensemble fixed in advance, no seed selection.
    if eligible is not None:
        ranked = []
        for pred in predictions:
            value = np.zeros_like(pred, dtype=float)
            for i, mask in enumerate(eligible):
                value[i, mask] = ranks(pred[i, mask])/max(mask.sum(), 1)
            ranked.append(value)
        return np.mean(ranked, axis=0)
    return np.mean([[ranks(row)/len(row) for row in pred] for pred in predictions], axis=0)


def choose_policy(data, val, seed_preds, ks):
    rows = []
    for method in METHODS:
        for k in ks:
            values = []
            for pred in seed_preds:
                for sub in np.array_split(np.arange(len(val)), 2):
                    metrics, _, _ = backtest(data, val[sub], pred[sub], k, method)
                    values.append(metrics['net_sharpe'])
            rows.append(dict(method=method, k=k, score=robust_score(values), component_sharpes=values))
    # Stable ties choose first declared configuration.
    selected = max(rows, key=lambda r:r['score'])
    return selected, rows


def bootstrap_difference(a, b, repeats=1000, block=20):
    """Paired circular moving-block bootstrap of DAILY Sharpe differences."""
    if len(a) != len(b):
        raise ValueError('Paired paths must align')
    rng, out = np.random.default_rng(20261004), []
    n = len(a)
    for _ in range(repeats):
        starts = rng.integers(0, n, int(np.ceil(n/block)))
        ids = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
        sa, sb = performance(a[ids])['sharpe'], performance(b[ids])['sharpe']
        if sa is not None and sb is not None:
            out.append(sa-sb)
    return dict(observed=performance(a)['sharpe']-performance(b)['sharpe'],
                low=float(np.quantile(out, .025)), high=float(np.quantile(out, .975)),
                probability_positive=float(np.mean(np.asarray(out)>0)),
                block_sessions=block, bootstrap_replicates=repeats,
                warning='Exploratory paired interval, not multiplicity-adjusted or proof of an edge.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/robust_2022_2025'))
    parser.add_argument('--datasets', nargs='+', default=['NYSE_recent', 'NASDAQ_recent'])
    parser.add_argument('--test-years', type=int, nargs='+', default=[2022, 2023, 2024, 2025])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--horizon', type=int, choices=[1, 5], default=5)
    parser.add_argument('--reuse-fits-from', type=Path, default=None,
                        help='Reuse completed fits from an interrupted run only after input/checkpoint/prediction checks.')
    parser.add_argument('--variants', nargs='+', default=VARIANTS)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 42])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.001)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a fresh output directory; no results are overwritten.')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = dict(arguments={k:str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    experiments=EXPERIMENTS, portfolio_methods=METHODS,
                    huber='2*Huber(delta=1 percentage point); validation MSE retained across losses to isolate training objective',
                    stop_rule='One predeclared batch; do not add variants after viewing test outcomes. 2024-25 already observed; all results exploratory.',
                    universe='Fixed source cohorts; see per-dataset metadata for dates and availability limitations.',
                    input_features=['daily return', 'close/MA5-1', 'close/MA20-1', '20-day return std', 'log dollar volume/ADV20'],
                    lookback=16, horizon=args.horizon, rebalance_sessions=args.horizon,
                    execution=f'Signal after close t; shares sized with t NAV/prices; fill at close t+1 plus costs; exit/rebalance t+{args.horizon+1}.',
                    liquidity='Signal-time 20-session dollar ADV >= $1m; max trade 1% lagged ADV; $1m initial NAV.',
                    cost='Per-side 2bp commission + 5bp spread/slippage + 3bp sqrt(participation/1%); 3% annual short borrow.',
                    selection='Early stopping: validation MSE. Policy: median validation half-year/seed net Sharpe minus .25 std; equal/confidence/inverse-vol/target-retention buffer; fixed K=5,10,20. No short selection without borrow feed.',
                    model_selection='Per-year architecture selected using validation-policy scores for that year and earlier folds only. No later validation can choose an earlier test architecture.',
                    evaluation='Three-seed equal-weight rank ensembles primary; individual seeds secondary. Gross matched-holdings P&L excludes fees/borrow.',
                    risk_free='Fixed 3% annual hurdle, plus RF=0 Sharpe reported; cash earns zero.',
                    short_warning='Long-short scenarios hypothetical: no historical locates or time-varying borrow fees.',
                    final_warning='Public data availability/survivorship bias remains. Previously observed windows are not pristine confirmation data.',
                    seed_policy='All declared seeds reported; no best-seed picking.',
                    sources=['https://github.com/SJTU-DMTai/StockMixer', 'https://github.com/SJTU-DMTai/MASTER',
                             'https://arxiv.org/abs/2609.25617', 'https://tylersnetwork.github.io/papers/icdm22-think.pdf'],
                    python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__,
                    code_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                                 ['run_robust_research.py', 'robust_models.py', 'robust_engine.py', 'research_engine.py', 'research_models.py', 'think_model.py']})
    protocol['experiment_plan_sha256'] = hashlib.sha256(Path('config/robust_experiment_plan.json').read_bytes()).hexdigest()
    (args.out/'EXPERIMENT_PLAN.json').write_bytes(Path('config/robust_experiment_plan.json').read_bytes())
    write_json(args.out/'PROTOCOL.json', protocol)
    if any(name.endswith('_recent') for name in args.datasets):
        extra_sources = ['config/recent_identity_exclusions.json', 'data/recent_cohorts/manifest.json',
                         'data/recent_cohorts/selection.json', 'fetch_recent_cohorts.py']
        write_json(args.out/'DATA_INPUT_LOCK.json',
                   {source:hashlib.sha256(Path(source).read_bytes()).hexdigest() for source in extra_sources})
        for source in extra_sources:
            (args.out/('input_'+Path(source).name)).write_bytes(Path(source).read_bytes())
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir()
    for source in protocol['code_hashes']:
        (snapshot/Path(source).name).write_bytes(Path(source).read_bytes())
    all_metrics, policies, training_records, stored, panels = [], [], [], {}, {}
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    reused = []
    if args.reuse_fits_from:
        previous_protocol = json.loads((args.reuse_fits_from/'PROTOCOL.json').read_text())
        for source in ['research_models.py', 'think_model.py', 'research_engine.py']:
            assert previous_protocol['code_hashes'][source] == protocol['code_hashes'][source]
        for source, expected in json.loads((args.reuse_fits_from/'DATA_INPUT_LOCK.json').read_text()).items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected
        for key in ['epochs', 'patience', 'batch', 'lr', 'seeds']:
            assert previous_protocol['arguments'][key] == protocol['arguments'][key]
    started = time.time()
    for name in args.datasets:
        data = prepare_panel(load_panel(name), horizon=args.horizon)
        panels[name] = data
        years = args.test_years or ([2024, 2025] if name.endswith('_recent') else
                                   [2022, 2023, 2024, 2025, 2026] if name == 'modern12' else [2015, 2016, 2017])
        ks = [3, 5, 8] if name == 'modern12' else [5, 10, 20]
        datahash = hashlib.sha256(data['panel'].close.tobytes()+data['panel'].volume.tobytes()).hexdigest()
        write_json(args.out/f'{name}_data.json', dict(tickers=data['panel'].tickers, notes=data['panel'].notes,
                   sha256_close_and_dollar_volume=datahash, dates=[str(data['panel'].dates[0]), str(data['panel'].dates[-1])],
                   eligible_min=int(data['mask'].sum(1).min()), eligible_max=int(data['mask'].sum(1).max()),
                   missing_future_labels=int((data['mask'] & ~data['labelmask']).sum())))
        for year in years:
            folder = args.out/f'{name}_{year}'
            folder.mkdir()
            train, val, test = fold_indices(data, year)
            x, mean, std = scale_inputs(data, train)
            np.savez_compressed(folder/'preprocessing.npz', mean=mean, std=std, train=train, validation=val, test=test)
            groups = training_groups(data, train)
            fold = dict(dataset=name, year=year, horizon=args.horizon, universe=data['panel'].tickers, seeds=args.seeds,
                        data_hash=datahash, features=protocol['input_features'], hidden=16, lr=args.lr,
                        train_signal=[str(data['panel'].dates[data['origin'][train[i]]].date()) for i in [0, -1]],
                        validation_signal=[str(data['panel'].dates[data['origin'][val[i]]].date()) for i in [0, -1]],
                        test_signal=[str(data['panel'].dates[data['origin'][test[i]]].date()) for i in [0, -1]],
                        train_labels_end=str(data['panel'].dates[data['exit'][train[-1]]].date()),
                        validation_labels_end=str(data['panel'].dates[data['exit'][val[-1]]].date()))
            write_json(folder/'fold.json', fold)
            for kind in args.variants+['ridge', 'momentum126', 'reversal5']:
                vals, tests = [], []
                seeds = args.seeds if kind in args.variants else [0]
                oldfolder = args.reuse_fits_from/f'{name}_{year}' if args.reuse_fits_from else None
                reuse = kind in args.variants and oldfolder is not None and (oldfolder/f'{kind}_predictions.npz').exists()
                jobs = None
                if kind in args.variants and not reuse and pool is not None:
                    jobs = [pool.submit(fit_seed_job, data, x, train, val, test, kind, seed, groups, args, folder)
                            for seed in seeds]
                for seed in seeds:
                    if kind in args.variants:
                        if reuse:
                            assert json.loads((oldfolder/'fold.json').read_text()) == fold
                            oldscale = np.load(oldfolder/'preprocessing.npz')
                            for key, value in [('mean',mean),('std',std),('train',train),('validation',val),('test',test)]:
                                np.testing.assert_array_equal(oldscale[key], value)
                            checkpoint = torch.load(oldfolder/f'{kind}_{seed}.pt', weights_only=False)
                            assert checkpoint['groups'] == [sorted(g) for g in groups]
                            model = make_model(kind, len(data['panel'].tickers), groups)
                            model.load_state_dict(checkpoint['state_dict'])
                            vp, tp = predict(model, x, data['mask'], val, args.batch), predict(model, x, data['mask'], test, args.batch)
                            previous = np.load(oldfolder/f'{kind}_predictions.npz')
                            np.testing.assert_array_equal(vp, previous['validation'][seeds.index(seed)])
                            np.testing.assert_array_equal(tp, previous['test'][seeds.index(seed)])
                            history = json.loads((oldfolder/f'{kind}_{seed}_history.json').read_text())
                            chosen, completed = checkpoint['epoch'], len(history)
                            parameters = sum(p.numel() for p in model.parameters())
                            for suffix in ['.pt', '_history.json']:
                                (folder/f'{kind}_{seed}{suffix}').write_bytes((oldfolder/f'{kind}_{seed}{suffix}').read_bytes())
                            reused.append(dict(dataset=name, year=year, model=kind, seed=seed,
                                               source=str(oldfolder), verification='Exact predictions, scalers, indices, fold and model hashes matched.'))
                        elif jobs is not None:
                            vp, tp, chosen, completed, parameters = jobs[seeds.index(seed)].result()
                        else:
                            vp, tp, chosen, completed, parameters = fit_seed_job(data, x, train, val, test, kind, seed, groups, args, folder)
                        training_records.append(dict(dataset=name, year=year, model=kind, seed=seed,
                                                     chosen_epoch=chosen, epochs_run=completed,
                                                     parameters=parameters))
                    elif kind == 'ridge':
                        design = x.transpose(0, 2, 1, 3).reshape(len(x), len(data['panel'].tickers), -1)
                        design = np.concatenate([design, np.ones((*design.shape[:2], 1))], axis=-1)
                        a, y = design[train][data['labelmask'][train]], (data['y'][train]*100)[data['labelmask'][train]]
                        penalty = np.eye(a.shape[-1])*10
                        penalty[-1, -1] = 0
                        coef = np.linalg.solve(a.T@a+penalty, a.T@y)
                        np.save(folder/'ridge_coef.npy', coef)
                        vp, tp = design[val]@coef, design[test]@coef
                    else:
                        key = 'momentum' if kind == 'momentum126' else 'reversal'
                        vp, tp = data[key][val], data[key][test]
                    vals.append(vp)
                    tests.append(tp)
                selected, candidates = choose_policy(data, val, vals, ks)
                record = dict(dataset=name, year=year, model=kind, selected=selected, candidates=candidates)
                policies.append(record)
                write_json(folder/f'{kind}_policy.json', record)
                # Predictions retained but no test metrics are computed or inspected in this phase.
                np.savez_compressed(folder/f'{kind}_predictions.npz', validation=np.array(vals), test=np.array(tests),
                                    test_ids=test, validation_ids=val, truth=data['y'][test],
                                    eligible=data['mask'][test], labelmask=data['labelmask'][test])
                stored[(name, year, kind)] = (test, tests, selected, ks)
                print(f'FIT {name} {year} {kind}: {len(seeds)} seed(s), elapsed {(time.time()-started)/60:.1f}m', flush=True)
    # Lock all choices BEFORE any test statistics are calculated.
    if pool is not None:
        pool.shutdown()
    write_json(args.out/'FIT_REUSE_AUDIT.json', reused)
    architecture_selection = {}
    for name in args.datasets:
        by_year = {}
        for year in sorted({r['year'] for r in policies if r['dataset']==name}):
            scores = {kind:robust_score([r['selected']['score'] for r in policies
                                        if r['dataset']==name and r['model']==kind and r['year']<=year])
                      for kind in args.variants+['ridge', 'momentum126', 'reversal5']}
            by_year[str(year)] = dict(winner=max(scores, key=scores.get), validation_scores=scores)
        architecture_selection[name] = dict(**by_year[str(max(map(int, by_year)))], by_year=by_year)
    write_json(args.out/'SELECTION_LOCK.json', architecture_selection)
    pd.DataFrame(training_records).to_csv(args.out/'training_records.csv', index=False)
    write_json(args.out/'validation_search.json', policies)
    print('Selection locked. Beginning one-pass held-out evaluation.', flush=True)
    paths = {}
    for (name, year, kind), (test, seeds_pred, selected, ks) in stored.items():
        data, folder = panels[name], args.out/f'{name}_{year}'
        ensemble = ensemble_predictions(seeds_pred, data['mask'][test]) if len(seeds_pred)>1 else seeds_pred[0]
        rank, ranking = ranking_metrics(ensemble, data['y'][test], data['mask'][test], data['labelmask'][test], ks)
        ranking.insert(0, 'signal_date', data['panel'].dates[data['origin'][test]].astype(str))
        ranking.to_csv(folder/f'{kind}_ranking.csv', index=False)
        for k in ks:
            for method in METHODS+['long_short']:
                metric, daily, _ = backtest(data, test, ensemble, k, method)
                daily.to_csv(folder/f'{kind}_{method}_k{k}_daily.csv', index=False)
                chosen = k == selected['k'] and method == selected['method']
                all_metrics.append(dict(dataset=name, year=year, model=kind, seed='ensemble', k=k, method=method,
                                        selected_policy=chosen, cost_multiplier=1., **metric, **rank))
                if chosen:
                    daily.to_csv(folder/f'{kind}_daily.csv', index=False)
                    paths[(name, year, kind)] = daily
                    for mult in [0., 2., 4.]:
                        stress, _, _ = backtest(data, test, ensemble, k, method, cost_multiplier=mult)
                        all_metrics.append(dict(dataset=name, year=year, model=kind, seed='ensemble', k=k, method=method,
                                                selected_policy=True, cost_multiplier=mult, **stress, **rank))
        for seed, pred in zip(args.seeds if len(seeds_pred)>1 else [0], seeds_pred):
            metric, _, _ = backtest(data, test, pred, selected['k'], selected['method'])
            srank, _ = ranking_metrics(pred, data['y'][test], data['mask'][test], data['labelmask'][test], ks)
            all_metrics.append(dict(dataset=name, year=year, model=kind, seed=str(seed), k=selected['k'], method=selected['method'],
                                    selected_policy=True, cost_multiplier=1., **metric, **srank))
    # Benchmarks on identical dates, each fold starts from cash, just like the models.
    for name in args.datasets:
        data = panels[name]
        years = sorted({key[1] for key in stored if key[0]==name})
        for year in years:
            _, _, test = fold_indices(data, year)
            for method in ['buy_hold', 'equal_universe']:
                metric, daily, _ = backtest(data, test, np.zeros_like(data['y'][test]), 0, method)
                paths[(name, year, method)] = daily
                daily.to_csv(args.out/f'{name}_{year}'/f'{method}_daily.csv', index=False)
                all_metrics.append(dict(dataset=name, year=year, model=method, seed='ensemble', k=0, method=method,
                                        selected_policy=True, cost_multiplier=1., **metric))
            benchmark = data['panel'].benchmark
            start, end = data['entry'][test[0]], data['exit'][test[-1]]
            br = benchmark[start+1:end+1]/benchmark[start:end]-1
            if not np.isfinite(br).all():
                raise ValueError('Missing benchmark prices')
            bnet = br.copy()
            bnet[0] = (1+bnet[0])*(1-.0007)-1
            bnet[-1] = (1+bnet[-1])*(1-.0007)-1
            daily = pd.DataFrame(dict(date=data['panel'].dates[start+1:end+1].astype(str), gross=br, net=bnet))
            paths[(name, year, 'market_proxy')] = daily
            daily.to_csv(args.out/f'{name}_{year}'/'market_proxy_daily.csv', index=False)
            metric = {f'gross_{k}':v for k,v in performance(br).items()}
            metric.update({f'net_{k}':v for k,v in performance(bnet).items()})
            all_metrics.append(dict(dataset=name, year=year, model='market_proxy', seed='ensemble', k=0, method='buy_hold',
                                    selected_policy=True, cost_multiplier=1., **metric, turnover_mean=2/len(test),
                                    turnover_annual=2*252/len(br), profitable_periods=float(np.mean(np.prod((1+bnet).reshape(-1,args.horizon),axis=1)>1))))
    frame = pd.DataFrame(all_metrics)
    frame.to_csv(args.out/'all_metrics.csv', index=False)
    intervals = {}
    for name, selection in architecture_selection.items():
        year = max(key[1] for key in paths if key[0]==name)
        winner = selection['winner']
        intervals[name] = {base:bootstrap_difference(paths[(name, year, winner)].net.to_numpy(), paths[(name, year, base)].net.to_numpy())
                           for base in ['think', 'buy_hold', 'market_proxy']}
    write_json(args.out/'paired_bootstrap_final.json', intervals)
    write_json(args.out/'completion.json', dict(elapsed_seconds=time.time()-started, neural_fits=len(training_records),
                                               metric_rows=len(frame), status='complete; conditional research evidence, not production validation'))
    print(f'COMPLETE: {len(training_records)} neural fits, {len(frame)} result rows in {args.out}', flush=True)


if __name__ == '__main__':
    main()
