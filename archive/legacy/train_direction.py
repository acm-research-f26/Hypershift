"""Run a reproducible up/down experiment on the existing historical price CSV.

python train_direction.py --csv data/prices.csv --horizon 21 --out runs/direction
Read DIRECTION_TUTORIAL.md for the interpretation and limitations.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from stock_gnn import load_prices, prepare, feature_names, window_features
from hypergraph import DirectionNet, correlation_hyperedges, hypergraph_geometry, four_point_delta
from direction_metrics import classification, select_threshold, ndcg, trading_backtest


def train(data, incidence, kind, seed, args):
    torch.manual_seed(seed)
    model = DirectionNet(data.x.shape[-1], args.hidden, kind, incidence, data.adjacency)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    x, y = torch.from_numpy(data.x), torch.from_numpy((data.y > 0).astype(np.float32))
    tr, val = data.splits['train'], data.splits['validation']
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, stale, history, state = float('inf'), 0, 0, [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for ids in torch.randperm(tr.stop, generator=generator).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[ids]), y[ids])
            if not torch.isfinite(loss):
                raise RuntimeError(f'{kind}: nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(ids)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x[val]), y[val]).item()
        if not np.isfinite(val_loss):
            raise RuntimeError(f'{kind}: nonfinite validation loss')
        history.append({'epoch': epoch, 'train_bce': total / tr.stop, 'validation_bce': val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.cat([model(batch).sigmoid() for batch in x.split(256)]).numpy()
    threshold = select_threshold(data.y[val] > 0, probabilities[val])
    return model, probabilities, threshold, history, best_epoch


def predict_direction(checkpoint, prices):
    """Restore a saved model and estimate direction after the latest available close."""
    tickers = checkpoint['tickers']
    if set(prices.columns) != set(tickers):
        raise ValueError('Prediction stock universe must match the training universe.')
    prices = prices[tickers]
    lookback = checkpoint['lookback']
    if len(prices) < lookback + 1:
        raise ValueError('Not enough history.')
    returns = prices.pct_change(fill_method=None).iloc[1:].to_numpy() * 100
    raw = torch.tensor(window_features(returns[-lookback:])[None], dtype=torch.float32)
    x = (raw - checkpoint['mean']) / checkpoint['scale']
    model = DirectionNet(raw.shape[-1], checkpoint['hidden'], checkpoint['kind'],
                         checkpoint['incidence'], checkpoint['adjacency'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        p = model(x).sigmoid()[0].numpy()
    return pd.DataFrame({'ticker': tickers, 'as_of': str(prices.index[-1].date()),
                         'horizon_sessions': checkpoint['horizon'], 'probability_up': p,
                         'validation_threshold': checkpoint['threshold'],
                         'predicted_up': p >= checkpoint['threshold']})


def evaluate(data, prices, probabilities, threshold, args):
    test = data.splits['test']
    actual, p, dates = data.y[test], probabilities[test], data.dates[test]
    trading, trades = trading_backtest(prices, dates, p, args.horizon, args.top_k, args.cost_bps)
    result = {'classification': classification(actual > 0, p, threshold),
              'classification_at_0_5': classification(actual > 0, p, 0.5),
              'nonoverlap_classification': classification(actual[::args.horizon] > 0, p[::args.horizon], threshold),
              'ranking': ndcg(actual, p, args.top_k), 'trading': trading}
    return result, trades


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('data/prices.csv'))
    parser.add_argument('--out', type=Path, default=Path('runs/direction'))
    parser.add_argument('--horizon', type=int, default=21)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27, 37, 47])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--neighbors', type=int, default=5)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--cost-bps', type=float, default=10)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.hidden < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('Training settings must be positive.')
    torch.set_num_threads(2)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Output directory is not empty; choose a new --out to preserve previous experiments.')
    args.out.mkdir(parents=True, exist_ok=True)
    prices = load_prices(args.csv)
    if not 1 <= args.top_k <= len(prices.columns):
        parser.error('top-k must be within the stock universe.')
    data = prepare(prices, lookback=20, k=min(3, len(prices.columns) - 1), horizon=args.horizon)
    train_part, val, test = [data.splits[k] for k in ('train', 'validation', 'test')]
    # Neither validation nor test prices can influence these relationships.
    cutoff = data.dates[train_part][-1]
    train_returns = prices.pct_change(fill_method=None).loc[:cutoff].iloc[1:].to_numpy() * 100
    h = correlation_hyperedges(train_returns, args.neighbors)
    trajectories = ((train_returns - train_returns.mean(0)) /
                    np.maximum(train_returns.std(0), 1e-8)).T / np.sqrt(len(train_returns))
    feature_distances = np.linalg.norm(trajectories[:, None] - trajectories[None], axis=-1)
    geometry = {'training_cutoff': str(cutoff.date()), 's1': hypergraph_geometry(h, 1),
                's2': hypergraph_geometry(h, 2),
                'training_trajectory_metric': four_point_delta(feature_distances),
                'trajectory_definition': 'Each stock = standardized daily training-return trajectory / sqrt(T).',
                'hyperedges': [[str(ticker) for ticker, present in zip(prices.columns, column) if present]
                               for column in h.T]}
    (args.out / 'geometry.json').write_text(json.dumps(geometry, indent=2), encoding='utf-8')
    metadata = {'data_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                'tickers': prices.columns.tolist(), 'torch_version': str(torch.__version__),
                'features': feature_names(20), 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'target': 'Up iff adjusted close(t+h) > adjusted close(t); exact ties are down/flat.',
                'splits': {name: {'origins': part.stop - part.start,
                                 'first_origin': str(data.dates[part][0].date()),
                                 'last_origin': str(data.dates[part][-1].date()),
                                 'last_target': str(data.target_dates[part][-1].date())}
                           for name, part in data.splits.items()}}
    (args.out / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    y_train = data.y[train_part] > 0
    stock_prior = (y_train.sum(0) + 1) / (len(y_train) + 2)
    baselines = {'always_up': np.ones_like(data.y),
                 'train_prior': np.full_like(data.y, (y_train.sum() + 1) / (y_train.size + 2)),
                 'stock_prior': np.broadcast_to(stock_prior, data.y.shape)}
    records, raw_results, prediction_frames = [], {}, []

    def record(name, seed, p, threshold, extra=None):
        result, trades = evaluate(data, prices, p, threshold, args)
        key = f'{name}_{seed}'
        result.update({'threshold': threshold, **(extra or {})})
        raw_results[key] = result
        pd.DataFrame(trades).to_csv(args.out / f'{key}_trades.csv', index=False)
        row = {'model': name, 'seed': seed, 'threshold': threshold,
               **{k: result['classification'][k] for k in ['accuracy', 'macro_f1', 'balanced_accuracy', 'mcc', 'roc_auc', 'brier', 'predicted_up_fraction']},
               'nonoverlap_macro_f1': result['nonoverlap_classification']['macro_f1'],
               'ndcg_at_k': result['ranking']['ndcg_at_k'], **result['trading']}
        records.append(row)
        shape = data.y[test].shape
        prediction_frames.append(pd.DataFrame({'model': name, 'seed': seed,
            'as_of': np.repeat(data.dates[test].astype(str), shape[1]),
            'target_date': np.repeat(data.target_dates[test].astype(str), shape[1]),
            'ticker': np.tile(prices.columns, shape[0]), 'actual_return_pp': data.y[test].ravel(),
            'actual_up': (data.y[test] > 0).ravel(), 'probability_up': p[test].ravel(),
            'threshold': threshold, 'predicted_up': (p[test] >= threshold).ravel()}))
        print(f'{key}: macro F1={row["macro_f1"]:.4f}, AUC={row["roc_auc"]:.4f}, net Sharpe={row["net_annualized_sharpe_rf0"]}', flush=True)

    for name, p in baselines.items():
        threshold = 0.5 if name == 'always_up' else select_threshold(data.y[val] > 0, p[val])
        record(name, 0, p, threshold)
    for kind in ['logistic', 'mlp', 'gcn', 'hypergraph', 'hyperbolic']:
        for seed in args.seeds:
            model, p, threshold, history, epoch = train(data, h, kind, seed, args)
            key = f'{kind}_{seed}'
            checkpoint = {'state_dict': model.state_dict(), 'kind': kind, 'hidden': args.hidden,
                          'incidence': torch.tensor(h), 'adjacency': torch.tensor(data.adjacency),
                          'tickers': prices.columns.tolist(), 'mean': torch.tensor(data.mean),
                          'scale': torch.tensor(data.scale), 'lookback': 20, 'horizon': args.horizon,
                          'threshold': threshold, 'training_cutoff': str(cutoff.date())}
            torch.save(checkpoint, args.out / f'{key}.pt')
            pd.DataFrame(history).to_csv(args.out / f'{key}_training.csv', index=False)
            record(kind, seed, p, threshold, {'best_epoch': epoch,
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'validation': classification(data.y[val] > 0, p[val], threshold)})
            predict_direction(checkpoint, prices).to_csv(args.out / f'{key}_latest.csv', index=False)
    pd.DataFrame(records).to_csv(args.out / 'metrics.csv', index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.out / 'predictions.csv', index=False)
    (args.out / 'metrics.json').write_text(json.dumps(raw_results, indent=2, allow_nan=False), encoding='utf-8')
    numeric = ['macro_f1', 'accuracy', 'balanced_accuracy', 'roc_auc', 'brier', 'nonoverlap_macro_f1',
               'ndcg_at_k', 'net_annualized_sharpe_rf0', 'net_total_return', 'net_daily_close_max_drawdown']
    table = pd.DataFrame(records).groupby('model')[numeric].agg(['mean', 'std'])
    table.columns = ['_'.join(c) for c in table.columns]
    table.to_csv(args.out / 'summary.csv')
    print(table.to_string(), flush=True)


if __name__ == '__main__':
    main()
