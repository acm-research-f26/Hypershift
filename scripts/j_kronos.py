"""Study J: zero-shot Kronos h5 ranking scores in the screen run-folder contract (plan: Study J amendment).

  prep      align adjusted OHLCV from bars.csv.gz to the screen panel -> ohlcv npz (no download)
  timing    seconds per signal day on given pre-2024 dates; nothing saved
  run       forecast one shard of the evaluation days of each (year, seed) -> shard npz (resumable)
  assemble  merge shards and write config.json / predictions.npz / daily.npz / metrics.json like the runner
  combine   kronos_static arm from finished static and kronos run folders (no fitting)

  python scripts/j_kronos.py prep --bars data/fresh/sp500_daily/bars.csv.gz --out data/fresh/sp500_daily/ohlcv_sp500_2015_2026.npz
  python scripts/j_kronos.py run --years 2025 2026 --seeds 7 --shard 0 2 --device cuda:0 ...
  python scripts/j_kronos.py assemble --years 2025 2026 --seeds 7 --shards 2 ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hypershift.screening.data import load_panel, sha256  # noqa: E402
from hypershift.screening.metrics import rank_metrics, trading  # noqa: E402

SHORT = 64  # stocks eligible (64 observed sessions) but without a full lookback use 64 sessions


def common(p):
    p.add_argument('--panel', default='data/fresh/sp500_daily/screen_sp500_2015_2026.npz')
    p.add_argument('--ohlcv', default='data/fresh/sp500_daily/ohlcv_sp500_2015_2026.npz')
    p.add_argument('--output', default='results/SCREEN_SP500_20261021_j')
    p.add_argument('--arm', default='kronos')
    p.add_argument('--years', nargs='+', type=int, default=[2025, 2026])
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 19, 42, 101, 202])
    p.add_argument('--start', help='first signal date kept (e.g. 2024-07-01 for fold 2024H2)')
    p.add_argument('--kronos-dir', default='../Kronos-ext')
    p.add_argument('--model', default='NeoQuasar/Kronos-small')
    p.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    p.add_argument('--max-context', type=int, default=512)
    p.add_argument('--lookback', type=int, default=256)
    p.add_argument('--paths', type=int, default=4)
    p.add_argument('--chunk', type=int, default=77, help='stocks per forward batch (x paths); fixed so RNG use is fixed')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-days', type=int, help='smoke only: first N evaluation days (output must contain "smoke")')


def prep(args):
    import pandas as pd
    z = np.load(args.panel)
    dates, tickers = z['dates'].astype('datetime64[D]'), z['tickers']
    b = pd.read_csv(args.bars, parse_dates=['timestamp'])
    b['timestamp'] = b.timestamp.values.astype('datetime64[D]')
    cols = ['open', 'high', 'low', 'close', 'volume']
    x = np.stack([b.pivot(index='ticker', columns='timestamp', values=c)
                  .reindex(index=tickers, columns=pd.DatetimeIndex(dates)).values for c in cols], -1)
    seen = z['mask'].astype(bool)
    assert not np.isnan(x[seen]).any(), 'bar missing where panel observed'
    ratio = np.where(seen, x[..., 3] / z['features'][..., -1], np.nan)
    spread = float(np.nanmax(np.nanmax(ratio, 1) / np.nanmin(ratio, 1) - 1))
    assert spread < 1e-5, spread
    np.savez_compressed(args.out, ohlcv=np.nan_to_num(x).astype(np.float64), dates=dates, tickers=tickers)
    print(args.out, x.shape, 'close/panel spread', spread, 'sha256', sha256(args.out))


class Forecaster:
    def __init__(self, args):
        import torch
        sys.path.insert(0, str(Path(args.kronos_dir).resolve()))
        from model import Kronos, KronosTokenizer
        from model.kronos import auto_regressive_inference
        self.torch, self.infer, self.args = torch, auto_regressive_inference, args
        self.tok = KronosTokenizer.from_pretrained(args.tokenizer).to(args.device).eval()
        self.model = Kronos.from_pretrained(args.model).to(args.device).eval()
        z = np.load(args.ohlcv)
        self.x, self.dates = z['ohlcv'], z['dates'].astype('datetime64[D]')
        d = self.dates.astype(object)
        # calc_time_stamps columns: minute, hour, weekday, day, month (daily bars: minute = hour = 0)
        self.stamp = np.array([[0, 0, t.weekday(), t.day, t.month] for t in d], np.float32)

    def day(self, t, ids, observed, seed):
        """Mean-of-paths and path-1 h5 scores for stocks ids at signal index t."""
        torch, a = self.torch, self.args
        full = observed[ids, t-a.lookback+1:t+1].all(1) if t-a.lookback+1 >= 0 else np.zeros(len(ids), bool)
        score, first = np.zeros(len(ids)), np.zeros(len(ids))
        for L, sel in ((a.lookback, full), (SHORT, ~full)):
            if not sel.any():
                continue
            g = ids[sel]
            x = self.x[g, t-L+1:t+1]
            x = np.concatenate([x, x[..., 4:5] * x[..., :4].mean(-1, keepdims=True)], -1)   # amount fill
            mu, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True)
            xn = np.clip((x - mu) / (sd + 1e-5), -5, 5).astype(np.float32)
            xs, ys = self.stamp[t-L+1:t+1], self.stamp[t+1:t+6]
            torch.manual_seed(int(seed) * 1_000_003 + int(t) * 17 + L)
            out = []
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=not a.no_amp and 'cuda' in a.device):
                for j in range(0, len(g), a.chunk):
                    xb = torch.tensor(xn[j:j+a.chunk], device=a.device).repeat_interleave(a.paths, 0)
                    n = len(xb)
                    sx = torch.tensor(xs, device=a.device)[None].expand(n, -1, -1).contiguous()
                    sy = torch.tensor(ys, device=a.device)[None].expand(n, -1, -1).contiguous()
                    # sample_count=1 on path-replicated rows == official sample_count=paths, before its mean
                    out.append(self.infer(self.tok, self.model, xb, sx, sy, a.max_context, 5, 5, 1., 0, .9, 1)[:, -1, 3])
            close = np.concatenate(out).reshape(len(g), a.paths) * (sd[:, 0, 3:4] + 1e-5) + mu[:, 0, 3:4]
            r = close / x[:, -1, 3:4] - 1
            score[sel], first[sel] = r.mean(1), r[:, 0]
        return score, first


def eval_signals(panel, args):
    s = panel.evaluate
    s = s[panel.dates[s] >= np.datetime64(args.start)] if args.start else s
    return s[:args.max_days] if args.max_days else s


def timing(args):
    f = Forecaster(args)
    z = np.load(args.panel)
    seen = z['mask'].astype(bool)
    for d in args.dates:
        assert np.datetime64(d) < np.datetime64('2024-01-01'), 'timing uses pre-2024 days only'
        t = int(np.searchsorted(f.dates, np.datetime64(d)))
        ids = np.flatnonzero(seen[:, t-63:t+1].all(1))
        f.torch.cuda.synchronize(); s = time.time()
        f.day(t, ids, seen, 7)
        f.torch.cuda.synchronize(); print('timing', d, 'stocks', len(ids), 'seconds/day', round(time.time()-s, 2), flush=True)


def shard_path(args, year, seed, i, n):
    return Path(args.output)/str(year)/args.arm/f'seed_{seed}'/f'shard_{i}_of_{n}.npz'


def run(args):
    f = Forecaster(args)
    i, n = args.shard
    for year in args.years:
        panel = load_panel(args.panel, year, 1)
        assert np.array_equal(f.dates[:len(panel.dates)], panel.dates)
        signals = eval_signals(panel, args)[i::n]
        for seed in args.seeds:
            path = shard_path(args, year, seed, i, n); path.parent.mkdir(parents=True, exist_ok=True)
            done = dict(np.load(path)) if path.exists() else {'signals': np.zeros(0, int)}
            if len(done['signals']) == len(signals):
                print('reuse', path, flush=True); continue
            pred = done.get('prediction', np.zeros((0, len(panel.tickers))))
            first = done.get('prediction_path1', np.zeros((0, len(panel.tickers))))
            todo = signals[len(done['signals']):]
            start = time.time()
            for k, t in enumerate(todo):
                ids = np.flatnonzero(panel.eligible[:, t])
                row, row1 = np.zeros((1, len(panel.tickers))), np.zeros((1, len(panel.tickers)))
                row[0, ids], row1[0, ids] = f.day(t, ids, panel.observed, seed)
                pred, first = np.concatenate([pred, row]), np.concatenate([first, row1])
                if (k + 1) % 20 == 0 or k + 1 == len(todo):
                    np.savez(path, signals=signals[:len(pred)], prediction=pred, prediction_path1=first)
                    print(f'{year} seed {seed} shard {i}/{n}: {len(pred)}/{len(signals)} days, '
                          f'{(time.time()-start)/(k+1):.2f} s/day', flush=True)


def provenance(args):
    kd = Path(args.kronos_dir).resolve()
    files = sorted(kd.glob('model/*.py'))

    def weights(name):
        p = Path(name)
        if not p.exists():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(name))
        return {q.name: sha256(q) for q in sorted(p.glob('*')) if q.is_file() and q.suffix in ('.safetensors', '.json')}
    return {'script_sha256': sha256(__file__), 'kronos_model_code': {q.name: sha256(q) for q in files},
            'model': args.model, 'model_files': weights(args.model), 'tokenizer': args.tokenizer,
            'tokenizer_files': weights(args.tokenizer)}


def assemble(args):
    import torch
    prov = provenance(args)
    for year in args.years:
        panel = load_panel(args.panel, year, 1)
        signals = eval_signals(panel, args)
        for seed in args.seeds:
            folder = Path(args.output)/str(year)/args.arm/f'seed_{seed}'
            if (folder/'metrics.json').exists():
                print('exists', folder); continue
            parts = [dict(np.load(shard_path(args, year, seed, i, args.shards))) for i in range(args.shards)]
            pred, first = np.zeros((len(signals), len(panel.tickers))), np.zeros((len(signals), len(panel.tickers)))
            got = 0
            for i, z in enumerate(parts):
                assert np.array_equal(z['signals'], signals[i::args.shards]), f'shard {i} incomplete'
                pred[i::args.shards], first[i::args.shards] = z['prediction'], z['prediction_path1']
                got += len(z['signals'])
            assert got == len(signals)
            env = json.loads((folder/'env.json').read_text()) if (folder/'env.json').exists() else {}
            cfg = {'arm': args.arm, 'year': year, 'seed': seed, 'dataset_sha256': panel.digest,
                   'ohlcv_sha256': sha256(args.ohlcv), 'source': prov, 'smoke': bool(args.max_days), 'max_days': args.max_days, 'protocol': 'study-j-kronos-zero-shot',
                   'start': args.start, 'lookback': args.lookback, 'short_lookback': SHORT, 'paths': args.paths,
                   'chunk': args.chunk, 'T': 1., 'top_k': 0, 'top_p': .9, 'max_context': args.max_context,
                   'amp_fp16': not args.no_amp, 'shards': args.shards, 'horizon': 5, 'environment': env,
                   'torch': torch.__version__}
            write_run(folder, cfg, panel, signals, pred, first)


def write_run(folder, cfg, panel, signals, pred, first=None):
    """Score pred like the runner and write config.json / predictions.npz / daily.npz / metrics.json."""
    (folder/'config.json').write_text(json.dumps(cfg, indent=2))
    raw, mask = panel.target(signals)
    raw *= panel.yscale
    result, daily = rank_metrics(pred, raw, mask)
    neutral, nm = panel.target(signals, 'sector')
    nr, nd = rank_metrics(pred, neutral*panel.yscale, nm)
    result.update({'neutral_RankIC': nr['RankIC'], 'neutral_spread10': nr['spread10'], 'neutral_kind': 'sector'})
    extra = {}
    if first is not None:
        r1, d1 = rank_metrics(first, raw, mask)
        result['RankIC_path1'] = r1['RankIC']
        extra = {'RankIC_path1': d1['RankIC']}
    portfolio, pnl = trading(pred, signals, panel)
    result.update(portfolio)
    result.update({'training': {'epochs': 0, 'parameters': None}, 'year': cfg['year'], 'seed': cfg['seed'], 'arm': cfg['arm'],
                   'experiment': 'J', 'parent': 'static', 'smoke': cfg['smoke'], 'status': 'complete'})
    np.savez_compressed(folder/'predictions.npz', prediction=pred, target=raw, mask=mask, signals=signals,
                        dates=panel.dates[signals], tickers=panel.tickers,
                        **({} if first is None else {'prediction_path1': first}))
    np.savez_compressed(folder/'daily.npz', **daily, neutral_RankIC=nd['RankIC'], portfolio=pnl, **extra)
    result['artifacts'] = {name: sha256(folder/name) for name in ('config.json', 'predictions.npz', 'daily.npz')}
    (folder/'metrics.json').write_text(json.dumps(result, indent=2))
    print('COMPLETE', cfg['arm'], cfg['year'], cfg['seed'], 'RankIC', round(result['RankIC'], 6), flush=True)


def combine(args):
    """kronos_static: per day, mean of cross-sectional z-scores (over eligible stocks) of static seed k and kronos seed k."""
    root = Path(args.output)
    for year in args.years:
        panel = load_panel(args.panel, year, 1)
        for seed in args.seeds:
            folder = root/str(year)/'kronos_static'/f'seed_{seed}'
            src = {a: root/str(year)/a/f'seed_{seed}' for a in ('static', 'kronos')}
            if (folder/'metrics.json').exists() or not all((d/'metrics.json').exists() for d in src.values()):
                print('skip', folder, flush=True); continue
            k, s = (np.load(src[a]/'predictions.npz') for a in ('kronos', 'static'))
            signals = k['signals']
            pos = {int(t): i for i, t in enumerate(s['signals'])}
            ps = s['prediction'][[pos[int(t)] for t in signals]]
            e = panel.eligible[:, signals].T

            def z(p):
                m = np.where(e, p, np.nan)
                return np.where(e, (p - np.nanmean(m, 1, keepdims=True)) / np.maximum(np.nanstd(m, 1, keepdims=True), 1e-12), 0)
            folder.mkdir(parents=True, exist_ok=True)
            cfg = {'arm': 'kronos_static', 'year': year, 'seed': seed, 'dataset_sha256': panel.digest, 'smoke': False,
                   'protocol': 'study-j-kronos-zero-shot', 'combine': 'mean of daily cross-sectional z-scores, eligible stocks',
                   'inputs': {a: sha256(src[a]/'predictions.npz') for a in src}, 'script_sha256': sha256(__file__)}
            write_run(folder, cfg, panel, signals, (z(k['prediction']) + z(ps)) / 2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('prep')
    q.add_argument('--panel', default='data/fresh/sp500_daily/screen_sp500_2015_2026.npz')
    q.add_argument('--bars', default='data/fresh/sp500_daily/bars.csv.gz')
    q.add_argument('--out', default='data/fresh/sp500_daily/ohlcv_sp500_2015_2026.npz')
    q = sub.add_parser('timing'); common(q); q.add_argument('--dates', nargs='+', required=True)
    q = sub.add_parser('run'); common(q); q.add_argument('--shard', nargs=2, type=int, default=[0, 1])
    q = sub.add_parser('assemble'); common(q); q.add_argument('--shards', type=int, default=1)
    q = sub.add_parser('combine'); common(q)
    args = p.parse_args()
    if getattr(args, 'max_days', None) and 'smoke' not in args.output.lower():
        p.error('--max-days output must contain smoke')
    if getattr(args, 'years', None) and any(y not in (2022, 2023, 2024, 2025, 2026) for y in args.years):
        p.error('Study J folds: 2024-2026 (reserved) or 2022/2023 (memorisation diagnostic)')
    {'prep': prep, 'timing': timing, 'run': run, 'assemble': assemble, 'combine': combine}[args.cmd](args)


if __name__ == '__main__':
    main()
