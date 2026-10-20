"""Analysis / report generator for the broad research screen (v2 RSR, v3 S&P 500).

Reads <root>/<year>/<arm>/seed_<k>/ run folders (never writes into them) and writes, prefixed by --tag:
  SCREENING_RESULTS.csv, INVENTORY.md, PAIRED.csv, SCREENING_TABLES.md,
  TRADING_SENSITIVITY.csv (unless --no-trading), figures/*.png.

Paired design (plan: docs/screening_20261019/RESEARCH_SCREENING_PLAN.md):
per fold, matched seeds and common dates; delta of daily RankIC, seed-averaged daily series, stationary bootstrap
(expected block 10, B=5000, fixed rng) for CI and p; pooled = equal-weight mean of fold means; Holm over the primary family.
Exploratory screen triage only; the verdict words are NOT the project's confirmatory verdicts.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')

from hypershift.eval.stats import holm  # noqa: E402
from hypershift.screening.runner import ARMS  # noqa: E402

SCALARS = ['IC', 'RankIC', 'MSE', 'spread10', 'prediction_std', 'unique_fraction',
           'NDCG5', 'random_NDCG5', 'NDCG10', 'random_NDCG10', 'NDCG20', 'random_NDCG20',
           'neutral_RankIC', 'neutral_spread10', 'gross_Sharpe', 'net_Sharpe', 'CAGR', 'max_drawdown',
           'turnover', 'missing_held_marks', 'market_correlation']
SINGLE_SEED = ('ridge', 'momentum', 'reversal')      # statistical arms run for the first seed only
FAMILY = [('A', 'residual_soft'), ('B', 'sector_target'), ('C', 'hybrid'), ('D', 'multiscale'), ('E', 'gated'),
          ('F', 'two_residual'), ('G', 'summary'), ('H', 'pretrain'), ('I', 'adaptive')]
REQ_SEEDS, REQ_FOLDS, MIN_DELTA = 3, 2, 0.003
TRADING_K, TRADING_COST = (5, 10, 20), (0.0, 0.001, 0.0025)
OTHER_DELTAS = ['NDCG10', 'spread10', 'IC', 'neutral_RankIC']


# ----------------------------------------------------------------------------- pure functions
def family_arms(present):
    """Primary Holm family as {direction letter: arm}; B falls back to market_target if sector_target absent."""
    fam = {}
    for letter, arm in FAMILY:
        if letter == 'B' and arm not in present and 'market_target' in present:
            arm = 'market_target'
        fam[letter] = arm
    return fam


def bootstrap_means(series, block=10, B=5000, rng=None):
    """Stationary bootstrap (Politis-Romano, same recursion as eval.stats.stationary_bootstrap_indices, vectorised):
    idx[t] = fresh uniform w.p. 1/block else idx[t-1]+1 (mod n). Returns B bootstrap means."""
    s = np.asarray(series, float)
    n = len(s)
    rng = np.random.default_rng(0) if rng is None else rng
    restart = rng.random((B, n)) < 1.0 / block
    restart[:, 0] = True
    fresh = rng.integers(n, size=(B, n))
    t = np.arange(n)[None, :]
    seg_start = np.maximum.accumulate(np.where(restart, t, 0), axis=1)
    start_val = np.take_along_axis(fresh, seg_start, axis=1)
    idx = (start_val + t - seg_start) % n
    return s[idx].mean(1)


def boot_p(m, boots):
    """Two-sided p: fraction of bootstrap means with |m* - m| >= |m| (m = observed mean)."""
    return float(np.mean(np.abs(boots - m) >= abs(m)))


def verdict(fold_seed_deltas, pooled, n_seeds=REQ_SEEDS, n_folds=REQ_FOLDS, min_delta=MIN_DELTA):
    """fold_seed_deltas: {fold: [per-seed delta RankIC]}. Screening triage label, exactly as in the plan."""
    if pooled is None or len(fold_seed_deltas) < n_folds or any(len(v) < n_seeds for v in fold_seed_deltas.values()):
        return 'Inconclusive'
    fold_means = [float(np.mean(v)) for v in fold_seed_deltas.values()]
    if (all(m > 0 for m in fold_means)
            and all(sum(d > 0 for d in v) >= 2 for v in fold_seed_deltas.values())
            and pooled >= min_delta):
        return 'Promising'
    if pooled <= -min_delta:
        return 'Negative preliminary result'
    return 'Neutral'


def sha12(x):
    return hashlib.sha256(x.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- loading
def load_runs(root):
    """Scan run folders. Returns (runs list of dicts, inventory dict)."""
    root = Path(root)
    folds = sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    status = {}  # (fold, arm, seed) -> 'complete'|'failed'|'smoke'|'incomplete'
    runs = []
    arms_present = set()
    for y in folds:
        for ad in sorted((root / str(y)).iterdir()):
            if not ad.is_dir() or ad.name not in ARMS:
                continue
            arms_present.add(ad.name)
            for sd in sorted(ad.glob('seed_*')):
                seed = int(sd.name.split('_')[1])
                key = (y, ad.name, seed)
                if (sd / 'failed.json').exists():
                    status[key] = 'failed'; continue
                if not (sd / 'metrics.json').exists():
                    status[key] = 'incomplete'; continue
                m = json.loads((sd / 'metrics.json').read_text())
                cfg = json.loads((sd / 'config.json').read_text()) if (sd / 'config.json').exists() else {}
                if m.get('smoke') or cfg.get('smoke'):
                    status[key] = 'smoke'; continue
                status[key] = 'complete'
                runs.append({'fold': y, 'arm': ad.name, 'seed': seed, 'dir': sd, 'm': m, 'cfg': cfg})
    return runs, status, folds, sorted(arms_present)


def expected_keys(status, folds, arms_present):
    """Expected = arms with a directory x seeds present x folds present (statistical single-seed arms: their seed;
    frozen: folds after the first only, since the first fold is identical to static)."""
    seeds_all = sorted({s for (_, a, s) in status if a not in SINGLE_SEED})
    keys = []
    for arm in arms_present:
        if arm in SINGLE_SEED:
            seen = sorted({s for (_, a, s) in status if a == arm})
            seeds = seen[:1] if seen else seeds_all[:1]
        else:
            seeds = seeds_all
        for y in folds:
            if arm == 'frozen' and y == folds[0]:
                continue
            for s in seeds:
                keys.append((y, arm, s))
    return keys


def flatten(run):
    m, cfg = run['m'], run['cfg']
    tr = m.get('training', {}) or {}
    hist = tr.get('history') or []
    select = cfg.get('select', 'mse')
    best_epoch = None
    if hist:
        if select == 'rankic' and all('validation_RankIC' in h for h in hist):
            best_epoch = int(max(hist, key=lambda h: h['validation_RankIC'])['epoch'])
        elif all('validation_MSE' in h for h in hist):
            best_epoch = int(min(hist, key=lambda h: h['validation_MSE'])['epoch'])
    row = {'fold': run['fold'], 'arm': run['arm'], 'seed': run['seed'],
           'experiment': m.get('experiment'), 'parent': m.get('parent')}
    for k in SCALARS:
        row[k] = m.get(k)
    for k in (5, 10, 20):
        a, b = m.get(f'NDCG{k}'), m.get(f'random_NDCG{k}')
        row[f'NDCG{k}_excess'] = None if a is None or b is None else a - b
    row.update({'epochs': tr.get('epochs'), 'best_epoch': best_epoch, 'parameters': tr.get('parameters'),
                'runtime_s': m.get('runtime_s'), 'device': cfg.get('device'), 'protocol': cfg.get('protocol'),
                'select': select,
                'dataset_sha12': (cfg.get('dataset_sha256') or '')[:12],
                'source_sha12': sha12(''.join(v for _, v in sorted((cfg.get('source') or {}).items())))
                if cfg.get('source') else ''})
    return row


class Store:
    def __init__(self, runs):
        self.runs = runs
        self.by = {}
        for r in runs:
            self.by[(r['fold'], r['arm'], r['seed'])] = r
        self._daily = {}

    def seeds(self, fold, arm):
        return sorted(s for (f, a, s) in self.by if f == fold and a == arm)

    def daily(self, fold, arm, seed, key='RankIC'):
        k = (fold, arm, seed)
        if k not in self._daily:
            d = self.by[k]['dir']
            z = np.load(d / 'daily.npz')
            dates = np.load(d / 'predictions.npz', allow_pickle=False)['dates']
            self._daily[k] = ({n: z[n] for n in z.files}, dates)
        return self._daily[k][0][key], self._daily[k][1]

    def scalar(self, fold, arm, seed, key):
        return self.by[(fold, arm, seed)]['m'].get(key)


def paired_fold(store, fold, arm, parent):
    """Per-fold paired quantities for arm vs parent, or None if not computable."""
    sa, sp = store.seeds(fold, arm), store.seeds(fold, parent)
    if not sa or not sp:
        return None
    if arm in SINGLE_SEED:
        a_seeds, base = sa[:1], sp          # compare with the parent's seed-mean series
        seed_pairs = [(sa[0], None)]
    else:
        matched = sorted(set(sa) & set(sp))
        if not matched:
            return None
        a_seeds, base = matched, matched
        seed_pairs = [(s, s) for s in matched]
    series_a, series_p, dates_list = {}, {}, []
    for s in a_seeds:
        v, d = store.daily(fold, arm, s); series_a[s] = v; dates_list.append(d)
    for s in base:
        v, d = store.daily(fold, parent, s); series_p[s] = v; dates_list.append(d)
    common = sorted(set.intersection(*[set(map(str, d)) for d in dates_list]))
    if len(common) < 20:
        return None

    def sel(v, d):
        pos = {str(x): i for i, x in enumerate(d)}
        return np.array([v[pos[c]] for c in common])
    aligned_a = {s: sel(series_a[s], store.daily(fold, arm, s)[1]) for s in a_seeds}
    aligned_p = {s: sel(series_p[s], store.daily(fold, parent, s)[1]) for s in base}
    pmean = np.mean([aligned_p[s] for s in base], axis=0)
    seed_delta, daily_deltas = {}, []
    for sa_, sp_ in seed_pairs:
        ref = pmean if sp_ is None else aligned_p[sp_]
        dd = aligned_a[sa_] - ref
        seed_delta[sa_] = float(dd.mean()); daily_deltas.append(dd)
    series = np.mean(daily_deltas, axis=0)
    other = {}
    for k in OTHER_DELTAS:
        vals = []
        for sa_, sp_ in seed_pairs:
            pv = (np.mean([store.scalar(fold, parent, s, k) for s in base]) if sp_ is None
                  else store.scalar(fold, parent, sp_, k))
            av = store.scalar(fold, arm, sa_, k)
            if av is not None and pv is not None:
                vals.append(av - pv)
        other[k] = float(np.mean(vals)) if vals else np.nan
    return {'series': series, 'seed_delta': seed_delta, 'mean': float(series.mean()), 'n_dates': len(common),
            'n_seeds': len(seed_delta), 'other': other}


def paired_arm(store, folds, arm, parent, B=5000, block=10):
    per = {}
    for f in folds:
        r = paired_fold(store, f, arm, parent)
        if r is None:
            continue
        r['boots'] = bootstrap_means(r['series'], block, B, np.random.default_rng((0, f)))
        r['p'] = boot_p(r['mean'], r['boots'])
        r['ci'] = tuple(np.quantile(r['boots'], [.025, .975]))
        per[f] = r
    if not per:
        return None
    pooled = float(np.mean([r['mean'] for r in per.values()]))
    pb = np.mean([r['boots'] for r in per.values()], axis=0)
    out = {'arm': arm, 'parent': parent, 'per_fold': per, 'pooled': pooled,
           'ci': tuple(np.quantile(pb, [.025, .975])), 'p': boot_p(pooled, pb),
           'other': {k: float(np.nanmean([r['other'][k] for r in per.values()])) for k in OTHER_DELTAS},
           'verdict': verdict({f: list(r['seed_delta'].values()) for f, r in per.items()}, pooled)}
    return out


# ----------------------------------------------------------------------------- tables
def f(x, nd=4, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f'{x:+.{nd}f}' if plus else f'{x:.{nd}f}'


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def pooled_metric(rows, arm, key):
    """Pooled = mean over folds of the fold's seed mean."""
    by = {}
    for r in rows:
        if r['arm'] == arm and r.get(key) is not None:
            by.setdefault(r['fold'], []).append(r[key])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def write_csv(path, rows, cols):
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            w.writerow(['' if (r.get(c) is None or (isinstance(r.get(c), float) and not np.isfinite(r[c])))
                        else r.get(c) for c in cols])


def build(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    tag = args.tag
    runs, status, folds, arms_present = load_runs(args.root)
    store = Store(runs)
    rows = [flatten(r) for r in runs]
    cols = (['fold', 'arm', 'seed', 'experiment', 'parent'] + SCALARS + ['NDCG5_excess', 'NDCG10_excess', 'NDCG20_excess']
            + ['epochs', 'best_epoch', 'parameters', 'runtime_s', 'device', 'protocol', 'select',
               'dataset_sha12', 'source_sha12'])
    write_csv(out / f'{tag}_SCREENING_RESULTS.csv', rows, cols)

    # ---- inventory
    exp = expected_keys(status, folds, arms_present)
    counts = {'expected': len(exp), 'complete': 0, 'failed': 0, 'missing': 0, 'smoke_excluded': 0}
    detail = []
    for k in exp:
        st = status.get(k)
        if st == 'complete': counts['complete'] += 1
        elif st == 'failed': counts['failed'] += 1; detail.append((k, 'failed'))
        elif st == 'smoke': counts['smoke_excluded'] += 1; detail.append((k, 'smoke (excluded)'))
        else: counts['missing'] += 1; detail.append((k, 'running/incomplete' if st == 'incomplete' else 'not started'))
    extra = [k for k, v in status.items() if k not in set(exp) and v == 'complete']
    inv = [f'# Inventory ({tag}) root `{args.root}`', '',
           f'Folds: {folds}; arms with a directory: {len(arms_present)}.',
           'Expected = arms with a directory x seeds present x folds present (ridge/momentum/reversal: first seed only; '
           'frozen: not in the first fold).', '',
           md_table(['expected', 'complete', 'failed', 'missing (incomplete or not started)', 'smoke excluded'],
                    [[counts['expected'], counts['complete'], counts['failed'], counts['missing'], counts['smoke_excluded']]]),
           '', '## Per arm and fold (complete / expected)', '']
    hdr = ['arm'] + [str(y) for y in folds]
    tbl = []
    for arm in arms_present:
        r = [arm]
        for y in folds:
            e = [k for k in exp if k[0] == y and k[1] == arm]
            c = [k for k in e if status.get(k) == 'complete']
            r.append(f'{len(c)}/{len(e)}' if e else '-')
        tbl.append(r)
    inv += [md_table(hdr, tbl), '', '## Not complete', '']
    inv += [md_table(['fold', 'arm', 'seed', 'state'], [[*k, s] for k, s in sorted(detail)])] if detail else ['None.']
    if extra:
        inv += ['', f'Complete runs outside the expected set: {len(extra)}.']
    (out / f'{tag}_INVENTORY.md').write_text('\n'.join(inv) + '\n', encoding='utf-8')

    # ---- paired
    present = set(arms_present)
    fam = family_arms(present)
    fam_arms = set(fam.values())
    paired = {}
    for arm in arms_present:
        parent = ARMS[arm][4]
        if arm == parent:
            continue
        r = paired_arm(store, folds, arm, parent, args.boot, args.block)
        if r:
            paired[arm] = r
    # Fixed family size 9: arms without paired data enter with p=1 (conservative), so m never shrinks.
    fam_p = {a: (paired[a]['p'] if a in paired else 1.0) for a in fam.values()}
    holm_p = {a: v for a, v in holm(fam_p).items() if a in paired}
    prow = []
    for arm, r in paired.items():
        d = {'arm': arm, 'parent': r['parent'], 'experiment': ARMS[arm][0],
             'primary_family': arm in fam_arms, 'pooled_dRankIC': r['pooled'],
             'ci_lo': r['ci'][0], 'ci_hi': r['ci'][1], 'p_boot': r['p'], 'holm_p': holm_p.get(arm),
             'verdict': r['verdict']}
        for k in OTHER_DELTAS:
            d[f'd{k}'] = r['other'][k]
        for y, fr in r['per_fold'].items():
            d[f'fold{y}_dRankIC'] = fr['mean']; d[f'fold{y}_ci_lo'] = fr['ci'][0]; d[f'fold{y}_ci_hi'] = fr['ci'][1]
            d[f'fold{y}_p'] = fr['p']; d[f'fold{y}_n_seeds'] = fr['n_seeds']; d[f'fold{y}_n_dates'] = fr['n_dates']
            d[f'fold{y}_seed_deltas'] = ';'.join(f'{s}:{v:+.5f}' for s, v in fr['seed_delta'].items())
        prow.append(d)
    pcols = ['arm', 'parent', 'experiment', 'primary_family', 'pooled_dRankIC', 'ci_lo', 'ci_hi', 'p_boot', 'holm_p',
             'verdict'] + [f'd{k}' for k in OTHER_DELTAS]
    for y in folds:
        pcols += [f'fold{y}_{s}' for s in ('dRankIC', 'ci_lo', 'ci_hi', 'p', 'n_seeds', 'n_dates', 'seed_deltas')]
    write_csv(out / f'{tag}_PAIRED.csv', prow, pcols)

    # ---- tables
    def runtime(arm):
        v = [r['runtime_s'] for r in rows if r['arm'] == arm and r['runtime_s'] is not None]
        return f'{np.mean(v):.0f}' if v else 'n/a'

    def seeds_cell(r):
        return ' / '.join(f"{y}:{fr['n_seeds']}" for y, fr in r['per_fold'].items()) if r else 'n/a'

    def line(label, arm, with_holm):
        r = paired.get(arm)
        if r is None:
            return [label, arm, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', runtime(arm), 'n/a', 'Inconclusive (no paired data)']
        base = pooled_metric(rows, r['parent'], 'RankIC'); new = pooled_metric(rows, arm, 'RankIC')
        return [label, f"{arm} (vs {r['parent']})", f(base), f(new),
                f"{f(r['pooled'], 4, True)} [{f(r['ci'][0], 4, True)}, {f(r['ci'][1], 4, True)}]",
                f(r['other']['NDCG10'], 4, True), f(r['other']['spread10'], 4, True), seeds_cell(r), runtime(arm),
                f(holm_p.get(arm), 3) if with_holm and arm in holm_p else ('n/a' if with_holm else '-'), r['verdict']]
    head = ['Experiment', 'Method', 'Baseline RankIC', 'New RankIC', 'dRankIC [95% CI]', 'dNDCG@10',
            'dTop-K spread (spread10)', 'Seeds (per fold)', 'Runtime (mean s/fit)', 'Holm p', 'Verdict']
    main = [line(letter, fam[letter], True) for letter in 'ABCDEFGHI']
    main.append(['J', 'Kronos-type frozen financial features', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', '0', 'n/a', '-',
                 'Data/resource blocked'])
    ctrl = [line(ARMS[a][0], a, False) for a in sorted(paired, key=lambda a: (ARMS[a][0], a)) if a not in fam_arms]
    absr = []
    for arm in arms_present:
        n_ = sum(1 for r in rows if r['arm'] == arm)
        absr.append((pooled_metric(rows, arm, 'RankIC'), [
            arm, ARMS[arm][0], n_, f(pooled_metric(rows, arm, 'RankIC')), f(pooled_metric(rows, arm, 'IC')),
            f(pooled_metric(rows, arm, 'NDCG5_excess')), f(pooled_metric(rows, arm, 'NDCG10_excess')),
            f(pooled_metric(rows, arm, 'NDCG20_excess')), f(pooled_metric(rows, arm, 'spread10')),
            f(pooled_metric(rows, arm, 'neutral_RankIC')), f(pooled_metric(rows, arm, 'net_Sharpe'), 2),
            f(pooled_metric(rows, arm, 'turnover'), 3)]))
    absr = [r for _, r in sorted(absr, key=lambda t: -(t[0] if t[0] is not None else -9))]
    md = [f'# Screening tables ({tag})', '',
          f'Root `{args.root}`; folds {folds}. Exploratory screen triage; verdict words are not the project confirmatory verdicts.',
          'Bootstrap: stationary, expected block %d, B=%d, rng seed 0 (per fold). Pooled = equal-weight mean of fold means; '
          'p = fraction of bootstrap means with |m*-m| >= |m|. Holm over the primary family only '
          '(A residual_soft, B %s, C hybrid, D multiscale, E gated vs mix, F two_residual, G summary, H pretrain, I adaptive).'
          % (args.block, args.boot, fam['B']), '',
          '## Direction matrix (principal arms)', '', md_table(head, main), '',
          '## Controls and other arms (descriptive, no Holm)', '', md_table(head, ctrl), '',
          '## Baseline table: absolute pooled metrics (mean over folds of seed means), sorted by RankIC', '',
          md_table(['Arm', 'Exp', 'Runs', 'RankIC', 'IC', 'NDCG5 excess', 'NDCG10 excess', 'NDCG20 excess', 'spread10',
                    'neutral RankIC', 'net Sharpe', 'turnover'], absr), '',
          'Net Sharpe, turnover: top-10, 10 bp (predeclared default). Single-seed statistical arms (ridge, momentum, reversal) '
          'are compared with the seed-mean series of their parent; random has its own seeds.', '']
    (out / f'{tag}_SCREENING_TABLES.md').write_text('\n'.join(md), encoding='utf-8')
    figures(out, tag, store, rows, paired, fam, folds)
    if not args.no_trading:
        trading_sensitivity(args, out, tag, store, folds)
    return counts, main, paired


# ----------------------------------------------------------------------------- trading
def trading_sensitivity(args, out, tag, store, folds):
    from hypershift.screening.data import load_panel
    from hypershift.screening.metrics import trading
    if not args.panel:
        print('no --panel: skipping trading sensitivity'); return
    rows = []
    for y in folds:
        keys = [k for k in store.by if k[0] == y]
        if not keys:
            continue
        stride = store.by[keys[0]]['cfg'].get('train_stride', 5)
        panel = load_panel(args.panel, y, stride)     # cached per fold: loaded once here
        res = {}
        for (fy, arm, seed) in keys:
            z = np.load(store.by[(fy, arm, seed)]['dir'] / 'predictions.npz', allow_pickle=False)
            for k in TRADING_K:
                for c in TRADING_COST:
                    m, _ = trading(z['prediction'], z['signals'], panel, k, c)
                    res.setdefault((arm, k, c), []).append(m)
        for (arm, k, c), ms in sorted(res.items()):
            r = {'fold': y, 'arm': arm, 'k': k, 'cost': c, 'n_seeds': len(ms),
                 'default_spec': k == 10 and c == 0.001}
            for key in ('net_Sharpe', 'CAGR', 'max_drawdown', 'turnover'):
                r[key] = float(np.mean([m[key] for m in ms]))
            rows.append(r)
    write_csv(out / f'{tag}_TRADING_SENSITIVITY.csv', rows,
              ['fold', 'arm', 'k', 'cost', 'n_seeds', 'default_spec', 'net_Sharpe', 'CAGR', 'max_drawdown', 'turnover'])


# ----------------------------------------------------------------------------- figures
OKABE = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999',
         '#882255', '#44AA99']
EXPS = ['baseline', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'extra']


def figures(out, tag, store, rows, paired, fam, folds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {e: OKABE[i] for i, e in enumerate(EXPS)}
    principal = set(fam.values())
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

    # (a) pooled delta RankIC vs parent
    arms = sorted(paired, key=lambda a: paired[a]['pooled'])
    if arms:
        fig, ax = plt.subplots(figsize=(8, 0.28 * len(arms) + 1.5))
        for i, a in enumerate(arms):
            r = paired[a]; c = col.get(ARMS[a][0], '#999999')
            ax.barh(i, r['pooled'], color=c, alpha=.85)
            ax.errorbar(r['pooled'], i, xerr=[[r['pooled'] - r['ci'][0]], [r['ci'][1] - r['pooled']]],
                        color='black', capsize=2, lw=1)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([('* ' if a in principal else '') + f'{ARMS[a][0]}  {a} (vs {paired[a]["parent"]})' for a in arms])
        ax.axvline(0, color='black', lw=1)
        ax.axvline(MIN_DELTA, color='grey', lw=.8, ls=':'); ax.axvline(-MIN_DELTA, color='grey', lw=.8, ls=':')
        ax.set_xlabel('pooled dRankIC vs parent, 95% CI (dotted: +/-0.003 gate; *: principal arm)', fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=col[e]) for e in EXPS if any(ARMS[a][0] == e for a in arms)]
        ax.legend(handles, [e for e in EXPS if any(ARMS[a][0] == e for a in arms)], title='experiment',
                  loc='lower right', fontsize=8)
        fig.tight_layout(); fig.savefig(out / 'figures' / f'{tag}_delta_rankic.png', dpi=150); plt.close(fig)

    # (b) NDCG excess
    allarms = sorted({r['arm'] for r in rows}, key=lambda a: -(pooled_metric(rows, a, 'NDCG10_excess') or -9))
    if allarms:
        fig, ax = plt.subplots(figsize=(max(8, .45 * len(allarms) + 2), 4.5))
        w = .27
        for j, k in enumerate((5, 10, 20)):
            vals = [pooled_metric(rows, a, f'NDCG{k}_excess') or 0 for a in allarms]
            ax.bar(np.arange(len(allarms)) + (j - 1) * w, vals, w, label=f'NDCG@{k}',
                   color=['#0072B2', '#E69F00', '#009E73'][j])
        ax.axhline(0, color='black', lw=1)
        ax.set_xticks(range(len(allarms))); ax.set_xticklabels(allarms, rotation=60, ha='right')
        ax.set_ylabel('NDCG excess over random (pooled seed means)'); ax.legend()
        fig.tight_layout(); fig.savefig(out / 'figures' / f'{tag}_ranking_quality.png', dpi=150); plt.close(fig)

    # (c) validation curves
    curve_arms = ['static'] + [a for a in fam.values()]
    fig, axes = plt.subplots(1, max(1, len(folds)), figsize=(6.5 * max(1, len(folds)), 4.2), squeeze=False)
    cmap = [OKABE[1], OKABE[2], OKABE[3], OKABE[5], OKABE[6], OKABE[7], OKABE[9], OKABE[10], '#E69F00', '#000000']
    any_curve = False
    for ax, y in zip(axes[0], folds):
        for i, a in enumerate(curve_arms):
            hs = [r['m']['training']['history'] for (fy, aa, s), r in store.by.items()
                  if fy == y and aa == a and (r['m'].get('training') or {}).get('history')]
            hs = [h for h in hs if h and 'validation_RankIC' in h[0]]
            if not hs:
                continue
            L = max(len(h) for h in hs)
            mean = [np.mean([h[e]['validation_RankIC'] for h in hs if len(h) > e]) for e in range(L)]
            ax.plot(range(L), mean, color=cmap[i % len(cmap)], lw=2.2 if a == 'static' else 1.3,
                    ls='-' if a == 'static' else ['-', '--', '-.', ':'][i % 4], label=a); any_curve = True
        ax.set_title(f'fold {y}'); ax.set_xlabel('epoch'); ax.set_ylabel('validation RankIC (seed mean)')
        if not any_curve:
            ax.text(.5, .5, 'validation_RankIC not recorded\n(MSE-selection protocol)', ha='center',
                    va='center', transform=ax.transAxes)
        else:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'figures' / f'{tag}_training_curves.png', dpi=150); plt.close(fig)

    # (d) seed x fold scatter
    pr = [a for a in fam.values() if a in paired]
    if pr:
        fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(pr) + 2), 4.2))
        marks = ['o', 's', '^', 'D']
        for i, a in enumerate(pr):
            for j, y in enumerate(folds):
                fr = paired[a]['per_fold'].get(y)
                if fr:
                    xs = i + (j - (len(folds) - 1) / 2) * .22 + np.linspace(-.04, .04, len(fr['seed_delta']))
                    ax.scatter(xs, list(fr['seed_delta'].values()), marker=marks[j % 4],
                               color=col.get(ARMS[a][0]), edgecolor='black', s=38, label=f'fold {y}' if i == 0 else None)
        ax.axhline(0, color='black', lw=1)
        ax.set_xticks(range(len(pr))); ax.set_xticklabels([f'{a}\n(vs {paired[a]["parent"]})' for a in pr], fontsize=8)
        ax.set_ylabel('per-seed dRankIC vs parent'); ax.legend(title='marker')
        fig.tight_layout(); fig.savefig(out / 'figures' / f'{tag}_seed_fold_scatter.png', dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', default='docs/screening_20261019')
    p.add_argument('--panel')
    p.add_argument('--boot', type=int, default=5000)
    p.add_argument('--block', type=int, default=10)
    p.add_argument('--tag', default='v3')
    p.add_argument('--no-trading', action='store_true')
    args = p.parse_args()
    counts, main_rows, _ = build(args)
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
