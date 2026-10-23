"""Build the point-in-time S&P 500 panel (2016-01-04 .. 2023-12-29) from fja05680 membership + Alpaca SIP bars.

  python scripts/pit_build_panel.py            -> data/pit/pit_sp500_2016_2023.npz, docs/strategy_pit_20261023/PIT_DATA.md
Rules: docs/strategy_pit_20261023/PREREG.md. No bar after 2023-12-31 is requested.
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from hypershift.data import pit_sp500 as PS  # noqa: E402
from hypershift.data.alpaca import fetch_bars, make_http  # noqa: E402

LO, HI = np.datetime64('2016-01-04'), np.datetime64('2023-12-29')
OUT = Path('data/pit')
NPZ = OUT / 'pit_sp500_2016_2023.npz'
DOC = Path('docs/strategy_pit_20261023/PIT_DATA.md')
SURV = 'data/fresh/sp500_daily/screen_sp500_2015_2026.npz'


def asof_of(end):
    return HI if end > HI else np.busday_offset(end, -1, roll='backward')


def fetch(ivs, adj):
    path = OUT / f"bars_{adj}.pkl"   # local cache written by this script only (never downloaded)
    if path.exists():
        return pickle.load(open(path, 'rb'))
    h, raw = make_http(), {}

    def http(p):
        s, js = h(p)
        if s == 200:
            for k, b in (js.get('bars') or {}).items():
                raw.setdefault((p['asof'], k), []).extend(b)
        return s, js
    by = defaultdict(set)
    for t, s, e in ivs:
        by[str(asof_of(e))].add(t)
    for a, syms in sorted(by.items()):
        syms = sorted(syms)
        for j in range(0, len(syms), 100):
            fetch_bars(syms[j:j + 100], '2016-01-01', '2023-12-31', adj, http, asof=a)
    pickle.dump(raw, open(path, 'wb'))
    return raw


MANUAL_NEUTRAL = {('UA', '2016-04-08')}   # Class C 1:1 share dividend, unadjusted by Alpaca (gap 0.527)
ALIAS = {'CCEP': 'CCE'}   # fja05680 lists Coca-Cola Enterprises (CCE) under its successor's ticker for 2016-01..05


def complete(raw, ivs, adj):
    """Multi-symbol requests can drop a symbol when two map to related entities: re-fetch missing keys one by one."""
    path = OUT / f"bars_{adj}.pkl"
    h, added = make_http(), 0
    for t, s, e in ivs:
        a = str(asof_of(e))
        if (a, t) in raw:
            continue
        got = {}

        def http(p):
            st, js = h(p)
            if st == 200:
                for k, b in (js.get('bars') or {}).items():
                    got.setdefault(k, []).extend(b)
            return st, js
        q = ALIAS.get(t, t)
        fetch_bars([q], '2016-01-01', '2023-12-31', adj, http, asof=a)
        if got.get(q):
            raw[(a, t)] = got[q]
            added += 1
    if added:
        pickle.dump(raw, open(path, 'wb'))
    return raw


def main():
    sd, snaps = PS.load_snapshots()
    ivs = PS.intervals(sd, snaps, LO, HI)
    all_, raw = complete(fetch(ivs, 'all'), ivs, 'all'), complete(fetch(ivs, 'raw'), ivs, 'raw')
    frames = {k: PS.bars_frame(b) for k, b in all_.items() if b}
    frames = {k: f[f.index <= '2023-12-31'] for k, f in frames.items()}
    rawf = {k: PS.bars_frame(b) for k, b in raw.items() if b}
    key_iv = defaultdict(list)
    missing = []
    for t, s, e in ivs:
        k = (str(asof_of(e)), t)
        if k in frames:
            key_iv[k].append((s, e))
        else:
            missing.append((t, str(s), str(e)))
    keys = sorted(key_iv)
    groups = PS.group_entities({k: frames[k]['c'] for k in keys})
    cal = pd.DatetimeIndex(sorted(set().union(*[frames[k].index for k in keys])))
    counts = pd.Series(0, cal)
    for k in keys:
        counts[frames[k].index] += 1
    cal = cal[(counts >= 100) & (cal >= str(LO)) & (cal <= str(HI))]
    dates = cal.values.astype('datetime64[D]')
    N, T = len(groups), len(dates)
    arr = {f: np.full((N, T), np.nan) for f in ('open', 'high', 'low', 'close', 'volume', 'raw_open', 'raw_close', 'raw_volume')}
    ent_iv, labels = [], []
    for i, g in enumerate(groups):
        rep = max(g, key=lambda k: len(frames[k]))
        f = frames[rep].reindex(cal)
        for c, name in zip('ohlcv', ('open', 'high', 'low', 'close', 'volume')):
            arr[name][i] = f[c].to_numpy()
        r = rawf.get(rep, pd.DataFrame(columns=list('ohlcv'))).reindex(cal)
        arr['raw_open'][i], arr['raw_close'][i], arr['raw_volume'][i] = r['o'].to_numpy(), r['c'].to_numpy(), r['v'].to_numpy()
        ent_iv.append([iv for k in g for iv in key_iv[k]])
        labels.append('|'.join(sorted({k[1] for k in g})))
    observed = np.isfinite(arr['raw_close']) & (arr['raw_close'] > 0) & np.isfinite(arr['raw_open']) & (arr['raw_open'] > 0)
    member = PS.member_from_intervals(ent_iv, dates)
    # Total-return index from RAW bars + plausibility-checked Alpaca adjustments (PREREG amendment A1)
    r_all, kind = PS.clean_returns(arr['raw_close'], arr['raw_open'], arr['close'], observed)
    for lab, day in MANUAL_NEUTRAL:
        i, t = labels.index(lab), int(np.searchsorted(dates, np.datetime64(day)))
        r_all[i, t], kind[i, t] = 0., 'neutralised'
    idx = np.exp(np.cumsum(np.log1p(r_all), 1))
    scale = np.where(observed, idx / arr['raw_close'], np.nan)
    arr['all_close'] = arr['close'].copy()
    for name, src in (('close', 'raw_close'), ('open', 'raw_open')):
        arr[name] = arr[src] * scale
    arr['open'] = np.where(kind == 'neutralised', arr['close'], arr['open'])   # value-neutral day: no gap, no intraday
    for name in ('high', 'low'):
        arr[name] = arr[name] / arr['all_close'] * arr['close']
    arr['volume'] = arr['raw_volume']
    obs_r = observed.copy()
    obs_r[:, 0] = False
    obs_r[:, 1:] &= observed[:, :-1]

    # survivor mapping (E2): a yfinance survivor maps to the entity carrying its ticker whose daily returns agree
    z = np.load(SURV)
    ident = z['identity_ok'].astype(bool)
    yd = z['dates'].astype('datetime64[D]')
    yp = z['features'][ident][:, :, -1].astype(np.float64)
    ym = z['mask'][ident].astype(bool)
    ytick, ysec = z['tickers'][ident], z['sectors'][ident].astype(int)
    common = np.intersect1d(dates, yd)
    ia, ib = np.searchsorted(dates, common), np.searchsorted(yd, common)
    ra = np.where(obs_r[:, ia], r_all[:, ia], np.nan)
    rb = np.full((len(ytick), len(common)), np.nan)
    yr = np.full(yp.shape, np.nan)
    yr[:, 1:] = np.where(ym[:, 1:] & ym[:, :-1], yp[:, 1:] / yp[:, :-1] - 1, np.nan)
    rb = yr[:, ib]
    lab_set = [set(l.split('|')) for l in labels]
    surv = np.full(N, -1)
    e2 = []
    for j, t in enumerate(ytick):
        t = str(t).replace('-', '.')
        best = None
        for i in [i for i in range(N) if t in lab_set[i]]:
            ok = np.isfinite(ra[i]) & np.isfinite(rb[j])
            if ok.sum() > 250:
                agree = float((np.abs(ra[i, ok] - rb[j, ok]) < 5e-3).mean())
                if best is None or agree > best[0]:
                    best = (agree, i, float(np.corrcoef(ra[i, ok], rb[j, ok])[0, 1]), int(ok.sum()))
        if best is not None and best[0] > .9 and surv[best[1]] < 0:
            surv[best[1]] = j
            e2.append((str(ytick[j]), best[0], best[2], best[3]))
    ref = np.where(surv >= 0, ysec[np.maximum(surv, 0)], -1)
    sectors = PS.sector_proxy(r_all, obs_r, ref, window=126)
    last = np.where(observed.any(1), T - 1 - observed[:, ::-1].argmax(1), -1)
    fail = PS.failure_flags(arr['close'], observed, last)
    np.savez_compressed(NPZ, dates=dates.astype(str), labels=np.array(labels), member=member, observed=observed,
                        sectors=sectors, gics_ref=ref, survivor_index=surv, failure=fail, last=last, kind=kind,
                        **{k: v.astype(np.float64) for k, v in arr.items()})

    # report
    big = []
    with np.errstate(invalid='ignore'):
        jump = np.abs(r_all) > .4
    for i, t in zip(*np.nonzero(jump & obs_r & member)):
        big.append(f'| {labels[i]} | {dates[t]} | {r_all[i, t]:+.2f} |')
    ended = [(labels[i], str(dates[last[i]]), bool(member[i, last[i]]), bool(fail[i]), float(arr['raw_close'][i, last[i]]))
             for i in range(N) if 0 <= last[i] < T - 1 and member[i, max(last[i] - 5, 0):last[i] + 1].any()]
    agree = (sectors[surv >= 0] == ref[surv >= 0]).mean()
    agr = np.array([x[1] for x in e2]); cor = np.array([x[2] for x in e2])
    n_neut = int(((kind == 'neutralised') & member).sum()); n_adj = int(((kind == 'adjusted') & member).sum())
    neut = [(labels[i], str(dates[t]), float(arr['raw_open'][i, t] / arr['raw_close'][i, t - 1] - 1)) for i, t in zip(*np.nonzero((kind == 'neutralised') & member))]
    multi = [l for l in labels if '|' in l]
    nomem = int((~member.any(1)).sum())
    L = ['# C1. Point-in-time S&P 500 panel from Alpaca: build and validation (2016-01-04 to 2023-12-29)', '',
         'Generated by `scripts/pit_build_panel.py` (rules in `PREREG.md`). Data stay local in `data/pit/` (git-ignored; '
         'Alpaca terms do not allow redistribution).', '',
         '## Coverage', '',
         f'- Membership intervals overlapping the window: **{len(ivs)}** (fja05680 `components.csv`).',
         f'- Intervals with Alpaca bars: **{len(ivs) - len(missing)}**; without: **{len(missing)}**: ' +
         ', '.join(f'{t} ({s}..{e})' for t, s, e in missing) + '.',
         f'- Entities after merging identical-return series: **{N}**; merged entities (renames/ticker changes): '
         f'**{len(multi)}**, e.g. ' + ', '.join(multi[:12]) + '.',
         f'- Entities never a member on a panel date: {nomem}.',
         f'- Trading days: **{T}**; member-days: {int(member.sum())}; member-days with a bar: '
         f'{int((member & observed).sum())} ({(member & observed).sum() / member.sum():.4f}).',
         f'- Members per day: min {member.sum(0).min()}, median {int(np.median(member.sum(0)))}, max {member.sum(0).max()}.', '',
         '## Return construction (PREREG amendment A1)', '',
         'Alpaca records several spin-offs as reverse splits (e.g. TGNA 2017-06-01 1:3, WRK 2016-05-16 1:6, UTX '
         '2020-04-03 1:2, DLPH 2017-12-05 1:3), so its adjusted series contain fake -70% to -89% days (and AIV an '
         'adjusted close of 0.83 vs raw 32). Returns are rebuilt from RAW bars (`pit_sp500.clean_returns`).',
         f'- Member-days using an Alpaca adjustment (dividends, consistent splits): {n_adj}.',
         f'- Member-days neutralised (implausible adjustment or unrecorded share-ratio gap; the day gets a zero '
         f'return): **{n_neut}**: ' + ', '.join(f'{a} {b} (raw gap {c:+.2f})' for a, b, c in neut) + '.', '',
         '## E2. Alpaca (cleaned) vs yfinance (survivor names, daily returns on common days)', '',
         f'- Survivors matched by ticker with > 90% of days agreeing within 0.5%: **{(surv >= 0).sum()} / {len(ytick)}** '
         '(the rest were not S&P 500 members in 2016-2023 or carry a different ticker).',
         f'- Share of days agreeing within 0.5%: median **{np.median(agr):.4f}**, min {agr.min():.4f}; daily-return '
         f'correlation median **{np.median(cor):.4f}**, 5th percentile {np.percentile(cor, 5):.4f}, min {cor.min():.4f}.',
         '- Lowest agreement: ' + ', '.join(f'{a} ({b:.3f}, corr {c:.3f})' for a, b, c, _ in sorted(e2, key=lambda x: x[1])[:8]) + '.', '',
         '## Sector rule', '',
         f'- One rule for all entities (return-correlation to leave-one-out GICS sector baskets, first 126 sessions). '
         f'Agreement with current GICS on matched survivors: **{agree:.3f}**.', '',
         '## Bars ending while a member (or within 5 sessions after removal): delistings', '',
         f'{len(ended)} entities. `failure` = last close < 50% of the close 60 sessions earlier.', '',
         '| Entity | Last bar | Member on last bar | Failure | Last close |', '|---|---|---|---|---|']
    L += [f'| {a} | {b} | {c} | {d} | {e:.2f} |' for a, b, c, d, e in ended]
    L += ['', '## E3. One-day moves above 40% on member days (inspect)', '', f'{len(big)} cases.', '',
          '| Entity | Date | Return |', '|---|---|---|'] + big
    DOC.write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L[:30]))


if __name__ == '__main__':
    main()
