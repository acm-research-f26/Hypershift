"""Freeze three 100-name draws per exchange before fetching public histories."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import yfinance as yf
from fetch_recent_cohorts import FIELDS

ROOT = Path('data/random_trials_20261010')

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'manifest.json').exists():
        print('Immutable completed snapshot exists'); return
    plan = json.loads(Path('config/random_trials_20261010.json').read_text())
    selection = {}
    sources = {}
    for market in ['NYSE', 'NASDAQ']:
        source = Path(f'config/{market}_source_cohort.txt')
        pool = sorted(set(source.read_text().splitlines()))
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        for seed in plan['sampling_seeds']:
            selection[f'{market}_draw{seed}'] = sorted(np.random.default_rng(seed).choice(pool, 100, replace=False).tolist())
    lock = ROOT/'selection.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == selection
    else:
        lock.write_text(json.dumps(selection, indent=2))
    (ROOT/'PLAN.json').write_text(json.dumps(plan, indent=2))
    symbols = sorted(set(sum(selection.values(), [])+['SPY','QQQ']))
    rawdir = ROOT/'raw'; rawdir.mkdir(exist_ok=True)
    for symbol in symbols:
        for old in ['recent_cohorts', 'largecap_20261010']:
            cached = Path('data')/old/'raw'/f'{symbol}.csv'
            if cached.exists() and not (rawdir/f'{symbol}.csv').exists():
                shutil.copyfile(cached, rawdir/f'{symbol}.csv')
    needed = [s for s in symbols if not (rawdir/f'{s}.csv').exists()
              or pd.read_csv(rawdir/f'{s}.csv')['Adj Close'].notna().sum() == 0]
    yf.set_tz_cache_location(str(Path('.cache/yfinance').resolve()))
    for start in range(0, len(needed), 15):
        batch = needed[start:start+15]
        aliases = {s:s.replace('.', '-') for s in batch}
        raw = yf.download(list(aliases.values()), start='2018-01-01', end='2026-01-01',
                          auto_adjust=False, actions=True, progress=False, threads=6, timeout=20, multi_level_index=True)
        for s in batch:
            frame = pd.DataFrame(index=raw.index)
            for field in FIELDS:
                key = (field, aliases[s])
                frame[field] = raw[key] if key in raw.columns else np.nan
            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            frame.to_csv(rawdir/f'{s}.csv', index_label='Date')
        print(f'Fetched {min(start+15,len(needed))}/{len(needed)} new symbols', flush=True)
    if needed and not any(pd.read_csv(rawdir/f'{s}.csv')['Adj Close'].notna().any() for s in needed):
        raise RuntimeError('No new histories downloaded: possible infrastructure failure. Do not finalize a missing-data snapshot.')
    spy = pd.read_csv(rawdir/'SPY.csv', index_col=0, parse_dates=True)
    dates = spy.loc['2018-01-01':'2025-12-31'].index
    assert len(dates) >= 1900
    known = json.loads(Path('config/recent_identity_exclusions.json').read_text())
    fields = {f:{} for f in ['Adj Close','Close','Volume']}
    coverage = []
    for s in symbols:
        frame = pd.read_csv(rawdir/f'{s}.csv', index_col=0, parse_dates=True).reindex(dates)
        valid = frame['Adj Close'].notna() & (frame['Adj Close'] > 0)
        first = dates[valid][0] if valid.any() else None
        reason = ('known issuer mismatch' if s in known else
                  'old-cohort issuer with no 2018 history; identity unresolved' if first is not None and first.year > 2018 else '')
        if reason:
            frame[:] = np.nan
        row = dict(ticker=s, first=str(first.date()) if first is not None else None,
                   rejection=reason, observations=int(frame['Adj Close'].notna().sum()))
        coverage.append(row)
        for f in fields:
            fields[f][s] = frame[f]
    for f, columns in fields.items():
        pd.DataFrame(columns, index=dates).to_csv(ROOT/(f.lower().replace(' ','_')+'.csv'), index_label='Date')
    pd.DataFrame(coverage).to_csv(ROOT/'coverage.csv', index=False)
    manifest = dict(selection=selection, source_hashes=sources, provider='Yahoo Finance via yfinance',
                    limitations=plan['limitations'], hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.csv')})
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Snapshot complete', flush=True)

if __name__ == '__main__':
    main()
