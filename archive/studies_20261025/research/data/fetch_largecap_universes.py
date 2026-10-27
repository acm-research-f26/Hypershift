"""Freeze current large-company lists and fetch an immutable 2018-2025 panel.

Membership is a current snapshot, not historical index membership. No missing
companies are replaced based on price availability or subsequent returns.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path('data/largecap_20261010')
NAMES = ['NYSE_top100_20261010', 'NASDAQ100_20261010']
FIELDS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows_from_web(path):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = re.sub(r'^L\d+:\s*', '', line)
        line = re.sub(r'\ue200cite\ue202[^\u2020]+\u2020([^\ue201]+)\ue201', r'\1', line)
        parts = [part.strip() for part in line.split('|')]
        if len(parts) == 7 and parts[0].isdigit():
            rows.append(parts)
    return rows

def freeze_lists():
    if (ROOT/'selection.json').exists():
        return json.loads((ROOT/'selection.json').read_text())
    nasdaq_rows = rows_from_web(ROOT/'sources/web_0.txt')
    nyse_rows = rows_from_web(ROOT/'sources/web_1.txt')
    assert len(nasdaq_rows) == 101, len(nasdaq_rows)
    assert len(nyse_rows) >= 110, len(nyse_rows)
    selected, members = [], []
    seen = set()
    for rank, ticker, company, cap, price, change, revenue in nyse_rows:
        if '.PR' in ticker or company in seen:
            continue
        seen.add(company)
        # Berkshire's more liquid Class B is one security for one company.
        ticker = 'BRK.B' if ticker == 'BRK.A' else ticker
        selected.append(ticker)
        members.append(dict(dataset=NAMES[0], rank=len(selected), source_rank=int(rank), ticker=ticker,
                            company=company, snapshot_market_cap=cap, selection='Largest company market cap; one common/ADR security per company'))
        if len(selected) == 100:
            break
    assert len(selected) == 100 and len(set(selected)) == 100
    nasdaq = []
    for rank, company, ticker, weight, price, change, percent in nasdaq_rows:
        assert re.fullmatch(r'[A-Z.]+', ticker), ticker
        nasdaq.append(ticker)
        members.append(dict(dataset=NAMES[1], rank=int(rank), source_rank=int(rank), ticker=ticker,
                            company=company, snapshot_market_cap=None, selection='Full Nasdaq-100 list; both Alphabet share classes retained'))
    selection = {NAMES[0]:sorted(selected), NAMES[1]:sorted(nasdaq)}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/'selection.json').write_text(json.dumps(selection, indent=2))
    pd.DataFrame(members).to_csv(ROOT/'members.csv', index=False)
    provenance = {
        'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'membership_basis':'Current lists retrieved 2026-10-10; not point-in-time 2022-2025 membership',
        'nyse_source':'https://stockanalysis.com/list/nyse-stocks/',
        'nasdaq_source':'https://www.slickcharts.com/nasdaq100',
        'nasdaq_definition':'https://indexes.nasdaq.com/docs/Methodology_NDX.pdf',
        'nasdaq_company_count':100, 'nasdaq_security_count':101,
        'nyse_company_count':100, 'nyse_security_count':100,
        'nyse_rule':'Company market cap descending, exclude preferred share rows and duplicate company names; Berkshire Class B; foreign ADRs included.',
        'no_replacement_rule':'Keep every selected ticker even if its 2018-2025 history is unavailable.',
        'source_hashes':{str(path.relative_to(ROOT)):digest(path) for path in [ROOT/'sources/web_0.txt',ROOT/'sources/web_1.txt']},
    }
    (ROOT/'membership_provenance.json').write_text(json.dumps(provenance, indent=2))
    print('Frozen NYSE 100 companies and Nasdaq-100 100 companies / 101 securities.', flush=True)
    return selection

def download():
    if (ROOT/'manifest.json').exists():
        raise SystemExit('Immutable completed snapshot already exists.')
    selection = freeze_lists()
    symbols = sorted(set(sum(selection.values(), []) + ['SPY', 'QQQ']))
    aliases = {ticker:ticker.replace('.', '-') for ticker in symbols}
    (ROOT/'aliases.json').write_text(json.dumps(aliases, indent=2))
    cache = ROOT/'raw'
    cache.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(Path('.cache/yfinance').resolve()))
    for start in range(0, len(symbols), 12):
        batch = symbols[start:start+12]
        needed = [ticker for ticker in batch if not (cache/f'{ticker}.csv').exists()]
        if needed:
            raw = yf.download([aliases[ticker] for ticker in needed], start='2018-01-01', end='2026-01-01',
                              auto_adjust=False, actions=True, progress=False, threads=4, timeout=30, multi_level_index=True)
            for ticker in needed:
                frame = pd.DataFrame(index=raw.index)
                for field in FIELDS:
                    key = (field, aliases[ticker])
                    frame[field] = raw[key] if key in raw.columns else np.nan
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                frame.to_csv(cache/f'{ticker}.csv', index_label='Date')
        print(f'Fetched/cached {min(start+12,len(symbols))}/{len(symbols)} securities', flush=True)
    # Retry empty responses individually so transient batch failures are explicit.
    for ticker in symbols:
        path = cache/f'{ticker}.csv'
        if pd.read_csv(path)['Adj Close'].notna().sum() == 0:
            frame = yf.Ticker(aliases[ticker]).history(start='2018-01-01', end='2026-01-01', auto_adjust=False, actions=True, raise_errors=False)
            if not frame.empty and 'Adj Close' in frame:
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                frame.reindex(columns=FIELDS).to_csv(path, index_label='Date')
                print(f'Retry recovered {ticker}', flush=True)
    spy = pd.read_csv(cache/'SPY.csv', index_col=0, parse_dates=True)
    dates = spy.loc[spy['Adj Close'].notna()].index
    assert len(dates) >= 1900 and dates.max() >= pd.Timestamp('2025-12-30'), 'Incomplete benchmark calendar'
    all_fields = {field:{} for field in FIELDS}
    coverage, missing = [], []
    for ticker in symbols:
        frame = pd.read_csv(cache/f'{ticker}.csv', index_col=0, parse_dates=True).reindex(dates)
        assert not frame.index.has_duplicates
        valid = frame['Adj Close'].notna() & (frame['Adj Close'] > 0)
        if not valid.any():
            missing.append(ticker)
        for dataset, names in selection.items():
            if ticker not in names:
                continue
            row = dict(dataset=dataset, ticker=ticker, provider_symbol=aliases[ticker],
                       first=str(dates[valid][0].date()) if valid.any() else None,
                       last=str(dates[valid][-1].date()) if valid.any() else None)
            for year in range(2018, 2026):
                row[f'observed_{year}'] = int(valid[dates.year == year].sum())
            coverage.append(row)
        for field in FIELDS:
            all_fields[field][ticker] = frame[field]
    for field, columns in all_fields.items():
        pd.DataFrame(columns,index=dates).to_csv(ROOT/(field.lower().replace(' ','_')+'.csv'),index_label='Date')
    pd.DataFrame(coverage).to_csv(ROOT/'coverage.csv',index=False)
    limitations = [
        'Current 2026 company lists are applied retrospectively to 2018-2025. This embeds future membership/market-cap selection and survivorship bias.',
        'Not a historical Nasdaq-100 membership backtest or a historically reconstituted NYSE top-100 portfolio.',
        'Recent IPOs and spin-offs remain missing before their listing; no fabricated history or substitution with predecessor companies.',
        'NASDAQ retains 101 securities for 100 companies, including both Alphabet classes. NYSE retains 100 distinct companies, one share class each, including ADRs.',
        'Yahoo adjusted histories may be revised for corporate actions; no certified delisting or point-in-time feed.',
        'Historical exchange transfers are not reconstructed; group labels describe the current snapshot.',
        '2022-2025 outcomes were inspected in prior research; all new results are exploratory.',
    ]
    manifest = dict(provider='Yahoo Finance via yfinance', version=yf.__version__, downloaded_utc=datetime.now(timezone.utc).isoformat(),
                    start='2018-01-01',end_exclusive='2026-01-01', selection=selection,no_history=missing,limitations=limitations,
                    membership_provenance=json.loads((ROOT/'membership_provenance.json').read_text()),
                    hashes={str(path.relative_to(ROOT)):digest(path) for path in ROOT.glob('*.csv')})
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    counts = pd.DataFrame(coverage).groupby('dataset').agg(selected=('ticker','size'),recovered=('first','count'))
    print(counts.to_string(), flush=True)
    print('No returned 2018-2025 history:',missing,flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only',action='store_true')
    args = parser.parse_args()
    freeze_lists() if args.freeze_only else download()
