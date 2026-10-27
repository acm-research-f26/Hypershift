"""Fetch deterministic 100-name subsets of the existing academic cohorts.

Selection uses ticker hashes, never returns, liquidity or future data. The parent
cohorts still have survivorship/selection bias; this does not cure that bias.
"""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import requests
import pandas as pd
import yfinance as yf

ROOT = Path('data/research_cohorts')
REV = 'cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d'


def download(item):
    market, ticker = item
    path = ROOT / 'raw' / f'{market}_{ticker}.csv'
    url = f'https://raw.githubusercontent.com/fulifeng/Temporal_Relational_Stock_Ranking/{REV}/data/google_finance/{market}_{ticker}_30Y.csv'
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    original = ROOT / 'raw_original' / path.name
    source = original if original.exists() else path
    return dict(market=market, ticker=ticker, url=url, source_file=str(source),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def main():
    ROOT.mkdir(exist_ok=True)
    (ROOT / 'raw').mkdir(exist_ok=True)
    selection = {}
    for market in ['NYSE', 'NASDAQ']:
        names = Path(f'config/{market}_source_cohort.txt').read_text().splitlines()
        selection[market] = sorted(sorted(names, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:100])
    (ROOT / 'selection.json').write_text(json.dumps(selection, indent=2))
    items = [(m, t) for m, names in selection.items() for t in names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        manifests = list(pool.map(download, items))
    (ROOT / 'sources.json').write_text(json.dumps(manifests, indent=2))
    print('Downloaded', len(manifests), 'original OHLCV histories.', flush=True)
    yf.set_tz_cache_location(str(Path('.cache/yfinance').resolve()))
    spy = yf.download(['SPY', 'QQQ'], start='2012-01-01', end='2018-01-01', auto_adjust=True,
                      progress=False, threads=False, multi_level_index=True)
    if spy is None or spy.empty:
        raise RuntimeError('No benchmark returned')
    spy['Close'].to_csv(ROOT / 'benchmark.csv')
    reference = yf.download(['EXPO', 'ZTR'], start='2012-01-01', end='2018-01-01',
                            auto_adjust=False, actions=True, progress=False, threads=False)
    if reference is None or reference.empty:
        raise RuntimeError('No corporate-action crosscheck returned')
    reference.to_csv(ROOT / 'corporate_action_reference.csv')
    from repair_research_corporate_actions import main as repair
    repair()


if __name__ == '__main__':
    main()
