"""Price-only supplement for verified continuing historical index members.

Historical membership comes from the separate dated source, never this ticker
union. Recycled/vanished companies use identity-checked Tiingo histories.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd
import yfinance as yf


def main():
    root = Path('data/nasdaq_monthly_pit/yahoo_raw')
    root.mkdir(parents=True, exist_ok=True)
    members = pd.read_csv('data/nasdaq_monthly_pit/ndx_preflight/monthly_members_PROVISIONAL.csv')
    # Free-source amendment: no unrecoverable recycled CA/old FOX/Discovery histories.
    members = members[members.month >= '2019-07-01']
    aliases = {'CTRP':'TCOM','FB':'META','SYMC':'GEN','NLOK':'GEN','WLTW':'WTW'}
    native = {'ALXN','ANSS','ATVI','CELG','CERN','CTXS','MXIM','MYL','SGEN','SPLK','WBA','XLNX'}
    names = sorted({aliases.get(t,t) for t in members.ticker if t not in native} | {'SPY','QQQ','BMY','AZN'})
    path = root / 'download.pkl'
    if not path.exists():
        data = yf.download(names,start='2018-01-01',end='2026-01-01',auto_adjust=False,
                           actions=True,threads=4,progress=False,group_by='column')
        data.to_pickle(path)
    else:
        data = pd.read_pickle(path)
    # Retry transient misses serially. Never substitute another ticker.
    for symbol in names:
        if symbol in {'EA','WBD'} or data['Adj Close'][symbol].notna().any():
            continue
        retry=yf.download(symbol,start='2018-01-01',end='2026-01-01',auto_adjust=False,
                          actions=True,threads=False,progress=False,group_by='column')
        for field in data.columns.levels[0]:
            if field in retry.columns.levels[0]:data[(field,symbol)]=retry[field][symbol].reindex(data.index)
        data.to_pickle(path)
    for name,field in [('adj_close','Adj Close'),('close','Close'),('volume','Volume'),
                       ('dividends','Dividends'),('splits','Stock Splits')]:
        data[field].to_csv(root / (name + '.csv'))
    records=[]
    for symbol in names:
        series=data['Adj Close'][symbol].dropna()
        records.append(dict(symbol=symbol,rows=len(series),
                            first_date=str(series.index.min().date()) if len(series) else None,
                            last_date=str(series.index.max().date()) if len(series) else None))
    pd.DataFrame(records).to_csv(root/'coverage.csv',index=False)
    (root/'manifest.json').write_text(json.dumps(dict(status='prices_only_identity_and_action_review_pending',
         provider='Yahoo Finance via yfinance',version=yf.__version__,start='2018-01-01',end_exclusive='2026-01-01',
         symbols=names,aliases=aliases,native_tiingo_symbols=sorted(native),
         hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.csv')}),indent=2),encoding='utf-8')
    print('Downloaded',len(names),'price symbols;',sum(r['rows']==0 for r in records),'empty histories; historical membership remains separate.')


if __name__ == '__main__':main()
