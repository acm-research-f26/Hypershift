"""Small resumable free Alpha Vantage historical-listing capability probe.

Set ALPHAVANTAGE_API_KEY locally. This probes listings, not historical market
caps, issuer identity or delisting payouts. It never launches model training.
"""
from pathlib import Path
from io import StringIO
import argparse
import hashlib
import json
import os
import pandas as pd
import requests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('data/nasdaq_monthly_pit/listing_probe'))
    parser.add_argument('--dates',nargs='+',default=['2021-12-31','2023-12-29'])
    args=parser.parse_args()
    key=os.environ.get('ALPHAVANTAGE_API_KEY')
    if not key:
        parser.error('Set ALPHAVANTAGE_API_KEY locally first. Free signup: https://www.alphavantage.co/support/#api-key . No request sent.')
    args.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for date in args.dates:
        if pd.Timestamp(date) < pd.Timestamp('2010-01-02'):
            parser.error('Historical listing endpoint requires dates later than 2010-01-01')
        for state in ['active','delisted']:
            path=args.out/f'{date}_{state}.csv'
            if not path.exists():
                try:
                    response=requests.get('https://www.alphavantage.co/query',params=dict(
                        function='LISTING_STATUS',date=date,state=state,apikey=key),timeout=30)
                    response.raise_for_status()
                except requests.RequestException:
                    # Never print exception URLs: they can contain the API key.
                    raise SystemExit(f'Request failed for {date}/{state}; existing valid files retained.') from None
                text=response.text
                if text.lstrip().startswith('{'):
                    raise SystemExit(f'Provider returned a limit/access message for {date}/{state}; no CSV saved.')
                frame=pd.read_csv(StringIO(text),dtype=str)
                if not {'symbol','name','exchange','assetType','ipoDate','delistingDate'}.issubset(frame.columns):
                    raise SystemExit('Unrecognized listing schema; no CSV saved.')
                path.write_text(text,encoding='utf-8')
            frame=pd.read_csv(path,dtype=str)
            records.append(dict(date=date,state=state,rows=len(frame),
                nasdaq_rows=int(frame.exchange.eq('NASDAQ').sum()),file=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.out/'probe.json').write_text(json.dumps(dict(records=records,
        status='listings_only_not_training_ready',requests_max=2*len(args.dates),
        limitations=['Historical exchange correctness and issuer IDs still need cross-checking',
                     'No historical company caps or share-count publication dates supplied',
                     'No delisted stock price/settlement history supplied']),indent=2),encoding='utf-8')
    print('Listing capability probe complete; historical cap and terminal-return gates remain unresolved.')


if __name__=='__main__':main()
