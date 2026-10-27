"""Two source-verified data corrections, specified before test metrics were scored.

Original downloads remain in raw_original/. No ticker is removed and no realized
trading result is used to choose a correction. This is not a complete CA audit.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd


def main():
    root = Path('data/research_cohorts')
    original = root/'raw_original'
    original.mkdir(exist_ok=True)
    notes = []
    for name in ['NASDAQ_EXPO', 'NYSE_ZTR']:
        path = root/'raw'/f'{name}.csv'
        backup = original/path.name
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        f = pd.read_csv(backup,index_col=0)
        if name=='NASDAQ_EXPO':
            # Independent Yahoo closes match the source before this date and half
            # its values thereafter; volume uses the inverse scale. Close*volume
            # is preserved. The +98% jump is an adjustment error, not alpha.
            mask = f.index.str[:10]>='2015-06-04'
            f.loc[mask,['Open','High','Low','Close']] /= 2
            f.loc[mask,'Volume'] *= 2
            reason = 'Inconsistent split normalization from 2015-06-04: OHLC /2 and volume *2; dollar volume preserved.'
            source = 'https://www.sec.gov/Archives/edgar/data/851520/000114420415034077/v411916_ex99-1.htm'
        else:
            # One erroneous pre-reverse-split bar, independently cross-checked.
            ref = pd.read_csv(root/'corporate_action_reference.csv',header=[0,1],index_col=0)
            mask = f.index.str[:10]=='2012-06-26'
            for col in ['Open','High','Low','Close','Volume']:
                f.loc[mask,col] = float(ref.loc['2012-06-26',(col,'ZTR')])
            reason = 'Replace one inconsistent 2012-06-26 OHLCV bar using independent split-adjusted Yahoo history.'
            source = 'https://ir.virtus.com/news/news-details/2012/Zweig-Fund-And-Zweig-Total-Return-Fund-Announce-Date-Of-Reverse-Stock-Split/default.aspx'
        f.to_csv(path)
        notes.append(dict(file=str(path),original_file=str(backup),
                          original_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
                          corrected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          reason=reason,corporate_action_source=source,
                          crosscheck='corporate_action_reference.csv: Yahoo adjusted-for-splits Close and Volume'))
    (root/'corrections.json').write_text(json.dumps(notes,indent=2))
    print('Saved two verified corporate-action corrections; originals preserved.')


if __name__=='__main__':
    main()
