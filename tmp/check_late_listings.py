from pathlib import Path
from datetime import datetime,timezone
import json
import yfinance as yf

yf.set_tz_cache_location(str(Path('.cache/yfinance').resolve()))
rows=[]
for ticker in ['HONA','SPCX']:
    try:
        history=yf.Ticker(ticker).history(start='2026-01-01',end='2026-10-11',auto_adjust=False,actions=True,raise_errors=True)
        rows.append({'ticker':ticker,'first_2026_observation':str(history.index.min().date()) if len(history) else None,
                     'last_observation':str(history.index.max().date()) if len(history) else None,'observations':len(history)})
    except Exception as error:
        rows.append({'ticker':ticker,'error':str(error)})
result={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'source':'Yahoo Finance via yfinance','purpose':'Explain absence from 2018-2025; not used for model fitting or selection','entries':rows}
Path('data/largecap_20261010/late_listing_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
