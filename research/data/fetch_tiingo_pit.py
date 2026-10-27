"""Resumable identity-first Tiingo price retrieval; tokens stay in the environment.

No response is accepted as a training input merely because its ticker matches.
Expected names must be supplied; recycled symbols fail rather than being spliced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
import requests


def fetch(symbol, expected_name, out, start='2016-01-01', end='2025-12-31'):
    token = os.environ['TIINGO_API_KEY']
    headers = {'Authorization': 'Token ' + token, 'Content-Type': 'application/json'}
    base = 'https://api.tiingo.com/tiingo/daily/' + symbol.lower()
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / (symbol + '_metadata.json')
    price_path = out / (symbol + '_prices.json')
    record = dict(symbol=symbol, expected_name=expected_name, start=start, end=end)
    try:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        else:
            response = requests.get(base, headers=headers, timeout=30)
            if response.status_code == 429:
                return dict(record, status='rate_limit_stop')
            if response.status_code != 200:
                return dict(record, status='metadata_access_failed', http_status=response.status_code)
            metadata = response.json()
            metadata_path.write_text(json.dumps(metadata, indent=2).replace(token, '[REDACTED]'), encoding='utf-8')
        name = str(metadata.get('name', ''))
        record['provider_name'] = name
        record['provider_start'] = metadata.get('startDate')
        record['provider_end'] = metadata.get('endDate')
        if expected_name.casefold() not in name.casefold():
            return dict(record, status='issuer_mismatch_rejected')
        if price_path.exists():
            payload = json.loads(price_path.read_text(encoding='utf-8'))
        else:
            response = requests.get(base + '/prices', headers=headers,
                                    params=dict(startDate=start, endDate=end), timeout=60)
            if response.status_code == 429:
                return dict(record, status='rate_limit_stop')
            if response.status_code != 200:
                return dict(record, status='price_access_failed', http_status=response.status_code)
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                return dict(record, status='empty_or_unrecognized_prices')
            price_path.write_text(json.dumps(payload, indent=2).replace(token, '[REDACTED]'), encoding='utf-8')
        frame = pd.DataFrame(payload)
        needed = {'date', 'close', 'adjClose', 'volume', 'divCash', 'splitFactor'}
        if not needed.issubset(frame.columns):
            return dict(record, status='unrecognized_price_schema')
        dates = pd.to_datetime(frame.date, utc=True)
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            return dict(record, status='invalid_dates')
        if ((frame.close <= 0) | (frame.adjClose <= 0)).any():
            return dict(record, status='invalid_prices')
        record.update(status='downloaded_not_corporate_action_certified', rows=len(frame),
                      first_date=str(dates.iloc[0].date()), last_date=str(dates.iloc[-1].date()),
                      last_close=float(frame.close.iloc[-1]),
                      dividend_events=int((frame.divCash != 0).sum()),
                      split_events=int((frame.splitFactor != 1).sum()),
                      hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [metadata_path, price_path]})
        return record
    except (requests.RequestException, ValueError, KeyError):
        # HTTP exception text may reflect credentials; never print it.
        return dict(record, status='request_or_schema_failure; details suppressed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', nargs='+', default=['ATVI', 'XLNX', 'CA'])
    parser.add_argument('--out', type=Path, default=Path('data/nasdaq_monthly_pit/tiingo_raw'))
    args = parser.parse_args()
    expected = {'ATVI': 'Activision', 'XLNX': 'Xilinx', 'CA': 'CA Inc',
                'ALXN': 'Alexion', 'ANSS': 'ANSYS', 'WBA': 'Walgreens', 'SPLK': 'Splunk',
                'SGEN': 'Seagen', 'CELG': 'Celgene', 'CERN': 'Cerner', 'CTXS': 'Citrix',
                'ESRX': 'Express Scripts', 'MXIM': 'Maxim', 'YHOO': 'Yahoo',
                'SHPG': 'Shire', 'DISH': 'DISH', 'MYL': 'Mylan', 'DISCA': 'Discovery',
                'EA': 'Electronic Arts', 'WBD': 'Warner'}
    if not os.environ.get('TIINGO_API_KEY'):
        parser.error('Set TIINGO_API_KEY in the local environment. No request sent.')
    if any(symbol not in expected for symbol in args.symbols):
        parser.error('No reviewed expected company name for one or more requested symbols.')
    records = []
    for symbol in args.symbols:
        record = fetch(symbol, expected[symbol], args.out)
        records.append(record)
        print(json.dumps(record))
        if record['status'] == 'rate_limit_stop':
            break
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'last_probe.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
