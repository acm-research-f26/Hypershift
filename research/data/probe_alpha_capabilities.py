"""Small credential-safe capability check. Uses an ephemeral environment key."""
import hashlib
import json
import os
import time
from pathlib import Path
import requests


def main():
    key = os.environ['ALPHAVANTAGE_API_KEY']
    out = Path('data/nasdaq_monthly_pit/alpha_capabilities')
    out.mkdir(parents=True, exist_ok=True)
    probes = [dict(function='SHARES_OUTSTANDING', symbol='MSFT'),
              dict(function='TIME_SERIES_DAILY', symbol='ATVI', outputsize='full'),
              dict(function='TIME_SERIES_DAILY_ADJUSTED', symbol='ATVI', outputsize='full')]
    records = []
    for params in probes:
        name = params['function'] + '_' + params['symbol']
        path = out / (name + '.json')
        if path.exists():
            payload = json.loads(path.read_text(encoding='utf-8'))
        else:
            try:
                response = requests.get('https://www.alphavantage.co/query',
                                        params={**params, 'apikey': key}, timeout=30)
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError):
                print(name + ': request failed; credential-bearing exception suppressed')
                break
            # A burst warning is not evidence that the daily allowance is spent.
            if '1 request per second' in str(payload) and 'Information' in payload:
                time.sleep(2)
                try:
                    response = requests.get('https://www.alphavantage.co/query',
                                            params={**params, 'apikey': key}, timeout=30)
                    response.raise_for_status()
                    payload = response.json()
                except (requests.RequestException, ValueError):
                    print(name + ': retry failed; credential-bearing exception suppressed')
                    break
            # Even provider messages must not be allowed to reflect a credential.
            serialized = json.dumps(payload, indent=2).replace(key, '[REDACTED]')
            path.write_text(serialized, encoding='utf-8')
        record = dict(request=params, file=str(path),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      keys=list(payload))
        messages = {k: payload[k] for k in ('Information', 'Note', 'Error Message') if k in payload}
        if messages:
            record['messages'] = messages
            record['status'] = 'access_or_limit_message'
        else:
            record['status'] = 'data_response_needs_review'
            record['record_counts'] = {k: len(v) for k, v in payload.items() if isinstance(v, (list, dict))}
        records.append(record)
        print(json.dumps(record))
        if any('25 requests' in str(v).lower() for v in messages.values()):
            break
        time.sleep(2)
    (out / 'probe.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
