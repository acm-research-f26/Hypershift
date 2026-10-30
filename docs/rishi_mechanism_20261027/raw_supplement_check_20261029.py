"""Check published Alpaca chunks against their original sidecars without loading pickles."""
from pathlib import Path
import gzip
import hashlib
import json
from collections import Counter

root = Path('data/nasdaq_monthly_pit/alpaca_15min/sip_raw_20180101_20260101')
sidecars = sorted((root / 'bars').glob('*/*.json'))
errors = []
statuses = Counter()
rows = 0
for path in sidecars:
    metadata = json.loads(path.read_text(encoding='utf-8'))
    chunk = path.with_suffix('.csv.gz')
    digest = hashlib.sha256(chunk.read_bytes()).hexdigest()
    with gzip.open(chunk, 'rt', encoding='utf-8') as source:
        count = max(0, sum(1 for _ in source) - 1)
    rows += count
    statuses[metadata['status']] += 1
    if digest != metadata['sha256'] or count != metadata['rows']:
        errors.append(str(path))
print(f'sidecars={len(sidecars)} rows={rows} statuses={dict(statuses)} errors={len(errors)}')
if errors:
    print('\n'.join(errors))
    raise SystemExit(1)
