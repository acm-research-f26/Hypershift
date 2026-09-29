"""Fetch cited public source data with URLs, commit pins and SHA256 provenance."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import requests

ROOT = Path('data/think_audit')
ROOT.mkdir(parents=True, exist_ok=True)
PGT = 'https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/ea40a6a396b6688a94d7482d9d5fd288eaa2cb3b/'
RSR = 'https://raw.githubusercontent.com/fulifeng/Temporal_Relational_Stock_Ranking/cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d/'
SOURCES = {'chickenpox.json': PGT + 'dataset/chickenpox.json',
           'twitter_tennis_rg17.json': PGT + 'dataset/twitter_tennis_rg17.json',
           'twitter_tennis_uo17.json': PGT + 'dataset/twitter_tennis_uo17.json',
           'windmill_output.json': 'https://anl.app.box.com/shared/static/wgwb75lt3ty3pv5a15y9bilx1mjhcq59',
           'relation.tar.gz': RSR + 'data/relation.tar.gz',
           'RSR_load_data.py.txt': RSR + 'training/load_data.py',
           'RSR_wikidata.py.txt': RSR + 'preprocess/wikidata.py',
           'STHAN_model.py.txt': 'https://raw.githubusercontent.com/midas-research/sthan-sr-aaai/05a23fca0f737f7a6d6924a38ce57e96c8aff04f/model.py',
           'hats_README.md': 'https://raw.githubusercontent.com/dmis-lab/hats/master/README.md'}
for market in ['NYSE','NASDAQ']:
    for suffix in ['tickers_qualify_dr-0.98_min-5_smooth.csv','aver_line_dates.csv','wiki.csv']:
        SOURCES[f'{market}_{suffix}'] = RSR + f'data/{market}_{suffix}'

def fetch(item):
    name, url = item
    target = ROOT / name
    result = {'file': name, 'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat()}
    try:
        if not target.exists():
            response = requests.get(url, timeout=120)
            result['http_status'] = response.status_code
            response.raise_for_status()
            target.write_bytes(response.content)
        data = target.read_bytes()
        result.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:
        result['error'] = str(exc)
    print(json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch, SOURCES.items()))
    (ROOT / 'sources.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
