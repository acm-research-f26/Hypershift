"""Fetch the selected original stock panels and inspect cited preprocessing."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import requests
import tarfile
from fetch_think_sources import ROOT, RSR

if __name__ == '__main__':
    # Read only small JSON/text members; the archive also contains multi-GB arrays.
    with tarfile.open(ROOT / 'relation.tar.gz') as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(('.json','.csv')):
                destination = ROOT / Path(member.name).name
                if not destination.exists():
                    destination.write_bytes(archive.extractfile(member).read())
    jobs = []
    for market in ['NYSE', 'NASDAQ']:
        folder = ROOT / market
        folder.mkdir(exist_ok=True)
        tickers = (ROOT / f'{market}_tickers_qualify_dr-0.98_min-5_smooth.csv').read_text().splitlines()
        for ticker in tickers:
            filename = f'{market}_{ticker}_1.csv'
            jobs.append((folder / filename, RSR + 'data/2013-01-01/' + filename))
    def download(job):
        path, url = job
        if not path.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        return {'file': str(path.relative_to(ROOT)), 'url': url,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = []
        for i, item in enumerate(executor.map(download, jobs), 1):
            records.append(item)
            if i % 200 == 0:
                print(f'Downloaded {i}/{len(jobs)} original stock series', flush=True)
    (ROOT / 'stock_sources.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    for filename in ['training/load_data_nyse.py','training/load_data_nasdaq.py','training/load_data_tse.py','training/train_hgat_tse.py']:
        url = 'https://raw.githubusercontent.com/midas-research/sthan-sr-aaai/05a23fca0f737f7a6d6924a38ce57e96c8aff04f/' + filename
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        (ROOT / (Path(filename).name + '.txt')).write_text(response.text, encoding='utf-8')
    print('Stock source downloads complete.', flush=True)
