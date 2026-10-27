from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import json

root = Path('data/largecap_20261010/sources')
root.mkdir(parents=True, exist_ok=True)
urls = {
    'nasdaq100_api': 'https://api.nasdaq.com/api/nasdaq100',
    'nyse_stockanalysis': 'https://stockanalysis.com/list/nyse-stocks/',
    'nasdaq100_slickcharts': 'https://www.slickcharts.com/nasdaq100',
}
def fetch(item):
    name, url = item
    try:
        result = requests.get(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json,text/html,*/*', 'Referer': 'https://www.nasdaq.com/'}, timeout=35)
        (root / f'{name}.txt').write_text(result.text, encoding='utf-8')
        return {'source':name, 'url':url,'status':result.status_code,'bytes':len(result.content),'preview':result.text[:220]}
    except Exception as error:
        return {'source':name,'error':str(error)}
with ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(fetch, urls.items()):
        print(json.dumps(result), flush=True)
