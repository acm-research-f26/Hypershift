"""Check public dataset files against their pre-publication repository revisions."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT=Path('data/think_audit')
REPO='benedekrozemberczki/pytorch_geometric_temporal'
PATHS=['dataset/chickenpox.json','dataset/twitter_tennis_rg17.json','dataset/twitter_tennis_uo17.json',
       'torch_geometric_temporal/dataset/windmilllarge.py']

def check(path):
    response=requests.get(f'https://api.github.com/repos/{REPO}/commits',
                          params={'path':path,'until':'2022-10-01T00:00:00Z','per_page':1},timeout=30)
    response.raise_for_status()
    commits=response.json()
    result={'path':path,'before':'2022-10-01'}
    if not commits:
        return {**result,'status':'no historical revision found'}
    sha=commits[0]['sha']
    url=f'https://raw.githubusercontent.com/{REPO}/{sha}/{path}'
    download=requests.get(url,timeout=60); download.raise_for_status()
    original=ROOT/Path(path).name
    snapshot=ROOT/('pre2022_'+Path(path).name.replace('.py','.py.txt'))
    snapshot.write_bytes(download.content)
    result.update(commit=sha,url=url,sha256=hashlib.sha256(download.content).hexdigest())
    if original.exists():
        result['identical_to_downloaded_current_file']=original.read_bytes()==download.content
    if path.endswith('.py'):
        result['loader_urls']=[line.strip() for line in download.text.splitlines() if 'http' in line]
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as executor:
        results=list(executor.map(check,PATHS))
    (ROOT/'historical_revision_check.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
