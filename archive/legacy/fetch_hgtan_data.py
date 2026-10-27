"""Download author-published CSV data; parse as data, never execute."""
import hashlib,json,time
from pathlib import Path
import requests
root=Path('data/hgtan_source'); tree=json.loads((root/'tree.json').read_text()); sha=tree['sha']
manifest=[]; session=requests.Session()
items=[(x['path'],f'https://raw.githubusercontent.com/lixiaojieff/HGTAN/{sha}/'+x['path']) for x in tree['tree'] if x['path'].endswith('.csv') and 'result' not in x['path']]
items.insert(0,('data/result(758)_label.csv','https://drive.usercontent.google.com/download?id=1T7OHfe8lOrv_fED545yArLDVj8sLVqP2&export=download&confirm=t'))
for name,url in items:
    target=root/name; target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size>1000:
        body=target.read_bytes(); manifest.append({'path':name,'url':url,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}); continue
    r=session.get(url,stream=True,timeout=(30,120)); r.raise_for_status()
    if 'text/html' in r.headers.get('content-type',''): raise RuntimeError('Expected CSV, received HTML: '+name)
    digest=hashlib.sha256(); count=0; start=time.monotonic()
    temp=target.with_suffix('.download')
    with temp.open('wb') as f:
        for chunk in r.iter_content(1024*1024):
            if count==0: print(name,'FIRST',chunk[:250],flush=True)
            f.write(chunk); digest.update(chunk); count+=len(chunk)
            if count%(32*1024*1024)<1024*1024: print(name,count,'bytes',flush=True)
    temp.replace(target)
    manifest.append({'path':name,'url':url,'bytes':count,'sha256':digest.hexdigest()})
    print(name,count,'seconds',round(time.monotonic()-start,1),flush=True)
(root/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
