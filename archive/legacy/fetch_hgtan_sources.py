"""Pin and download public HGTAN source for inspection, without executing it."""
import hashlib,json
from pathlib import Path
import requests

root=Path('data/hgtan_source'); root.mkdir(parents=True,exist_ok=True)
s=requests.Session()
response=s.get('https://api.github.com/repos/lixiaojieff/HGTAN/git/trees/main?recursive=1',timeout=60); response.raise_for_status()
tree=response.json(); sha=tree['sha']
(root/'tree.json').write_text(json.dumps(tree,indent=2))
manifest=[]
for entry in tree['tree']:
    name=entry['path']
    print(name,entry.get('size',''),flush=True)
    if entry['type']!='blob' or not (name.endswith(('.py','.md','.txt')) or 'license' in name.lower()): continue
    url=f'https://raw.githubusercontent.com/lixiaojieff/HGTAN/{sha}/{name}'
    r=s.get(url,timeout=90); r.raise_for_status()
    target=root/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(r.content)
    manifest.append({'path':name,'url':url,'sha256':hashlib.sha256(r.content).hexdigest()})
(root/'manifest.json').write_text(json.dumps({'commit':sha,'files':manifest},indent=2))
print('COMMIT',sha,flush=True)
