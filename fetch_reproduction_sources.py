import hashlib
import json
from pathlib import Path
import requests

root=Path('data/think_reproduction_sources'); root.mkdir(parents=True,exist_ok=True)
sources={
 'think.pdf':'https://tylersnetwork.github.io/papers/icdm22-think.pdf',
 'train_nyse.py.txt':'https://raw.githubusercontent.com/midas-research/sthan-sr-aaai/05a23fca0f737f7a6d6924a38ce57e96c8aff04f/training/train_nyse.py',
 'evaluator.py.txt':'https://raw.githubusercontent.com/midas-research/sthan-sr-aaai/05a23fca0f737f7a6d6924a38ce57e96c8aff04f/training/evaluator.py',
 'think_issues.json':'https://api.github.com/repos/shivamag125/ICDM22-THINK/issues?state=all',
 'hnn_tree.json':'https://api.github.com/repos/mil-tokyo/hyperbolic_nn_plusplus/git/trees/main?recursive=1',
 'think_issue_1_comments.json':'https://api.github.com/repos/shivamag125/ICDM22-THINK/issues/1/comments',
 'think_issue_2_comments.json':'https://api.github.com/repos/shivamag125/ICDM22-THINK/issues/2/comments',
 'linear.py.txt':'https://raw.githubusercontent.com/mil-tokyo/hyperbolic_nn_plusplus/28737e22822562ac18d5ea03f8c3e3929e945a83/geoopt_plusplus/modules/linear.py',
 'math.py.txt':'https://raw.githubusercontent.com/mil-tokyo/hyperbolic_nn_plusplus/28737e22822562ac18d5ea03f8c3e3929e945a83/geoopt_plusplus/manifolds/stereographic/math.py',
 'multinomial_logistic_regression.py.txt':'https://raw.githubusercontent.com/mil-tokyo/hyperbolic_nn_plusplus/28737e22822562ac18d5ea03f8c3e3929e945a83/geoopt_plusplus/modules/multinomial_logistic_regression.py',
}
manifest=[]
for name,url in sources.items():
 r=requests.get(url,timeout=90)
 if name=='hnn_tree.json' and r.status_code==404:
  url=url.replace('/main?','/master?'); r=requests.get(url,timeout=90)
 r.raise_for_status()
 (root/name).write_bytes(r.content)
 manifest.append({'file':name,'url':url,'status':r.status_code,'sha256':hashlib.sha256(r.content).hexdigest()})
 print(name,r.status_code,flush=True)
(root/'sources.json').write_text(json.dumps(manifest,indent=2))
if (root/'hnn_tree.json').exists():
 tree=json.loads((root/'hnn_tree.json').read_text())
 print('HNN relevant paths',[x['path'] for x in tree.get('tree',[]) if any(t in x['path'] for t in ['poincare','mobius','hyperbolic'])])
