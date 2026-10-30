"""Verify or restore the published research outputs without replacing local files."""
import argparse,hashlib,json,zipfile
from pathlib import Path,PurePosixPath

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--restore',action='store_true');args=parser.parse_args();root=Path(__file__).resolve().parents[2];bundle=root/'data/historical_sharpe_exports_20261029';archive=bundle/'experiment_outputs.zip';manifest=json.loads((bundle/'manifest.json').read_text());assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
 with zipfile.ZipFile(archive) as z:
  assert set(z.namelist())==set(manifest['files']);pending=[]
  for name,record in manifest['files'].items():
   p=PurePosixPath(name)
   if p.is_absolute() or '..' in p.parts or p.parts[0]!='results':raise ValueError('unsafe archive path')
   target=root.joinpath(*p.parts).resolve();target.relative_to(root/'results');payload=z.read(name)
   if len(payload)!=record['bytes'] or hashlib.sha256(payload).hexdigest()!=record['sha256']:raise ValueError('archive payload changed: '+name)
   if target.exists():
    if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('existing file differs; preserve it: '+name)
   else:pending.append(name)
  if args.restore:
   for name in pending:
    target=root.joinpath(*PurePosixPath(name).parts);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(z.read(name))
 print(f'Verified {len(manifest["files"])} files; {len(pending)} missing; '+('restored missing files' if args.restore else 'verification only'))
if __name__=='__main__':main()
