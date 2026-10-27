"""Read-only original-file hash gate; a hash pass is not scientific raw replay."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath,PureWindowsPath


def verify_contract(contract,root):
    root=Path(root).resolve();files=contract.get('files',[])
    if not root.is_dir() or not files:raise ValueError('Missing root or empty contract')
    seen=set();rows=[]
    for item in files:
        name=item['path'].replace('\\','/');p=PurePosixPath(name)
        windows=PureWindowsPath(name);sha=item['sha256'];name=p.as_posix()
        if p.is_absolute() or windows.drive or '..' in p.parts or not p.parts or name.casefold() in seen:
            raise ValueError('Unsafe or duplicate contract path: '+name)
        if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('Invalid SHA256')
        seen.add(name.casefold());target=(root/name).resolve()
        if not target.is_relative_to(root):raise ValueError('Contract path resolves outside root')
        row=dict(path=name,expected_sha256=sha)
        if not target.is_file():row['state']='missing'
        else:
            h=hashlib.sha256()
            with target.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            row.update(actual_sha256=h.hexdigest(),state='matched' if h.hexdigest()==sha else 'mismatched')
        rows.append(row)
    counts={s:sum(r['state']==s for r in rows) for s in ['matched','missing','mismatched']}
    return dict(status='hashes_ready_not_raw_replayed' if counts['matched']==len(rows) else 'not_ready',
                root=str(root),required=len(rows),**counts,files=rows,
                scope='Original file integrity only; forecasts, IDs, timing and account replay require subsequent validation')


def write_receipt(path,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--contract',type=Path,default=Path('docs/rishi_mechanism_20261027/ARTIFACT_CONTRACT_V2.json'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve earlier verification report')
    raw=args.contract.read_bytes();result=verify_contract(json.loads(raw),args.root)
    result['contract_sha256']=hashlib.sha256(raw).hexdigest()
    write_receipt(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
    return 0 if result['status']=='hashes_ready_not_raw_replayed' else 2


if __name__=='__main__':raise SystemExit(main())
