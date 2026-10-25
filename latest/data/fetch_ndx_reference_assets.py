"""Restore hash-pinned public research assets; never execute downloaded code.

python fetch_ndx_reference_assets.py
Only missing assets are fetched; existing bytes must match. The manifest pins
author commits, source hashes, the pure-Python einops wheel and JKP data vintage.
The public factor URL can change: a hash mismatch stops rather than accepting a
new data vintage. No credentials, private endpoints or package installation.
"""
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent


def restore(path,url,expected):
    path=path.resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Asset outside workspace')
    if path.exists():payload=path.read_bytes()
    else:
        response=requests.get(url,timeout=180);response.raise_for_status();payload=response.content
    if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Asset hash mismatch: '+str(path))
    if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)


def main():
    manifest=json.loads((ROOT/'config/ndx_reference_assets_20261024.json').read_text())
    for name,info in manifest['repositories'].items():
        folder=ROOT/'.cache/architecture_references_20261024'/name
        for relative,digest in info['files'].items():
            url=f'https://raw.githubusercontent.com/{info["repository"]}/{info["commit"]}/{relative}'
            restore(folder/relative,url,digest)
        metadata=folder/'REFERENCE.json';expected=json.dumps(info['reference_metadata'],indent=2)
        if metadata.exists() and metadata.read_text()!=expected:raise ValueError('Changed reference metadata')
        if not metadata.exists():metadata.write_text(expected,encoding='utf8')
    dependency=manifest['dependency']
    restore(ROOT/'.cache/reference_dependencies_20261024'/dependency['filename'],dependency['url'],dependency['digests']['sha256'])
    metadata=ROOT/'.cache/reference_dependencies_20261024/einops.json'
    if not metadata.exists():metadata.write_text(json.dumps(dependency,indent=2),encoding='utf8')
    folder=ROOT/'data/nasdaq_monthly_pit/jkp_themes_20261024'
    for info in manifest['factor_downloads']:restore(folder/info['file'],info['url'],info['sha256'])
    metadata=folder/'DOWNLOAD.json'
    if not metadata.exists():metadata.write_text(json.dumps(manifest['factor_downloads'],indent=2),encoding='utf8')
    print('All pinned public reference assets verified; no external scripts executed')


if __name__=='__main__':main()
