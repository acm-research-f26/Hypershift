"""Seal audited local experiment artifacts without producing documents."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(name):
    folder=ROOT/'runs'/name
    audit=json.loads((folder/'AUDIT.json').read_text())
    progress=json.loads((folder/'PROGRESS.json').read_text())
    if audit['status']!='passed' or progress['stage']!='completed_audited':raise ValueError('Audit incomplete')
    for lockname in ['INPUT_CODE_LOCK.json','EVALUATION_SOURCE_LOCK.json']:
        for path,h in json.loads((folder/lockname).read_text()).items():
            if digest(Path(path))!=h:raise ValueError('Frozen source/input changed: '+path)
    for path,h in json.loads((folder/'FITS_COMPLETE_LOCK.json').read_text())['hashes'].items():
        if digest(Path(path))!=h:raise ValueError('Fit artifact changed: '+path)
    files=[p for p in folder.iterdir() if p.is_file() and p.name!='FINAL_PUBLICATION.json']
    for name in ['source_snapshot','complete_source_snapshot','resume_source_snapshot','parity']:
        files.extend(p for p in (folder/name).rglob('*') if p.is_file())
    hashes={str(p.relative_to(folder)).replace('\\','/'):digest(p) for p in sorted(files)}
    target=folder/'FINAL_PUBLICATION.json'
    if target.exists():
        if json.loads(target.read_text())['artifact_hashes']!=hashes:raise ValueError('Sealed output changed')
    else:target.write_text(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='audited_and_sealed',artifact_hashes=hashes),indent=2),encoding='utf8')
    print('Sealed',len(hashes),'artifacts in',folder.name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name',choices=['ndx_loss_optimizer_20261024','ndx_reference_networks_20261024'])
    seal(parser.parse_args().name)
