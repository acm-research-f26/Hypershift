"""Continue the currently authorized local run into its results analysis."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path('runs/largecap_20261010')
while not all((root/name/'completion.json').exists() for name in ['five_session','daily']):
    try:
        status=json.loads((root/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        status={}
    if status.get('stage')=='failed':
        raise SystemExit('Training failed; inspect the training log before analyzing incomplete results.')
    time.sleep(10)
print('Training finished; starting the results analysis.',flush=True)
with (root/'reporting.log').open('a',encoding='utf-8') as log:
    process=subprocess.Popen([sys.executable,'-u','analyze_largecap_retraining.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,encoding='utf-8',errors='replace')
    for line in process.stdout:
        log.write(line);log.flush();print(line,end='',flush=True)
    raise SystemExit(process.wait())
