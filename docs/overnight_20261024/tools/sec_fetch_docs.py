"""Fetch a few named primary documents for the extraction-robustness diagnostic (OVERNIGHT_RISK_REVIEW.md).

  python sec_fetch_docs.py OUTDIR URL [URL ...]     (max 4 URLs, 2.5 s apart; User-Agent from SEC_USER_AGENT)
Raw documents go to OUTDIR (outside the repository); nothing is written under data/.
"""
import os
import sys
import time
from pathlib import Path

import requests

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
ua = os.environ['SEC_USER_AGENT']
for k, u in enumerate(sys.argv[2:6]):
    if k:
        time.sleep(2.5)
    r = requests.get(u, headers={'User-Agent': ua}, timeout=120)
    name = u.rstrip('/').split('/')[-1]
    if r.status_code == 200:
        (out / name).write_bytes(r.content)
    print(u, r.status_code, len(r.content))
