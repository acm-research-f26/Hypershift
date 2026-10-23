"""Fetch the first 400 bytes of a few known primary documents to see whether they start with an XML declaration.

Used once by the overnight review (OVERNIGHT_RISK_REVIEW.md, issue C1). Max 4 requests, 2.5 s apart.
User-Agent from SEC_USER_AGENT (never printed).
"""
import os
import sys
import time

import requests

URLS = sys.argv[1:]
ua = os.environ['SEC_USER_AGENT']
for k, u in enumerate(URLS[:4]):
    if k:
        time.sleep(2.5)
    r = requests.get(u, headers={'User-Agent': ua, 'Range': 'bytes=0-399'}, timeout=60)
    head = r.content[:400]
    print(f'--- {u}\nstatus {r.status_code}, bytes {len(r.content)}\n{head[:200]!r}\n')
