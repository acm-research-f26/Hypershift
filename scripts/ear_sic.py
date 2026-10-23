"""Dated industry codes for the EAR test: the SIC recorded in the EDGAR header of each CIK's earliest Item 2.02 filing
in the window (as filed at that date, not today's code).

  SEC_USER_AGENT=... python scripts/ear_sic.py   -> data/edgar/sic_hist.json {cik: [sic, filing_date]}
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, 'src')
from hypershift.data import edgar as ED  # noqa: E402

EVT = Path('data/edgar/ear_events.json')
OUT = Path('data/edgar/sic_hist.json')


def main():
    sec = ED.SecClient()
    ev = json.loads(EVT.read_text())
    first = {}
    for evs in ev.values():
        for e in evs:
            c = e['cik']
            if c not in first or e['filingDate'] < first[c]['filingDate']:
                first[c] = e
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for c, e in sorted(first.items()):
        if str(c) in out:
            continue
        acc = e['accession']
        try:
            x = sec.get(f"https://www.sec.gov/Archives/edgar/data/{c}/{acc.replace('-', '')}/{acc}-index-headers.html")
            m = re.search(r'STANDARD INDUSTRIAL CLASSIFICATION:[^\[]*\[(\d{4})\]', x.decode('latin-1'))
            out[str(c)] = [int(m.group(1)) if m else None, e['filingDate']]
        except RuntimeError as ex:
            out[str(c)] = [None, e['filingDate']]
            print('fail', c, ex, flush=True)
    OUT.write_text(json.dumps(out, indent=0))
    print(len(out), 'CIKs;', sum(v[0] is None for v in out.values()), 'without SIC', flush=True)


if __name__ == '__main__':
    main()
