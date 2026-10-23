"""Download primary 10-K/10-Q documents for mapped CIKs, reduce them to term counts, and compute the frozen
year-over-year cosine similarity (PREREG). Resumable; filings accepted after 2023-12-31 are never requested.

  SEC_USER_AGENT=... python scripts/lazy_signals.py   -> data/edgar/tokens/*.json.gz, data/edgar/signals.csv
Stores term counts plus provenance (URL, sha256 of the raw document, retrieval time), not the raw HTML.
"""
import csv
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
from hypershift.data import edgar as ED  # noqa: E402

TOK = Path('data/edgar/tokens')
SIG = Path('data/edgar/signals.csv')
LO, HI, FIRST = np.datetime64('2015-01-01'), np.datetime64('2023-12-31T23:59:59'), np.datetime64('2016-01-01')


def parse(args):
    acc, cik, url, sha, body = args
    try:
        tok = ED.extract_tokens(body)
        err = ''
    except Exception as e:     # recorded, never silently dropped
        tok, err = {}, repr(e)[:200]
    rec = {'accession': acc, 'cik': cik, 'url': url, 'sha256': sha, 'retrieved': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'n_tokens': int(sum(tok.values())), 'error': err, 'parser': 2, 'counts': dict(tok)}
    (TOK / f'{acc}.json.gz').write_bytes(gzip.compress(json.dumps(rec).encode()))
    return acc, rec['n_tokens'], err


def load(acc):
    return json.loads(gzip.decompress((TOK / f'{acc}.json.gz').read_bytes()))


def main():
    sec = ED.SecClient()
    cmap = json.loads(Path('data/edgar/cik_map.json').read_text())
    ciks = sorted({int(seg[2]) for v in cmap.values() for seg in v.get('segments', [])})
    TOK.mkdir(parents=True, exist_ok=True)
    plan, meta = [], {}
    for cik in ciks:
        _, rows = sec.submissions(cik)
        orig = [f for f in ED.original_filings(rows) if LO <= f['acceptance_et'] <= HI]
        prior = ED.match_prior(orig)
        by = {f['accession']: f for f in orig}
        need = set()
        for a, p in prior.items():
            if by[a]['acceptance_et'] >= FIRST:
                need |= {a, p}
        for a in need:
            plan.append((cik, by[a]))
        for f in orig:
            meta[f['accession']] = (cik, f, prior.get(f['accession']))
    def bad(a):
        q = TOK / f'{a}.json.gz'
        if not q.exists():
            return True
        r = json.loads(gzip.decompress(q.read_bytes()))
        return bool(r['error']) or r['n_tokens'] == 0 and r.get('parser') != 2
    todo = [(c, f) for c, f in plan if bad(f['accession'])]   # missing, failed, or empty under the old parser
    print(f'{len(ciks)} CIKs, {len(plan)} documents needed, {len(todo)} to fetch', flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending, done = [], 0
        for k, (cik, f) in enumerate(todo):
            try:
                url, sha, body = sec.document(cik, f['accession'], f['primaryDocument'])
            except RuntimeError as e:
                print('FETCH FAIL', f['accession'], e, flush=True)
                continue
            pending.append(pool.submit(parse, (f['accession'], cik, url, sha, body)))
            while len(pending) > 16:
                pending.pop(0).result()
                done += 1
            if k % 500 == 0:
                print(f'{k}/{len(todo)} fetched', flush=True)
        for p in pending:
            p.result()
    with open(SIG, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['cik', 'accession', 'form', 'reportDate', 'acceptance_et', 'filingDate', 'prior_accession', 'similarity',
                    'n_tokens', 'n_tokens_prior', 'inline_xbrl', 'prior_inline_xbrl', 'error'])
        for a, (cik, f, p) in sorted(meta.items(), key=lambda x: x[1][1]['acceptance_et']):
            if f['acceptance_et'] < FIRST:
                continue
            if p is None:          # no prior-year comparison: the filing still counts and gives NO signal (PREREG)
                w.writerow([cik, a, f['form'], f['reportDate'], str(f['acceptance_et']), f['filingDate'], '', '', '', '',
                            int(f['inline_xbrl']), '', 'no comparison'])
                continue
            ok = (TOK / f'{a}.json.gz').exists() and (TOK / f'{p}.json.gz').exists()
            ra, rp = (load(a), load(p)) if ok else ({'counts': {}, 'n_tokens': 0, 'error': 'missing'}, {'counts': {}, 'n_tokens': 0, 'error': ''})
            err = ra['error'] or rp['error'] or ('empty document' if not ra['n_tokens'] or not rp['n_tokens'] else '')
            from collections import Counter
            sim = '' if err else f"{ED.cosine(Counter(ra['counts']), Counter(rp['counts'])):.6f}"
            pf = meta[p][1]
            w.writerow([cik, a, f['form'], f['reportDate'], str(f['acceptance_et']), f['filingDate'], p, sim, ra['n_tokens'],
                        rp['n_tokens'], int(f['inline_xbrl']), int(pf['inline_xbrl']), err])
    print('wrote', SIG, flush=True)


if __name__ == '__main__':
    main()
