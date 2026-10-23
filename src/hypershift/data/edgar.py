"""SEC EDGAR filings for the filing-text strategy (docs/strategy_lazyprices_20261023/PREREG.md).

Only what the frozen signal needs: original 10-K/10-Q lists per CIK with acceptance times, primary-document text
reduced to term counts, prior-year matching, cosine similarity, and SIC -> Fama-French 12 industries.
Network access goes through SecClient: User-Agent from the SEC_USER_AGENT environment variable (SEC fair-access
policy), at most 8 requests/second, JSON responses cached on disk with provenance.
"""
import gzip
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

FORMS = ('10-K', '10-Q')
CACHE = Path('data/edgar')

FF12 = {1: ((100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)),
        2: ((2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751),
            (3792, 3792), (3900, 3939), (3990, 3999)),
        3: ((2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709),
            (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)),
        4: ((1200, 1399), (2900, 2999)),
        5: ((2800, 2829), (2840, 2899)),
        6: ((3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)),
        7: ((4800, 4899),),
        8: ((4900, 4949),),
        9: ((5000, 5999), (7200, 7299), (7600, 7699)),
        10: ((2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)),
        11: ((6000, 6999),)}   # Ken French Siccodes12.txt (checked 2026-10-23); everything else is 12 (Other)


def ff12(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return 12
    for k, ranges in FF12.items():
        if any(a <= s <= b for a, b in ranges):
            return k
    return 12


def _us_dst(local):
    """US Eastern daylight time in force at this local wall-clock time (rules since 2007)."""
    y = local.astype('datetime64[Y]').astype(int) + 1970
    mar1 = np.datetime64(f'{y}-03-01')
    nov1 = np.datetime64(f'{y}-11-01')
    start = np.busday_offset(mar1, 1, roll='forward', weekmask='Sun') + np.timedelta64(2, 'h')   # 2nd Sunday of March
    end = np.busday_offset(nov1, 0, roll='forward', weekmask='Sun') + np.timedelta64(2, 'h')     # 1st Sunday of Nov
    return start <= local < end


def acceptance_et(s):
    """Eastern wall-clock acceptance time from a submissions-JSON `acceptanceDateTime`.

    The JSON value labelled UTC equals ET + 2 x (UTC offset): +8 h in EDT, +10 h in EST (verified against index-page
    'Accepted' times on 8 filings, 2026-10-23)."""
    t = np.datetime64(s.replace('Z', '').split('.')[0], 's')
    for h, dst in ((8, True), (10, False)):
        et = t - np.timedelta64(h, 'h')
        if _us_dst(et) == dst:
            return et
    return t - np.timedelta64(8, 'h')


def available_index(acc_et, cal, filing_date=None):
    """Index in trading calendar `cal` of the first trading day strictly after the acceptance date (ET), and never
    before the day after SEC's official filing date (guards against the JSON timestamp's inconsistent time zone)."""
    d = np.datetime64(acc_et, 'D')
    if filing_date:
        d = max(d, np.datetime64(filing_date, 'D'))
    return int(np.searchsorted(cal, d, side='right'))


_HIDDEN = re.compile(r'display\s*:\s*none', re.I)
_TOKEN = re.compile(r'[a-z]{2,}')


def extract_tokens(doc):
    """Term counts of a 10-K/10-Q primary document under the frozen rules (PREREG): hidden inline-XBRL and
    display:none elements, scripts and styles removed; tables with > 15% numeric characters removed; lowercase
    alphabetic tokens of 2+ letters; stop words kept."""
    import lxml.html
    if isinstance(doc, bytes):
        doc = doc.decode('utf-8', errors='replace')
    if '<' not in doc[:2000] and '<html' not in doc.lower()[:5000]:
        return Counter(_TOKEN.findall(doc.lower()))
    doc = re.sub(r'^\s*<\?xml[^>]*\?>', '', doc)       # lxml refuses str input with an encoding declaration
    root = lxml.html.fromstring(doc)
    drop = [el for el in root.iter() if isinstance(el.tag, str) and (
        el.tag.lower() in ('script', 'style') or el.tag.lower().endswith('header') and ':' in el.tag
        or _HIDDEN.search(el.get('style') or ''))]
    for el in drop:
        if el.getparent() is not None:
            el.drop_tree()
    for tab in list(root.iter('table')):
        txt = tab.text_content()
        digits = sum(c.isdigit() for c in txt)
        letters = sum(c.isalpha() for c in txt)
        if digits + letters and digits / (digits + letters) > .15 and tab.getparent() is not None:
            tab.drop_tree()
    return Counter(_TOKEN.findall(root.text_content().lower()))


def cosine(a, b):
    if not a or not b:
        return 0.
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def original_filings(filings):
    """Original 10-K / 10-Q only (exact form strings), sorted by acceptance time."""
    return sorted((f for f in filings if f['form'] in FORMS), key=lambda f: f['acceptance_et'])


def match_prior(filings, lo=330, hi=400):
    """{accession: prior-year accession}: same form, accepted earlier, report date lo..hi days before; latest wins."""
    out = {}
    for f in filings:
        r = np.datetime64(f['reportDate'])
        best = None
        for p in filings:
            if p['form'] != f['form'] or p['acceptance_et'] >= f['acceptance_et'] or not p['reportDate']:
                continue
            gap = (r - np.datetime64(p['reportDate'])).astype(int)
            if lo <= gap <= hi and (best is None or p['acceptance_et'] > best['acceptance_et']):
                best = p
        if best is not None:
            out[f['accession']] = best['accession']
    return out


def filings_from_submissions(sub):
    """Rows of a submissions JSON block (`filings.recent` or an additional `files` page) as filing dicts."""
    keys = ('accessionNumber', 'form', 'filingDate', 'reportDate', 'acceptanceDateTime', 'primaryDocument',
            'isInlineXBRL')
    n = len(sub['form'])
    out = []
    for k in range(n):
        row = {c: sub.get(c, [None] * n)[k] for c in keys}
        out.append({'accession': row['accessionNumber'], 'form': row['form'], 'filingDate': row['filingDate'],
                    'reportDate': row['reportDate'] or None, 'acceptance_et': acceptance_et(row['acceptanceDateTime']),
                    'primaryDocument': row['primaryDocument'], 'inline_xbrl': bool(row['isInlineXBRL'])})
    return out


class SecClient:
    """Polite SEC client: declared User-Agent, <= 8 requests/s, retries, JSON cache with provenance."""

    def __init__(self, cache=CACHE, rate=8.):
        ua = os.environ.get('SEC_USER_AGENT')
        if not ua:
            raise RuntimeError('set SEC_USER_AGENT ("Name email") as required by SEC fair-access policy')
        import requests
        self.s = requests.Session()
        self.s.headers.update({'User-Agent': ua, 'Accept-Encoding': 'gzip, deflate'})
        self.cache, self.gap, self.last = Path(cache), 1. / rate, 0.

    def get(self, url):
        for attempt in range(6):
            wait = self.gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.s.get(url, timeout=60)
            except Exception:          # dropped connection / timeout: back off and retry
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f'SEC {r.status_code} for {url}')
        raise RuntimeError(f'SEC retries exhausted for {url}')

    def json(self, url):
        p = self.cache / 'json' / (hashlib.sha256(url.encode()).hexdigest()[:24] + '.json.gz')
        if p.exists():
            return json.loads(gzip.decompress(p.read_bytes()))['body']
        body = json.loads(self.get(url))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(json.dumps({'url': url, 'retrieved': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                'body': body}).encode()))
        return body

    def submissions(self, cik):
        """All filings of a CIK (recent block plus older pages) as filing dicts, with registrant metadata."""
        j = self.json(f'https://data.sec.gov/submissions/CIK{int(cik):010d}.json')
        rows = filings_from_submissions(j['filings']['recent'])
        for f in j['filings'].get('files', []):
            rows += filings_from_submissions(self.json(f"https://data.sec.gov/submissions/{f['name']}"))
        meta = {k: j.get(k) for k in ('cik', 'name', 'sic', 'tickers', 'formerNames', 'fiscalYearEnd')}
        return meta, rows

    def document(self, cik, accession, primary):
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{primary}"
        body = self.get(url)
        return url, hashlib.sha256(body).hexdigest(), body


NAME_STOP = {'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited', 'plc', 'the', 'holdings',
             'holding', 'group', 'class', 'new', 'de', 'nv', 'sa', 'lp', 'and', 'of', 'international', 'common', 'stock',
             'llc', 'trust'}


def _sig(name):
    import unicodedata
    name = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    toks = re.findall(r'[a-z0-9]+', name.lower().replace("'", ''))
    return [t for t in toks if t not in NAME_STOP]


def _pair_match(a, b, strict=False):
    if not a or not b:
        return False
    ca, cb = ''.join(a), ''.join(b)
    if ca == cb:
        return True
    if strict:
        return any(len(x) >= 2 and set(x) <= set(y) for x, y in ((a, b), (b, a)))
    for x, y in ((a, b), (b, a)):
        cx = ''.join(x)
        if len(cx) >= 3 and set(x) <= set(y):
            return True
        if len(cx) >= 4 and any(''.join(y[:k]) == cx for k in range(1, len(y) + 1)):
            return True
    return False


def name_match(names_a, names_b, strict=False):
    """Same company name: equal canonical names, one name's significant words a subset of the other's, or one name
    equal to a word-aligned prefix of the other (\"DuPont\" ~ \"DU PONT E I DE NEMOURS\"; \"Endo\" !~ \"ENDONOVO\").
    strict=True (candidates found by name search): only equal names or a subset of at least two significant words."""
    return any(_pair_match(_sig(x), _sig(y), strict) for x in names_a for y in names_b)
