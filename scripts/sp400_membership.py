"""S&P 400 point-in-time membership feasibility (docs/sp400_feasibility_20261024). No prices or returns.

  python scripts/sp400_membership.py   -> data/sp400/wiki/*.json (cache), data/sp400/membership_probe.json, stdout summary
Monthly Wikipedia revisions of "List of S&P 400 companies" (last revision before the 1st of each month, 2016-01..2024-01)
are parsed for constituent tickers; month-to-month differences are reconciled with the dated change table of the
2024-01 revision, and with an independent source: S&P 500 additions (fja05680) that came from the S&P 400 must leave
the S&P 400 list at the same time.
"""
import csv
import json
import re
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TITLE = 'List of S&P 400 companies'
WIKI = Path('data/sp400/wiki')
HDR = {'User-Agent': 'HyperShift research (reads S&P 400 list revisions)'}
API = 'https://en.wikipedia.org/w/api.php'


def get(url, **kw):
    for k in range(6):
        try:
            r = requests.get(url, headers=HDR, timeout=120, **kw)
            if r.status_code == 200 and r.text.strip():
                return r
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    raise RuntimeError(url)


def revision(when):
    p = WIKI / f'{when}.json'
    if p.exists():
        return json.loads(p.read_text())
    rq = get(API, params=dict(action='query', prop='revisions', titles=TITLE, rvlimit=1, rvstart=f'{when}T00:00:00Z',
                              rvdir='older', format='json', rvprop='ids|timestamp')).json()
    rev = list(rq['query']['pages'].values())[0]['revisions'][0]
    html = get('https://en.wikipedia.org/w/index.php', params=dict(title=TITLE, oldid=rev['revid'])).text
    t = pd.read_html(StringIO(html))
    c = t[0]
    c.columns = [str(x) for x in c.columns]
    tk = [x for x in c.columns if 'ymbol' in x or 'icker' in x][0]
    nm = [x for x in c.columns if x in ('Security', 'Company')][0]
    chg = []
    if len(t) > 1:
        for r in t[1].astype(str).values:
            if len(r) >= 5:
                chg.append([str(x) for x in r[:6]])
    out = {'rev': rev, 'tickers': [str(x).strip().upper() for x in c[tk]], 'names': [str(x) for x in c[nm]],
           'changes': chg}
    WIKI.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


def parse_date(s):
    for f in ('%B %d, %Y', '%b %d, %Y', '%Y-%m-%d'):
        try:
            return np.datetime64(datetime.strptime(s.strip(), f).date())
        except ValueError:
            pass
    return None


def main():
    months = [str(m) + '-01' for m in np.arange(np.datetime64('2016-01'), np.datetime64('2024-02'))]
    revs = {}
    for m in months:
        revs[m] = revision(m)
        time.sleep(.5)
    sets = {m: set(r['tickers']) for m, r in revs.items()}
    counts = {m: len(s) for m, s in sets.items()}
    stale = {m: (np.datetime64(m) - np.datetime64(revs[m]['rev']['timestamp'][:10])).astype(int) for m in months}
    # change log from the last revision
    log = []
    for r in revs[months[-1]]['changes']:
        d = parse_date(r[0])
        if d is None:
            continue
        log.append((d, r[1].upper() if r[1] not in ('nan', '') else None, r[3].upper() if r[3] not in ('nan', '') else None, r[5]))
    log16 = [x for x in log if np.datetime64('2016-01-01') <= x[0] < np.datetime64('2024-01-01')]
    # reconcile: snapshot diffs vs log (within the same or next month)
    diffs = []
    for a, b in zip(months[:-1], months[1:]):
        for t in sets[b] - sets[a]:
            diffs.append(('add', t, a, b))
        for t in sets[a] - sets[b]:
            diffs.append(('rem', t, a, b))
    def explained(kind, t, a, b):
        lo, hi = np.datetime64(a) - np.timedelta64(31, 'D'), np.datetime64(b) + np.timedelta64(1, 'D')
        for d, ad, rm, _ in log:
            if lo <= d <= hi and ((kind == 'add' and ad == t) or (kind == 'rem' and rm == t)):
                return True
        return False
    exp = [explained(*x) for x in diffs]
    # log entries reflected in snapshots
    def reflected(d, t, kind):
        k = np.searchsorted(np.array(months, dtype='datetime64[D]'), d, side='right')   # first month start after d
        for j in range(k, min(k + 3, len(months))):
            s = sets[months[j]]
            if (kind == 'add' and t in s) or (kind == 'rem' and t not in s):
                return j - k                                                             # lag in months
        return None
    lags = []
    for d, ad, rm, _ in log16:
        if d >= np.datetime64(months[-1]):
            continue
        if ad:
            lags.append(('add', ad, str(d), reflected(d, ad, 'add')))
        if rm:
            lags.append(('rem', rm, str(d), reflected(d, rm, 'rem')))
    # independent check: S&P 500 additions (fja05680) that were S&P 400 members the month before
    rows = list(csv.DictReader(open('data/audit/membership/components.csv', encoding='utf-8')))
    snaps = sorted((np.datetime64(r['date']), set(t.strip() for t in r['tickers'].split(','))) for r in rows)
    mig = []
    for (d0, s0), (d1, s1) in zip(snaps[:-1], snaps[1:]):
        if not (np.datetime64('2016-02-01') <= d1 < np.datetime64('2024-01-01')):
            continue
        for t in s1 - s0:
            m_before = str((d1.astype('datetime64[M]') - 1).astype('datetime64[D]'))
            m_after = str((d1.astype('datetime64[M]') + 1).astype('datetime64[D]'))
            if m_before in sets and t in sets[m_before]:
                mig.append((t, str(d1), m_after in sets and t not in sets.get(m_after, set())))
    out = {'counts': counts, 'stale_days': {k: int(v) for k, v in stale.items()}, 'n_log_2016_2023': len(log16),
           'snapshot_diffs': len(diffs), 'diffs_explained': int(sum(exp)),
           'unexplained': [d for d, e in zip(diffs, exp) if not e],
           'log_reflection': lags, 'sp500_migrations': mig}
    Path('data/sp400/membership_probe.json').write_text(json.dumps(out, indent=0, default=str))
    c = np.array(list(counts.values()))
    lg = [x[3] for x in lags]
    print('months', len(c), 'count min/median/max', c.min(), int(np.median(c)), c.max(),
          'within ±5% of 400:', np.mean(np.abs(c - 400) <= 20))
    print('revision staleness days: median', int(np.median(list(stale.values()))), 'max', max(stale.values()))
    print('change-log entries 2016-2023:', len(log16), '| snapshot diffs', len(diffs), 'explained by log', int(sum(exp)))
    print('log entries reflected: lag0', lg.count(0), 'lag1', lg.count(1), 'lag2', lg.count(2), 'never(<=2m)', lg.count(None))
    print('S&P500 migrations from S&P400:', len(mig), 'removed from 400 list by next month:', sum(m[2] for m in mig))


if __name__ == '__main__':
    main()
