"""S&P 400 pilot B: 20 deleted constituents - Alpaca asof bars and identity checks. No returns are computed.

  SEC_USER_AGENT=... python scripts/sp400_pilot_deleted.py   -> data/sp400/pilot_deleted.json, stdout table
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from hypershift.data import edgar as ED  # noqa: E402
from hypershift.data.alpaca import fetch_bars, make_http  # noqa: E402
from sp400_membership import parse_date  # noqa: E402

DISTRESS = {'AKRX', 'DNR', 'OAS', 'WLL', 'CHK', 'FTR', 'ENDP', 'MNK', 'DO', 'NE', 'VAL', 'RIG', 'HTZ', 'JCP', 'GNC',
            'WFT', 'CBL', 'SFIX', 'BBBY', 'DF', 'TUES', 'GNW', 'SUP'}


def pick(rows):
    cats = {'acquired': [], 'sp500': [], 'sp600': [], 'mcap': [], 'other': [], 'distress': []}
    for d, t, n, why in rows:
        w = why.lower()
        k = 'distress' if t in DISTRESS else 'sp500' if 's&p 500' in w else 'sp600' if ('s&p 600' in w or 'smallcap' in w) \
            else 'acquired' if ('acqui' in w or 'merg' in w) else 'mcap' if 'market cap' in w else 'other'
        cats[k].append((d, t, n, why, k))
    want = {'acquired': 5, 'sp500': 3, 'sp600': 3, 'mcap': 4, 'other': 2, 'distress': 3}
    out = []
    for k, n in want.items():
        v = sorted(cats[k])
        if v:
            idx = np.linspace(0, len(v) - 1, min(n, len(v))).round().astype(int)
            out += [v[i] for i in idx]
    return out[:20]


def main():
    last = json.loads(Path('data/sp400/wiki/2024-01-01.json').read_text())
    rows = []
    for r in last['changes']:
        d = parse_date(r[0])
        if d is not None and np.datetime64('2016-01-01') <= d < np.datetime64('2024-01-01') and r[3] not in ('nan', ''):
            rows.append((str(d), r[3].upper(), r[4], r[5][:120]))
    sel = pick(rows)
    sec, http0 = ED.SecClient(), make_http()
    tick = {v['ticker'].upper(): int(v['cik_str']) for v in sec.json('https://www.sec.gov/files/company_tickers.json').values()}
    res = []
    for d, t, name, why, cat in sel:
        asof = str(np.busday_offset(np.datetime64(d), -1, roll='backward'))
        got = {}

        def http(p):
            s, js = http0(p)
            if s == 200:
                for k, b in (js.get('bars') or {}).items():
                    got.setdefault(k, []).extend(b)
            return s, js
        try:
            fetch_bars([t.replace('-', '.')], '2015-06-01', '2023-12-31', 'raw', http, asof=asof)
        except RuntimeError:
            pass
        bars = got.get(t.replace('-', '.'), [])
        first = bars[0]['t'][:10] if bars else None
        lastb = bars[-1]['t'][:10] if bars else None
        px = None
        for b in bars:
            if b['t'][:10] <= asof:
                px = b['c']
        # identity: EDGAR ticker lookup, else current ticker table
        cik, how = None, None
        try:
            x = sec.get(f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}&type=10-K&dateb=&owner=include&count=10&output=atom').decode('latin-1')
            m = re.search(r'<cik>(\d+)</cik>', x)
            if m:
                cik, how = int(m.group(1)), 'EDGAR ticker lookup'
        except RuntimeError:
            pass
        if cik is None and t in tick:
            cik, how = tick[t], 'current ticker table'
        ename, name_ok, filed_near, mcap = None, False, False, None
        if cik:
            meta, frows = sec.submissions(cik)
            ename = meta.get('name')
            names = [ename or ''] + [f.get('name', '') for f in (meta.get('formerNames') or [])]
            name_ok = ED.name_match([name], names)
            acc = [np.datetime64(f['acceptance_et'], 'D') for f in ED.original_filings(frows)]
            filed_near = any(0 <= (np.datetime64(d) - a).astype(int) <= 200 for a in acc)
            try:
                cf = sec.json(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json')
                sh = cf.get('facts', {}).get('dei', {}).get('EntityCommonStockSharesOutstanding', {}).get('units', {}).get('shares', [])
                sh = [s for s in sh if s['end'] <= d]
                if sh and px:
                    mcap = sorted(sh, key=lambda s: s['end'])[-1]['val'] * px / 1e9
            except RuntimeError:
                pass
        ok = bool(bars) and name_ok and filed_near and (mcap is None or .3 <= mcap <= 40)
        res.append(dict(date=d, ticker=t, name=name, cat=cat, reason=why, asof=asof, first_bar=first, last_bar=lastb,
                        close_asof=px, cik=cik, how=how, edgar_name=ename, name_ok=bool(name_ok), filed_near=bool(filed_near),
                        mcap_bn=None if mcap is None else round(mcap, 2), identity_ok=ok))
        print(f"{d} {t:6s} {cat:8s} bars {first}..{lastb} px {px} cik {cik} {ename} name {name_ok} filed {filed_near} "
              f"mcap {None if mcap is None else round(mcap, 2)} -> {'OK' if ok else 'CHECK'}", flush=True)
    Path('data/sp400/pilot_deleted.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
