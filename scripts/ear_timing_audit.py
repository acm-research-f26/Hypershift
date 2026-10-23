"""EAR timing audit (docs/strategy_ear_20261023/TIMING_AUDIT.md). No return after d0+1 of any event is read.

  SEC_USER_AGENT=... python scripts/ear_timing_audit.py
Builds the Item 2.02 event table from the cached submissions JSON, locates the market reaction day inside the
pre-signal window (d0-3..d0+1, market-adjusted |return|), samples 16 events for index-page acceptance times and the
press-release exhibit, and checks that the 2.02 filing precedes the quarter's 10-Q/10-K.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
from hypershift.data import edgar as ED  # noqa: E402
from hypershift.strategy import ear as EA  # noqa: E402

NPZ = 'data/pit/pit_sp500_2016_2023.npz'
OUT = Path('docs/strategy_ear_20261023/TIMING_AUDIT.md')
EVT = Path('data/edgar/ear_events.json')


def blocks(sec, cik):
    j = sec.json(f'https://data.sec.gov/submissions/CIK{int(cik):010d}.json')
    out = [j['filings']['recent']]
    for f in j['filings'].get('files', []):
        out.append(sec.json(f"https://data.sec.gov/submissions/{f['name']}"))
    return out


def build_events(sec, z, cmap):
    """Per-entity Item 2.02 events (dict entity_index -> list of events), joined through dated CIK segments."""
    cal = z['dates'].astype('datetime64[D]')
    labels = list(z['labels'])
    out, tenq = {}, {}
    for i, lab in enumerate(labels):
        evs = []
        for s, e, c in cmap.get(lab, {}).get('segments', []):
            lo, hi = np.datetime64(s) - np.timedelta64(120, 'D'), np.datetime64(e)
            for blk in blocks(sec, c):
                for ev in EA.events(blk, cal):
                    if lo <= np.datetime64(ev['filingDate']) <= hi and np.datetime64(ev['filingDate']) <= np.datetime64('2023-12-31'):
                        evs.append(dict(ev, cik=int(c)))
                n = len(blk['form'])
                for k in range(n):
                    if blk['form'][k] in ('10-Q', '10-K') and lo <= np.datetime64(blk['filingDate'][k]) <= hi:
                        tenq.setdefault(i, []).append(blk['filingDate'][k])
        evs.sort(key=lambda x: x['d0'])
        dedup = []
        for ev in evs:
            if dedup and ev['d0'] - dedup[-1]['d0'] < 5:
                continue
            dedup.append(ev)
        out[i] = dedup
    return out, tenq


def main():
    sec = ED.SecClient()
    z = np.load(NPZ)
    cal = z['dates'].astype('datetime64[D]')
    labels = list(z['labels'])
    member, observed = z['member'], z['observed']
    close = z['close']
    cmap = json.loads(Path('data/edgar/cik_map.json').read_text())
    ev, tenq = build_events(sec, z, cmap)
    EVT.write_text(json.dumps({labels[i]: [{k: (str(v) if k == 'acceptance_et' else v) for k, v in e.items()}
                                           for e in es] for i, es in ev.items()}, indent=0))
    m = EA.market_returns(np.nan_to_num(close, nan=1.), observed, member)
    T = len(cal)
    both = observed[:, 1:] & observed[:, :-1]
    rr = np.zeros(close.shape)
    rr[:, 1:] = np.where(both, close[:, 1:] / np.where(both, close[:, :-1], 1.) - 1, np.nan)
    ar = rr - m[None]
    base = np.nanmean(np.abs(np.where(member, ar, np.nan)))

    # reaction-day location inside the pre-signal window (offsets -3..+1); events while a member only
    offs = np.arange(-3, 2)
    peak, absby, rows = [], [], []
    for i, es in ev.items():
        for e in es:
            d0 = e['d0']
            if d0 - 3 < 1 or d0 + 1 >= T or not member[i, min(d0, T - 1)]:
                continue
            w = ar[i, d0 - 3:d0 + 2]
            if not np.isfinite(w).all():
                continue
            k = int(np.argmax(np.abs(w)))
            peak.append(offs[k])
            absby.append(np.abs(w))
            rows.append((i, e, offs[k], w))
    peak = np.array(peak)
    absby = np.array(absby)
    yrs = {}
    for i, es in ev.items():
        for e in es:
            yrs[e['filingDate'][:4]] = yrs.get(e['filingDate'][:4], 0) + 1

    # coverage: eligible member-quarters with an event
    q = cal.astype('datetime64[M]').astype(int) // 3
    cov = {}
    for i in range(len(labels)):
        dq = {q[e['d0']] for e in ev.get(i, []) if e['d0'] < T}
        for qq in np.unique(q[member[i]]):
            yr = int(qq // 4) + 1970
            hit = qq in dq
            a, b = cov.get(yr, (0, 0))
            cov[yr] = (a + hit, b + 1)

    # 2.02 precedes the quarter's 10-Q/10-K
    prec, gaps = [], []
    for i, fds in tenq.items():
        evd = np.array(sorted(np.datetime64(e['filingDate']) for e in ev.get(i, [])))
        for fd in fds:
            f = np.datetime64(fd)
            if not (np.datetime64('2016-03-01') <= f <= np.datetime64('2023-12-31')) or not len(evd):
                continue
            k = np.searchsorted(evd, f, side='right') - 1
            ok = k >= 0 and (f - evd[k]).astype(int) <= 90
            prec.append(ok)
            if ok:
                gaps.append((f - evd[k]).astype(int))

    # sample: stratified by peak offset
    rng = np.random.default_rng(1)
    picks = []
    for o, nk in ((0, 5), (1, 5), (-1, 3), (-2, 2), (-3, 1)):
        cand = [r for r in rows if r[2] == o]
        if cand:
            picks += [cand[j] for j in rng.choice(len(cand), size=min(nk, len(cand)), replace=False)]
    samp = []
    for i, e, o, w in picks:
        acc = e['accession']
        base_url = f"https://www.sec.gov/Archives/edgar/data/{e['cik']}/{acc.replace('-', '')}"
        try:
            x = sec.get(f'{base_url}/{acc}-index.htm').decode('latin-1')
        except RuntimeError:
            continue
        mm = re.search(r'Accepted</div>\s*<div class="info">([^<]+)<', x)
        accepted = mm.group(1).strip() if mm else '?'
        link = re.search(r'<td scope="row">EX-99[^<]*</td>\s*<td scope="row"><a href="(/Archives/edgar/data/[^"]+)"', x)
        text = ''
        if link:
            try:
                body = sec.get('https://www.sec.gov' + link.group(1))
                t = ED.extract_tokens  # noqa: F841  (tokeniser not needed; keep raw text)
                import lxml.html
                doc = re.sub(r'^\s*<\?xml[^>]*\?>', '', body.decode('utf-8', 'replace'))
                text = re.sub(r'\s+', ' ', lxml.html.fromstring(doc).text_content())[:700]
            except Exception as ex:  # noqa: BLE001
                text = f'(exhibit unreadable: {ex!r:.80})'
        dates = re.findall(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
                           r'\.? \d{1,2},? \d{4}', text)
        samp.append((labels[i], acc, e['filingDate'], accepted, o, float(w[o + 3]), dates[:2], text[:220]))

    L = ['# A. EAR timing audit (2026-10-23)', '',
         'Generated by `scripts/ear_timing_audit.py`. Returns used: market-adjusted daily returns on trading days '
         'd0−3..d0+1 only (the pre-signal window); no post-signal return was read.', '',
         '## Event table', '',
         f'- Original 8-K filings with Item 2.02 for mapped S&P 500 entities, 2016–2023: **{sum(yrs.values())}** '
         '(8-K/A excluded; filings within 5 trading days collapsed). By year: '
         + ', '.join(f'{k} {v}' for k, v in sorted(yrs.items())) + '.',
         '- Member-quarters with at least one event (coverage): '
         + ', '.join(f'{y} {a / b:.3f}' for y, (a, b) in sorted(cov.items())) + '.',
         f'- The quarter\'s 10-Q/10-K is preceded by a 2.02 filing within 90 days in **{np.mean(prec):.3f}** of '
         f'{len(prec)} periodic filings; median gap {int(np.median(gaps))} days.', '',
         '## Where is the market reaction? (d0 = trading day of SEC\'s official filing date)', '',
         f'{len(peak)} member events with all bars. Day of the largest |market-adjusted return| within d0−3..d0+1:', '',
         '| Offset | Share of events | Mean abs(AR) | × non-event mean |', '|---|---|---|---|']
    for k, o in enumerate(offs):
        L.append(f'| {o:+d} | {np.mean(peak == o):.3f} | {absby[:, k].mean():.4f} | {absby[:, k].mean() / base:.1f} |')
    L += ['', f'Non-event baseline mean abs(AR) on member-days: {base:.4f}. Under no information, each offset would hold '
          '0.20 of peaks.', '',
          '## Sample timelines (index-page acceptance in ET; press-release exhibit text)', '',
          '| Entity | Accession | SEC filing date | Accepted (index, ET) | Peak offset | AR at peak | Dates in release | Release text (start) |',
          '|---|---|---|---|---|---|---|---|']
    for s in samp:
        L.append(f'| {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]:+d} | {s[5]:+.3f} | {"; ".join(s[6])} | '
                 f'{s[7].replace("|", "/")} |')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
