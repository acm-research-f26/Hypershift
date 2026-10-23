"""Map PIT S&P 500 entities to SEC CIKs, by month, and verify every CIK (PREREG check P2).

  SEC_USER_AGENT=... python scripts/lazy_cik_map.py   -> data/edgar/cik_map.json, docs/strategy_lazyprices_20261023/CIK_MAP.md

Source: monthly revisions (2016-01..2024-01) of Wikipedia's "List of S&P 500 companies"; its constituent table
carries each member's CIK as of that revision, so a company that re-registers gets dated CIK segments. Every CIK is
verified against SEC: EDGAR current/former name matches the Wikipedia security name (`edgar.name_match`) and an
original 10-K/10-Q was accepted within 200 days before >= 90% of the member months assigned to it (months before
the registrant's first such filing are not counted). Fallback for unlisted or failed months: SEC ticker table and
EDGAR name search, same checks. No filing text and no returns are read.
"""
import json
import re
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, 'src')
from hypershift.data import edgar as ED  # noqa: E402

NPZ = 'data/pit/pit_sp500_2016_2023.npz'
WIKI = Path('data/edgar/wiki')
OUT_JSON = Path('data/edgar/cik_map.json')
OUT_MD = Path('docs/strategy_lazyprices_20261023/CIK_MAP.md')
API = 'https://en.wikipedia.org/w/api.php'
HDR = {'User-Agent': 'HyperShift research (reads S&P 500 list revisions)'}


def words(s):
    return {w for w in re.findall(r'[a-z0-9]+', str(s).lower().replace('&', ' ')) if w not in ED.NAME_STOP and len(w) > 1}


def _get(url, **kw):
    for k in range(6):
        try:
            r = requests.get(url, headers=HDR, timeout=120, **kw)
            if r.status_code == 200 and r.text.strip():
                return r
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    raise RuntimeError(f'wikipedia request failed: {url}')


def revision_tables(when):
    """(constituents [ticker, security, cik], change-table rows [ticker, security]) of the last revision before `when`."""
    p = WIKI / f'{when}.json'
    if p.exists():
        j = json.loads(p.read_text())
        return pd.DataFrame(j['cons']), pd.DataFrame(j['chg'])
    rq = _get(API, params=dict(action='query', prop='revisions', titles='List of S&P 500 companies', rvlimit=1,
                               rvstart=f'{when}T00:00:00Z', rvdir='older', format='json', rvprop='ids|timestamp')).json()
    rev = list(rq['query']['pages'].values())[0]['revisions'][0]
    html = _get('https://en.wikipedia.org/w/index.php', params=dict(title='List of S&P 500 companies',
                                                                   oldid=rev['revid'])).text
    t = pd.read_html(StringIO(html))
    c = t[0]
    c.columns = [str(x) for x in c.columns]
    tk = [x for x in c.columns if 'Symbol' in x or 'Ticker' in x][0]
    cons = pd.DataFrame({'ticker': c[tk].astype(str).str.strip().str.upper(), 'security': c['Security'].astype(str),
                         'cik': pd.to_numeric(c['CIK'], errors='coerce')})
    rows = []
    if len(t) > 1:
        for r in t[1].astype(str).values:
            if len(r) >= 5:
                rows += [(r[1], r[2]), (r[3], r[4])]
    chg = pd.DataFrame(rows, columns=['ticker', 'security'])
    chg = chg[chg.ticker.str.match(r'^[A-Z][A-Z\.\-]*$', na=False)]
    WIKI.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({'rev': rev, 'cons': cons.to_dict('list'), 'chg': chg.to_dict('list')}))
    return cons, chg


def coverage_months(rows, months):
    """Share of member months with an original 10-K/10-Q accepted in the 200 days before the month's first member day;
    months before the registrant's first such filing (new registrants, spin-offs) are not counted."""
    acc = np.sort(np.array([np.datetime64(f['acceptance_et'], 'D') for f in ED.original_filings(rows)]))
    if not len(acc):
        return 0., None
    use = [d for d in months if d > acc[0]]
    if not use:
        return 0., acc[-1]
    hits = 0
    for d in use:
        k = np.searchsorted(acc, d, side='left') - 1
        hits += k >= 0 and (d - acc[k]).astype(int) <= 200
    return hits / len(use), acc[-1]


def main():
    sec = ED.SecClient()
    z = np.load(NPZ)
    labels, member, last = list(z['labels']), z['member'], z['last']
    dates = z['dates'].astype('datetime64[D]')
    T = len(dates)
    month = dates.astype('datetime64[M]')
    snaps = [str(m) + '-01' for m in np.arange(np.datetime64('2016-01'), np.datetime64('2024-02'))]
    tables = {s: revision_tables(s) for s in snaps}
    names = {}
    for s in snaps:
        cons, chg = tables[s]
        for tk, sname in list(zip(chg.ticker, chg.security)) + list(zip(cons.ticker, cons.security)):
            names.setdefault(tk, set()).add(sname)
    tick_now = sec.json('https://www.sec.gov/files/company_tickers.json')
    cache = {}

    def sub(cik):
        if cik not in cache:
            try:
                cache[cik] = sec.submissions(cik)
            except RuntimeError:
                cache[cik] = ({}, [])
        return cache[cik]

    def verify(cik, months, wiki_names, strict=False):
        meta, rows = sub(cik)
        edgar_names = [meta.get('name') or ''] + [f.get('name', '') for f in (meta.get('formerNames') or [])]
        cov, last_acc = coverage_months(rows, months)
        nm = ED.name_match(wiki_names, edgar_names, strict=strict) if wiki_names else False
        return cov >= .9 and nm, cov, nm, meta, last_acc

    out = {}
    for i, lab in enumerate(labels):
        mem = np.flatnonzero(member[i])
        if not len(mem):
            continue
        firsts = mem[np.r_[True, month[mem][1:] != month[mem][:-1]]]
        mlist = [month[k] for k in firsts]
        mday = {month[k]: dates[k] for k in firsts}
        syms = lab.split('|')
        tick_set = {x for tk in syms for x in (tk, tk.replace('.', '-'))}
        wiki_names = set()
        for tk in tick_set:
            wiki_names |= names.get(tk, set())
        assign = {}
        for sname in snaps:
            m = np.datetime64(sname[:7])
            if m not in mday:
                continue
            hit = tables[sname][0]
            hit = hit[hit.ticker.isin(tick_set)]
            ciks = [int(c) for c in hit.cik if np.isfinite(c)]
            if ciks:
                assign[m] = ciks[0]
                wiki_names |= set(hit.security)
        for m in mlist:                      # months without a revision listing: nearest listed month within 2
            if m not in assign:
                near = [(abs((m - k).astype(int)), c) for k, c in assign.items() if abs((m - k).astype(int)) <= 2]
                if near:
                    assign[m] = min(near)[1]
        report = {}
        for c in set(assign.values()):
            ms = [mday[m] for m in mlist if assign.get(m) == c]
            ok, cov, nm, meta, last_acc = verify(c, ms, wiki_names)
            report[c] = (ok, cov, nm, (meta or {}).get('name'), 'wikipedia revision', last_acc, (meta or {}).get('sic'))
            if not ok:
                for m in [m for m in mlist if assign.get(m) == c]:
                    del assign[m]
        rest = [m for m in mlist if m not in assign]
        own = [c for c, r in report.items() if r[0]]          # the entity's own verified CIKs come first
        for c in own:
            if rest:
                ok, cov, nm, meta, last_acc = verify(c, [mday[m] for m in rest], wiki_names)
                if ok:
                    for m in rest:
                        assign[m] = c
                    rest = []
        if rest:
            cands = [int(v['cik_str']) for v in tick_now.values() if v['ticker'].upper() in tick_set]
            for nm_ in sorted(wiki_names):
                q = ' '.join(sorted(words(nm_), key=len, reverse=True)[:3])
                if q:
                    hits = sec.json(f'https://efts.sec.gov/LATEST/search-index?keysTyped={requests.utils.quote(q)}')
                    cands += [int(h['_id']) for h in hits.get('hits', {}).get('hits', [])[:5]]
            best = None
            for c in dict.fromkeys(cands):
                ok, cov, nm, meta, last_acc = verify(c, [mday[m] for m in rest], wiki_names, strict=True)
                if ok and (best is None or cov > best[1]):
                    best = (c, cov, nm, meta, last_acc)
            if best:
                for m in rest:
                    assign[m] = best[0]
                report[best[0]] = (True, best[1], best[2], (best[3] or {}).get('name'), 'ticker table / name search',
                                   best[4], (best[3] or {}).get('sic'))
        segs = []
        for m in mlist:
            c = assign.get(m)
            if c is None:
                continue
            end = str((m + 1).astype('datetime64[D]') - 1)
            if segs and segs[-1][2] == c:
                segs[-1][1] = end
            else:
                segs.append([str(m.astype('datetime64[D]')), end, c])
        last_c = segs[-1][2] if segs else None
        delisted = 0 <= last[i] < T - 5
        la = report[last_c][5] if last_c else None
        cont = bool(last_c and delisted and la is not None and (la - dates[last[i]]).astype(int) > 200)
        out[lab] = {'segments': segs, 'cik': last_c, 'sic': report[last_c][6] if last_c else None,
                    'member_months': len(mlist), 'mapped_months': int(sum(m in assign for m in mlist)),
                    'checked': {str(c): {'ok': bool(r[0]), 'coverage': round(float(r[1]), 3), 'name_ok': bool(r[2]), 'edgar_name': r[3],
                                         'how': r[4]} for c, r in report.items()},
                    'wiki_names': sorted(wiki_names), 'continues_filing_after_last_bar': cont}
        print(f'{i+1}/{len(labels)} {lab}: ' + '; '.join(f'{a}..{b} {c}' for a, b, c in segs) +
              f' ({out[lab]["mapped_months"]}/{len(mlist)})', flush=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    mm = sum(v['member_months'] for v in out.values())
    mapped = sum(v['mapped_months'] for v in out.values())
    part = {k: v for k, v in out.items() if v['mapped_months'] < v['member_months']}
    multi_seg = {k: v['segments'] for k, v in out.items() if len({s[2] for s in v['segments']}) > 1}
    shared = {}
    for k, v in out.items():
        for c in {s[2] for s in v['segments']}:
            shared.setdefault(c, []).append(k)
    multi = {c: ks for c, ks in shared.items() if len(ks) > 1}
    cont = [k for k, v in out.items() if v['continues_filing_after_last_bar']]
    nfall = sum(1 for v in out.values() for r in v['checked'].values() if r['ok'] and r['how'] != 'wikipedia revision')
    L = ['# CIK map: PIT S&P 500 entities to SEC registrants (PREREG check P2)', '',
         'Generated by `scripts/lazy_cik_map.py`; map in `data/edgar/cik_map.json` (git-ignored).',
         '- **Source:** the CIK that monthly revisions (2016-01..2024-01) of Wikipedia\'s S&P 500 constituent table list '
         'for the entity\'s ticker in that month, so a company that re-registers (new CIK) gets dated segments.',
         '- **Verification of every CIK:** EDGAR current or former name matches the Wikipedia security name '
         '(`edgar.name_match`, tested), and an original 10-K/10-Q was accepted within 200 days before ≥ 90% of the '
         'member months assigned to it (months before the registrant\'s first such filing are not counted).',
         '- **Fallback** for unlisted or failed months: SEC ticker table and EDGAR name search, same checks.',
         '- No filing text or returns were read.', '',
         f'- Entities with member months: {len(out)}; CIKs verified through the fallback: {nfall}.',
         f'- **Member-months mapped: {mapped} / {mm} = {mapped / max(mm, 1):.3f}** (pass condition ≥ 0.90).',
         f'- Entities with dated CIK changes ({len(multi_seg)}): ' + '; '.join(
             f'{k}: ' + ' → '.join(f'{s[2]} from {s[0]}' for s in v) for k, v in multi_seg.items()) + '.',
         f'- Entities not fully mapped ({len(part)}): ' + '; '.join(
             f'{k} {v["mapped_months"]}/{v["member_months"]} (' + ', '.join(
                 f'{c}: cov {r["coverage"]}, name {r["name_ok"]}, {r["edgar_name"]}' for c, r in v['checked'].items()) + ')'
             for k, v in part.items()) + '.',
         f'- CIKs shared by several entities (one company under several ticker labels; disjoint membership): '
         f'{len(multi)}: ' + '; '.join(f'{c}: {"/".join(ks)}' for c, ks in multi.items()) + '.',
         f'- Delisted entities whose last CIK files 10-K/10-Q > 200 days after the last bar (subsidiaries that keep '
         f'reporting, or identity breaks: inspect): {len(cont)}: ' + ', '.join(
             f'{k} ({out[k]["checked"][str(out[k]["cik"])]["edgar_name"]})' for k in cont) + '.']
    OUT_MD.write_text('\n'.join(L) + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
