"""Map PIT S&P 500 entities to SEC CIKs and verify each mapping against the CIK's filing history (PREREG P2).

  SEC_USER_AGENT=... python scripts/lazy_cik_map.py   -> data/edgar/cik_map.json, docs/strategy_lazyprices_20261023/CIK_MAP.md
Candidates, in order: current SEC ticker table; EDGAR ticker lookup (keeps some retired tickers); Alpaca asset
name -> EDGAR entity search. A candidate is accepted only if an original 10-K/10-Q was accepted within the 200 days
before at least 90% of the entity's member months. No filing text and no returns are read here.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'src')
from hypershift.data import edgar as ED  # noqa: E402

NPZ = 'data/pit/pit_sp500_2016_2023.npz'
OUT_JSON = Path('data/edgar/cik_map.json')
OUT_MD = Path('docs/strategy_lazyprices_20261023/CIK_MAP.md')
SUFFIX = re.compile(r'\b(common stock|class [a-c]|ordinary shares|shares|inc|incorporated|corp|corporation|co|company|'
                    r'ltd|limited|plc|the|holdings?|group|n\.?v|s\.?a|lp|l\.p|new|de|ag)\b\.?', re.I)


def coverage(rows, member_days):
    """Share of member months with an original 10-K/10-Q accepted in the 200 days before the month's first member day."""
    acc = np.sort(np.array([np.datetime64(f['acceptance_et'], 'D') for f in ED.original_filings(rows)]))
    if not len(member_days) or not len(acc):
        return 0., None
    hits = 0
    for d in member_days:
        k = np.searchsorted(acc, d, side='left') - 1
        hits += k >= 0 and (d - acc[k]).astype(int) <= 200
    return hits / len(member_days), acc[-1]


def alpaca_name(sym):
    import requests
    r = requests.get(f'https://paper-api.alpaca.markets/v2/assets/{sym}', timeout=30,
                     headers={'APCA-API-KEY-ID': os.environ['APCA_API_KEY_ID'],
                              'APCA-API-SECRET-KEY': os.environ['APCA_API_SECRET_KEY']})
    return r.json().get('name') if r.status_code == 200 else None


def main():
    sec = ED.SecClient()
    z = np.load(NPZ)
    labels, member, last = list(z['labels']), z['member'], z['last']
    dates = z['dates'].astype('datetime64[D]')
    T = len(dates)
    month = dates.astype('datetime64[M]')
    tick = {v['ticker'].upper(): int(v['cik_str']) for v in sec.json('https://www.sec.gov/files/company_tickers.json').values()}
    cache = {}

    def sub(cik):
        if cik not in cache:
            try:
                cache[cik] = sec.submissions(cik)
            except RuntimeError:
                cache[cik] = ({}, [])
        return cache[cik]

    out, rows_md = {}, []
    for i, lab in enumerate(labels):
        mem = np.flatnonzero(member[i])
        first_of_month = mem[np.r_[True, month[mem][1:] != month[mem][:-1]]]
        md = dates[first_of_month]
        syms = lab.split('|')
        tried, best = [], None

        def consider(cik, how):
            nonlocal best
            if cik in [c for c, _ in tried]:
                return
            meta, rows = sub(cik)
            cov, last_acc = coverage(rows, md)
            tried.append((cik, how))
            if best is None or cov > best[1]:
                best = (cik, cov, how, meta, last_acc)
        for s in syms:
            for v in (s, s.replace('.', '-'), s.replace('.', '')):
                if v in tick:
                    consider(tick[v], 'ticker table')
        if best is None or best[1] < .9:
            for s in syms:
                try:
                    x = sec.get(f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={s}&type=10-K'
                                f'&dateb=&owner=include&count=10&output=atom').decode('latin-1')
                except RuntimeError:
                    continue
                m = re.search(r'<cik>(\d+)</cik>', x)
                if m:
                    consider(int(m.group(1)), 'EDGAR ticker lookup')
        if best is None or best[1] < .9:
            for s in syms:
                name = alpaca_name(s)
                if not name:
                    continue
                q = re.sub(r'\s+', ' ', SUFFIX.sub(' ', re.sub(r'[^A-Za-z0-9&\. ]', ' ', name))).strip()
                if not q:
                    continue
                hits = sec.json(f'https://efts.sec.gov/LATEST/search-index?keysTyped={q.replace(" ", "%20")}')
                for h in hits.get('hits', {}).get('hits', [])[:5]:
                    consider(int(h['_id']), f'name search ({name})')
        ok = best is not None and best[1] >= .9
        delisted = 0 <= last[i] < T - 5
        cont = bool(ok and delisted and best[4] is not None and (best[4] - dates[last[i]]).astype(int) > 200)
        out[lab] = {'cik': best[0] if ok else None, 'coverage': round(best[1], 3) if best else 0.,
                    'how': best[2] if best else None, 'name': (best[3] or {}).get('name') if best else None,
                    'sic': (best[3] or {}).get('sic') if best else None,
                    'member_months': int(len(md)), 'continues_filing_after_last_bar': cont,
                    'candidates_tried': len(tried)}
        print(f'{i+1}/{len(labels)} {lab}: {out[lab]["cik"]} {out[lab]["name"]} cov {out[lab]["coverage"]} ({out[lab]["how"]})',
              flush=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))

    mm = sum(v['member_months'] for v in out.values())
    mapped = sum(v['member_months'] for v in out.values() if v['cik'])
    unm = [k for k, v in out.items() if not v['cik'] and v['member_months']]
    shared = {}
    for k, v in out.items():
        if v['cik']:
            shared.setdefault(v['cik'], []).append(k)
    multi = {c: ks for c, ks in shared.items() if len(ks) > 1}
    cont = [k for k, v in out.items() if v['continues_filing_after_last_bar']]
    hows = {}
    for v in out.values():
        if v['cik']:
            h = v['how'].split(' (')[0]
            hows[h] = hows.get(h, 0) + 1
    L = ['# CIK map: PIT S&P 500 entities to SEC registrants (PREREG check P2)', '',
         'Generated by `scripts/lazy_cik_map.py`; map in `data/edgar/cik_map.json` (git-ignored). A CIK is accepted '
         'only if an original 10-K/10-Q was accepted within the 200 days before at least 90% of the entity\'s member '
         'months. No filing text or returns were read.', '',
         f'- Entities: {len(out)}; mapped: {sum(1 for v in out.values() if v["cik"])}; by source: '
         + ', '.join(f'{a} {b}' for a, b in sorted(hows.items())) + '.',
         f'- **Member-months mapped: {mapped} / {mm} = {mapped / max(mm, 1):.3f}** (pass condition ≥ 0.90).',
         f'- Unmapped entities with member months ({len(unm)}): '
         + ', '.join(f'{k} ({out[k]["member_months"]} mo, best cov {out[k]["coverage"]})' for k in unm) + '.',
         f'- CIKs shared by several entities (same company under different ticker labels, disjoint membership; one '
         f'extra round trip at the ticker change, no double counting): {len(multi)}: '
         + '; '.join(f'{c}: {"/".join(ks)}' for c, ks in list(multi.items())[:40]) + '.',
         f'- Delisted entities whose CIK keeps filing 10-K/10-Q more than 200 days after the last bar (identity breaks or '
         f'subsidiaries that keep reporting; inspect): {len(cont)}: '
         + ', '.join(f'{k} ({out[k]["name"]})' for k in cont) + '.']
    OUT_MD.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
