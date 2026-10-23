"""Extraction-robustness diagnostic on real 10-Ks (no returns): OVERNIGHT_RISK_REVIEW.md issues C1 and I1.

  PYTHONPATH=src python docs/overnight_20261024/tools/extraction_probe.py DOCDIR

For each document: does the frozen `edgar.extract_tokens` parse it; how many tokens differ when block elements are
separated by whitespace (a formatting-only change); and the year-over-year cosine under both extractions.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import lxml.html

from hypershift.data import edgar as ED

DECL = re.compile(r'^\s*<\?xml[^>]*\?>', re.S)


def frozen(raw):
    """The frozen extractor; if it raises, the same extractor after stripping the XML declaration (patch C1)."""
    try:
        return ED.extract_tokens(raw), 'ok'
    except ValueError as e:
        txt = raw.decode('utf-8', errors='replace').lstrip('﻿')
        return ED.extract_tokens(DECL.sub('', txt, count=1)), f'frozen raised {type(e).__name__}; declaration stripped'


def spaced(raw):
    """Frozen rules, but every element boundary is a word boundary (whitespace-insensitive to source formatting)."""
    txt = DECL.sub('', raw.decode('utf-8', errors='replace').lstrip('﻿'), count=1)
    root = lxml.html.fromstring(txt)
    drop = [el for el in root.iter() if isinstance(el.tag, str) and (
        el.tag.lower() in ('script', 'style') or el.tag.lower().endswith('header') and ':' in el.tag
        or ED._HIDDEN.search(el.get('style') or ''))]
    for el in drop:
        if el.getparent() is not None:
            el.drop_tree()
    for tab in list(root.iter('table')):
        t = tab.text_content()
        d = sum(c.isdigit() for c in t)
        a = sum(c.isalpha() for c in t)
        if d + a and d / (d + a) > .15 and tab.getparent() is not None:
            tab.drop_tree()
    return Counter(ED._TOKEN.findall(' '.join(root.itertext()).lower()))


def main(docdir):
    docs = {p.name: p.read_bytes() for p in sorted(Path(docdir).glob('*.htm'))}
    res = {}
    for name, raw in docs.items():
        f, note = frozen(raw)
        s = spaced(raw)
        merged = sum(v for k, v in f.items() if k not in s)
        res[name] = (f, s)
        print(f'{name}: {note}; frozen tokens {sum(f.values())}, spaced tokens {sum(s.values())}, '
              f'frozen tokens absent from spaced extraction {merged} ({merged / max(sum(f.values()), 1):.2%}); '
              f'cosine(frozen, spaced) same document {ED.cosine(f, s):.5f}')
    pairs = [('aapl-20230930.htm', 'aapl-20220924.htm'), ('msft-20230630.htm', 'msft-10k_20220630.htm')]
    for a, b in pairs:
        if a in res and b in res:
            print(f'YoY {a} vs {b}: cosine frozen {ED.cosine(res[a][0], res[b][0]):.5f}, '
                  f'spaced {ED.cosine(res[a][1], res[b][1]):.5f}')


if __name__ == '__main__':
    main(sys.argv[1])
