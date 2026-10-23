"""Write the overnight patch files (against a14fd5d) without touching the frozen sources.

  python docs/overnight_20261024/tools/make_patches.py WORKDIR
Copies each source to WORKDIR/{a,b}/..., applies exact string edits to the b/ copy (CRLF preserved), and writes
`diff -u` output to docs/overnight_20261024/patches/. Fails loudly if an anchor is not found exactly once.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'docs/overnight_20261024/patches'

PATCHES = {
    'C1_xml_declaration_and_refetch.patch': {
        'src/hypershift/data/edgar.py': [
            ("_TOKEN = re.compile(r'[a-z]{2,}')\n",
             "_TOKEN = re.compile(r'[a-z]{2,}')\n_XML_DECL = re.compile(r'^\\s*<\\?xml[^>]*\\?>', re.S)\n"),
            ("    root = lxml.html.fromstring(doc)\n",
             "    try:\n"
             "        root = lxml.html.fromstring(doc)\n"
             "    except ValueError:      # str with an XML encoding declaration (DFIN inline XBRL): lxml refuses it\n"
             "        root = lxml.html.fromstring(_XML_DECL.sub('', doc.lstrip('\\ufeff'), count=1))\n"),
        ],
        'scripts/lazy_signals.py': [
            ("    todo = [(c, f) for c, f in plan if not (TOK / f\"{f['accession']}.json.gz\").exists()]\n",
             "    def stored_ok(acc):        # a token file written after a parse error must be fetched and parsed again\n"
             "        p = TOK / f'{acc}.json.gz'\n"
             "        return p.exists() and not load(acc)['error']\n"
             "    todo = [(c, f) for c, f in plan if not stored_ok(f['accession'])]\n"),
        ],
    },
    'I2_prechecks_exclude_errored_rows.patch': {
        'scripts/lazy_checks.py': [
            ("    ntok = np.array([int(r['n_tokens']) for r in recs])\n"
             "    small = [(r['accession'], r['form'], r['n_tokens']) for r in recs if int(r['n_tokens']) < 1000]\n"
             "    errs = [r['accession'] for r in recs if r['error']]\n"
             "    L += [f'**P4 extraction:** {len(recs)} signal filings; parse errors {len(errs)}; documents under 1,000 tokens: '\n",
             "    errs = [r['accession'] for r in recs if r['error']]\n"
             "    recs = [r for r in recs if not r['error']]        # errored pairs carry similarity 0 and 0 tokens\n"
             "    ntok = np.array([int(r['n_tokens']) for r in recs])\n"
             "    small = [(r['accession'], r['form'], r['n_tokens']) for r in recs if int(r['n_tokens']) < 1000]\n"
             "    L += [f'**P4 extraction:** {len(recs) + len(errs)} signal filings; parse errors {len(errs)} (excluded from P4-P6); '\n"
             "          f'documents under 1,000 tokens: '\n"),
        ],
    },
}


def main(work):
    work = Path(work)
    OUT.mkdir(parents=True, exist_ok=True)
    for pname, files in PATCHES.items():
        chunks = []
        for rel, edits in files.items():
            raw = subprocess.run(['git', '-C', str(REPO), 'show', f'a14fd5d:{rel}'], capture_output=True,
                                 check=True).stdout.decode('utf-8')
            crlf = '\r\n' in raw
            text = raw.replace('\r\n', '\n')
            new = text
            for old, rep in edits:
                if new.count(old) != 1:
                    raise SystemExit(f'{pname}: anchor not unique in {rel}: {old[:60]!r}')
                new = new.replace(old, rep)
            a, b = work / 'a' / rel, work / 'b' / rel
            for p, t in ((a, text), (b, new)):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((t.replace('\n', '\r\n') if crlf else t).encode('utf-8'))
            d = subprocess.run(['diff', '-u', '--label', f'a/{rel}', '--label', f'b/{rel}', str(a), str(b)],
                               capture_output=True).stdout.decode('utf-8')
            chunks.append(d)
        (OUT / pname).write_bytes(''.join(chunks).encode('utf-8'))
        print('wrote', OUT / pname)


if __name__ == '__main__':
    main(sys.argv[1])
