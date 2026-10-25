"""Edit report tables in place, preserving all fonts, images and OOXML parts."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
import csv
import json
import hashlib

ROOT = Path('docs/largecap_retraining_20261010')
DOC = ROOT / 'Hypershift_Large_Company_Retraining_Results.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + NS['w'] + '}'


def text(e):
    return ''.join(e.xpath('.//w:t/text()', namespaces=NS))


def replace(e, value):
    nodes = e.xpath('.//w:t', namespaces=NS)
    assert nodes
    nodes[0].text = value
    for n in nodes[1:]:
        n.text = ''


def main():
    with (ROOT / 'primary_results.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    bykey = {(r['dataset'].split('_')[0], 'Five-session' if r['procedure'] == 'Five-session' else 'Daily'): r for r in rows}
    with ZipFile(DOC) as z:
        original = {i.filename: (i, z.read(i.filename)) for i in z.infolist()}
    xml = etree.fromstring(original['word/document.xml'][1])
    tables = xml.xpath('./w:body/w:tbl', namespaces=NS)
    assert len(tables) == 13
    for index in [1, 2]:
        table_rows = tables[index].xpath('./w:tr', namespaces=NS)
        header = table_rows[0].xpath('./w:tc', namespaces=NS)
        replace(header[3], 'Benchmark Sharpe' if index == 1 else 'Gain over benchmark')
        for tr in table_rows[1:]:
            c = tr.xpath('./w:tc', namespaces=NS)
            group = 'NYSE' if 'NYSE' in text(c[0]) else 'NASDAQ100'
            procedure = 'Five-session' if 'Five day' in text(c[0]) else 'Daily'
            r = bykey[(group, procedure)]
            replace(c[3], f'{float(r["benchmark_sharpe" if index == 1 else "delta_benchmark"]):.2f}')
            if index == 2:
                replace(c[4], f'{float(r["delta_benchmark_ci_low"]):.2f} to {float(r["delta_benchmark_ci_high"]):.2f}')
    for index in [5, 7]:
        for tr in list(tables[index].xpath('./w:tr', namespaces=NS)):
            c = tr.xpath('./w:tc', namespaces=NS)
            if text(c[0]) == 'QQQ':
                tables[index].remove(tr)
            elif text(c[0]) == 'Rebalance the full group':
                replace(c[0], 'NYSE basket (passive)')
        assert not any(text(tr.xpath('./w:tc', namespaces=NS)[0]) == 'QQQ'
                       for tr in tables[index].xpath('./w:tr', namespaces=NS))
    for p in xml.xpath('./w:body/w:p', namespaces=NS):
        t = text(p)
        if t.startswith('We completed 648 model training runs'):
            replace(p, t.replace('both SPY, a broad US market fund, and QQQ, a NASDAQ-100 fund,',
                                 'SPY and the relevant cohort benchmark (passive NYSE basket or QQQ),'))
        elif t.startswith('For each test year, the model and trading rule'):
            replace(p, 'Earlier validation selected each model and trading rule. Tables compare NYSE models with '
                       'a passive basket of all eligible NYSE stocks, and NASDAQ models with QQQ. SPY provides '
                       'broad-market context. These charts retain the original fund overlays.')
        elif t.startswith('Total return combines the saved annual-fold returns.'):
            replace(p, 'Benchmark means the passive NYSE basket for NYSE and QQQ for NASDAQ. The basket is equally '
                       'weighted at scheduled rebalances, with modeled costs. Total return combines annual folds; '
                       'worst fall is the largest historical drop from an earlier high.')
        elif 'both comparison funds' in t:
            replace(p, t.replace('both comparison funds', 'SPY and the relevant cohort benchmark'))
        elif t.startswith('These ranges estimate uncertainty'):
            replace(p, t.replace('model and funds', 'model and benchmarks'))
    replacement = etree.tostring(xml, xml_declaration=True, encoding='UTF-8', standalone=True)
    pending = DOC.with_suffix('.pending.docx')
    with ZipFile(pending, 'w', ZIP_DEFLATED) as z:
        for name, (info, body) in original.items():
            z.writestr(info, replacement if name == 'word/document.xml' else body)
    with ZipFile(pending) as z:
        assert z.testzip() is None
        assert all(z.read(name) == body for name, (_, body) in original.items() if name != 'word/document.xml')
    pending.replace(DOC)
    # Refresh the downloadable data archive so it contains the updated tables.
    package = ROOT / 'Hypershift_Large_Company_Results_Data.zip'
    with ZipFile(package) as z:
        contents = [(i, z.read(i.filename)) for i in z.infolist()]
    pending_zip = package.with_suffix('.pending.zip')
    comparison = Path('runs/largecap_20261010/comparison')
    with ZipFile(pending_zip, 'w', ZIP_DEFLATED) as z:
        for info, body in contents:
            candidate = ROOT / info.filename if info.filename == 'primary_results.csv' else comparison / info.filename
            z.writestr(info, candidate.read_bytes() if candidate.exists() else body)
    with ZipFile(pending_zip) as z:
        assert z.testzip() is None
    pending_zip.replace(package)
    print(json.dumps({'docx_updated': str(DOC), 'zip_updated': str(package),
                      'fonts_and_images_preserved': True}, indent=2))


if __name__ == '__main__':
    main()
