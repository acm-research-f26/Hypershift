"""Small deterministic OOXML edits using bundled runtime, preserving layouts."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib
import json
from lxml import etree

def replace_text(element,value):
    nodes=element.xpath('.//*[local-name()="t"]')
    if nodes:
        nodes[0].text=value
        for n in nodes[1:]:n.text=''

def write_parts(path,parts):
    result=path
    pending=path.with_suffix('.benchmark-pending'+path.suffix)
    with ZipFile(pending,'w',ZIP_DEFLATED) as z:
        for info,body in parts:z.writestr(info,body)
    with ZipFile(pending) as z:assert z.testzip() is None
    # Some OneDrive files allow writing but deny rename-over-existing/delete.
    try:
        path.write_bytes(pending.read_bytes())
    except PermissionError:
        target=path.with_name(path.stem+'_Benchmark_Corrected'+path.suffix)
        target.write_bytes(pending.read_bytes())
        result=target
        print('Locked original; saved corrected copy:',target)
    pending.unlink(missing_ok=True)
    return result

def main():
    records=[]
    doc=Path('docs/largecap_retraining_20261010/Hypershift_Large_Company_Retraining_Results.docx')
    with ZipFile(doc) as z:parts=[(i,z.read(i.filename)) for i in z.infolist()]
    for i,(info,body) in enumerate(parts):
        if info.filename=='word/document.xml':
            xml=etree.fromstring(body)
            for p in xml.xpath('//*[local-name()="body"]/*[local-name()="p"]'):
                t=''.join(p.xpath('.//*[local-name()="t"]/text()'))
                if t.startswith('Earlier validation selected each model'):
                    replace_text(p,'Earlier validation selected each model and trading rule. NYSE charts use a simulated cohort basket; NASDAQ charts use QQQ. NYSX tracks the official NYSE 100 growth index across exchanges, launched March 25, 2026, and has no 2022–2025 ETF history.')
                elif t.startswith('Benchmark means the passive NYSE basket'):
                    replace_text(p,'Benchmark means the simulated NYSE cohort basket for NYSE and QQQ for NASDAQ. The basket is equally weighted at scheduled rebalances with modeled costs. NYSX performance is unavailable in these test years. Total returns combine annual folds.')
            parts[i]=(info,etree.tostring(xml,xml_declaration=True,encoding='UTF-8',standalone=True))
        elif info.filename in ['word/media/image2.png','word/media/image3.png']:
            h=5 if info.filename.endswith('image2.png') else 1
            parts[i]=(info,Path(f'tmp/benchmark_artifact_images/procedure_h{h}.png').read_bytes())
    target=write_parts(doc,parts);records.append(dict(path=str(target),original=str(doc),original_replaced=target==doc))
    decks=[p for p in list(Path('docs/slides').glob('*.pptx'))+list(Path('presentations/output').glob('*.pptx')) if 'Benchmark_Corrected' not in p.stem]
    for deck in decks:
        with ZipFile(deck) as z:parts=[(i,z.read(i.filename)) for i in z.infolist()]
        changed=[]
        for i,(info,body) in enumerate(parts):
            if not info.filename.endswith('.xml') or not info.filename.startswith(('ppt/slides/slide','ppt/notesSlides/notesSlide')):continue
            xml=etree.fromstring(body);modified=False
            for node in xml.xpath('//*[local-name()="t"]'):
                values={'SPY and QQQ':'Cohort controls','2 ETFs':'3 controls','Both available':'2022–2025',
                        'Funds tracking the S&P 500':'NYSE: simulated basket.',
                        'and Nasdaq-100. We compare':'NASDAQ: QQQ. SPY: context.',
                        'results on the same dates.':'NYSX: no 2022–2025 history.',
                        'COHORT CONTROLS':'CONTROLS','SPY / QQQ':'CONTROLS'}
                if node.text in values:node.text=values[node.text];modified=True
            for shape in xml.xpath('//*[local-name()="sp"]'):
                t=' '.join(shape.xpath('.//*[local-name()="t"]/text()'))
                value=None
                if t.startswith('Controls: SPY'):
                    value='NYSE: simulated basket. NASDAQ: QQQ. SPY: broad market. NYSX has no 2022–2025 history.'
                elif t.startswith('SPY and QQQ use the same'):
                    value='Cohort references use matched dates: a simulated basket for NYSE, QQQ for NASDAQ and SPY for context. NYSX launched March 25, 2026.'
                elif t.startswith('SPY tracks the S&P 500. QQQ tracks'):
                    value='NYSE uses a simulated basket. NASDAQ uses QQQ. SPY supplies context. NYSX launched in March 2026 and has no test-period history.'
                elif t=='SPY and QQQ':value='Cohort controls'
                elif t=='2 ETFs':value='3 controls'
                elif t=='Both available':value='2022–2025'
                elif t in ['SPY / QQQ','COHORT CONTROLS']:value='CONTROLS'
                elif t=='Funds tracking the S&P 500':value='NYSE: simulated basket.'
                elif t=='and Nasdaq-100. We compare':value='NASDAQ: QQQ. SPY: context.'
                elif t=='results on the same dates.':value='NYSX: no 2022–2025 history.'
                elif t=='Matched ETF controls':value='Matched cohort controls'
                if value is not None:replace_text(shape,value);modified=True
            if modified:
                parts[i]=(info,etree.tostring(xml,xml_declaration=True,encoding='UTF-8',standalone=True));changed.append(info.filename)
        target=write_parts(deck,parts);records.append(dict(path=str(target),changed_parts=changed))
    Path('docs/audit/benchmark_artifact_update.json').write_text(json.dumps(dict(updated=records,
        preserved='Existing tables, slide positions, fonts and all unrelated OOXML parts; document procedure images regenerated with correct cohort references',
        NYSX_history='unavailable before 2026-03-25'),indent=2))
    print('Updated existing DOCX and',len(decks),'published PPTX decks')

if __name__=='__main__':main()
