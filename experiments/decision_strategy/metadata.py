"""Recover pre-2025 groups; never use the 2026 retrospective labels."""
import csv
import difflib
import re
from pathlib import Path
from bs4 import BeautifulSoup
from experiments.strategy_search.storage import atomic, read, sha, utc
from .config import STUDY


def normalize(name):
    name=re.sub(r'/(?:OH|MD)$','',name).lower().replace('&','and')
    name=re.split(r'\b(?:class|common stock|ordinary shares|ads|adr|registered|nvs)\b',name)[0]
    name=re.sub(r'\([^)]*\)|\*','',name)
    name=re.sub(r'\b(?:incorporated|inc|corp|corporation|co|company|plc|ltd|limited|sa|se|nv|ag|the|holdings|holding)\b','',name)
    return re.sub('[^a-z0-9]','',name)


def recover(output):
    output=Path(output);root=output/'metadata_sources';p=root/'schwab_us_2024.html'
    s=BeautifulSoup(p.read_bytes(),'lxml');rows=s.find_all('tr')
    groups=set()
    for r in rows[:589]:
        cells=[' '.join(x.stripped_strings) for x in r.find_all(['td','th'],recursive=False)]
        if cells and re.fullmatch(r'.+ \d+\.\d+%',cells[0]):
            groups.add(re.sub(r' \d+\.\d+%$','',cells[0]))
    group=None;by_name={}
    for i,r in enumerate(rows):
        cells=[' '.join(x.stripped_strings) for x in r.find_all(['td','th'],recursive=False)]
        if not cells:continue
        title=re.sub(r' \d+\.\d+%$','',cells[0])
        if title in groups:group=title
        elif len(cells)==1 and ('%' in cells[0] or 'TOTAL' in cells[0] or 'COMMON STOCK' in cells[0]):group=None
        elif group and len(cells)==3 and len(cells[0])<180 and re.search('[A-Za-z]',cells[0]) and re.fullmatch('[0-9,]+',cells[1]):
            name=normalize(cells[0]);v=dict(name=cells[0],industry_group=group,source_row=i,source=p.name,
                source_sha256=sha(p) if not by_name else None,published='2024-09-25',effective='2024-07-31')
            by_name.setdefault(name,[]).append(v)
    repo=Path(__file__).parents[2]
    u=list(csv.DictReader((repo/'local-data/metadata/universes/us_equities_250/2026-09-26/universe.csv').open()))
    # Symbol-to-issuer matching only; none of this later file's sector labels is used.
    identities={r['node_id']:r['provider_name'] for r in csv.DictReader((repo/'local-data/metadata/gics/vanguard/2026-10-02/classifications.csv').open())}
    matched=[];missing=[]
    for r in u:
        symbol=r['symbol'];key=normalize(identities[symbol]);hits=by_name.get(key,[])
        if hits and len({h['industry_group'] for h in hits})==1:
            matched.append(dict(symbol=symbol,identity_name=identities[symbol],**hits[0]))
        else:
            near=difflib.get_close_matches(key,by_name,n=3,cutoff=.35)
            missing.append(dict(symbol=symbol,identity_name=identities[symbol],normalized=key,
                suggestions=[by_name[k][0] for k in near],conflicts=hits))
    doc=dict(updated_UTC=utc(),matched=matched,missing=missing,matched_count=len(matched),groups=sorted(groups),
        source_sha256=sha(p),status='REVIEW_REQUIRED',later_taxonomy_values_used=False,
        source='Schwab filing published2024-09-25; table asof2024-07-31; coarser industry GROUPS, not74 GICS industries')
    atomic(output/'metadata_recovery.json',doc)
    print('EXACT_MATCH',len(matched),'MISSING',len(missing),flush=True)
    for r in missing:print(r['symbol'],r['identity_name'],[(x['name'],x['industry_group']) for x in r['suggestions']],flush=True)
    return doc


if __name__=='__main__':recover(STUDY/'decision_strategy_execution_2026-10-27')


SECTOR_GROUPS={
    'Information Technology':['Software & Services','Technology Hardware & Equipment','Semiconductors & Semiconductor Equipment'],
    'Financials':['Banks','Financial Services','Insurance'],
    'Industrials':['Capital Goods','Commercial & Professional Services','Transportation'],
    'Consumer Discretionary':['Automobiles & Components','Consumer Durables & Apparel','Consumer Services','Consumer Discretionary Distribution & Retail'],
    'Consumer Staples':['Consumer Staples Distribution & Retail','Food, Beverage & Tobacco','Household & Personal Products'],
    'Communication Services':['Media & Entertainment','Telecommunication Services'],
    'Health Care':['Health Care Equipment & Services','Pharmaceuticals, Biotechnology & Life Sciences'],
    'Energy':['Energy'],'Materials':['Materials'],'Real Estate':['Equity Real Estate Investment Trusts (REITs)','Real Estate Management & Development'],
    'Utilities':['Utilities'],
}

# These assignments are deliberately coarser BUSINESS GROUPS reconstructed
# from pre-2025 primary disclosures. They are not certified historical GICS
# codes and do not reuse the later classification values. Freeze this choice
# before any development outcomes; report the coarsening and nine judgments.
FOREIGN_EVIDENCE={
    'SAP':('Software & Services','2024-02-29','sap_business_2024.html','https://news.sap.com/2024/02/sap-releases-integrated-report-2023-and-files-annual-report-2023-on-form-20-f/','Enterprise application software business'),
    'LOGI':('Technology Hardware & Equipment','2024-05-16',None,'https://s1.q4cdn.com/104539020/files/doc_financials/2024/ar/annual-report.pdf','Fiscal2024 operating segment Peripherals; indexed report excerpt; original download HTTP403'),
    'ENB':('Energy','2024-08-31',None,'https://meketacapital.com/wp-content/uploads/2024/08/Meketa-6.30.24-N-PORT-Part-F-FINAL.pdf','June2024 schedule places issuer under ENERGY; full PDF read through web tool; original download HTTP404'),
    'TRP':('Energy','2024-08-31',None,'https://meketacapital.com/wp-content/uploads/2024/08/Meketa-6.30.24-N-PORT-Part-F-FINAL.pdf','June2024 schedule places issuer under ENERGY; full PDF read through web tool; original download HTTP404'),
    'AZN':('Pharmaceuticals, Biotechnology & Life Sciences','2024-03-31',None,'https://www.astrazeneca.com/investor-relations/annual-reports/annual-report-2023.html','Annual2023 report describes prescription pharmaceutical business; web text read; original download HTTP403'),
    'NVS':('Pharmaceuticals, Biotechnology & Life Sciences','2024-03-31',None,'https://www.novartis.com/sites/novartis_com/files/novartis-annual-report-2023.pdf','Annual2023 report Chair letter identifies innovative pharmaceutical company; web PDF read'),
    'SNY':('Pharmaceuticals, Biotechnology & Life Sciences','2024-03-31','sanofi_2023.html','https://www.sanofi.com/assets/dotcom/content-app/publications/xbrl-package/ixbrl-viewer-20f-2023-htm.htm','Annual2023 Biopharma business segment'),
    'GSK':('Pharmaceuticals, Biotechnology & Life Sciences','2024-03-31','gsk_2023.pdf','https://www.gsk.com/media/iwxac4h1/annual-report-2023.pdf','Annual2023 focused biopharma company'),
    'UL':('Household & Personal Products','2024-03-31',None,'https://www.unilever.com/files/66bc4aea-608f-46ee-8da3-cde0ec8ebe90/unilever-annual-report-and-accounts-2023.pdf','Annual2023 Beauty,Personal and Home Care revenue38.476bn of59.604bn; majority business rule; indexed excerpt; original403'),
}


def finalize(output):
    output=Path(output);doc=recover(output);rows=doc['matched']
    assert {r['symbol'] for r in doc['missing']}==set(FOREIGN_EVIDENCE)
    for r in doc['missing']:
        group,published,local,url,rationale=FOREIGN_EVIDENCE[r['symbol']]
        rows.append(dict(symbol=r['symbol'],identity_name=r['identity_name'],industry_group=group,published=published,
            effective='2023-12-31',source=local,source_url=url,source_sha256=sha(output/'metadata_sources'/local) if local else None,
            judgment=rationale,official_historical_GICS_assignment=False))
    mapping={g:s for s,gs in SECTOR_GROUPS.items() for g in gs}
    for r in rows:r['sector_name']=mapping[r['industry_group']]
    result=dict(status='verified',verified_UTC=utc(),rows=sorted(rows,key=lambda r:r['symbol']),
        grouping='Coarse pre-2025 business industry groups;241 archived classifications plus9 documented manual business classifications. Not supplier-certified historical GICS.',
        static_grouping_asof='2024-12-31',future_values_used=False,universe_coverage=250,
        identities='Later universe and provider names used only to align the user-mandated250 symbols; later sector/industry VALUES never used.',
        limitation='Fixed2024 groups carried forward; no claim of intraperiod reclassification coverage. Retrospectively fixed250-stock universe remains exposed/survivorship-conditioned.',
        rejected_sources={'blackrock_us_2024.pdf':'URL served October2025 report despite cached search text2024; excluded',
            'vanguard_world_2023.pdf':'Country-sorted holdings lack per-issuer industry; excluded from classification'})
    dest=output/'causal_classifications.json'
    if dest.exists():
        old=read(dest);result['verified_UTC']=old['verified_UTC']
        if old!=result:raise ValueError('Changed classification bindings')
    else:atomic(dest,result)
    return result
