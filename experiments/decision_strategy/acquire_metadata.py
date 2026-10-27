"""Cache bounded, public pre-2025 classification evidence without credentials."""
from pathlib import Path
import urllib.request
from experiments.strategy_search.storage import atomic, sha, utc
from .config import STUDY

SOURCES = [
    ('sap_business_2024.html','https://news.sap.com/2024/02/sap-releases-integrated-report-2023-and-files-annual-report-2023-on-form-20-f/','2024-02-29'),
    ('logitech_2024.pdf','https://s1.q4cdn.com/104539020/files/doc_financials/2024/ar/annual-report.pdf','2024-05-16'),
    ('energy_2024.pdf','https://meketacapital.com/wp-content/uploads/2024/08/Meketa-6.30.24-N-PORT-Part-F-FINAL.pdf','2024-08-31'),
    ('astrazeneca_2023.html','https://www.astrazeneca.com/investor-relations/annual-reports/annual-report-2023.html','2024-03-31'),
    ('sanofi_2023.html','https://www.sanofi.com/assets/dotcom/content-app/publications/xbrl-package/ixbrl-viewer-20f-2023-htm.htm','2024-03-31'),
    ('gsk_2023.pdf','https://www.gsk.com/media/iwxac4h1/annual-report-2023.pdf','2024-03-31'),
    ('unilever_2023.pdf','https://www.unilever.com/files/66bc4aea-608f-46ee-8da3-cde0ec8ebe90/unilever-annual-report-and-accounts-2023.pdf','2024-03-31'),
    ('m_funds_2024.html','https://content-archive.fast-edgar.com/20240516/AP2LKQ2DZ2229ZZ2222G2ZOELQABZQ22J262/NPORT_1TM3_42748965_0324.htm','2024-05-16'),
    ('goldman_international_2024.pdf','https://am.gs.com/public-assets/documents/aaaf297b-a397-11ef-9531-3769b6f0cf37?view=true','2024-11-15'),
    ('vanguard_world_2023.pdf','https://fund-docs.vanguard.com/q6280.pdf','2023-12-26'),
    ('vanguard_world_2023.html','https://www.sec.gov/Archives/edgar/data/857489/000110465923130096/tm2331781d7_ncsr.htm','2023-12-26'),
    ('schwab_us_2024.html','https://content-archive.fast-edgar.com/20240925/A422D22EZC226JZZ232R2ZOA8I3GOQ22Z262/NPORT_2MU5_86144328_0724.htm','2024-09-25'),
    ('blackrock_us_2024.pdf','https://www.blackrock.com/us/individual/literature/first-quarter-report/fqr-retail-ishares-total-us-stock-market-index-fund.pdf','2024-12-31'),
]


def main():
    root=STUDY/'decision_strategy_execution_2026-10-27'/'metadata_sources';root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,url,published in SOURCES:
        p=root/name
        try:
            if not p.exists():
                req=urllib.request.Request(url,headers={'User-Agent':'Hypershift Academic Research Ethan Cochran contact via acm-research-f26 on GitHub'})
                with urllib.request.urlopen(req,timeout=45) as response:
                    data=response.read(40_000_001)
                if len(data)>40_000_000:raise ValueError('Metadata source exceeds bounded40MB')
                p.write_bytes(data)
            row=dict(name=name,url=url,published_candidate=published,bytes=p.stat().st_size,sha256=sha(p),status='downloaded_requires_date_and_classification_verification')
        except Exception as e:
            row=dict(name=name,url=url,status='failed',error=str(e))
        rows.append(row);print(row,flush=True)
        atomic(root/'acquisition.json',dict(updated_UTC=utc(),sources=rows))


if __name__=='__main__':main()
