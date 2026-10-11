from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json

names=['01-data-sources.png','02-yearly-tests.png','03-model-inputs.png',
       '04-forecast-targets.png','05-data-limits.png','06-reading-results.png']
output=Path('presentations/output')
records=[]
for suffix,render,deck in [('', 'style_benchmark_after','Hypershift_Datasets_Canva_Style_v3.pptx'),
                          ('_Poppins','poppins_benchmark_after','Hypershift_Datasets_Canva_Poppins.pptx')]:
    parts={name:Path(f'tmp/{render}/slide-{i}.png').read_bytes() for i,name in enumerate(names,1)}
    parts['README.txt']=(
        'Rendered from the corrected published deck. NYSE uses a simulated cohort basket; '
        'NASDAQ uses QQQ; SPY supplies market context. NYSX launched March 25, 2026 '
        'and has no 2022-2025 ETF history. Its index spans exchanges. '
        'Fresh trial results are in docs/random_trials_20261010/CHARTS.md.\n'
    ).encode('utf-8')
    archive=output/f'Hypershift_Dataset_Slide_Images{suffix}.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for name,body in parts.items():z.writestr(name,body)
    folders=[output/'Hypershift_Dataset_Slide_Images_Poppins',output/'Hypershift_Canva_PNGs_Poppins'] if suffix else []
    for folder in folders:
        for name,body in parts.items():(folder/name).write_bytes(body)
    records.append(dict(deck=str(output/deck),deck_sha256=hashlib.sha256((output/deck).read_bytes()).hexdigest(),
                        archive=str(archive),folders=list(map(str,folders)),
                        members={name:hashlib.sha256(body).hexdigest() for name,body in parts.items()}))
Path('docs/audit/slide_export_update.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Refreshed both published PNG archives and both Poppins export folders from verified corrected renders')
