"""Independent scalar metrics and materialized-sample checks of the new table."""
import json
import math
import statistics
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from compare_ndx_strategies_20261020 import OUT,ROOT,sha


def main():
    record=json.loads((OUT/'ANALYSIS.json').read_text())
    for name,value in record['source_hashes'].items():
        if sha(ROOT/name)!=value:raise ValueError('Analysis source data changed: '+name)
    if sha(ROOT/'compare_ndx_strategies_20261020.py')!=record['analysis_source_sha256']:raise ValueError('Analysis code changed')
    table=pd.read_csv(OUT/'all_strategies.csv').set_index('id')
    with np.load(OUT/'analysis_paths.npz') as saved:
        paths=saved['returns'].copy();ids=saved['ids'].copy();dates=saved['dates'].copy()
    if len(dates)!=1003 or len(table)!=267 or table.index.has_duplicates:raise ValueError('Table inventory mismatch')
    for j,name in enumerate(ids):
        a=paths[:,j].tolist();n=len(a);wealth=1.;peak=1.;dd=0.
        for value in a:
            wealth*=1+value;peak=max(peak,wealth);dd=min(dd,wealth/peak-1)
        actual=dict(sharpe=math.sqrt(252)*(math.fsum(a)/n-math.expm1(math.log1p(.03)/252))/statistics.stdev(a),
                    cagr=math.expm1(math.fsum(map(math.log1p,a))*252/n),max_drawdown=dd)
        for field,value in actual.items():np.testing.assert_allclose(value,table.loc[name,field],atol=1e-10,rtol=1e-10)
    selected=['original/SPY/10','original/QQQ/10','original/think_mix/10'];positions=[list(ids).index(name) for name in selected]
    values=paths[:,positions];n=len(values);block=20;repeats=10000;rng=np.random.default_rng(20261020+block);draws=[]
    for first in range(0,repeats,100):
        starts=rng.integers(0,n,(100,int(np.ceil(n/block))))
        indices=((starts[:,:,None]+np.arange(block))%n).reshape(100,-1)[:,:n]
        sample=values[indices]
        draws.append(np.sqrt(252)*(sample.mean(1)-((1.03)**(1/252)-1))/sample.std(1,ddof=1))
    low,high=np.quantile(np.concatenate(draws),[.025,.975],axis=0)
    for j,name in enumerate(selected):
        np.testing.assert_allclose([low[j],high[j]],table.loc[name,['sr_low','sr_high']].to_numpy(float),atol=1e-11,rtol=0)
    wb=OUT/'NASDAQ100_strategy_comparison.xlsx';ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(wb) as archive:
        shared=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings=ET.fromstring(archive.read('xl/sharedStrings.xml'))
            shared=[''.join(element.itertext()) for element in strings.findall('s:si',ns)]
        expected=json.loads((OUT/'workbook_data.json').read_text())
        workbook=ET.fromstring(archive.read('xl/workbook.xml'));sheets=workbook.find('s:sheets',ns)
        if [s.attrib['name'] for s in sheets]!=['Models','Defensive','Portfolio variants','Method']:raise ValueError('Workbook sheet inventory mismatch')
        for index,count in [(1,64),(2,33),(3,174)]:
            xml=ET.fromstring(archive.read(f'xl/worksheets/sheet{index}.xml'))
            entries=xml.findall('s:sheetData/s:row',ns)
            rows={int(row.attrib['r']):row for row in entries}
            if max(rows)!=count+5:raise ValueError('Workbook output count mismatch')
            cells={cell.attrib['r']:cell for row in entries for cell in row.findall('s:c',ns)}
            def value_at(address):
                cell=cells[address];kind=cell.attrib.get('t');value=cell.find('s:v',ns)
                if kind=='s':return shared[int(value.text)]
                if kind=='inlineStr':return ''.join(cell.find('s:is',ns).itertext())
                if kind=='str':return value.text if value is not None else ''
                return float(value.text) if value is not None else None
            for number,row in enumerate(expected['sheets'][index-1]['rows'],6):
                values=[row['strategy'],row['study'],row['sharpe'],row['cagr'],
                        'n.a.' if row['rankic'] is None else row['rankic'],row['max_drawdown'],row['sr_low'],row['sr_high']]
                for column,value in zip('ABCDEFGH',values):
                    actual=value_at(f'{column}{number}')
                    if isinstance(value,str):
                        if actual!=value:raise ValueError('Workbook label differs from audited comparison')
                    else:np.testing.assert_allclose(actual,value,atol=1e-12,rtol=1e-12)
            for position in [6,count//2+6,count+5]:
                for column in ['C','D','F','G','H']:
                    cell=cells[f'{column}{position}']
                    if cell.attrib.get('t') not in [None,'n'] or cell.find('s:v',ns) is None:raise ValueError('Metric exported as nonnumeric')
            if xml.find('s:sheetViews/s:sheetView/s:pane',ns) is None:raise ValueError('Header freeze missing')
    summary=dict(status='passed',strategies_independently_recomputed=len(table),daily_observations=1003,
                 full_materialized_sharpe_bootstraps=selected,bootstrap_replicates=10000,block_sessions=20,
                 workbook_sheets=4,source_hashes_verified=len(record['source_hashes']),
                 workbook_comparison_rows_verified=271,
                 audit_source_sha256=sha(__file__),workbook_builder_sha256=sha(OUT/'build_workbook.mjs'),
                 output_sha256={p.name:sha(p) for p in [wb,OUT/'all_strategies.csv',OUT/'COMPARISON.md',OUT/'sharpe_block_sensitivity.csv']})
    record['status']='completed_audited';record['workbook']='NASDAQ100_strategy_comparison.xlsx'
    (OUT/'ANALYSIS.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (OUT/'AUDIT.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
