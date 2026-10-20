import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const out=path.dirname(fileURLToPath(import.meta.url));
const data=JSON.parse(await fs.readFile(path.join(out,'workbook_data.json'),'utf8'));
const wb=Workbook.create();
const headers=['Strategy','Study','Sharpe','CAGR','RankIC','Max drawdown','SR 95% lower','SR 95% upper'];
const inspections=[];
for (const [index,section] of data.sheets.entries()) {
  const sheet=wb.worksheets.add(section.name);
  sheet.showGridLines=false;
  sheet.tabColor=index===0?'#18344F':'#456582';
  const end=5+section.rows.length;
  const used=sheet.getRange(`A1:H${end}`);
  used.format.font={name:'Arial',size:10,color:'#202D39'};
  used.format.rowHeight=23;
  used.format.verticalAlignment='center';
  const widths=[section.name==='Portfolio variants'?99:66,27,13,14,14,17,17,17];
  for(let j=0;j<widths.length;j++)sheet.getRangeByIndexes(0,j,end,1).format.columnWidth=widths[j];
  sheet.getRange('A2').values=[[section.name==='Models'?'Nasdaq-100 model comparisons':section.name]];
  sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#18344F'};
  sheet.getRange('A2:H2').format.borders={bottom:{style:'thin',color:'#9DAFBD'}};
  sheet.getRange('A3').values=[['2022–2025 • net base-cost returns • 3% Sharpe hurdle • 10,000 block resamples']];
  sheet.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#52606B'};
  sheet.getRange('A5:H5').values=[headers];
  const values=section.rows.map(r=>[r.strategy,r.study,r.sharpe,r.cagr,r.rankic===null?'n.a.':r.rankic,r.max_drawdown,r.sr_low,r.sr_high]);
  sheet.getRange(`A6:H${end}`).values=values;
  const table=sheet.tables.add(`A5:H${end}`,true,`Comparison${index+1}`);
  table.style='TableStyleMedium2';
  table.showFilterButton=true;
  sheet.getRange('A5:H5').format={fill:'#18344F',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
    horizontalAlignment:'center',verticalAlignment:'center',rowHeight:30};
  sheet.getRange(`C6:H${end}`).format.horizontalAlignment='right';
  sheet.getRange(`C6:C${end}`).setNumberFormat('0.000');
  sheet.getRange(`D6:D${end}`).setNumberFormat('0.00%');
  sheet.getRange(`E6:E${end}`).setNumberFormat('0.0000');
  sheet.getRange(`F6:F${end}`).setNumberFormat('0.00%');
  sheet.getRange(`G6:H${end}`).setNumberFormat('0.000');
  sheet.freezePanes.freezeRows(5);
  for(let j=0;j<section.rows.length;j++){
    const r=section.rows[j];
    if(r.strategy==='SPY'||r.strategy==='QQQ')sheet.getRange(`A${j+6}:H${j+6}`).format.font.bold=true;
  }
  inspections.push(await wb.inspect({kind:'table',range:`'${section.name}'!A5:H9`,include:'values,formulas',tableMaxRows:5,tableMaxCols:8,maxChars:3500}));
}
const method=wb.worksheets.add('Method');
method.showGridLines=false;method.tabColor='#9CA9B3';
method.getRange(`A1:B${data.methodology.length+5}`).format.font={name:'Arial',size:10,color:'#202D39'};
method.getRange(`A1:A${data.methodology.length+5}`).format.columnWidth=28;
method.getRange(`B1:B${data.methodology.length+5}`).format.columnWidth=125;
method.getRange('A2').values=[['Method']];
method.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#18344F'};
method.getRange('A2:B2').format.borders={bottom:{style:'thin',color:'#9DAFBD'}};
method.getRange('A5:B5').values=[['Definition','Method and interpretation']];
method.getRange('A5:B5').format={fill:'#18344F',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:28};
method.getRange(`A6:B${data.methodology.length+5}`).values=data.methodology;
method.getRange(`A6:B${data.methodology.length+5}`).format.wrapText=true;
method.getRange(`A6:B${data.methodology.length+5}`).format.rowHeight=52;
method.getRange(`A6:B${data.methodology.length+5}`).format.verticalAlignment='center';
method.getRange(`A6:A${data.methodology.length+5}`).format.font.bold=true;
method.freezePanes.freezeRows(5);

wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options:{useRegex:true,maxResults:100},summary:'Final error scan',maxChars:2500});
await fs.writeFile(path.join(out,'workbook_inspection.json'),JSON.stringify({inspections:inspections.map(r=>r.ndjson),errors:errors.ndjson},null,2));
for(const section of [...data.sheets,{name:'Method'}]){
  const range=section.name==='Method'?'A1:B13':'A1:H15';
  const image=await wb.render({sheetName:section.name,range,scale:1.3,format:'png'});
  await fs.writeFile(path.join(out,`${section.name.replaceAll(' ','_')}_preview.png`),new Uint8Array(await image.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(out,'NASDAQ100_strategy_comparison.xlsx'));
console.log(JSON.stringify({workbook:path.join(out,'NASDAQ100_strategy_comparison.xlsx'),
  rows:data.sheets.map(s=>[s.name,s.rows.length]),errorScan:errors.ndjson}));
