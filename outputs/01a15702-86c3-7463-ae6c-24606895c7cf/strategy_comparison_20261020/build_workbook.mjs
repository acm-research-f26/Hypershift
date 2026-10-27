import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile,FileBlob} from '@oai/artifact-tool';

const out=path.dirname(fileURLToPath(import.meta.url));
if(process.argv.includes('--new-findings') || process.argv.includes('--inspect-new')){
  const payload=JSON.parse(await fs.readFile(path.join(out,'new_findings_workbook_data.json'),'utf8'));
  const crypto=await import('node:crypto');
  if(crypto.createHash('sha256').update(await fs.readFile(payload.input_workbook)).digest('hex')!==payload.input_sha256)throw Error('Archived current workbook changed');
  const book=await SpreadsheetFile.importXlsx(await FileBlob.load(payload.input_workbook));
  const previews=path.dirname(path.dirname(payload.input_workbook));
  const savePreview=async(name,range,label)=>{
    const png=await book.render({sheetName:name,range,scale:1.1,format:'png'});
    await fs.writeFile(path.join(previews,`new_findings_${label}_preview.png`),new Uint8Array(await png.arrayBuffer()));
  };
  if(process.argv.includes('--inspect-new')){
    for(const [name,range,label] of [['Models','A1:H14','before_models'],['Models','A68:H77','before_tail'],['Portfolio variants','A175:H184','before_portfolio']])await savePreview(name,range,label);
    const inspection=await book.inspect({kind:'workbook,sheet,table',maxChars:5000,tableMaxRows:3,tableMaxCols:8});
    await fs.writeFile(path.join(previews,'before_inspection.ndjson'),inspection.ndjson);
    console.log(inspection.ndjson);process.exit(0);
  }
  const changes=[];
  for(const name of ['Models','Portfolio variants']){
    const sheet=book.worksheets.getItem(name),table=sheet.tables.items[0];
    const begin=payload.existing_counts[name]+6;
    const rows=payload.additions.filter(r=>r.section===name);
    table.rows.add(null,rows.map(r=>[r.strategy,r.study,r.sharpe,r.cagr,r.rankic,r.max_drawdown,r.sr_low,r.sr_high]));
    const end=begin+rows.length-1;
    const range=sheet.getRange(`A${begin}:H${end}`);
    range.format.font={name:'Arial',size:10,color:'#202D39'};range.format.rowHeight=36;range.format.verticalAlignment='center';
    sheet.getRange(`A${begin}:B${end}`).format.wrapText=true;
    sheet.getRange(`C${begin}:H${end}`).format.horizontalAlignment='right';
    for(const [col,fmt] of [['C','0.000'],['D','0.00%'],['E','0.0000'],['F','0.00%'],['G','0.000'],['H','0.000']])sheet.getRange(`${col}${begin}:${col}${end}`).setNumberFormat(fmt);
    changes.push({name,begin,end});
  }
  const method=book.worksheets.getItem('Method');
  const noteStart=34,noteEnd=noteStart+payload.notes.length-1;
  method.getRange(`A${noteStart}:B${noteEnd}`).values=payload.notes;
  method.getRange(`A${noteStart}:B${noteEnd}`).format.font={name:'Arial',size:10,color:'#202D39'};
  method.getRange(`A${noteStart}:B${noteEnd}`).format.wrapText=true;
  method.getRange(`A${noteStart}:B${noteEnd}`).format.rowHeight=68;
  method.getRange(`A${noteStart}:B${noteEnd}`).format.verticalAlignment='center';
  method.getRange(`A${noteStart}:A${noteEnd}`).format.font.bold=true;
  const createTable=(name,title,subtitle,headers,rows,widths,tableName)=>{
    const sheet=book.worksheets.add(name),last=rows.length+5,letter=String.fromCharCode(64+headers.length);
    sheet.showGridLines=false;sheet.tabColor='#456582';
    sheet.getRange(`A1:${letter}${last}`).format.font={name:'Arial',size:10,color:'#202D39'};
    sheet.getRange(`A1:${letter}${last}`).format.rowHeight=30;
    sheet.getRange(`A1:${letter}${last}`).format.verticalAlignment='center';
    for(const [i,width] of widths.entries())sheet.getRangeByIndexes(0,i,last,1).format.columnWidth=width;
    sheet.getRange('A2').values=[[title]];sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#18344F'};
    sheet.getRange(`A2:${letter}2`).format.borders={bottom:{style:'thin',color:'#9DAFBD'}};
    sheet.getRange('A3').values=[[subtitle]];sheet.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#52606B'};
    sheet.getRange(`A5:${letter}5`).values=[headers];sheet.getRange(`A6:${letter}${last}`).values=rows;
    const table=sheet.tables.add(`A5:${letter}${last}`,true,tableName);table.style='TableStyleMedium2';table.showFilterButton=true;
    sheet.getRange(`A5:${letter}5`).format={fill:'#18344F',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:42,wrapText:true};
    sheet.getRange(`A6:A${last}`).format.wrapText=true;
    sheet.freezePanes.freezeRows(5);sheet.freezePanes.freezeColumns(1);return sheet;
  };
  const na=v=>v===null?'n.a.':v;
  const summary=createTable('New findings','New findings and seed comparisons','2022–2025 development history. Ten seeds. Top ten primary; top five secondary. Base costs.',
    ['Strategy','Holdings','Sharpe','CAGR','RankIC','Max drawdown','SR 95% lower','SR 95% upper','Original-five SR','New-five SR','Original-five CAGR','New-five CAGR'],
    payload.summary.map(r=>[r.strategy,r.k,r.sharpe,r.cagr,na(r.rankic),r.max_drawdown,r.sr_low,r.sr_high,na(r.original_sr),na(r.new_sr),na(r.original_cagr),na(r.new_cagr)]),
    [70,10,12,13,12,15,16,16,18,18,19,19],'NewFindings');
  summary.getRange('B6:L31').format.horizontalAlignment='right';
  for(const [col,fmt] of [['B','0'],['C','0.000'],['D','0.00%'],['E','0.0000'],['F','0.00%'],['G','0.000'],['H','0.000'],['I','0.000'],['J','0.000'],['K','0.00%'],['L','0.00%']])summary.getRange(`${col}6:${col}31`).setNumberFormat(fmt);
  const explanation=[
    [35,'How the dynamic rank blend works'],
    [36,'It retains THINK+mixer AH+PM and adds a separately trained restricted dynamic hypergraph network.'],
    [37,'Each branch averages stock percentile ranks across its fixed seeds. Final score = 0.5 × THINK rank + 0.5 × dynamic rank.'],
    [38,'One equal-weighted top-five or top-ten portfolio trades at the regular close every five exchange sessions.'],
    [40,'Evidence and limits'],
    [41,'Seed halves share one market history. Five holdings are secondary. A higher historical Sharpe does not establish benchmark superiority.'],
    [42,'Intervals are absolute SR intervals with a 3% hurdle. Paired and adjusted intervals remain inconclusive for a reliable edge.'],
    [43,'Author networks use declared NASDAQ feature/training adaptations. Dynamic graph inputs and JKP factor publication timing remain limitations.'],
    [45,'Sources and definitions are in Method. New accounts retains every account from both audited batches.']
  ];
  for(const [row,value] of explanation){summary.getRange(`A${row}`).values=[[value]];summary.getRange(`A${row}:L${row}`).format.font={name:'Arial',size:10,color:'#202D39'};summary.getRange(`A${row}:L${row}`).format.rowHeight=23;}
  summary.getRange('A35').format.font.bold=true;summary.getRange('A40').format.font.bold=true;
  const details=createTable('New accounts','All new experiment accounts','346 declared accounts. Includes matched controls, original/new seed halves, individual seeds and 2×/4× fees.',
    ['Strategy','Study','Seed group','Holdings','Fee multiple','Seed','Sharpe','CAGR','RankIC','Max drawdown','SR 95% lower','SR 95% upper','Role'],
    payload.details.map(r=>[r.strategy,r.study,r.group,r.k,r.cost_multiplier,r.seed,r.sharpe,r.cagr,na(r.rankic),r.max_drawdown,r.sr_low,r.sr_high,r.role]),
    [76,29,19,10,13,17,12,13,12,15,16,16,33],'NewExperimentAccounts');
  details.getRange('A6:M351').format.rowHeight=36;details.getRange('B6:B351').format.wrapText=true;
  details.getRange('D6:L351').format.horizontalAlignment='right';
  for(const [col,fmt] of [['D','0'],['E','0"x"'],['F','@'],['G','0.000'],['H','0.00%'],['I','0.0000'],['J','0.00%'],['K','0.000'],['L','0.000']])details.getRange(`${col}6:${col}351`).setNumberFormat(fmt);
  book.recalculate();
  const inspection=await book.inspect({kind:'table',range:"'New findings'!A5:L12",include:'values,formulas',tableMaxRows:8,tableMaxCols:12,maxChars:5000});
  const errors=await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},maxChars:2000});
  await fs.writeFile(path.join(previews,'new_findings_inspection.json'),JSON.stringify({changes,noteStart,noteEnd,inspection:inspection.ndjson,errors:errors.ndjson},null,2));
  const views=process.argv.includes('--reuse-new-previews')?[]:[['New findings','A1:L13','summary'],['New findings','A35:L45','explanation'],['New accounts','A1:M12','accounts'],['Method',`A${noteStart}:B${noteEnd}`,'method']];
  for(const [name,range,label] of [...views,...changes.map(r=>[r.name,`A${r.begin}:H${r.begin+8}`,r.name.replaceAll(' ','_')])])await savePreview(name,range,label);
  const file=await SpreadsheetFile.exportXlsx(book);await file.save(payload.output_workbook);
  console.log(JSON.stringify({updated:true,changes,seedComparisonRows:26,newAccounts:346,output:payload.output_workbook,errors:errors.ndjson}));
  process.exit(0);
}
if(process.argv.includes('--top5') || JSON.parse(await fs.readFile(path.join(out,'workbook_data.json'),'utf8')).top5_accounts){
  const root=path.resolve(out,'../../..');
  const payload=JSON.parse(await fs.readFile(path.join(out,'top5_workbook_data.json'),'utf8'));
  const crypto=await import('node:crypto');
  if(crypto.createHash('sha256').update(await fs.readFile(payload.input_workbook)).digest('hex')!==payload.input_sha256)throw Error('Archived workbook changed');
  const book=await SpreadsheetFile.importXlsx(await FileBlob.load(payload.input_workbook));
  const portfolio=book.worksheets.getItem('Portfolio variants');
  portfolio.tables.items[0].rows.add(null,payload.additions.map(r=>[r.strategy,r.study,r.sharpe,r.cagr,r.rankic,r.max_drawdown,r.sr_low,r.sr_high]));
  const begin=180,end=184;
  portfolio.getRange(`A${begin}:H${end}`).format.font={name:'Arial',size:10,color:'#202D39'};
  portfolio.getRange(`A${begin}:H${end}`).format.rowHeight=23;
  portfolio.getRange(`A${begin}:H${end}`).format.verticalAlignment='center';
  portfolio.getRange(`C${begin}:H${end}`).format.horizontalAlignment='right';
  for(const [column,format] of [['C','0.000'],['D','0.00%'],['E','0.0000'],['F','0.00%'],['G','0.000'],['H','0.000']])portfolio.getRange(`${column}${begin}:${column}${end}`).setNumberFormat(format);
  const method=book.worksheets.getItem('Method');
  const methodology=JSON.parse(await fs.readFile(path.join(out,'workbook_data.json'),'utf8')).methodology;
  method.getRange('A32:B33').values=methodology.slice(-2);
  method.getRange('A32:B33').format.font={name:'Arial',size:10,color:'#202D39'};
  method.getRange('A32:B33').format.wrapText=true;method.getRange('A32:B33').format.rowHeight=52;
  method.getRange('A32:B33').format.verticalAlignment='center';method.getRange('A32:A33').format.font.bold=true;
  const sheet=book.worksheets.add('AH PM holding counts');sheet.showGridLines=false;sheet.tabColor='#456582';
  const last=payload.details.length+5;
  sheet.getRange(`A1:K${last}`).format.font={name:'Arial',size:10,color:'#202D39'};
  sheet.getRange(`A1:K${last}`).format.rowHeight=23;sheet.getRange(`A1:K${last}`).format.verticalAlignment='center';
  for(const [i,width] of [66,22,11,13,17,13,14,14,17,17,17].entries())sheet.getRangeByIndexes(0,i,last,1).format.columnWidth=width;
  sheet.getRange('A2').values=[['After-hours + premarket: five versus ten holdings']];
  sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#18344F'};
  sheet.getRange('A2:K2').format.borders={bottom:{style:'thin',color:'#9DAFBD'}};
  sheet.getRange('A3').values=[['2022–2025 • fixed saved forecasts • all 62 accounts • 3% SR hurdle • no new training']];
  sheet.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#52606B'};
  sheet.getRange('A5:K5').values=[['Strategy','Seed group','Holdings','Fee multiple','Seed','Sharpe','CAGR','RankIC','Max drawdown','SR 95% lower','SR 95% upper']];
  sheet.getRange(`A6:K${last}`).values=payload.details.map(r=>[r.strategy,r.group,r.k,r.cost_multiplier,r.seed,r.sharpe,r.cagr,r.rankic===null?'n.a.':r.rankic,r.max_drawdown,r.sr_low,r.sr_high]);
  const table=sheet.tables.add(`A5:K${last}`,true,'AHPMHoldingCounts');table.style='TableStyleMedium2';table.showFilterButton=true;
  sheet.getRange('A5:K5').format={fill:'#18344F',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:30};
  sheet.getRange(`C6:K${last}`).format.horizontalAlignment='right';
  for(const [col,fmt] of [['C','0'],['D','0"x"'],['F','0.000'],['G','0.00%'],['H','0.0000'],['I','0.00%'],['J','0.000'],['K','0.000']])sheet.getRange(`${col}6:${col}${last}`).setNumberFormat(fmt);
  sheet.freezePanes.freezeRows(5);book.recalculate();
  const inspection=await book.inspect({kind:'table',range:"'AH PM holding counts'!A5:K12",include:'values,formulas',tableMaxRows:8,tableMaxCols:11,maxChars:3500});
  const errors=await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},summary:'Top-five update error scan',maxChars:1500});
  await fs.writeFile(path.join(out,'workbook_inspection.json'),JSON.stringify({inspection:inspection.ndjson,errors:errors.ndjson},null,2));
  for(const [name,range,label] of [['Portfolio variants','A178:H184','new_rows'],['AH PM holding counts','A1:K14','top'],['AH PM holding counts','A24:K34','requested_model'],['Method','A32:B33','notes']]){
    const preview=await book.render({sheetName:name,range,scale:1.1,format:'png'});
    await fs.writeFile(path.join(out,`top5_${label}_preview.png`),new Uint8Array(await preview.arrayBuffer()));
  }
  const output=await SpreadsheetFile.exportXlsx(book);await output.save(path.join(root,'docs/NASDAQ100_strategy_comparison.xlsx'));
  console.log(JSON.stringify({updated:true,addedMainStrategies:5,followupAccounts:62,errorScan:errors.ndjson}));
  process.exit(0);
}
const data=JSON.parse(await fs.readFile(path.join(out,'workbook_data.json'),'utf8'));
if(data.updated_date==='2026-10-24'){
  const root=path.resolve(out,'../../..');
  const prior=path.join(root,'.cache/strategy_comparison_20261024_previous');
  const original=JSON.parse(await fs.readFile(path.join(prior,'workbook_data.json'),'utf8'));
  const book=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(prior,'NASDAQ100_strategy_comparison.xlsx')));
  const originalModels=original.sheets[0].rows.length;
  const additions=data.sheets[0].rows.slice(originalModels);
  if(additions.length!==8)throw Error('Expected exactly eight new ensemble rows');
  const models=book.worksheets.getItem('Models');
  models.tables.items[0].rows.add(null,additions.map(r=>[r.strategy,r.study,r.sharpe,r.cagr,r.rankic,r.max_drawdown,r.sr_low,r.sr_high]));
  const first=originalModels+6,last=originalModels+additions.length+5;
  models.getRange(`A${first}:H${last}`).format.font={name:'Arial',size:10,color:'#202D39'};
  models.getRange(`A${first}:H${last}`).format.rowHeight=23;
  models.getRange(`A${first}:H${last}`).format.verticalAlignment='center';
  models.getRange(`C${first}:H${last}`).format.horizontalAlignment='right';
  for(const [column,format] of [['C','0.000'],['D','0.00%'],['E','0.0000'],['F','0.00%'],['G','0.000'],['H','0.000']])models.getRange(`${column}${first}:${column}${last}`).setNumberFormat(format);
  models.getRange('A3').values=[['2022–2025 • updated October 24 • base costs • 3% SR hurdle • 10,000 block resamples']];
  const method=book.worksheets.getItem('Method');
  method.getRange(`A6:B${data.methodology.length+5}`).values=data.methodology;
  const methodFirst=original.methodology.length+6,methodLast=data.methodology.length+5;
  method.getRange(`A${methodFirst}:B${methodLast}`).format.font={name:'Arial',size:10,color:'#202D39'};
  method.getRange(`A${methodFirst}:B${methodLast}`).format.wrapText=true;
  method.getRange(`A${methodFirst}:B${methodLast}`).format.rowHeight=52;
  method.getRange(`A${methodFirst}:B${methodLast}`).format.verticalAlignment='center';
  method.getRange(`A${methodFirst}:A${methodLast}`).format.font.bold=true;
  const detail=book.worksheets.add('Extended-hours');
  detail.showGridLines=false;detail.tabColor='#456582';
  const end=5+data.extended_hours.length;
  detail.getRange(`A1:J${end}`).format.font={name:'Arial',size:10,color:'#202D39'};
  detail.getRange(`A1:J${end}`).format.rowHeight=23;
  detail.getRange(`A1:J${end}`).format.verticalAlignment='center';
  for(const [i,width] of [66,29,17,14,13,14,14,17,17,17].entries())detail.getRangeByIndexes(0,i,end,1).format.columnWidth=width;
  detail.getRange('A2').values=[['Extended-hours accounts and sensitivities']];
  detail.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#18344F'};
  detail.getRange('A2:J2').format.borders={bottom:{style:'thin',color:'#9DAFBD'}};
  detail.getRange('A3').values=[['2022–2025 • all 66 accounts • fixed primary ten seeds; other groups diagnostic • 20-session SR intervals']];
  detail.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#52606B'};
  detail.getRange('A5:J5').values=[['Strategy','Role','Seed','Fee multiple','Sharpe','CAGR','RankIC','Max drawdown','SR 95% lower','SR 95% upper']];
  detail.getRange(`A6:J${end}`).values=data.extended_hours.map(r=>[r.strategy,r.role,r.seed,r.cost_multiplier,r.sharpe,r.cagr,r.rankic===null?'n.a.':r.rankic,r.max_drawdown,r.sr_low,r.sr_high]);
  const table=detail.tables.add(`A5:J${end}`,true,'ExtendedHoursAccounts');table.style='TableStyleMedium2';table.showFilterButton=true;
  detail.getRange('A5:J5').format={fill:'#18344F',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:30};
  detail.getRange(`D6:J${end}`).format.horizontalAlignment='right';
  for(const [column,format] of [['D','0"x"'],['E','0.000'],['F','0.00%'],['G','0.0000'],['H','0.00%'],['I','0.000'],['J','0.000']])detail.getRange(`${column}6:${column}${end}`).setNumberFormat(format);
  detail.freezePanes.freezeRows(5);
  book.recalculate();
  const checks=[];
  for(const [name,range] of [['Models','A68:H77'],['Extended-hours','A5:J12'],['Method',`A${methodFirst}:B${methodLast}`]])checks.push((await book.inspect({kind:'table',range:`'${name}'!${range}`,include:'values,formulas',tableMaxRows:10,tableMaxCols:10,maxChars:5000})).ndjson);
  const errors=await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},summary:'Final error scan',maxChars:2500});
  await fs.writeFile(path.join(out,'workbook_inspection.json'),JSON.stringify({checks,errors:errors.ndjson},null,2));
  for(const [name,range,suffix] of [['Models','A65:H77','new_rows'],['Extended-hours','A1:J15','top'],['Extended-hours','A59:J71','benchmarks'],['Method',`A${methodFirst}:B${methodLast}`,'updates']]){
    const preview=await book.render({sheetName:name,range,scale:1.15,format:'png'});
    await fs.writeFile(path.join(out,`${name.replaceAll(' ','_')}_${suffix}_preview.png`),new Uint8Array(await preview.arrayBuffer()));
  }
  const output=await SpreadsheetFile.exportXlsx(book);
  await output.save(path.join(root,'docs','NASDAQ100_strategy_comparison.xlsx'));
  console.log(JSON.stringify({updatedWorkbook:true,mainModels:data.sheets[0].rows.length,extendedAccounts:data.extended_hours.length,errorScan:errors.ndjson}));
  process.exit(0);
}
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
