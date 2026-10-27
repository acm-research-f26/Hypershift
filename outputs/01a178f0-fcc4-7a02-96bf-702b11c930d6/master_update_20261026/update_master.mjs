import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';

const out=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(out,'../../..');
const payloadPath=path.join(out,'payload.json');
const input=path.join(root,'docs/NASDAQ100_strategy_comparison.xlsx');
const inspectOnly=process.argv.includes('--inspect');
const payload=inspectOnly?null:JSON.parse(await fs.readFile(payloadPath,'utf8'));
const book=await SpreadsheetFile.importXlsx(await FileBlob.load(inspectOnly?input:payload.input_workbook));
if(inspectOnly){
  console.log((await book.inspect({kind:'workbook,sheet,table',maxChars:3500,tableMaxRows:2,tableMaxCols:8})).ndjson);
  for(const [name,range,label] of [['New findings','A1:H28','before_findings'],['Defensive','A1:H12','before_defensive']]){
    const image=await book.render({sheetName:name,range,scale:1.1,format:'png'});
    await fs.writeFile(path.join(out,label+'_preview.png'),new Uint8Array(await image.arrayBuffer()));
  }
  await fs.writeFile(path.join(out,'existing_values.json'),JSON.stringify({newFindings:book.worksheets.getItem('New findings').getRange('A1:L45').values}));
  process.exit(0);
}
const crypto=await import('node:crypto');
const digest=buffer=>crypto.createHash('sha256').update(buffer).digest('hex');
if(digest(await fs.readFile(payload.input_workbook))!==payload.input_sha256)throw Error('Input workbook snapshot changed');
const font={name:'Arial',size:10,color:'#202D39'};
const header={fill:'#18344F',font:{...font,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:42,wrapText:true};
const letters=index=>{let text='';for(index++;index>0;index=Math.floor((index-1)/26))text=String.fromCharCode(65+(index-1)%26)+text;return text;};
const makeTable=(sheet,range,name)=>{const table=sheet.tables.add(range,true,name);table.style='TableStyleMedium2';table.showFilterButton=true;return table;};
const createSheet=(name,title,subtitle,widths,rows)=>{
  const sheet=book.worksheets.add(name);sheet.showGridLines=false;
  sheet.getRangeByIndexes(0,0,rows,widths.length).format.font=font;
  sheet.getRangeByIndexes(0,0,rows,widths.length).format.rowHeight=25;
  sheet.getRangeByIndexes(0,0,rows,widths.length).format.verticalAlignment='center';
  widths.forEach((width,j)=>sheet.getRangeByIndexes(0,j,rows,1).format.columnWidth=width);
  sheet.getRange('A2').values=[[title]];sheet.getRange('A2').format.font={...font,size:14,bold:true,color:'#18344F'};
  sheet.getRange('A3').values=[[subtitle]];sheet.getRange('A3').format.font={...font,italic:true,color:'#52606B'};
  return sheet;
};

// Flat research account source retains all failures and all higher-cost accounts.
const accounts=createSheet('Drawdown accounts','All drawdown-control accounts','2022-2025. 915 base-cost accounts and 54 additional higher-cost accounts. Screen qualification refers to the base-cost recipe.',
  [72,18,13,12,14,16,15,17,17,19,18,75],payload.accounts.length+5);
const accountLast=payload.accounts.length+5;
accounts.getRange('A5:L5').values=[['Strategy','Seed group','Cost multiple','Sharpe','CAGR','Max drawdown','Annual volatility','Annual turnover','Modeled costs (USD)','Terminal NAV (USD)','Base screen pass','Recipe ID']];
accounts.getRange(`A6:L${accountLast}`).values=payload.accounts;
makeTable(accounts,`A5:L${accountLast}`,'DrawdownAccounts');accounts.getRange('A5:L5').format=header;
accounts.getRange(`A6:B${accountLast}`).format.wrapText=true;accounts.getRange(`A6:L${accountLast}`).format.rowHeight=38;
accounts.getRange(`C6:K${accountLast}`).format.horizontalAlignment='right';
for(const [col,format] of [['C','0"x"'],['D','0.000'],['E','0.00%'],['F','0.00%'],['G','0.00%'],['H','0.0"x"'],['I','"$"#,##0.00'],['J','"$"#,##0']])accounts.getRange(`${col}6:${col}${accountLast}`).setNumberFormat(format);
accounts.freezePanes.freezeRows(5);accounts.freezePanes.freezeColumns(1);

// Daily source returns and formula-linked growth paths for the two native charts.
const pathLast=payload.dates.length+5;
const paths=createSheet('Leader paths','Daily source returns and growth paths','Initial row sets $100 before January 3, 2022. Each following row is one exchange session after modeled costs. No dates are omitted.',
  Array(30).fill(18),pathLast);
paths.getRange('A5:N5').values=[['Date','Chart date',...payload.source_headers]];
paths.getRange(`A6:A${pathLast}`).values=payload.dates.map(date=>[new Date(date+'T00:00:00Z')]);
paths.getRange(`A6:A${pathLast}`).setNumberFormat('yyyy-mm-dd');
paths.getRange(`C6:N${pathLast}`).values=payload.returns;
paths.getRange(`C6:N${pathLast}`).setNumberFormat('0.0000%');
paths.getRange('B6').formulas=[['=TEXT(A6,"yyyy-mm-dd")']];paths.getRange(`B6:B${pathLast}`).fillDown();
paths.getRange('P4').values=[['Original five seeds']];paths.getRange('X4').values=[['Ten seeds']];
for(let group=0;group<2;group++){
  const start=group===0?15:23;
  paths.getRangeByIndexes(4,start,1,7).values=[payload.chart_legends[group]];
  paths.getRangeByIndexes(5,start,1,7).values=[Array(7).fill(100)];
  paths.getRangeByIndexes(6,start,1,7).formulas=[payload.chart_indices[group].map((source,j)=>`=${letters(start+j)}6*(1+${letters(2+source)}7)`)];
  paths.getRangeByIndexes(6,start,pathLast-6,7).fillDown();
  paths.getRangeByIndexes(5,start,pathLast-5,7).setNumberFormat('"$"0.00');
  paths.getRangeByIndexes(4,start,1,7).format=header;
  const ddStart=group===0?31:39;
  paths.getRangeByIndexes(3,ddStart,1,1).values=[[group===0?'Original five: drawdown':'Ten seeds: drawdown']];
  paths.getRangeByIndexes(4,ddStart,1,7).values=[payload.chart_legends[group]];
  paths.getRangeByIndexes(5,ddStart,1,7).formulas=[Array.from({length:7},(_,j)=>`=${letters(start+j)}6/MAX(${letters(start+j)}$6:${letters(start+j)}6)-1`)];
  paths.getRangeByIndexes(5,ddStart,pathLast-5,7).fillDown();
  paths.getRangeByIndexes(5,ddStart,pathLast-5,7).setNumberFormat('0.00%');
  paths.getRangeByIndexes(4,ddStart,1,7).format=header;
  paths.getRangeByIndexes(4,ddStart,pathLast-4,7).format.columnWidth=18;
}
for(const [first,last] of [[15,21],[23,29],[31,37],[39,45]]){
  paths.getRangeByIndexes(4,first,1,last-first+1).format.rowHeight=95;
}
paths.getRange('A5:N5').format=header;paths.getRange('A5:N5').format.rowHeight=65;
paths.freezePanes.freezeRows(5);paths.freezePanes.freezeColumns(2);

// Append the 18 qualified main accounts to the established Defensive table.
const defensive=book.worksheets.getItem('Defensive');
defensive.tables.items[0].rows.add(null,payload.additions.map(row=>[row.strategy,row.study,row.sharpe,row.cagr,'n.a.',row.max_drawdown,row.sr_low,row.sr_high]));
defensive.getRange('A39:H56').format.font=font;defensive.getRange('A39:H56').format.rowHeight=36;
defensive.getRange('A39:B56').format.wrapText=true;defensive.getRange('A39:H56').format.verticalAlignment='center';
defensive.getRange('C39:H56').format.horizontalAlignment='right';
for(const [col,format] of [['C','0.000'],['D','0.00%'],['E','0.0000'],['F','0.00%'],['G','0.000'],['H','0.000']])defensive.getRange(`${col}39:${col}56`).setNumberFormat(format);
const method=book.worksheets.getItem('Method');
method.getRange('A44:B48').values=payload.method_notes;method.getRange('A44:B48').format.font=font;
method.getRange('A44:B48').format.wrapText=true;method.getRange('A44:B48').format.rowHeight=70;
method.getRange('A44:B48').format.verticalAlignment='center';method.getRange('A44:A48').format.font.bold=true;

// Preserve all prior findings below the new chart-first opening view.
const findings=book.worksheets.getItem('New findings');
findings.getRange('A81:L125').copyFrom(findings.getRange('A1:L45'),'all');
findings.tables.items[0].delete();findings.getRange('A1:L45').clear({applyTo:'all'});
makeTable(findings,'A85:L111','NewFindings');
findings.getRange('A1:L79').format.font=font;findings.getRange('A1:L79').format.rowHeight=22;
findings.getRange('A1:L79').format.verticalAlignment='center';
for(const [row,height] of Object.entries(payload.prior_findings_row_heights))findings.getRange(`A${Number(row)+80}:L${Number(row)+80}`).format.rowHeight=height;
findings.getRange('A2').values=[['Highest-Sharpe models and drawdown controls']];
findings.getRange('A2').format.font={...font,size:14,bold:true,color:'#18344F'};
findings.getRange('A3').values=[['2022-2025 development history. Same saved forecasts. Base costs. Original-five and ten-seed comparisons are separate.']];
findings.getRange('A3').format.font={...font,italic:true,color:'#52606B'};
findings.getRange('A31:H31').values=[['Strategy','Seed group','Sharpe','CAGR','Max drawdown','SR 95% lower','SR 95% upper','Historical distinction']];
const headlineLast=31+payload.headline.length;
const roleLabels={'Higher-return original-five alternative':'Higher return','Lower-drawdown original-five alternative':'Lower drawdown','Issuer-aware return-preserving alternative':'Return-preserving','Return-preserving ten-seed alternative':'Return-preserving','Prior highest-Sharpe reference':'Prior leader','Matched ten-seed reference':'Matched baseline','Issuer-aware 20% target':'Issuer control'};
findings.getRange(`A32:H${headlineLast}`).values=payload.headline.map(row=>[row.label,row.group,row.sharpe,row.cagr,row.dd,row.low,row.high,roleLabels[row.role]||row.role]);
makeTable(findings,`A31:H${headlineLast}`,'DrawdownLeaders');findings.getRange('A31:H31').format=header;
findings.getRange(`A32:H${headlineLast}`).format.rowHeight=39;findings.getRange(`A32:B${headlineLast}`).format.wrapText=true;
findings.getRange(`H32:H${headlineLast}`).format.wrapText=true;findings.getRange(`C32:G${headlineLast}`).format.horizontalAlignment='right';
for(const [col,format] of [['C','0.000'],['D','0.00%'],['E','0.00%'],['F','0.000'],['G','0.000']])findings.getRange(`${col}32:${col}${headlineLast}`).setNumberFormat(format);
payload.headline.forEach((row,j)=>{if(['Highest observed Sharpe','Highest ten-seed Sharpe'].includes(row.role)||['SPY','QQQ'].includes(row.recipe))findings.getRange(`A${j+32}:H${j+32}`).format.font.bold=true;});
findings.getRange('A44').values=[['305 recipes tested. Six passed the base-cost screen. New-five seed results remain weaker. Paired Sharpe-improvement intervals include zero.']];
findings.getRange('A45').values=[['Risk controls change allocation and exposure. They retain the same THINK/dynamic forecasts and regular-close execution. No future-performance confirmation.']];
findings.getRange('A44:H45').format.font={...font,size:10,color:'#52606B'};
findings.getRange('A73:L73').values=[['Risk-control recipe','Holdings','Ten-seed SR','Ten-seed CAGR','Ten-seed DD','Original-five SR','Original-five CAGR','Original-five DD','New-five SR','New-five CAGR','New-five DD','Base screen pass']];
findings.getRange('A74:L79').values=payload.summary;makeTable(findings,'A73:L79','DrawdownSeedSummary');
findings.getRange('A73:L73').format=header;findings.getRange('A74:L79').format.rowHeight=40;
findings.getRange('A74:A79').format.wrapText=true;findings.getRange('B74:L79').format.horizontalAlignment='right';
for(const col of ['C','F','I'])findings.getRange(`${col}74:${col}79`).setNumberFormat('0.000');
for(const col of ['D','E','G','H','J','K'])findings.getRange(`${col}74:${col}79`).setNumberFormat('0.00%');
findings.freezePanes.unfreeze();findings.freezePanes.freezeRows(3);

findings.getRange('A5:H28').format.rowHeight=34;
findings.getRange('A47:H70').format.rowHeight=34;
for(const [group,start,from,to,title,isDD] of [
  [0,15,'A5','H19','Original five seeds compared with SPY and QQQ',false],
  [0,31,'A20','H28','Drawdown from previous peak: original five',true],
  [1,23,'A47','H61','Ten seeds compared with SPY and QQQ',false],
  [1,39,'A62','H70','Drawdown from previous peak: ten seeds',true]]){
  const ranges=[paths.getRange(`B5:B${pathLast}`),...Array.from({length:7},(_,j)=>paths.getRange(`${letters(start+j)}5:${letters(start+j)}${pathLast}`))];
  const chart=findings.charts.add('line',ranges);chart.setPosition(from,to);chart.title=title;
  chart.titleTextStyle.typeface='Arial';chart.titleTextStyle.fontSize=14;
  chart.legend={position:'bottom',textStyle:{typeface:'Arial',fontSize:11}};
  chart.hasLegend=!isDD;
  chart.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:10},tickLabelInterval:250};
  chart.yAxis={numberFormatCode:isDD?'0%':'"$"0',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:11}};
  chart.yAxis.title.text=isDD?'Decline from previous peak':'Growth of $100 after costs';
  chart.series.items.forEach((series,j)=>{series.line={fill:payload.chart_colors[j],style:[5,6].includes(j)?'dashed':'solid',width:j===4?1.4:2.6};});
}

// Verify that native chart helpers respond to their raw-return inputs.
book.recalculate();
const first=paths.getRange('C7').values[0][0], before=paths.getRange('P7').values[0][0];
const ddBefore=paths.getRange('AF7').values[0][0];
paths.getRange('C7').values=[[-.10]];book.recalculate();
if(Math.abs(paths.getRange('AF7').values[0][0]+.10)>1e-10)throw Error('Drawdown formula did not respond to source edit');
paths.getRange('C7').values=[[first+.01]];book.recalculate();
const after=paths.getRange('P7').values[0][0];
if(Math.abs(after-before-1)>1e-8)throw Error('Growth-path formula did not respond to source edit');
paths.getRange('C7').values=[[first]];book.recalculate();
const returned=paths.getRange('P7').values[0][0];if(Math.abs(returned-before)>1e-10)throw Error('Source restoration failed');
if(Math.abs(paths.getRange('AF7').values[0][0]-ddBefore)>1e-10)throw Error('Drawdown restoration failed');
const check=await book.inspect({kind:'table',range:"'New findings'!A31:H42",include:'values,formulas',maxChars:2500,tableMaxRows:5,tableMaxCols:8});
const errors=await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},maxChars:2000});
await fs.writeFile(path.join(out,'workbook_inspection.json'),JSON.stringify({headlines:check.ndjson,errors:errors.ndjson,formulaMutationVerified:true,drawdownMutationVerified:true,sheets:10,nativeCharts:4,pathLast,headlineLast},null,2));
for(const [name,range,label] of [['New findings','A1:H28','leaders_chart'],['New findings','A30:H45','leaders_table'],['New findings','A47:H70','ten_seed_chart'],['New findings','A73:L79','seed_comparison'],['Defensive','A39:H47','new_defensive'],['Drawdown accounts','A1:L12','all_accounts']]){
  const image=await book.render({sheetName:name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(out,label+'_preview.png'),new Uint8Array(await image.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(book);await file.save(payload.output_workbook);
console.log(JSON.stringify({staged:true,output:payload.output_workbook,newMainModels:18,fullStudyAccounts:payload.accounts.length,nativeCharts:4,formulaMutationVerified:true,errors:errors.ndjson}));
