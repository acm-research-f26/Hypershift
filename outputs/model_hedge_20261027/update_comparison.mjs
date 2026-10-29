import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';

const out=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(out,'../..');
const inspectOnly=process.argv.includes('--inspect');
const input=inspectOnly?path.join(root,'docs/NASDAQ100_strategy_comparison.xlsx'):path.join(out,'before.xlsx');
const book=await SpreadsheetFile.importXlsx(await FileBlob.load(input));
if(inspectOnly){
  console.log((await book.inspect({kind:'workbook,sheet,table',maxChars:4500,tableMaxRows:2,tableMaxCols:8})).ndjson);
  const preview=await book.render({sheetName:'New findings',range:'A1:H20',scale:1,format:'png'});
  await fs.writeFile(path.join(out,'before_preview.png'),new Uint8Array(await preview.arrayBuffer()));
  console.log((await book.inspect({kind:'region',sheetId:'Defensive',range:'A1:H8',maxChars:2200})).ndjson);
  process.exit(0);
}
const payload=JSON.parse(await fs.readFile(path.join(out,'payload.json'),'utf8'));
const crypto=await import('node:crypto');
if(crypto.createHash('sha256').update(await fs.readFile(input)).digest('hex')!==payload.input_sha256)throw Error('Workbook snapshot changed');
const font={name:'Arial',size:10,color:'#202D39'};
const header={fill:'#18344F',font:{...font,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:38,wrapText:true};
const letters=i=>{let t='';for(i++;i>0;i=Math.floor((i-1)/26))t=String.fromCharCode(65+(i-1)%26)+t;return t;};
function sheet(name,title,subtitle,widths,rows){
  const s=book.worksheets.add(name);s.showGridLines=false;
  s.getRangeByIndexes(0,0,rows,widths.length).format.font=font;
  s.getRangeByIndexes(0,0,rows,widths.length).format.rowHeight=25;
  s.getRangeByIndexes(0,0,rows,widths.length).format.verticalAlignment='center';
  widths.forEach((w,j)=>s.getRangeByIndexes(0,j,rows,1).format.columnWidth=w);
  s.getRange('A2').values=[[title]];s.getRange('A2').format.font={...font,size:14,bold:true,color:'#18344F'};
  s.getRange('A3').values=[[subtitle]];s.getRange('A3').format.font={...font,italic:true,color:'#52606B'};
  return s;
}
function table(s,address,name){const t=s.tables.add(address,true,name);t.style='TableStyleMedium2';t.showFilterButton=true;return t;}

// Preserve all existing sheets, data and drawings. Extend the existing comparison.
const defensive=book.worksheets.getItem('Defensive');
defensive.tables.items[0].rows.add(null,payload.defensive_rows);
const endDef=56+payload.defensive_rows.length;
defensive.getRange(`A57:H${endDef}`).format.rowHeight=38;
defensive.getRange(`A57:B${endDef}`).format.wrapText=true;
defensive.getRange(`C57:H${endDef}`).format.horizontalAlignment='right';
for(const [c,f] of [['C','0.000'],['D','0.00%'],['E','0.0000'],['F','0.00%'],['G','0.000'],['H','0.000']])defensive.getRange(`${c}57:${c}${endDef}`).setNumberFormat(f);
book.worksheets.getItem('New findings').getRange('A1').values=[['October 27 model hedge comparisons: open Model hedges. Prior findings remain below.']];

// The new output is distinct from historical findings; raw accounts and live
// chart paths stay in separate source/build sheets because of their size.
const summary=sheet('Model hedges','Model hedge signals and highest observed Sharpe',
  '2022–2025 development history. Five stock issuers in every neural account. Base costs and assumed 3% annual borrow.',[68,18,12,14,16,14,14,39],105);
const accounts=sheet('Hedge accounts','All own-model hedge accounts',
  '495 accounts. All signal rules, seed groups, transaction costs and borrowing assumptions, including inactive rules.',
  [66,18,34,25,14,14,12,14,16,18,20,20,21,70,15,15],payload.accounts.length+5);
const lastAccount=payload.accounts.length+5;
accounts.getRange('A5:P5').values=[payload.account_headers];accounts.getRange(`A6:P${lastAccount}`).values=payload.accounts;
table(accounts,`A5:P${lastAccount}`,'OwnModelHedgeAccounts');accounts.getRange('A5:P5').format=header;
accounts.getRange(`A6:D${lastAccount}`).format.wrapText=true;accounts.getRange(`A6:P${lastAccount}`).format.rowHeight=34;
accounts.getRange(`E6:P${lastAccount}`).format.horizontalAlignment='right';
for(const [c,f] of [['E','0"x"'],['F','0.0%'],['G','0.000'],['H','0.00%'],['I','0.00%'],['J','0.00%'],['K','"$"#,##0.00'],['L','"$"#,##0.00'],['M','"$"#,##0'],['O','0.000'],['P','0.000']])accounts.getRange(`${c}6:${c}${lastAccount}`).setNumberFormat(f);
accounts.freezePanes.freezeRows(5);accounts.freezePanes.freezeColumns(1);

const accountMap=new Map(payload.accounts.map((r,i)=>[r[13],i+6]));
const headlineEnd=6+payload.headline.length;
summary.getRange('A6:H6').values=[['Strategy','Seed ensemble','Sharpe','CAGR','Max drawdown','95% SR low','95% SR high','Interpretation']];
summary.getRange(`A7:H${headlineEnd}`).values=payload.headline.map(r=>[r.label,r.group,r.sharpe,r.cagr,r.dd,r.low,r.high,r.role]);
for(let i=0;i<payload.headline.length;i++){
  const item=payload.headline[i];if(!item.id)continue;const src=accountMap.get(item.id);const dest=i+7;
  summary.getRange(`C${dest}:G${dest}`).formulas=[[`='Hedge accounts'!G${src}`,`='Hedge accounts'!H${src}`,`='Hedge accounts'!I${src}`,`='Hedge accounts'!O${src}`,`='Hedge accounts'!P${src}`]];
}
table(summary,`A6:H${headlineEnd}`,'OwnModelHedgeLeaders');summary.getRange('A6:H6').format=header;
summary.getRange(`A7:H${headlineEnd}`).format.rowHeight=38;summary.getRange(`A7:B${headlineEnd}`).format.wrapText=true;
summary.getRange(`H7:H${headlineEnd}`).format.wrapText=true;summary.getRange(`C7:G${headlineEnd}`).format.horizontalAlignment='right';
for(const [c,f] of [['C','0.000'],['D','0.00%'],['E','0.00%'],['F','0.000'],['G','0.000']])summary.getRange(`${c}7:${c}${headlineEnd}`).setNumberFormat(f);
payload.headline.forEach((r,i)=>{if(r.role==='Continuous-hedge control')summary.getRange(`A${i+7}:H${i+7}`).format.font.bold=true;});
summary.getRange('A19').values=[['Raw model signals never hedge. Bias-corrected signals hedge throughout 2023 and miss the 2022 decline.']];
summary.getRange('A20').values=[['The highest observed Sharpe uses continuous hedging, independent of model timing. New-five results remain weaker.']];
summary.getRange('A21').values=[['Selected historical findings. Pointwise intervals do not adjust the full research search or establish future superiority.']];
summary.getRange('A19:H21').format.font={...font,color:'#52606B'};

const lastPath=payload.dates.length+5;
const paths=sheet('Hedge paths','Daily hedge-study sources and calculated chart paths',
  'Raw daily returns drive editable wealth and drawdown formulas. Each exchange session is retained.',Array(38).fill(19),lastPath);
paths.getRange('A5:M5').values=[['Date',...payload.chart_keys]];
paths.getRange(`A6:A${lastPath}`).values=payload.dates.map(d=>[new Date(d+'T00:00:00Z')]);paths.getRange(`A6:A${lastPath}`).setNumberFormat('yyyy-mm-dd');
paths.getRange(`B6:M${lastPath}`).values=payload.returns;paths.getRange(`B6:M${lastPath}`).setNumberFormat('0.0000%');
const legendHeaders=payload.chart_legends.flat();
paths.getRange('N5:Y5').values=[legendHeaders];paths.getRange('Z5:AK5').values=[legendHeaders];paths.getRange('AL5').values=[['Chart date']];
paths.getRange('N6:Y6').values=[Array(12).fill(100)];
paths.getRange('N7:Y7').formulas=[Array.from({length:12},(_,j)=>`=${letters(13+j)}6*(1+${letters(1+j)}7)`)];paths.getRange(`N7:Y${lastPath}`).fillDown();
paths.getRange('Z6:AK6').formulas=[Array.from({length:12},(_,j)=>`=${letters(13+j)}6/MAX($${letters(13+j)}$6:${letters(13+j)}6)-1`)];paths.getRange(`Z6:AK${lastPath}`).fillDown();
paths.getRange('AL6').formulas=[['=TEXT(A6,"yyyy-mm-dd")']];paths.getRange(`AL6:AL${lastPath}`).fillDown();
paths.getRange(`N6:Y${lastPath}`).setNumberFormat('"$"0.00');paths.getRange(`Z6:AK${lastPath}`).setNumberFormat('0.00%');
paths.getRange('A5:AL5').format=header;paths.freezePanes.freezeRows(5);

for(const [group,startRow,endRow,dd] of [[0,24,43,false],[0,44,54,true],[1,57,76,false],[1,77,87,true]]){
  summary.getRange(`A${startRow}:H${endRow}`).format.rowHeight=28;
  const start=dd?25:13;
  const ranges=[paths.getRange(`AL5:AL${lastPath}`),...payload.chart_indices[group].map(j=>paths.getRange(`${letters(start+j)}5:${letters(start+j)}${lastPath}`))];
  const chart=summary.charts.add('line',ranges);chart.setPosition(`A${startRow}`,`H${endRow}`);
  chart.title=(group===0?'Original five seeds':'Ten seeds')+(dd?' drawdown':' compared with SPY and QQQ');
  chart.titleTextStyle.typeface='Arial';chart.titleTextStyle.fontSize=14;
  chart.hasLegend=!dd;chart.legend={position:'bottom',textStyle:{typeface:'Arial',fontSize:11}};
  chart.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:10},tickLabelInterval:250};
  chart.yAxis={numberFormatCode:dd?'0%':'"$"0',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:11}};
  chart.yAxis.title.text=dd?'Drawdown from previous peak':'Growth of $100 after costs';
  chart.series.items.forEach((s,j)=>{s.line={fill:payload.colors[j],style:j>=4?'dashed':'solid',width:2.4};});
}
summary.getRange('A90').values=[['Signal activation counts by year']];summary.getRange('A90').format.font.bold=true;
summary.getRange('A92:K92').values=[payload.signal_headers];summary.getRange(`A93:K${92+payload.signal_rows.length}`).values=payload.signal_rows;
summary.getRange('A92:K92').format=header;summary.getRange(`A93:K${92+payload.signal_rows.length}`).setNumberFormat('0');
summary.getRange(`A93:A${92+payload.signal_rows.length}`).setNumberFormat('@');
summary.getRange('I92:K104').format.columnWidth=18;
book.recalculate();
const original=paths.getRange('B7').values[0][0],oldWealth=paths.getRange('N7').values[0][0],oldDD=paths.getRange('Z7').values[0][0];
paths.getRange('B7').values=[[-.1]];book.recalculate();
if(Math.abs(paths.getRange('N7').values[0][0]-90)>1e-9||Math.abs(paths.getRange('Z7').values[0][0]+.1)>1e-10)throw Error('Chart formula mutation failed');
paths.getRange('B7').values=[[original]];book.recalculate();
if(Math.abs(paths.getRange('N7').values[0][0]-oldWealth)>1e-9||Math.abs(paths.getRange('Z7').values[0][0]-oldDD)>1e-10)throw Error('Formula restoration failed');
const check=await book.inspect({kind:'table',range:`'Model hedges'!A6:H${headlineEnd}`,include:'values,formulas',maxChars:3500,tableMaxRows:6,tableMaxCols:8});
const errors=await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},summary:'final formula error scan',maxChars:2000});
await fs.writeFile(path.join(out,'workbook_inspection.json'),JSON.stringify({formulaMutationVerified:true,headline:check.ndjson,errors:errors.ndjson}));
console.log('Workbook edited and recalculated; exporting',errors.ndjson);
await(await SpreadsheetFile.exportXlsx(book)).save(path.join(out,'comparison_staged.xlsx'));
for(const [name,range,file] of [['Model hedges','A1:H21','summary_preview.png'],['Model hedges','A24:H54','charts_preview.png'],['Hedge accounts','A1:J12','accounts_preview.png']]){
  const image=await book.render({sheetName:name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(out,file),new Uint8Array(await image.arrayBuffer()));
}
console.log('Staged workbook and visual previews saved');
