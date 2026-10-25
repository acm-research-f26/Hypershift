"""Create a plain-language report from the completed, audited new-universe run."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import zipfile
import uuid
from lxml import etree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path('runs/largecap_20261010/comparison')
OUT=Path('docs/largecap_retraining_20261010')
BUILD=Path('tmp/largecap_report')
DATA=Path('data/largecap_20261010')
FONT=Path('presentations/build/canva_dataset/poppins/fonts')
NAMES={'NYSE_top100_20261010':'NYSE top 100','NASDAQ100_20261010':'NASDAQ 100'}
SHORT={'think':'THINK','think_rank':'THINK with ranking','think_gate':'THINK with input gate','think_mix':'THINK + mixer',
       'think_risk':'THINK with risk output','mlp':'Simple neural network','lstm':'Sequence neural network',
       'stockmixer':'StockMixer variant','market_transformer':'Market transformer','think_huber':'THINK + Huber',
       'think_mix_huber':'THINK + mixer + Huber','stockmixer_huber':'StockMixer + Huber',
       'think_mix_shrink':'THINK + weaker mixer','think_mix_shrink_huber':'THINK + weaker mixer + Huber',
       'multiscale_mse':'Multi scale mixer','multiscale_huber':'Multi scale mixer + Huber',
       'think_mix_huber_rank_01':'Mixer + Huber + rank 0.1','think_mix_huber_rank_03':'Mixer + Huber + rank 0.3',
       'think_mix_huber_rank_10':'Mixer + Huber + rank 1.0','ridge':'Linear prediction model',
       'momentum126':'Buy recent winners','reversal5':'Buy recent losers','buy_hold':'Buy and hold the group',
       'equal_universe':'Rebalance the full group','market_proxy':'SPY','qqq':'QQQ'}

def number(x,d=2):
    return '—' if pd.isna(x) else f'{x:.{d}f}'
def percent(x):
    return '—' if pd.isna(x) else f'{x*100:.1f}%'
def interval(a,b):
    return f'{a:.2f} to {b:.2f}'

def shade(cell,fill):
    props=cell._tc.get_or_add_tcPr()
    item=OxmlElement('w:shd');item.set(qn('w:fill'),fill);props.append(item)
def table(doc,headers,rows,widths,font_size=9.5):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    props=t._tbl.tblPr
    borders=OxmlElement('w:tblBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement(f'w:{side}');e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
    props.append(borders)
    for width,col in zip(widths,t.columns):col.width=Inches(width)
    for j,text in enumerate(headers):t.rows[0].cells[j].text=str(text)
    for values in rows:
        cells=t.add_row().cells
        for j,text in enumerate(values):cells[j].text=str(text)
    for i,row in enumerate(t.rows):
        for j,cell in enumerate(row.cells):
            cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp=cell._tc.get_or_add_tcPr()
            margins=OxmlElement('w:tcMar')
            for side,value in [('top',50),('bottom',50),('left',95),('right',95)]:
                e=OxmlElement(f'w:{side}');e.set(qn('w:w'),str(value));e.set(qn('w:type'),'dxa');margins.append(e)
            cp.append(margins)
            shade(cell,'333333' if i==0 else ('F4F4F2' if i%2==0 else 'FFFFFF'))
            for p in cell.paragraphs:
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0);p.paragraph_format.line_spacing=1.0
                for r in p.runs:
                    r.font.name='Poppins';r.font.size=Pt(font_size);r.bold=i==0;r.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '000000')
        # Rows can grow, but a single record should stay together.
        trPr=row._tr.get_or_add_trPr();trPr.append(OxmlElement('w:cantSplit'))
    header=t.rows[0]._tr.get_or_add_trPr();header.append(OxmlElement('w:tblHeader'))
    doc.add_paragraph(style='Table Spacer')
    return t

def paragraph(doc,text,style=None):
    p=doc.add_paragraph(text,style=style);return p
def page(doc,title):
    heading=doc.add_heading(title,level=1)
    heading.paragraph_format.page_break_before=True
def picture(doc,path,width=6.9):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(path),width=Inches(width))
    p.paragraph_format.space_after=Pt(3)

def setup_doc():
    doc=Document();s=doc.sections[0]
    doc.core_properties.title='Stock Forecasting Model Retraining Results'
    doc.core_properties.author='Hypershift'
    doc.core_properties.subject='Current large company lists with retrospective tests from 2022 to 2025'
    s.page_width=Inches(8.5);s.page_height=Inches(11)
    s.top_margin=Inches(.65);s.bottom_margin=Inches(.6);s.left_margin=s.right_margin=Inches(.8)
    for name,size in [('Normal',10.5),('Title',25),('Subtitle',10.5),('Heading 1',18),('Heading 2',12.5),('Caption',9)]:
        st=doc.styles[name];st.font.name='Poppins Black' if name=='Title' else 'Poppins';st.font.size=Pt(size)
        fonts=st.element.get_or_add_rPr().rFonts
        for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
            fonts.attrib.pop(qn('w:'+attr),None)
        st.font.color.rgb=RGBColor(0,0,0)
        st.paragraph_format.space_after=Pt(8)
        st.paragraph_format.space_before=Pt(0 if name in ['Normal','Title','Subtitle'] else 10)
    doc.styles['Normal'].paragraph_format.line_spacing=1.05
    spacer=doc.styles.add_style('Table Spacer',WD_STYLE_TYPE.PARAGRAPH)
    spacer.font.name='Poppins';spacer.font.size=Pt(1)
    spacer.paragraph_format.line_spacing=Pt(1);spacer.paragraph_format.space_after=Pt(3)
    spacer.paragraph_format.space_before=Pt(0)
    for name in ['Heading 1','Heading 2']:
        doc.styles[name].font.bold=True
        doc.styles[name].paragraph_format.keep_with_next=True
    title_props=doc.styles['Title'].element.get_or_add_pPr()
    for border in list(title_props.findall(qn('w:pBdr'))):title_props.remove(border)
    header=s.header.paragraphs[0];header.text='HYPERSHIFT   LARGE COMPANY RETRAINING';header.style=doc.styles['Caption']
    footer=s.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=footer.add_run('Exploratory research   |   ');r.font.name='Poppins';r.font.size=Pt(8)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    return doc

def embed_fonts(path):
    """Embed OFL-licensed Poppins faces so Word can preserve the typography."""
    w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    rel='http://schemas.openxmlformats.org/package/2006/relationships'
    ct='http://schemas.openxmlformats.org/package/2006/content-types'
    with zipfile.ZipFile(path) as archive:parts={name:archive.read(name) for name in archive.namelist()}
    theme=ET.fromstring(parts['word/theme/theme1.xml'])
    for node in theme.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}latin'):
        node.set('typeface','Poppins')
    parts['word/theme/theme1.xml']=ET.tostring(theme,encoding='utf-8',xml_declaration=True)
    fonts=ET.fromstring(parts['word/fontTable.xml'])
    relations=ET.fromstring(parts['word/_rels/fontTable.xml.rels']) if 'word/_rels/fontTable.xml.rels' in parts else ET.Element('{'+rel+'}Relationships')
    for family,face,file in [('Poppins','embedRegular','Poppins-Regular.ttf'),('Poppins','embedBold','Poppins-Bold.ttf'),('Poppins Black','embedRegular','Poppins-Black.ttf')]:
        node=next((node for node in fonts if node.get('{'+w+'}name')==family),None)
        if node is None:node=ET.SubElement(fonts,'{'+w+'}font',{'{'+w+'}name':family})
        key='{'+str(uuid.uuid4()).upper()+'}'
        mask=bytes.fromhex(key.strip('{}').replace('-',''))[::-1]
        content=bytearray((FONT/file).read_bytes())
        for i in range(32):content[i]^=mask[i%16]
        target='fonts/'+file.replace('.ttf','.odttf')
        identifier='rIdPoppins'+face+family.replace(' ','')
        ET.SubElement(relations,'{'+rel+'}Relationship',{'Id':identifier,'Type':r+'/font','Target':target})
        ET.SubElement(node,'{'+w+'}'+face,{'{'+r+'}id':identifier,'{'+w+'}fontKey':key,'{'+w+'}subsetted':'0'})
        parts['word/'+target]=bytes(content)
    parts['word/fontTable.xml']=ET.tostring(fonts,encoding='utf-8',xml_declaration=True)
    parts['word/_rels/fontTable.xml.rels']=ET.tostring(relations,encoding='utf-8',xml_declaration=True)
    types=ET.fromstring(parts['[Content_Types].xml'])
    ET.SubElement(types,'{'+ct+'}Default',{'Extension':'odttf','ContentType':'application/vnd.openxmlformats-officedocument.obfuscatedFont'})
    if not any(item.get('Extension')=='txt' for item in types):
        ET.SubElement(types,'{'+ct+'}Default',{'Extension':'txt','ContentType':'text/plain'})
    parts['[Content_Types].xml']=ET.tostring(types,encoding='utf-8',xml_declaration=True)
    settings=ET.fromstring(parts['word/settings.xml'])
    ET.SubElement(settings,'{'+w+'}embedTrueTypeFonts')
    parts['word/settings.xml']=ET.tostring(settings,encoding='utf-8',xml_declaration=True)
    staged=path.with_suffix('.embedded.docx')
    with zipfile.ZipFile(staged,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,content in parts.items():archive.writestr(name,content)
        archive.writestr('Poppins-OFL.txt',(FONT/'OFL.txt').read_bytes())
    staged.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replace-unpublished-draft',action='store_true')
    args=parser.parse_args()
    assert (ROOT/'largecap_reporting_complete.json').exists(), 'Results and audits must finish first'
    assert json.loads((ROOT/'chronology_audit.json').read_text())['audited_folds']==16
    OUT.mkdir(parents=True,exist_ok=True);BUILD.mkdir(parents=True,exist_ok=True)
    for font in FONT.glob('*.ttf'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Poppins','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    pooled=pd.read_csv(ROOT/'combined_2022_2025_metrics.csv')
    ci=pd.read_csv(ROOT/'pooled_confidence_intervals.csv')
    procedure=pd.read_csv(ROOT/'pooled_selection_procedures.csv')
    pci=pd.read_csv(ROOT/'procedure_confidence_intervals.csv')
    controls=pd.read_csv(ROOT/'selection_procedures_vs_spy_qqq.csv')
    selected=pd.read_csv(ROOT/'validation_selected_by_year.csv')
    curves=pd.read_csv(ROOT/'combined_2022_2025_curves.csv')
    annual=pd.read_csv(ROOT/'annual_comparison.csv')
    changes=pd.read_csv(ROOT/'old_new_fixed_k10_comparison.csv')
    frequency_pairs=pd.read_csv(ROOT/'fixed_k10_horizon_pairs.csv')
    eligibility=pd.read_csv(DATA/'eligibility_preflight.csv')
    coverage=pd.read_csv(DATA/'coverage.csv')
    plan=json.loads(Path('config/largecap_retraining_20261010.json').read_text())
    summary=procedure.merge(pci[['procedure','dataset','delta_spy_ci_low','delta_spy_ci_high','delta_qqq_ci_low','delta_qqq_ci_high']],on=['procedure','dataset']).merge(
        controls[['procedure','dataset','qqq_sharpe','delta_qqq']],on=['procedure','dataset'])
    primary=summary[summary.procedure.isin(['Five-session','Daily'])].copy()
    primary.to_csv(OUT/'primary_results.csv',index=False)
    significant=int(((primary.delta_spy_ci_low>0)&(primary.delta_qqq_ci_low>0)).sum())
    improved=int((changes.delta_sharpe>0).sum())
    total=len(changes)
    cost_view=primary.sort_values(['dataset','procedure'],ascending=[False,False])
    cost_labels=[NAMES[r.dataset]+'\n'+('Five day' if r.procedure=='Five-session' else 'One day') for r in cost_view.itertuples()]
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.6))
    colors=['#B48900' if r.procedure=='Five-session' else '#777777' for r in cost_view.itertuples()]
    for ax,values,title in [(axes[0],cost_view.turnover_annual,'Amount of trading each year'),
                             (axes[1],cost_view.gross_sharpe-cost_view.sharpe,'Sharpe decrease from costs')]:
        bars=ax.bar(np.arange(len(cost_view)),values,color=colors,width=.65)
        ax.set_xticks(np.arange(len(cost_view)),cost_labels,fontsize=7.5)
        ax.set_title(title,fontsize=10,fontweight='bold');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.margins(y=.2)
        for bar,value in zip(bars,values):
            ax.annotate(f'{value:.1f}' if ax is axes[0] else f'{value:.3f}',
                        (bar.get_x()+bar.get_width()/2,value),xytext=(0,4 if value>=0 else -10),
                        textcoords='offset points',ha='center',fontsize=8)
    axes[0].set_ylabel('Trading value / portfolio value',fontsize=8)
    fig.tight_layout();fig.savefig(BUILD/'costs.png',bbox_inches='tight');plt.close(fig)
    ranking=pooled[pooled.horizon==5].set_index(['dataset','model'])
    rank_models=plan['five_session_models']
    fig,ax=plt.subplots(figsize=(6.9,5.2))
    rank_y=np.arange(len(rank_models))
    for offset,dataset,color in [(-.12,plan['datasets'][0],'#B48900'),(.12,plan['datasets'][1],'#666666')]:
        values=[ranking.loc[(dataset,model),'RankIC'] for model in rank_models]
        ax.scatter(values,rank_y+offset,s=25,color=color,label=NAMES[dataset],zorder=3)
    ax.axvline(0,color='#888888',ls='--',lw=1)
    ax.set_yticks(rank_y,[SHORT[model] for model in rank_models],fontsize=8)
    ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
    ax.set_xlabel('Average agreement with future stock ordering',fontsize=9)
    ax.spines['left'].set_visible(False)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(BUILD/'ranking.png',bbox_inches='tight');plt.close(fig)
    # Coverage figure has the same denominator meaning for each series.
    fig,ax=plt.subplots(figsize=(6.9,1.7));y=np.array([1,0])
    ax.barh(y+.16,[68,67],height=.27,color='#898985',label='Earlier sample')
    ax.barh(y-.16,[100,99/101*100],height=.27,color='#DFBD36',label='New company list')
    ax.set_yticks(y,['NYSE','NASDAQ']);ax.set_xlim(0,108);ax.set_xlabel('Chosen securities with any price history (%)')
    for yy,old,new in zip(y,[68,67],[100,99/101*100]):
        ax.text(old+1,yy+.16,f'{old:.0f}%',va='center',fontsize=9)
        ax.text(new+1,yy-.16,f'{new:.0f}%',va='center',fontsize=9)
    ax.spines['left'].set_visible(False);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=8.5)
    fig.tight_layout();fig.savefig(BUILD/'coverage.png',bbox_inches='tight');plt.close(fig)
    # Model choice follows earlier validation scores for each year, not test winners.
    for h in [5,1]:
        fig,axes=plt.subplots(1,2,figsize=(6.9,2.9),sharey=True)
        for ax,dataset in zip(axes,plan['datasets']):
            rows=selected[(selected.horizon==h)&(selected.dataset==dataset)].sort_values('year')
            parts=[]
            for row in rows.itertuples():
                parts.append(curves[(curves.horizon==h)&(curves.dataset==dataset)&(curves.fold_year==row.year)&(curves.model==row.model)])
            model=pd.concat(parts,ignore_index=True)
            for label,kind,color,style in [('Model chosen earlier',None,'#B48900','-'),('SPY','market_proxy','#555555','-'),('QQQ','qqq','#969696','--')]:
                path=model if kind is None else curves[(curves.horizon==h)&(curves.dataset==dataset)&(curves.model==kind)]
                np.testing.assert_array_equal(model.date,path.date)
                ax.plot(pd.to_datetime(path.date),np.cumprod(1+path.net),label=label,color=color,ls=style,lw=1.8 if kind is None else 1.3)
            ax.set_title(NAMES[dataset],fontsize=11,fontweight='bold');ax.grid(alpha=.18);ax.tick_params(labelsize=8)
            ax.set_xticks(pd.to_datetime(['2022-01-01','2023-01-01','2024-01-01','2025-01-01','2026-01-01']),['2022','2023','2024','2025','2026'])
        axes[0].set_ylabel('Growth of $1 after costs',fontsize=9)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8)
        fig.tight_layout(rect=[0,.09,1,1]);fig.savefig(BUILD/f'procedure_h{h}.png',bbox_inches='tight');plt.close(fig)
    doc=setup_doc()
    doc.add_paragraph('Stock Forecasting Model\nRetraining Results',style='Title')
    paragraph(doc,'Current company lists with historical tests from 2022 to 2025',style='Subtitle')
    finding=(f'None of the {len(primary)} main tests established a Sharpe advantage after trading costs over both SPY, a broad US market fund, and QQQ, a NASDAQ-100 fund, within the estimated uncertainty ranges.' if significant==0 else f'In {significant} of the {len(primary)} main tests, the estimated uncertainty range supported a Sharpe advantage after trading costs over both SPY and QQQ.')
    paragraph(doc,f'We completed 648 model training runs using the current NASDAQ-100 and the 100 largest NYSE-listed companies. A model learns patterns from earlier prices to rank stocks by expected returns. Coverage improved substantially. {finding} All declared models are reported below.')
    paragraph(doc,'These results describe a retrospective study of today’s large companies. Selecting companies using their 2026 size or membership gives us information that would not have been available in 2022. Better coverage makes the sample easier to explain, but does not remove that bias.')
    doc.add_heading('Company coverage',level=1)
    table(doc,['Group','Companies chosen','Stocks chosen','Stocks with history','Eligible stocks in tests'],[
        ['NYSE top 100','100','100','100','99 to 100'],['NASDAQ 100','100','101','99','91 to 99']], [1.4,1.2,1.15,1.25,1.9],10)
    picture(doc,BUILD/'coverage.png')
    paragraph(doc,'NASDAQ has 101 stocks because Alphabet has two included share classes. The 99 available stock histories represent 98 companies. The first available Yahoo prices for HONA and SPCX begin in June 2026, after the 2018–2025 training window. Recent listings and spin-offs also explain why fewer stocks qualify in earlier test years. We kept every chosen company and did not replace missing names.')
    paragraph(doc,'NYSE selection uses company market value, removes preferred-share rows and duplicate companies, and uses Berkshire Class B. It includes foreign companies with NYSE-listed depositary shares. This is a custom group of the largest NYSE-listed companies, separate from the official NYSE 100 technology and growth index.')

    page(doc,'Returns after trading costs')
    paragraph(doc,'For each test year, the model and trading rule were chosen using earlier validation data. The yellow line follows that procedure; it does not pick the best model after seeing the year’s results. SPY tracks the broad US market and QQQ tracks the NASDAQ-100.')
    doc.add_heading('Five day forecasts',level=2);picture(doc,BUILD/'procedure_h5.png')
    doc.add_heading('One day forecasts',level=2);picture(doc,BUILD/'procedure_h1.png')
    paragraph(doc,'Each year starts a separate simulated account. These curves combine the annual percentage returns and omit untested boundary sessions. They are research summaries, rather than the record of one continuously held account.',style='Caption')

    page(doc,'How well forecasts ordered the stocks')
    paragraph(doc,'A model tries to put stocks with higher future returns above stocks with lower future returns. This chart shows its average agreement with the actual order over the following five trading days. It includes every declared five-day neural model in both company groups.')
    picture(doc,BUILD/'ranking.png')
    paragraph(doc,'Zero means no consistent agreement. Positive scores mean the model tended to put future winners higher; negative scores mean it tended to reverse the order. The mathematical score ranges from minus one to plus one. It is a rank correlation, called RankIC, rather than the percentage of predictions that were correct.')
    paragraph(doc,'Prediction quality and portfolio returns answer different questions. A portfolio holds only a subset of stocks, while this score evaluates the entire eligible group. Position weights, price variability and trading costs can change the investment outcome even when ordering scores look similar.')

    page(doc,'Performance of the model selection procedure')
    paragraph(doc,'Sharpe measures return relative to how much returns fluctuate, after a fixed 3% annual hurdle. Higher is better within this study. A positive return alone does not mean that a model outperformed an index fund.')
    rows=[]
    for r in primary.itertuples():
        rows.append([NAMES[r.dataset]+'\n'+('Five day' if r.procedure=='Five-session' else 'One day'),number(r.sharpe),number(r.spy_sharpe),number(r.qqq_sharpe),percent(r.cumulative_return),percent(r.maximum_drawdown)])
    table(doc,['Group and forecast','Model Sharpe','SPY Sharpe','QQQ Sharpe','Total return','Worst fall'],rows,[1.7,1.0,1.0,1.0,1.1,1.1],10)
    paragraph(doc,'Total return combines the saved annual-fold returns. Worst fall is the largest decline from an earlier high in that research curve. It is not a forecast of the maximum loss possible in future trading.')
    doc.add_heading('Uncertainty in the comparison',level=2)
    rows=[]
    for r in primary.itertuples():
        rows.append([NAMES[r.dataset]+'\n'+('Five day' if r.procedure=='Five-session' else 'One day'),number(r.delta_spy),interval(r.delta_spy_ci_low,r.delta_spy_ci_high),number(r.delta_qqq),interval(r.delta_qqq_ci_low,r.delta_qqq_ci_high)])
    table(doc,['Group and forecast','Gain over SPY','95% range','Gain over QQQ','95% range'],rows,[1.6,.9,1.7,.9,1.8],9.5)
    paragraph(doc,'These ranges estimate uncertainty in the Sharpe difference using 1,000 resampled versions of the daily returns. A range that crosses zero does not establish an advantage. The resampling preserves blocks of 20 trading days within each year and uses the same dates for the model and funds. It does not repeat model training or correct for testing many candidates.')
    page(doc,'How trading costs changed performance')
    daily_helped=int((frequency_pairs.daily_minus_five_session_sharpe>0).sum())
    paragraph(doc,f'At a fixed rule of ten equally weighted stocks, daily forecasts had higher net Sharpe in {daily_helped} of {len(frequency_pairs)} matched model, group and year cases. These cases reuse the same companies and years; they are not independent experiments. Each horizon retains its saved test dates and annual boundary gaps.')
    paragraph(doc,'The chart compares the amount of trading with the change in Sharpe caused by modeled costs. A trading value of ten means that annual transactions total about ten times the portfolio’s value. Yellow marks five-day forecasts; grey marks daily forecasts.')
    picture(doc,BUILD/'costs.png')
    rows=[[NAMES[r.dataset]+'\n'+('Five day' if r.procedure=='Five-session' else 'One day'),number(r.gross_sharpe),number(r.sharpe),number(r.turnover_annual,1)] for r in primary.itertuples()]
    table(doc,['Group and forecast','Sharpe before costs','Sharpe after costs','Annual trading value divided by portfolio value'],rows,[1.7,1.4,1.4,2.4],9.5)
    paragraph(doc,'The before-and-after columns use the same holdings and trades. Turnover describes how much trading the strategy requires. More trading creates more opportunities for fees, spreads and price impact to consume a small prediction advantage.')
    page(doc,'Annual model choices')
    paragraph(doc,'These choices were locked from prior validation scores before the new held-out performance statistics were computed. Each annual test uses the model and stock count listed below.')
    rows=[]
    for r in selected.sort_values(['horizon','dataset','year']).itertuples():
        rows.append([NAMES[r.dataset],str(r.horizon)+' day',str(r.year),SHORT.get(r.model,r.model),str(r.k)])
    table(doc,['Group','Forecast','Test year','Chosen model','Target stocks'],rows,[1.15,.75,.8,3.35,.85],9)

    for h,dataset in [(5,plan['datasets'][0]),(5,plan['datasets'][1])]:
        page(doc,NAMES[dataset]+' five day model results')
        paragraph(doc,'Each model uses its own trading rule chosen on prior validation data. The table includes all declared five-day models, simple strategies and both comparison funds. Rows retain the declared model order; the table does not hide unsuccessful variants.')
        block=pooled[(pooled.horizon==h)&(pooled.dataset==dataset)].copy()
        block=block.drop(columns=['delta_spy_ci_low','delta_spy_ci_high'],errors='ignore')
        block=block.merge(ci[['horizon','dataset','model','delta_spy_ci_low','delta_spy_ci_high']],on=['horizon','dataset','model'])
        order=plan['five_session_models']+['ridge','momentum126','reversal5','buy_hold','equal_universe','market_proxy','qqq']
        block=block.set_index('model').loc[order].reset_index()
        rows=[[SHORT[r.model],number(r.net_sharpe),percent(r.net_cumulative_return),percent(r.net_maximum_drawdown),interval(r.delta_spy_ci_low,r.delta_spy_ci_high)] for r in block.itertuples()]
        table(doc,['Model or strategy','Net Sharpe','Total return','Worst fall','95% Sharpe gain vs SPY'],rows,[2.75,.75,.95,.95,1.5],9.2)
        paragraph(doc,'Neural models are local adaptations. See the method section for definitions.',style='Caption')

    page(doc,'One day model results')
    paragraph(doc,'One-day forecasts use the same input features, three training seeds and prior-validation selection rules. Daily models use a larger training batch, so this comparison changes both horizon and batch size. Frequent transactions can also consume more of a small forecasting advantage.')
    for index,dataset in enumerate(plan['datasets']):
        if index:page(doc,NAMES[dataset]+' one day model results')
        else:doc.add_heading(NAMES[dataset],level=2)
        block=pooled[(pooled.horizon==1)&(pooled.dataset==dataset)].set_index('model')
        order=plan['daily_models']+['ridge','momentum126','reversal5','buy_hold','equal_universe','market_proxy','qqq']
        rows=[[SHORT[k],number(block.loc[k,'net_sharpe']),percent(block.loc[k,'net_cumulative_return']),percent(block.loc[k,'net_maximum_drawdown'])] for k in order]
        table(doc,['Model or strategy','Net Sharpe','Total return','Worst fall'],rows,[3.35,1.05,1.25,1.25],9)

    page(doc,'Comparison with the earlier company samples')
    paragraph(doc,f'For the {total} shared model and group comparisons with saved results, the new company lists increased net Sharpe in {improved} cases at a fixed rule of ten equally weighted stocks. The rule and test dates are held constant. The universe, data download and CPU-to-GPU training backend differ, so the changes cannot be attributed solely to better coverage.')
    paragraph(doc,'The table highlights four existing model families. It uses the same fixed ten-stock rule in both experiments. The earlier sample was selected by ticker hashes from historical research lists and had only 50–59 eligible Nasdaq securities or 56–63 eligible NYSE securities on test dates.')
    focus=['think','think_mix_huber','stockmixer_huber','multiscale_huber']
    for h in [1,5]:
        if h==5:page(doc,'Five day comparison with earlier samples')
        else:doc.add_heading('One day forecasts',level=2)
        rows=[]
        for r in changes[changes.model.isin(focus)&(changes.horizon==h)].sort_values(['dataset','model']).itertuples():
            rows.append([NAMES[r.dataset],SHORT[r.model],number(r.old_net_sharpe),number(r.new_net_sharpe),number(r.delta_sharpe)])
        table(doc,['Group','Model','Earlier Sharpe','New Sharpe','Change'],rows,[1.25,2.85,.9,.9,1.0],9.5)
    paragraph(doc,'The full comparison is retained in the supporting CSV files. Model choices from prior validation and fixed-rule comparisons answer different questions: the former tests a selection procedure, while the latter helps compare forecasting models with the trading rule held constant.')

    page(doc,'Study method and interpretation')
    paragraph(doc,'Each annual test freshly trains all neural models on earlier prices. The simple winner and loser rules use earlier prices directly. We used 16 trading days of input and five measurements: daily price change, price relative to its five-day and twenty-day averages, recent price variability, and trading activity. Models predict a price-change score used to rank eligible stocks.')
    table(doc,['Test year','Training signals','Validation year'],[['2022','2019 to 2020','2021'],['2023','2019 to 2021','2022'],['2024','2019 to 2022','2023'],['2025','2020 to 2023','2024']],[1.4,3.6,1.9],10)
    paragraph(doc,'THINK is the original model family. A mixer combines patterns across stocks; the weaker version reduces its influence. Huber reduces the influence of extreme forecast errors. Rank variants add a training objective for ordering future winners above losers. These are local adaptations, rather than official reproductions of the research papers.')
    paragraph(doc,'We trained each neural configuration from three different starting weights, called seeds 7, 19 and 42. Predictions from these runs are combined by their ranks; we do not choose the seed with the best test return. Training makes at most thirty passes through the training data and stops after six passes without improvement in validation forecast error. The optimizer is AdamW with learning rate 0.001. Five-day models process batches of eight observations; daily models process forty. All 648 declared fits finished.')
    trading_text='A stock qualifies only with forty consecutive observed prices, finite model inputs and at least $1 million of average daily trading value over twenty sessions. A score produced at today’s close trades at the next close. Each annual simulated account starts at $1 million. Normal orders are capped at 1% of lagged average daily trading value; forced year-end liquidations may exceed that cap. Per-side costs include a 2-basis-point commission, 5 basis points for spread and slippage, and participation-based impact. A basis point is 0.01 percentage points. Cash earns zero in the simulation, while Sharpe uses a fixed 3% annual hurdle. Days in this report mean trading days.'
    baseline_text='The buy-recent-winners baseline also needs a finite six-month price-change measurement. It excludes a newly listed stock until that history is available; the stock can still qualify for the neural models earlier. This causal availability check was added after an error during validation, before any new held-out performance metrics were computed.'
    paragraph(doc,'Portfolio selection tests five weighting or retention methods and holdings of five, ten or twenty stocks. It uses the median prior-validation half-year and seed Sharpe minus a variability penalty. The architecture choice for an annual test uses that year’s and earlier validation scores. The new five-day procedure also includes the three previously studied ranking-loss variants; the earlier main procedure had sixteen models.')
    paragraph(doc,'The sixteen cohort, horizon and year folds passed saved-split, training-only scaling, mature-label and future-observation perturbation checks. Those checks support the chronology of the saved-data pipeline. They do not certify historical membership, unrevised vendor data or an independent confirmation test.')
    page(doc,'Trading rules and research limits')
    paragraph(doc,trading_text)
    paragraph(doc,baseline_text)
    doc.add_heading('Limits that remain',level=2)
    paragraph(doc,'Current 2026 lists create future-universe selection and survivorship bias when applied to 2022–2025. The NASDAQ group is not a reconstruction of historical index membership, and the NYSE group is not reselected by historical size. Company businesses can change through restructurings. Price adjustments may include later revisions, and recent listings cannot provide earlier histories. The test years were already examined during prior research. Any apparent advantage remains conditional on this retrospective setup.')
    paragraph(doc,'Before using the result as evidence of future trading performance, repeat the experiment with membership and company-size information available at each historical decision date, plus a genuinely new test period. The current report can support a presentation about exploratory results on clearly defined large-company groups.')

    page(doc,'Sources and supporting results')
    paragraph(doc,'Company lists were frozen on 10 October 2026 before the new price download and before inspection of new test metrics. Complete ticker lists, company names, source snapshots, coverage counts, training checkpoints, validation choices and return paths are retained separately.')
    for title,url in [('NYSE-listed company ranking','https://stockanalysis.com/list/nyse-stocks/'),('NASDAQ-100 component list','https://www.slickcharts.com/nasdaq100'),('Official NASDAQ-100 methodology','https://indexes.nasdaq.com/docs/Methodology_NDX.pdf'),('NYSE 100 definition','https://www.nyse.com/nyse-100-index'),('Price history provider','https://finance.yahoo.com/'),('Data download library','https://ranaroussi.github.io/yfinance/')]:
        paragraph(doc,title+'\n'+url)
    doc.add_heading('Files retained with the study',level=2)
    paragraph(doc,'The supporting data package includes full model results, paired confidence intervals, annual model choices, fixed ten-stock comparisons with the previous run, company membership and coverage. It also includes the secondary procedure that chooses between both forecast horizons using earlier validation scores. The training run retains per-seed histories and checkpoints. The original experiment and presentation files remain unchanged.')
    details=json.loads((ROOT/'largecap_reporting_complete.json').read_text())
    table(doc,['Forecast','Neural fits','Epochs completed'],[[str(r['horizon'])+' day',r['neural_fits'],r['epoch_sum']] for r in details['training']],[2.3,2.3,2.3],10)
    paragraph(doc,'Report generated after the complete training and reporting batch. All figures and numeric tables come from saved simulated returns; no model was selected from the test charts.',style='Caption')
    destination=OUT/'Hypershift_Large_Company_Retraining_Results.docx'
    assert not destination.exists() or args.replace_unpublished_draft, 'Keep earlier published document versions intact'
    doc.save(destination)
    embed_fonts(destination)
    files=[DATA/'members.csv',DATA/'coverage.csv',DATA/'eligibility_preflight.csv',DATA/'membership_provenance.json',DATA/'late_listing_check.json',
           ROOT/'combined_2022_2025_metrics.csv',ROOT/'pooled_confidence_intervals.csv',ROOT/'procedure_confidence_intervals.csv',
           ROOT/'pooled_selection_procedures.csv',ROOT/'selection_procedures_vs_spy_qqq.csv',ROOT/'combined_validation_selected.csv',
           ROOT/'seed_stability.csv',ROOT/'random_top10_controls.csv',ROOT/'huber_mse_pooled_intervals.csv',
           ROOT/'annual_comparison.csv',ROOT/'validation_selected_by_year.csv',ROOT/'old_new_fixed_k10_comparison.csv',
           ROOT/'fixed_k10_pooled.csv',ROOT/'fixed_k10_annual.csv',ROOT/'fixed_k10_horizon_pairs.csv',
           ROOT/'chronology_audit.json',OUT/'primary_results.csv',Path('config/largecap_retraining_20261010.json')]
    package=OUT/'Hypershift_Large_Company_Results_Data.zip'
    if package.exists():
        assert args.replace_unpublished_draft
        with zipfile.ZipFile(package) as archive:
            assert archive.testzip() is None
            for file in files:assert archive.read(file.name)==file.read_bytes()
    else:
        with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as archive:
            for file in files:archive.write(file,file.name)
    (BUILD/'document_summary.json').write_text(json.dumps({'positive_paired_ranges_vs_both':significant,'primary_procedures':len(primary),
        'shared_fixed_k10_improvements':improved,'shared_fixed_k10_comparisons':total,'docx':str(destination)},indent=2))
    print(json.dumps({'docx':str(destination),'summary':json.loads((BUILD/'document_summary.json').read_text())}),flush=True)

if __name__=='__main__':main()
