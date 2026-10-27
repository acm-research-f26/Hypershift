"""Update current prose/selected-procedure graphics and record asset replacements."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_engine import performance
from refresh_audited_charts import BENCHMARK_NOTE, LABEL, draw, OLD_IMAGES

def main():
    replacement=json.loads(Path('docs/audit/benchmark_image_replacements.json').read_text())
    studies=[(Path('runs/robust_comparison'),Path('docs')),
             (Path('runs/largecap_20261010/comparison'),Path('docs/largecap_retraining_20261010'))]
    for root,dest in studies:
        curves=pd.read_csv(root/'audited_curves.csv')
        byyear=pd.read_csv(root/'validation_selected_by_year.csv')
        joint=pd.read_csv(root/'combined_validation_selected.csv')
        datasets=list(curves.dataset.unique())
        fig,axes=plt.subplots(len(datasets),3,figsize=(18,10),squeeze=False)
        for i,d in enumerate(datasets):
            for j,(name,h) in enumerate([('Five-session',5),('Daily',1),('Joint validation choice',None)]):
                chosen=joint[joint.dataset==d] if h is None else byyear[(byyear.dataset==d)&(byyear.horizon==h)]
                models=[];bench=[];spy=[]
                for row in chosen.sort_values('year').itertuples():
                    group=curves[(curves.dataset==d)&(curves.horizon==int(row.horizon))&(curves.fold_year==row.year)]
                    for target,kind in [(models,row.model),(bench,'equal_universe' if d.startswith('NYSE') else 'qqq'),(spy,'market_proxy')]:
                        target.append(group[group.model==kind])
                for parts,label,color in [(models,'Selected procedure','#0072b2'),(bench,'Simulated NYSE basket' if d.startswith('NYSE') else 'QQQ','#228b22' if d.startswith('NYSE') else '#8844aa'),(spy,'SPY','#111111')]:
                    p=pd.concat(parts);axes[i,j].plot(pd.to_datetime(p.date),np.cumprod(1+p.net),label=f'{label} SR {performance(p.net)["sharpe"]:.2f}',color=color)
                axes[i,j].set_title(f'{d} / {name}');axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=8)
        fig.suptitle('Prior-validation-selected procedures and cohort references, 2022–2025')
        fig.text(.06,.01,'Annual cash resets and omitted boundary dates. NYSX history unavailable; NYSE basket simulated. Public-data/universe bias remains.')
        fig.tight_layout(rect=[0,.04,1,.96]);path=root/'figures/selected_procedures.png'
        if path.exists():replacement[hashlib.sha256(path.read_bytes()).hexdigest()]=str(path.resolve())
        fig.savefig(path,dpi=130);plt.close(fig);shutil.copyfile(path,dest/'figures'/path.name)
        # Remaining former NYSE QQQ overview images used by ranking follow-up pages.
        for path in (root/'figures').glob('*NYSE*.png'):
            if path.name.startswith('huber_ranking_all_models_') or path.name.startswith('huber_ranking_NYSE'):
                data=curves[(curves.dataset.str.startswith('NYSE'))&(curves.horizon==5)]
                draw(data,path,'NYSE cohort | five-session | frozen model comparison')
                if (dest/'figures'/path.name).exists():shutil.copyfile(path,dest/'figures'/path.name)
    replacement.update(OLD_IMAGES)
    Path('docs/audit/benchmark_image_replacements.json').write_text(json.dumps(replacement,indent=2))
    marker_start='<!-- benchmark-audit-20261010:start -->';marker_end='<!-- benchmark-audit-20261010:end -->'
    files=[Path('README.md'),Path('ROBUST_RESEARCH.md'),Path('ARCHITECTURE_RESEARCH.md'),Path('STOCKMIXER_PAPER_REVIEW.md')]
    files+=list(Path('docs').rglob('*.md'))
    for root in ['runs/robust_comparison','runs/largecap_20261010/comparison','runs/recent_2024_2025','runs/architecture_research']:
        files+=list(Path(root).glob('*.md'))
    updated=[]
    for path in files:
        if not path.exists() or path.name=='CHARTS.md':continue
        text=path.read_text(encoding='utf-8')
        if marker_start in text:
            before,rest=text.split(marker_start,1);_,after=rest.split(marker_end,1);text=before+after.lstrip('\n')
        # Current prose describes the new references; filenames/source URLs remain for provenance.
        text=text.replace('SPY/QQQ controls','SPY and cohort controls').replace('SPY/QQQ evaluation','SPY and cohort evaluation')
        text=text.replace('SPY and QQQ for both cohorts','SPY and the cohort reference')
        text=text.replace('Both SPY (black) and QQQ (purple dashed) are now included in every all-model overview/full-size panel and in the selected-procedure chart.',
                          'SPY appears on both cohorts; NASDAQ uses QQQ and NYSE uses the simulated cohort basket.')
        text=text.replace('every model and both ETFs','every model and cohort references').replace('all models and both ETFs','all models and cohort references')
        first,sep,rest=text.partition('\n')
        notice=(marker_start+'\n'+BENCHMARK_NOTE+'\n\n'+'The refreshed CHARTS.md pages are the current visual record. Earlier ETF-overlay descriptions are superseded by this correction. '
                'Audited return paths and training selections remain preserved.\n'+marker_end)
        path.write_text(first+'\n\n'+notice+'\n\n'+rest.lstrip('\n'),encoding='utf-8');updated.append(str(path))
    Path('docs/audit/benchmark_markdown_update.json').write_text(json.dumps(dict(updated=updated,reason='Explicit NYSX availability and cohort/ETF identity correction'),indent=2))
    print('Updated',len(updated),'markdown files and selected-procedure images')

if __name__=='__main__':main()
