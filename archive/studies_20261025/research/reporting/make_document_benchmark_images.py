from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('runs/largecap_20261010/comparison')
curves=pd.read_csv(root/'audited_curves.csv')
selected=pd.read_csv(root/'validation_selected_by_year.csv')
out=Path('tmp/benchmark_artifact_images');out.mkdir(exist_ok=True)
for h in [5,1]:
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.9),sharey=True)
    for ax,d in zip(axes,['NYSE_top100_20261010','NASDAQ100_20261010']):
        chosen=selected[(selected.horizon==h)&(selected.dataset==d)].sort_values('year')
        model=pd.concat([curves[(curves.horizon==h)&(curves.dataset==d)&(curves.fold_year==r.year)&(curves.model==r.model)] for r in chosen.itertuples()])
        control='equal_universe' if d.startswith('NYSE') else 'qqq'
        for label,kind,color,style in [('Chosen model',None,'#b48900','-'),('SPY','market_proxy','#555555','-'),
                                       ('NYSE basket' if d.startswith('NYSE') else 'QQQ',control,'#228b22' if d.startswith('NYSE') else '#969696','--')]:
            p=model if kind is None else curves[(curves.horizon==h)&(curves.dataset==d)&(curves.model==kind)]
            np.testing.assert_array_equal(model.date,p.date)
            ax.plot(pd.to_datetime(p.date),np.cumprod(1+p.net),label=label,color=color,ls=style,lw=1.5)
        ax.set_title('NYSE top 100' if d.startswith('NYSE') else 'NASDAQ 100',fontsize=11,fontweight='bold');ax.grid(alpha=.18)
        ax.tick_params(labelsize=8);ax.legend(fontsize=7,frameon=False)
        ax.set_xticks(pd.to_datetime(['2022-01-01','2023-01-01','2024-01-01','2025-01-01','2026-01-01']),['2022','2023','2024','2025','2026'])
    axes[0].set_ylabel('Growth of $1 after costs',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1]);fig.text(.05,.01,'NYSE basket is simulated. NYSX has no 2022–2025 ETF history.',fontsize=7)
    fig.savefig(out/f'procedure_h{h}.png',dpi=215);plt.close(fig)
