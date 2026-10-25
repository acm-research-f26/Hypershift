"""Publish corrected cohort benchmarks from frozen returns; never refit/select."""
from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_engine import performance
from report_robust_research import LABELS
from report_pooled_uncertainty import indices, bootstrap_sharpe, interval

BENCHMARK_NOTE = ('NYSX is the passive Global X NYSE 100 ETF. It launched March 25, 2026 and has no tradable '
    '2022–2025 history. Its index covers technology and growth companies across exchanges, not the 100 largest '
    'NYSE-listed companies. NYSX performance is unavailable here. NYSE historical charts use the simulated '
    'equal-weight cohort basket (scheduled rebalances and modeled costs), not an ETF. NASDAQ charts retain QQQ; SPY supplies broad-market context. '
    '[Issuer and inception](https://www.globalxetfs.com/funds/nysx/) and '
    '[prospectus](https://assets.globalxetfs.com/funds/documents/nysx/prospectus-regulatory/Summary-Prospectus_NYSX.pdf).')
FOCUS = ['think','think_mix','think_huber','think_mix_huber','ridge','momentum126','equal_universe','market_proxy','qqq']
LABEL = dict(LABELS, equal_universe='Cohort basket (simulated)',market_proxy='SPY',qqq='QQQ',momentum126='126-session momentum',ridge='Ridge')
COLORS = dict(think='#0072b2',think_mix='#d55e00',think_huber='#56b4e9',think_mix_huber='#cc79a7',
              ridge='#6e4c9e',momentum126='#e6ab02',equal_universe='#228b22',market_proxy='#111111',qqq='#8844aa')
OLD_IMAGES = {}

def draw(frame,path,title,focus=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): OLD_IMAGES[hashlib.sha256(path.read_bytes()).hexdigest()]=str(path.resolve())
    fig,ax=plt.subplots(figsize=(16,8))
    palette=list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('Dark2').colors)
    for j,(kind,piece) in enumerate(frame.groupby('model',sort=False)):
        if focus and kind not in FOCUS: continue
        sr=performance(piece.net)['sharpe']
        ax.plot(pd.to_datetime(piece.date),np.cumprod(1+piece.net),label=f'{LABEL.get(kind,kind)}  SR {sr:.2f}',
                color=COLORS.get(kind,palette[j%len(palette)]),lw=2 if kind in FOCUS else 1,
                ls='--' if 'mix' in kind or kind=='qqq' else '-')
    ax.set(title=title,ylabel='Net wealth per initial $1')
    ax.grid(alpha=.2); ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=9)
    fig.subplots_adjust(left=.07,right=.67,bottom=.16,top=.90)
    fig.text(.07,.08,'Frozen predictions and prior-validation policies. Annual accounts reset; untested boundary dates omitted.',fontsize=10)
    fig.text(.07,.045,'NYSX: unavailable before its March 2026 launch. NYSE basket is simulated. Retrospective universe/data bias remains.',fontsize=10)
    fig.savefig(path,dpi=130); plt.close(fig)

def metrics(frame):
    rows=[]
    for (h,d,m),p in frame.groupby(['horizon','dataset','model'],sort=False):
        b=frame[(frame.horizon==h)&(frame.dataset==d)&(frame.model==('equal_universe' if d.startswith('NYSE') else 'qqq'))]
        np.testing.assert_array_equal(p.date,b.date)
        stats=performance(p.net); sampling=indices(p.groupby('fold_year',sort=True).size().tolist())
        samples=bootstrap_sharpe(p.net,sampling)
        low,high=interval(samples-bootstrap_sharpe(b.net,sampling))
        rows.append(dict(horizon=h,dataset=d,model=m,**stats,benchmark_model=b.model.iloc[0],
                         delta_benchmark=stats['sharpe']-performance(b.net)['sharpe'],delta_low=low,delta_high=high,
                         NYSX_sharpe=np.nan,NYSX_status='unavailable: inception 2026-03-25'))
    return pd.DataFrame(rows)

def main():
    studies=[('original',Path('runs/robust_comparison'),Path('docs')),
             ('top100',Path('runs/largecap_20261010/comparison'),Path('docs/largecap_retraining_20261010'))]
    audit=pd.read_csv('docs/audit/backtests_20261010/pooled.csv')
    textfiles=[]
    for study,root,dest in studies:
        base=pd.read_csv(root/'combined_2022_2025_curves.csv')
        rankfile=root/'huber_ranking_curves.csv'
        if rankfile.exists():
            rank=pd.read_csv(rankfile); rank=rank[rank.policy=='validation-selected'].copy(); rank['horizon']=5
            keys=set(zip(base.horizon,base.dataset,base.model))
            rank=rank[[ (5,d,m) not in keys for d,m in zip(rank.dataset,rank.model) ]]
            base=pd.concat([base,rank],ignore_index=True)
        base=base[~(base.dataset.str.startswith('NYSE')&(base.model=='qqq'))].copy()
        base.to_csv(root/'audited_curves.csv',index=False)
        out=metrics(base); out.to_csv(root/'audited_metrics.csv',index=False)
        for name in ['audited_curves.csv','audited_metrics.csv']: shutil.copyfile(root/name,dest/name)
        sections=[f'# {"Original random samples" if study=="original" else "Current top-100 universes"} charts and audit','',BENCHMARK_NOTE,'',
                  'The audit reproduced saved momentum and ridge forecasts and replayed THINK variants. '
                  'A separate ledger in share units agreed with all 384 selected/fixed portfolio paths across the two studies. '
                  'Future-price perturbations preserved earlier features, momentum, masks, scaling and graph construction. '
                  'This supports the forecast/accounting implementation. It does not remove future-universe selection, survivorship bias or repeated examination of test results.','',
                  'Current top-100 membership was chosen in 2026 and applied to 2022–2025. The original samples recover only 68 NYSE and 67 NASDAQ issuer histories. '
                  'These are different, biased populations. Strong top-100 results need not transfer to random samples.','',
                  '[Audited metrics](audited_metrics.csv) · [Dated curves](audited_curves.csv)','',
                  '126-session momentum scores each eligible stock as adjusted close today / adjusted close 126 sessions ago − 1. '
                  'It ranks past winners, then trades at the next session close; it does not predict using future prices. '
                  'This implementation includes the most recent month. Ridge pools eligible training stock/date observations and '
                  'fits 80 lagged features plus an intercept by minimizing squared forecast error + 10 × squared coefficient size '
                  '(intercept unpenalized). Features are scaled on training data only. It forecasts next-close-to-exit returns; the '
                  'portfolio ranks those forecasts. Both baselines use earlier validation to choose K and allocation.','',
                  'THINK+mixer adds a learned cross-stock residual to THINK scores. Extra parameters can help one cohort and overfit '
                  'another. A higher return in one panel is insufficient evidence of an architectural advantage. '
                  'Compare the same fixed portfolio, seed spread and paired uncertainty before attributing a difference to the mixer.','']
        for (h,d),group in base.groupby(['horizon','dataset'],sort=False):
            f=root/'figures'/f'combined_h{h}_{d}.png'
            draw(group,f,f'{d} | {h}-session forecasts | all declared models')
            focus=root/'figures'/f'audited_focus_h{h}_{d}.png'; draw(group,focus,f'{d} | {h}-session forecasts | baselines and THINK',True)
            for path in [f,focus]: shutil.copyfile(path,dest/'figures'/path.name)
            sections += [f'## {d}, {h}-session forecasts','',f'![All models](figures/{f.name})','',f'![Baselines and THINK](figures/{focus.name})','',
                         '| Model | Net Sharpe | Annualized net return | Net total return | Drawdown | ΔSharpe vs cohort reference [95% interval] |',
                         '|---|---:|---:|---:|---:|---:|']
            table=out[(out.horizon==h)&(out.dataset==d)]
            for r in table.itertuples():
                if r.model in FOCUS:
                    sections.append(f'| {LABEL.get(r.model,r.model)} | {r.sharpe:.3f} | {r.annualized_return:.1%} | {r.cumulative_return:.1%} | {r.maximum_drawdown:.1%} | {r.delta_benchmark:.3f} [{r.delta_low:.3f}, {r.delta_high:.3f}] |')
            fixed=[]
            source=Path('runs/robust_2022_2025_final' if h==5 else 'runs/daily_2022_2025_final') if study=='original' else Path('runs/largecap_20261010')/('five_session' if h==5 else 'daily')
            for kind in FOCUS:
                if kind not in group.model.unique(): continue
                if kind in ['qqq','market_proxy','equal_universe']:
                    fixed.append(group[group.model==kind]); continue
                for year in [2022,2023,2024,2025]:
                    piece=pd.read_csv(source/f'{d}_{year}'/f'{kind}_equal_k10_daily.csv')
                    piece['model']=kind; piece['horizon']=h;piece['dataset']=d;piece['fold_year']=year;fixed.append(piece)
            fixed=pd.concat(fixed,ignore_index=True)
            f=root/'figures'/f'audited_fixed_k10_h{h}_{d}.png';draw(fixed,f,f'{d} | {h}-session forecasts | fixed equal top 10',True)
            shutil.copyfile(f,dest/'figures'/f.name)
            sections += ['',f'![Fixed equal top 10](figures/{f.name})','',
                         'The fixed-top-10 panel holds allocation constant across models. SPY/QQQ and the full-cohort basket retain their control rules.','']
            # Replace every NYSE annual panel linked by the old documents, removing QQQ.
            if d.startswith('NYSE'):
                for year,piece in group.groupby('fold_year',sort=True):
                    f=root/'figures'/f'panel_h{h}_{d}_{int(year)}.png';draw(piece,f,f'{d} | {int(year)} | {h}-session forecasts')
                    if (dest/'figures'/f.name).exists(): shutil.copyfile(f,dest/'figures'/f.name)
                f=root/'figures'/f'all_models_h{h}_{d}.png'; draw(group,f,f'{d} | {h}-session forecasts | cohort reference')
                if (dest/'figures'/f.name).exists(): shutil.copyfile(f,dest/'figures'/f.name)
        if study=='original':
            sections += ['## Fresh random draws','', '[Three fresh samples per exchange](random_trials_20261010/CHARTS.md). These draws use the full published source pools before downloading prices.','']
        else:
            sections += ['## Data and selection limitations','',
                         'The NYSE top-100 is a custom market-cap cohort, not the official NYSE 100 growth index. NASDAQ includes 101 share classes for 100 companies. '
                         'Current membership creates future selection. Revised public prices and missing delistings remain unresolved. '
                         'Intervals use 1,000 paired 20-session block samples within annual folds and are exploratory, with no correction for repeated comparisons.','']
        content='\n'.join(sections)
        for target in [root/'CHARTS.md',dest/'CHARTS.md']: target.write_text(content,encoding='utf-8');textfiles.append(str(target))
    Path('docs/audit/benchmark_image_replacements.json').write_text(json.dumps(OLD_IMAGES,indent=2))
    Path('docs/audit/benchmark_availability.json').write_text(json.dumps(dict(as_of='2026-10-10',requested_etf='NYSX',
        inception='2026-03-25',historical_2022_2025='unavailable',NYSE_historical_reference='equal_universe simulated cohort basket',
        NASDAQ_reference='QQQ',source='https://www.globalxetfs.com/funds/nysx/',updated_charts=textfiles),indent=2))
    print('Refreshed all-model, focused, fixed-K10 and NYSE annual charts',flush=True)

if __name__=='__main__':main()
