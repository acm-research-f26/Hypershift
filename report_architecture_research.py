"""Describe locked experiments without choosing anything from held-out results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_engine import performance

LABELS = {'think':'THINK reconstruction', 'think_rank':'THINK + ranking loss',
          'think_gate':'THINK + market feature gate', 'think_mix':'THINK + stock mixer residual',
          'think_risk':'THINK + volatility auxiliary task', 'mlp':'Shared MLP', 'lstm':'LSTM',
          'stockmixer':'StockMixer-inspired MLP', 'market_transformer':'MASTER-inspired Transformer',
          'ridge':'Ridge regression', 'momentum126':'126-session momentum', 'reversal5':'5-session reversal',
          'buy_hold':'Equal-weight buy-and-hold', 'equal_universe':'Equal-weight rebalanced universe',
          'market_proxy':'Market ETF proxy'}


def fmt(x, percent=False):
    if x is None or pd.isna(x):
        return '—'
    return f'{100*x:.1f}%' if percent else f'{x:.3f}'


def table(frame, columns, percentage=()):
    lines = ['| '+' | '.join(label for _,label in columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        values = []
        for key,_ in columns:
            val = row.get(key, np.nan)
            if key in ('year', 'k', 'days', 'prediction_periods') and not pd.isna(val):
                values.append(str(int(val)))
            else:
                values.append(fmt(val, key in percentage) if not isinstance(val,str) else val)
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=Path('runs/architecture_research'))
    a = p.parse_args()
    root = a.run
    f = pd.read_csv(root/'all_metrics.csv')
    if all(str(name).endswith('_recent') for name in f.dataset.unique()):
        from report_recent_research import generate
        generate(root)
        return
    locked = json.loads((root/'SELECTION_LOCK.json').read_text())
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    complete = json.loads((root/'completion.json').read_text())
    ci = json.loads((root/'paired_bootstrap_final.json').read_text())
    training = pd.read_csv(root/'training_records.csv')
    primary = f[(f.seed=='ensemble') & f.selected_policy & (f.cost_multiplier==1)].copy()
    lines = ['# Controlled architecture research: results and limits', '',
             'This run tests architectural ideas against the local THINK reconstruction. It is **conditional research evidence, not a verified investable edge**. The test results are descriptive; every deployed fold policy and final architecture choice was selected from chronological validation data before test scoring.', '',
             f'Completed **{complete["neural_fits"]} neural fits**, three fixed seeds, nine neural architectures/ablations, three simple ranking strategies, and three portfolio benchmarks. The data comprise the existing 12-stock basket and deterministic 100-name subsets of each academic NYSE/NASDAQ cohort. These are **not the full exchange universes**.', '',
             '## Data and protocol limitations', '',
             '- All three universes inherit survivor/selection bias. There is no verified historical constituent or delisting feed. Rebuilding causal features cannot remove this upstream bias.',
             '- The published exchange cohorts include securities such as closed-end funds (for example ZTR), not a verified point-in-time common-stock-only universe. Historical security-type filtering is unavailable.',
             '- Modern12 uses revised Yahoo adjusted prices, a total-return approximation. The old Google price/dividend/split conventions are unverified, so NYSE/NASDAQ are **price-series diagnostics**, not certified total-return backtests. Comparisons against total-return ETFs have this mismatch.',
             '- Adjusted price units represent return accounting, not literal exchange share prices. The simulator uses next-close price ratios and modeled slippage as an auction proxy; ex-dividend fill sizing, dividend reinvestment and actual auction fills are approximations, not audited broker executions.',
             '- Old 2017 data and some modern historical windows were examined in earlier work. Only the modern 2026 segment was newly introduced here; it still shares the biased universe. No claim of a wholly pristine historical research process is made.',
             '- Fixed ticker-hash sampling precedes examining outcomes. Each forecast ranks every signal-time eligible name in that fixed sample; it does not screen on future returns. Missing outcomes are counted and omitted from predictive metrics only, never used to remove positions in the simulator.',
             '- Unavailable held marks are carried forward and explicitly counted. Historical trading suspensions, delisting payouts, auction depth, securities lending and corporate-action vintages are unavailable. Nonzero stale marks or liquidation-cap breaches weaken economic interpretation.',
             '- This is a bounded 30-epoch CPU comparison, not an exhaustive architecture/hyperparameter search. StockMixer/MASTER variants are small adaptations of their core ideas, not official-paper reproductions. No pretrained model is used.', '',
             'Before old-cohort fitting or any test scoring, two independently verified adjustment errors were repaired with originals preserved: EXPO prices from 2015-06-04 were divided by two and volume multiplied by two; one ZTR bar on 2012-06-26 was replaced using the independent split-adjusted reference. The actions are corroborated by [Exponent’s SEC release](https://www.sec.gov/Archives/edgar/data/851520/000114420415034077/v411916_ex99-1.htm) and [Virtus’s reverse-split notice](https://ir.virtus.com/news/news-details/2012/Zweig-Fund-And-Zweig-Total-Return-Fund-Announce-Date-Of-Reverse-Stock-Split/default.aspx). `corrections.json` records original/corrected hashes. This is a documented data repair, not a comprehensive certification of all corporate actions.', '',
             '## Frozen experimental design', '',
             'Models see 16 observations of five causal inputs: daily return, price relative to 5- and 20-session means, 20-session volatility and log dollar-volume/ADV. A scaler fitted only on the training fold standardizes inputs; values are clipped to ±8 standard deviations and multiplied by 0.2 for hyperbolic numerical stability. Graph groups use up to four positive return-correlation neighbors estimated from training history. Static future-dated industry/Wikidata relations and the old future-maximum price normalization are not used.', '',
             'The existing `ThinkReconstruction` backbone is reused with kernel sizes (4,4), width 16 and a direct 5-session return target in percentage points. This changes the old pilot’s task/inputs/training protocol; it does not claim to reproduce the published THINK experiment. Each ablation changes one component relative to this matched baseline.', '',
             'Rolling training uses at most four years before the one-year validation window. Training labels mature before validation begins, and validation labels mature before test begins. Windows use increasing dates, including training batches. Models are fitted afresh annually, frozen during their test year, and never updated using immature labels. Test-year training can legitimately include outcomes observed during earlier years.', '',
             'Each fold starts from cash and retains only complete five-session holding periods. Models and benchmarks share those exact dates. Boundary gaps are not stitched into a claimed continuous live-investment history; annualization uses the observed sessions within each fold.', '',
             'AdamW: learning rate 0.001, weight decay 0.001, batch eight, 30 epochs maximum, six-epoch patience, gradient norm cap one. Seeds 7, 19, 42. Checkpoints minimize validation return MSE. Portfolio selection then maximizes median validation half-year/seed net Sharpe minus 0.25 times its standard deviation. This penalizes unstable configurations; it does not directly optimize test Sharpe.', '',
             'Rebalance every five sessions. Scores are formed after close t; share targets use signal-time prices and NAV and fill at close t+1, plus modeled costs. The label is the realized close t+1 to t+6 return. No same-close entry. Daily marks within each holding period are used for volatility and drawdown. A $1m portfolio trades at most 1% of trailing 20-session dollar ADV; eligibility requires ADV ≥ $1m and 40 valid price observations. Cash earns zero. Initial purchases and an estimated terminal liquidation charge are included.', '',
             'Cost per traded dollar per side: **2bp fee + 5bp spread/slippage + 3bp × sqrt(participation / 1%)**. Participation is measured against lagged ADV. Terminal liquidation costs are estimates; any trades exceeding the cap are flagged, not certified executable. Gross metrics use matched actual holdings without fees/borrow; a separate zero-cost rerun is in the stress results. Neither is a literal frictionless execution guarantee.', '',
             'Long-only policies: equal-weight or softmax confidence-weighted top K. K={3,5,8} for Modern12; K={5,10,20} for NYSE/NASDAQ. Confidence weights are relative scores, not calibrated return probabilities. Long-short scenarios allocate 50% long/50% short and charge 3% annual borrow, but are **excluded from model/policy selection** because historical locates and borrow rates are unavailable.', '',
             'Primary models are equal-weight rank ensembles of all three seeds. Sharpe uses a fixed 3% annual hurdle, with a zero-hurdle version also recorded. Annualization uses 252 trading sessions. Turnover is total absolute traded notional divided by NAV (buys plus sells), with annual turnover reported in multiples of NAV. ICIR and RankICIR are unannualized mean/sample-standard-deviation ratios across nonoverlapping prediction periods.', '',
             'NDCG uses linear percentile relevance derived from realized cross-sectional returns; the least-returning stock has relevance zero and the best one relevance one. This keeps relevance meaningful even on all-negative days. Exact score ties share expected discounts; tie allocations in portfolios are fractional. IC/RankIC cover the entire observed eligible cross-section, never only the selected top K.', '',
             '## Architecture mapping', '',
             '| Variant | Isolated idea / distinction |', '|---|---|',
             '| THINK | Reused local temporal–hypergraph–temporal hyperbolic backbone; training-only correlation hyperedges |',
             '| THINK + ranking | Adds 0.1 × pairwise wrong-order return-gap loss to MSE |',
             '| THINK + gate | Multiplicative feature gate driven by eligible cross-sectional mean features, inspired by MASTER |',
             '| THINK + mixer | Adds stock→16-dimensional market→stock MLP residual to scores, inspired by StockMixer |',
             '| THINK + risk | Adds a volatility head on the shared first temporal representation; auxiliary weight 0.1, inspired by LiMT multitask learning |',
             '| Shared MLP / LSTM | Graph-free nonlinear and recurrent controls, same inputs and horizon |',
             '| StockMixer-inspired | Temporal, channel and cross-stock residual MLPs; ranking auxiliary loss |',
             '| MASTER-inspired | Market feature gate, positional temporal attention and masked cross-stock attention; ranking auxiliary loss |',
             '| Ridge / momentum / reversal | Ridge penalty 10 with unpenalized intercept; 126-session momentum; negative five-session return |', '',
             'Official sources: [THINK paper](https://tylersnetwork.github.io/papers/icdm22-think.pdf), [StockMixer](https://github.com/SJTU-DMTai/StockMixer), [MASTER](https://github.com/SJTU-DMTai/MASTER), [LiMT](https://arxiv.org/abs/2609.25617). LiMT-style volume forecasting, expert routing and portfolio optimization were not implemented. DoubleAdapt meta-learning, HIST concept discovery, LLMs, foundation models and order-book models were not tested; their required data/training assumptions do not match this bounded experiment.', '',
             '## All tested models: out-of-sample comparison', '',
             'Each row uses its own **validation-selected** portfolio in each fold. Sharpe medians/worst values measure stability across years; they are not pooled Sharpe ratios. Final means the last chronological test window (2026 or 2017). Tables retain the predeclared architecture order rather than sorting by test performance.', '']
    summaries = []
    for name in locked:
        subset = primary[primary.dataset==name]
        rows = []
        for kind in LABELS:
            part = subset[subset.model==kind].sort_values('year')
            if part.empty:
                continue
            last = part.iloc[-1]
            k = 5 if name=='modern12' else 10
            row = dict(dataset=name, model=kind, label=LABELS[kind],
                       median_net_sharpe=part.net_sharpe.median(), worst_net_sharpe=part.net_sharpe.min(),
                       median_gross_sharpe=part.gross_sharpe.median(), final_net_sharpe=last.net_sharpe,
                       IC=part.IC.mean(), RankIC=part.RankIC.mean(), ndcg=part[f'NDCG@{k}'].mean(),
                       median_drawdown=part.net_maximum_drawdown.median(), median_turnover=part.turnover_annual.median())
            rows.append(row)
            summaries.append(row)
        lines += [f'### {name}', '',
                  f'Validation-selected candidate: **{LABELS[locked[name]["winner"]]}**. Test windows: '+', '.join(map(str, sorted(subset.year.unique())))+'.', '',
                  table(pd.DataFrame(rows), [('label','Model'),('median_net_sharpe','Median net SR'),('worst_net_sharpe','Worst SR'),
                        ('median_gross_sharpe','Median gross SR'),('final_net_sharpe','Final net SR'),('IC','Mean IC'),('RankIC','Mean RankIC'),
                        ('ndcg',f'NDCG@{k}'),('median_drawdown','Median DD'),('median_turnover','Turnover/year')], ['median_drawdown']), '']
    pd.DataFrame(summaries).to_csv(root/'architecture_comparison.csv', index=False)
    lines += ['## Latest test: validation-selected candidate versus benchmarks', '']
    detail_rows = []
    for name, selection in locked.items():
        sub = primary[primary.dataset==name]
        year = sub.year.max()
        sub = sub[(sub.year==year)&sub.model.isin([selection['winner'],'think','buy_hold','market_proxy'])].copy()
        sub['label'] = sub.model.map(LABELS)
        detail_rows.extend(sub.to_dict('records'))
        lines += [f'### {name}, {year}', '',
                  table(sub, [('label','Model'),('method','Policy'),('k','K'),('gross_sharpe','Gross SR'),('net_sharpe','Net SR'),
                              ('net_annualized_return','Ann. return'),('net_cumulative_return','Cumulative'),('net_annualized_volatility','Ann. vol'),
                              ('net_maximum_drawdown','Max DD'),('profitable_periods','Win rate')],
                        ['net_annualized_return','net_cumulative_return','net_annualized_volatility','net_maximum_drawdown','profitable_periods']), '']
        ks = [3,5,8] if name=='modern12' else [5,10,20]
        lines += [table(sub[sub.IC.notna()], [('label','Model'),('IC','IC'),('RankIC','RankIC'),
                      ('ICIR','ICIR'),('RankICIR','RankICIR')]+[(f'NDCG@{k}',f'NDCG@{k}') for k in ks]+
                      [('turnover_mean','Mean turnover/rebalance'),('turnover_annual','Annual turnover')]), '']
        winner = selection['winner']
        lines += ['Paired 20-session moving-block bootstrap of **net Sharpe differences**, 1,000 resamples:', '',
                  '| Comparison | Difference | 95% interval |', '|---|---|---|']
        for base, result in ci[name].items():
            lines.append(f'| {LABELS[winner]} minus {LABELS[base]} | {fmt(result["observed"])} | [{fmt(result["low"])}, {fmt(result["high"])}] |')
        lines += ['', 'Intervals are exploratory, unadjusted for multiple comparisons, conditional on fitted models/data and short historical samples. They do not account for universe selection or data-vintage bias.', '']
    pd.DataFrame(detail_rows).to_csv(root/'final_selected_comparison.csv', index=False)
    lines += ['## Controlled THINK ablations', '',
              'To isolate architecture effects from portfolio selection, these comparisons fix **equal weights, K=5 (Modern12) or K=10 (old cohorts), identical costs, dates and seeds**. Deltas are medians of paired fold differences. A positive selected-policy Sharpe alone cannot establish a component’s value.', '',
              '| Dataset | Component | Δ net Sharpe | Δ IC | Δ RankIC | Positive Sharpe folds |', '|---|---|---|---|---|---|']
    for name in locked:
        k = 5 if name=='modern12' else 10
        sub = f[(f.dataset==name)&(f.seed=='ensemble')&(f.cost_multiplier==1)&(f.method=='equal')&(f.k==k)]
        base = sub[sub.model=='think'].set_index('year')
        for kind in ['think_rank','think_gate','think_mix','think_risk']:
            v = sub[sub.model==kind].set_index('year')
            delta = v[['net_sharpe','IC','RankIC']]-base[['net_sharpe','IC','RankIC']]
            lines.append(f'| {name} | {LABELS[kind]} | {fmt(delta.net_sharpe.median())} | {fmt(delta.IC.median())} | {fmt(delta.RankIC.median())} | {int((delta.net_sharpe>0).sum())}/{len(delta)} |')
    lines += ['', '## Seed, K and cost robustness', '',
              'The following diagnostics are reported without changing the locked selection. Net Sharpe ranges can be wide when sample size is small. Cost stress scales commission, spread/slippage and impact together, with the policy fixed.', '']
    for name, selection in locked.items():
        winner = selection['winner']
        year = primary[primary.dataset==name].year.max()
        lines += [f'### {name}: {LABELS[winner]}, {year}', '']
        sub = f[(f.dataset==name)&(f.year==year)&(f.model==winner)]
        seeds = sub[(sub.seed!='ensemble')&(sub.cost_multiplier==1)]
        lines += [table(seeds, [('seed','Seed'),('net_sharpe','Net SR'),('IC','IC'),('RankIC','RankIC')]), '']
        policies = sub[(sub.seed=='ensemble')&(sub.cost_multiplier==1)]
        lines += [table(policies, [('method','Policy'),('k','K'),('net_sharpe','Net SR'),('net_maximum_drawdown','Max DD'),('turnover_annual','Turnover/year')], ['net_maximum_drawdown']), '']
        stress = sub[(sub.seed=='ensemble')&sub.selected_policy].sort_values('cost_multiplier')
        lines += [table(stress, [('cost_multiplier','Cost multiple'),('net_sharpe','Net SR'),('net_annualized_return','Ann. return')], ['net_annualized_return']), '']
    lines += ['## Training and data diagnostics', '',
              f'{int((training.epochs_run>=protocol["arguments"]["epochs"]).sum())} of {len(training)} fits reached the epoch budget. This limits any claim that all architectures were optimally trained.', '',
              'Data-quality counts below are reported rather than repaired by dropping stocks after looking at future outcomes:', '',
              '| Dataset | Minimum eligible | Maximum eligible | Missing future labels | Max stale-position days | Max terminal cap breaches |', '|---|---|---|---|---|---|']
    for name in locked:
        meta = json.loads((root/f'{name}_data.json').read_text())
        sub = primary[primary.dataset==name]
        lines.append(f'| {name} | {meta["eligible_min"]} | {meta["eligible_max"]} | {meta["missing_future_labels"]} | {int(sub.stale_position_marks.max())} | {int(sub.terminal_liquidation_over_cap.max())} |')
    lines += ['', '## Interpretation', '',
              'A component is promising only if its matched-policy improvements recur across folds/universes/seeds, survive costs, and are accompanied by useful ranking quality. The validation-selected candidate is the deployment-style comparison; the highest descriptive test Sharpe is not a newly selected winner. Wide paired intervals, fragile seed results, high turnover or contradictory cross-universe ablations argue against promotion.', '',
              'Even a positive interval here is insufficient to establish an investable edge: unverified corporate actions, biased source cohorts, historical-data reuse, multiple comparisons and missing lending/execution data are not captured by the bootstrap. A production claim requires point-in-time constituent/delisting data, verified total returns, borrow/auction information, an externally frozen protocol and a genuinely new forward or paper-trading period. Further architecture changes must use new validation data, not these now-observed test windows.', '',
              '## Reproduction and audit trail', '',
              '```powershell', '.\\.venv\\Scripts\\python.exe -m unittest -v test_research',
              '.\\.venv\\Scripts\\python.exe run_architecture_research.py --out runs\\architecture_research_new',
              '.\\.venv\\Scripts\\python.exe report_architecture_research.py --run runs\\architecture_research_new', '```', '',
              'To acquire the same data on a fresh machine, run `fetch_research_panel.py` and `fetch_research_cohorts.py` first. The latter invokes `repair_research_corporate_actions.py`. Downloads from a changing provider may differ; use saved snapshot hashes when reproducing these exact numbers.', '',
              '- `PROTOCOL.json`: fixed architecture/portfolio search, costs, software versions and source hashes.',
              '- `source_snapshot/`: copies of the hashed training/model/accounting source files used for this run.',
              '- `SELECTION_LOCK.json`, `validation_search.json`: choices and all validation candidates, saved before test scoring.',
              '- Each dataset/year folder: date boundaries, scaler, graph memberships, model checkpoints, loss histories, predictions, full eligible masks, true outcomes, per-date IC/RankIC/NDCG and daily gross/net portfolio returns.',
              '- `all_metrics.csv`: every model × year × portfolio × K, all requested metrics, seeds and cost sensitivity. Benchmark prediction metrics are undefined rather than invented.',
              '- `architecture_comparison.csv`, `final_selected_comparison.csv`: compact comparison tables.',
              '- `paired_bootstrap_final.json`, `training_records.csv`, `completion.json`: uncertainty, training budgets and completion evidence.',
              '- `regime_diagnostics.csv`: signal-time rising/falling market trend crossed with high/low volatility, using a training-only volatility threshold. Conditional subsamples are descriptive, not separate tradable strategies.',
              '- `accounting_audit.csv`: independent recomputation of saved daily portfolio metrics and dates; `additional_source_hashes.json` covers benchmark/reference snapshots and corrections.',
              '- Data snapshots/source hashes remain in `data/research_panel/` and `data/research_cohorts/`; generated data/results are Git-ignored.', '']
    interpretation = root/'INTERPRETATION.md'
    if interpretation.exists():
        lines[4:4] = interpretation.read_text(encoding='utf-8').splitlines()+['']
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, axes = plt.subplots(1, len(locked), figsize=(16, 4.5), squeeze=False)
    for ax, (name, selection) in zip(axes[0], locked.items()):
        year = primary[primary.dataset==name].year.max()
        for kind in dict.fromkeys([selection['winner'],'think','buy_hold','market_proxy']):
            daily = pd.read_csv(root/f'{name}_{year}'/f'{kind}_daily.csv')
            ax.plot(pd.to_datetime(daily.date), np.cumprod(1+daily.net), label=LABELS[kind])
        ax.set_title(f'{name}: final {year} test')
        ax.set_ylabel('Net wealth per $1')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
        ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(root/'final_equity.png', dpi=150)
    plt.close(fig)
    print(root/'REPORT.md')


if __name__ == '__main__':
    main()
