"""Reports for the 2024–2025 public-data extension, with matched-date SPY."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report_architecture_research import LABELS, table, fmt


def all_model_equity(root, primary, locks):
    """Every tested architecture/strategy/benchmark, with consistent line styles."""
    models = list(LABELS)
    assert set(primary.model.unique()) == set(models)
    palette = list(plt.get_cmap('tab20').colors)
    colors = {kind: palette[i] for i, kind in enumerate(models)}
    colors.update(market_proxy='#111111', buy_hold='#777777', equal_universe='#217a45')
    styles = {kind: ('--' if kind in ['buy_hold','equal_universe'] else '-') for kind in models}
    years = sorted(primary.year.unique())
    out = root/'figures'
    out.mkdir(exist_ok=True)
    inventory = []

    def draw(ax, name, year, kinds):
        for kind in kinds:
            daily = pd.read_csv(root/f'{name}_{year}'/f'{kind}_daily.csv')
            ax.plot(pd.to_datetime(daily.date), np.cumprod(1+daily.net),
                    label='SPY (S&P 500 ETF)' if kind=='market_proxy' else LABELS[kind],
                    color=colors[kind], linestyle=styles[kind],
                    linewidth=2.3 if kind=='market_proxy' else 1.25, alpha=1 if kind=='market_proxy' else .9)
            inventory.append(dict(dataset=name, year=int(year), model=kind, source=str(root/f'{name}_{year}'/f'{kind}_daily.csv')))
        ax.set_title(f'{name.replace("_recent", " cohort")} · {year}', fontsize=13)
        ax.set_ylabel('Net wealth per $1')
        ax.grid(alpha=.2)
        ax.tick_params(axis='x', rotation=25)

    fig, axes = plt.subplots(len(years), len(locks), figsize=(17, 11), squeeze=False)
    for x, name in enumerate(locks):
        for y, year in enumerate(years):
            draw(axes[y,x], name, year, models)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5,.04), frameon=False)
    fig.suptitle('Every tested model and benchmark · net equity in 2024 and 2025', fontsize=17)
    fig.text(.02,.015,'Separate annual portfolios; validation-selected K/weights; matched dates and costs. Public-data survivorship bias remains.',fontsize=10)
    fig.tight_layout(rect=(0,.23,1,.96))
    fig.savefig(out/'all_models_equity.png',dpi=170)
    plt.close(fig)
    # Full-size individual panels make all 15 overlapping lines easier to inspect.
    for name in locks:
        for year in years:
            fig, ax = plt.subplots(figsize=(14,6.5))
            draw(ax,name,year,models)
            ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=9,frameon=False)
            fig.text(.02,.015,'All 15 tested models/benchmarks; costs included. Fixed public-data cohort, not a survivorship-free exchange universe.',fontsize=9)
            fig.tight_layout(rect=(0,.06,1,1))
            fig.savefig(out/f'all_models_{name}_{year}.png',dpi=170)
            plt.close(fig)
    # Additional focused comparisons retain the full overview above.
    families = {'think_ablations': ['think','think_rank','think_gate','think_mix','think_risk','market_proxy'],
                'alternative_models': ['mlp','lstm','stockmixer','market_transformer','ridge','momentum126','reversal5','market_proxy']}
    for family, kinds in families.items():
        fig, axes = plt.subplots(len(years),len(locks),figsize=(16,10),squeeze=False)
        for x,name in enumerate(locks):
            for y,year in enumerate(years):
                draw(axes[y,x],name,year,kinds)
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,.04),frameon=False)
        fig.suptitle('THINK component ablations' if family=='think_ablations' else 'Alternative models and simple strategies',fontsize=17)
        fig.text(.02,.015,'Net equity; validation-selected K/weights per model; separate annual windows. Public-data survivorship bias remains.',fontsize=10)
        fig.tight_layout(rect=(0,.19,1,.96))
        fig.savefig(out/f'{family}_equity.png',dpi=170)
        plt.close(fig)
    (root/'line_chart_inventory.json').write_text(json.dumps(inventory,indent=2))


def generate(root):
    f = pd.read_csv(root/'all_metrics.csv')
    primary = f[(f.seed=='ensemble') & f.selected_policy & (f.cost_multiplier==1)].copy()
    locks = json.loads((root/'SELECTION_LOCK.json').read_text())
    completed = json.loads((root/'completion.json').read_text())
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    relative = pd.read_csv(root/'spy_relative_metrics.csv')
    coverage = pd.read_csv(root/'universe_coverage.csv')
    random = pd.read_csv(root/'random_topk_controls.csv')
    ci = json.loads((root/'paired_bootstrap_by_year.json').read_text())
    training = pd.read_csv(root/'training_records.csv')
    reused = json.loads((root/'FIT_REUSE_AUDIT.json').read_text()) if (root/'FIT_REUSE_AUDIT.json').exists() else []
    primary['label'] = primary.model.map(LABELS).where(primary.model!='market_proxy', 'SPY (S&P 500 ETF)')
    primary = primary.merge(relative, on=['dataset','year','model'], validate='one_to_one')
    primary['delta_sharpe_spy'] = primary.net_sharpe-primary.spy_net_sharpe
    primary.to_csv(root/'yearly_comparison.csv', index=False)
    deployed = []
    for name, selection in locks.items():
        for year, chosen in selection['by_year'].items():
            row = primary[(primary.dataset==name)&(primary.year==int(year))&(primary.model==chosen['winner'])].iloc[0]
            deployed.append(row.to_dict())
    deployed = pd.DataFrame(deployed)
    deployed.to_csv(root/'validation_selected_by_year.csv', index=False)
    positive = int((deployed.delta_sharpe_spy>0).sum())
    summaries = []
    for name in locks:
        for kind in LABELS:
            part = primary[(primary.dataset==name)&(primary.model==kind)].sort_values('year')
            if part.empty:
                continue
            summaries.append(dict(dataset=name, model=kind, label=part.label.iloc[0],
                                  median_net_sharpe=part.net_sharpe.median(), worst_net_sharpe=part.net_sharpe.min(),
                                  median_gross_sharpe=part.gross_sharpe.median(), final_net_sharpe=part.net_sharpe.iloc[-1],
                                  IC=part.IC.mean(), RankIC=part.RankIC.mean(), ndcg=part['NDCG@10'].mean(),
                                  median_drawdown=part.net_maximum_drawdown.median(), median_turnover=part.turnover_annual.median()))
    pd.DataFrame(summaries).to_csv(root/'architecture_comparison.csv', index=False)
    lines = ['# Modern stock-ranking evaluation: 2024–2025 versus SPY', '',
             '**This supersedes the 2015–2017 exchange-cohort tests as the active comparison.** '
             'The previous experiment remains available for provenance; its numbers have not been relabeled as modern results.', '',
             f'Completed **{completed["neural_fits"]} neural fits**: nine architectures × three fixed seeds × two cohorts × two test years. '
             f'The validation-selected architecture exceeded SPY’s net Sharpe in **{positive}/{len(deployed)} cohort/year comparisons**. '
             'Selection used only validation information available before the relevant test year; the highest test result is not promoted after the fact.', '',
             '[Open all graphs](CHARTS.md) · [Every model/year metric](yearly_comparison.csv) · '
             '[All K, weighting, seed and cost scenarios](all_metrics.csv)', '',
             '## Data quality is the principal limitation', '',
             'This is a **public-data, availability-conditioned research test**, not a survivorship-free simulation of the NYSE or NASDAQ. '
             'We retained the same 100 SHA256-selected tickers from each published 2017 research cohort, chosen before fetching recent prices. '
             'Names were not replaced because their returns were poor or their downloads failed. '
             'Nevertheless, many original securities have no recoverable Yahoo history, and cannot become eligible: '
             '**that omission leaves material survivorship bias, including bankrupt companies.** '
             'The remaining data do not establish what the model would have achieved on all original names.', '',
             'The cohort labels describe their historical source, not verified current exchange membership. '
             'They include funds and preferred securities as well as common stocks. '
             'The universe is fixed and does not add later IPOs or reconstruct historical index membership.', '',
             'Two identity errors were blocked before any training: original AVX was acquired by Kyocera, while today’s AVX '
             'series belongs to AVAX One; original CAMP was CalAmp, while the returned series belongs to CAMP4 Therapeutics. '
             'Their unrelated replacement-company prices remain in the raw snapshot for audit but are masked from all features, labels and trading. '
             'The original issuer histories remain unavailable. Evidence: '
             '[Kyocera history](https://global.kyocera.com/company/history/2020yrs.html), '
             '[AVAX One SEC filing](https://www.sec.gov/Archives/edgar/data/1826397/000149315225023464/form10-q.htm), '
             '[CalAmp bankruptcy filing](https://www.sec.gov/Archives/edgar/data/730255/000119312524180486/0001193125-24-180486-index.htm), '
             '[CAMP4 investor FAQ](https://investors.camp4tx.com/resources/investor-faqs).', '',
             '| Cohort | Frozen names | No returned history | Wrong issuer rejected | Original-issuer histories available |',
             '|---|---:|---:|---:|---:|']
    for name in locks:
        c = coverage[coverage.dataset==name]
        any_data = c[[x for x in c.columns if x.startswith('observed_')]].sum(axis=1)>0
        lines.append(f'| {name} | {len(c)} | {int((~any_data).sum())} | {int(c.identity_rejected.sum())} | {int((any_data&~c.identity_rejected).sum())} |')
    lines += ['', 'Per-security coverage is recorded in `universe_coverage.csv`; all missing names are retained there. '
              'Adjusted closes approximate total returns for stocks and SPY, using revised data downloaded now. '
              'No price-vintage, comprehensive delisting-payout or historical lending database is available.', '',
              '## Chronological design and execution', '',
              '- 2024: training signals from 2019–2022, validation 2023, test 2024. 2025: training 2020–2023, validation 2024, test 2025. '
              'Exact purged boundaries are saved in each `fold.json`. Labels mature before the next split; dates are never shuffled.',
              '- Same nine model variants as the earlier comparison: THINK, ranking-loss/gating/mixing/volatility-task ablations, '
              'MLP, LSTM, StockMixer-inspired MLP and MASTER-inspired Transformer. These remain local adaptations, not full paper reproductions. '
              'This architecture menu was designed retrospectively; chronological price access does not imply the research decisions were actually made in 2024.',
              '- Fixed 16-session lookback, five causal features, width 16, five-session prediction horizon, seeds 7/19/42. '
              'AdamW learning rate 0.001, weight decay 0.001, batch eight, 30 epochs maximum, six-epoch validation-MSE patience. '
              'Feature scaling and correlation hyperedges are training-only.',
              '- Scores are generated after close t; orders are sized using information at t, with a next-close t+1 fill proxy. '
              'Hold to t+6 and rebalance every five sessions. Each annual fold starts with $1m cash and evaluates complete holding periods only.',
              '- Eligibility requires 40 valid price observations and trailing 20-session dollar ADV ≥ $1m. '
              'Trades are capped at 1% of lagged ADV. Per-side costs: 2bp fee + 5bp slippage/spread + 3bp × sqrt(participation/1%). '
              'Cash earns zero; Sharpe uses a fixed 3% annual hurdle and 252-session annualization. RF=0 Sharpe is also saved.',
              '- Equal-weight or confidence-weighted top K, K={5,10,20}, selected solely on validation stability. '
              'Long-short 50/50 portfolios with 3% borrow are diagnostics only; no verified borrow availability.',
              '- Equal-weight ensembles rank each seed within the **eligible** cross-section. Unavailable symbols cannot affect ensemble ranks. '
              'This explicit masking improves the earlier evaluator’s handling of sparse universes.',
              '- Architecture choice is locked separately for each year, using that year’s validation and earlier validation folds only. '
              'The 2025 choice can use 2024 information; it cannot retroactively choose the 2024 architecture. '
              'All choices are serialized before test metrics are computed.',
              ('- An initial attempt stopped during momentum validation because an ineligible NaN score contaminated confidence weights. '
              'This was repaired and regression-tested before any test metrics were produced. The 27 completed neural fits were reused '
              'only after exact checks of predictions, preprocessing, split indices, model hashes and graph groups; '
              '`FIT_REUSE_AUDIT.json` records these checks. Remaining seed fits ran in independent worker processes.'
              if reused else '- All fits were trained fresh for this run; seed workers are independent.'),
              '- SPY is the primary index-fund benchmark for **both** cohorts, with adjusted-close total-return approximation and 7bp entry/exit costs. '
              'Every comparison uses exactly matching observed trading dates. Gross model results exclude fees/borrow on matched holdings; '
              'separate zero-, double- and quadruple-cost scenarios are saved.',
              '- Prices and fills are adjusted-price accounting approximations, not audited broker executions. '
              'Boundary gaps between annual folds are not stitched into a claimed continuous live portfolio.', '',
              '## Validation-selected models versus SPY', '',
              table(deployed, [('dataset','Cohort'),('year','Test year'),('label','Validation choice'),('method','Weights'),('k','K'),
                               ('net_sharpe','Net SR'),('spy_net_sharpe','SPY SR'),('delta_sharpe_spy','Δ SR'),
                               ('net_cumulative_return','Net return'),('net_maximum_drawdown','Max DD'),
                               ('IC','IC'),('RankIC','RankIC')], ['net_cumulative_return','net_maximum_drawdown']), '',
              '![Every tested model and benchmark](figures/all_models_equity.png)', '',
              'All 15 model/benchmark kinds appear in the line-chart overview. The graph collection also contains '
              'full-size panels and focused architecture-family comparisons. No test winner is substituted for the validation choice.', '',
              '## Every architecture and benchmark, by test year', '']
    investment = [('label','Model'),('method','Weights'),('k','K'),('gross_sharpe','Gross SR'),('net_sharpe','Net SR'),
                  ('net_annualized_return','Ann. return'),('net_cumulative_return','Cumulative'),
                  ('net_annualized_volatility','Ann. vol'),('net_maximum_drawdown','Max DD'),
                  ('turnover_annual','Turnover/year'),('profitable_periods','Win periods')]
    percentages = ['net_annualized_return','net_cumulative_return','net_annualized_volatility','net_maximum_drawdown','profitable_periods']
    ranking = [('label','Model'),('IC','IC'),('RankIC','RankIC'),('ICIR','ICIR'),('RankICIR','RankICIR'),
               ('NDCG@5','NDCG@5'),('NDCG@10','NDCG@10'),('NDCG@20','NDCG@20')]
    for name in locks:
        for year in sorted(primary[primary.dataset==name].year.unique()):
            sub = primary[(primary.dataset==name)&(primary.year==year)].set_index('model').reindex(LABELS).reset_index()
            one = sub.iloc[0]
            lines += [f'### {name}, {year}', '',
                      f'Daily P&L dates: **{one.first_date} to {one.last_date}**; {int(one.net_days)} sessions, '
                      f'{int(one.prediction_periods)} prediction periods. Eligible universe: '
                      f'{int(one.eligible_min)}–{int(one.eligible_max)} names. Missing eligible test labels: {int(one.missing_test_labels)}.', '',
                      table(sub, investment, percentages), '', table(sub[sub.IC.notna()], ranking), '',
                      'SPY-relative diagnostics (descriptive OLS alpha is not proof of skill):', '',
                      table(sub, [('label','Model'),('delta_sharpe_spy','Δ SR vs SPY'),('beta_to_spy','SPY beta'),
                                  ('alpha_annualized_ols','Ann. OLS alpha'),('tracking_error','Tracking error'),
                                  ('information_ratio','Information ratio')], ['alpha_annualized_ols','tracking_error']), '']
    lines += ['IC and RankIC measure each prediction against realized returns across the entire observed eligible cross-section. '
              'ICIR/RankICIR are unannualized mean/sample-standard-deviation ratios across prediction periods. '
              'NDCG uses linear percentile realized-return relevance, with tie-aware discounts. '
              'Benchmarks have no stock scores, so their prediction metrics are undefined. '
              'Turnover counts both buys and sells in multiples of NAV, including entry and estimated exit. '
              'Win rate is the fraction of profitable five-session periods after costs.', '',
              '## Fixed-policy component ablations', '',
              'Equal weights, K=10, same dates/costs/seeds. Each delta is relative to THINK within the same fold; '
              'this separates the component effect from selecting different K or weights.', '',
              '| Cohort | Year | Component | Δ net SR | Δ IC | Δ RankIC |', '|---|---:|---|---:|---:|---:|']
    fixed = f[(f.seed=='ensemble')&(f.cost_multiplier==1)&(f.method=='equal')&(f.k==10)]
    for (name, year), group in fixed.groupby(['dataset','year'], sort=False):
        base = group[group.model=='think'].iloc[0]
        for kind in ['think_rank','think_gate','think_mix','think_risk']:
            row = group[group.model==kind].iloc[0]
            lines.append(f'| {name} | {year} | {LABELS[kind]} | {fmt(row.net_sharpe-base.net_sharpe)} | {fmt(row.IC-base.IC)} | {fmt(row.RankIC-base.RankIC)} |')
    lines += ['', '## Random top-K controls', '',
              'Fifty fixed random score paths per cohort/year/K, equal-weight portfolios, same trading rules and costs. '
              'These test how much performance a random selection from this available universe can produce. '
              'They share the universe bias and cannot repair missing securities. Ranges below are empirical 5th–95th percentiles, '
              'not confidence intervals or a multiple-testing-adjusted significance test.', '',
              '| Cohort | Year | K | Random median SR | Random 5th–95th SR | THINK SR | Mixer SR | SPY SR |',
              '|---|---:|---:|---:|---|---:|---:|---:|']
    for (name, year, k), group in random.groupby(['dataset','year','k']):
        matched = f[(f.dataset==name)&(f.year==year)&(f.k==k)&(f.method=='equal')&(f.seed=='ensemble')&(f.cost_multiplier==1)]
        spy = primary[(primary.dataset==name)&(primary.year==year)&(primary.model=='market_proxy')].net_sharpe.iloc[0]
        lines.append(f'| {name} | {year} | {k} | {fmt(group.net_sharpe.median())} | '
                     f'[{fmt(group.net_sharpe.quantile(.05))}, {fmt(group.net_sharpe.quantile(.95))}] | '
                     f'{fmt(matched[matched.model=="think"].net_sharpe.iloc[0])} | '
                     f'{fmt(matched[matched.model=="think_mix"].net_sharpe.iloc[0])} | {fmt(spy)} |')
    lines += ['', '## Uncertainty of validation-selected results', '',
              'Paired circular 20-session block bootstrap, 1,000 replicates, net Sharpe differences. '
              'These intervals are conditional on fitted models and available data, unadjusted for multiple comparisons. '
              'They do not capture survival/availability bias or all model-selection uncertainty.', '',
              '| Cohort/year | Validation-selected model | Comparator | Δ SR | Exploratory 95% interval |',
              '|---|---|---|---:|---|']
    for fold, record in ci.items():
        for base, interval in record['comparisons'].items():
            lines.append(f'| {fold} | {LABELS[record["winner"]]} | {"SPY" if base=="market_proxy" else LABELS[base]} | '
                         f'{fmt(interval["observed"])} | [{fmt(interval["low"])}, {fmt(interval["high"])}] |')
    lines += ['', '## Seed, K and cost stability of the validation selections', '']
    for _, choice in deployed.iterrows():
        sub = f[(f.dataset==choice.dataset)&(f.year==choice.year)&(f.model==choice.model)]
        lines += [f'### {choice.dataset}, {choice.year}: {choice.label}', '',
                  table(sub[(sub.seed!='ensemble')&(sub.cost_multiplier==1)], [('seed','Seed'),('net_sharpe','Net SR'),('IC','IC'),('RankIC','RankIC')]), '',
                  table(sub[(sub.seed=='ensemble')&(sub.cost_multiplier==1)], [('method','Weights'),('k','K'),('net_sharpe','Net SR'),
                        ('net_maximum_drawdown','Max DD'),('turnover_annual','Turnover/year')], ['net_maximum_drawdown']), '',
                  table(sub[(sub.seed=='ensemble')&sub.selected_policy].sort_values('cost_multiplier'),
                        [('cost_multiplier','Cost multiplier'),('net_sharpe','Net SR'),('net_cumulative_return','Cumulative return')], ['net_cumulative_return']), '']
    lines += ['## Accounting and interpretation', '',
              f'{int((training.epochs_run>=protocol["arguments"]["epochs"]).sum())}/{len(training)} fits reached the 30-epoch ceiling. '
              'This remains a bounded architecture comparison, not an exhaustive optimization.', '',
              f'Maximum stale-position marks across primary stock portfolios: {int(primary.stale_position_marks.max())}. '
              f'Maximum terminal-liquidation orders above the participation cap: {int(primary.terminal_liquidation_over_cap.max())}. '
              'Terminal charges are modeled exit estimates. An absence of stale held marks does not establish complete delisting coverage: '
              'missing entire histories can prevent those securities from ever entering a portfolio.', '',
              'Two adjacent test years provide limited regime coverage. Any apparent winner is descriptive unless it was the validation choice. '
              'Do not tune the architecture against these now-observed years. Credible promotion requires recoverable delisted histories, '
              'historical security identifiers/membership, and new forward evaluation. StockMixer/MASTER remain local adaptations.', '',
              '## Reproduction', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe fetch_recent_cohorts.py',
              '.\\.venv\\Scripts\\python.exe -m unittest -v test_research',
              '.\\.venv\\Scripts\\python.exe run_architecture_research.py --datasets NYSE_recent NASDAQ_recent --test-years 2024 2025 --workers 3 --out runs\\recent_2024_2025_new',
              '.\\.venv\\Scripts\\python.exe audit_architecture_research.py --run runs\\recent_2024_2025_new',
              '.\\.venv\\Scripts\\python.exe analyze_recent_research.py --run runs\\recent_2024_2025_new',
              '.\\.venv\\Scripts\\python.exe report_recent_research.py --run runs\\recent_2024_2025_new',
              '.\\.venv\\Scripts\\python.exe plot_research_metrics.py --run runs\\recent_2024_2025_new', '```', '',
              'The downloader refuses to overwrite a completed snapshot. Use the saved prices for exact reproduction; '
              'new vendor downloads may differ. `DATA_INPUT_LOCK.json`, `PROTOCOL.json`, input copies, source snapshots, '
              'per-fold checkpoints/predictions/scalers/policies and all daily paths preserve provenance. '
              'Data and runs are Git-ignored; retain them with the code when sharing the experiment.', '']
    interpretation = root/'INTERPRETATION.md'
    if interpretation.exists():
        lines[6:6] = interpretation.read_text(encoding='utf-8').splitlines()+['']
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    deployed.to_csv(root/'final_selected_comparison.csv', index=False)
    years = sorted(primary.year.unique())
    for filename, plot_years in [('test_equity.png', years), ('final_equity.png', years[-1:])]:
        fig, axes = plt.subplots(len(plot_years), len(locks), figsize=(14, 4.8*len(plot_years)), squeeze=False)
        for x, name in enumerate(locks):
            for y, year in enumerate(plot_years):
                ax = axes[y, x]
                winner = locks[name]['by_year'][str(year)]['winner']
                for kind in dict.fromkeys([winner, 'think', 'buy_hold', 'market_proxy']):
                    daily = pd.read_csv(root/f'{name}_{year}'/f'{kind}_daily.csv')
                    ax.plot(pd.to_datetime(daily.date), np.cumprod(1+daily.net),
                            label=('SPY (S&P 500)' if kind=='market_proxy' else LABELS[kind])+
                            (' [validation choice]' if kind==winner else ''), linewidth=1.6)
                ax.set_title(f'{name.replace("_recent", " cohort")} · {year}')
                ax.set_ylabel('Net wealth per $1')
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
                ax.tick_params(axis='x', rotation=25)
        fig.suptitle('Matched-date out-of-sample portfolios versus SPY', fontsize=16)
        fig.text(.02, .01, 'Separate annual portfolios; costs included. Public-data availability/survivorship bias remains.', fontsize=10)
        fig.tight_layout(rect=(0,.04,1,.96))
        fig.savefig(root/filename, dpi=170)
        plt.close(fig)
    print(root/'REPORT.md', flush=True)
    all_model_equity(root, primary, locks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('runs/recent_2024_2025'))
    generate(parser.parse_args().run)
