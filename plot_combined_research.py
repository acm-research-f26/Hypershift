"""Plot the frozen annual accounts as combined 2022-2025 research curves.

No training, tuning or portfolio selection occurs here. Annual cash resets and
boundary gaps remain part of these synthetic fold-compounded paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research_engine import performance
from report_robust_research import LABELS
from report_architecture_research import table


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def insert_section(path, heading, body):
    text = path.read_text(encoding='utf-8')
    start, end = '<!-- combined-curves:start -->', '<!-- combined-curves:end -->'
    if start in text:
        a, rest = text.split(start, 1)
        _, b = rest.split(end, 1)
        text = a + b.lstrip('\n')
    first, rest = text.split('\n', 1)
    path.write_text(first + '\n\n' + start + '\n' + heading + '\n\n' + body +
                    '\n\n' + end + '\n\n' + rest.lstrip('\n'), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('runs/robust_comparison'))
    root = parser.parse_args().run
    input_hashes = {}

    def read(path):
        input_hashes[str(path)] = digest(path)
        return pd.read_csv(path)

    old = read(root / 'pooled_model_vs_spy_qqq.csv')
    annual = read(root / 'annual_comparison.csv')
    etf_annual = read(root / 'etf_controls_by_year.csv')
    lock_path = root / 'COMBINED_SELECTION_LOCK.json'
    input_hashes[str(lock_path)] = digest(lock_path)
    lock = json.loads(lock_path.read_text())
    roots = {int(c['horizon']): Path(c['run']) for group in lock['candidates'].values() for c in group}
    palette = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('Dark2').colors)
    colors = {kind: palette[i] for i, kind in enumerate(LABELS)}
    colors.update(think_mix_huber='#0072B2', market_proxy='#111111', qqq='#8024A8',
                  buy_hold='#888888', equal_universe='#247A45')
    metrics, curve_rows, figures, inventory = [], [], [], []
    paths = {}
    groups = [(h, d) for h in [5, 1] for d in ['NASDAQ_recent', 'NYSE_recent']]
    for h, dataset in groups:
        subset = old[(old.horizon == h) & (old.dataset == dataset)]
        kinds = subset.model.tolist() + ['qqq']
        dates = None
        for kind in kinds:
            pieces = []
            for year in [2022, 2023, 2024, 2025]:
                file = (root / 'benchmarks' / f'QQQ_h{h}_{dataset}_{year}.csv' if kind == 'qqq'
                        else roots[h] / f'{dataset}_{year}' / f'{kind}_daily.csv')
                piece = read(file)
                piece['fold_year'] = year
                pieces.append(piece)
            daily = pd.concat(pieces, ignore_index=True)
            assert daily.date.is_unique and daily.date.is_monotonic_increasing
            assert np.isfinite(daily[['gross', 'net']]).all().all()
            if dates is None:
                dates = daily.date.to_numpy()
            else:
                np.testing.assert_array_equal(dates, daily.date)
            computed = {f'{side}_{key}': value for side in ['net', 'gross']
                        for key, value in performance(daily[side]).items()}
            if kind != 'qqq':
                row = subset[subset.model == kind].iloc[0].to_dict()
                for key, value in computed.items():
                    np.testing.assert_allclose(value, row[key], rtol=1e-10, atol=1e-12,
                                               err_msg=f'{h}/{dataset}/{kind}/{key}')
            else:
                row = dict(horizon=h, dataset=dataset, model=kind, label='QQQ')
                eq = etf_annual[(etf_annual.horizon == h) & (etf_annual.dataset == dataset)
                                & (etf_annual.symbol == 'QQQ')]
                row['turnover_annual'] = np.average(eq.turnover_annual, weights=eq.net_days)
                # Use the same h-session evaluation blocks as the original SPY
                # control, without pretending that the ETF rebalances each block.
                assert all(len(piece) % h == 0 for piece in pieces)
                row['profitable_periods'] = np.average([
                    np.mean(np.prod((1 + piece.net.to_numpy()).reshape(-1, h), axis=1) > 1)
                    for piece in pieces], weights=[len(piece) for piece in pieces])
                np.testing.assert_allclose(computed['net_sharpe'], subset.qqq_sharpe, rtol=1e-10)
            row.update(computed, start=daily.date.iloc[0], end=daily.date.iloc[-1],
                       accounting='annual-reset fold compounding; boundary gaps omitted')
            metrics.append(row)
            daily['net_wealth'] = (1 + daily.net).cumprod()
            daily['gross_wealth'] = (1 + daily.gross).cumprod()
            # Include initial capital in the drawdown peak, consistent with performance().
            daily['net_drawdown'] = daily.net_wealth / np.maximum(1, daily.net_wealth.cummax()) - 1
            np.testing.assert_allclose(daily.net_drawdown.min(), computed['net_maximum_drawdown'], atol=1e-12)
            np.testing.assert_allclose(daily.net_wealth.iloc[-1] - 1, computed['net_cumulative_return'], atol=1e-12)
            daily['horizon'], daily['dataset'], daily['model'] = h, dataset, kind
            curve_rows.append(daily)
            paths[(h, dataset, kind)] = daily
            inventory.append(dict(horizon=h, dataset=dataset, model=kind, observations=len(daily),
                                  net_sharpe=computed['net_sharpe'], net_end_wealth=daily.net_wealth.iloc[-1]))

    output = pd.DataFrame(metrics)
    for (h, dataset), group in output.groupby(['horizon', 'dataset']):
        for etf, suffix in [('market_proxy', 'spy'), ('qqq', 'qqq')]:
            control = group[group.model == etf].iloc[0]
            output.loc[group.index, f'delta_{suffix}'] = group.net_sharpe - control.net_sharpe
    confidence_file = root / 'pooled_confidence_intervals.csv'
    if confidence_file.exists():
        confidence = read(confidence_file)
        ci_columns = [c for c in confidence.columns if '_ci_' in c]
        output = output.merge(confidence[['horizon', 'dataset', 'model'] + ci_columns],
                              on=['horizon', 'dataset', 'model'], validate='one_to_one')
    output.to_csv(root / 'combined_2022_2025_metrics.csv', index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(root / 'combined_2022_2025_curves.csv', index=False)

    def draw(h, dataset, focus=False):
        group = output[(output.horizon == h) & (output.dataset == dataset)]
        if focus:
            group = group[group.model.isin(['think', 'think_mix', 'think_mix_huber', 'market_proxy', 'qqq'])]
        fig, ax = plt.subplots(figsize=(19, 10))
        for _, row in group.iterrows():
            kind = row.model
            p = paths[(h, dataset, kind)]
            prominent = kind in ['think_mix_huber', 'market_proxy', 'qqq']
            # First point is realized wealth, without fabricating an initial return/date.
            ax.plot(pd.to_datetime(p.date), p.net_wealth,
                    label=f'{row.label}  [SR {row.net_sharpe:.3f}]', color=colors[kind],
                    lw=2.8 if prominent else 1.25, alpha=1 if prominent else .82,
                    ls='--' if kind == 'qqq' or 'huber' in kind else '-',
                    zorder=5 if prominent else 2)
        for year in [2023, 2024, 2025]:
            ax.axvline(pd.Timestamp(f'{year}-01-01'), color='#666666', lw=.7, ls=':', alpha=.5)
        ax.set_title(f'{dataset.replace("_recent", " cohort")} | {h}-session forecasts | 2022–2025 combined returns', fontsize=17, pad=18)
        ax.set_ylabel('Net fold-compounded wealth per initial $1', fontsize=12)
        ax.grid(alpha=.18)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, .5), frameon=False, fontsize=10,
                  title='Pooled net Sharpe (3% annual hurdle)', title_fontsize=10)
        ax.margins(x=.01)
        fig.subplots_adjust(left=.065, right=.64, top=.9, bottom=.14)
        fig.text(.065, .065, 'Each model uses its prior-validation-selected portfolio each year. All curves include modeled costs.', fontsize=11)
        fig.text(.065, .037, 'Annual accounts reset; boundary gaps are omitted. These are compounded annual folds, not a continuous live account.', fontsize=11)
        note = ('Focused view; the accompanying all-model chart includes every architecture. ' if focus else 'All tested architectures are shown. ')
        fig.text(.065, .01, note + 'This retrospective comparison does not establish a new winner; public-data survivorship bias remains.', fontsize=10, color='#555555')
        name = f'combined_h{h}_{dataset}' + ('_focus' if focus else '') + '.png'
        fig.savefig(root / 'figures' / name, dpi=160)
        plt.close(fig)
        figures.append(name)
        return name

    image_sections = []
    sections = ['# Combined 2022–2025 model and ETF comparison', '',
                'These charts combine the saved annual backtest returns without changing predictions, portfolios or costs. '
                'All original pooled gross/net performance metrics were recomputed and verified against the previous report. '
                'QQQ now also has a complete standalone pooled performance row.', '',
                '**Interpretation:** annual accounts reset to cash, sizing restarts at $1m and boundary sessions are omitted. '
                'Compounding their returns creates a synthetic research curve, not an uninterrupted four-year live account. '
                'Each horizon uses its own exact matched-date ETF calendar. Differences between daily and five-session ETF '
                'Sharpes reflect those calendars. Sharpe uses 252 sessions and a fixed 3% annual hurdle.', '',
                'Every model keeps its separately validation-selected portfolio for each year. The highlighted hybrid is '
                'an exploratory candidate identified after seeing results, not the validation-selected architecture procedure. '
                'Its NASDAQ five-session Sharpe remains **0.675**, versus **SPY 0.621** and **QQQ 0.626**. '
                'The existing confidence intervals include zero advantage.', '',
                '[All metrics CSV](combined_2022_2025_metrics.csv) | [Underlying dated returns and curves](combined_2022_2025_curves.csv)', '']
    for h, dataset in groups:
        name = draw(h, dataset)
        heading = f'{dataset.replace("_recent", " cohort")}, {h}-session forecasts'
        image_sections += [f'### {heading}', '', f'![{heading}: combined net wealth and Sharpe](figures/{name})', '']
        group = output[(output.horizon == h) & (output.dataset == dataset)]
        sections += [f'## {heading}', '', f'![{heading}](figures/{name})', '', '### Investment metrics', '',
                     table(group, [('label','Model / control'), ('gross_sharpe','Gross SR'), ('net_sharpe','Net SR'),
                                   ('net_annualized_return','Ann. net return'), ('net_cumulative_return','Cum. net return'),
                                   ('net_annualized_volatility','Ann. volatility'), ('net_maximum_drawdown','Max drawdown'),
                                   ('turnover_annual','Annual turnover'), ('net_profitable_days','Profitable days'),
                                   ('profitable_periods','Profitable evaluation periods')],
                           percentage=['net_annualized_return','net_cumulative_return','net_annualized_volatility',
                                       'net_maximum_drawdown','net_profitable_days','profitable_periods']), '',
                     '### Ranking metrics', '',
                     table(group, [('label','Model / control'), ('IC','IC'), ('RankIC','RankIC'), ('ICIR','ICIR'),
                                   ('RankICIR','RankICIR'), ('NDCG@5','NDCG@5'), ('NDCG@10','NDCG@10'), ('NDCG@20','NDCG@20')]), '']
        if confidence_file.exists():
            sections += ['### Sharpe confidence intervals', '',
                         'Exploratory 95% paired block-bootstrap intervals. Difference intervals are for model Sharpe minus ETF Sharpe, '
                         'not for the model Sharpe itself. [Method and full uncertainty tables](CONFIDENCE_INTERVALS.md).', '',
                         '| Model / control | Net SR | 95% SR CI | ΔSR vs SPY | 95% ΔSR CI | ΔSR vs QQQ | 95% ΔSR CI |',
                         '|---|---:|---|---:|---|---:|---|']
            for _, r in group.iterrows():
                sections.append(f'| {r.label} | {r.net_sharpe:.3f} | [{r.net_sharpe_ci_low:.3f}, {r.net_sharpe_ci_high:.3f}] | '
                                f'{r.delta_spy:.3f} | [{r.delta_spy_ci_low:.3f}, {r.delta_spy_ci_high:.3f}] | '
                                f'{r.delta_qqq:.3f} | [{r.delta_qqq_ci_low:.3f}, {r.delta_qqq_ci_high:.3f}] |')
            sections.append('')
    focus = draw(5, 'NASDAQ_recent', True)
    sections += ['## Focus: THINK, the mixer hybrid and ETF controls', '', f'![Hybrid comparison](figures/{focus})', '',
                 'Ranking metrics pool prediction periods across the eligible stock universe, not only portfolio holdings. '
                 'ICIR/RankICIR are unannualized mean-to-standard-deviation ratios. Turnover and profitable-rebalance-period '
                 'rates retain the original trading-day-weighted annual aggregation. ETF ranking metrics are undefined, '
                 'shown as missing. ETF profitable evaluation periods use the same horizon-sized return blocks as model '
                 'holding periods, without extra ETF rebalancing. Profitable trading days are reported for all curves. '
                 'All gross metrics, zero-hurdle Sharpes and existing exploratory uncertainty intervals remain in the CSV.', '']
    (root / 'COMBINED_2022_2025.md').write_text('\n'.join(sections), encoding='utf-8')
    intro = ('The **0.675** NASDAQ THINK+mixer+Huber result is the blue dashed curve in the five-session NASDAQ chart below. '
             'Each legend shows pooled net Sharpe. [Complete pooled metric tables](COMBINED_2022_2025.md).\n\n'
             'Annual cash resets and boundary gaps remain; these are synthetic fold-compounded curves. '
             'Models and portfolio choices are unchanged.\n\n')
    insert_section(root / 'CHARTS.md', '## Combined 2022–2025 curves: every model and both ETFs', intro + '\n'.join(image_sections))
    insert_section(root / 'REPORT.md', '## Combined curves and complete pooled ETF tables',
                   '[Open the combined 2022–2025 curves and tables](COMBINED_2022_2025.md). '
                   'All tested models appear alongside SPY and QQQ, with pooled Sharpe in each legend. '
                   'The NASDAQ five-session hybrid remains 0.675. Existing experiment results and selections are unchanged.')
    assert len(output) == 76 and len(inventory) == 76
    assert round(output.query("horizon == 5 and dataset == 'NASDAQ_recent' and model == 'think_mix_huber'").net_sharpe.iloc[0], 3) == .675
    (root / 'combined_curve_audit.json').write_text(json.dumps(dict(
        model_control_series=len(inventory), figures=figures, curves=inventory, inputs=input_hashes,
        source_sha256=digest(__file__), existing_pooled_metrics_verified=True,
        selection_changed=False, accounting='Annual-reset fold compounding; untested boundary days omitted'), indent=2))
    print(f'Saved {len(figures)} figures, {len(output)} pooled model/control rows; every original pooled performance metric verified.')


if __name__ == '__main__':
    main()
