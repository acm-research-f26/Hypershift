"""Reporting-only additions after the frozen experiment batch completed.

These descriptive summaries do not change model/policy selection or saved results.
"""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd


def main():
    root = Path('runs/robust_comparison')
    all_rows = pd.read_csv(root / 'all_metrics.csv')
    equal = all_rows[(all_rows.seed == 'ensemble') & (all_rows.method == 'equal')
                     & (all_rows.cost_multiplier == 1)]
    pairs = equal.pivot(index=['horizon', 'dataset', 'year', 'model'],
                        columns='k', values='net_sharpe').dropna()
    effect = pd.read_csv(root / 'fixed_k10_portfolio_effects.csv')
    marker = '## Completed portfolio and candidate interpretation'
    text = [marker, '',
            'These additional aggregate descriptions were written after the tests. They are not new selection rules. '
            'The comparisons are correlated and should not be treated as independent statistical trials.', '',
            '### Does increasing K help?', '',
            '| Equal-weight comparison | Positive windows | Total windows | Median net Sharpe change |',
            '|---|---:|---:|---:|']
    for k in [10, 20]:
        delta = pairs[k] - pairs[5]
        text.append(f'| K={k} minus K=5 | {(delta > 0).sum()} | {len(delta)} | {delta.median():+.3f} |')
    text += ['', 'This pools the declared ranking models across both horizons and cohorts. '
             'Top-10 alone did not solve inconsistency. Top-20 produced a modest descriptive improvement, '
             'but it cannot be selected using these test results.', '', '### Do portfolio changes help?', '',
             '| Horizon | K=10 policy versus equal weights | Positive windows | Windows | Median Sharpe change | Median annual turnover change |',
             '|---|---|---:|---:|---:|---:|']
    for (h, method), group in effect.groupby(['horizon', 'method']):
        text.append(f'| {h} sessions | {method} | {(group.delta_sharpe > 0).sum()} | {len(group)} | '
                    f'{group.delta_sharpe.median():+.3f} | {group.delta_turnover.median():+.2f} |')
    text += ['', 'The equal-weight retention buffer reduced turnover and helped daily net Sharpe most clearly. '
             'That improvement did not make the daily validation-selected procedure beat either ETF. '
             'Inverse-volatility weights alone did not provide a consistent gain.', '',
             '### Promising hybrid and its limits', '',
             'The five-session NASDAQ THINK+mixer+Huber variant achieved pooled net Sharpe **0.675**, '
             'versus **0.621 SPY** and **0.626 QQQ**, using each year’s validation-selected portfolio. '
             'Its net annualized return was **19.5%**, cumulative return **99.5%**, annualized volatility **27.8%**, '
             'and maximum drawdown **30.0%**. Mean IC was **0.0118**, RankIC **0.0110**, and NDCG@10 **0.5475**. '
             'The same model’s NYSE pooled Sharpe was only **0.194**. It beat both ETFs in only **one of four NASDAQ years**.', '',
             'The exploratory 95% paired block-bootstrap interval for the NASDAQ Sharpe advantage is '
             '**[-0.396, 0.524] versus SPY** and **[-0.445, 0.596] versus QQQ**. Both include zero. '
             'The model was identified after inspecting this batch, and its architecture was not the validation-selected '
             'choice each year. Its result supports further study, not an established economic or statistical edge. '
             'Public-data survivorship bias, repeated historical evaluation and unadjusted multiple comparisons remain material.', '',
             'Four-year statistics concatenate annual accounts and omit boundary sessions. ETF values are matched-window '
             'controls, not published full-calendar-year fund returns. See [all gross/net and ranking metrics](pooled_model_vs_spy_qqq.csv).', '']
    section = '\n'.join(text)
    for name in ['REPORT.md', 'FINDINGS.md']:
        path = root / name
        original = path.read_text(encoding='utf-8').split(marker)[0].rstrip()
        path.write_text(original + '\n\n' + section, encoding='utf-8')
    charts = root / 'CHARTS.md'
    content = charts.read_text(encoding='utf-8').replace('All 22 model curves:', 'All 23 model/control curves:')
    content = content.replace('All 14 model curves:', 'All 15 model/control curves:')
    charts.write_text(content, encoding='utf-8')
    snapshot = root / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    names = ['plot_robust_diagnostics.py', 'summarize_robust_findings.py', 'add_etf_controls.py',
             'audit_robust_regimes.py', 'verify_robust_outputs.py', 'finalize_robust_reporting.py']
    hashes = {}
    for name in names:
        target = snapshot / name
        if target.exists():
            assert target.read_bytes() == Path(name).read_bytes(), name
        else:
            shutil.copyfile(name, target)
        hashes[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    (root / 'FINAL_REPORTING_PROVENANCE.json').write_text(json.dumps(dict(
        reporting_only=True, training_and_selection_unchanged=True,
        supplementary_interpretation='post-test descriptive aggregates; no new winners selected',
        sources=hashes), indent=2))
    print('Final report, K/portfolio interpretation and reporting source snapshots saved.')


if __name__ == '__main__':
    main()
