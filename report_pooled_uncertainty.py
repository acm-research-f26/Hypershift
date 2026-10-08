"""Recompute matched pooled Sharpe intervals; never tune or select models."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_engine import performance

ROOT = Path('runs/robust_comparison')
REPEATS, BLOCK, SEED = 1000, 20, 20261005


def insert_section(path, heading, body):
    text = path.read_text(encoding='utf-8')
    # Use a distinct marker so combined-curve reporting and CI reporting coexist.
    start, end = '<!-- uncertainty:start -->', '<!-- uncertainty:end -->'
    for a, b in [(start, end), ('<!-- combined-curves:start -->', '<!-- combined-curves:end -->')]:
        if a in text:
            before, tail = text.split(a, 1)
            section, after = tail.split(b, 1)
            if a == start or '## Verified pooled confidence intervals' in section:
                text = before + after.lstrip('\n')
    first, rest = text.split('\n', 1)
    path.write_text(first+'\n\n'+start+'\n'+heading+'\n\n'+body+'\n\n'+end+'\n\n'+rest.lstrip('\n'), encoding='utf-8')


def indices(lengths):
    """Match the original repeat-then-year RNG ordering exactly."""
    rng = np.random.default_rng(SEED)
    result = []
    offsets = np.cumsum([0] + lengths[:-1])
    for _ in range(REPEATS):
        row = []
        for n, offset in zip(lengths, offsets):
            starts = rng.integers(0, n, int(np.ceil(n / BLOCK)))
            row.extend((((starts[:, None] + np.arange(BLOCK)) % n).ravel()[:n] + offset).tolist())
        result.append(row)
    return np.asarray(result)


def bootstrap_sharpe(returns, ids):
    a = np.asarray(returns)[ids]
    return (a.mean(1) - ((1.03)**(1/252)-1)) * np.sqrt(252) / a.std(1, ddof=1)


def interval(values):
    return np.quantile(values, [.025, .975])


def fmt_ci(low, high):
    return f'[{low:.3f}, {high:.3f}]'


def main():
    curves = pd.read_csv(ROOT / 'combined_2022_2025_curves.csv')
    original = pd.read_csv(ROOT / 'pooled_model_vs_spy_qqq.csv')
    records, boot, actual = [], {}, {}
    for (h, d), group in curves.groupby(['horizon', 'dataset']):
        controls = group[group.model == 'market_proxy']
        lengths = controls.groupby('fold_year', sort=True).size().tolist()
        ids = indices(lengths)
        for kind, path in group.groupby('model', sort=False):
            np.testing.assert_array_equal(path.date, controls.date)
            boot[(h, d, kind)] = bootstrap_sharpe(path.net, ids)
            actual[(h, d, kind)] = performance(path.net)['sharpe']
        for kind in group.model.unique():
            values = boot[(h, d, kind)]
            low, high = interval(values)
            row = dict(horizon=h, dataset=d, model=kind,
                       net_sharpe=actual[(h, d, kind)], net_sharpe_ci_low=low, net_sharpe_ci_high=high)
            for name, control in [('spy', 'market_proxy'), ('qqq', 'qqq')]:
                a, b = interval(values - boot[(h, d, control)])
                row.update({f'delta_{name}':actual[(h, d, kind)] - actual[(h, d, control)],
                            f'delta_{name}_ci_low':a, f'delta_{name}_ci_high':b})
            old = original[(original.horizon == h) & (original.dataset == d) & (original.model == kind)]
            if len(old):
                old = old.iloc[0]
                for new_key, old_key in [('delta_spy_ci_low', 'low'), ('delta_spy_ci_high', 'high'),
                                         ('delta_qqq_ci_low', 'qqq_delta_low'), ('delta_qqq_ci_high', 'qqq_delta_high')]:
                    np.testing.assert_allclose(row[new_key], old[old_key], rtol=1e-9, atol=1e-10,
                                               err_msg=f'{h}/{d}/{kind}/{new_key}')
            records.append(row)
    result = pd.DataFrame(records)
    result.to_csv(ROOT / 'pooled_confidence_intervals.csv', index=False)
    # Selected architecture procedures, including joint horizon choices.
    selected = pd.read_csv(ROOT / 'validation_selected_by_year.csv')
    joint = pd.read_csv(ROOT / 'combined_validation_selected.csv')
    old_procedures = pd.read_csv(ROOT / 'pooled_selection_procedures.csv')
    old_etf = pd.read_csv(ROOT / 'selection_procedures_vs_spy_qqq.csv')
    procedure_records = []
    for name, frame in [('Five-session', selected[selected.horizon == 5]),
                        ('Daily', selected[selected.horizon == 1]), ('Joint validation choice', joint)]:
        for d, rows in frame.groupby('dataset'):
            net, spy, qqq, lengths = [], [], [], []
            for _, row in rows.sort_values('year').iterrows():
                window = curves[(curves.horizon == row.horizon) & (curves.dataset == d) & (curves.fold_year == row.year)]
                arrays = [window[window.model == m].net.to_numpy() for m in [row.model, 'market_proxy', 'qqq']]
                assert len(set(map(len, arrays))) == 1
                net.extend(arrays[0]); spy.extend(arrays[1]); qqq.extend(arrays[2]); lengths.append(len(arrays[0]))
            ids = indices(lengths)
            samples = bootstrap_sharpe(net, ids)
            low, high = interval(samples)
            record = dict(procedure=name, dataset=d, net_sharpe=performance(net)['sharpe'],
                          net_sharpe_ci_low=low, net_sharpe_ci_high=high)
            for symbol, control in [('spy', spy), ('qqq', qqq)]:
                low, high = interval(samples - bootstrap_sharpe(control, ids))
                record.update({f'delta_{symbol}':performance(net)['sharpe']-performance(control)['sharpe'],
                               f'delta_{symbol}_ci_low':low, f'delta_{symbol}_ci_high':high})
            old = old_procedures[(old_procedures.procedure == name) & (old_procedures.dataset == d)].iloc[0]
            qq = old_etf[(old_etf.procedure == name) & (old_etf.dataset == d)].iloc[0]
            np.testing.assert_allclose([record['delta_spy_ci_low'], record['delta_spy_ci_high']], [old.low, old.high], atol=1e-10)
            np.testing.assert_allclose([record['delta_qqq_ci_low'], record['delta_qqq_ci_high']], [qq.qqq_delta_low, qq.qqq_delta_high], atol=1e-10)
            procedure_records.append(record)
    procedures = pd.DataFrame(procedure_records)
    procedures.to_csv(ROOT / 'procedure_confidence_intervals.csv', index=False)
    # Paired Huber-minus-MSE comparison distinguishes loss ablation from policy changes.
    ablations = []
    for h, d in boot_keys(result):
        for policy in ['validation-selected per model', 'fixed equal K10']:
            if policy.startswith('validation'):
                a, b = boot[(h,d,'think_mix_huber')], boot[(h,d,'think_mix')]
                sr_a, sr_b = actual[(h,d,'think_mix_huber')], actual[(h,d,'think_mix')]
            else:
                root = Path('runs/robust_2022_2025_final' if h == 5 else 'runs/daily_2022_2025_final')
                paths = {}
                for kind in ['think_mix', 'think_mix_huber']:
                    pieces = [pd.read_csv(root / f'{d}_{y}' / f'{kind}_equal_k10_daily.csv') for y in range(2022,2026)]
                    paths[kind] = pd.concat(pieces, ignore_index=True)
                np.testing.assert_array_equal(paths['think_mix'].date, paths['think_mix_huber'].date)
                ids = indices([len(x) for x in pieces])
                a, b = [bootstrap_sharpe(paths[k].net, ids) for k in ['think_mix_huber', 'think_mix']]
                sr_a, sr_b = [performance(paths[k].net)['sharpe'] for k in ['think_mix_huber', 'think_mix']]
            low, high = interval(a-b)
            ablations.append(dict(horizon=h,dataset=d,policy=policy,mse_net_sharpe=sr_b,huber_net_sharpe=sr_a,
                                  delta_huber_minus_mse=sr_a-sr_b,delta_ci_low=low,delta_ci_high=high))
    ablations = pd.DataFrame(ablations)
    ablations.to_csv(ROOT / 'huber_mse_pooled_intervals.csv', index=False)
    lines = ['# Pooled Sharpe confidence intervals', '',
             '**These are exploratory 95% percentile block-bootstrap intervals.** Each of 1,000 replicates '
             'resamples contiguous 20-session circular blocks separately within each annual test fold. '
             'Model and ETF returns use identical sampled dates. The random seed is 20261005, matching the '
             'predeclared analysis. Every previously published pooled model-versus-ETF interval was independently reproduced.', '',
             '- **Sharpe CI** describes uncertainty in a model’s own pooled net Sharpe.',
             '- **ΔSR vs SPY/QQQ CI** describes the paired difference between the two Sharpes. '
             'It is not obtained by subtracting endpoints of separate Sharpe intervals.',
             '- An interval containing zero does not establish an advantage. These intervals condition on '
             'the four observed years, fixed trained models and selected policies. They do not rerun training/selection '
             'or correct for repeated testing, multiple comparisons, survivorship bias or future regime uncertainty.',
             '- Annual folds reset sizing and omit boundary sessions. All comparisons use identical dates within '
             'each horizon. Returns include modeled costs and Sharpe uses a 3% annual hurdle.', '',
             '## Validation-selected architecture procedures', '',
             '| Procedure | Cohort | Net SR | 95% SR CI | ΔSR vs SPY | 95% ΔSR CI | ΔSR vs QQQ | 95% ΔSR CI |',
             '|---|---|---:|---|---:|---|---:|---|']
    for _, r in procedures.iterrows():
        lines.append(ci_row(r, f'{r.procedure} | {r.dataset}'))
    for h,d in boot_keys(result):
        lines += ['', f'## {d}: {h}-session models and controls', '',
                  '| Model | Net SR | 95% SR CI | ΔSR vs SPY | 95% ΔSR CI | ΔSR vs QQQ | 95% ΔSR CI |',
                  '|---|---:|---|---:|---|---:|---|']
        for _, r in result[(result.horizon==h)&(result.dataset==d)].iterrows():
            lines.append(ci_row(r, 'SPY' if r.model=='market_proxy' else r.model))
    lines += ['', '## Huber versus MSE: paired differences', '',
              'The fixed-equal-top-10 comparison holds portfolio construction constant. The validation-selected '
              'comparison permits each model’s previously selected portfolio and K to differ.', '',
              '| Horizon | Cohort | Portfolio | MSE SR | Huber SR | Huber minus MSE | 95% difference CI |',
              '|---|---|---|---:|---:|---:|---|']
    for _, r in ablations.iterrows():
        lines.append(f'| {r.horizon} | {r.dataset} | {r.policy} | {r.mse_net_sharpe:.3f} | '
                     f'{r.huber_net_sharpe:.3f} | {r.delta_huber_minus_mse:.3f} | {fmt_ci(r.delta_ci_low,r.delta_ci_high)} |')
    lines += ['', '[All model intervals](pooled_confidence_intervals.csv) | '
              '[Procedure intervals](procedure_confidence_intervals.csv) | '
              '[Huber/MSE intervals](huber_mse_pooled_intervals.csv)', '']
    (ROOT/'CONFIDENCE_INTERVALS.md').write_text('\n'.join(lines), encoding='utf-8')
    for name in ['REPORT.md','ETF_CONTROLS.md','FINDINGS.md']:
        insert_section(ROOT/name, '## Verified pooled confidence intervals',
                       '[Updated uncertainty tables](CONFIDENCE_INTERVALS.md) show 95% intervals for net Sharpe '
                       'and paired Sharpe differences versus both ETFs, plus Huber-versus-MSE ablations. '
                       'All previously reported pooled benchmark-difference intervals were reproduced. '
                       'Intervals remain exploratory and unadjusted for model selection or multiple testing.')
    (ROOT/'confidence_interval_audit.json').write_text(json.dumps(dict(
        status='verified', series=len(result), procedures=len(procedures), huber_comparisons=len(ablations),
        replicates=REPEATS,block_sessions=BLOCK,seed=SEED,confidence_level=.95,
        all_existing_paired_intervals_reproduced=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(result.query("horizon==5 and model=='think_mix_huber'").to_string(index=False))
    print(ablations.to_string(index=False))


def boot_keys(frame):
    return [(h,d) for h in [5,1] for d in ['NASDAQ_recent','NYSE_recent']]


def ci_row(r, prefix):
    return (f'| {prefix} | {r.net_sharpe:.3f} | {fmt_ci(r.net_sharpe_ci_low,r.net_sharpe_ci_high)} | '
            f'{r.delta_spy:.3f} | {fmt_ci(r.delta_spy_ci_low,r.delta_spy_ci_high)} | '
            f'{r.delta_qqq:.3f} | {fmt_ci(r.delta_qqq_ci_low,r.delta_qqq_ci_high)} |')


if __name__=='__main__':
    main()
