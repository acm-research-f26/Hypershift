"""Post-process saved comparison tables; never train, select or alter returns.

Run after the ordinary report generators. NYSE uses the saved equal_universe
control; Nasdaq continues to use QQQ. Archived curves/figures are untouched.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from research_engine import performance, load_panel, prepare_panel, fold_indices
from report_pooled_uncertainty import indices, bootstrap_sharpe, interval

LABEL = 'NYSE basket — passive benchmark'
STAGED = {}
AUDIT = {'inputs': {}, 'updated_files': [], 'matched_paths': 0,
         'training_changed': False, 'selection_changed': False,
         'returns_changed': False, 'figures_changed': False,
         'bootstrap': {'replicates': 1000, 'block_sessions': 20, 'seed': 20261005}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    AUDIT['inputs'][str(path)] = digest(path)
    return pd.read_csv(path)


def stage_csv(path, frame):
    STAGED[path] = frame.to_csv(index=False).encode('utf-8')


class Study:
    def __init__(self, root):
        self.root = root
        self.large = 'largecap' in str(root)
        self.paths = {}
        self.cache = {}
        self.procedures = {}
        self.frames = {}
        self.annual = read(root / 'annual_comparison.csv')
        base = read(root / 'combined_2022_2025_curves.csv')
        for (h, d, m), p in base.groupby(['horizon', 'dataset', 'model'], sort=False):
            self.paths[(int(h), d, 'validation-selected', m)] = p.reset_index(drop=True)
        ranking = root / 'huber_ranking_curves.csv'
        if ranking.exists():
            for (d, policy, m), p in read(ranking).groupby(['dataset', 'policy', 'model'], sort=False):
                key = (5, d, policy, m)
                if key in self.paths:
                    np.testing.assert_array_equal(self.paths[key].date, p.date)
                    np.testing.assert_allclose(self.paths[key].net, p.net, atol=1e-12)
                self.paths[key] = p.reset_index(drop=True)
        self.datasets = list(base.dataset.unique())
        self.lock = json.loads((root / 'COMBINED_SELECTION_LOCK.json').read_text())
        AUDIT['inputs'][str(root / 'COMBINED_SELECTION_LOCK.json')] = digest(root / 'COMBINED_SELECTION_LOCK.json')

    def path(self, h, d, policy, m):
        if m == 'equal_universe' and (h, d, policy, m) not in self.paths:
            return self.paths[(h, d, 'validation-selected', m)]
        return self.paths[(h, d, policy, m)]

    def stats(self, h, d, policy, m, year=None):
        key = (h, d, policy, m, year)
        if key in self.cache:
            return self.cache[key]
        p = self.path(h, d, policy, m)
        bench = self.path(h, d, policy, 'equal_universe' if d.startswith('NYSE') else 'qqq')
        spy = self.path(h, d, policy, 'market_proxy')
        if year is not None:
            p, bench, spy = [x[x.fold_year == year].reset_index(drop=True) for x in [p, bench, spy]]
        np.testing.assert_array_equal(p.date, bench.date)
        np.testing.assert_array_equal(p.fold_year, bench.fold_year)
        np.testing.assert_array_equal(p.date, spy.date)
        assert p.date.is_unique and p.date.is_monotonic_increasing
        AUDIT['matched_paths'] += 1
        out = {f'{side}_{k}': v for side in ['net', 'gross'] for k, v in performance(p[side]).items()}
        out['benchmark_model'] = 'equal_universe' if d.startswith('NYSE') else 'qqq'
        out['benchmark_label'] = LABEL if d.startswith('NYSE') else 'QQQ'
        out['benchmark_sharpe'] = performance(bench.net)['sharpe']
        out['delta_benchmark'] = out['net_sharpe'] - out['benchmark_sharpe']
        out['beats_benchmark'] = out['delta_benchmark'] > 0
        out['beats_both'] = out['beats_benchmark'] and out['net_sharpe'] > performance(spy.net)['sharpe']
        out['start'], out['end'] = p.date.iloc[0], p.date.iloc[-1]
        if year is None:
            ids = indices(p.groupby('fold_year', sort=True).size().tolist())
            samples = bootstrap_sharpe(p.net, ids)
            low, high = interval(samples)
            out.update(net_sharpe_ci_low=low, net_sharpe_ci_high=high)
            for control, prefix in [(bench, 'benchmark'), (spy, 'spy')]:
                low, high = interval(samples - bootstrap_sharpe(control.net, ids))
                out[f'delta_{prefix}_ci_low'], out[f'delta_{prefix}_ci_high'] = low, high
                out[f'delta_{prefix}'] = out['net_sharpe'] - performance(control.net)['sharpe']
            out['benchmark_delta_low'] = out['delta_benchmark_ci_low']
            out['benchmark_delta_high'] = out['delta_benchmark_ci_high']
            huber_key = (h, d, policy, 'think_mix_huber')
            if huber_key in self.paths:
                huber = self.path(*huber_key)
                np.testing.assert_array_equal(p.date, huber.date)
                low, high = interval(samples - bootstrap_sharpe(huber.net, ids))
                out.update(delta_huber=out['net_sharpe'] - performance(huber.net)['sharpe'],
                           delta_huber_ci_low=low, delta_huber_ci_high=high)
            annual_sr = [performance(g.net)['sharpe'] for _, g in p.groupby('fold_year', sort=True)]
            bs = [performance(g.net)['sharpe'] for _, g in bench.groupby('fold_year', sort=True)]
            ss = [performance(g.net)['sharpe'] for _, g in spy.groupby('fold_year', sort=True)]
            out['years_beating_benchmark'] = sum(a > b for a, b in zip(annual_sr, bs))
            out['years_beating_both'] = sum(a > b and a > s for a, b, s in zip(annual_sr, bs, ss))
            out['years_beating_spy'] = sum(a > s for a, s in zip(annual_sr, ss))
            if huber_key in self.paths:
                hs = [performance(g.net)['sharpe'] for _, g in huber.groupby('fold_year', sort=True)]
                out['years_beating_huber'] = sum(a > b for a, b in zip(annual_sr, hs))
            out.update(median_yearly_sharpe=float(np.median(annual_sr)), worst_yearly_sharpe=min(annual_sr),
                       std_yearly_sharpe=float(np.std(annual_sr, ddof=1)))
        a = self.annual[(self.annual.horizon == h) & (self.annual.dataset == d) & (self.annual.model == m)]
        if year is not None:
            a = a[a.year == year]
        if len(a):
            for field in ['turnover_annual', 'profitable_periods']:
                out[field] = float(np.average(a[field], weights=a.net_days))
        self.cache[key] = out
        return out

    def proc(self, name, d):
        key = (name, d)
        if key in self.procedures:
            return self.procedures[key]
        chosen = read(self.root / ('combined_validation_selected.csv' if name == 'Joint validation choice'
                                  else 'validation_selected_by_year.csv'))
        chosen = chosen[chosen.dataset == d]
        if name != 'Joint validation choice':
            chosen = chosen[chosen.horizon == (5 if name == 'Five-session' else 1)]
        ns, bs, lengths, years_both, years_bench = [], [], [], 0, 0
        for row in chosen.sort_values('year').itertuples():
            p = self.path(int(row.horizon), d, 'validation-selected', row.model)
            b = self.path(int(row.horizon), d, 'validation-selected', 'equal_universe' if d.startswith('NYSE') else 'qqq')
            s = self.path(int(row.horizon), d, 'validation-selected', 'market_proxy')
            p, b, s = [x[x.fold_year == row.year] for x in [p, b, s]]
            np.testing.assert_array_equal(p.date, b.date)
            np.testing.assert_array_equal(p.date, s.date)
            ns.extend(p.net); bs.extend(b.net); lengths.append(len(p))
            p_sr, b_sr, s_sr = [performance(x.net)['sharpe'] for x in [p, b, s]]
            years_bench += p_sr > b_sr
            years_both += p_sr > b_sr and p_sr > s_sr
        ids = indices(lengths)
        low, high = interval(bootstrap_sharpe(ns, ids) - bootstrap_sharpe(bs, ids))
        out = dict(net_sharpe=performance(ns)['sharpe'], benchmark_sharpe=performance(bs)['sharpe'],
                   delta_benchmark=performance(ns)['sharpe'] - performance(bs)['sharpe'],
                   delta_benchmark_ci_low=low, delta_benchmark_ci_high=high,
                   years_beating_both=int(years_both), years_beating_benchmark=int(years_bench),
                   benchmark_model='equal_universe' if d.startswith('NYSE') else 'qqq',
                   benchmark_label=LABEL if d.startswith('NYSE') else 'QQQ')
        out['benchmark_delta_low'], out['benchmark_delta_high'] = low, high
        self.procedures[key] = out
        return out

    def update_frame(self, filename):
        path = self.root / filename
        frame = read(path)
        original = frame.copy(deep=True)
        if filename == 'regime_diagnostics.csv':
            return self.regimes(frame)
        aliases = {}
        for c in frame.columns:
            if 'qqq' in c:
                aliases[c] = c.replace('qqq', 'benchmark')
        frame = frame.rename(columns={a: b for a, b in aliases.items() if a in frame.columns})
        # Fill explicit benchmark identity even in tables containing both markets.
        frame['benchmark_model'] = np.where(frame.dataset.str.startswith('NYSE'), 'equal_universe', 'qqq')
        frame['benchmark_label'] = np.where(frame.dataset.str.startswith('NYSE'), LABEL, 'QQQ')
        drop = []
        for i, r in frame.iterrows():
            d = r.dataset
            if not d.startswith('NYSE'):
                # Populate newly introduced identity/summary columns without
                # replacing any previously published Nasdaq comparison value.
                h = int(r.get('horizon', 5))
                policy = r.get('policy', 'validation-selected')
                if pd.isna(policy):
                    policy = 'validation-selected'
                if 'procedure' in frame:
                    extra = self.proc(r.procedure, d)
                elif 'model' in frame:
                    extra = self.stats(h, d, policy, r.model, int(r.year) if 'year' in frame else None)
                else:
                    extra = {}
                for field, value in extra.items():
                    if ((field.startswith('benchmark_') or field.startswith('delta_benchmark') or
                         field.startswith('years_beating_benchmark')) and pd.isna(r.get(field, np.nan))):
                        frame.at[i, field] = value
                continue
            h = int(r.get('horizon', 5))
            policy = r.get('policy', 'validation-selected')
            if pd.isna(policy):
                policy = 'validation-selected'
            replacement = False
            if r.get('model') == 'qqq' or r.get('symbol') == 'QQQ':
                replacement = True
                if 'model' in frame:
                    dup = (frame.dataset == d) & (frame.model == 'equal_universe')
                    if 'horizon' in frame:
                        dup &= frame.horizon == h
                    if 'policy' in frame:
                        dup &= frame.policy.fillna('validation-selected') == policy
                    if 'year' in frame:
                        dup &= frame.year == r.year
                    if dup.any():
                        drop.append(i)
                        continue
                    frame.at[i, 'model'] = 'equal_universe'
                    if 'label' in frame:
                        frame.at[i, 'label'] = LABEL
                else:
                    frame.at[i, 'symbol'] = 'NYSE_BASKET'
                if 'year' in frame:
                    control = self.annual[(self.annual.horizon == h) & (self.annual.dataset == d) &
                                          (self.annual.year == r.year) & (self.annual.model == 'equal_universe')]
                    assert len(control) == 1
                    for field, value in control.iloc[0].items():
                        if field in frame and field not in ['model', 'label', 'policy']:
                            frame.at[i, field] = value
            if 'procedure' in frame:
                computed = self.proc(r.procedure, d)
            elif 'year' in frame:
                m = 'equal_universe' if replacement or 'model' not in frame else r.model
                computed = self.stats(h, d, policy, m, int(r.year))
            else:
                m = 'equal_universe' if replacement else r.model
                computed = self.stats(h, d, policy, m)
            # Keep every existing model/ETF statistic unchanged; update comparisons only.
            for field, value in computed.items():
                if field.startswith('benchmark_') or field.startswith('delta_benchmark') or field in [
                        'beats_benchmark', 'beats_both', 'years_beating_both', 'years_beating_benchmark']:
                    frame.at[i, field] = value
                elif replacement and field in frame.columns:
                    frame.at[i, field] = value
            if r.get('model') == 'equal_universe' and 'label' in frame:
                frame.at[i, 'label'] = LABEL
        frame = frame.drop(index=drop).reset_index(drop=True)
        # Nasdaq numeric values must remain exactly as read (only column names change).
        old_n = original[original.dataset.str.startswith('NASDAQ')].reset_index(drop=True)
        new_n = frame[frame.dataset.str.startswith('NASDAQ')].reset_index(drop=True)
        for c in old_n.columns:
            mask = old_n[c].notna() if 'benchmark' in c else pd.Series(True, index=old_n.index)
            pd.testing.assert_series_equal(old_n.loc[mask, c], new_n.loc[mask, aliases.get(c, c)], check_names=False)
        self.frames[filename] = frame
        return frame

    def regimes(self, frame):
        frame = frame.rename(columns={'qqq_conditional_sharpe': 'benchmark_conditional_sharpe'})
        frame['benchmark_model'] = np.where(frame.dataset.str.startswith('NYSE'), 'equal_universe', 'qqq')
        frame['benchmark_label'] = np.where(frame.dataset.str.startswith('NYSE'), LABEL, 'QQQ')
        for (h, d, y), group in frame[frame.dataset.str.startswith('NYSE')].groupby(['horizon', 'dataset', 'year']):
            data = prepare_panel(load_panel(d), horizon=int(h))
            p = data['panel']
            b = pd.Series(p.benchmark, index=p.dates)
            risk = b.pct_change(fill_method=None).rolling(20).std().to_numpy()
            trend = (b / b.shift(126) - 1).to_numpy()
            train, _, test = fold_indices(data, int(y))
            threshold = np.nanmedian(risk[data['origin'][train]])
            tags = np.array([('rising' if trend[t] >= 0 else 'falling') + '_' +
                             ('high_vol' if risk[t] >= threshold else 'low_vol') for t in data['origin'][test]])
            bench = self.path(int(h), d, 'validation-selected', 'equal_universe')
            bench = bench[bench.fold_year == y]
            for tag, subset in group.groupby('regime'):
                mask = np.repeat(tags == tag, int(h))
                assert len(mask) == len(bench) and np.all(subset.sessions == mask.sum())
                frame.loc[subset.index, 'benchmark_conditional_sharpe'] = performance(bench.net.to_numpy()[mask])['sharpe']
        self.frames['regime_diagnostics.csv'] = frame
        return frame


def cells(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def line(c):
    return '| ' + ' | '.join(c) + ' |'


def context(stack, study, row, heads):
    text = ' '.join(stack.values())
    d = next((c for c in row if c in study.datasets), None)
    if d is None:
        market = None
        # Nearest heading containing a cohort takes priority over higher headings.
        for title in reversed(list(stack.values())):
            if 'NYSE' in title:
                market = 'NYSE'; break
            if 'NASDAQ' in title or 'Nasdaq' in title:
                market = 'NASDAQ'; break
        d = next((d for d in study.datasets if market and d.startswith(market)), None)
    h = 5
    for title in reversed(list(stack.values())):
        match = re.search(r'(1|5)[- ]session', title)
        if match:
            h = int(match[1]); break
        if 'Daily' in title or 'daily forecasts' in title:
            h = 1; break
    if 'Horizon' in heads:
        match = re.match(r'\d+', row[heads.index('Horizon')])
        if match:
            h = int(match[0])
    policy = 'equal K10' if 'equal K10' in text else 'validation-selected'
    if 'Portfolio rule' in heads:
        policy = row[heads.index('Portfolio rule')]
    year = None
    for title in ['Year', 'Test year']:
        if title in heads:
            year = int(float(row[heads.index(title)]))
    return d, h, policy, year


def find_model(study, label):
    if label in ['QQQ', 'qqq', LABEL, 'NYSE_BASKET']:
        return 'equal_universe'
    if label in ['SPY', 'market_proxy']:
        return 'market_proxy'
    if 'Equal-weight rebalanced universe' == label or label == 'equal_universe':
        return 'equal_universe'
    for filename in ['huber_ranking_pooled.csv', 'combined_2022_2025_metrics.csv']:
        f = study.frames.get(filename)
        if f is not None:
            matches = f[f.model.eq(label) | f.get('label', pd.Series('', index=f.index)).eq(label)]
            if len(matches):
                return matches.model.iloc[0]
    aliases = {'THINK+mixer+Huber': 'think_mix_huber', 'THINK': 'think',
               'Huber+ranking, weight 0.1': 'think_mix_huber_rank_01',
               'Huber+ranking, weight 0.3': 'think_mix_huber_rank_03',
               'Huber+ranking, weight 1.0': 'think_mix_huber_rank_10',
               'Validation choice of ranking weight': 'ranking_choice'}
    return aliases.get(label)


def ci(s, prefix):
    return f'[{s[prefix + "_ci_low"]:.3f}, {s[prefix + "_ci_high"]:.3f}]'


def control_cell(head, s, old):
    metric = {'Gross SR': 'gross_sharpe', 'Net SR': 'net_sharpe', 'Net return': 'net_cumulative_return',
              'Ann. net return': 'net_annualized_return', 'Cum. net return': 'net_cumulative_return',
              'Ann. volatility': 'net_annualized_volatility', 'Max drawdown': 'net_maximum_drawdown',
              'Annual turnover': 'turnover_annual', 'Profitable days': 'net_profitable_days',
              'Profitable evaluation periods': 'profitable_periods', 'CAGR': 'net_annualized_return',
              'Cum. return': 'net_cumulative_return', 'Vol.': 'net_annualized_volatility',
              'Max DD': 'net_maximum_drawdown', 'Turnover/year': 'turnover_annual',
              'Win periods': 'profitable_periods', 'Ann. vol': 'net_annualized_volatility'}
    if head == '95% SR CI':
        return ci(s, 'net_sharpe')
    if head in metric:
        key = metric[head]
        value = s.get(key)
        if value is None:
            return '—'
        if '%' in old:
            return f'{value:.1%}'
        if old.endswith('x'):
            return f'{value:.2f}x'
        return f'{value:.3f}'
    if head in ['IC', 'RankIC', 'ICIR', 'RankICIR'] or head.startswith('NDCG'):
        return '—'
    return old


def update_md(path, study):
    text = path.read_text(encoding='utf-8')
    rows = text.splitlines()
    stack, output, i, changes = {}, [], 0, 0
    while i < len(rows):
        match = re.match(r'^(#+) (.+)', rows[i])
        if match:
            level = len(match[1])
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = match[2]
        if rows[i].startswith('|') and i + 1 < len(rows) and re.match(r'^\|[ :|-]+\|$', rows[i + 1]):
            table_rows = []
            j = i
            while j < len(rows) and rows[j].startswith('|'):
                table_rows.append(cells(rows[j])); j += 1
            heads = table_rows[0]
            qcols = [k for k, c in enumerate(heads) if 'QQQ' in c]
            # A mixed-cohort table has one benchmark column with explicit per-row identity in CSV.
            table_ds = [context(stack, study, r, heads)[0] for r in table_rows[2:]]
            ny_table = any(d and d.startswith('NYSE') for d in table_ds)
            dedicated = ny_table and all(d and d.startswith('NYSE') for d in table_ds)
            newheads = [c.replace('QQQ', 'NYSE basket' if dedicated else 'benchmark') if ny_table else c for c in heads]
            transformed = [newheads, table_rows[1]]
            has_basket_row = any(r[0] in ['Equal-weight rebalanced universe', 'equal_universe', LABEL] for r in table_rows[2:])
            for r in table_rows[2:]:
                assert len(r) == len(heads), (path, r)
                d, h, policy, year = context(stack, study, r, heads)
                # Side-by-side Nasdaq/NYSE summary uses a single QQQ row.
                if r[0] == 'QQQ' and 'NYSE net SR' in heads:
                    r = r.copy(); r[0] = 'Comparison benchmark'
                    nd = next(d for d in study.datasets if d.startswith('NYSE'))
                    r[heads.index('NYSE net SR')] = f'{study.stats(5, nd, "validation-selected", "equal_universe")["net_sharpe"]:.3f}'
                    changes += 1; transformed.append(r); continue
                if not d or not d.startswith('NYSE'):
                    transformed.append(r); continue
                r = r.copy()
                is_control = r[0] in ['QQQ', 'qqq'] or ('ETF' in heads and r[heads.index('ETF')] == 'QQQ')
                if is_control and has_basket_row:
                    changes += 1; continue
                if 'Procedure' in heads:
                    s = study.proc(r[heads.index('Procedure')], d)
                elif 'Validation-chosen weight' in heads:
                    s = study.stats(h, d, policy, 'ranking_choice', year)
                elif 'ETF' in heads:
                    if not is_control:
                        transformed.append(r); continue
                    s = study.stats(h, d, policy, 'equal_universe', year)
                else:
                    m = find_model(study, r[0])
                    if not m:
                        transformed.append(r); continue
                    s = study.stats(h, d, policy, m, year)
                for k in qcols:
                    if 'ΔSR' in heads[k]:
                        r[k] = (f'{s["delta_benchmark"]:+.3f} {ci(s, "delta_benchmark")}'
                                if '[95% CI]' in heads[k] else f'{s["delta_benchmark"]:.3f}')
                        if k + 1 < len(heads) and heads[k + 1] == '95% ΔSR CI':
                            r[k + 1] = ci(s, 'delta_benchmark')
                    elif 'Minus' in heads[k]:
                        r[k] = f'{s["delta_benchmark"]:.3f}'
                    elif 'SR' in heads[k]:
                        r[k] = f'{s["benchmark_sharpe"]:.3f}'
                    else:
                        raise ValueError((path, heads[k]))
                    changes += 1
                if 'Years > both' in heads:
                    r[heads.index('Years > both')] = str(s['years_beating_both'])
                if is_control:
                    r[heads.index('ETF') if 'ETF' in heads else 0] = LABEL
                    for k, head in enumerate(heads):
                        if head.isdigit() and len(head) == 4:
                            r[k] = f'{study.stats(h, d, policy, "equal_universe", int(head))["net_sharpe"]:.3f}'
                        else:
                            r[k] = control_cell(head, s, r[k])
                        if 'ΔSR vs Huber' in head:
                            r[k] = f'{s["delta_huber"]:+.3f} {ci(s, "delta_huber")}'
                        if 'ΔSR vs SPY' in head:
                            r[k] = (f'{s["delta_spy"]:+.3f} {ci(s, "delta_spy")}'
                                    if '[95% CI]' in head else f'{s["delta_spy"]:.3f}')
                            if k + 1 < len(heads) and heads[k + 1] == '95% ΔSR CI':
                                r[k + 1] = ci(s, 'delta_spy')
                    changes += 1
                elif r[0] in ['equal_universe', 'Equal-weight rebalanced universe']:
                    r[0] = LABEL; changes += 1
                transformed.append(r)
            output.extend(line(r) for r in transformed)
            i = j; continue
        output.append(rows[i]); i += 1
    if changes:
        result = '\n'.join(output) + '\n'
        note = ('**Comparison benchmark:** NYSE tables use **NYSE basket — passive benchmark** '
                '(saved `equal_universe` portfolio: equal weights across all stocks eligible at each scheduled '
                'rebalance, with modeled execution constraints and costs). Earlier tables use their original '
                'NYSE cohort; top-100 tables use the new NYSE cohort. This is a simulated portfolio, not an ETF. '
                'Nasdaq retains QQQ; SPY remains a secondary reference. A mixed-cohort “benchmark” column means '
                'NYSE basket for NYSE rows and QQQ for Nasdaq rows. Paired confidence intervals were recalculated '
                'on identical dates using 1,000 circular 20-session block resamples within each annual fold '
                '(seed 20261005). Saved model returns and selection choices are unchanged. Embedded charts '
                'retain their previous benchmark overlays; this update applies to tables.\n\n')
        first, rest = result.split('\n', 1)
        result = first + '\n\n' + '<!-- nyse-benchmark:start -->\n' + note + '<!-- nyse-benchmark:end -->\n\n' + rest.lstrip('\n')
        result = result.replace('versus both ETFs', 'versus SPY and the cohort comparison benchmark')
        result = result.replace('with both ETFs within each regime', 'with SPY and the cohort comparison benchmark within each regime')
        result = result.replace('Model and ETF returns use identical sampled dates.', 'Model and benchmark returns use identical sampled dates.')
        STAGED[path] = result.encode('utf-8')


def main():
    studies = [Study(Path('runs/robust_comparison')), Study(Path('runs/largecap_20261010/comparison'))]
    files = ['combined_2022_2025_metrics.csv', 'combined_2022_2025_metrics_current.csv',
             'pooled_confidence_intervals.csv', 'pooled_confidence_intervals_current.csv',
             'model_vs_spy_qqq_by_year.csv', 'pooled_model_vs_spy_qqq.csv',
             'procedure_confidence_intervals.csv', 'selection_procedures_vs_spy_qqq.csv',
             'etf_controls_by_year.csv', 'huber_ranking_pooled.csv', 'huber_ranking_annual.csv',
             'regime_diagnostics.csv']
    for study in studies:
        print(f'Calculating tables: {study.root}', flush=True)
        for filename in files:
            if not (study.root / filename).exists():
                continue
            frame = study.update_frame(filename)
            stage_csv(study.root / filename, frame)
            if not study.large and (Path('docs') / filename).exists():
                stage_csv(Path('docs') / filename, frame)
        print(f'Checking Markdown tables: {study.root}', flush=True)
        for p in study.root.glob('*.md'):
            update_md(p, study)
        if not study.large:
            for p in Path('docs').glob('*.md'):
                update_md(p, study)
        else:
            p = Path('docs/largecap_retraining_20261010/primary_results.csv')
            f = read(p)
            renamed = f.rename(columns={c: c.replace('qqq', 'benchmark') for c in f.columns})
            for i, r in renamed.iterrows():
                s = study.proc(r.procedure, r.dataset)
                for k, v in s.items():
                    if k != 'net_sharpe':
                        renamed.at[i, k] = v
            stage_csv(p, renamed)
    # Verify all frozen inputs are still intact before applying the complete update.
    for path, expected in AUDIT['inputs'].items():
        assert digest(Path(path)) == expected, f'Concurrent input change: {path}'
    for path, body in STAGED.items():
        AUDIT['updated_files'].append({'path': str(path), 'before_sha256': digest(path),
                                       'after_sha256': hashlib.sha256(body).hexdigest()})
        path.write_bytes(body)
    AUDIT['passive_sharpes'] = [dict(study=str(s.root), dataset=d, horizon=h,
                                    net_sharpe=s.stats(h, d, 'validation-selected', 'equal_universe')['net_sharpe'])
                                for s in studies for d in s.datasets if d.startswith('NYSE') for h in [5, 1]]
    audit_path = Path('docs/audit/nyse_benchmark_update.json')
    audit_path.write_text(json.dumps(AUDIT, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'updated_files': len(STAGED), 'matched_paths': AUDIT['matched_paths'],
                      'passive_sharpes': AUDIT['passive_sharpes']}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
