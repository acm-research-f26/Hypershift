"""Frozen allocation-only top-five experiment, with independent accounting audit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from audit_historical_ndx import independent_metrics
from audit_historical_ndx_ranking import average_ranks
from compare_ndx_strategies_20261020 import bootstrap_sharpe
from ndx_policy_engine import policy_replay
from pit_research_engine import load_data, continuous_backtest, etf_backtest
from report_ndx_extended_hours import forecast_maps
from report_ndx_ranking import joint_intervals
from research_engine import performance
from run_ndx_extended_hours import complete
from run_ndx_alpaca_training import sha

ROOT = Path(__file__).resolve().parent
PARENT = ROOT/'runs/ndx_extended_hours_20261024'
PRIOR = ROOT/'docs/ndx_extended_hours_20261024'
RUN = ROOT/'runs/ndx_extended_hours_top5_20261024'
DOC = ROOT/'docs/ndx_extended_hours_top5_20261024'
CONFIG = ROOT/'config/ndx_extended_hours_top5_20261024.json'
GROUPS = ['original_five', 'ten_seed', 'new_five']
VIEWS = ['daily', 'after_hours_premarket']


def write(p, obj): p.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
def read(name): return pd.read_csv(RUN/name, dtype={'seed': str})
def key(view, group, k, seed='ensemble', cost=1): return f'{view}/{group}/{seed}/{k}/{cost}'


def verify():
    for p, h in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items():
        if sha(p) != h: raise ValueError('Frozen top-five input changed: '+p)


def freeze():
    complete()
    RUN.mkdir(parents=True, exist_ok=True)
    sources = [CONFIG, Path(__file__), PARENT/'FITS_COMPLETE_LOCK.json', PARENT/'PROTOCOL.json',
               PARENT/'INPUT_CODE_LOCK.json', PARENT/'FINAL_PUBLICATION.json',
               ROOT/'runs/nasdaq_topk_pseudohuber_20261011/metrics.csv',
               ROOT/'runs/nasdaq_topk_pseudohuber_20261011/curves.csv']
    sources += [ROOT/n for n in ['pit_research_engine.py', 'research_engine.py', 'ndx_policy_engine.py',
        'report_ndx_extended_hours.py', 'report_ndx_ranking.py', 'compare_ndx_strategies_20261020.py',
        'audit_historical_ndx.py', 'audit_historical_ndx_ranking.py', 'run_robust_research.py']]
    publication = json.loads((PARENT/'FINAL_PUBLICATION.json').read_text())
    for name, h in publication['artifact_hashes'].items():
        p = PRIOR/name
        if sha(p) != h: raise ValueError('Published parent changed: '+name)
        sources.append(p)
    # Complete() checks every original checkpoint/forecast, including unused AH controls.
    lock = json.loads((PARENT/'INPUT_CODE_LOCK.json').read_text())
    for name, h in json.loads((PARENT/'FITS_COMPLETE_LOCK.json').read_text())['hashes'].items(): lock[str(PARENT/name)] = h
    lock.update({str(p.resolve()): sha(p) for p in sources})
    target = RUN/'INPUT_CODE_LOCK.json'
    if target.exists():
        if json.loads(target.read_text()) != lock: raise ValueError('Refusing to replace frozen top-five inputs')
    else:
        write(target, lock); shutil.copyfile(CONFIG, RUN/'PROTOCOL.json')
        shutil.copyfile(__file__, RUN/'SOURCE.py')
        write(RUN/'FREEZE.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), new_fits=0,
            protocol_sha256=sha(CONFIG), new_account_metrics_absent=not (RUN/'metrics.csv').exists()))
    verify(); print('Frozen saved-forecast allocation test: 62 accounts, zero new fits', flush=True)


def inputs():
    verify(); data = load_data(); plan = json.loads((PARENT/'PROTOCOL.json').read_text())
    ids, raw, scores = forecast_maps(data, plan)
    def ranker(available, mask):
        ranked = []
        for values in available:
            out = np.zeros_like(values, dtype=float)
            for j, selected in enumerate(mask): out[j, selected] = average_ranks(values[j, selected])/selected.sum()
            ranked.append(out)
        return np.mean(ranked, axis=0)
    _, independent_raw, independent_scores = forecast_maps(data, plan, ranker)
    for view in VIEWS:
        for group in GROUPS: np.testing.assert_allclose(scores[view, group], independent_scores[view, group], atol=1e-12, rtol=0)
    return data, ids, raw, scores


def cagr_samples(values, block):
    """Circular-block sums of log growth; same date draws as Sharpe helper."""
    n, m = values.shape; logs = np.log1p(values)
    prefix = np.vstack([np.zeros((1, m)), np.cumsum(np.concatenate([logs, logs]), axis=0)])
    starts = np.arange(n); full, tail = divmod(n, block)
    sums = prefix[starts+block]-prefix[starts]; ends = prefix[starts+tail]-prefix[starts]
    rng = np.random.default_rng(20261020+block); result = []
    for _ in range(100):
        draw = rng.integers(0, n, (100, full+bool(tail)))
        growth = sums[draw[:, :full]].sum(1)
        if tail: growth += ends[draw[:, -1]]
        result.append(np.expm1(growth*252/n))
    return np.concatenate(result)


def contrasts():
    result = []
    for group in GROUPS:
        a = key('after_hours_premarket', group, 5)
        for label, b in [('top-five minus top-ten', key('after_hours_premarket', group, 10)),
                         ('top-five minus SPY', key('SPY', 'benchmark', 0)),
                         ('top-five minus QQQ', key('QQQ', 'benchmark', 0)),
                         ('top-five AH+PM minus top-five daily', key('daily', group, 5))]:
            result.append((group+': '+label, ('account', a), ('account', b)))
    return result


def evaluate():
    verify()
    if (RUN/'EVALUATED.json').exists(): raise ValueError('Completed evaluation is immutable')
    data, ids, raw, scores = inputs(); metrics = []; curves = []; ledgers = []
    rankings = pd.read_csv(PRIOR/'ranking.csv').set_index(['view', 'group']).rankic
    quality = pd.read_csv(PRIOR/'prediction_quality.csv', dtype={'seed': str}).set_index(['view', 'seed']).rankic
    def record(view, group, k, seed, cost, m, path, ledger=None):
        tags = dict(id=key(view, group, k, seed, cost), view=view, group=group, k=k, seed=str(seed), cost_multiplier=cost)
        ic = None if view in ['SPY', 'QQQ'] else float(quality.loc[view, str(seed)]) if group == 'individual' else float(rankings.loc[view, group])
        metrics.append(dict(**tags, rankic=ic, **m)); curves.append(path.assign(**tags))
        if ledger is not None: ledgers.append(ledger.assign(**tags))
    for view in VIEWS:
        for group in GROUPS:
            for k in [5, 10]:
                for cost in [1, 2, 4]:
                    m, p, l = continuous_backtest(data, ids, scores[view, group], k, cost)
                    record(view, group, k, 'ensemble', cost, m, p, l)
            print('Evaluated fixed sizes/costs', view, group, flush=True)
    for seed, prediction in raw['after_hours_premarket'].items():
        for k in [5, 10]:
            m, p, l = continuous_backtest(data, ids, prediction, k, 1)
            record('after_hours_premarket', 'individual', k, seed, 1, m, p, l)
    for ticker in ['SPY', 'QQQ']:
        for cost in [1, 2, 4]:
            m, p = etf_backtest(data, ticker, cost); record(ticker, 'benchmark', 0, 'ensemble', cost, m, p)
    mf = pd.DataFrame(metrics); cf = pd.concat(curves, ignore_index=True)
    assert len(mf) == 62 and not mf.id.duplicated().any()
    mf.to_csv(RUN/'metrics.csv', index=False); cf.to_csv(RUN/'curves.csv', index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(RUN/'ledgers.csv', index=False)
    annual = []; monthly = []
    for account, p in cf.groupby('id', sort=False):
        date = pd.to_datetime(p.date)
        for year, q in p.groupby(date.dt.year): annual.append(dict(id=account, year=int(year), **performance(q.net)))
        for month, q in p.groupby(date.dt.to_period('M').astype(str)): monthly.append(dict(id=account, month=month, net_return=float(np.prod(1+q.net)-1)))
    pd.DataFrame(annual).to_csv(RUN/'annual.csv', index=False); pd.DataFrame(monthly).to_csv(RUN/'monthly.csv', index=False)
    account_paths = {account: p.net.to_numpy() for account, p in cf.groupby('id', sort=False)}
    ordering = mf.id.tolist(); matrix = np.column_stack([account_paths[k] for k in ordering])
    sr_ci = []; paired = []; family = contrasts()
    for block in [5, 20, 60]:
        sr = bootstrap_sharpe(matrix, block=block); growth = cagr_samples(matrix, block)
        low, high = np.quantile(sr, [.025, .975], axis=0)
        for j, account in enumerate(ordering): sr_ci.append(dict(id=account, block_sessions=block, sr_low=low[j], sr_high=high[j]))
        for measure, samples, observed in [('sharpe_difference', sr, mf.sharpe.to_numpy()), ('cagr_difference', growth, mf.annualized_return.to_numpy())]:
            draws = np.column_stack([samples[:, ordering.index(a[1])]-samples[:, ordering.index(b[1])] for _, a, b in family])
            estimate = np.array([observed[ordering.index(a[1])]-observed[ordering.index(b[1])] for _, a, b in family])
            se = draws.std(0, ddof=1); critical = np.quantile(np.max(np.abs((draws-estimate)/np.maximum(se, 1e-12)), axis=1), .95)
            for j, (label, a, b) in enumerate(family):
                paired.append(dict(contrast=label, measure=measure, left_id=a[1], right_id=b[1], block_sessions=block, replicates=10000,
                    estimate=estimate[j], paired_low=np.quantile(draws[:, j], .025), paired_high=np.quantile(draws[:, j], .975),
                    simultaneous_low=estimate[j]-critical*se[j], simultaneous_high=estimate[j]+critical*se[j], family_comparisons=12))
    pd.DataFrame(sr_ci).to_csv(RUN/'sharpe_intervals.csv', index=False)
    pd.DataFrame(paired).to_csv(RUN/'paired_intervals.csv', index=False)
    joint_intervals({('account', k): p for k, p in account_paths.items()}, family, [5, 20, 60], annualizer=252.).to_csv(RUN/'active_mean_intervals.csv', index=False)
    write(RUN/'EVALUATED.json', dict(accounts=62, fits=0, status='complete_pending_independent_audit'))
    print('All 62 accounts and paired intervals computed; beginning independent audit', flush=True)


def audit():
    verify(); data, ids, raw, scores = inputs(); mf = read('metrics.csv').set_index('id'); cf = read('curves.csv')
    parent_curves = pd.read_csv(PRIOR/'curves.csv', dtype={'seed': str}); prior = pd.read_csv(ROOT/'runs/nasdaq_topk_pseudohuber_20261011/curves.csv', dtype={'seed': str})
    reference_checks = 0; max_difference = 0.; checks = 0
    for account, p in cf.groupby('id', sort=False):
        row = mf.loc[account]; view, group, k, cost = row['view'], row.group, int(row.k), int(row.cost_multiplier)
        assert len(p) == 1003 and p.date.is_unique and p.date.is_monotonic_increasing
        for field, value in independent_metrics(p.net).items(): np.testing.assert_allclose(value, row[field], atol=1e-10, rtol=1e-10)
        if view in ['SPY', 'QQQ']:
            root = Path(data['manifest']['root']); first = data['entry'][ids[0]]
            q = pd.read_csv(root/'etf_adj_close.csv')[view].to_numpy(); volume = pd.read_csv(root/'etf_dollar_volume.csv')[view].to_numpy()
            rate = cost*(.0007+.0003*np.sqrt(1e6/volume[first-20:first].mean()/.01))
            nav = 1e6/(1+rate)/q[first]*q[first:]
            nav[-1] *= 1-cost*(.0007+.0003*np.sqrt(nav[-1]/volume[-21:-1].mean()/.01))
        else:
            prediction = raw[view][int(row.seed)] if group == 'individual' else scores[view, group]
            nav = policy_replay(data, ids, prediction, k, cost)
        actual = 1e6*np.cumprod(1+p.net.to_numpy())
        max_difference = max(max_difference, float(np.abs(nav-actual).max()))
        np.testing.assert_allclose(nav, actual, atol=1e-5, rtol=1e-10); checks += 1
        if k == 10 or view in ['SPY', 'QQQ']:
            match = parent_curves[(parent_curves['view'] == view)&(parent_curves.group == ('ten_seed' if group == 'benchmark' else group))&(parent_curves.seed == row.seed)&(parent_curves.cost_multiplier == cost)]
            np.testing.assert_array_equal(match.date.to_numpy(), p.date.to_numpy()); np.testing.assert_allclose(match.net.to_numpy(), p.net.to_numpy(), atol=1e-14, rtol=0)
            reference_checks += 1
        elif view == 'daily' and group == 'original_five':
            match = prior[(prior.model == 'think_mix')&(prior.seed == 'ensemble')&(prior.k == 5)&(prior.cost_multiplier == cost)]
            np.testing.assert_array_equal(match.date.to_numpy(), p.date.to_numpy()); np.testing.assert_allclose(match.net.to_numpy(), p.net.to_numpy(), atol=1e-14, rtol=0)
            reference_checks += 1
    ledger = read('ledgers.csv'); np.testing.assert_allclose(ledger.nav, ledger.cash+ledger.holdings, atol=1e-7, rtol=1e-12)
    assert ledger.cash.min() >= -1e-5 and checks == 62 and reference_checks == 37
    # Independent materialized resamples verify optimized SR and CAGR contrasts.
    selected = [key('after_hours_premarket', 'original_five', k) for k in [5, 10]]+[key('SPY', 'benchmark', 0)]
    matrix = np.column_stack([cf[cf.id == k].net.to_numpy() for k in selected]); n = len(matrix)
    rng = np.random.default_rng(20261040); ss = []; gg = []
    for _ in range(100):
        starts = rng.integers(0, n, (100, int(np.ceil(n/20))))
        index = ((starts[:, :, None]+np.arange(20)) % n).reshape(100, -1)[:, :n]; sample = matrix[index]
        ss.append(np.sqrt(252)*(sample.mean(1)-((1.03)**(1/252)-1))/sample.std(1, ddof=1))
        gg.append(np.expm1(np.log1p(sample).sum(1)*252/n))
    optimized_sr = bootstrap_sharpe(matrix, block=20); optimized_growth = cagr_samples(matrix, 20)
    np.testing.assert_allclose(np.concatenate(ss), optimized_sr, atol=2e-10, rtol=1e-10)
    np.testing.assert_allclose(np.concatenate(gg), optimized_growth, atol=2e-10, rtol=1e-10)
    write(RUN/'AUDIT.json', dict(status='passed', independent_accounts=62, matching_prior_accounts=37,
        rank_ensembles_independently_reconstructed=6, independent_materialized_bootstrap_accounts=3,
        bootstrap_replicates=10000, largest_nav_difference_dollars=max_difference,
        source_hashes_verified=len(json.loads((RUN/'INPUT_CODE_LOCK.json').read_text())), new_training_fits=0))
    print('Independent accounting, source, rank and bootstrap audits passed', flush=True)


def publish():
    verify(); proof = json.loads((RUN/'AUDIT.json').read_text()); assert proof['status'] == 'passed'
    DOC.mkdir(parents=True, exist_ok=True); figures = DOC/'figures'; figures.mkdir(exist_ok=True)
    mf = read('metrics.csv'); cf = read('curves.csv'); ci = read('sharpe_intervals.csv'); paired = read('paired_intervals.csv')
    annual = read('annual.csv'); base = mf[(mf.cost_multiplier == 1)&(mf.seed == 'ensemble')].set_index('id')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for group in GROUPS:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        for account, label in [(key('after_hours_premarket', group, 5), 'AH + PM / top 5'), (key('after_hours_premarket', group, 10), 'AH + PM / top 10'),
                               (key('daily', group, 5), 'Daily only / top 5'), (key('SPY', 'benchmark', 0), 'SPY'), (key('QQQ', 'benchmark', 0), 'QQQ')]:
            p = cf[cf.id == account]; wealth = np.r_[1., np.cumprod(1+p.net.to_numpy())]
            line, = axes[0].plot(pd.to_datetime(p.date), wealth[1:], label=f'{label}: {base.loc[account].annualized_return:.2%} CAGR')
            axes[1].plot(pd.to_datetime(p.date), (wealth/np.maximum.accumulate(wealth)-1)[1:]*100, color=line.get_color())
        axes[0].set(title=f'Five versus ten holdings: {group}, exploratory 2022–2025', ylabel='Net wealth per initial $1')
        axes[0].legend(fontsize=9); axes[1].set(ylabel='Drawdown (%)')
        for ax in axes: ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(figures/(group+'.png'), dpi=150); plt.close(fig)
    lines = ['# After-hours + premarket: five versus ten holdings', '', 'October 24, 2026. Allocation-only follow-up on previously inspected 2022–2025 history. Saved forecasts are unchanged; zero new model fits.', '',
        '| Seed group | Holdings | CAGR | Sharpe | Max drawdown | RankIC | 95% SR interval |', '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for group in GROUPS:
        for k in [5, 10]:
            account = key('after_hours_premarket', group, k); row = base.loc[account]; bounds = ci[(ci.id == account)&(ci.block_sessions == 20)].iloc[0]
            lines.append(f'| {group} | {k} | {row.annualized_return:.2%} | {row.sharpe:.3f} | {row.maximum_drawdown:.2%} | {row.rankic:.4f} | [{bounds.sr_low:.3f}, {bounds.sr_high:.3f}] |')
    lines += ['', 'All groups use their original saved rank ensembles. Top-five targets approximately 20% per security versus 10% for top-ten, with fractional ties, 1% ADV share caps, cash scaling, missing-fill cancellation, modeled fees and terminal liquidation unchanged. RankIC describes the unchanged stock forecasts and therefore is identical across holding counts.', '',
        'AH + PM bars end by 09:15 Eastern, decision is 09:20 on entry day, and fixed shares fill at that session’s regular close. Shares are sized from the known preceding close. The account engine’s origin-date field is bookkeeping alignment, not a claim the forecast existed before the premarket features. No overnight move before entry is earned.', '',
        '## Paired holding-count differences', '', '| Group | CAGR difference | Paired 95% interval | Sharpe difference | Paired 95% interval |', '| --- | ---: | --- | ---: | --- |']
    for group in GROUPS:
        part = paired[(paired.contrast == group+': top-five minus top-ten')&(paired.block_sessions == 20)].set_index('measure')
        a, b = part.loc['cagr_difference'], part.loc['sharpe_difference']
        lines.append(f'| {group} | {a.estimate*100:+.2f} pp | [{a.paired_low*100:.2f}, {a.paired_high*100:.2f}] pp | {b.estimate:+.3f} | [{b.paired_low:.3f}, {b.paired_high:.3f}] |')
    lines += ['', 'Intervals use 10,000 paired circular-block resamples. Twenty sessions is primary; five and sixty are disclosed. Twelve contrasts cover all three groups against matched top-ten, SPY, QQQ and matched daily-only top-five. CSVs include simultaneous diagnostics. These fitted-path intervals omit prior research-search, retraining and future-regime uncertainty. No winning group or holding count is promoted as an unseen validation result.', '',
        '## Annual returns for the requested original-five ensemble', '', '| Year | AH + PM / top 5 | AH + PM / top 10 | Daily / top 5 | SPY | QQQ |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    accounts = [key('after_hours_premarket', 'original_five', k) for k in [5, 10]]+[key('daily', 'original_five', 5), key('SPY', 'benchmark', 0), key('QQQ', 'benchmark', 0)]
    for year in [2022, 2023, 2024, 2025]:
        values = [annual[(annual.id == account)&(annual.year == year)].iloc[0].cumulative_return for account in accounts]
        lines.append('| '+str(year)+' | '+' | '.join(f'{value:.2%}' for value in values)+' |')
    lines += ['', '## Cost stress', '', '| Group | Holdings | Base-cost CAGR | 2x fees | 4x fees |', '| --- | ---: | ---: | ---: | ---: |']
    for group in GROUPS:
        for k in [5, 10]:
            values = [mf[mf.id == key('after_hours_premarket', group, k, cost=c)].iloc[0].annualized_return for c in [1, 2, 4]]
            lines.append(f'| {group} | {k} | '+' | '.join(f'{value:.2%}' for value in values)+' |')
    lines += ['', '## Wealth and drawdown comparisons', '']
    for group in GROUPS: lines += [f'![{group}](figures/{group}.png)', '']
    selected = mf[(mf.group == 'individual')&(mf.cost_multiplier == 1)].pivot(index='seed', columns='k', values='annualized_return')
    lines += ['## Individual-seed checks', '', f'Top-five CAGR exceeded its paired top-ten in {int((selected[5] > selected[10]).sum())}/10 individual runs. All twenty individual accounts are disclosed alongside the ensemble and fee-stress accounts.', '',
        '## Verification', '', f'All 62 accounts were independently replayed. All 37 matching prior paths reproduced, including the original daily top-five control. Six rank ensembles were independently reconstructed. Three accounts’ 10,000 bootstrap replicates were materialized to verify the optimized calculations. Maximum account NAV difference was ${proof["largest_nav_difference_dollars"]:.8f}. Parent model/data/source hashes were checked before evaluation.', '',
        '[All metrics](metrics.csv) · [Daily returns](curves.csv) · [Annual returns](annual.csv) · [Monthly returns](monthly.csv) · [Absolute SR intervals](sharpe_intervals.csv) · [Paired CAGR and SR differences](paired_intervals.csv) · [Arithmetic active-return intervals](active_mean_intervals.csv) · [Protocol](PROTOCOL.json) · [Audit](AUDIT.json)', '']
    (DOC/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    for name in ['metrics.csv', 'curves.csv', 'annual.csv', 'monthly.csv', 'sharpe_intervals.csv', 'paired_intervals.csv', 'active_mean_intervals.csv', 'PROTOCOL.json', 'FREEZE.json', 'AUDIT.json']:
        shutil.copyfile(RUN/name, DOC/name)
    write(RUN/'PUBLICATION.json', dict(status='completed_audited_exploratory', accounts=62, new_training_fits=0,
        artifact_hashes={str(p.relative_to(DOC)): sha(p) for p in DOC.rglob('*') if p.is_file()}))
    print('Published', DOC/'REPORT.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'evaluate', 'audit', 'publish'])
    globals()[parser.parse_args().stage]()
