"""Compare locked daily account paths without rerunning or modifying experiments.

Run directly with the workspace Python. Only this dated publication directory
is written; completed training/evaluation sources and artifacts stay untouched.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/core'))
from research_engine import performance

DEST = ROOT / 'docs/ndx_model_comparison_20261030'
SOURCES = {
    'old': ('ndx_retention_replication_20261029', ''),
    'stable': ('ndx_rank_stability_20261029_v2', ''),
    'new': ('ndx_rank_retraining_20261029', 'accounts'),
    'factor': ('ndx_factor_controls_20261029', ''),
}
GROUPS = {
    'original_five': 'Original five seeds',
    'new_five': 'Second five seeds',
    'fresh_five': 'Third five seeds',
    'ten_seed': 'Combined ten seeds',
    'fifteen_seed': 'Combined fifteen seeds',
}
# Named representatives, not an exhaustive search or a new selection policy.
# Each family's choice and its historical selection are disclosed in README.md.
SERIES = {
    'old': ('Old retention (historical selected recipe)', '#4b5563', '--'),
    'stable': ('Old forecasts: stable consensus / soft ten / 50% cap', '#b87b08', ':'),
    'loss_issuer': ('Retrained: loss checkpoint / five issuers', '#007f73', '-'),
    'loss_retain': ('Retrained: loss checkpoint / retention five', '#2463c9', '-'),
    'rank': ('Retrained: rank checkpoint / retention five', '#bf426b', '-'),
    'snapshots': ('Retrained: three snapshots / five issuers', '#8657b8', '-'),
    'factor': ('Rank checkpoint + liquid-consensus factor (50:50)', '#b7591c', '-.'),
    'control': ('Training-mean-rank control / retention five', '#5d6368', ':'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources():
    sources = {}; hashes = {}
    for name, (folder, subfolder) in SOURCES.items():
        run = ROOT / 'runs' / folder
        lock_path = run / 'COMPLETE.json'
        lock = json.loads(lock_path.read_text())['output_sha256']
        hashes[str(lock_path.relative_to(ROOT)).replace('\\', '/')] = sha(lock_path)
        source = run / subfolder
        for path in (source / 'PATHS.npz', run / 'METRICS.csv'):
            relative = path.relative_to(run).as_posix()
            actual = sha(path)
            assert actual == lock[relative], f'Locked artifact changed: {path}'
            hashes[path.relative_to(ROOT).as_posix()] = actual
        with np.load(source / 'PATHS.npz', allow_pickle=False) as saved:
            paths = {key: saved[key].copy() for key in saved.files}
        sources[name] = (paths, pd.read_csv(run / 'METRICS.csv').set_index('id'))
    dates = sources['new'][0]['dates']
    assert len(dates) == 1003 and dates[0] == '2022-01-03' and dates[-1] == '2025-12-31'
    for paths, _ in sources.values():
        np.testing.assert_array_equal(paths['dates'], dates)
    return sources, pd.DatetimeIndex(dates), hashes


def account_id(series, group):
    if series == 'old':
        return 'old', f'{group}__retention__cap75__b03__c1'
    if series == 'stable':
        return 'stable', f'stable_consensus__soft10__{group}__cap50__b03__c1'
    if series == 'factor':
        return 'factor', f'liquid_consensus__nn50__{group}__retain5__c1__b03'
    if series == 'control':
        return 'new', 'training_mean_rank__seed_free__retain5__cap75__c1__b03'
    method, allocation = {
        'loss_issuer': ('loss', 'issuer5'), 'loss_retain': ('loss', 'retain5'),
        'rank': ('rank', 'retain5'), 'snapshots': ('snapshots', 'issuer5'),
    }[series]
    return 'new', f'{method}__{group}__{allocation}__cap75__c1__b03'


def collect(sources):
    accounts = {}; rows = []; seen = set()
    for group in GROUPS:
        for series in SERIES:
            if series in ('old', 'stable') and group in ('fresh_five', 'fifteen_seed'):
                continue  # No historical forecasts fitted for those groups.
            source, key = account_id(series, group)
            paths, metrics = sources[source]
            daily = paths[key]
            assert daily.shape == (1003,) and np.isfinite(daily).all()
            recomputed = performance(daily)
            row = metrics.loc[key]
            for field, value in recomputed.items():
                np.testing.assert_allclose(value, row[field], rtol=1e-10, atol=1e-12,
                                           err_msg=f'{source}/{key}/{field}')
            accounts[group, series] = (daily, recomputed)
            if (source, key) not in seen:
                seen.add((source, key))
                rows.append(dict(source=source, id=key, series=series,
                                 group='seed_free' if series == 'control' else group,
                                 **recomputed))
    return accounts, pd.DataFrame(rows)


def figure(dates, accounts, groups, series, filename, title):
    multi = len(groups) > 1
    fig, axes = plt.subplots(2, len(groups), figsize=(18, 9) if multi else (14, 9),
                             squeeze=False, sharex=True, sharey='row',
                             gridspec_kw={'height_ratios': [2, 1]})
    handles = {}; labels = {}
    # An explicit initial NAV includes loss on the first evaluated trading day.
    x = dates.insert(0, dates[0] - pd.offsets.BDay(1))
    for col, group in enumerate(groups):
        for name in series:
            if (group, name) not in accounts:
                continue
            daily, metrics = accounts[group, name]
            label, color, linestyle = SERIES[name]
            equity = np.r_[1., np.cumprod(1. + daily)]
            drawdown = equity / np.maximum.accumulate(equity) - 1.
            line, = axes[0, col].plot(x, 100 * equity, color=color, ls=linestyle,
                                     lw=2.05 if name == 'loss_issuer' else 1.55,
                                     alpha=.96, zorder=4 if name == 'loss_issuer' else 2)
            axes[1, col].plot(x, drawdown, color=color, ls=linestyle, lw=1.2, alpha=.9)
            handles[name] = line
            labels[name] = label if multi else (
                f'{label}   |   SR {metrics["sharpe"]:.3f}, '
                f'CAGR {metrics["annualized_return"]:.1%}, DD {metrics["maximum_drawdown"]:.1%}')
        axes[0, col].set_title(GROUPS[group], loc='left', fontsize=12, weight='bold', pad=10)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axes[0, col].yaxis.set_major_formatter(StrMethodFormatter('${x:.0f}'))
        axes[0, col].axhline(100, color='#9ca3af', linewidth=.7)
        axes[1, col].axhline(0, color='#9ca3af', linewidth=.7)
        for ax in axes[:, col]:
            ax.grid(axis='y', color='#d8dee7', linewidth=.6)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['bottom', 'left']].set_color('#bac2cb')
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_xlim(x[0], dates[-1])
            ax.tick_params(labelsize=10)
    axes[0, 0].set_ylabel('Net wealth from $100')
    axes[1, 0].set_ylabel('Drawdown from prior peak')
    fig.suptitle(title, x=.07, y=.97, ha='left', fontsize=18, weight='bold')
    fig.text(.07, .928, '2022–2025 historical development paths • 1,003 sessions • fees and 3% short borrowing included',
             fontsize=11, color='#505c6b')
    ordered = [name for name in series if name in handles]
    fig.legend([handles[n] for n in ordered], [labels[n] for n in ordered],
               loc='lower left', bbox_to_anchor=(.064, .056), frameon=False,
               fontsize=9.4 if multi else 10, ncol=2 if multi else 1,
               handlelength=3, columnspacing=2, labelspacing=.65)
    risk_note = '20% stock-volatility target; continuous QQQ short up to 75% NAV'
    risk_note += ' (stable-consensus comparator: 50%).' if 'stable' in series else '.'
    scope_note = 'Same market history, different seed cohorts.' if multi else 'Seed cohort held fixed; control is seed-free.'
    fig.text(.07, .025, risk_note + '\n' + scope_note +
             ' Historical selection is disclosed; unseen-period performance is untested.',
             fontsize=9, color='#505c6b')
    fig.subplots_adjust(left=.07, right=.975, top=.88, bottom=.24 if multi else .32,
                        hspace=.13, wspace=.13)
    fig.savefig(DEST / filename, dpi=160, facecolor='white')
    plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.labelsize': 11})
    sources, dates, hashes = load_sources()
    accounts, metrics = collect(sources)
    metrics.to_csv(DEST / 'COMPARISON_METRICS.csv', index=False)
    figure(dates, accounts, ['ten_seed'], list(SERIES), 'ten_seed_paths.png',
           'Model comparison: the same ten-seed ensemble')
    figure(dates, accounts, ['original_five', 'new_five', 'fresh_five'],
           ['old', 'loss_issuer', 'loss_retain', 'rank', 'snapshots'],
           'disjoint_seed_paths.png', 'Three separate seed cohorts: wealth and drawdown')
    figure(dates, accounts, ['fifteen_seed'],
           ['loss_issuer', 'loss_retain', 'rank', 'snapshots', 'factor', 'control'],
           'fifteen_seed_paths.png', 'Model comparison: the same fifteen-seed ensemble')
    report = '''# Daily model-path comparison — October 30, 2026

These figures replot locked daily returns; they do not train, retune, or simulate
new accounts. All paths share 1,003 sessions, January 3, 2022–December 31, 2025.
Net wealth compounds the saved daily returns from $100; drawdowns include the
initial $100 NAV. Sharpe uses a 3% annual risk-free hurdle and 252 sessions.
Modeled trading fees and 3% calendar-day short borrowing are already deducted.

![Combined ten](ten_seed_paths.png)

![Separate cohorts](disjoint_seed_paths.png)

![Combined fifteen](fifteen_seed_paths.png)

## What is being compared

- The old retention strategy is the historically selected recipe, not a fresh
  control selected without these outcomes. Its original/second/ten SR is
  1.731 / 0.648 / 1.381. Old models were not fitted for the third/fifteen groups;
  no missing old paths are extrapolated or replaced with original-five results.
- Four trained representatives expose both successful and failed choices:
  loss-selected five issuers, loss-selected retention five, rank-selected
  retention five, and three-snapshot five issuers. The issuer loss row was the
  observed best minimum across all five ensembles; retention loss was the best
  minimum across the three disjoint groups. These choices are descriptive.
- Stable-consensus / soft ten was the earlier postprocessing study's best
  weaker-half candidate. It uses a 50% hedge ceiling; other displayed accounts
  use 75%. Both ceilings are subject to remaining gross budget and trailing
  volatility limits. That comparator changes portfolio and hedge rules as well
  as model aggregation; it is not an isolated model-only comparison.
- The factor hybrid is the strongest weaker-group hybrid in its fixed study:
  50:50 ranking-checkpoint forecasts and liquid-consensus factor, retention five.
  It is outcome-selected and did not meet the performance target. The separate
  study blended the ranking checkpoint, not the subsequently stronger loss one.
- The training-mean-rank control uses only each fold's training-period stock
  return ranks, with shrinkage. It is deterministic, seed-free, and repeated
  visually for context rather than counted as independent seed evidence.

All neural ensembles blend THINK and dynamic-hypergraph rank forecasts equally.
New models use 12 epochs; the old schedule used up to 30 with early stopping.
Initialization, objective, validation selection, and duration changed together.
Five issuers consolidates share classes; retention five selects securities and
adds 0.02 to previous target selections. Both use the 20% stock-basket volatility
target and the existing weekly execution/cost model. Portfolio rules differ,
so aggregate curves do not isolate architecture or initialization effects.

## Interpretation and limitations

Loss-selected five issuers reduces the three disjoint cohort SR spread to
approximately 0.053, with SR 1.171 / 1.147 / 1.200; ten/fifteen SR is 1.023 / 1.235.
No trained recipe cleared SR>1.5 across every predefined ensemble. Pooled ranks
are not averages of portfolio returns: top-k selection, retention, and risk sizing
are nonlinear. More seeds can therefore change or worsen the account path.

Neutral initialization, ranking-aligned training, validation separation, and
all-cohort reporting improve the design and observed seed stability. They do
not establish unseen-market generalization. All 2022–2025 outcomes have been
repeatedly inspected, and all cohorts share those market realizations. The ten
and fifteen groups also overlap constituent models. No new untouched evaluation,
multiple-search correction, or causal single-change ablation is supplied here.

The plots are a representative overview, not the full experiment grid. All
recipes, costs, failed cases, and controls remain in the original reports:
[retraining](../ndx_rank_retraining_20261029/REPORT.md),
[postprocessing](../ndx_rank_stability_20261029_v2/REPORT.md),
[retention](../ndx_retention_replication_20261029/REPORT.md), and
[factor controls](../ndx_factor_controls_20261029/REPORT.md).

## Verification and reproduction

Every source PATHS.npz and METRICS.csv is checked against its completed-run SHA256
manifest. Date arrays must match exactly. Every performance field for the 32
distinct compared accounts is recomputed from daily returns and checked against
its saved CSV (relative tolerance 1e-10, absolute tolerance 1e-12). No experiment
source, checkpoint, ledger, or completion manifest is modified. The publication
manifest records source hashes, chart hashes, and the plotting script hash.

From the workspace root:

```powershell
.\\.venv\\Scripts\\python.exe research/reporting/plot_ndx_model_comparison_20261030.py
```

[Compared account metrics](COMPARISON_METRICS.csv) · [Provenance](MANIFEST.json)
'''
    # The deterministic control occurs once across all five visual groupings.
    assert len(metrics) == 32
    (DEST / 'README.md').write_text(report, encoding='utf-8')
    artifacts = {path.name: sha(path) for path in sorted(DEST.iterdir())
                 if path.is_file() and path.name != 'MANIFEST.json'}
    manifest = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    script=Path(__file__).relative_to(ROOT).as_posix(),
                    script_sha256=sha(Path(__file__)), source_sha256=hashes,
                    output_sha256=artifacts, days=len(dates), accounts=len(metrics),
                    metrics_reconstruction='passed', source_hash_checks='passed',
                    new_training=False, new_account_evaluation=False)
    (DEST / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(DEST), compared_accounts=len(metrics), checks='passed')))


if __name__ == '__main__':
    main()
