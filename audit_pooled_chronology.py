"""Audit saved chronological folds and causal preprocessing without retraining."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_engine import load_panel, prepare_panel, fold_indices, training_groups
from run_architecture_research import scale_inputs, robust_score


ROOT = Path('runs/robust_comparison')
RUNS = {5: Path('runs/robust_2022_2025_final'), 1: Path('runs/daily_2022_2025_final')}


def main():
    rows = []
    source_checks = 0
    for horizon, root in RUNS.items():
        protocol = json.loads((root / 'PROTOCOL.json').read_text())
        for path, expected in protocol['code_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
            source_checks += 1
        policies = json.loads((root / 'validation_search.json').read_text())
        lock = json.loads((root / 'SELECTION_LOCK.json').read_text())
        for policy in policies:
            assert policy['selected'] == max(policy['candidates'], key=lambda r: r['score'])
        for dataset in ['NYSE_recent', 'NASDAQ_recent']:
            panel = load_panel(dataset)
            data = prepare_panel(panel, horizon=horizon)
            for year in [2022, 2023, 2024, 2025]:
                folder = root / f'{dataset}_{year}'
                fold = json.loads((folder / 'fold.json').read_text())
                saved = np.load(folder / 'preprocessing.npz')
                train, val, test = fold_indices(data, year)
                for key, ids in [('train', train), ('validation', val), ('test', test)]:
                    np.testing.assert_array_equal(ids, saved[key])
                    assert (np.diff(ids) > 0).all()
                x, mean, std = scale_inputs(data, train)
                np.testing.assert_array_equal(mean, saved['mean'])
                np.testing.assert_array_equal(std, saved['std'])
                dates = panel.dates
                signal, outcome = dates[data['origin']], dates[data['exit']]
                assert outcome[train].max() < pd.Timestamp(year - 1, 1, 1)
                assert outcome[val].max() < pd.Timestamp(year, 1, 1)
                assert data['entry'][test[0]] == data['origin'][test[0]] + 1
                assert np.all(data['exit'][test] == data['entry'][test] + horizon)
                for label, ids in [('train', train), ('validation', val), ('test', test)]:
                    actual = [str(signal[ids[0]].date()), str(signal[ids[-1]].date())]
                    assert actual == fold[f'{label}_signal']
                assert str(outcome[train[-1]].date()) == fold['train_labels_end']
                assert str(outcome[val[-1]].date()) == fold['validation_labels_end']
                # Perturb every raw observation after the first test signal.
                # Earlier features, masks, liquidity, training labels, scaling
                # and relationship groups must remain identical.
                changed = copy.deepcopy(panel)
                cutoff = data['origin'][test[0]]
                rng = np.random.default_rng(year + horizon)
                size = changed.close[cutoff+1:].shape
                changed.close[cutoff+1:] *= rng.uniform(.2, 3., size)
                changed.volume[cutoff+1:] *= .01
                changed.benchmark[cutoff+1:] *= 2.
                altered = prepare_panel(changed, horizon=horizon)
                past = data['origin'] <= cutoff
                for key in ['x', 'mask', 'momentum', 'reversal']:
                    np.testing.assert_array_equal(data[key][past], altered[key][past])
                np.testing.assert_array_equal(data['adv'][:cutoff+1], altered['adv'][:cutoff+1])
                for key in ['y', 'risk', 'labelmask']:
                    np.testing.assert_array_equal(data[key][train], altered[key][train])
                    np.testing.assert_array_equal(data[key][val], altered[key][val])
                altered_x, altered_mean, altered_std = scale_inputs(altered, train)
                np.testing.assert_array_equal(mean, altered_mean)
                np.testing.assert_array_equal(std, altered_std)
                np.testing.assert_array_equal(x[past], altered_x[past])
                assert training_groups(data, train) == training_groups(altered, train)
                choice = lock[dataset]['by_year'][str(year)]
                expected = {kind: robust_score([r['selected']['score'] for r in policies
                            if r['dataset'] == dataset and r['model'] == kind and r['year'] <= year])
                            for kind in choice['validation_scores']}
                assert choice['validation_scores'] == expected
                assert choice['winner'] == max(expected, key=expected.get)
                rows.append(dict(horizon=horizon, dataset=dataset, test_year=year,
                                 train_start=fold['train_signal'][0], train_end=fold['train_signal'][1],
                                 train_labels_end=fold['train_labels_end'],
                                 validation_start=fold['validation_signal'][0], validation_end=fold['validation_signal'][1],
                                 validation_labels_end=fold['validation_labels_end'],
                                 test_start=fold['test_signal'][0], test_end=fold['test_signal'][1],
                                 training_origins=len(train), validation_origins=len(val), test_origins=len(test),
                                 saved_scaler_verified=True, future_perturbation_invariant=True,
                                 prior_validation_selection_verified=True))
                print(f'Chronology verified: horizon {horizon}, {dataset}, test {year}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'chronology_audit.csv', index=False)
    (ROOT / 'chronology_audit.json').write_text(json.dumps(dict(
        status='passed within the saved-data pipeline', audited_folds=len(rows), code_hash_checks=source_checks,
        first_training_signal=frame.train_start.min(),
        limitation='Does not certify point-in-time vendor data, missing delisting histories, survivor cohort selection, or independence of previously inspected tests.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    lines = ['# Training dates and leakage audit', '',
             '**Result:** the 16 cohort/horizon/year folds passed the chronology and preprocessing checks. '
             'No future-observation leakage was detected in these audited code paths. This is not a certification '
             'that the public source data are point-in-time or that the research history is free of test reuse.', '',
             '| Test year | Actual training-signal years | Validation year |', '|---|---|---|',
             '| 2022 | 2019–2020 | 2021 |', '| 2023 | 2019–2021 | 2022 |',
             '| 2024 | 2019–2022 | 2023 |', '| 2025 | 2020–2023 | 2024 |', '',
             'Raw prices start in 2018. A 260-session warmup moves the first model-training signal to '
             '**2019-01-15**. Models refit for each annual test. Earlier test years can become training '
             'or validation data for later years because they are then in the past. The 2025 fit never '
             'contributes predictions to the 2022 curve. NASDAQ and NYSE remain separate cohorts.', '',
             '## Exact saved signal dates', '',
             'Dates below apply to both cohorts. Labels mature after signal dates, as shown in the next table.', '',
             '| Horizon | Test year | Training signals | Validation signals | Test signals |', '|---|---|---|---|---|']
    for _, r in frame[frame.dataset == 'NYSE_recent'].iterrows():
        lines.append(f'| {r.horizon} | {r.test_year} | {r.train_start} to {r.train_end} | '
                     f'{r.validation_start} to {r.validation_end} | {r.test_start} to {r.test_end} |')
    lines += ['', '| Horizon | Test year | Last training outcome | First validation signal | Last validation outcome | First test signal |',
              '|---|---|---|---|---|---|']
    for _, r in frame[frame.dataset == 'NYSE_recent'].iterrows():
        lines.append(f'| {r.horizon} | {r.test_year} | {r.train_labels_end} | {r.validation_start} | '
                     f'{r.validation_labels_end} | {r.test_start} |')
    lines += ['', '## Checks performed', '',
              '- Reconstructed train/validation/test indices exactly match saved preprocessing files.',
              '- Training labels mature before validation; validation labels mature before testing.',
              '- Training-only feature means and standard deviations exactly match the saved scalers.',
              '- Changing all future prices, volume and the benchmark leaves earlier inputs, eligibility, liquidity, '
              'training/validation labels, scalers and correlation groups unchanged.',
              '- Saved portfolio winners equal the best recorded validation candidate. Architecture selection uses '
              'only validation scores from the current or earlier folds.',
              '- Frozen training-source hashes match. Inputs remain chronological, with next-close execution and '
              'one- or five-session holding periods.', '',
              '## Remaining limitations', '',
              'Yahoo adjusted histories can reflect later corporate-action revisions. The fixed research cohorts '
              'and unavailable delisted securities create selection/survivorship bias. Correlation groups use all '
              'available price history through the last training signal, including older history before a rolling '
              'training window. That is historical information, not future information. '
              'The 2024–2025 tests were previously inspected and the wider period was chosen retrospectively. '
              'Thus the results are exploratory. Pooled curves concatenate separate annual accounts, reset sizing '
              'each year and omit boundary sessions.', '',
              '[Machine-readable fold audit](chronology_audit.csv) | [Audit status](chronology_audit.json)', '']
    (ROOT / 'DATA_AND_SPLITS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
