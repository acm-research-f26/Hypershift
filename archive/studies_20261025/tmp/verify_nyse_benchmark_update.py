"""Verify the approved passive-basket replacement against untouched returns."""
from pathlib import Path
import sys
import hashlib
import json
from zipfile import ZipFile, ZIP_DEFLATED
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from research_engine import performance
from report_pooled_uncertainty import indices, bootstrap_sharpe, interval


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    checked = 0
    roots = [Path('runs/robust_comparison'), Path('runs/largecap_20261010/comparison')]
    for root in roots:
        curves = pd.read_csv(root / 'combined_2022_2025_curves.csv')
        stats = pd.read_csv(root / 'combined_2022_2025_metrics.csv')
        for (h, d), group in stats[stats.dataset.str.startswith('NYSE')].groupby(['horizon', 'dataset']):
            p = curves[(curves.horizon == h) & (curves.dataset == d)]
            bench = p[p.model == 'equal_universe']
            ids = indices(bench.groupby('fold_year', sort=True).size().tolist())
            sb = performance(bench.net)['sharpe']
            samples = bootstrap_sharpe(bench.net, ids)
            assert not group.model.eq('qqq').any()
            assert group.benchmark_model.eq('equal_universe').all()
            for r in group.itertuples():
                path = p[p.model == r.model]
                np.testing.assert_array_equal(path.date, bench.date)
                s = performance(path.net)['sharpe']
                np.testing.assert_allclose([r.net_sharpe, r.benchmark_sharpe, r.delta_benchmark], [s, sb, s - sb], atol=1e-12)
                expected = interval(bootstrap_sharpe(path.net, ids) - samples)
                np.testing.assert_allclose([r.delta_benchmark_ci_low, r.delta_benchmark_ci_high], expected, atol=1e-12)
                checked += 1
        for p in root.glob('*.csv'):
            if p.name in ['combined_2022_2025_curves.csv', 'combined_2022_2025_curves_current.csv', 'huber_ranking_curves.csv']:
                continue  # frozen dated return records retain the archived QQQ series
            f = pd.read_csv(p)
            if 'benchmark_model' not in f:
                continue
            ny = f[f.dataset.str.startswith('NYSE')]
            assert ny.benchmark_model.eq('equal_universe').all(), p
            assert not any('qqq' in c for c in f.columns), p
            if 'model' in f:
                assert not ny.model.eq('qqq').any(), p
            if 'symbol' in f:
                assert not ny.symbol.eq('QQQ').any(), p
            copy = Path('docs') / p.name
            if root == roots[0] and copy.exists():
                assert copy.read_bytes() == p.read_bytes(), p
    # Publication checksums reflect the revised artifacts without republishing
    # unrelated source files over the user's existing documentation changes.
    manifest = Path('docs/publication_manifest.json')
    m = json.loads(manifest.read_text(encoding='utf-8'))
    for item in m.get('files', []):
        for field, target in [('source_sha256', item.get('source')), ('published_sha256', item.get('published'))]:
            if target and Path(target).exists():
                item[field] = sha(Path(target))
    m['nyse_benchmark_update'] = 'Passive equal-weight eligible cohort; Nasdaq retains QQQ; tables only'
    manifest.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding='utf-8')
    package = Path('docs/largecap_retraining_20261010/Hypershift_Large_Company_Results_Data.zip')
    with ZipFile(package) as z:
        contents = [(i, z.read(i.filename)) for i in z.infolist()]
    pending = package.with_suffix('.pending.zip')
    with ZipFile(pending, 'w', ZIP_DEFLATED) as z:
        for info, body in contents:
            source = package.parent / info.filename if info.filename == 'primary_results.csv' else roots[1] / info.filename
            z.writestr(info, source.read_bytes() if source.exists() else body)
    pending.replace(package)
    with ZipFile(package) as z:
        assert z.testzip() is None
        for name in ['primary_results.csv', 'combined_2022_2025_metrics.csv', 'procedure_confidence_intervals.csv']:
            source = package.parent / name if name == 'primary_results.csv' else roots[1] / name
            assert z.read(name) == source.read_bytes()
    audit = Path('docs/audit/nyse_benchmark_update.json')
    a = json.loads(audit.read_text(encoding='utf-8'))
    for name, expected in a['inputs'].items():
        if name.endswith('_curves.csv') or name.endswith('SELECTION_LOCK.json'):
            assert sha(Path(name)) == expected, f'Frozen source changed: {name}'
    for item in a['updated_files']:
        item['after_sha256'] = sha(Path(item['path']))
    a['verification'] = {'status': 'passed', 'pooled_nyse_comparisons_recomputed': checked,
                         'published_csvs_match_sources': True, 'download_archive_matches_tables': True,
                         'frozen_returns_and_selection_locks_unchanged': True}
    a['table_artifacts'] = {str(p): sha(p) for root in [Path('docs'), *roots] for p in root.glob('*.md')
                            if '<!-- nyse-benchmark:start -->' in p.read_text(encoding='utf-8')}
    audit.write_text(json.dumps(a, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(a['verification'], indent=2))


if __name__ == '__main__':
    main()
