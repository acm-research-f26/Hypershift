"""Independent artifact/accounting checks and descriptive regime diagnostics.

Reads completed runs only. Does not retrain, select policies or change predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_engine import load_panel, prepare_panel, fold_indices, performance
from run_architecture_research import robust_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('runs/architecture_research'))
    a = parser.parse_args()
    f = pd.read_csv(a.run/'all_metrics.csv')
    protocol = json.loads((a.run/'PROTOCOL.json').read_text())
    input_lock = a.run/'DATA_INPUT_LOCK.json'
    diagnostic_lock = a.run/'DIAGNOSTIC_LOCK.json'
    if diagnostic_lock.exists():
        locked = json.loads(diagnostic_lock.read_text())
        assert hashlib.sha256(Path('analyze_recent_research.py').read_bytes()).hexdigest() == locked['analysis_sha256']
    if input_lock.exists():
        for source, expected in json.loads(input_lock.read_text()).items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, source
        manifest_path = Path('data/recent_cohorts/manifest.json')
        manifest = json.loads(manifest_path.read_text())
        for source, expected in manifest['hashes'].items():
            assert hashlib.sha256((manifest_path.parent/source).read_bytes()).hexdigest() == expected, source
    snapshot = a.run/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for source, expected in protocol['code_hashes'].items():
        saved = snapshot/Path(source).name
        if not saved.exists():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected
            saved.write_bytes(Path(source).read_bytes())
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == expected
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, 'Audit with the matching source snapshot: '+source
    locks = json.loads((a.run/'SELECTION_LOCK.json').read_text())
    policies = json.loads((a.run/'validation_search.json').read_text())
    for row in policies:
        assert row['selected'] == max(row['candidates'], key=lambda r:r['score'])
    for name, selection in locks.items():
        for year, choice in selection.get('by_year', {}).items():
            for kind, score in choice['validation_scores'].items():
                expected = robust_score([row['selected']['score'] for row in policies
                                         if row['dataset']==name and row['model']==kind and row['year']<=int(year)])
                assert np.isclose(score, expected), 'Future validation leaked into selection'
            assert choice['winner'] == max(choice['validation_scores'], key=choice['validation_scores'].get)
    primary = f[(f.seed=='ensemble') & f.selected_policy & (f.cost_multiplier==1)]
    regimes, audit = [], []
    for name in primary.dataset.unique():
        data = prepare_panel(load_panel(name))
        p = data['panel']
        recorded = json.loads((a.run/f'{name}_data.json').read_text())
        assert hashlib.sha256(p.close.tobytes()+p.volume.tobytes()).hexdigest() == recorded['sha256_close_and_dollar_volume']
        b = pd.Series(p.benchmark, index=p.dates)
        br = b.pct_change(fill_method=None)
        vol = br.rolling(20).std().to_numpy()
        trend = (b/b.shift(126)-1).to_numpy()
        for year in sorted(primary[primary.dataset==name].year.unique()):
            train,val,test = fold_indices(data, int(year))
            end = data['origin'][train[-1]]
            vol_threshold = np.nanmedian(vol[:end+1])
            signals = data['origin'][test]
            tag = np.array([('rising' if trend[t]>=0 else 'falling')+'_'+('high_vol' if vol[t]>=vol_threshold else 'low_vol') for t in signals])
            for _,row in primary[(primary.dataset==name)&(primary.year==year)].iterrows():
                path = a.run/f'{name}_{year}'/f'{row.model}_daily.csv'
                daily = pd.read_csv(path)
                # Independent daily-series recomputation, not comparison to a copied metric.
                for prefix in ['gross','net']:
                    result = performance(daily[prefix])
                    for key in ['sharpe','annualized_return','cumulative_return','annualized_volatility','maximum_drawdown']:
                        assert np.isclose(result[key], row[f'{prefix}_{key}'], atol=1e-9), (name,year,row.model,key)
                assert len(daily)==len(test)*5
                assert pd.to_datetime(daily.date).is_monotonic_increasing
                assert not pd.to_datetime(daily.date).duplicated().any()
                audit.append(dict(dataset=name,year=int(year),model=row.model,days=len(daily),status='verified'))
                rankfile = a.run/f'{name}_{year}'/f'{row.model}_ranking.csv'
                ranking = pd.read_csv(rankfile) if rankfile.exists() else None
                for regime in sorted(set(tag)):
                    mask = tag==regime
                    rd = daily.net.to_numpy()[np.repeat(mask,5)]
                    result = performance(rd)
                    regimes.append(dict(dataset=name,year=int(year),model=row.model,regime=regime,
                                        periods=int(mask.sum()),sessions=len(rd),
                                        conditional_net_sharpe=result['sharpe'],
                                        IC=float(ranking.IC[mask].mean()) if ranking is not None else None,
                                        RankIC=float(ranking.RankIC[mask].mean()) if ranking is not None else None,
                                        note='Conditional noncontiguous subsample; not a tradable strategy. Regime known at signal time.'))
    pd.DataFrame(regimes).to_csv(a.run/'regime_diagnostics.csv',index=False)
    pd.DataFrame(audit).to_csv(a.run/'accounting_audit.csv',index=False)
    files = [*Path('data/research_panel').glob('*.csv'), *Path('data/research_cohorts').glob('*.csv'),
             *Path('data/recent_cohorts').glob('*.csv'), *Path('data/recent_cohorts').glob('*.json'),
             Path('data/research_cohorts/selection.json'),Path('data/research_cohorts/sources.json'),
             Path('data/research_cohorts/corrections.json')]
    manifest = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (a.run/'additional_source_hashes.json').write_text(json.dumps(manifest,indent=2))
    extras = ['audit_architecture_research.py', 'analyze_recent_research.py', 'report_recent_research.py',
              'report_architecture_research.py', 'plot_research_metrics.py', 'test_research.py',
              'config/recent_identity_exclusions.json']
    hashes = {}
    for name in extras:
        source = Path(name)
        if source.exists():
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            (snapshot/source.name).write_bytes(source.read_bytes())
    (a.run/'report_source_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(f'Verified {len(audit)} saved daily portfolio paths; saved {len(regimes)} regime diagnostics.')


if __name__=='__main__':
    main()
