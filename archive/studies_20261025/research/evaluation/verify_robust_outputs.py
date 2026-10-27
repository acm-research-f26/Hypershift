"""Read-only verification of completed experiment outputs and chart coverage."""
import hashlib
import json
import re
from pathlib import Path
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('runs/robust_comparison')
    locked = json.loads((root / 'comparison_input_hashes.json').read_text())
    failures = [p for p, h in locked.items() if sha(p) != h]
    assert not failures, failures
    analysis = json.loads(Path('config/robust_diagnostics_lock.json').read_text())
    assert sha('report_robust_research.py') == analysis['analysis_sha256']
    for p, h in json.loads(Path('config/robust_presentation_lock.json').read_text()).items():
        assert sha(p) == h, p
    charts = (root / 'CHARTS.md').read_text(encoding='utf-8')
    images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', charts)
    assert all((root / p).is_file() for p in images)
    annual = pd.read_csv(root / 'annual_comparison.csv')
    inventory = pd.DataFrame(json.loads((root / 'etf_control_inventory.json').read_text()))
    keys = ['horizon', 'dataset', 'year']
    for key, rows in annual.groupby(keys):
        panel = inventory
        for col, val in zip(keys, key):
            panel = panel[panel[col] == val]
        expected = set(rows.model.replace({'market_proxy': 'spy'}).str.lower()) | {'qqq'}
        assert set(panel.series.str.lower()) == expected, (key, panel.series.tolist())
    audit = pd.read_csv(root / 'accounting_audit.csv')
    assert len(audit) == 288 and set(audit.status) == {'verified'}
    counts = {}
    for name in ['robust_2022_2025_final', 'daily_2022_2025_final']:
        run = Path('runs') / name
        completion = json.loads((run / 'completion.json').read_text())
        counts[name] = completion['neural_fits']
        for source in (run / 'source_snapshot').glob('*.py'):
            assert sha(source) == sha(source.name), source.name
    result = dict(status='verified', locked_input_files=len(locked), neural_configurations=counts,
                  primary_paths_audited=len(audit), primary_and_etf_curves=len(inventory),
                  embedded_images=len(images), missing_images=0,
                  training_and_predeclared_analysis_sources_unchanged=True)
    (root / 'FINAL_VERIFICATION.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
