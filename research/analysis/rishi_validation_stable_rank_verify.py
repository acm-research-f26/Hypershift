"""Independently rebuild validation/test rank summaries with pandas ranking."""
import numpy as np
import pandas as pd
from pathlib import Path
import run_ndx_drawdown_research as study


def main():
    data = study.load_data(); saved = pd.read_csv('docs/rishi_mechanism_20261027/execution_batch46_validation_rank/CELLS.csv')
    root = {'B':Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),
            'D':Path('runs/ndx_reference_networks_20261024/dynamic_restricted')}
    errors = []; cells = 0
    for year in (2022, 2023, 2024, 2025):
        _, val, test = study.fold(data, year)
        for group, seeds in study.GROUPS.items():
            values = {}
            for family, path in root.items():
                a = {'validation': [], 'test': []}
                for seed in seeds:
                    with np.load(path / str(year) / f'{seed}_predictions.npz', allow_pickle=False) as z:
                        for split, ids in [('validation',val),('test',test)]:
                            p = pd.DataFrame(z[split]).where(data['mask'][ids])
                            a[split].append(p.rank(axis=1, method='average', pct=True).to_numpy())
                values[family] = {k: np.mean(v,axis=0) for k,v in a.items()}
            values['E'] = {k: (values['B'][k]+values['D'][k])/2 for k in ('validation','test')}
            for family in ('B','D','E'):
                v = pd.DataFrame(values[family]['validation']); t = pd.DataFrame(values[family]['test'])
                good = (v.count()>=5)&(t.count()>=5)
                x = v.mean()[good]; y = t.mean()[good]
                row = saved[(saved.year==year)&(saved.group==group)&(saved.family==family)].iloc[0]
                assert row.matched == len(x)
                errors += [abs(row.spearman-x.corr(y,method='spearman')),
                           abs(row.median_absolute_rank_difference-np.median(abs(x-y)))]
                cells += 1
    assert cells==36 and max(errors)<1e-12
    print(f'{cells} cells verified with pandas; max error {max(errors):.3g}')


if __name__ == '__main__': main()
