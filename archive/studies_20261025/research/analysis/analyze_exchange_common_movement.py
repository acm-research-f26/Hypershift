"""Training-only return co-movement on the same complete-history geometry stocks."""
import numpy as np
import pandas as pd
from research_engine import load_panel, prepare_panel, fold_indices

rows = []
for dataset in ['NYSE_top100_20261010', 'NASDAQ100_20261010']:
    data = prepare_panel(load_panel(dataset))
    for year in range(2022, 2026):
        train = fold_indices(data, year)[0]
        keep = data['mask'][train].all(0)
        returns = data['x'][train][:, :, keep, 0].reshape(-1, int(keep.sum()))
        correlation = np.corrcoef(returns, rowvar=False)
        pairs = correlation[np.triu_indices(len(correlation), 1)]
        rows.append(dict(dataset=dataset, test_year=year, stocks=int(keep.sum()),
                         median_pairwise_return_correlation=float(np.median(pairs)),
                         mean_pairwise_return_correlation=float(pairs.mean()),
                         pc1_correlation_variance_fraction=float(np.linalg.eigvalsh(correlation)[-1]/len(correlation))))
pd.DataFrame(rows).to_csv('runs/hyperbolicity_20261010_exact/common_movement_diagnostics.csv', index=False)
