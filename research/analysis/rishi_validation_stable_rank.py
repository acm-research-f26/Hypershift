"""Compare validation and test per-security average saved percentile ranks."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import run_ndx_drawdown_research as study


def ranks(predictions, mask):
    ranks = np.full(mask.shape, np.nan)
    for j in range(len(mask)):
        assert mask[j].any() and np.isfinite(predictions[j, mask[j]]).all()
        ranks[j, mask[j]] = rankdata(predictions[j, mask[j]], method='average') / mask[j].sum()
    return ranks


def main():
    out = Path('docs/rishi_mechanism_20261027/execution_batch46_validation_rank')
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    data = study.load_data(); paths = {'B': Path('runs/ndx_extended_hours_20261024/after_hours_premarket'),
                                      'D': Path('runs/ndx_reference_networks_20261024/dynamic_restricted')}
    rows = []; hashes = {}
    for year in (2022, 2023, 2024, 2025):
        _, val, test = study.fold(data, year)
        assert data['exit'][val].max() < data['origin'][test].min()
        for group, seeds in study.GROUPS.items():
            scores = {}
            for family, folder in paths.items():
                per = {'validation': [], 'test': []}
                for seed in seeds:
                    path = folder / str(year) / f'{seed}_predictions.npz'
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path, allow_pickle=False) as z:
                        for split, ids in [('validation', val), ('test', test)]:
                            prediction = z[split].astype(float); mask = data['mask'][ids]
                            assert prediction.shape == mask.shape
                            per[split].append(ranks(prediction, mask))
                scores[family] = {split: np.mean(arrays, axis=0) for split, arrays in per.items()}
            scores['E'] = {split: (scores['B'][split]+scores['D'][split])/2 for split in ('validation','test')}
            for family in ('B','D','E'):
                means = []; support = []
                for split in ('validation','test'):
                    a = scores[family][split]; n = np.sum(np.isfinite(a), axis=0)
                    means.append(np.divide(np.nansum(a, axis=0), n, out=np.full(a.shape[1], np.nan), where=n>0))
                    support.append(n)
                common = (support[0]>=5)&(support[1]>=5)
                a, b = means[0][common], means[1][common]
                corr = float(spearmanr(a,b).statistic) if len(a)>=3 and np.ptp(a)>0 and np.ptp(b)>0 else np.nan
                rows.append(dict(year=year,group=group,family=family,matched=int(common.sum()),
                                 validation_dates=len(val),test_dates=len(test),spearman=corr,
                                 median_absolute_rank_difference=float(np.median(np.abs(a-b))) if len(a) else np.nan))
    frame = pd.DataFrame(rows); assert len(frame)==36
    frame.to_csv(out/'CELLS.csv',index=False)
    (out/'INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(frame.to_string(index=False)); print('files hashed',len(hashes))


if __name__ == '__main__': main()
