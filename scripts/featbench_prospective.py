"""Study FB prospective scorer (run only after enough post-freeze data exists; see PROSPECTIVE_PROTOCOL.md).

  python scripts/featbench_prospective.py --panel <extended panel npz> --ohlcv <extended ohlcv npz> \
      --manifest docs/featbench_20261022/FREEZE_MANIFEST.json --out results/FEATBENCH_PROSPECTIVE
The extended files must have the same format as data/fresh/sp500_daily/{screen_sp500_2015_2026,ohlcv_sp500_2015_2026}.npz.
Signals on or before 2026-10-22 are never scored (hard guard in hypershift.featbench.prospective).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
from screen_analysis import bootstrap_means, boot_p  # noqa: E402
from hypershift.featbench.prospective import load_range, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--ohlcv', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    fb = load_range(a.panel, a.ohlcv, man['horizon'])
    n = len(fb.evaluate)
    print('post-freeze scored signals:', n, 'minimum for the primary test:', man['min_days'])
    out = score(fb, man)
    res = {'days': n, 'first': str(fb.dates[fb.evaluate[0]]) if n else None, 'interim': n < man['min_days'], 'arms': {}}
    for name, v in out.items():
        s = v['neutral_RankIC']
        b = bootstrap_means(s, 10, 5000, np.random.default_rng(1))
        res['arms'][name] = {'neutral_RankIC': float(s.mean()), 'ci': list(np.quantile(b, [.025, .975])), 'raw_RankIC': v['raw']}
    a_, b_ = man['primary'][0], man['primary'][1]
    d = out[a_]['neutral_RankIC'] - out[b_]['neutral_RankIC']
    bt = bootstrap_means(d, 10, 5000, np.random.default_rng(2))
    res['primary'] = {'contrast': f'{a_} - {b_}', 'mean': float(d.mean()), 'ci': list(np.quantile(bt, [.025, .975])),
                      'p': boot_p(float(d.mean()), bt)}
    res['primary']['verdict'] = ('INTERIM (descriptive)' if res['interim'] else
                                 'SUPPORTED' if res['primary']['p'] < .05 and (res['primary']['ci'][0] > 0 or res['primary']['ci'][1] < 0)
                                 else 'NO EVIDENCE')
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"prospective_{res['days']}d.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
