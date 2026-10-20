import sys

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')

from screen_analysis import bootstrap_means, boot_p, verdict, family_arms  # noqa: E402
from hypershift.eval.stats import holm  # noqa: E402


def test_bootstrap_p_known_series():
    ps = []
    for seed in range(20):                           # under the null the p-values are roughly uniform
        x = np.random.default_rng(100 + seed).standard_normal(250)
        ps.append(boot_p(x.mean(), bootstrap_means(x, 10, 1000, np.random.default_rng(0))))
    assert 0.25 < np.mean(ps) < 0.75
    noise = np.random.default_rng(1).standard_normal(250)
    strong = noise * 0.2 + 1.0                       # mean 1, se ~0.013
    b1 = bootstrap_means(strong, 10, 4000, np.random.default_rng(0))
    assert boot_p(strong.mean(), b1) < 0.01
    assert abs(b1.mean() - strong.mean()) < 0.02
    assert boot_p(0.0, b1) == 1.0                    # m=0: every |m*-m| >= 0
    assert boot_p(1.0, np.full(50, 1.0)) == 0.0      # constant series: no spread, p=0


def test_verdict_gate():
    ok = {2015: [.01, .02, -.001], 2016: [.01, .004, .02]}
    assert verdict(ok, 0.01) == 'Promising'
    assert verdict(ok, 0.002) == 'Neutral'                       # pooled below 0.003
    assert verdict({2015: [.01, -.02, -.001], 2016: [.01, .004, .02]}, 0.01) == 'Neutral'   # <2/3 seeds positive
    assert verdict({2015: [-.01, -.02, -.001], 2016: [.01, .004, .02]}, 0.01) == 'Neutral'  # fold mean negative
    assert verdict({2015: [-.01, -.02, -.03], 2016: [-.01, -.02, -.03]}, -0.02) == 'Negative preliminary result'
    assert verdict({2015: [.01, .02, .01]}, 0.01) == 'Inconclusive'                          # one fold
    assert verdict({2015: [.01, .02], 2016: [.01, .02]}, 0.01) == 'Inconclusive'             # two seeds


def test_holm_family():
    fam = family_arms({'sector_target', 'market_target'})
    assert fam['B'] == 'sector_target' and len(fam) == 9 and fam['E'] == 'gated'
    assert family_arms({'market_target'})['B'] == 'market_target'
    h = holm({a: p for a, p in zip(fam.values(), [.001, .02, .5, 1, 1, 1, 1, 1, 1])})
    assert abs(h['residual_soft'] - .009) < 1e-9 and abs(h['sector_target'] - .16) < 1e-9
