"""Fast end-to-end check: the pipeline learns a planted signal (scripts/known_signal.py, tiny market)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import known_signal as ks  # noqa: E402
from conftest import make_synthetic_market  # noqa: E402
from hypershift.data.hypergraph import Hypergraph  # noqa: E402


def _tiny():
    real = make_synthetic_market(N=40, T=260, valid_index=150, test_index=210)
    edges = tuple(tuple(range(i, i + 5)) for i in range(0, 40, 5))
    return real, Hypergraph(40, edges)


def test_plant_signal_is_learnable_by_oracle():
    real, hg = _tiny()
    ks.LEVELS["_t"] = (0.3, 0.5, 0.01)
    syn, A = ks.plant_signal(real, hg, 0.3, 0.5, 0.01, seed=0)
    ref = ks.reference_metrics(syn, A, 0.3, 0.5)
    assert ref["oracle"]["ic"] > 0.25 and ref["oracle"]["ic"] > ref["own_only"]["ic"] > 0.05
    assert np.allclose(A.sum(1), 1.0, atol=1e-5)
    # no look-ahead: the oracle uses only day t-1 returns; gt of the test day is the target
    assert syn.gt.shape == real.gt.shape and (syn.mask == real.mask).all()


def test_pipeline_recovers_planted_signal(tmp_path):
    real, hg = _tiny()
    ks.LEVELS["_t"] = (0.3, 0.5, 0.01)
    for arm, mode in (("EE_hyper", "relative"), ("HH_hyper", "relative")):
        r = ks.run_cell(real, hg, "_t", arm, mode, 0, epochs=8, out_root=str(tmp_path), seq=8, kernel=2, hidden=8,
                        lr=5e-3, batch_days=4, patience=50)
        assert r["test_ic"] > 0.1, (arm, r["test_ic"])
