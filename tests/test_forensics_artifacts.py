import hashlib
from pathlib import Path
import numpy as np
import pytest
from hypershift.eval import forensics as F

RUN = Path("results/R5_f2_alpha0_train/HH")
pytestmark = [pytest.mark.data, pytest.mark.skipif(not RUN.exists(), reason="R5_f2 artifacts not present")]


@pytest.mark.parametrize("seed", range(5))
def test_saved_daily_series_and_metrics_reproduce_readonly(seed):
    p = RUN / f"seed_{seed}"
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir()}
    a = F.load_run("R5_f2_alpha0_train", "HH", seed)
    r, _ = F.portfolio(a.pred, a.gt, a.mask)
    # saved arrays are float32 and loop.py averaged float32 gt, so agreement is to float32 precision, not 1e-12
    np.testing.assert_allclose(r, a.daily, atol=1e-7)
    assert a.pred.shape == (1737, 237)
    assert F.perf(r)["sr"] == pytest.approx(a.metrics["test"]["sr"], abs=1e-5)
    # the saved arrays really are from the selected epoch
    assert a.history[a.metrics["best_epoch"]]["test"]["sr"] == pytest.approx(F.perf(r)["sr"], abs=1e-5)
    assert a.config["weight_decay"] == 0.0 and a.config["input_mode"] == "relative" and a.config["alpha"] == 0.0
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir()}
    assert before == after
