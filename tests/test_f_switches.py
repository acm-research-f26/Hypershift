"""Phase 1.5 F: learnability switches keep defaults unchanged and behave as documented."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import known_signal as ks  # noqa: E402
from conftest import make_synthetic_market  # noqa: E402
from hypershift.config import RunConfig  # noqa: E402
from hypershift.data.hypergraph import Hypergraph  # noqa: E402
from hypershift.models.think import THINK  # noqa: E402
from hypershift.train.loop import input_transform, apply_input_mode, gather_batch, window_offsets  # noqa: E402


def _tiny():
    real = make_synthetic_market(N=40, T=260, valid_index=150, test_index=210)
    return real, Hypergraph(40, tuple(tuple(range(i, i + 5)) for i in range(0, 40, 5)))


def test_defaults_unchanged():
    c = RunConfig()
    assert (c.input_std, c.input_scale, c.init_gain, c.head_scale, c.spatial_residual, c.decoupled_wd, c.log_ic) == \
           (False, 1.0, 1.0, 0.0, False, False, False)
    real, _ = _tiny()
    x, *_ = gather_batch(real, window_offsets(real, 16, "train")[:3], 16)
    np.testing.assert_array_equal(input_transform(c, real)(x), apply_input_mode(x, "level"))
    a, b = THINK(hidden=8), THINK(hidden=8, init_gain=1.0, head_scale=0.0, spatial_residual=False)
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    assert not any("head" in n for n, _ in a.named_parameters())


@pytest.mark.parametrize("spatial,temporal", [("hyp", "hyp"), ("euc", "euc"), ("hyp", "euc")])
def test_residual_and_head_scale_forward_backward(spatial, temporal):
    real, hg = _tiny()
    thg = hg.to_torch(torch.device("cpu"))
    m = THINK(in_dim=5, hidden=8, seq=16, kernel=4, spatial=spatial, temporal=temporal,
              spatial_residual=True, head_scale=50.0, init_gain=3.0)
    x = torch.rand(2, 40, 16, 5) * 0.1
    y = m(x, thg)
    assert y.shape == (2, 40) and torch.isfinite(y).all()
    y.sum().backward()
    assert m.log_head_scale.grad is not None and torch.isfinite(m.log_head_scale.grad)
    assert m.log_head_scale.exp().item() == pytest.approx(50.0, rel=1e-5)


def test_residual_keeps_self_signal():
    """Without a self path the node output ignores u when it has edges; with residual the output depends on u."""
    real, hg = _tiny()
    thg = hg.to_torch(torch.device("cpu"))
    torch.manual_seed(0)
    base = THINK(in_dim=5, hidden=8, spatial="euc", temporal="euc", spatial_residual=True)
    u = torch.randn(1, 4, 40, 8) * 0.1
    out = base.spatial(u, thg)
    assert torch.all(out - u >= -1e-6)      # u + relu(agg) >= u


def test_init_gain_scales_z():
    torch.manual_seed(0)
    a = THINK(hidden=8, init_gain=1.0)
    torch.manual_seed(0)
    b = THINK(hidden=8, init_gain=4.0)
    assert (b.tconv1.fc.z.norm() / a.tconv1.fc.z.norm()).item() == pytest.approx(4.0, rel=1e-5)


def test_input_std_uses_train_stats_only():
    real, _ = _tiny()
    c = RunConfig(input_mode="relative", input_std=True, input_scale=0.5)
    f = input_transform(c, real)
    x, m, *_ = gather_batch(real, window_offsets(real, 16, "train")[:64], 16)
    z = f(x)
    assert np.isfinite(z).all() and 0.2 < z[m > 0.5].std() < 1.0
    # changing the future (test period) must not change the stats
    real2 = make_synthetic_market(N=40, T=260, valid_index=150, test_index=210)
    real2.features[:, 210:] *= 2.0
    z2 = input_transform(c, real2)(x)
    np.testing.assert_allclose(z, z2, rtol=1e-6)


def test_grad_clip_zero_means_off_and_log_ic(tmp_path):
    real, hg = _tiny()
    ks.LEVELS["_t"] = (0.3, 0.5, 0.01)
    r = ks.run_cell(real, hg, "_t", "EE_hyper", "relative", 0, epochs=2, out_root=str(tmp_path), seq=8, kernel=2,
                    hidden=8, batch_days=4, patience=50, grad_clip=0.0, log_ic=True, decoupled_wd=True,
                    input_std=True, input_scale=0.5)
    hist = [__import__("json").loads(l) for l in (Path(tmp_path) / "known_signal__t_relative/EE_hyper/seed_0/history.jsonl").read_text().splitlines()]
    assert len(hist) == 2 and all({"val_ic", "test_ic", "test_pred_sd"} <= set(h) for h in hist)
    assert np.isfinite(r["test_ic"])


def test_weight_decay_collapses_hyperbolic_output(tmp_path):
    """The F root cause: coupled L2 at 5e-4 dominates the ~1e-5 loss gradient and shrinks |z| (and the output) to ~0."""
    real, hg = _tiny()
    ks.LEVELS["_t"] = (0.3, 0.5, 0.01)
    sd = {}
    for wd in (5e-4, 0.0):
        r = ks.run_cell(real, hg, "_t", "HH_hyper", "relative", 0, epochs=10, exp=f"wd{wd}", out_root=str(tmp_path),
                        seq=8, kernel=2, hidden=8, lr=1e-3, batch_days=1, patience=50, weight_decay=wd)
        sd[wd] = r["pred_std"]
    assert sd[5e-4] < 0.2 * sd[0.0], sd
