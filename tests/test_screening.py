import numpy as np
import pytest
import torch


def test_five_session_target_and_path_mask():
    from hypershift.screening.data import forward_labels
    p = np.array([[1., 2., 3., 4., 5., 6., 7.]])
    m = np.ones_like(p, bool)
    y, ok = forward_labels(p, m, 5)
    assert y[0, 0] == pytest.approx(5.)
    assert y[0, 1] == pytest.approx(2.5)
    assert not ok[0, 2:].any()
    m[0, 3] = False
    assert not forward_labels(p, m, 5)[1].any()


def test_split_labels_mature_before_boundary():
    from hypershift.screening.data import split_signals
    train, val, test = split_signals(np.arange('2018-01-01', '2021-01-01', dtype='datetime64[D]'), 2020)
    dates = np.arange('2018-01-01', '2021-01-01', dtype='datetime64[D]')
    assert (dates[train + 5] < np.datetime64('2019-01-01')).all()
    assert (dates[val + 5] < np.datetime64('2020-01-01')).all()
    assert (dates[test + 5] < np.datetime64('2021-01-01')).all()
    assert train.min() >= 63


def test_ndcg_random_expectation_and_perfect_order():
    from hypershift.screening.metrics import rank_metrics
    y = np.arange(30.)[None, :]
    r, daily = rank_metrics(y, y, np.ones_like(y, bool))
    assert r['RankIC'] == pytest.approx(1.)
    assert r['NDCG10'] == pytest.approx(1.)
    assert 0 < r['random_NDCG10'] < 1
    assert daily['RankIC'].shape == (1,)


def test_tied_predictions_have_zero_rank_information():
    from hypershift.screening.metrics import rank_metrics
    y = np.arange(30.)[None, :]
    r, _ = rank_metrics(np.ones_like(y), y, np.ones_like(y, bool))
    assert r['RankIC'] == 0
    assert r['IC'] == 0


@pytest.mark.parametrize('kind', ['static', 'hyp', 'none', 'soft', 'residual_soft', 'rolling', 'two', 'two_residual', 'multiscale', 'mix', 'gated', 'lowrank', 'long', 'summary', 'pretrain', 'adaptive', 'master', 'mlp'])
def test_models_mask_ineligible_inputs_and_backpropagate(kind):
    from hypershift.screening.models import ScreenModel
    torch.manual_seed(7)
    model = ScreenModel(kind, nodes=24, hidden=8)
    x = torch.randn(2, 24, 64, 5) * .02
    eligible = torch.ones(2, 24, dtype=torch.bool)
    eligible[:, -1] = False
    pred = model(x, eligible)
    x[:, -1] = 1000
    assert torch.allclose(pred[:, :-1], model(x, eligible)[:, :-1], atol=1e-6)
    assert pred.shape == (2, 24)
    pred[:, :-1].square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_listwise_loss_learns_known_order():
    from hypershift.screening.models import objective
    p = torch.zeros(2, 30, requires_grad=True)
    y = torch.linspace(-1, 1, 30).repeat(2, 1)
    m = torch.ones_like(p, dtype=torch.bool)
    opt = torch.optim.Adam([p], lr=.1)
    first = objective(p, y, m, 'listwise').item()
    for _ in range(50):
        opt.zero_grad(); loss = objective(p, y, m, 'listwise'); loss.backward(); opt.step()
    assert loss.item() < first
    assert torch.corrcoef(torch.stack([p[0], y[0]]))[0, 1] > .95


def test_warmup_dates_with_no_usable_labels_are_removed():
    from hypershift.screening.data import usable_signals
    mask = np.zeros((50, 10), bool)
    mask[:, 4:] = True
    assert usable_signals(np.arange(10), mask).tolist() == list(range(4, 10))


def test_listwise_empty_date_does_not_poison_gradients():
    from hypershift.screening.models import objective
    p = torch.zeros(2, 30, requires_grad=True)
    y = torch.arange(30.).repeat(2, 1)
    m = torch.ones_like(p, dtype=torch.bool); m[0] = False
    objective(p, y, m, 'listwise').backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad[0].abs().sum() == 0


def test_repository_temporal_model_learns_planted_signal():
    from hypershift.screening.models import ScreenModel
    torch.set_num_threads(2); torch.manual_seed(11)
    model = ScreenModel('none', nodes=48, hidden=16)
    x = torch.randn(8, 48, 64, 5)*.1
    eligible = torch.ones(8, 48, dtype=torch.bool)
    y = x[:, :, -4:, 0].mean(-1)*5
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    first = ((model(x, eligible)-y)**2).mean().item()
    for _ in range(100):
        opt.zero_grad(); loss = ((model(x, eligible)-y)**2).mean()
        loss.backward(); opt.step()
    assert loss.item() < first*.2


def test_trading_does_not_choose_using_future_label_availability():
    from types import SimpleNamespace
    from hypershift.screening.metrics import trading
    n, t = 50, 20
    price = np.ones((n, t)); observed = np.ones((n, t), bool)
    observed[0, 7] = False
    panel = SimpleNamespace(price=price, observed=observed, eligible=np.ones((n, t), bool))
    signals = np.arange(3, 10)
    pred = np.tile(-np.arange(n), (len(signals), 1))
    r, pnl = trading(pred, signals, panel)
    assert r['missing_held_marks'] > 0
    assert pnl.sum() < 0  # Costs charged even with flat prices.


def test_static_graph_preserves_stock_specific_information():
    from hypershift.screening.models import ScreenModel
    torch.set_num_threads(2); torch.manual_seed(11)
    groups = np.arange(48) % 4
    model = ScreenModel('static', nodes=48, hidden=16, groups=groups)
    x = torch.randn(4, 48, 64, 5)*.1
    e = torch.ones(4, 48, dtype=torch.bool)
    p = model(x, e)
    assert len(torch.unique(p[0])) > 24
    y = x[:, :, -4:, 0].mean(-1)*5
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    first = ((p-y)**2).mean().item()
    for _ in range(100):
        opt.zero_grad(); loss = ((model(x, e)-y)**2).mean(); loss.backward(); opt.step()
    assert loss.item() < first*.3


def test_staggered_sleeves_are_self_financing():
    from types import SimpleNamespace
    from hypershift.screening.metrics import trading
    price = np.ones((50, 12))
    price[0, 5:] = 2
    price[1:5, 6:] = 2
    panel = SimpleNamespace(price=price, observed=np.ones_like(price, bool), eligible=np.ones_like(price, bool))
    signals = np.arange(5)
    pred = np.zeros((5, 50))
    for i in range(5):
        pred[i, i] = 1
    _, daily = trading(pred, signals, panel, k=1, cost=0)
    assert np.prod(1+daily) == pytest.approx(2.)
