import numpy as np
import pytest

from hypershift.data.rsr import MarketData


def make_synthetic_market(N=12, T=80, C=5, valid_index=40, test_index=60, seed=0, signal=0.5):
    """Random-walk prices; next-day return partly predictable from last-day return (planted signal)."""
    rng = np.random.default_rng(seed)
    ret = np.zeros((N, T), dtype=np.float32)
    ret[:, 0] = rng.normal(0, 0.01, N)
    for t in range(1, T):
        ret[:, t] = signal * ret[:, t - 1] + rng.normal(0, 0.01, N)
    close = np.cumprod(1 + ret, axis=1)
    close = close / close[:, :valid_index].max(axis=1, keepdims=True)
    feats = np.repeat(close[:, :, None], C, axis=2).astype(np.float32)
    mask = np.ones((N, T), dtype=np.float32)
    mask[0, 50] = 0.0  # one missing day
    gt = np.zeros((N, T), dtype=np.float32)
    gt[:, 1:] = close[:, 1:] / close[:, :-1] - 1
    gt[0, 50] = gt[0, 51] = 0.0
    return MarketData([f"S{i}" for i in range(N)], feats, mask, gt, close.astype(np.float32), valid_index, test_index)


def make_synthetic_hypergraph(N=12):
    from hypershift.data.hypergraph import Hypergraph  # lazy: Task 3 creates this module
    return Hypergraph(N, ((0, 1, 2), (2, 3, 4, 5), (6, 7), (8, 9, 10)))  # node 11 isolated


@pytest.fixture
def synthetic_market():
    return make_synthetic_market()


@pytest.fixture
def synthetic_hypergraph():
    return make_synthetic_hypergraph()
