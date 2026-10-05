from types import SimpleNamespace

import numpy as np
import pytest

from experiments.market_ablation.config import Variant
from experiments.market_ablation.dataset import MarketDataset
from experiments.market_ablation.feature_gather import feature_batch


@pytest.mark.parametrize("pack", ["R", "RV", "F"])
@pytest.mark.parametrize("context", [False, True])
def test_cached_gather_preserves_batches_missingness_and_dataset(pack, context):
    rng = np.random.default_rng(19023)
    dataset = MarketDataset.__new__(MarketDataset)
    dataset.config = SimpleNamespace(lookback=20)
    dataset.features = rng.normal(size=(37, 250, 10)).astype(np.float32)
    dataset.feature_mask = rng.random((37, 250, 10)) > 0.3
    dataset.feature_mask[:, 0] = False
    dataset.features[~dataset.feature_mask] = np.nan
    dataset.valid = rng.random((37, 250)) > 0.2
    dataset.valid[:, 1] = False
    dataset.bars = rng.uniform(50, 150, size=(37, 250, 5))
    dataset.bars[:, :, 3][~dataset.valid] = 0
    moments = {"mean": rng.normal(size=(250, 10)).astype(np.float32),
               "std": rng.uniform(0.5, 2, size=(250, 10)).astype(np.float32),
               "target_scale": np.float32(0.01)}
    variant = Variant("test", (), features=pack, ph_context=context)
    original_features, original_mask = dataset.features, dataset.feature_mask
    for origins in (np.array([19, 30, 20, 19]), np.array([31, 22, 28])):
        expected = dataset.batch(origins, variant, moments)
        actual = feature_batch(dataset, origins, variant, moments)
        for before, after in zip(expected, actual, strict=True):
            assert before.dtype == after.dtype and before.shape == after.shape
            assert before.strides == after.strides
            np.testing.assert_array_equal(before, after)
        assert dataset.features is original_features and dataset.feature_mask is original_mask
        assert np.isfinite(actual[0]).all() and np.isfinite(actual[2]).all()
        assert not actual[1][:, :2].any() and not actual[3][:, :2].any()
