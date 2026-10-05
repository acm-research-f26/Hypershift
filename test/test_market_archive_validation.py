from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest

from experiments.market_ablation.archive_validation import RISK_KEYS, verify_portfolio_archive
from experiments.market_ablation.config import SweepConfig
from experiments.market_ablation.portfolio import portfolios


@pytest.fixture
def saved_portfolios(tmp_path):
    bars = np.zeros((40, 250, 5), np.float64)
    bars[:, :, 0] = 100 + np.arange(40)[:, None] * .04
    bars[:, :, 3] = bars[:, :, 0] + .02
    dataset = SimpleNamespace(interval="1m", bars=bars, times=np.arange(40, dtype=np.int64),
                              starts=np.arange(40, dtype=np.int64),
                              sessions=np.repeat(np.arange(2, dtype=np.int32), 20),
                              session_dates=np.array(["2025-01-02", "2025-01-03"]), expected=np.ones(40))
    origins = np.arange(20, 39, dtype=np.int64)
    prediction = np.broadcast_to(np.linspace(-.01, .01, 250), (len(origins), 250)).copy()
    config = replace(SweepConfig(), output=str(tmp_path))
    directory = tmp_path / "portfolios"
    portfolios(config, dataset, origins, prediction, np.ones_like(prediction, bool), directory)
    return directory, dataset, origins


def rewrite(path, key, mutate):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    mutate(arrays[key])
    np.savez_compressed(path, **arrays)


def test_verifies_every_saved_portfolio_value_and_hash(saved_portfolios):
    result = verify_portfolio_archive(*saved_portfolios)
    assert result["status"] == "verified" and len(result["files"]) == 39
    assert result["execution_marks"] == 18 and result["market_sessions"] == 1
    assert all(len(file["sha256"]) == 64 and file["bytes"] > 0 for file in result["files"])


@pytest.mark.parametrize("damage,match", [
    ("nonfinite", "Nonfinite"), ("wealth", "Wealth disagrees"),
    ("timing", "Delayed execution support"), ("unflagged_action", "Unflagged incomplete accounting")])
def test_rejects_corrupt_or_unflagged_saved_paths(saved_portfolios, damage, match):
    directory, dataset, origins = saved_portfolios
    if damage == "wealth":
        rewrite(directory / "long_only.npz", "wealth", lambda a: a.__setitem__(1, a[1] + .01))
    elif damage == "timing":
        rewrite(directory / "long_only.npz", "execution", lambda a: a.__setitem__(0, a[0] - 1))
    else:
        path = directory / "sensitivities/long_only/cost0-borrow0.npz"
        key, value = ("net_returns", np.nan) if damage == "nonfinite" else ("bad_actions", True)
        rewrite(path, key, lambda a: a.__setitem__(0, value))
    with pytest.raises(ValueError, match=match):
        verify_portfolio_archive(directory, dataset, origins)


@pytest.mark.parametrize("condition", ["incomplete_accounting", "portfolio_ruin"])
def test_accepts_explicitly_flagged_undefined_cases(saved_portfolios, condition):
    directory, dataset, origins = saved_portfolios
    path = directory / "sensitivities/long_only/cost0-borrow0.npz"
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    values = metrics["strategies"]["long_only"]["cost0-borrow0"]
    values.update({key: None for key in RISK_KEYS})
    if condition == "incomplete_accounting":
        arrays["bad_actions"][0] = True
        values["accounting_status"] = "incomplete_economic_valuation"
    else:
        arrays["net_returns"][0] = -1
        arrays["session_returns"] = np.array([], np.float64)
        values.update(status="undefined", reason="portfolio_ruin")
    np.savez_compressed(path, **arrays)
    metrics_path.write_text(json.dumps(metrics, allow_nan=False))
    assert verify_portfolio_archive(directory, dataset, origins)["status"] == "verified"
