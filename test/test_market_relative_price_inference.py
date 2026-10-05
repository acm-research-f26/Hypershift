from dataclasses import replace
import json
import numpy as np
import pytest
from experiments.market_ablation import relative_price_inference as relative
from experiments.market_ablation.config import SweepConfig


def contributions(ratio, n=20):
    naive = np.tile([1., 9.], n // 2)
    ratio = np.broadcast_to(ratio, naive.shape)
    return {"count": np.full(n, 100), "price_loss_sum": naive * ratio,
            "persistence_price_loss_sum": naive}


def test_primary_weighting_can_reverse_pooled_superiority():
    ratio = np.tile([.5, 1.1], 10)
    seeds = {1001: contributions(ratio - .02), 2001: contributions(ratio + .02)}
    result = relative.relative_price_inference(seeds, replicates=100)
    assert result["relative_price_MSE"] == pytest.approx(.8)
    assert result["price_skill"] == pytest.approx(.2)
    assert result["seed_relative_price_MSEs"] == pytest.approx({"1001": .78, "2001": .82})
    assert result["seed_relative_MSE_std"] == pytest.approx(np.sqrt(.0008))
    assert np.sum(contributions(ratio)["price_loss_sum"]) / np.sum(contributions(ratio)["persistence_price_loss_sum"]) == pytest.approx(1.04)
    inference = result["persistence_comparison"]
    assert inference["mean_model_minus_reference_loss"] == pytest.approx(-.2)
    for lag, values in result["price_skill_uncertainty"]["HAC"].items():
        assert values["ci95"] == pytest.approx([-inference["HAC"][lag]["ci95"][1], -inference["HAC"][lag]["ci95"][0]])


def test_matching_seed_reference_and_reproducible_market_bootstrap():
    ratio = np.tile([.5, 1.1], 10)
    seeds = {2001: contributions(ratio + .02), 1001: contributions(ratio - .02)}
    refs = {1001: contributions(ratio + .03), 2001: contributions(ratio + .07)}
    first = relative.relative_price_inference(seeds, refs, replicates=100, seed=21)
    second = relative.relative_price_inference(seeds, refs, replicates=100, seed=21)
    assert first == second
    assert first["reference_relative_price_MSE"] == pytest.approx(.85)
    assert first["skill_change_vs_matched_reference"] == pytest.approx(.05)
    assert first["sessions"] == 20
    assert first["seeds"] == [1001, 2001]


@pytest.mark.parametrize("damage", ["count", "baseline", "seed"])
def test_rejects_unmatched_reference_history_or_seed(damage):
    seeds = {1001: contributions(.8), 2001: contributions(.9)}
    refs = {1001: contributions(.9), 2001: contributions(1.)}
    if damage == "count":
        refs[1001]["count"][0] += 1
    elif damage == "baseline":
        refs[1001]["persistence_price_loss_sum"][0] += 1
    else:
        refs[3001] = refs.pop(2001)
    with pytest.raises(ValueError):
        relative.relative_price_inference(seeds, refs, replicates=50)


def test_zero_persistence_sessions_are_explicit_and_insufficient_support_is_undefined():
    row = contributions(.8)
    row["persistence_price_loss_sum"][:] = 0
    result = relative.relative_price_inference({1001: row}, replicates=50)
    assert result["sessions"] == 0 and result["zero_persistence_loss_sessions_excluded"] == 20
    assert result["relative_price_MSE"] is None
    assert result["persistence_comparison"]["status"] == "undefined"
    small = relative.relative_price_inference({1001: contributions(.8, 4)}, replicates=50)
    assert small["price_skill"] == pytest.approx(.2)
    assert small["price_skill_uncertainty"]["status"] == "undefined"


@pytest.mark.parametrize("damage", ["nan", "negative", "fractional", "empty_session"])
def test_invalid_saved_contributions_cannot_enter_inference(damage):
    row = contributions(.8)
    if damage == "nan":
        row["price_loss_sum"][0] = np.nan
    elif damage == "negative":
        row["price_loss_sum"][0] = -1
    elif damage == "fractional":
        row["count"] = row["count"].astype(float)
        row["count"][0] = .5
    else:
        row["count"][0] = 0
    with pytest.raises(ValueError, match="Invalid saved session"):
        relative.relative_price_inference({1001: row}, replicates=50)


@pytest.fixture
def saved_score_report(tmp_path, monkeypatch):
    config = replace(SweepConfig(), output=str(tmp_path), intervals=("1m",), bootstrap_replicates=50)
    jobs = []
    settings = {"training_loss": "raw_price_MSE", "hidden": 32}
    (tmp_path / "settings").mkdir()
    (tmp_path / "settings/1m.json").write_text(json.dumps(settings))
    for name, features, scope, ratio in (("T", "R", "core", .9), ("T-F-Q0", "F", "latest_fold_contrast", .8)):
        for seed in (1001, 2001):
            variant = {"name": name, "features": features, "families": (), "ph_context": False,
                       "control": "none", "parameter": ""}
            job = {"id": f"1m/fold-2/{name}/seed-{seed}", "interval": "1m", "fold": 2,
                   "variant": variant, "seed": seed, "scope": scope}
            jobs.append(job)
            path = tmp_path / "fits" / job["id"]
            path.mkdir(parents=True)
            (path / "result.json").write_text(json.dumps({"seed": seed, "variant": variant,
                                                         "settings": settings, "test_price_persistence_relative_mse": ratio}))
            np.savez(path / "test_contributions.npz", **contributions(ratio))
    monkeypatch.setattr(relative, "manifest", lambda _: {"jobs": jobs})
    return config, jobs


def test_report_records_primary_scores_reference_seeds_and_all_input_hashes(saved_score_report):
    config, jobs = saved_score_report
    records = relative.relative_price_report(config, jobs)
    assert len(records) == 2
    contrast = next(r for r in records if r["scope"] == "latest_fold_contrast")
    assert contrast["reference"] == "T"
    assert contrast["skill_change_vs_matched_reference"] == pytest.approx(.1)
    payload = json.loads((config.root / "report/primary_relative_price_inference.json").read_text())
    assert payload["final_fits"] == len(payload["inputs"]) == 4
    assert all(len(row["test_contributions_sha256"]) == 64 for row in payload["inputs"])


def test_report_refuses_incomplete_manifest_and_changed_primary_score(saved_score_report):
    config, jobs = saved_score_report
    with pytest.raises(ValueError, match="entire final manifest"):
        relative.relative_price_report(config, jobs[:-1])
    path = config.root / "fits" / jobs[0]["id"] / "result.json"
    result = json.loads(path.read_text())
    result["test_price_persistence_relative_mse"] += .1
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="differs from session contributions"):
        relative.relative_price_report(config, jobs)
