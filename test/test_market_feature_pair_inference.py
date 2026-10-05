from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from experiments.market_ablation.config import SweepConfig, manifest
from experiments.market_ablation.feature_pair_inference import feature_pair_design, feature_pair_report
from experiments.market_ablation.statistics import holm


def test_full_design_isolates_ph_and_feature_changes_with_matching_seeds():
    config = SweepConfig()
    jobs = manifest(config)["jobs"]
    pairs = feature_pair_design(config, jobs)
    assert len(pairs) == 168
    for interval in config.intervals:
        selected = [pair for pair in pairs if pair["interval"] == interval]
        assert sum(pair["kind"] == "PH_context" for pair in selected) == 18
        assert sum(pair["kind"] == "feature_pack" for pair in selected) == 24
    for pair in pairs:
        assert pair["fold"] == 2
        assert [job["seed"] for job in pair["candidate_jobs"]] == [1001, 2001]
        assert [job["seed"] for job in pair["reference_jobs"]] == [1001, 2001]
        a, b = pair["candidate_jobs"][0]["variant"], pair["reference_jobs"][0]["variant"]
        assert a["families"] == b["families"]
        if pair["kind"] == "PH_context":
            assert a["features"] == b["features"] and a["ph_context"] and not b["ph_context"]
        else:
            assert a["ph_context"] == b["ph_context"] and b["features"] == "R"
    rv = next(pair for pair in pairs if pair["kind"] == "PH_context" and pair["anchor"] == "A"
              and pair["candidate_pack"] == "RV" and pair["interval"] == "1m")
    assert rv["reference"] == "A-RV-Q0"


@pytest.mark.parametrize("damage", ["duplicate", "missing", "modified"])
def test_design_rejects_noncanonical_jobs(damage):
    config = replace(SweepConfig(), intervals=("1m",))
    jobs = copy.deepcopy(manifest(config)["jobs"])
    if damage == "duplicate":
        jobs.append(jobs[0])
    elif damage == "missing":
        jobs.pop()
    else:
        jobs[0]["variant"]["ph_context"] = True
    with pytest.raises(ValueError, match="manifest"):
        feature_pair_design(config, jobs)


@pytest.fixture
def saved_feature_forecasts(tmp_path):
    config = replace(SweepConfig(), output=str(tmp_path), intervals=("1m",), bootstrap_replicates=50)
    jobs = manifest(config)["jobs"]
    pairs = feature_pair_design(config, jobs)
    required = {job["id"]: job for pair in pairs for side in ("candidate_jobs", "reference_jobs")
                for job in pair[side]}
    settings = {"training_loss": "raw_price_MSE", "hidden": 32}
    (tmp_path / "settings").mkdir()
    (tmp_path / "settings/1m.json").write_text(json.dumps(settings))
    naive = np.tile([1., 9.], 10)
    for job in required.values():
        pack = job["variant"]["features"]
        effect = {"R": 0, "RV": -.03, "F": -.07}[pack]
        ph_effect = (.02 if job["seed"] == 1001 else .04) if job["variant"]["ph_context"] else 0
        ratio = .9 + effect - ph_effect * np.linspace(.5, 1.5, 20)
        directory = tmp_path / "fits" / job["id"]
        directory.mkdir(parents=True)
        np.savez(directory / "test_contributions.npz", count=np.full(20, 100),
                 price_loss_sum=naive * ratio, persistence_price_loss_sum=naive)
        (directory / "result.json").write_text(json.dumps({"seed": job["seed"], "variant": job["variant"],
              "settings": settings, "tuning": False, "training_loss": "raw_price_MSE",
              "test_price_persistence_relative_mse": float(ratio.mean())}))
    return config, jobs


def test_saved_direct_contrasts_preserve_estimand_uncertainty_and_provenance(saved_feature_forecasts):
    config, jobs = saved_feature_forecasts
    result = feature_pair_report(config, jobs)
    assert result["groups_count"] == 42 and result["source_fits"] == 72
    assert result["complete_scope"] and len(result["inputs"]) == 72
    assert all(row["result_sha256"] and row["test_contributions_sha256"] for row in result["inputs"])
    row = next(row for row in result["groups"] if row["kind"] == "PH_context"
               and row["anchor"] == "A" and row["candidate_pack"] == "F")
    assert row["reference"] == "A-F-Q0" and row["seeds"] == [1001, 2001]
    assert row["skill_change_vs_matched_reference"] == pytest.approx(.03)
    assert row["seed_skill_changes"] == pytest.approx({"1001": .02, "2001": .04})
    assert row["seed_skill_change_std"] == pytest.approx(np.sqrt(.0002))
    assert row["market_uncertainty_unit"] == "whole_market_session"
    assert row["skill_change_uncertainty"]["stationary_bootstrap"]["10"]["replications"] == 50
    for kind in ("PH_context", "feature_pack"):
        family = [row for row in result["groups"] if row["kind"] == kind]
        expected = holm([row["matched_reference_comparison"]["primary_p"] for row in family])
        assert [row["Holm_p_matched_reference"] for row in family] == pytest.approx(expected)


@pytest.mark.parametrize("damage", ["missing", "settings", "support", "score"])
def test_saved_pair_rejects_missing_or_mismatched_evidence(saved_feature_forecasts, damage):
    config, jobs = saved_feature_forecasts
    directory = config.root / "fits/1m/fold-2/A-F-Q0/seed-1001"
    if damage == "missing":
        (directory / "result.json").unlink()
        error, message = RuntimeError, "incomplete"
    elif damage == "support":
        with np.load(directory / "test_contributions.npz") as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["count"][0] += 1
        np.savez(directory / "test_contributions.npz", **arrays)
        error, message = ValueError, "differ"
    else:
        path = directory / "result.json"
        record = json.loads(path.read_text())
        if damage == "settings":
            record["settings"]["hidden"] = 128
        else:
            record["test_price_persistence_relative_mse"] += .1
        path.write_text(json.dumps(record))
        error, message = ValueError, "differ"
    with pytest.raises(error, match=message):
        feature_pair_report(config, jobs)
