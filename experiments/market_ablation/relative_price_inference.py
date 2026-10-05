"""Inference for the runner's equal-session persistence-relative price score."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from .config import manifest
from .statistics import paired_inference, holm
from .storage import atomic_json, digest_file


def _aligned(contributions):
    if not contributions:
        raise ValueError("At least one saved seed is required")
    seeds = sorted(contributions)
    baseline = None
    losses = []
    for seed in seeds:
        row = contributions[seed]
        arrays = [np.asarray(row[key], np.float64) for key in
                  ("count", "price_loss_sum", "persistence_price_loss_sum")]
        count, model, naive = arrays
        if (count.ndim != 1 or not len(count) or any(a.shape != count.shape for a in arrays)
                or any(not np.isfinite(a).all() or (a < 0).any() for a in arrays)
                or not np.equal(count, np.floor(count)).all()
                or np.any((count == 0) & ((model != 0) | (naive != 0)))):
            raise ValueError("Invalid saved session price losses/counts")
        if baseline is None:
            baseline = (count, naive)
        elif any(not np.array_equal(a, b) for a, b in zip((count, naive), baseline)):
            raise ValueError("Seed target support or persistence losses differ")
        losses.append(model)
    eligible = (baseline[0] > 0) & (baseline[1] > 0)
    ratios = np.array(losses)[:, eligible] / baseline[1][eligible]
    if not np.isfinite(ratios).all():
        raise ValueError("Nonfinite session-relative price loss")
    return seeds, baseline, eligible, ratios


def _skill_intervals(inference):
    if inference["status"] != "ok":
        return {"status": inference["status"], "reason": inference["reason"]}
    return {"status": "ok", "units": "fraction; multiply by 100 for percent or percentage-point changes",
            "HAC": {lag: {"standard_error": value["standard_error"],
                           "ci95": [-value["ci95"][1], -value["ci95"][0]]}
                    for lag, value in inference["HAC"].items()},
            "stationary_bootstrap": {length: {"ci95": [-value["ci95"][1], -value["ci95"][0]],
                                               "replications": value["replications"],
                                               "centered_two_sided_p": value["centered_two_sided_p"]}
                                     for length, value in inference["bootstrap"].items()}}


def relative_price_inference(contributions, references=None, *, replicates=2000, seed=1003):
    """Average saved seed losses, then resample whole market sessions together."""
    seeds, baseline, eligible, ratios = _aligned(contributions)
    n = int(eligible.sum())
    result = {"seeds": seeds, "sessions": n,
              "zero_persistence_loss_sessions_excluded": int(((baseline[0] > 0) & (baseline[1] == 0)).sum()),
              "estimand": "1-mean_session(mean_seed(model_price_loss_sum)/persistence_price_loss_sum)",
              "market_uncertainty_unit": "whole_market_session",
              "training_uncertainty": "conditional on these saved training runs; seed variation is separate",
              "relative_price_MSE": float(ratios.mean()) if n else None,
              "price_skill": float(1-ratios.mean()) if n else None,
              "seed_relative_price_MSEs": {str(s): float(r.mean()) if n else None for s, r in zip(seeds, ratios)},
              "seed_relative_MSE_std": float(np.std(ratios.mean(axis=1), ddof=1)) if n and len(seeds) > 1 else None}
    inference = paired_inference(ratios.mean(axis=0)-1, replicates, seed)
    result["persistence_comparison"] = inference
    result["price_skill_uncertainty"] = _skill_intervals(inference)
    if references is not None:
        ref_seeds, ref_baseline, ref_eligible, ref_ratios = _aligned(references)
        if (ref_seeds != seeds or not np.array_equal(ref_eligible, eligible)
                or any(not np.array_equal(a, b) for a, b in zip(baseline, ref_baseline))):
            raise ValueError("Reference seeds, target support or persistence losses differ")
        delta = ratios.mean(axis=0)-ref_ratios.mean(axis=0)
        reference = paired_inference(delta, replicates, seed)
        result.update(reference_relative_price_MSE=float(ref_ratios.mean()) if n else None,
                      skill_change_vs_matched_reference=float(-delta.mean()) if n else None,
                      matched_reference_comparison=reference,
                      skill_change_uncertainty=_skill_intervals(reference))
    return result


def _reference_name(variant):
    control = variant["control"]
    if control in ("frozen", "window"):
        return "A"
    if control in ("rewire", "uniform"):
        return "frozen-reference"
    return variant["name"].split("-", 1)[0]


def relative_price_report(config, jobs):
    """Require the full manifest and export the primary score's separate inference."""
    canonical = {job["id"]: job for job in manifest(config)["jobs"]}
    if len(jobs) != len(canonical) or {j["id"] for j in jobs} != set(canonical):
        raise ValueError("Primary score inference requires the entire final manifest")
    groups = defaultdict(list)
    for job in jobs:
        groups[(job["interval"], job["fold"], job["variant"]["name"])].append(job)
    saved, inputs = {}, []

    def load(job):
        if job["id"] in saved:
            return saved[job["id"]]
        directory = config.root / "fits" / job["id"]
        result = json.loads((directory / "result.json").read_text())
        settings = json.loads((config.root / "settings" / f"{job['interval']}.json").read_text())
        if (result["seed"] != job["seed"] or result["settings"] != settings
                or result["variant"] != json.loads(json.dumps(job["variant"]))):
            raise ValueError(f"Saved primary score identity/settings differ: {job['id']}")
        with np.load(directory / "test_contributions.npz", allow_pickle=False) as archive:
            row = {key: archive[key] for key in archive.files}
        _, _, eligible, ratio = _aligned({job["seed"]: row})
        if not eligible.any() or not np.isclose(ratio.mean(), result["test_price_persistence_relative_mse"], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Saved primary score differs from session contributions: {job['id']}")
        inputs.append({"job": job["id"], "result_sha256": digest_file(directory / "result.json"),
                       "test_contributions_sha256": digest_file(directory / "test_contributions.npz")})
        saved[job["id"]] = row
        return row

    records = []
    for (interval, fold, name), group in groups.items():
        seeds = [job["seed"] for job in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Repeated training seed in a comparison")
        contributions = {job["seed"]: load(job) for job in group}
        references, reference_name = None, None
        if group[0]["scope"] == "latest_fold_contrast":
            reference_name = _reference_name(group[0]["variant"])
            references = {seed: load(canonical[f"{interval}/fold-{fold}/{reference_name}/seed-{seed}"]) for seed in seeds}
        record = {"interval": interval, "fold": fold, "variant": name, "scope": group[0]["scope"],
                  **relative_price_inference(contributions, references, replicates=config.bootstrap_replicates,
                                             seed=config.constructor_seed)}
        if reference_name is not None:
            record["reference"] = reference_name
        records.append(record)
    for interval in config.intervals:
        for scope in ("core", "latest_fold_contrast"):
            family = [r for r in records if r["interval"] == interval and r["scope"] == scope]
            for comparison, field in (("persistence_comparison", "Holm_p_persistence"),
                                      ("matched_reference_comparison", "Holm_p_matched_reference")):
                present = [r for r in family if comparison in r]
                adjusted = holm([r[comparison].get("primary_p") if r[comparison].get("primary_p") is not None else np.nan for r in present])
                for record, p in zip(present, adjusted):
                    record[field] = float(p) if np.isfinite(p) else None
    atomic_json(config.root / "report/primary_relative_price_inference.json",
                {"manifest_identity": config.identity, "final_fits": len(jobs), "groups": records,
                 "bootstrap_replicates": config.bootstrap_replicates,
                 "multiplicity_families": "per interval: all core fold/configuration comparisons together; latest-fold contrasts separately",
                 "scope": "Primary equal-session persistence-relative raw-price score. Complements pooled price metrics and ratio-of-session-means inference; does not treat seeds or stocks as independent market histories.",
                 "source_sha256": digest_file(Path(__file__)),
                 "statistics_source_sha256": digest_file(Path(__file__).with_name("statistics.py")), "inputs": inputs})
    return records
