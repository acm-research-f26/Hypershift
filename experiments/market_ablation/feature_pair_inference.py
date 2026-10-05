"""Direct PH and feature-pack contrasts from existing matching-seed forecasts."""
from collections import defaultdict
from pathlib import Path
import json

import numpy as np

from .config import FEATURE_PACKS, manifest
from .relative_price_inference import relative_price_inference
from .statistics import holm
from .storage import atomic_json, digest_file


FEATURE_ANCHORS = ("T", "GKL", "P", "C", "GKLPC", "A")


def feature_pair_design(config, jobs, intervals=None):
    """Resolve the planned latest-fold factorial without selecting by performance."""
    intervals = tuple(config.intervals if intervals is None else intervals)
    if not intervals or len(set(intervals)) != len(intervals) or not set(intervals) <= set(config.intervals):
        raise ValueError("Unknown or repeated feature comparison intervals")
    canonical = {job["id"]: job for job in manifest(config)["jobs"]}
    supplied = {job["id"]: job for job in jobs}
    if len(supplied) != len(jobs) or supplied.keys() != canonical.keys():
        raise ValueError("Feature comparisons require the entire canonical job manifest")
    if any(json.dumps(supplied[key], sort_keys=True) != json.dumps(job, sort_keys=True)
           for key, job in canonical.items()):
        raise ValueError("Feature comparison job definitions differ from the manifest")
    design = []

    def resolve(interval, anchor, pack, ph):
        name = anchor if pack == "R" and not ph else f"{anchor}-{pack}-Q{int(ph)}"
        group = [canonical[f"{interval}/fold-2/{name}/seed-{seed}"] for seed in config.seeds[:2]]
        if len({job["seed"] for job in group}) != 2:
            raise ValueError("Two distinct matching feature seeds are required")
        return group

    def add(interval, anchor, kind, candidate_pack, reference_pack, candidate_ph, reference_ph):
        candidate = resolve(interval, anchor, candidate_pack, candidate_ph)
        reference = resolve(interval, anchor, reference_pack, reference_ph)
        left, right = candidate[0]["variant"], reference[0]["variant"]
        controlled = ("families", "control", "parameter")
        if any(left[key] != right[key] for key in controlled) or left["control"] != "none":
            raise ValueError("Feature comparison changes a constructor or control")
        if kind == "PH_context" and left["features"] != right["features"]:
            raise ValueError("PH context comparison changes the feature pack")
        if kind == "feature_pack" and left["ph_context"] != right["ph_context"]:
            raise ValueError("Feature pack comparison changes PH context")
        design.append({"interval": interval, "fold": 2, "anchor": anchor, "kind": kind,
                       "candidate": left["name"], "reference": right["name"],
                       "candidate_pack": candidate_pack, "reference_pack": reference_pack,
                       "candidate_ph_context": candidate_ph, "reference_ph_context": reference_ph,
                       "candidate_jobs": candidate, "reference_jobs": reference})

    for interval in intervals:
        for anchor in FEATURE_ANCHORS:
            for pack in FEATURE_PACKS:
                add(interval, anchor, "PH_context", pack, pack, True, False)
            for ph in (False, True):
                for pack in ("RV", "F"):
                    add(interval, anchor, "feature_pack", pack, "R", ph, ph)
    return design


def feature_pair_report(config, jobs, *, intervals=None, directory=None):
    """Full reporting uses all intervals; an explicit interim subset stays labelled."""
    design = feature_pair_design(config, jobs, intervals)
    required = {job["id"]: job for pair in design for side in ("candidate_jobs", "reference_jobs")
                for job in pair[side]}
    missing = [identity for identity in required
               if not (config.root / "fits" / identity / "result.json").is_file()]
    if missing:
        raise RuntimeError(f"Feature comparison fits are incomplete: {len(missing)} missing")
    loaded, inputs = {}, []
    for identity, job in required.items():
        path = config.root / "fits" / identity
        result = json.loads((path / "result.json").read_text())
        settings = json.loads((config.root / "settings" / f"{job['interval']}.json").read_text())
        if (result["seed"] != job["seed"] or result["settings"] != settings
                or result["variant"] != json.loads(json.dumps(job["variant"]))
                or result["tuning"] or result["training_loss"] != "raw_price_MSE"):
            raise ValueError(f"Feature forecast identity/settings differ: {identity}")
        with np.load(path / "test_contributions.npz", allow_pickle=False) as archive:
            row = {key: archive[key] for key in archive.files}
        loaded[identity] = (row, result["test_price_persistence_relative_mse"])
        inputs.append({"job": identity, "result_sha256": digest_file(path / "result.json"),
                       "test_contributions_sha256": digest_file(path / "test_contributions.npz")})
    records = []
    for pair in design:
        candidate = {job["seed"]: loaded[job["id"]][0] for job in pair["candidate_jobs"]}
        reference = {job["seed"]: loaded[job["id"]][0] for job in pair["reference_jobs"]}
        inference = relative_price_inference(candidate, reference, replicates=config.bootstrap_replicates,
                                             seed=config.constructor_seed)
        seed_changes = {}
        for candidate_job, reference_job in zip(pair["candidate_jobs"], pair["reference_jobs"]):
            candidate_row, candidate_score = loaded[candidate_job["id"]]
            reference_row, reference_score = loaded[reference_job["id"]]
            for row, score in ((candidate_row, candidate_score), (reference_row, reference_score)):
                eligible = (row["count"] > 0) & (row["persistence_price_loss_sum"] > 0)
                reconstructed = np.mean(row["price_loss_sum"][eligible] / row["persistence_price_loss_sum"][eligible])
                if not np.isclose(reconstructed, score, rtol=1e-12, atol=1e-12):
                    raise ValueError("Feature seed score differs from saved session contributions")
            seed_changes[str(candidate_job["seed"])] = float(reference_score - candidate_score)
        record = {key: value for key, value in pair.items() if not key.endswith("_jobs")}
        record.update(inference, scope="latest_fold_direct_feature_contrast",
                      candidate_jobs=[job["id"] for job in pair["candidate_jobs"]],
                      reference_jobs=[job["id"] for job in pair["reference_jobs"]],
                      seed_skill_changes=seed_changes,
                      seed_skill_change_std=float(np.std(list(seed_changes.values()), ddof=1)))
        records.append(record)
    families = defaultdict(list)
    for record in records:
        families[(record["interval"], record["kind"])].append(record)
    for family in families.values():
        adjusted = holm([row["matched_reference_comparison"].get("primary_p")
                         if row["matched_reference_comparison"].get("primary_p") is not None else np.nan
                         for row in family])
        for row, p in zip(family, adjusted):
            row["Holm_p_matched_reference"] = float(p) if np.isfinite(p) else None
    selected = tuple(dict.fromkeys(pair["interval"] for pair in design))
    directory = config.root / "report" if directory is None else Path(directory)
    payload = {"manifest_identity": config.identity, "groups": records, "intervals": selected,
               "groups_count": len(records), "expected_final_groups": 42 * len(config.intervals),
               "complete_scope": set(selected) == set(config.intervals), "source_fits": len(required),
               "bootstrap_replicates": config.bootstrap_replicates,
               "multiplicity_families": "Per interval: 18 direct PH on/off comparisons; 24 feature-pack comparisons separately. These are additional to the primary persistence and base-feature comparison families.",
               "scope": "Latest test fold only, first two matching seeds, frozen model settings and constructors; equal-session persistence-relative raw-price skill changes. Market-session inference and training-seed variation are separate. No retraining, test-driven selection, or claim of individually optimal configurations.",
               "source_sha256": digest_file(Path(__file__)),
               "inference_source_sha256": digest_file(Path(__file__).with_name("relative_price_inference.py")),
               "statistics_source_sha256": digest_file(Path(__file__).with_name("statistics.py")), "inputs": inputs}
    atomic_json(directory / "feature_context_pair_inference.json", payload)
    return payload


def feature_pair_figures(payload, directory):
    """Show pointwise market intervals and matching-seed variation in separate panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for interval in payload["intervals"]:
        for kind in ("PH_context", "feature_pack"):
            rows = [row for row in payload["groups"] if row["interval"] == interval and row["kind"] == kind]
            fig, axes = plt.subplots(1, 2, figsize=(13, max(7, len(rows) * .34 + 2)), sharey=True)
            labels = []
            for index, row in enumerate(rows):
                labels.append(f"{row['anchor']}: {row['candidate_pack']} PH on/off" if kind == "PH_context"
                              else f"{row['anchor']}: {row['candidate_pack']} vs R, PH {'on' if row['candidate_ph_context'] else 'off'}")
                estimate = row["skill_change_vs_matched_reference"]
                uncertainty = row["skill_change_uncertainty"]
                if uncertainty["status"] == "ok":
                    lo, hi = np.array(uncertainty["stationary_bootstrap"]["10"]["ci95"]) * 100
                    axes[0].plot([lo, hi], [index, index], color="#537d86", linewidth=1.4)
                if estimate is not None:
                    significant = row["Holm_p_matched_reference"] is not None and row["Holm_p_matched_reference"] < .05
                    axes[0].scatter(100 * estimate, index, s=32, edgecolors="#1b555f",
                                    facecolors="#1b555f" if significant else "white", zorder=3)
                changes = np.array(list(row["seed_skill_changes"].values())) * 100
                axes[1].plot([changes.min(), changes.max()], [index, index], color="#a5a5a5", linewidth=1)
                for value, marker, color in zip(changes, ("o", "x"), ("#1b555f", "#bf7733")):
                    axes[1].scatter(value, index, marker=marker, color=color, s=22, zorder=3)
            axes[0].set_yticks(range(len(rows)), labels)
            axes[0].invert_yaxis()
            for ax in axes:
                ax.axvline(0, color="black", linewidth=.8)
                ax.grid(axis="x", alpha=.2)
                ax.set_xlabel("Price-skill change (percentage points); positive improves")
            axes[0].set_title("Market history: pointwise 95% paired bootstrap intervals")
            axes[1].set_title("Training variation: two matching seeds, no market CI")
            title = "PH context on versus off" if kind == "PH_context" else "Feature packs at fixed PH status"
            fig.suptitle(f"{interval}: {title}; latest test fold", fontsize=13)
            fig.text(.5, .012, f"Whole-market session blocks, mean length 10; {payload['bootstrap_replicates']:,} replications. Filled market points: Holm-adjusted HAC p < 0.05.", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .035, 1, .96))
            path = directory / f"{interval}-direct-{kind}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs.append({"path": str(path), "sha256": digest_file(path)})
    return outputs
