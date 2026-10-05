from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
import json

from experiments.config import TimeSpan
from experiments.datasets import ALPACA_1MIN_ARCHIVE, LOCAL_DATA
from hyperedges.common.types import ConstructorSpec

FAMILIES = tuple("GKLEJMPC")
INTERVALS = ("1m", "15m", "1h", "1d")
ANCHORS = ("T", "GKL", "A")
FEATURE_NAMES = ("log_return", "log1p_volume", "volatility20", "range", "close_open",
                 "opening_gap", "relative_volume20", "time_sin", "time_cos", "partial")
FEATURE_PACKS = {"R": (0,), "RV": (0, 1, 2), "F": tuple(range(len(FEATURE_NAMES)))}


@dataclass(frozen=True)
class Variant:
    name: str
    families: tuple[str, ...]
    features: str = "R"
    ph_context: bool = False
    control: str = "none"
    parameter: str = ""


@dataclass(frozen=True)
class SweepConfig:
    output: str = str(LOCAL_DATA.parent / "local-experiments" / "runs" / "four_day")
    archive: str = str(LOCAL_DATA / ALPACA_1MIN_ARCHIVE)
    gics: str = str(LOCAL_DATA / "metadata/gics/vanguard/2026-10-02/classifications.csv")
    intervals: tuple[str, ...] = INTERVALS
    lookback: int = 20
    minute_training_stride: int = 5
    tuning_trials: int = 12
    tuning_origin_cap: int = 20000
    tuning_epochs: int = 8
    tuning_patience: int = 3
    epochs: int = 20
    patience: int = 4
    bootstrap_replicates: int = 2000
    reserve_gib: float = 15
    target_hours: float = 96
    constructor_seed: int = 1003
    seeds: tuple[int, ...] = (1001, 2001, 3001)
    batch_sizes: tuple[int, ...] = (64, 128)
    device: str = "auto"
    training_loss: str = "raw_price_MSE"
    start: str = "2021-09-26"
    end: str = "2026-09-26"
    schema_version: int = 1

    def __post_init__(self):
        if not set(self.intervals) <= set(INTERVALS) or not self.intervals:
            raise ValueError("Unknown market intervals")
        if self.training_loss not in ("raw_price_MSE","log_return_MSE"):
            raise ValueError("Unknown forecasting objective")
        if min(self.lookback, self.minute_training_stride, self.epochs, self.tuning_trials) < 1:
            raise ValueError("Positive history, sampling and training budgets required")

    @property
    def root(self):
        return Path(self.output)

    @property
    def identity(self):
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def families(label):
    return FAMILIES if label == "A" else tuple(f for f in FAMILIES if f in label)


def core_variants():
    labels = ["T", *FAMILIES, "GK", "GL", "KL", "GKL", "KEJ", "A"]
    result = [Variant(label, families(label) if label != "T" else ()) for label in labels]
    result += [Variant(f"A-minus-{f}", tuple(other for other in FAMILIES if other != f)) for f in FAMILIES]
    result += [Variant(label, families(label)) for label in ("KP", "LP", "KC", "LC", "GKLP", "GKLC", "PC", "GKLPC")]
    result += [Variant(f"{label}-{similarity}", families(label), parameter=similarity)
               for similarity in ("signed", "covariance") for label in ("K", "GKL")]
    assert len(result) == 35 and len({v.name for v in result}) == 35
    return result


def contrast_variants():
    result = []
    for label in ("T", "GKL", "P", "C", "GKLPC", "A"):
        for pack in FEATURE_PACKS:
            for ph in (False, True):
                if pack == "R" and not ph:
                    continue
                result.append(Variant(f"{label}-{pack}-Q{int(ph)}", families(label) if label != "T" else (), pack, ph))
    result += [Variant("frozen-reference", FAMILIES, control="frozen")]
    result += [Variant(f"rewire-{f}", FAMILIES, control="rewire", parameter=f) for f in FAMILIES]
    result += [Variant("rewire-all", FAMILIES, control="rewire", parameter="all"),
               Variant("uniform-all", FAMILIES, control="uniform", parameter="all")]
    result += [Variant(f"{label}-no-family-norm", families(label), control="no_family_norm") for label in ("GKLPC", "A")]
    result += [Variant(f"{label}-cover-return-pca", families(label), control="cover_graph") for label in ("C", "GKLPC", "A")]
    result += [Variant(f"A-window-{scale}", FAMILIES, control="window", parameter=scale) for scale in ("0.5", "2")]
    assert len(result) == 48
    return result


def folds():
    return [{"id": i, "train_start": "2021-09-26T00:00:00Z", "fit_cutoff": f"{year}-06-26T00:00:00Z",
             "validation_end": f"{year}-09-26T00:00:00Z", "test_end": f"{year+1}-09-26T00:00:00Z"}
            for i, year in enumerate((2023, 2024, 2025))]


def manifest(config):
    jobs = []
    for interval in config.intervals:
        for fold in folds():
            for variant in core_variants() + (contrast_variants() if fold["id"] == 2 else []):
                seeds = config.seeds if variant in core_variants() else config.seeds[:2]
                for seed in seeds:
                    jobs.append({"id": f"{interval}/fold-{fold['id']}/{variant.name}/seed-{seed}",
                                 "interval": interval, "fold": fold["id"], "seed": seed,
                                 "variant": asdict(variant), "scope": "core" if variant in core_variants() else "latest_fold_contrast"})
    return {"schema_version": 1, "configuration": asdict(config), "identity": config.identity,
            "jobs": jobs, "final_fits": len(jobs), "joint_tuning_trials": len(config.intervals) * config.tuning_trials,
            "short_tuning_fits": len(config.intervals) * config.tuning_trials * 3,
            "inference_scope": "fixed_forecasts", "comparison_support": "common_R_eligibility",
            "constructor_recipes": {interval:{v.name:[asdict(s) for s in constructor_specs(v,interval,config.gics)]
                                               for v in core_variants()+contrast_variants()} for interval in config.intervals},
            "model": "shared_flattened_20_bar_history_projection_plus_incidence_preserving_nonlinear_edge_messages"}


def constructor_specs(variant, interval, gics, cover_version="0.1.0"):
    daily = interval == "1d"
    multiplier = float(variant.parameter) if variant.control == "window" else 1
    window = TimeSpan(max(1, round((252 if daily else 60) * multiplier)), "sessions")
    event_window = TimeSpan(max(1, round((126 if daily else 20) * multiplier)), "sessions")
    descriptor = {"features": ["mean", "std", "q10", "q50", "q90", "downside_frequency", "upside_frequency", "lag1"],
                  "coverage_threshold": .95, "standardize": True}
    knn = {"neighbors": 5, "similarity": "absolute_covariance" if variant.parameter == "covariance" else ("signed_correlation" if variant.parameter == "signed" else "absolute_correlation"),
           "absolute": variant.parameter != "signed", "coverage_threshold": .95, "min_rows": 30}
    specs = {"G": [ConstructorSpec("G", "gics", {"source": gics, "level": "industry_group", "min_size": 2,
             "metadata_protocol": "retrospective_static"}, window)],
        "K": [ConstructorSpec("K", "covariance_knn", knn, window)],
        "L": [ConstructorSpec("L", "learned_membership", {"slots": 16}, window)],
        "E": [ConstructorSpec(f"E-{direction}", "event_dowker", {"quantiles": [.2, .8], "direction": direction,
               "min_support": .02, "size_bounds": [3, 4], "edge_budget": 250, "candidate_budget": 10000,
               "candidate_mode": "sampled"}, event_window) for direction in ("down", "up")],
        "J": [ConstructorSpec("J", "joint_information", {"bin_spec": {"kind": "binary_sign"}, "smoothing": .5,
               "min_observations": 100, "tolerance": 1e-8, "max_iterations": 1000, "candidate_budget": 5000,
               "order": 3, "estimation_fraction": .8, "selection_spec": {"edge_budget": 100,
               "min_holdout_gain_bits": 0., "min_selection_observations": 30}}, window)],
        "M": [ConstructorSpec("M", "mapper_cover", {"descriptor_spec": {**descriptor, "features": ["mean", "std", "q10", "q90"]},
               "lens_spec": {"kind": "feature", "feature": "std"}, "cover_spec": {"bins": 8, "overlap": .4},
               "clustering_spec": {"kind": "single_linkage", "radius": 1.}, "min_size": 2, "edge_budget": 250}, window)],
        "P": [ConstructorSpec("P", "ph_localized", {}, window)],
        "C": [ConstructorSpec("C", "cover_learning", {"descriptor_spec": descriptor,
               "graph_spec": {"neighbors": 10, "algorithm": "umap", "backend_version": cover_version},
               "backend_spec": {"name": "shapediscover", "version": cover_version},
               "objective_spec": {"preset": "ShapeDiscover", "parameters": {"n_cover": 32, "loss_weights": [1,10,1,10],
               "initialization_algorithm": "spectral_clustering", "model": "set_function", "n_max_iter": 250,
               "learning_rate": .001}},
               "membership_rule": {"threshold": .5, "min_size": 2},
               "graph_representation": "return_pca" if variant.control == "cover_graph" else "descriptors"}, window)]}
    return tuple(spec for family in variant.families for spec in specs[family])
