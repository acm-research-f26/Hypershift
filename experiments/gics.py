"""Offline GICS recipes for the verified 250-stock public snapshot."""

from pathlib import Path

from hyperedges.common.types import ConstructorSpec, HyperedgePipelineConfig
from .datasets import LOCAL_DATA


PUBLIC_GICS_DIRECTORY = LOCAL_DATA / "metadata" / "gics" / "vanguard" / "2026-10-02"
PUBLIC_GICS_FILE = PUBLIC_GICS_DIRECTORY / "classifications.csv"


def public_gics_spec(instance_id="gics", *, level="industry_group", source=PUBLIC_GICS_FILE):
    """Use published real classifications with explicit retrospective scope."""
    return ConstructorSpec(instance_id, "gics", {"source": str(Path(source)), "level": level,
                                               "min_size": 2, "metadata_protocol": "retrospective_static"})


def public_gics_pipeline(*, include_knn=True, source=PUBLIC_GICS_FILE, level="industry_group"):
    """Ready-to-use frozen GICS or GICS+KNN configuration; no network access."""
    specs = (public_gics_spec(source=source, level=level),)
    if include_knn:
        specs += (ConstructorSpec("knn", "covariance_knn", {"neighbors": 5}),)
    return HyperedgePipelineConfig(specs)
