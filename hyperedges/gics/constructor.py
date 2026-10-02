"""Supplied classifications, with explicit historical availability and no taxonomy substitution."""

from pathlib import Path
from numbers import Integral

import pandas as pd

from ..common.types import make_hyperedge_family, _utc_timestamp

LEVELS = {"sector", "industry_group", "industry", "sub_industry"}


def load_classification_records(source):
    records = source.copy(deep=True) if isinstance(source, pd.DataFrame) else pd.read_csv(Path(source), dtype=str)
    if not {"node_id", "source"} <= set(records.columns):
        raise ValueError("Classification records require node_id and source columns")
    if records["node_id"].isna().any() or records["source"].isna().any():
        raise ValueError("Classification IDs and provenance cannot be missing")
    if not records["node_id"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Classification node IDs must be nonempty strings")
    return records


def select_classifications_at_cutoff(records, node_ids, cutoff, *, level="industry_group", metadata_protocol="point_in_time"):
    cutoff = _utc_timestamp(cutoff)
    if level not in LEVELS or level not in records:
        raise ValueError(f"Missing or unsupported GICS level {level!r}")
    if metadata_protocol not in {"point_in_time", "retrospective_static"}:
        raise ValueError("Unknown classification metadata protocol")
    records = records.loc[records.node_id.isin(node_ids)].copy()
    if metadata_protocol == "point_in_time":
        if not {"effective_from", "available_at"} <= set(records.columns):
            raise ValueError("Point-in-time GICS requires effective_from and available_at")
        for column in ("effective_from", "available_at"):
            records[column] = records[column].map(_utc_timestamp)
        records = records.loc[(records.effective_from <= cutoff) & (records.available_at <= cutoff)]
        if "effective_to" in records:
            ends = records.effective_to.map(lambda x: None if pd.isna(x) else _utc_timestamp(x))
            if any(end is not None and end <= start for start, end in zip(records.effective_from, ends, strict=True)):
                raise ValueError("Classification effective_to must follow effective_from")
            records = records.loc[ends.isna() | (ends > cutoff)]
        if records.duplicated(["node_id", "effective_from", "available_at"]).any():
            raise ValueError("Ambiguous classification records at the same information time")
        records = records.sort_values(["node_id", "effective_from", "available_at"]).drop_duplicates("node_id", keep="last")
    elif records.node_id.duplicated().any():
        raise ValueError("Retrospective static metadata requires one row per stock")
    # Missing classifications leave the stock uncovered, rather than joining all unknown stocks.
    return records.set_index("node_id").reindex(node_ids)


def gics_groups(classifications, level="industry_group", min_size=2):
    if isinstance(min_size, bool) or not isinstance(min_size, Integral) or min_size < 1:
        raise ValueError("min_size must be a positive integer")
    if level not in LEVELS or level not in classifications:
        raise ValueError("Missing GICS classification level")
    labels = classifications[level]
    valid = labels.notna() & labels.map(lambda x: isinstance(x, str) and bool(x.strip()))
    return {
        f"gics:{level}:{code}": tuple(classifications.index[valid & labels.eq(code)])
        for code in sorted(set(labels[valid]))
        if int((valid & labels.eq(code)).sum()) >= min_size
    }


def build_gics_family(context, params):
    source = params.get("source", context.metadata.get("gics"))
    if source is None:
        raise ValueError("GICS requires supplied classification records; manual universe labels are not a substitute")
    level, minimum = params.get("level", "industry_group"), params.get("min_size", 2)
    protocol = params.get("metadata_protocol", "point_in_time")
    classifications = select_classifications_at_cutoff(load_classification_records(source), context.node_ids, context.cutoff,
                                                      level=level, metadata_protocol=protocol)
    groups = gics_groups(classifications, level, minimum)
    known_groups = gics_groups(classifications, level, 1)
    return make_hyperedge_family(
        groups, context,
        {edge: {"gics_level": level, "classification_code": edge.split(":", 2)[2]} for edge in groups},
        diagnostics={"missing_labels": list(classifications.index[classifications[level].isna()]),
                     "omitted_small_groups": [edge for edge in known_groups if edge not in groups],
                     "metadata_protocol": protocol},
        state={"classifications": classifications.reset_index().where(pd.notna(classifications.reset_index()), None).to_dict("records")},
    )
