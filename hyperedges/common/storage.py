"""Versioned JSON snapshots and checkpoints; no executable pickle payloads."""

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .validation import _validate_identifiers
from .types import ContextFeatures, HyperedgeFamily, HyperedgeSnapshot, _utc_timestamp, validate_hyperedge_family

__all__ = ["save_hyperedge_snapshot", "load_hyperedge_snapshot"]


def _encode(value):
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, pd.DataFrame):
        return {"__kind__": "frame", "index": _encode(tuple(value.index)),
                "columns": _encode(tuple(value.columns)), "values": _encode(value.to_numpy()),
                "dtypes": [str(x) for x in value.dtypes], "index_name": value.index.name, "columns_name": value.columns.name}
    if isinstance(value, pd.Timestamp):
        return {"__kind__": "timestamp", "value": value.isoformat()}
    if isinstance(value, np.ndarray):
        return {"__kind__": "array", "dtype": str(value.dtype), "shape": list(value.shape), "values": _encode(value.tolist())}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"__kind__": "float", "value": str(value)}
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "values": [_encode(x) for x in value]}
    if isinstance(value, list):
        return [_encode(x) for x in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Artifact dictionary keys must be strings")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _encode(asdict(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Artifact value cannot be serialized: {type(value).__name__}")


def _decode(value):
    if isinstance(value, list):
        return [_decode(x) for x in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__kind__")
    if kind == "timestamp":
        return pd.Timestamp(value["value"])
    if kind == "float":
        return float(value["value"])
    if kind == "tuple":
        return tuple(_decode(x) for x in value["values"])
    if kind == "array":
        return np.asarray(_decode(value["values"]), dtype=value["dtype"]).reshape(value["shape"])
    if kind == "frame":
        frame = pd.DataFrame(_decode(value["values"]), index=_decode(value["index"]), columns=_decode(value["columns"]))
        for column, dtype in zip(frame.columns, value["dtypes"], strict=True):
            frame[column] = frame[column].astype(dtype)
        frame.index.name, frame.columns.name = value["index_name"], value["columns_name"]
        return frame
    return {key: _decode(item) for key, item in value.items()}


def _artifact_digest(value):
    return sha256(json.dumps(_encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_hyperedge_snapshot(snapshot, destination):
    _validate_snapshot(snapshot)
    payload = _encode(asdict(snapshot))
    envelope = {"schema_version": 1, "sha256": _artifact_digest(asdict(snapshot)), "snapshot": payload}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(envelope, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_hyperedge_snapshot(source):
    envelope = json.loads(Path(source).read_text(encoding="utf-8"))
    if envelope.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot schema")
    payload = _decode(envelope["snapshot"])
    if _artifact_digest(payload) != envelope["sha256"]:
        raise ValueError("Snapshot checksum mismatch")
    payload["families"] = tuple(HyperedgeFamily(**family) for family in payload["families"])
    payload["context_features"] = tuple(ContextFeatures(**feature) for feature in payload["context_features"])
    snapshot = HyperedgeSnapshot(**payload)
    _validate_snapshot(snapshot)
    return snapshot


def _validate_snapshot(snapshot):
    _validate_identifiers(snapshot.node_ids, "node")
    if snapshot.protocol not in {"fold_frozen", "rolling"}:
        raise ValueError("Unknown snapshot protocol")
    if not _utc_timestamp(snapshot.cutoff) <= _utc_timestamp(snapshot.available_at) <= _utc_timestamp(snapshot.effective_time):
        raise ValueError("Invalid snapshot timing")
    identifiers = [family.instance_id for family in snapshot.families]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) & set(snapshot.learned_references):
        raise ValueError("Duplicate snapshot family identifiers")
    for family in snapshot.families:
        validate_hyperedge_family(family)
        if family.node_ids != tuple(snapshot.node_ids):
            raise ValueError("Snapshot family axes are not aligned")
    for name, state in snapshot.learned_references.items():
        if tuple(state["node_ids"]) != tuple(snapshot.node_ids) or state["spec"]["instance_id"] != name:
            raise ValueError("Learned checkpoint identities or axes are not aligned")
        if _utc_timestamp(state["provenance"]["cutoff"]) > _utc_timestamp(snapshot.available_at):
            raise ValueError("Learned checkpoint is unavailable at snapshot completion")
