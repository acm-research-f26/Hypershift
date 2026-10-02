"""Explicit snapshot availability and sample attachment."""

from dataclasses import replace

from .types import _utc_timestamp

__all__ = ["select_snapshot_for_origin", "attach_snapshot_to_sample"]


def select_snapshot_for_origin(snapshots, origin, *, fold_id=None):
    origin = _utc_timestamp(origin)
    eligible = [snapshot for snapshot in snapshots
                if _utc_timestamp(snapshot.available_at) <= origin
                and _utc_timestamp(snapshot.effective_time) <= origin
                and _utc_timestamp(snapshot.cutoff) <= origin
                and (fold_id is None or snapshot.fold_id == fold_id)]
    if not eligible:
        raise ValueError("No snapshot was available at this forecast origin")
    return max(eligible, key=lambda snapshot: (snapshot.effective_time, snapshot.cutoff, snapshot.snapshot_id))


def attach_snapshot_to_sample(sample, snapshot, *, offline_training=False):
    if tuple(sample.node_ids) != tuple(snapshot.node_ids):
        raise ValueError("Sample and snapshot stock axes must match")
    if offline_training:
        if snapshot.protocol != "fold_frozen" or sample.target_availability > snapshot.cutoff:
            raise ValueError("Offline training attachment requires a frozen fold and training-available labels")
    else:
        select_snapshot_for_origin((snapshot,), sample.origin_time)
    return replace(sample, snapshot_id=snapshot.snapshot_id)
