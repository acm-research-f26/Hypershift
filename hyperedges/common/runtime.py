"""Mask-aware runtime incidence, retaining family identity and autograd."""

import numpy as np
import torch

from .types import FamilyTensor


def runtime_family_from_incidence(instance_id, node_ids, edge_ids, incidence, active_nodes, *,
                                  edge_weights=None, valid_edges=None, attributes=None, regularization=None):
    if incidence.ndim != 2 or incidence.shape != (len(node_ids), len(edge_ids)):
        raise ValueError("Runtime incidence axes must match declared node and edge IDs")
    active = torch.as_tensor(active_nodes, device=incidence.device)
    if active.dtype != torch.bool or active.shape != (len(node_ids),):
        raise ValueError("active_nodes must be Boolean with one entry per canonical stock")
    weights = (torch.ones(len(edge_ids), device=incidence.device, dtype=incidence.dtype)
               if edge_weights is None else torch.as_tensor(edge_weights, device=incidence.device, dtype=incidence.dtype))
    if weights.shape != (len(edge_ids),) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Runtime edge weights must be finite and nonnegative")
    masked = incidence * active[:, None]
    degrees = masked.sum(dim=0)
    edge_mask = degrees.detach() >= 2
    if valid_edges is not None:
        edge_mask &= torch.as_tensor(valid_edges, device=incidence.device, dtype=torch.bool)
    masked = masked * edge_mask[None, :]
    return FamilyTensor(instance_id, tuple(node_ids), tuple(edge_ids), masked, weights, active, edge_mask,
                        (masked * weights[None, :]).sum(dim=1), masked.sum(dim=0), attributes or {}, regularization)


def materialize_family_inputs(snapshot, learned_modules=None, active_nodes=None, training=False, *, device=None, dtype=torch.float32):
    if active_nodes is None:
        active_nodes = np.ones(len(snapshot.node_ids), dtype=bool)
    if learned_modules is None:
        from ..learned_membership import LearnedMembershipConstructor
        learned_modules = {name: LearnedMembershipConstructor.from_export_state(state)
                           for name, state in snapshot.learned_references.items()}
        if training and learned_modules:
            raise ValueError("Training must supply optimizer-owned learned modules, rather than checkpoint copies")
    if set(learned_modules) != set(snapshot.learned_references):
        raise ValueError("Learned module identities must match snapshot references")
    families = []
    for family in snapshot.families:
        if family.node_ids != tuple(snapshot.node_ids):
            raise ValueError("Snapshot family stock axes must match")
        tensor = torch.as_tensor(family.incidence.to_numpy(copy=True), device=device, dtype=dtype)
        weights = [family.attributes.get(edge, {}).get("weight", 1.0) for edge in family.edge_ids]
        families.append(runtime_family_from_incidence(family.instance_id, family.node_ids, family.edge_ids,
                                                     tensor, active_nodes, edge_weights=weights, attributes=family.attributes))
    for name, module in learned_modules.items():
        if tuple(module.node_ids) != tuple(snapshot.node_ids):
            raise ValueError("Learned stock axes must match the snapshot")
        if device is not None and module.logits.device != torch.device(device):
            raise ValueError("Move the optimizer-owned learned module to the consumer device first")
        families.append(module(active_nodes=active_nodes, training=training))
    return tuple(families)
