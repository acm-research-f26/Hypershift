"""An incidence-preserving consumer of existing temporal stock representations.

This Euclidean boundary module does not change the temporal encoder or prescribe
hyperbolic operations. Nonlinear edge processing retains hyperedge identities.
"""

import torch
from torch import nn

from hyperedges.common.runtime import materialize_family_inputs


class HyperedgeConsumer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, *, family_ids=(), learned_modules=None, context_channels=0, family_normalization=True):
        super().__init__()
        if min(in_channels, hidden_channels, out_channels) < 1 or context_channels < 0:
            raise ValueError("Channel dimensions must be positive (context can be zero)")
        self.family_ids = tuple(family_ids)
        if len(set(self.family_ids)) != len(self.family_ids):
            raise ValueError("Family IDs must be unique")
        self.temporal_projection = nn.Linear(in_channels, hidden_channels)
        self.edge_transform = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                                            nn.Linear(hidden_channels, hidden_channels))
        self.family_logits = nn.Parameter(torch.zeros(len(self.family_ids)))
        # Numeric keys permit any string instance ID while remaining compatible with nn.ModuleDict.
        learned_modules = learned_modules or {}
        self.learned_ids = tuple(learned_modules)
        self.membership_modules = nn.ModuleDict({str(index): module for index, module in enumerate(learned_modules.values())})
        if not set(self.learned_ids) <= set(self.family_ids):
            raise ValueError("Learned modules must have declared family IDs")
        self.context_channels = context_channels
        self.context_projection = nn.Linear(context_channels, hidden_channels, bias=False) if context_channels else None
        self.head = nn.Linear(hidden_channels, out_channels)
        self.family_normalization = family_normalization
        self._incidence_cache = {}

    @property
    def learned_modules(self):
        return {name: self.membership_modules[str(index)] for index, name in enumerate(self.learned_ids)}

    def forward(self, stock_representations, snapshot, active_nodes=None, *, context_values=None, context_mask=None, disabled_families=()):
        if stock_representations.ndim == 3:
            return self._forward_batch(stock_representations, snapshot, active_nodes, context_values,
                                       context_mask, disabled_families)
        if stock_representations.ndim != 2 or stock_representations.shape[0] != len(snapshot.node_ids):
            raise ValueError("Temporal representations must be [canonical stocks, channels]")
        active = (torch.ones(len(snapshot.node_ids), dtype=torch.bool, device=stock_representations.device)
                  if active_nodes is None else torch.as_tensor(active_nodes, device=stock_representations.device))
        if active.dtype != torch.bool or active.shape != (len(snapshot.node_ids),):
            raise ValueError("Active stock mask must be Boolean and aligned")
        if not torch.isfinite(stock_representations[active]).all():
            raise ValueError("Active temporal representations must be finite")
        families = materialize_family_inputs(snapshot, self.learned_modules, active_nodes, self.training,
                                             device=stock_representations.device, dtype=stock_representations.dtype)
        if not {family.instance_id for family in families} <= set(self.family_ids):
            raise ValueError("Snapshot contains undeclared consumer families")
        if not set(disabled_families) <= set(self.family_ids):
            raise ValueError("Unknown disabled family")
        direct = self.temporal_projection(torch.where(active[:, None], stock_representations, 0))
        messages, availability, indices = [], [], []
        for family in families:
            if family.instance_id in disabled_families:
                continue
            incidence = family.incidence.to(direct.dtype)
            edge_means = incidence.T @ direct / family.edge_degrees.to(direct.dtype).clamp_min(1)[:, None]
            edges = self.edge_transform(edge_means) * family.edge_mask[:, None]
            messages.append(incidence @ (edges * family.edge_weights[:, None]) /
                            (family.node_degrees.to(direct.dtype).clamp_min(1)[:, None] if self.family_normalization else 1))
            availability.append(family.node_degrees > 0)
            indices.append(self.family_ids.index(family.instance_id))
        combined = direct
        if messages:
            present = torch.stack(availability, dim=1)
            logits = self.family_logits[indices][None, :].expand(len(direct), -1).masked_fill(~present, -torch.inf)
            # Nodes uncovered by every family retain the direct pathway without NaN softmax.
            logits = torch.where(present.any(dim=1, keepdim=True), logits, torch.zeros_like(logits))
            gates = torch.softmax(logits, dim=1) * present
            combined = combined + (torch.stack(messages, dim=1) * gates[:, :, None]).sum(dim=1)
        if self.context_projection is not None:
            if context_values is None or context_mask is None:
                raise ValueError("Configured context input requires values and validity mask")
            values = torch.as_tensor(context_values, device=direct.device, dtype=direct.dtype)
            mask = torch.as_tensor(context_mask, device=direct.device)
            if values.shape != (self.context_channels,) or mask.shape != values.shape or mask.dtype != torch.bool:
                raise ValueError("Context axes and mask must match configured context channels")
            if not torch.isfinite(values[mask]).all():
                raise ValueError("Valid context values must be finite")
            combined = combined + self.context_projection(torch.where(mask, values, 0))[None, :]
        return self.head(combined)

    def _forward_batch(self, x, snapshot, active_nodes, context_values, context_mask, disabled_families):
        if set(self.learned_ids) != set(snapshot.learned_references):
            raise ValueError("Learned module identities must match snapshot references")
        if x.shape[1] != len(snapshot.node_ids) or x.shape[2] != self.temporal_projection.in_features:
            raise ValueError("Batched representations must be [batch, canonical stocks, channels]")
        active = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if active_nodes is None else torch.as_tensor(active_nodes, device=x.device)
        if active.dtype != torch.bool or active.shape != x.shape[:2]:
            raise ValueError("Batched active mask must be Boolean [batch, stocks]")
        if not set(disabled_families) <= set(self.family_ids):
            raise ValueError("Unknown disabled family")
        key = (snapshot.snapshot_id, str(x.device), x.dtype)
        if key not in self._incidence_cache:
            self._incidence_cache = {key: [(family.instance_id,
                torch.tensor(family.incidence.to_numpy(copy=True), device=x.device, dtype=x.dtype),
                torch.tensor([family.attributes.get(edge, {}).get("weight", 1.) for edge in family.edge_ids], device=x.device, dtype=x.dtype))
                for family in snapshot.families]}
        families = list(self._incidence_cache[key])
        for name, module in self.learned_modules.items():
            runtime = module(training=self.training)
            families.append((name, runtime.incidence, runtime.edge_weights))
        if not {name for name, _, _ in families} <= set(self.family_ids):
            raise ValueError("Snapshot contains undeclared family")
        direct = self.temporal_projection(torch.where(active[:, :, None], x, 0))
        messages, available, indices = [], [], []
        for name, incidence, weights in families:
            if name in disabled_families or incidence.shape[1] == 0:
                continue
            # Share [N,E] incidence across origins. Masked matrix products avoid
            # materializing [B,N,E] membership tensors for every family.
            degrees = active.to(x.dtype) @ incidence
            usable = (degrees.detach() >= 2).to(x.dtype)
            edge_means = torch.matmul(incidence.T, direct * active[:, :, None]) / degrees.clamp_min(1)[:, :, None]
            edges = self.edge_transform(edge_means) * usable[:, :, None]
            node_degree = ((usable * weights[None, :]) @ incidence.T) * active
            message = torch.matmul(incidence, edges * weights[None, :, None]) * active[:, :, None]
            if self.family_normalization:
                message = message / node_degree.clamp_min(1)[:, :, None]
            messages.append(message)
            available.append(node_degree > 0)
            indices.append(self.family_ids.index(name))
        combined = direct
        if messages:
            present = torch.stack(available, dim=2)
            logits = self.family_logits[indices][None, None, :].expand_as(present).masked_fill(~present, -torch.inf)
            logits = torch.where(present.any(dim=2, keepdim=True), logits, torch.zeros_like(logits))
            gates = torch.softmax(logits, dim=2) * present
            combined = combined + (torch.stack(messages, dim=2) * gates[:, :, :, None]).sum(dim=2)
        if self.context_projection is not None:
            if context_values is None or context_mask is None:
                raise ValueError("Configured context requires values and masks")
            values = torch.as_tensor(context_values, dtype=x.dtype, device=x.device)
            mask = torch.as_tensor(context_mask, device=x.device)
            if values.shape != (len(x), self.context_channels) or mask.shape != values.shape or mask.dtype != torch.bool:
                raise ValueError("Batched context must be [batch, context_channels]")
            combined = combined + self.context_projection(torch.where(mask, values, 0))[:, None, :]
        return self.head(combined)

    def membership_penalty(self):
        from hyperedges.learned_membership import membership_regularization
        penalties = [membership_regularization(torch.sigmoid(module.logits) * module.observed_nodes[:, None], module.settings)["total"]
                     for module in self.membership_modules.values()]
        return sum(penalties, self.family_logits.sum() * 0)

    def freeze_memberships(self):
        from hyperedges.learned_membership import freeze_learned_memberships
        return tuple(freeze_learned_memberships(module) for module in self.membership_modules.values())
