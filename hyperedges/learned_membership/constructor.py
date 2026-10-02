"""Variable-size hard memberships with independent binary gates and surrogate gradients."""

from dataclasses import asdict
from numbers import Integral

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ..common.validation import validate_positive_integer
from ..common.runtime import runtime_family_from_incidence
from ..common.types import ConstructorSpec, HyperedgeFamily, validate_hyperedge_family

DEFAULTS = {
    "slots": 16, "initial_size": 6, "selected_probability": 0.9, "unselected_probability": 0.001,
    "temperature": 0.5, "min_size": 3, "max_size": 25, "size_weight": 0.01,
    "duplicate_weight": 0.01, "confidence_weight": 0.001, "duplicate_threshold": 0.95,
}


def initialize_membership_logits(node_ids, slots=16, seed=0, params=None, *, observed_nodes=None):
    settings = {**DEFAULTS, **(params or {}), "slots": slots}
    validate_positive_integer(slots, "slots")
    validate_positive_integer(settings["initial_size"], "initial size")
    high, low = settings["selected_probability"], settings["unselected_probability"]
    if not 0 < low < 0.5 < high < 1:
        raise ValueError("Initialization requires unselected probability < 0.5 < selected probability")
    observed = np.ones(len(node_ids), dtype=bool) if observed_nodes is None else np.asarray(observed_nodes, dtype=bool)
    logits = np.full((len(node_ids), slots), np.log(low / (1 - low)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    choices = np.flatnonzero(observed)
    for slot in range(slots):
        selected = rng.choice(choices, size=min(settings["initial_size"], len(choices)), replace=False)
        logits[selected, slot] = np.log(high / (1 - high))
    return torch.from_numpy(logits)


def sample_binary_memberships(logits, temperature=0.5, training=True, *, generator=None):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Concrete temperature must be positive and finite")
    if training:
        uniform = torch.rand(logits.shape, generator=generator, device="cpu").to(logits.device)
        uniform = uniform.clamp(torch.finfo(logits.dtype).eps, 1 - torch.finfo(logits.dtype).eps)
        soft = torch.sigmoid((logits + torch.log(uniform) - torch.log1p(-uniform)) / temperature)
        hard = (soft >= 0.5).to(logits.dtype)
        return hard + (soft - soft.detach())
    return (logits >= 0).to(logits.dtype)


def validate_learned_edges(memberships, size_bounds=(3, 25)):
    minimum, maximum = size_bounds
    validate_positive_integer(minimum, "minimum learned size")
    validate_positive_integer(maximum, "maximum learned size")
    if minimum > maximum:
        raise ValueError("Learned size bounds must be ordered")
    sizes = memberships.detach().sum(dim=0)
    return (sizes >= minimum) & (sizes <= maximum)


def membership_regularization(probabilities, params=None):
    settings = {**DEFAULTS, **(params or {})}
    sizes = probabilities.sum(dim=0)
    minimum, maximum = settings["min_size"], settings["max_size"]
    size_loss = ((F.relu(minimum - sizes) / minimum).square()
                 + (F.relu(sizes - maximum) / maximum).square()).mean()
    confidence_loss = (probabilities * (1 - probabilities)).mean()
    slots = probabilities.shape[1]
    if slots > 1:
        normalized = F.normalize(probabilities.T, dim=1, eps=1e-12)
        similarities = normalized @ normalized.T
        mask = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
        duplicate_loss = F.relu(similarities[mask] - settings["duplicate_threshold"]).square().mean()
    else:
        duplicate_loss = probabilities.sum() * 0
    total = (settings["size_weight"] * size_loss + settings["duplicate_weight"] * duplicate_loss
             + settings["confidence_weight"] * confidence_loss)
    return {"size": size_loss, "duplicates": duplicate_loss, "confidence": confidence_loss, "total": total}


class LearnedMembershipConstructor(nn.Module):
    trainable_membership = True

    def __init__(self, spec, seed=0):
        super().__init__()
        self.spec, self.seed = spec, seed
        self.settings = {**DEFAULTS, **spec.params}
        if set(spec.params) - DEFAULTS.keys():
            raise ValueError("Unknown learned membership parameters")
        for name in ("slots", "initial_size", "min_size", "max_size"):
            validate_positive_integer(self.settings[name], name)
        if self.settings["min_size"] > self.settings["max_size"]:
            raise ValueError("Learned size bounds must be ordered")
        for name in ("size_weight", "duplicate_weight", "confidence_weight"):
            if not np.isfinite(self.settings[name]) or self.settings[name] < 0:
                raise ValueError("Regularization weights must be finite and nonnegative")
        if not 0 <= self.settings["duplicate_threshold"] < 1:
            raise ValueError("Duplicate threshold must be in [0, 1)")
        if not np.isfinite(self.settings["temperature"]) or self.settings["temperature"] <= 0:
            raise ValueError("Concrete temperature must be positive")
        self.node_ids = ()
        self.register_parameter("logits", None)
        self.register_buffer("observed_nodes", torch.empty(0, dtype=torch.bool))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer("rng_state", generator.get_state())
        self.provenance = {}
        self.frozen = False

    def fit(self, context):
        if self.logits is not None:
            raise RuntimeError("Create a new learned constructor for each training fold")
        observed = (context.history.mask.to_numpy() & context.history.structural_mask[:, None]).any(axis=0)
        if observed.sum() < self.settings["min_size"]:
            raise ValueError("Too few training-observed stocks for the learned minimum group size")
        self.node_ids = tuple(context.node_ids)
        self.settings["max_size"] = min(self.settings["max_size"], int(observed.sum()))
        self.logits = nn.Parameter(initialize_membership_logits(self.node_ids, self.settings["slots"], self.seed,
                                                              self.settings, observed_nodes=observed))
        self.observed_nodes = torch.as_tensor(observed, dtype=torch.bool)
        self.provenance = {"cutoff": context.cutoff, "fold_id": context.fold_id, "seed": self.seed,
                           "builder_version": 1, "parameters": self.settings.copy(),
                           "unobserved_nodes": [node for node, valid in zip(self.node_ids, observed, strict=True) if not valid]}

    def forward(self, active_nodes=None, training=None):
        if self.logits is None:
            raise RuntimeError("Fit the membership constructor before use")
        training = self.training if training is None else training
        if self.frozen and training:
            raise ValueError("Frozen memberships cannot sample or train")
        generator = torch.Generator(device="cpu")
        generator.set_state(self.rng_state.cpu())
        memberships = sample_binary_memberships(self.logits, self.settings["temperature"], training, generator=generator)
        if training:
            self.rng_state.copy_(generator.get_state().to(self.rng_state.device))
        memberships = memberships * self.observed_nodes[:, None]
        valid = validate_learned_edges(memberships, (self.settings["min_size"], self.settings["max_size"]))
        probabilities = torch.sigmoid(self.logits) * self.observed_nodes[:, None]
        regularization = membership_regularization(probabilities, self.settings)["total"]
        if active_nodes is None:
            active_nodes = torch.ones(len(self.node_ids), dtype=torch.bool, device=self.logits.device)
        return runtime_family_from_incidence(
            self.spec.instance_id, self.node_ids, tuple(f"learned:slot:{i}" for i in range(self.settings["slots"])),
            memberships, active_nodes, valid_edges=valid, regularization=regularization,
            attributes={"full_membership_sizes": memberships.detach().sum(dim=0).cpu().tolist()},
        )

    def export_state(self):
        if self.logits is None:
            raise RuntimeError("Cannot export an unfitted membership module")
        return {"spec": asdict(self.spec), "seed": self.seed, "node_ids": self.node_ids,
                "settings": self.settings.copy(), "logits": self.logits.detach().cpu().numpy().copy(),
                "observed_nodes": self.observed_nodes.cpu().numpy().copy(),
                "rng_state": self.rng_state.cpu().numpy().copy(), "provenance": self.provenance.copy(), "frozen": self.frozen}

    @classmethod
    def from_export_state(cls, state):
        from experiments.config import TimeSpan
        spec_data = dict(state["spec"])
        spec_data["history_window"] = TimeSpan(**spec_data["history_window"])
        module = cls(ConstructorSpec(**spec_data), state["seed"])
        module.node_ids, module.settings = tuple(state["node_ids"]), dict(state["settings"])
        logits = torch.as_tensor(state["logits"], dtype=torch.float32).clone()
        if logits.shape != (len(module.node_ids), module.settings["slots"]) or not torch.isfinite(logits).all():
            raise ValueError("Invalid learned checkpoint logits")
        module.logits = nn.Parameter(logits, requires_grad=not state["frozen"])
        module.observed_nodes = torch.as_tensor(state["observed_nodes"], dtype=torch.bool).clone()
        if module.observed_nodes.shape != (len(module.node_ids),):
            raise ValueError("Invalid learned checkpoint observation axis")
        module.rng_state = torch.as_tensor(state["rng_state"], dtype=torch.uint8).clone()
        module.provenance, module.frozen = dict(state["provenance"]), bool(state["frozen"])
        if module.frozen:
            module.eval()
        return module


def freeze_learned_memberships(module):
    if module.logits is None:
        raise RuntimeError("Fit the membership module before freezing")
    module.eval()
    module.logits.requires_grad_(False)
    module.frozen = True
    runtime = module(training=False)
    # Export original full memberships; runtime masks are kept separate from historical incidence.
    values = (module.logits.detach() >= 0) & module.observed_nodes[:, None]
    valid = validate_learned_edges(values, (module.settings["min_size"], module.settings["max_size"]))
    edge_ids = [edge for edge, keep in zip(runtime.edge_ids, valid.cpu().tolist(), strict=True) if keep]
    incidence = pd.DataFrame(values[:, valid].cpu().numpy(), index=module.node_ids, columns=edge_ids, dtype=bool)
    family = HyperedgeFamily(module.spec.instance_id, "learned_membership", incidence,
                             diagnostics=summarize_learned_memberships(module), provenance=module.provenance.copy(),
                             state={"checkpoint": module.export_state()})
    validate_hyperedge_family(family)
    return family


def summarize_learned_memberships(module):
    with torch.no_grad():
        probabilities = torch.sigmoid(module.logits) * module.observed_nodes[:, None]
        hard = (probabilities >= 0.5)
        sizes = hard.sum(dim=0)
        valid = validate_learned_edges(hard, (module.settings["min_size"], module.settings["max_size"]))
        memberships = [tuple(torch.nonzero(hard[:, index]).flatten().tolist()) for index in range(hard.shape[1])]
        return {"sizes": sizes.cpu().tolist(), "expected_sizes": probabilities.sum(dim=0).cpu().tolist(),
                "empty_slots": int((sizes == 0).sum()), "invalid_slots": int((~valid).sum()),
                "duplicate_slots": len(memberships) - len(set(memberships)),
                "overlapping_stocks": int((hard.sum(dim=1) > 1).sum()),
                "mean_uncertainty": float((probabilities * (1 - probabilities)).mean()), "frozen": module.frozen}
