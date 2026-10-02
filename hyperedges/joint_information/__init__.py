"""Public API for the joint_information method."""

from .constructor import (
    generate_candidate_groups,
    split_estimation_selection_history,
    StateEncoder,
    fit_state_encoder,
    encode_joint_states,
    estimate_joint_distribution,
    derive_pairwise_marginals,
    fit_pairwise_maxent,
    score_joint_information,
    compare_joint_models,
    select_information_groups,
    build_joint_information_family,
)

__all__ = [
    "generate_candidate_groups",
    "split_estimation_selection_history",
    "StateEncoder",
    "fit_state_encoder",
    "encode_joint_states",
    "estimate_joint_distribution",
    "derive_pairwise_marginals",
    "fit_pairwise_maxent",
    "score_joint_information",
    "compare_joint_models",
    "select_information_groups",
    "build_joint_information_family",
]
