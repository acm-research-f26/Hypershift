"""Compatibility imports for the original incidence/KNN convenience module.

Implementations live in common/ and covariance_knn/. Prefer those packages
or the top-level hyperedges API for new code.
"""

from .common.incidence import incidence_from_groups
from .covariance_knn import (
    build_covariance_knn_family,
    correlation_knn_hyperedges,
    covariance_to_correlation,
    estimate_covariance,
    knn_groups_from_similarity,
)

__all__ = [
    "incidence_from_groups",
    "build_covariance_knn_family",
    "correlation_knn_hyperedges",
    "covariance_to_correlation",
    "estimate_covariance",
    "knn_groups_from_similarity",
]
