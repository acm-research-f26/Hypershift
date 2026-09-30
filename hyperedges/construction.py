"""A common output interface for comparing hyperedge construction methods.

The user implements these functions; the assistant supplies tests. Both are
intentionally empty. Sector, correlation, cover, event, and joint-information
constructors can all express their stock groups through the first function.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def incidence_from_groups(
    node_ids: Sequence[str], groups: Mapping[str, Sequence[str]]
) -> pd.DataFrame:

    raise NotImplementedError("Implement incidence_from_groups first")


def correlation_knn_hyperedges(
    returns: pd.DataFrame, *, neighbors: int = 2, absolute: bool = True
) -> pd.DataFrame:

    raise NotImplementedError("Implement after incidence_from_groups")
