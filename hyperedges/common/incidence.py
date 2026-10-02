"""Boolean stock-to-hyperedge incidence with a canonical ordered stock axis."""

from collections.abc import Mapping, Sequence

import pandas as pd

from .validation import _validate_identifiers

__all__ = ["incidence_from_groups"]


def incidence_from_groups(node_ids: Sequence[str], groups: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    nodes = _validate_identifiers(node_ids, "node")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("groups must be a nonempty mapping")
    edges = _validate_identifiers(groups, "edge")
    result = pd.DataFrame(False, index=nodes, columns=edges)
    known = set(nodes)
    for edge, members in groups.items():
        members = _validate_identifiers(members, "member")
        if not set(members) <= known:
            raise ValueError(f"Unknown members in {edge!r}")
        result.loc[list(members), edge] = True
    return result

