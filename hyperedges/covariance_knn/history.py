"""Coverage eligibility and jointly complete covariance estimation history."""

from numbers import Integral


def select_covariance_history(history, coverage_threshold=0.95, min_rows=30):
    if not 0 < coverage_threshold <= 1:
        raise ValueError("coverage_threshold must be in (0, 1]")
    if isinstance(min_rows, bool) or not isinstance(min_rows, Integral) or min_rows < 2:
        raise ValueError("min_rows must be an integer of at least two")
    structural = history.structural_mask
    if not structural.any():
        raise ValueError("No structurally valid return observations")
    valid = history.mask.loc[structural]
    coverage = valid.mean(axis=0)
    nodes = coverage.index[coverage >= coverage_threshold]
    if len(nodes) < 2:
        raise ValueError("Fewer than two stocks satisfy history coverage")
    complete = valid.loc[:, nodes].all(axis=1)
    returns = history.returns.loc[structural, nodes].loc[complete].copy()
    if len(returns) < min_rows:
        raise ValueError(f"Only {len(returns)} jointly complete rows; require {min_rows}")
    return returns, {
        "coverage": coverage.to_dict(), "eligible_nodes": list(nodes),
        "excluded_nodes": list(coverage.index[coverage < coverage_threshold]),
        "structural_rows": int(structural.sum()), "complete_rows": len(returns),
    }

