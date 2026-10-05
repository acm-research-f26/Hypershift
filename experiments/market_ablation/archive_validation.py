"""Validate saved economic paths without recomputing forecasts or portfolios."""
from pathlib import Path
import json
import numpy as np
from .storage import digest_file


STRATEGIES = ("long_only", "long_short", "persistence_cash")
CASES = tuple(f"cost{cost}-borrow{borrow:g}" for cost in (0, 1, 5, 10) for borrow in (0, .03, .1))
RISK_KEYS = ("Sharpe", "Sortino", "IR", "Calmar", "MaxDD", "VaR95", "VaR99", "ES95", "ES99")
PRIMARY_FIELDS = ("net_returns", "gross_returns", "turnover", "gross_exposure", "net_exposure",
                  "transaction_costs", "borrowing_costs", "carried_unquoted_exposure",
                  "corporate_action_liquidation_costs", "corporate_cash_flows")
SHARED_FIELDS = ("net_returns", "session_returns", "bad_quotes", "bad_actions",
                 "corporate_action_liquidation_costs", "corporate_cash_flows")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_arrays(path, required):
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == set(required), f"Unexpected portfolio schema: {path}")
        arrays = {key: saved[key] for key in saved.files}
    for key, value in arrays.items():
        _require(value.dtype.kind in "bifu" and np.isfinite(value).all(), f"Nonfinite/non-numeric {key}: {path}")
        _require(value.ndim == 1, f"Unaligned portfolio axis {key}: {path}")
    return arrays


def expected_execution(dataset, origins):
    """Derive the specified delayed execution support from archived forecast origins."""
    origins = np.asarray(origins)
    if dataset.interval == "1d":
        execution = origins + 1
        keep = execution + 1 < len(dataset.times)
        keep[keep] &= dataset.starts[execution[keep] + 1] <= dataset.times[origins[-1] + 1]
    else:
        execution = origins + 2
        keep = execution < len(dataset.times)
        keep[keep] &= dataset.sessions[execution[keep]] == dataset.sessions[origins[keep]]
    return execution[keep]


def _session_check(arrays, sessions, session_ids, metrics, path):
    net, daily = arrays["net_returns"], arrays["session_returns"]
    ruined = bool(np.any(net <= -1))
    if ruined:
        _require(not len(daily) and metrics.get("reason") == "portfolio_ruin"
                 and metrics.get("status") == "undefined", f"Unflagged portfolio ruin: {path}")
        _require(all(metrics.get(key) is None for key in RISK_KEYS), f"Risk ratios available after portfolio ruin: {path}")
    else:
        expected = np.expm1(np.bincount(sessions, weights=np.log1p(net))[session_ids])
        _require(daily.shape == expected.shape and np.allclose(daily, expected, rtol=1e-12, atol=1e-14),
                 f"Session returns disagree with execution returns: {path}")
    _require(arrays["bad_quotes"].dtype == np.bool_ and arrays["bad_actions"].dtype == np.bool_,
             f"Economic validity flags must be Boolean: {path}")
    for key in ("net_returns", "bad_quotes", "bad_actions", "corporate_action_liquidation_costs", "corporate_cash_flows"):
        _require(arrays[key].shape == net.shape, f"Economic path support differs for {key}: {path}")
    _require(np.all(arrays["corporate_action_liquidation_costs"] >= 0), f"Negative liquidation costs: {path}")


def verify_portfolio_archive(directory, dataset, origins):
    """Read every path value and certify support, wealth, flags, and archive hashes."""
    directory = Path(directory)
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    execution = expected_execution(dataset, origins)
    if not len(execution):
        _require(metrics.get("status") == "undefined" and metrics.get("reason") == "no_executable_signals",
                 f"Missing explicit empty-execution status: {metrics_path}")
        _require(not list(directory.rglob("*.npz")), f"Unexpected empty-support paths: {directory}")
        return {"status": "undefined", "reason": "no_executable_signals", "files": []}
    _require(set(metrics["strategies"]) == set(STRATEGIES), f"Missing portfolio strategy: {metrics_path}")
    sessions = dataset.sessions[execution]
    session_ids = np.unique(sessions)
    paths = {directory / f"{kind}.npz" for kind in STRATEGIES}
    paths.update(directory / "sensitivities" / kind / f"{case}.npz" for kind in STRATEGIES for case in CASES)
    _require(paths == set(directory.rglob("*.npz")), f"Incomplete or unexpected portfolio path set: {directory}")
    checks = []
    for kind in STRATEGIES:
        variants = metrics["strategies"][kind]
        _require(set(variants) == set(CASES), f"Missing cost/borrowing case: {metrics_path}")
        primary_path = directory / f"{kind}.npz"
        primary = _read_arrays(primary_path, (*PRIMARY_FIELDS, "execution", "sessions", "session_ids",
                                              "session_returns", "benchmark_sessions", "wealth", "drawdown",
                                              "bad_quotes", "bad_actions"))
        _require(primary["execution"].dtype == np.int64 and np.array_equal(primary["execution"], execution),
                 f"Delayed execution support disagrees with forecasts: {primary_path}")
        _require(np.array_equal(primary["sessions"], sessions) and np.array_equal(primary["session_ids"], session_ids),
                 f"Portfolio session support disagrees with dataset: {primary_path}")
        for key in PRIMARY_FIELDS:
            _require(primary[key].dtype == np.float64 and primary[key].shape == execution.shape,
                     f"Invalid primary path {key}: {primary_path}")
        wealth = np.r_[1., np.cumprod(1 + primary["net_returns"])]
        drawdown = wealth / np.maximum.accumulate(wealth) - 1
        _require(primary["wealth"].shape == wealth.shape and np.array_equal(primary["wealth"], wealth),
                 f"Wealth disagrees with execution returns: {primary_path}")
        _require(primary["drawdown"].shape == drawdown.shape and np.array_equal(primary["drawdown"], drawdown),
                 f"Drawdown disagrees with wealth: {primary_path}")
        _require(len(primary["benchmark_sessions"]) in (0, len(session_ids)), f"Unaligned benchmark: {primary_path}")
        for key in ("turnover", "gross_exposure", "transaction_costs", "borrowing_costs", "carried_unquoted_exposure"):
            _require(np.all(primary[key] >= 0), f"Negative {key}: {primary_path}")
        _require(np.all(np.abs(primary["net_exposure"]) <= primary["gross_exposure"] + 1e-12),
                 f"Net exposure exceeds gross exposure: {primary_path}")
        if kind == "persistence_cash":
            _require(all(np.all(primary[key] == 0) for key in PRIMARY_FIELDS)
                     and np.all(primary["wealth"] == 1), f"Naive persistence portfolio is not cash: {primary_path}")
        for case in CASES:
            path = directory / "sensitivities" / kind / f"{case}.npz"
            arrays = _read_arrays(path, SHARED_FIELDS)
            values = variants[case]
            _require(arrays["net_returns"].shape == execution.shape, f"Sensitivity execution support differs: {path}")
            _session_check(arrays, sessions, session_ids, values, path)
            incomplete = bool(arrays["bad_actions"].any() or (dataset.interval == "1d" and arrays["bad_quotes"].any()))
            if incomplete:
                _require(values.get("accounting_status") == "incomplete_economic_valuation"
                         and all(values.get(key) is None for key in RISK_KEYS), f"Unflagged incomplete accounting: {path}")
            else:
                _require(values.get("accounting_status") == "complete_on_executable_support", f"Missing accounting status: {path}")
            if not metrics.get("benchmark_accounting_complete"):
                _require(values.get("IR") is None and values.get("IR_undefined_reason") == "benchmark_economic_valuation_incomplete",
                         f"Unflagged incomplete benchmark: {path}")
            if case == "cost5-borrow0.03":
                _require(all(np.array_equal(arrays[key], primary[key]) for key in SHARED_FIELDS),
                         f"Primary path differs from its sensitivity: {primary_path}")
                _require(values.get("MaxDD_all_execution_marks") is None if incomplete else
                         np.isclose(values["MaxDD_all_execution_marks"], -drawdown.min(), rtol=1e-12, atol=1e-14),
                         f"MaxDD differs from archived drawdown: {primary_path}")
            if kind == "persistence_cash":
                _require(np.all(arrays["net_returns"] == 0), f"Naive persistence sensitivity is not cash: {path}")
            checks.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path),
                           "accounting_status": values["accounting_status"], "status": values["status"]})
        checks.append({"path": str(primary_path), "bytes": primary_path.stat().st_size, "sha256": digest_file(primary_path)})
    return {"status": "verified", "execution_marks": len(execution), "market_sessions": len(session_ids),
            "metrics_sha256": digest_file(metrics_path), "files": checks}
