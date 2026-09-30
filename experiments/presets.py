from .config import ComponentConfig, SeedBundle, TimeSpan

# Reusable feature and seed choices for experiments. These are used in the experiment config to avoid repetition.

# KEY
# session= "regular" means that the session is the regular trading session, which is 9:30am to 4:00pm Eastern Time.
# output_interval is the interval of the output data. For example, if the native data is 15min and the output_interval is 1h, then the data will be aggregated to 1h intervals.
# overnight_policy="exclude" means that the overnight period is excluded from the calculation of the feature. For example, if the feature is log_return, then the log return will be calculated using only the regular trading session data.
# cross_session_policy="exclude" means that the cross-session period is excluded from the calculation of the feature. For example, if the feature is log_return, then the log return will be calculated using only the regular trading session data.

RETURN_FEATURES = (
    ComponentConfig(
        "log_returns",
        {
            "field": "close",
            "overnight_policy": "exclude",
        },
    ),
    ComponentConfig(
        "standardize",
        {
            "fit_scope": "training_fold",
            "axis": "per_node_per_feature",
        },
    ),
)

RETURN_VOLUME_FEATURES = (
    ComponentConfig(
        "market_features",
        {
            "features": (
                "log_return",
                "log1p_volume",
                "realized_volatility",
            ),
            "volatility_window": TimeSpan(20, "steps"),
            "overnight_policy": "exclude",
        },
    ),
    ComponentConfig(
        "standardize",
        {
            "fit_scope": "training_fold",
            "axis": "per_node_per_feature",
        },
    ),
)

COUNT_FEATURES = (
    ComponentConfig("log1p", {"field": "case_count"}),
    ComponentConfig(
        "standardize",
        {
            "fit_scope": "training_fold",
            "axis": "per_node_per_feature",
        },
    ),
)


SMOKE_SEEDS = (
    SeedBundle(
        initialization=1001,
        batching=1002,
        hyperedge_candidates=1003,
        tuning=1004,
        uncertainty=1005,
    ),
)

RESEARCH_SEEDS = tuple(
    SeedBundle(
        initialization=base + 1,
        batching=base + 2,
        hyperedge_candidates=base + 3,
        tuning=base + 4,
        uncertainty=base + 5,
    )
    for base in (1000, 2000, 3000, 4000, 5000)
)