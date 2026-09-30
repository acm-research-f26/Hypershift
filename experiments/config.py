from dataclasses import dataclass, field
from typing import Any, Literal

# Contains dataclasses to describe experiment choices. The runner is to resolve and execute components.


@dataclass(frozen=True)
class ComponentConfig:
    """Implementation name and its parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TaskConfig:
    """What to predict, what the model outputs and the evaluation scale"""
    
    kind: Literal["regression", "classification", "ranking"]
    target: ComponentConfig
    output: ComponentConfig
    evaluation_scale: Literal["original", "transformed"]
    
@dataclass(frozen=True)
class TimeSpan:
    """A duration of time and its unit of measure."""
    
    value: int
    unit: Literal[
        "steps", "minutes", "hours", "days", "weeks", "sessions"
    ]
    
@dataclass(frozen=True)
class DatasetGroup:
    """Source observations, node selection, and sampling rules for a dataset."""
    
    name: str
    domain: str
    source: ComponentConfig
    nodes: ComponentConfig
    sampling: ComponentConfig
    

@dataclass(frozen=True)
class SeedBundle:
    """Random seeds for different stages of the experiment."""
    
    initialization: int
    batching: int
    hyperedge_candidates: int
    tuning: int
    uncertainty: int
    
@dataclass(frozen=True)
class AnalysisConfig:
    """A single statistical question and its inference procedure."""
    
    name: str
    statistic: ComponentConfig
    aggregation: ComponentConfig
    uncertainty: ComponentConfig | None
    hypothesis: ComponentConfig | None
    inference_scope: Literal[
        "fixed_forecasts",
        "refit_pipeline",
    ]


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluating the model: Metrics, analyses, alignment, and seed reporting."""
    
    metrics: tuple[ComponentConfig, ...]
    analyses: tuple[AnalysisConfig, ...]
    alignment: ComponentConfig
    seed_reporting: ComponentConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Container for all configuration of a single experiment, including dataset, features, model, training, evaluation, and seeds."""
    
    name: str
    dataset: DatasetGroup

    features: tuple[ComponentConfig, ...]
    task: TaskConfig

    hyperedge_features: tuple[ComponentConfig, ...]
    hyperedge_builders: tuple[ComponentConfig, ...]
    hyperedge_learning: ComponentConfig

    model: ComponentConfig
    geometry: ComponentConfig
    initialization: ComponentConfig

    optimizer: ComponentConfig
    objective: ComponentConfig
    scheduler: ComponentConfig | None
    training: ComponentConfig

    validation: ComponentConfig
    tuning: ComponentConfig

    evaluation: EvaluationConfig
    diagnostics: tuple[ComponentConfig, ...]
    portfolio: ComponentConfig | None

    seeds: tuple[SeedBundle, ...]
    output_directory: str
