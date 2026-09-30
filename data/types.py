"""Shared observation containers for the data-preparation pipeline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObservationPanel:
    """Aligned raw observations and masks, with values shaped [time, node, field].

    Observation masks identify finite aggregates. Eligibility masks additionally
    apply the experiment's bar policy. Availability times are declared estimates,
    not measured provider publication times. Frozen fields do not make contained
    NumPy arrays or dictionaries immutable.
    """

    timestamps: pd.DatetimeIndex
    node_ids: tuple[str, ...]
    field_names: tuple[str, ...]
    values: np.ndarray                 # [time, node, field]; raw aggregates
    observation_mask: np.ndarray       # finite values, independent of policy
    eligible_mask: np.ndarray          # observation_mask AND bar eligibility
    session_ids: pd.DatetimeIndex
    bar_starts: pd.DatetimeIndex
    availability_times: pd.DatetimeIndex
    expected_minutes: np.ndarray       # [time]
    observed_minutes: np.ndarray       # [time, node]
    is_partial: np.ndarray             # [time]
    units: dict[str, str]
    provenance: dict[str, object]
