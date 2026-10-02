# %% [markdown]
# # Hyperedges from scratch: an executable experiment notebook
#
# Start here if you want to create memberships, inspect them, train a forecaster,
# and change the construction recipe without editing the pipeline.
# The companion `hyperedge_walkthrough.ipynb` contains this walkthrough with
# outputs cleared for version control. This file is also a runnable Python script
# and a VS Code `# %%` interactive document.
# The validation helper saves executed copies under ignored `local-data/`.
#
# **Setup:** run this command from the repository root to install the project
# and the `notebooks` dependency group into uv's managed `.venv`:
#
# ```powershell
# uv sync --group notebooks
# ```
#
# Then open the notebook in VS Code/Jupyter, select `.venv/Scripts/python.exe`
# as the kernel, and use **Restart Kernel → Run All**. Dependencies are declared
# in `pyproject.toml` and their resolved versions are recorded in `uv.lock`.
# To run the paired script instead:
#
# ```powershell
# uv run --group notebooks python notebooks/hyperedge_walkthrough.py
# ```
#
# The data below are generated locally: 12 invented stocks, short toy sessions,
# persistent return factors, negative correlations, and one incompletely observed
# stock. The classification fixture is **mock metadata, not real GICS**. Numerical
# results demonstrate the software; they are not evidence of financial performance.
# No credentials, downloads, optional Cover Learning backend, or GPU are required.
#
# You will work through:
# 1. Prices → masks → returns → chronological samples and training-only scaling.
# 2. Member sets → incidence; GICS-shaped labels; three KNN similarity choices.
# 3. Event, joint-information, and Mapper recipes; a custom constructor.
# 4. Hard learned memberships → forecast loss → gradients → frozen evaluation.
# 5. Retrained ablations, checkpoint masking, and persistence context.
# 6. Rolling availability, save/reload, and replacement with real observations.
#
# The predictor here uses flattened return histories plus `HyperedgeConsumer`.
# Substitute your own temporal encoder's `[stocks, channels]` output at that
# boundary when integrating this construction pipeline into a larger model.

# %%
from pathlib import Path
import sys

# Works when launched from the repository root, notebooks/, or its parent.
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "hyperedges" / "common" / "pipeline.py").exists()), None)
if ROOT is None:
    candidate = Path.cwd() / "Hypershift"
    ROOT = candidate if (candidate / "hyperedges" / "common" / "pipeline.py").exists() else None
if ROOT is None:
    raise RuntimeError("Open this notebook inside the Hypershift repository.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from copy import deepcopy
from dataclasses import replace
from importlib import metadata, util
from time import perf_counter
import json

import numpy as np
import pandas as pd
import torch
from IPython import get_ipython
from IPython.display import display
import matplotlib

if get_ipython() is None:
    matplotlib.use("Agg")
else:
    get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt

from data.types import ObservationPanel
from data.windows import make_forecast_samples
from experiments.config import TimeSpan
from experiments.splits import split_forecast_samples
from features.market import log_returns
from features.pipeline import training_feature_mask, transform_forecast_sample
from features.transforms import fit_standardizer, StandardizerState
from hyperedges import (
    ConstructorSpec, HyperedgePipelineConfig, incidence_from_groups,
    correlation_knn_hyperedges, make_construction_context,
    prepare_construction_history, select_covariance_history,
    fit_hyperedge_pipeline, build_hyperedge_snapshot, build_context_features,
    save_hyperedge_snapshot, load_hyperedge_snapshot,
    select_snapshot_for_origin, attach_snapshot_to_sample, register_constructor,
)
from hyperedges.covariance_knn import (
    estimate_covariance, covariance_to_correlation, knn_groups_from_similarity,
)
from hyperedges.common.runtime import materialize_family_inputs
from hyperedges.common.types import make_hyperedge_family
from hyperedges.learned_membership import (
    LearnedMembershipConstructor, summarize_learned_memberships,
)
from src.hypergraph import HyperedgeConsumer
from experiments.hyperedge_ablation import (
    run_hyperedge_ablation, compare_hyperedge_ablations,
    evaluate_frozen_family_masking,
)

def _show(value):
    """Rich notebook tables; readable output when running the paired script."""
    if get_ipython() is None:
        print(value.to_string() if hasattr(value, "to_string") else value)
    else:
        display(value)

print("Repository:", ROOT)
print("Interpreter:", sys.executable)
print("Versions:", {name: metadata.version(name)
                    for name in ("numpy", "pandas", "torch", "matplotlib")})

# %% [markdown]
# ## 1. The control panel
#
# Change these values, then restart the kernel and run all cells. Keep the
# constructor instance IDs stable when comparing ablations: their independent
# seeds derive from those IDs. A constructor's history window, the predictor's
# lookback, and the fold's training interval are three separate choices.
#
# The default construction history is 60 sessions, KNN has five neighbors plus
# its center, and learned membership has 16 independent binary slots. Short
# training runs make this a CPU demonstration. For a research experiment, use
# several chronological folds, seeds, and an aligned tuning budget.
#
# `SAMPLE_STRIDE` subsamples **prediction origins**, equally for all methods. It
# does not subsample construction history. Keep it at 1 for every usable origin.
# Artifacts are written under ignored `local-data/tutorial-artifacts/`.

# %%
SEED = 1003
N_STOCKS = 12
TRAIN_SESSIONS = 60
VALIDATION_SESSIONS = 6
TEST_SESSIONS = 6
BARS_PER_SESSION = 9
BAR_MINUTES = 15
LOOKBACK = 3
SAMPLE_STRIDE = 2
KNN_NEIGHBORS = 5
CONSTRUCTION_WINDOW = TimeSpan(60, "sessions")
LEARNED_SLOTS = 16
HIDDEN_CHANNELS = 16
EPOCHS = 8
PATIENCE = 4
LEARNING_RATE = 0.002

# Explicitly select recipes for the first manually trained model here.
TRAIN_FAMILIES = ("gics", "knn_absolute", "learned")
ABLATION_NAMES = ("temporal_only", "knn", "gics", "gics_knn", "gics_knn_learned")
RUN_PUBLISHED_COVER = False  # Requires your separately installed exact backend version.
RUN_EXTRA_CONSTRUCTOR_TRAINING = True  # Set False to skip the three extra training runs.

assert N_STOCKS > KNN_NEIGHBORS and N_STOCKS >= 6
assert BARS_PER_SESSION >= LOOKBACK + 3 and SAMPLE_STRIDE >= 1
OUT = ROOT / "local-data" / "tutorial-artifacts"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)  # Small CPU matrices are faster without large thread pools.
torch.manual_seed(SEED)
np.random.seed(SEED)
plt.rcParams.update({"figure.figsize": (10, 3.8), "axes.grid": True})

def _finish_plot(filename):
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=140, bbox_inches="tight")
    if get_ipython() is not None:
        plt.show()
    plt.close()

# %% [markdown]
# ## 2. Generate prices and an observation panel
#
# The panel preserves the expected time grid, stock order, observed/eligible
# masks, session boundaries, and information availability. A missing close is
# unknown; it is not zero. Each close is declared available one second after
# the bar ends. The first return of each session and returns across gaps are
# excluded by the existing `log_returns` function.
#
# The business-day calendar and nine-bar sessions here are intentionally toy
# schedules. Real experiments must use the exchange calendar, complete expected
# bar grid, and their actual eligibility/availability policy.
#
# Three persistent latent factors create correlated groups. Some stocks load
# with the opposite sign; later stocks have higher volatility. Stock `S11` has
# repeated missing closes, making it fail KNN coverage while retaining its place
# in the panel and classification family.

# %%
def _make_demo_panel():
    rng = np.random.default_rng(SEED)
    days = pd.bdate_range("2025-01-06", periods=(TRAIN_SESSIONS +
                            VALIDATION_SESSIONS + TEST_SESSIONS))
    starts = pd.DatetimeIndex([
        day.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        + pd.Timedelta(minutes=BAR_MINUTES * bar)
        for day in days for bar in range(BARS_PER_SESSION)
    ]).tz_convert("UTC")
    ends = starts + pd.Timedelta(minutes=BAR_MINUTES)
    sessions = pd.DatetimeIndex(np.repeat(days.to_numpy(), BARS_PER_SESSION))
    nodes = tuple(f"S{i:02d}" for i in range(N_STOCKS))
    factor = np.zeros((len(ends), 3))
    for t in range(1, len(ends)):
        factor[t] = 0.45 * factor[t - 1] + rng.normal(0, 0.0008, 3)
    returns = rng.normal(0, 0.00045, (len(ends), N_STOCKS))
    for i in range(N_STOCKS):
        group = min(2, 3 * i // N_STOCKS)
        sign = -1 if i in (3, 7) else 1
        returns[:, i] += sign * factor[:, group]
    lagged_stock = min(8, N_STOCKS - 1)
    returns[1:, lagged_stock] += 0.2 * returns[:-1, 0]  # A planted lagged cross-stock signal.
    returns[:, 8:] *= 2.5
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    close[4::10, -1] = np.nan
    valid = np.isfinite(close)
    panel = ObservationPanel(
        timestamps=ends, node_ids=nodes, field_names=("close",),
        values=close[:, :, None], observation_mask=valid[:, :, None],
        eligible_mask=valid[:, :, None].copy(), session_ids=sessions,
        bar_starts=starts, availability_times=ends + pd.Timedelta(seconds=1),
        expected_minutes=np.full(len(ends), BAR_MINUTES),
        observed_minutes=valid.astype(int) * BAR_MINUTES,
        is_partial=np.zeros(len(ends), dtype=bool), units={"close": "USD"},
        provenance={"source": "synthetic tutorial", "seed": SEED,
                    "calendar": "toy business days; shortened sessions"},
    )
    return panel, days

panel, session_days = _make_demo_panel()
raw_returns, return_mask = log_returns(panel)
print("Panel [time, stocks, fields]:", panel.values.shape)
_show(pd.DataFrame(panel.values[:6, :, 0], index=panel.timestamps[:6],
                  columns=panel.node_ids).round(3))
plt.plot(panel.timestamps, panel.values[:, :4, 0], linewidth=0.8)
plt.title("Invented close prices: four demo stocks")
plt.ylabel("Close (USD)")
plt.legend(panel.node_ids[:4], ncol=4)
_finish_plot("synthetic_prices.png")

# %% [markdown]
# ## 3. Split before fitting anything
#
# Every example predicts the next adjacent same-session log return. Its input
# has shape `[lookback, stocks, features]`; its target has shape `[stocks]`.
# `split_forecast_samples` requires training labels to be available by the
# cutoff, and validation labels to be available before test model selection.
# We use the first validation session's start as the fitting cutoff.
#
# Fit input scaling on the union of actual training inputs, counting overlapping
# coordinates once. Apply that saved state to validation and test. **Targets
# remain unscaled log returns.** The constructors receive raw historical returns
# from the panel, independently of this predictor input scaling.

# %%
def _session_open(position):
    return (session_days[position].tz_localize("America/New_York")
            + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")

fold = {
    "train_start": _session_open(0),
    "fit_cutoff": _session_open(TRAIN_SESSIONS),
    "validation_end": _session_open(TRAIN_SESSIONS + VALIDATION_SESSIONS),
    "test_end": panel.availability_times[-1] + pd.Timedelta(seconds=1),
}
all_samples = tuple(make_forecast_samples(panel, LOOKBACK))
raw_splits = split_forecast_samples(all_samples, **fold)
raw_splits = {name: tuple(samples[::SAMPLE_STRIDE])
              for name, samples in raw_splits.items()}
assert all(raw_splits.values())
_show(pd.DataFrame([
    {"split": name, "origins": len(samples), "first_origin": samples[0].origin_time,
     "last_origin": samples[-1].origin_time,
     "supported_targets": sum(int((s.eligible_nodes & s.target_mask).sum())
                              for s in samples)}
    for name, samples in raw_splits.items()
]))

axes = {"node_ids": panel.node_ids, "feature_names": ("log_return",)}
fit_mask = training_feature_mask(panel.timestamps, raw_splits["train"],
                                fit_cutoff=fold["fit_cutoff"], **axes)
scaler = fit_standardizer(raw_returns[:, :, None],
                         return_mask[:, :, None] & fit_mask,
                         fit_mask.any(axis=(1, 2)), **axes)
splits = {name: tuple(transform_forecast_sample(s, scaler) for s in samples)
          for name, samples in raw_splits.items()}
assert not fit_mask[panel.timestamps >= fold["fit_cutoff"]].any()
_show(pd.DataFrame({"mean": scaler.mean[:, 0], "scale": scaler.scale[:, 0],
                   "training_input_count": scaler.count[:, 0]}, index=panel.node_ids))
print("One prepared input:", splits["train"][0].values.shape,
      "target units:", splits["train"][0].target_units)

# %% [markdown]
# ## 4. Memberships first: what is a hyperedge?
#
# A hyperedge is a named set of stocks. An incidence matrix has stocks on its
# rows and hyperedges on its columns: `H[i, e] = True` means stock `i` belongs to
# edge `e`. One stock can belong to several edges. An uncovered stock still
# remains on the canonical stock axis and can use the direct temporal pathway.
# Different edge IDs may deliberately have identical memberships.
#
# `incidence_from_groups` validates identifiers and preserves supplied order.
# An empty mapping is rejected by this convenience function; the pipeline can
# still represent empty discovered families and a zero-constructor experiment.

# %%
manual_groups = {
    "hand:abc": panel.node_ids[:3],
    "hand:overlap": panel.node_ids[2:6],
    "hand:abc_copy": panel.node_ids[:3],
}
manual_H = incidence_from_groups(panel.node_ids, manual_groups)
_show(manual_H.astype(int))
assert manual_H["hand:abc"].equals(manual_H["hand:abc_copy"])
assert manual_H.loc[panel.node_ids[2]].sum() == 3

def _edge_table(family):
    rows = []
    for edge in family.edge_ids:
        members = family.incidence.index[family.incidence[edge]].tolist()
        rows.append({"edge_id": edge, "size": len(members), "members": ", ".join(members)})
    return pd.DataFrame(rows, columns=("edge_id", "size", "members"))

def _family_summary(families):
    rows = []
    for family in families:
        H = family.incidence
        sizes = H.sum(axis=0)
        members = [tuple(np.flatnonzero(H[e])) for e in H.columns]
        rows.append({"instance": family.instance_id, "method": family.method,
                     "edges": H.shape[1], "covered_stocks": int(H.any(axis=1).sum()),
                     "overlapping_stocks": int((H.sum(axis=1) > 1).sum()),
                     "min_size": int(sizes.min()) if len(sizes) else 0,
                     "max_size": int(sizes.max()) if len(sizes) else 0,
                     "duplicate_memberships": len(members) - len(set(members))})
    return pd.DataFrame(rows)

def _inspect_family(family, limit=8):
    _show(_family_summary((family,)))
    _show(_edge_table(family).head(limit))

# %% [markdown]
# ## 5. Covariance → correlation → KNN, one step at a time
#
# Select a historical window ending at the fitting cutoff. Coverage is computed
# over structurally valid return positions. Stocks need at least 95% coverage;
# after choosing stocks, use jointly complete rows and require at least 30.
# An excluded stock stays on the full axis for other constructors and forecasting.
#
# Estimate one sample covariance matrix from that common window. Convert it to
# correlation by dividing by marginal standard deviations. For each center,
# choose five neighbors, with ties ordered by stock identifier.
#
# - **Absolute correlation:** strong positive or negative associations count.
# - **Signed correlation:** the largest signed correlations rank first.
# - **Absolute covariance:** volatility affects ranking as well as dependence.
#
# The strict convenience function handles complete finite input only. Missing
# observations are handled by `select_covariance_history` before calling it.

# %%
history = prepare_construction_history(panel, fold["fit_cutoff"], CONSTRUCTION_WINDOW)
complete_returns, coverage = select_covariance_history(history, 0.95, 30)
covariance = estimate_covariance(complete_returns)
correlation = covariance_to_correlation(covariance)
knn_groups = knn_groups_from_similarity(correlation, neighbors=KNN_NEIGHBORS, absolute=True)
strict_H = correlation_knn_hyperedges(complete_returns, neighbors=KNN_NEIGHBORS)
assert strict_H.equals(incidence_from_groups(complete_returns.columns, knn_groups))
print("Sessions used:", history.session_ids.nunique(),
      "complete rows:", coverage["complete_rows"],
      "KNN exclusions:", coverage["excluded_nodes"])
_show(pd.Series(coverage["coverage"], name="return_coverage").to_frame())
_show(correlation.round(2))
print("Center S00:", knn_groups["knn:S00"])
assert (strict_H.sum(axis=0) == KNN_NEIGHBORS + 1).all()

# %% [markdown]
# ## 6. Supplied classifications and reusable constructor specifications
#
# GICS creates one edge per represented industry-group code, omitting singleton
# groups. Unknown classifications leave stocks uncovered. It needs an actual
# classification source when used on real stocks. **The project's manual sector
# and peer-group labels are not GICS and must not be substituted silently.**
#
# This fixture uses obvious `MOCK_*` labels to exercise the same metadata API.
# `effective_from` and `available_at` constrain cutoff selection. Use a supplied
# CSV/DataFrame with authentic provider provenance for a real experiment.
#
# A specification is `(unique instance ID, registered method, params, window)`.
# Several instances of the same registered method are valid. Each has its own
# seed, settings, historical window, and provenance. The list can be any length,
# including zero. Adding another method requires no orchestration changes.

# %%
mock_gics = pd.DataFrame({
    "node_id": panel.node_ids,
    "industry_group": [f"MOCK_{min(2, 3 * i // N_STOCKS)}" for i in range(N_STOCKS)],
    "source": "synthetic tutorial fixture; not a GICS provider",
    "effective_from": pd.Timestamp("2024-01-01", tz="UTC"),
    "available_at": pd.Timestamp("2024-01-01", tz="UTC"),
})
context = make_construction_context(panel, fold["fit_cutoff"], seed=SEED,
                                    metadata={"gics": mock_gics}, fold_id="tutorial:0")

gics_spec = ConstructorSpec("gics", "gics", {"level": "industry_group", "min_size": 2},
                            CONSTRUCTION_WINDOW)
knn_spec = ConstructorSpec("knn_absolute", "covariance_knn", {
    "neighbors": KNN_NEIGHBORS, "similarity": "absolute_correlation",
    "coverage_threshold": 0.95, "min_rows": 30,
}, CONSTRUCTION_WINDOW)
signed_spec = replace(knn_spec, instance_id="knn_signed",
                      params={**knn_spec.params, "similarity": "signed_correlation"})
covariance_spec = replace(knn_spec, instance_id="knn_covariance",
                          params={**knn_spec.params, "similarity": "absolute_covariance"})
learned_spec = ConstructorSpec("learned", "learned_membership", {
    "slots": LEARNED_SLOTS, "initial_size": 6, "temperature": 0.5,
    "selected_probability": 0.9, "unselected_probability": 0.001,
    "min_size": 3, "max_size": 25,
    "size_weight": 0.01, "duplicate_weight": 0.01, "confidence_weight": 0.001,
}, CONSTRUCTION_WINDOW)
recipes = {spec.instance_id: spec for spec in
           (gics_spec, knn_spec, signed_spec, covariance_spec, learned_spec)}

baseline_config = HyperedgePipelineConfig((gics_spec, knn_spec))
baseline_pipeline = fit_hyperedge_pipeline(context, baseline_config)
baseline_snapshot = build_hyperedge_snapshot(baseline_pipeline)
baseline_families = {f.instance_id: f for f in baseline_snapshot.families}
_show(_family_summary(baseline_snapshot.families))
_inspect_family(baseline_families["gics"])
_inspect_family(baseline_families["knn_absolute"])
assert baseline_families["gics"].incidence.loc[panel.node_ids[-1]].any()
assert not baseline_families["knn_absolute"].incidence.loc[panel.node_ids[-1]].any()
assert baseline_families["knn_absolute"].incidence.index.tolist() == list(panel.node_ids)

# %% [markdown]
# Compare the three KNN recipes side by side. They share the same ordered stock
# axis but have distinct family identities. Correlation rankings are invariant
# to positive per-stock rescaling; raw covariance rankings are not.

# %%
knn_variants = HyperedgePipelineConfig((knn_spec, signed_spec, covariance_spec))
variant_snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, knn_variants))
_show(pd.DataFrame([
    {"instance": f.instance_id, "S00_members": ", ".join(
        f.incidence.index[f.incidence["knn:S00"]])}
    for f in variant_snapshot.families
]))
scaled_returns = complete_returns.copy()
scaled_returns.iloc[:, 1] *= 20
scaled_correlation = covariance_to_correlation(estimate_covariance(scaled_returns))
np.testing.assert_allclose(correlation, scaled_correlation, atol=1e-12)
assert not np.allclose(covariance, estimate_covariance(scaled_returns), atol=1e-10)

H = baseline_families["knn_absolute"].incidence
plt.imshow(H.to_numpy(dtype=int), cmap="Blues", vmin=0, vmax=1, aspect="auto")
plt.xticks(range(H.shape[1]), H.columns, rotation=75, ha="right")
plt.yticks(range(H.shape[0]), H.index)
plt.title("KNN Boolean incidence: stocks × center edges")
plt.grid(False)
_finish_plot("knn_incidence.png")

# %% [markdown]
# ## 7. Other historical construction methods
#
# These constructors fit independently on past data. They do not consume GICS,
# KNN, learned memberships, or each other's proposals. Build them with exactly
# the same pipeline API; add their specifications to a training configuration
# to consume their messages. An empty discovered family is a valid result.
#
# ### 7a. Event Dowker: recurring common witnesses
#
# Fit each stock's historical participation thresholds. Each timestamp is an
# event transaction; retain groups simultaneously participating at recurring
# timestamps. Pairwise co-occurrence alone cannot create an unwitnessed triple.
# Attributes retain support count and the group's jointly observable denominator.
# Missing observations are unknown rather than nonparticipation evidence.
#
# The example creates separate downside and upside instances. Quantiles, support,
# size bounds, edge budget, and candidate budget are editable research choices.

# %%
events_down = ConstructorSpec("events_down", "event_dowker", {
    "quantiles": [0.2, 0.8], "direction": "down", "min_support": 6,
    "size_bounds": [3, 4], "edge_budget": 24, "candidate_budget": 10_000,
}, CONSTRUCTION_WINDOW)
events_up = replace(events_down, instance_id="events_up",
                    params={**events_down.params, "direction": "up"})
recipes.update({s.instance_id: s for s in (events_down, events_up)})
event_snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, (events_down, events_up)))
_inspect_family(event_snapshot.families[0])
_show(pd.DataFrame.from_dict(event_snapshot.families[0].attributes, orient="index").head(5))

from hyperedges.event_dowker import mine_recurring_groups
pairwise_triangle = pd.DataFrame([[1, 1, 0], [1, 0, 1], [0, 1, 1]],
                                 columns=["A", "B", "C"], dtype=bool)
assert mine_recurring_groups(pairwise_triangle, 1, (3, 3)) == ()
witnessed = pd.concat([pairwise_triangle,
                      pd.DataFrame([[True, True, True]], columns=["A", "B", "C"])],
                     ignore_index=True)
assert mine_recurring_groups(witnessed, 1, (3, 3)) == (("A", "B", "C"),)
print("A pairwise triangle has no triple edge; a shared triple witness creates one.")

# %% [markdown]
# ### 7b. Joint information: dependence beyond a pairwise model
#
# Start with unrestricted triples. Split the constructor's training history into
# an earlier estimation segment and a later internal holdout. Fit the state
# encoder on estimation observations, estimate a smoothed joint probability
# table, and derive all pairwise marginals from that same table. Fit the pairwise
# maximum-entropy model and compare both models on the holdout.
#
# The residual is `KL(P_joint || Q_pairwise)` in **bits**. Fits that fail to
# converge cannot create edges. This recipe caps candidate cost and requires
# holdout gain; it is not a dependence-aware significance test. A history with
# only pairwise structure can legitimately retain few or zero edges.
#
# First verify the interpretation with exact XOR/parity states: all pairs are
# independent, but the triple has one bit of connected information.

# %%
from hyperedges.joint_information import (
    estimate_joint_distribution, derive_pairwise_marginals,
    fit_pairwise_maxent, score_joint_information,
)
parity_states = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
P = estimate_joint_distribution(parity_states, smoothing=0, cardinality=2)
Q = fit_pairwise_maxent(derive_pairwise_marginals(P), tolerance=1e-10, max_iterations=1000)
parity_bits = score_joint_information(P, Q)
assert Q["converged"] and np.isclose(parity_bits, 1)
print("Exact parity residual:", parity_bits, "bit")

information_spec = ConstructorSpec("joint", "joint_information", {
    "bin_spec": {"kind": "binary_sign"}, "smoothing": 0.5,
    "min_observations": 100, "tolerance": 1e-8, "max_iterations": 1000,
    "candidate_budget": 40, "order": 3, "estimation_fraction": 0.8,
    "selection_spec": {"edge_budget": 12, "min_holdout_gain_bits": 0.0,
                       "min_selection_observations": 30},
}, CONSTRUCTION_WINDOW)
recipes[information_spec.instance_id] = information_spec
information_snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, (information_spec,)))
information_family = information_snapshot.families[0]
_inspect_family(information_family)
info_diagnostics = pd.DataFrame(information_family.diagnostics["results"])
_show(info_diagnostics.sort_values("joint_information_bits", ascending=False).head(8))
# For diagnostics only, use replace(information_spec, params={**information_spec.params,
#                                                            "mode": "diagnose"}).

# %% [markdown]
# ### 7c. Mapper: overlapping descriptor-space clusters
#
# Stocks are the points. Build one historical descriptor vector per stock;
# choose a lens; cover the lens space with overlapping elements; cluster stocks
# locally inside each element. Each resulting member set becomes a hyperedge.
# The optional nerve is a diagnostic, not the stock incidence matrix itself.
#
# Change `lens_spec` to `{"kind": "pca", "components": 1}`, cover bins/overlap,
# descriptors, or clustering radius to experiment. Larger radii tend to merge
# local clusters; overlaps can put a stock into several groups.

# %%
mapper_spec = ConstructorSpec("mapper", "mapper_cover", {
    "descriptor_spec": {"features": ["mean", "std", "q10", "q90"],
                        "coverage_threshold": 0.95, "standardize": True},
    "lens_spec": {"kind": "feature", "feature": "std"},
    "cover_spec": {"bins": 4, "overlap": 0.4},
    "clustering_spec": {"kind": "single_linkage", "radius": 2.5},
    "min_size": 2,
}, CONSTRUCTION_WINDOW)
recipes[mapper_spec.instance_id] = mapper_spec
mapper_snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, (mapper_spec,)))
_inspect_family(mapper_snapshot.families[0])
_show(pd.DataFrame.from_dict(mapper_snapshot.families[0].attributes, orient="index").head(8))

# %% [markdown]
# ### 7d. Published Cover Learning: an explicit optional backend
#
# This baseline optimizes a cover using its specified neighborhood graph and
# published objective. It is separate from memberships trained by forecast loss.
# The adapter requires `shapediscover` at the **exact declared installed version**.
# Install that backend separately in a compatible environment after choosing
# your experiment's version; then set `RUN_PUBLISHED_COVER = True` and rerun.
#
# The guarded example supplies every graph, objective, and membership conversion
# choice. With the flag false, Run All skips only this backend-dependent method.
# We do not substitute another algorithm when the backend is missing.

# %%
cover_snapshot = None
if RUN_PUBLISHED_COVER:
    if util.find_spec("shapediscover") is None:
        raise ImportError("Install your chosen shapediscover version before enabling this cell.")
    cover_version = metadata.version("shapediscover")
    cover_spec = ConstructorSpec("cover", "cover_learning", {
        "descriptor_spec": mapper_spec.params["descriptor_spec"],
        "graph_spec": {"neighbors": 5, "algorithm": "umap", "backend_version": cover_version},
        "backend_spec": {"name": "shapediscover", "version": cover_version},
        "objective_spec": {"preset": "ShapeDiscover", "parameters": {
            "n_cover": 6, "loss_weights": [1, 10, 1, 10],
            "initialization_algorithm": "spectral_clustering",
            "model": "set_function", "n_max_iter": 100,
        }},
        "membership_rule": {"threshold": 0.5, "min_size": 2},
    }, CONSTRUCTION_WINDOW)
    recipes[cover_spec.instance_id] = cover_spec
    cover_snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, (cover_spec,)))
    _inspect_family(cover_snapshot.families[0])
else:
    print("Cover Learning skipped: RUN_PUBLISHED_COVER=False; no backend was installed by the demo.")

# %% [markdown]
# ## 8. Toggle any combination and verify independence
#
# Construct arbitrary-length lists from the recipe dictionary. Here is one
# snapshot with GICS, three KNN variants, two event families, joint information,
# and Mapper. Construction is independent: adding/removing/reordering other
# recipes must leave the retained historical memberships and seeds unchanged.
#
# **To train a different combination**, change `TRAIN_FAMILIES` in the control
# panel, or pass the new config to `run_hyperedge_ablation` below. The same API
# works for temporal-only: `HyperedgePipelineConfig(constructors=())`.

# %%
all_historical_specs = (gics_spec, knn_spec, signed_spec, covariance_spec,
                        events_down, events_up, information_spec, mapper_spec)
all_historical_pipeline = fit_hyperedge_pipeline(context, all_historical_specs)
all_historical_snapshot = build_hyperedge_snapshot(all_historical_pipeline)
_show(_family_summary(all_historical_snapshot.families))

retained_knn = next(f for f in all_historical_snapshot.families if f.instance_id == "knn_absolute")
assert retained_knn.incidence.equals(baseline_families["knn_absolute"].incidence)
assert retained_knn.provenance["seed"] == baseline_families["knn_absolute"].provenance["seed"]
reverse_snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, (knn_spec, gics_spec)))
assert reverse_snapshot.families[0].incidence.equals(retained_knn.incidence)
empty_snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, ()))
assert empty_snapshot.families == ()
later_values = panel.values.copy()
later_values[panel.timestamps >= fold["fit_cutoff"]] *= 10
later_panel = replace(panel, values=later_values)
same_cutoff_context = make_construction_context(later_panel, fold["fit_cutoff"], seed=SEED,
                          metadata={"gics": mock_gics}, fold_id=context.fold_id)
same_cutoff_snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(same_cutoff_context,
                                                                     baseline_config))
for original, changed in zip(baseline_snapshot.families, same_cutoff_snapshot.families, strict=True):
    assert original.incidence.equals(changed.incidence)
print("Verified independent KNN, a valid empty pipeline, and no influence from later prices.")

def _config_for(*instance_ids, context_providers=()):
    return HyperedgePipelineConfig(tuple(recipes[name] for name in instance_ids),
                                   tuple(context_providers))

if RUN_EXTRA_CONSTRUCTOR_TRAINING:
    extra_results = compare_hyperedge_ablations(context, {
        "gics_knn_events": _config_for("gics", "knn_absolute", "events_down"),
        "gics_knn_information": _config_for("gics", "knn_absolute", "joint"),
        "gics_knn_mapper": _config_for("gics", "knn_absolute", "mapper"),
    }, splits["train"], splits["validation"], splits["test"],
       seed=SEED, hidden_channels=HIDDEN_CHANNELS, epochs=EPOCHS,
       patience=PATIENCE, learning_rate=LEARNING_RATE)
    extra_table = pd.DataFrame({name: r["test_metrics"] for name, r in extra_results.items()}).T
    _show(extra_table)
    extra_table.to_csv(OUT / "optional_constructor_ablations.csv")

# %% [markdown]
# ## 9. Register your own constructor without changing orchestration
#
# A historical plugin implements `fit(context)` and `build(context)` and packages
# named groups with `make_hyperedge_family`. Its context already has its own
# window, parameters, seed, and ordered stocks. It cannot see other families.
#
# This simple example sorts eligible stocks by historical volatility and groups
# them into buckets. It demonstrates registration rather than a proposed
# validated financial method. Rerunning the cell is safe within this kernel.
# Change its recipe rather than the registry name when trying new parameters.

# %%
class VolatilityBuckets:
    def __init__(self, spec, seed):
        self.spec, self.seed, self.family = spec, seed, None

    def fit(self, context):
        valid = context.history.mask & context.history.structural_mask[:, None]
        std = context.history.returns.where(valid).std().dropna()
        ordered = sorted(std.index, key=lambda node: (std[node], node))
        groups = {}
        for i, members in enumerate(np.array_split(ordered, self.spec.params["buckets"])):
            if len(members) >= 2:
                groups[f"volatility:bucket:{i}"] = tuple(members.tolist())
        self.family = make_hyperedge_family(groups, context,
                                           state={"volatility": std.to_dict()})

    def build(self, context):
        return deepcopy(self.family)

if not globals().get("_volatility_registered", False):
    register_constructor("tutorial_volatility", VolatilityBuckets,
                         required_params=("buckets",), allowed_params=("buckets",))
    _volatility_registered = True
custom_spec = ConstructorSpec("volatility", "tutorial_volatility", {"buckets": 3},
                              TimeSpan(30, "sessions"))
recipes[custom_spec.instance_id] = custom_spec
custom_snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, (custom_spec,)))
_inspect_family(custom_snapshot.families[0])
# Try _config_for("knn_absolute", "volatility") with the same training functions.

# %% [markdown]
# ## 10. Runtime masks and the incidence-preserving consumer
#
# A saved family is historical membership. A `FamilyTensor` is its runtime view:
# inactive stocks are masked, degrees are recomputed within each family, and
# edges with fewer than two active members are suppressed. Historical incidence
# is preserved. Edge scores/attributes and learned model weights remain separate.
#
# The consumer processes stock → edge → stock messages separately per family,
# normalizes inside each family, and learns family importance. Its direct stock
# pathway remains available even when no relational family covers a stock.
# You supply the same ordered stock representations used by the snapshot.

# %%
active_demo = np.zeros(N_STOCKS, dtype=bool)
active_demo[:2] = True
runtime_demo = materialize_family_inputs(baseline_snapshot, active_nodes=active_demo, training=False)
_show(pd.DataFrame([
    {"family": f.instance_id, "original_edges": len(f.edge_ids),
     "active_edges": int(f.edge_mask.sum()),
     "active_members_per_edge": f.edge_degrees.tolist()}
    for f in runtime_demo
]))
assert baseline_families["gics"].incidence.equals(baseline_snapshot.families[0].incidence)

def _prepare_sample(sample):
    """Flatten [lookback,N,F] to [N,lookback*F] without adding future information."""
    active = sample.eligible_nodes & sample.mask.all(axis=(0, 2))
    values = np.where(sample.mask, sample.values, 0)
    x = torch.as_tensor(values.transpose(1, 0, 2).reshape(len(sample.node_ids), -1),
                        dtype=torch.float32)
    support = active & sample.target_mask & np.isfinite(sample.target)
    target = torch.as_tensor(sample.target, dtype=torch.float32)
    return x, active, support, target

first_x, first_active, first_support, first_target = _prepare_sample(splits["train"][0])
print("Consumer inputs [stocks, temporal channels]:", tuple(first_x.shape))

# %% [markdown]
# ## 11. Hard learned memberships and a visible training loop
#
# The optional learned constructor initializes 16 stock/slot binary gates.
# Memberships overlap and can cross classification boundaries. Slots have
# variable sizes; hard slots outside size bounds carry no message but still
# receive regularization. During training, hard forward values use a soft
# straight-through gradient. Evaluation uses deterministic probability ≥ 0.5.
#
# Build the model **with the pipeline's live modules**, then build the optimizer
# from `model.parameters()`. Passing detached incidence would break membership
# learning. Add `model.membership_penalty()` to the supervised loss.
# Default regularization weights are pilot controls; because targets here are
# small raw returns, check their scale relative to forecast MSE in a real run.
#
# Training uses the training-cutoff structure throughout this fold, including
# earlier training examples. This is the declared **offline frozen-fold protocol**,
# not a reconstruction of the graph available at every historical training time.
# Validation selects the best checkpoint; test scoring follows selection.

# %%
training_config = _config_for(*TRAIN_FAMILIES)
training_pipeline = fit_hyperedge_pipeline(context, training_config)
training_snapshot = build_hyperedge_snapshot(training_pipeline)
torch.manual_seed(SEED)
model = HyperedgeConsumer(first_x.shape[1], HIDDEN_CHANNELS,
                          family_ids=TRAIN_FAMILIES,
                          learned_modules=training_pipeline.learned_modules)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
initial_logits = {name: module.logits.detach().clone()
                  for name, module in model.learned_modules.items()}

model.train()
prediction = model(first_x, training_snapshot, first_active).flatten()
forecast_loss = (prediction[first_support] - first_target[first_support]).square().mean()
penalty = model.membership_penalty()
forecast_gate_gradients = {
    name: torch.autograd.grad(forecast_loss, module.logits, retain_graph=True,
                              allow_unused=True)[0]
    for name, module in model.learned_modules.items()
}
optimizer.zero_grad(set_to_none=True)
(forecast_loss + penalty).backward()
for name, module in model.learned_modules.items():
    assert forecast_gate_gradients[name] is not None and forecast_gate_gradients[name].abs().sum() > 0
    assert module.logits.grad is not None and module.logits.grad.abs().sum() > 0
    runtime = module(training=True)
    assert set(runtime.incidence.detach().unique().tolist()) <= {0.0, 1.0}
    print(name, "forecast-only membership gradient norm:", float(forecast_gate_gradients[name].norm()),
          "total gradient norm:", float(module.logits.grad.norm()))
    _show(pd.DataFrame(torch.sigmoid(module.logits).detach().numpy(),
                      index=panel.node_ids,
                      columns=[f"slot:{i}" for i in range(LEARNED_SLOTS)]).iloc[:, :5].round(3))
print("First forecast MSE:", float(forecast_loss.detach()),
      "membership penalty:", float(penalty.detach()))
optimizer.zero_grad(set_to_none=True)

attached_training = attach_snapshot_to_sample(splits["train"][0], training_snapshot,
                                              offline_training=True)
attached_validation = attach_snapshot_to_sample(splits["validation"][0], training_snapshot)
assert attached_training.snapshot_id == attached_validation.snapshot_id

# %% [markdown]
# The next cell contains the full optimization loop. Each step scores eligible
# stocks at one origin. It handles masks, updates the forecast model and live
# membership logits together, and tracks target-weighted training/validation MSE.
# The notebook keeps the loop explicit so you can add a temporal encoder, batching,
# or different losses later. No validation/test labels update model parameters.

# %%
def _evaluate(model, samples, snapshot, context_by_origin=None):
    model.eval()
    squared = absolute = 0.0
    correct = count = 0
    forecasts = []
    with torch.no_grad():
        for sample in samples:
            x, active, support, target = _prepare_sample(sample)
            kwargs = {}
            if context_by_origin is not None:
                features = context_by_origin[sample.origin_time]
                kwargs = {"context_values": np.concatenate([f.values for f in features]),
                          "context_mask": np.concatenate([f.mask for f in features])}
            pred = model(x, snapshot, active, **kwargs).flatten().numpy()
            errors = pred[support] - sample.target[support]
            squared += float(np.square(errors).sum())
            absolute += float(np.abs(errors).sum())
            correct += int(((pred[support] >= 0) == (sample.target[support] >= 0)).sum())
            count += int(support.sum())
            forecasts.append({"origin_time": sample.origin_time,
                              "prediction": pred, "support": support.copy()})
    if count == 0:
        raise ValueError("No supported targets in this evaluation split.")
    return {"mse": squared / count, "mae": absolute / count,
            "directional_accuracy": correct / count, "target_count": count}, forecasts

training_trace = []
best_state, best_mse, stalled = None, float("inf"), 0
started = perf_counter()
for epoch in range(EPOCHS):
    model.train()
    total, count = 0.0, 0
    for sample in splits["train"]:
        x, active, support, target = _prepare_sample(sample)
        if not support.any():
            continue
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, training_snapshot, active).flatten()
        mse = (prediction[support] - target[support]).square().mean()
        loss = mse + model.membership_penalty()
        loss.backward()
        optimizer.step()
        total += float(mse.detach()) * int(support.sum())
        count += int(support.sum())
    validation_metrics, _ = _evaluate(model, splits["validation"], training_snapshot)
    training_trace.append({"epoch": epoch + 1, "train_mse": total / count,
                           "validation_mse": validation_metrics["mse"]})
    if validation_metrics["mse"] < best_mse:
        best_state = deepcopy(model.state_dict())
        best_mse, stalled = validation_metrics["mse"], 0
    else:
        stalled += 1
        if stalled >= PATIENCE:
            break
assert best_state is not None
model.load_state_dict(best_state)
print("Manual training seconds:", round(perf_counter() - started, 2))
_show(pd.DataFrame(training_trace))
for name, module in model.learned_modules.items():
    change = float((module.logits.detach() - initial_logits[name]).abs().max())
    assert change > 0
    print(name, "largest logit change:", change)

trace = pd.DataFrame(training_trace).set_index("epoch")
plt.plot(trace.index, trace.train_mse, "o-", label="train")
plt.plot(trace.index, trace.validation_mse, "o-", label="validation")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Raw log-return MSE")
plt.title("Validation selects a checkpoint; the test block is untouched here")
plt.legend()
_finish_plot("training_loss.png")

# %% [markdown]
# ### Freeze the selected memberships and score test once
#
# Restore the best validation checkpoint first, freeze its learned memberships,
# and build the final snapshot. Frozen evaluation is deterministic. Historical
# GICS/KNN memberships were never optimizer parameters. Inspect Boolean sets and
# slot-size diagnostics separately from probabilities and family gates.
#
# A short run may change logits without moving them across the 0.5 boundary.
# That is expected with confident initialization; it is not a guarantee that
# the learned memberships have found a useful structure. Experiment with training
# duration, initialization probabilities, temperature, and regularization.

# %%
frozen_learned_families = model.freeze_memberships()
final_snapshot = build_hyperedge_snapshot(training_pipeline)
manual_test_metrics, manual_forecasts = _evaluate(model, splits["test"], final_snapshot)
_show(pd.Series(manual_test_metrics, name="manual_model_test").to_frame())
_show(_family_summary((*final_snapshot.families, *frozen_learned_families)))
for family in frozen_learned_families:
    _inspect_family(family)
for name, module in model.learned_modules.items():
    print(name, summarize_learned_memberships(module))
for family in final_snapshot.families:
    if family.instance_id in baseline_families:
        assert family.incidence.equals(baseline_families[family.instance_id].incidence)

example = splits["test"][0]
_show(pd.DataFrame({"prediction": manual_forecasts[0]["prediction"],
                   "actual": example.target, "scoring_support": manual_forecasts[0]["support"]},
                  index=panel.node_ids))

# %% [markdown]
# ## 12. Retrained ablations versus frozen-checkpoint masking
#
# `compare_hyperedge_ablations` repeats construction and training for each config
# on identical prepared splits and evaluation support. Use it when asking how
# well a model trains with a particular family combination. Shared consumer
# initialization is aligned; learned memberships are fitted anew for each run.
#
# `evaluate_frozen_family_masking` removes messages from an already trained model
# without retraining. It answers a different question about that checkpoint's
# reliance on a family. Family gates renormalize over remaining available families.
# Do not mix these two protocols in one claimed ablation result.
#
# The table reports original-unit error, directional accuracy, parameter count,
# and runtime. Toy metrics can vary and need not improve with more constructors.
# Report coverage, folds/seeds, uncertainty, and construction cost for real work.

# %%
ablation_configs = {
    "temporal_only": _config_for(),
    "knn": _config_for("knn_absolute"),
    "gics": _config_for("gics"),
    "gics_knn": _config_for("gics", "knn_absolute"),
    "gics_knn_learned": _config_for("gics", "knn_absolute", "learned"),
}
ablation_results = compare_hyperedge_ablations(context,
    {name: ablation_configs[name] for name in ABLATION_NAMES},
    splits["train"], splits["validation"], splits["test"],
    seed=SEED, hidden_channels=HIDDEN_CHANNELS, epochs=EPOCHS,
    patience=PATIENCE, learning_rate=LEARNING_RATE)
ablation_table = pd.DataFrame([
    {"experiment": name, **result["test_metrics"],
     "parameters": result["parameter_count"],
     "construction_seconds": result["construction_seconds"],
     "training_seconds": result["training_seconds"]}
    for name, result in ablation_results.items()
]).set_index("experiment")
_show(ablation_table)
assert ablation_table.target_count.nunique() == 1
ablation_table.to_csv(OUT / "retrained_ablations.csv")

manual_result = {"model": model, "snapshot": final_snapshot}
mask_results = {
    "all_trained_families": manual_test_metrics,
    **{f"mask:{family}": evaluate_frozen_family_masking(
        manual_result, splits["test"], (family,))["test_metrics"]
       for family in TRAIN_FAMILIES},
}
_show(pd.DataFrame(mask_results).T)

# %% [markdown]
# ## 13. PH is an actual context input, separate from memberships
#
# PH uses historical timestamps as points and selected stocks as coordinates.
# It returns persistence summaries with validity masks, not stock-member sets.
# We use the native exact H0/H1 backend on a small, explicitly bounded cloud,
# avoiding an external persistence dependency. For larger clouds/higher dimensions,
# choose a declared supported backend and its exact version.
#
# Fit coordinate scaling on the training construction history. Build actual PH
# summaries from observations available at each forecast origin, even while
# memberships stay frozen. Then fit summary-feature scaling on **training
# origins only**. Under this offline fold protocol, training examples use
# coordinate scaling fitted at the fold cutoff; online reconstruction would
# require its own earlier fitted scaling state.
#
# The cell trains a GICS+KNN+PH experiment through the same runner and verifies
# that enabling PH leaves GICS and KNN incidence unchanged. The per-origin mapping
# is passed explicitly; the runner refuses to invent or omit configured context.

# %%
from hyperedges.ph_context import fit_context_transform, transform_context_features

ph_spec = ConstructorSpec("ph", "ph_context", {
    "node_ids": panel.node_ids[:3],
    "cloud_spec": {"max_points": 12, "min_rows": 20},
    "backend_spec": {"name": "native_rips", "version": "1", "max_simplices": 10_000},
    "dimensions": [0, 1],
    "summary_spec": ["finite_count", "total_persistence", "max_persistence", "mean_persistence"],
}, TimeSpan(15, "sessions"))
ph_config = _config_for("gics", "knn_absolute", context_providers=(ph_spec,))
ph_pipeline = fit_hyperedge_pipeline(context, ph_config)
ph_snapshot = build_hyperedge_snapshot(ph_pipeline)
for original, with_ph in zip(baseline_snapshot.families, ph_snapshot.families, strict=True):
    assert original.incidence.equals(with_ph.incidence)

ph_by_origin_raw = {}
for split_samples in splits.values():
    for sample in split_samples:
        origin_context = make_construction_context(panel, sample.origin_time, seed=SEED,
                           metadata={"gics": mock_gics}, fold_id=context.fold_id)
        ph_by_origin_raw[sample.origin_time] = build_context_features(ph_pipeline, origin_context)

ph_transform = fit_context_transform(
    (ph_by_origin_raw[s.origin_time][0] for s in splits["train"]),
    fit_cutoff=fold["fit_cutoff"])
ph_by_origin = {origin: tuple(transform_context_features(f, ph_transform) for f in features)
                for origin, features in ph_by_origin_raw.items()}
first_ph = ph_by_origin[splits["test"][0].origin_time][0]
_show(pd.DataFrame({"feature": first_ph.names, "scaled_value": first_ph.values,
                   "valid": first_ph.mask}))
print("PH channels:", len(first_ph.names), "cloud points:", first_ph.provenance["cloud_points"])

ph_result = run_hyperedge_ablation(context, ph_config, splits["train"],
    splits["validation"], splits["test"], seed=SEED, hidden_channels=HIDDEN_CHANNELS,
    epochs=EPOCHS, patience=PATIENCE, learning_rate=LEARNING_RATE,
    context_features=ph_by_origin)
assert ph_result["model"].context_channels == len(first_ph.names)
_show(pd.Series(ph_result["test_metrics"], name="gics_knn_ph_test").to_frame())
# To persist a PH experiment, save its per-origin policy, coordinate scaling,
# ph_transform, provider recipe, snapshot, predictor weights, and feature scaler.

# %% [markdown]
# ## 14. Separate rolling-snapshot construction experiment
#
# Frozen evaluation reuses training-cutoff memberships. A rolling experiment
# explicitly refreshes historical constructors on later available history.
# **Data cutoff**, **build availability**, and **effective time** are distinct.
# A future-built snapshot cannot be selected for an earlier prediction.
#
# Here we create a new historical snapshot between two validation origins,
# declare a one-second construction delay, and choose the latest available
# snapshot for the second origin. This demonstrates scheduling and construction;
# the frozen-fold ablation runner above does not train rolling-snapshot models.
# For that experiment, use `select_snapshot_for_origin` inside a separately
# specified chronological training/evaluation loop and supply each selected
# snapshot to the consumer. Record build cost and turnover.

# %%
rolling_config = replace(baseline_config, protocol="rolling")
rolling_pipeline = fit_hyperedge_pipeline(context, rolling_config)
rolling_initial = build_hyperedge_snapshot(rolling_pipeline,
                     available_at=context.cutoff + pd.Timedelta(seconds=1))
refresh_cutoff = splits["validation"][0].origin_time
refresh_context = make_construction_context(panel, refresh_cutoff, seed=SEED,
                    metadata={"gics": mock_gics}, fold_id=context.fold_id)
rolling_refreshed = build_hyperedge_snapshot(rolling_pipeline, refresh_context,
                       available_at=refresh_cutoff + pd.Timedelta(seconds=1))
later_origin = splits["validation"][1].origin_time
chosen = select_snapshot_for_origin((rolling_initial, rolling_refreshed),
                                    later_origin, fold_id=context.fold_id)
assert chosen.snapshot_id == rolling_refreshed.snapshot_id
before_refresh = select_snapshot_for_origin((rolling_initial, rolling_refreshed), refresh_cutoff)
assert before_refresh.snapshot_id == rolling_initial.snapshot_id
_show(pd.DataFrame([
    {"snapshot": s.snapshot_id[:12], "cutoff": s.cutoff,
     "available_at": s.available_at, "effective_time": s.effective_time}
    for s in (rolling_initial, rolling_refreshed)
]))
print("Selected for", later_origin, ":", chosen.snapshot_id[:12])

# %% [markdown]
# ## 15. Save construction, preprocessing, and predictor weights
#
# A snapshot contains aligned historical incidence, constructor recipes/state,
# learned membership references, and timing/provenance. It does **not** contain
# the forecast consumer's weights or input standardizer. Save all three.
# Reload learned modules from the snapshot's references, reconstruct the same
# consumer architecture, load its tensor weights, and apply the saved scaler.
# The round trip below verifies equal input transformation and equal predictions.
# Files use JSON, a tensor state dict, and NumPy arrays with `allow_pickle=False`.
# `run_manifest.json` also records axes, fold times, demo settings, and versions.

# %%
snapshot_path = save_hyperedge_snapshot(final_snapshot, OUT / "hyperedges.json")
torch.save(model.state_dict(), OUT / "forecaster.pt")
np.savez(OUT / "feature_scaler.npz", node_ids=np.asarray(scaler.node_ids),
         feature_names=np.asarray(scaler.feature_names), mean=scaler.mean,
         scale=scaler.scale, count=scaler.count, fitted_mask=scaler.fitted_mask)
pd.DataFrame(training_trace).to_csv(OUT / "manual_training.csv", index=False)

manifest = {
    "dataset": panel.provenance, "classification": "MOCK metadata; not GICS",
    "fold": {key: value.isoformat() for key, value in fold.items()},
    "seed": SEED, "lookback": LOOKBACK, "sample_stride": SAMPLE_STRIDE,
    "node_ids": panel.node_ids, "feature_names": scaler.feature_names,
    "consumer": {"in_channels": first_x.shape[1], "hidden_channels": HIDDEN_CHANNELS,
                 "out_channels": 1, "family_ids": TRAIN_FAMILIES, "context_channels": 0},
    "training": {"epochs": EPOCHS, "patience": PATIENCE, "learning_rate": LEARNING_RATE},
    "versions": {name: metadata.version(name) for name in ("numpy", "pandas", "torch")},
    "snapshot_id": final_snapshot.snapshot_id,
    "test_metrics": manual_test_metrics,
}
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

loaded_snapshot = load_hyperedge_snapshot(snapshot_path)
loaded_manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
loaded_learned = {name: LearnedMembershipConstructor.from_export_state(state)
                  for name, state in loaded_snapshot.learned_references.items()}
architecture = dict(loaded_manifest["consumer"])
restored_model = HyperedgeConsumer(**architecture, learned_modules=loaded_learned)
restored_model.load_state_dict(torch.load(OUT / "forecaster.pt", map_location="cpu", weights_only=True))
restored_model.eval()
with np.load(OUT / "feature_scaler.npz", allow_pickle=False) as arrays:
    restored_scaler = StandardizerState(tuple(arrays["node_ids"].tolist()),
        tuple(arrays["feature_names"].tolist()), arrays["mean"].copy(), arrays["scale"].copy(),
        arrays["count"].copy(), arrays["fitted_mask"].copy())

restored_sample = transform_forecast_sample(raw_splits["test"][0], restored_scaler)
np.testing.assert_allclose(restored_sample.values, splits["test"][0].values, equal_nan=True)
x, active, support, _ = _prepare_sample(restored_sample)
model.eval()
with torch.no_grad():
    original_pred = model(x, final_snapshot, active)
    restored_pred = restored_model(x, loaded_snapshot, active)
torch.testing.assert_close(original_pred, restored_pred, rtol=0, atol=0)
assert loaded_snapshot.snapshot_id == final_snapshot.snapshot_id
print("Snapshot, preprocessing, and prediction round trips passed.")
print("Artifacts:", OUT)

# %% [markdown]
# ## 16. Replace the toy data with your own panel or explicit CSV inputs
#
# If your preparation pipeline already returns an `ObservationPanel`, replace
# `panel` in section 2 and define the real fold boundaries in section 3. Match
# classification `node_id` values to `panel.node_ids`. Restart and rerun from
# the split/scaling step; never reuse a scaler or learned checkpoint fitted to
# a different universe, feature axis, or fold.
#
# For a small CSV experiment, the helper below accepts two files:
#
# | File | Required columns |
# | --- | --- |
# | `bars.csv` | `timestamp` (bar end, offset-aware), `node_id`, `close` |
# | `grid.csv` | `timestamp`, `bar_start`, `session_id`, `available_at`, `expected_minutes`, `is_partial` |
#
# **The grid must include every expected bar**, even when nobody has a close.
# Session IDs are dates from the exchange calendar. The helper applies a
# declared complete-bar policy: partial bars are ineligible. Bar availability
# is one conservative common availability time for all closes in that bar;
# if your feeds differ by stock, prepare a panel using a policy that accounts
# for those delays. This minimal helper does not reconstruct provider minute
# coverage or corporate-action adjustments; choose and record those upstream.
#
# The cell writes and rereads toy CSVs to verify the format. Replace those paths,
# stock IDs, and classification source with your real inputs. The project's
# native Alpaca adapter currently requires a compatible **1-minute** source;
# do not pass a 15-minute archive to it. An already prepared panel is preferable.

# %%
def panel_from_csv(bars_path, grid_path, node_ids):
    bars = pd.read_csv(bars_path)
    grid = pd.read_csv(grid_path)
    for frame, columns in ((bars, ("timestamp",)),
                           (grid, ("timestamp", "bar_start", "available_at"))):
        for column in columns:
            # Reject naive values instead of silently assuming a timezone.
            values = frame[column].map(pd.Timestamp)
            if values.isna().any() or any(t.tzinfo is None for t in values):
                raise ValueError(f"{column} must contain timezone-aware timestamps.")
            frame[column] = pd.to_datetime(values, utc=True)
    grid = grid.sort_values("timestamp")
    if grid.timestamp.duplicated().any() or bars.duplicated(["timestamp", "node_id"]).any():
        raise ValueError("Expected grid times and observed time/stock pairs must be unique.")
    if not set(bars.timestamp) <= set(grid.timestamp) or not set(bars.node_id) <= set(node_ids):
        raise ValueError("Observed bars must lie on the declared grid and stock universe.")
    times = pd.DatetimeIndex(grid.timestamp)
    starts, available = pd.DatetimeIndex(grid.bar_start), pd.DatetimeIndex(grid.available_at)
    if not (starts < times).all() or not (available >= times).all():
        raise ValueError("Require bar_start < timestamp <= available_at.")
    flags = grid.is_partial.astype(str).str.lower()
    if not flags.isin(["true", "false", "0", "1"]).all():
        raise ValueError("is_partial must explicitly contain True/False or 0/1.")
    partial = flags.isin(["true", "1"]).to_numpy()
    expected = grid.expected_minutes.to_numpy(dtype=int)
    if (expected <= 0).any():
        raise ValueError("expected_minutes must be positive.")
    closes = bars.pivot(index="timestamp", columns="node_id", values="close").reindex(
        index=times, columns=node_ids).to_numpy(dtype=float)
    valid = np.isfinite(closes) & (closes > 0)
    closes = np.where(valid, closes, np.nan)
    return ObservationPanel(times, tuple(node_ids), ("close",), closes[:, :, None],
        valid[:, :, None], (valid & ~partial[:, None])[:, :, None],
        pd.DatetimeIndex(pd.to_datetime(grid.session_id)), starts, available,
        expected, valid.astype(int) * expected[:, None], partial, {"close": "USD"},
        {"source": str(bars_path), "grid_source": str(grid_path),
         "coverage_policy": "preaggregated complete closes; minute coverage not reconstructed"})

bars_csv = pd.DataFrame(panel.values[:, :, 0], index=panel.timestamps,
                        columns=panel.node_ids).rename_axis("timestamp").reset_index().melt(
                            id_vars="timestamp", var_name="node_id", value_name="close")
bars_csv.to_csv(OUT / "demo_bars.csv", index=False)
pd.DataFrame({"timestamp": panel.timestamps, "bar_start": panel.bar_starts,
              "session_id": panel.session_ids, "available_at": panel.availability_times,
              "expected_minutes": panel.expected_minutes,
              "is_partial": panel.is_partial}).to_csv(OUT / "demo_grid.csv", index=False)
mock_gics.to_csv(OUT / "mock_classifications.csv", index=False)
csv_panel = panel_from_csv(OUT / "demo_bars.csv", OUT / "demo_grid.csv", panel.node_ids)
np.testing.assert_allclose(csv_panel.values, panel.values, equal_nan=True)
assert csv_panel.timestamps.equals(panel.timestamps)
print("CSV panel round trip passed.")

# Real experiment replacement (edit paths; do not run with the mock source):
# panel = panel_from_csv(ROOT / "local-data/my_bars.csv",
#                        ROOT / "local-data/my_expected_grid.csv", your_ordered_stock_ids)
# real_gics_records = ROOT / "local-data/my_point_in_time_gics.csv"
# context = make_construction_context(panel, your_fit_cutoff, seed=SEED,
#             metadata={"gics": real_gics_records}, fold_id="real:0")
# Rebuild samples, splits, scaler, pipeline, and model as demonstrated above.

# %% [markdown]
# ## 17. A practical experiment sequence
#
# 1. Run this notebook unchanged and inspect the incidence and loss plots.
# 2. Change one KNN setting: neighbors, similarity, or constructor window.
# 3. Set `TRAIN_FAMILIES = ("knn_absolute",)`; then try GICS+KNN+events or Mapper.
# 4. Add/remove `"learned"`, inspect logit changes and hard slot sizes, and adjust
#    pilot learning controls using validation rather than test results.
# 5. Inspect the event/information/Mapper additions, enabled by
#    `RUN_EXTRA_CONSTRUCTOR_TRAINING`. Set it to false for a shorter run. Add
#    signed/covariance/custom configs to the ablation dictionary using `_config_for(...)`.
# 6. Keep PH in `context_providers` and pass real per-origin features and masks.
# 7. Replace synthetic inputs with real observations and supplied classifications.
# 8. Repeat the same fold/target/support/encoder/tuning protocol across several
#    seeds and folds; aggregate errors and uncertainty with construction coverage,
#    memberships, cost, and parameter counts.
#
# If KNN rejects a recipe, inspect coverage, complete-row count, eligible stock
# count, and constant histories. Do not silently reduce the requested neighbors.
# If another method returns zero edges, inspect its selection diagnostics and
# explicitly adjust the recipe. An empty family is often a valid scientific result.
#
# Rerun from a fresh kernel after changing global settings or the data. Re-running
# only downstream cells can otherwise mix artifacts from different experiments.
# Saved outputs in `local-data/tutorial-artifacts/` are overwritten by each run;
# choose a distinct output directory when keeping multiple experiment runs.

# %%
print("Walkthrough completed: construction, training, ablations, PH, rolling scheduling,")
print("checkpoint/prediction reload, and CSV input validation.")
print("Notebook outputs are synthetic demonstrations; research evaluation is a separate experiment.")
