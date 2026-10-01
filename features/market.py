import numpy as np

from data.types import ObservationPanel

# $r_{t} = \text{log}(C_t) - \text{log}(C_{t-1})$
# Returns this $r_t \in R^{(T x N)}$ 
# where T is the number of timestamps and N is the number of tickers
# Input value: $\text{ObservationPanel} \in R^{(T x N x F)}$ 
def log_returns(panel: ObservationPanel, overnight_policy="exclude", gap_policy="mask"):
    """Return close-to-close log returns and their Boolean mask, both [T,N]."""
    if overnight_policy != "exclude" or gap_policy != "mask":
        raise ValueError("Supported policies are overnight='exclude', gap='mask'")

    shape = (len(panel.timestamps), len(panel.node_ids), len(panel.field_names))
    arrays = (panel.values, panel.observation_mask, panel.eligible_mask)
    if any(array.shape != shape for array in arrays):
        raise ValueError("Panel values and masks must match the declared axes")
    if any(mask.dtype.kind != "b" for mask in arrays[1:]):
        raise ValueError("Panel masks must be Boolean")
    if len(panel.session_ids) != shape[0] or len(panel.bar_starts) != shape[0]:
        raise ValueError("Session IDs and bar starts must match the time axis")
    times = panel.timestamps
    if (times.tz is None or times.hasnans or not times.is_unique
            or not times.is_monotonic_increasing):
        raise ValueError("Timestamps must be aware, present, unique, and sorted")

    field = panel.field_names.index("close")
    close = panel.values[:, :, field]
    usable = (
        panel.observation_mask[:, :, field]
        & panel.eligible_mask[:, :, field]
        & np.isfinite(close)
        & (close > 0)
    )

    # Compute logs only at valid endpoints; missing/nonpositive prices stay NaN.
    log_close = np.full(close.shape, np.nan)
    np.log(close, out=log_close, where=usable)

    returns = np.full(close.shape, np.nan)
    valid = np.zeros(close.shape, dtype=bool)

    # A one-step return needs usable endpoints in adjacent bins of one session.
    same_session = np.asarray(panel.session_ids[1:] == panel.session_ids[:-1])
    contiguous = np.asarray(panel.bar_starts[1:] == times[:-1])
    valid[1:] = (
        usable[1:] & usable[:-1]
        & (same_session & contiguous)[:, None]
    )
    returns[1:] = np.where(
        valid[1:], log_close[1:] - log_close[:-1], np.nan
    )
    return returns, valid

