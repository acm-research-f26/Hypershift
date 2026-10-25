# Data acquisition and preparation

These sources acquire market histories, resolve historical membership and issuer
identity, review corporate actions, and build model-ready features. Raw data and
generated snapshots belong under the ignored root `data/`, not beside these sources.

| Sources | Scope |
| --- | --- |
| [fetch_alpaca_15min.py](fetch_alpaca_15min.py), [fetch_alpaca_batch.py](fetch_alpaca_batch.py) | Historical SIP 15-minute bars, including extended hours |
| [historical_nasdaq_universe.py](historical_nasdaq_universe.py), [historical_ndx_membership.py](historical_ndx_membership.py), [assemble_historical_ndx.py](assemble_historical_ndx.py) | Monthly membership and historical security identities |
| [fetch_tiingo_pit.py](fetch_tiingo_pit.py), [fetch_historical_ndx_yahoo.py](fetch_historical_ndx_yahoo.py), [review_historical_ndx_inputs.py](review_historical_ndx_inputs.py) | Reviewed daily economic histories and coverage |
| [prepare_ndx_alpaca.py](prepare_ndx_alpaca.py), [prepare_ndx_extended_hours.py](prepare_ndx_extended_hours.py) | Matched regular-session and AH/PM feature snapshots |
| [fetch_ndx_reference_assets.py](fetch_ndx_reference_assets.py) | Pinned reference implementations and declared factor inputs |

From the repository root, inspect the acquisition plan without making requests:

```powershell
python -m hypershift run fetch_alpaca_15min --dry-run
```

Credentials remain in the ignored root `.env`. See the
[Alpaca setup and dataset locations](../../docs/ALPACA_15MIN.md),
[historical data contract](../../docs/nasdaq_monthly_pit/DATA_REQUIREMENTS.md), and
[protocol index](../../config/README.md). Missing data are not evidence that a
stock was historically ineligible; preprocessing and membership review remain distinct.

The other cohort downloaders and capability probes document earlier experiments.
They should not replace the active historical-monthly universe.
