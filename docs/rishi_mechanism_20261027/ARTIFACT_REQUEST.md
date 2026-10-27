# Message to Rishi

Hey Rishi, I'm investigating why the THINK+mixer + dynamic blend works and which components contribute. The repo has the summaries/charts and selected return paths, but I couldn't find the saved per-stock forecasts, checkpoints or full account records.

Could you share a download link to these existing folders?

- `runs/ndx_reference_networks_20261024/`
- `runs/ndx_extended_hours_20261024/`
- `runs/ndx_drawdown_20261026/batch1/`

Ideally include per-stock predictions for each year/seed (including training/validation predictions if saved), checkpoints, preprocessing/scalers and graphs, and holdings, frozen orders/fills/cancellations and daily account ledgers. Stable security IDs, eligibility/label masks and original timestamps/hashes are important.

The reviewed `data/nasdaq_monthly_pit/assembled/` inputs and `extended_hours_features_20261024/` feature files would let us reproduce the forecasts too. Please retain their manifests and original ID/date order.

There is also a machine-readable list, `ARTIFACT_CONTRACT_V2.json`, derived from your committed hash locks: 339 existing fit/input files with expected hashes. It includes the trading calendar and the shared preparation routine's factor-file dependencies. This is a request list, not a claim that it includes every account/trade tape; those are still needed separately.

No new training needed—just the existing files. Leave out `.env`, API keys and credentials. If the export is too large, predictions + stock IDs/masks + daily holdings/trade records would be a useful first package. If some files weren't saved, just say which ones are available; training predictions can be regenerated from original checkpoints without refitting.
