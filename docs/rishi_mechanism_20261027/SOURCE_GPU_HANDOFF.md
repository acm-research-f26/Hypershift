# Source-device checkpoint trace: request and instructions

The local RTX 3050 cannot reproduce all saved checkpoint values to the frozen 1e-6 percentage-point tolerance. Rishi's recorded source device is an RTX 4070 laptop GPU. A trace on that original machine can identify whether the source device reproduces the saved values and where its intermediate tensors diverge from the local device. This needs inference only: no training, optimizer, new strategy, protected-window scoring, or tolerance amendment.

## What to give Rishi

Share `research/analysis/rishi_source_device_trace.py` from this checkout, plus these instructions. The current script includes the additional code-hash checks committed locally as b787818 but has **not** been pushed; that commit is not necessarily available in his checkout. Copy the file into the same relative location in a checkout with the organized `hypershift` launcher and the published original artifacts. The script expects the original reviewed daily/AH/PM inputs and 2022 parent seed-7 checkpoint/export at their existing paths. It refuses to overwrite the output directory.

From that repository root, with the source CUDA environment and GPU available, run:

```powershell
.\.venv\Scripts\python.exe -m hypershift run rishi_source_device_trace --out results/RISHI_SOURCE_DEVICE_TRACE_ORIGINAL
```

Return only the new output folder's `TRACE.json`, `FORECASTS.npz`, and `ACTIVATIONS.npz`; do not include `.env`, API keys, other credentials or unrelated files. `TRACE.json` contains local source paths and device metadata; review those before sharing. The layer activations are model-derived research data and should be shared only with intended collaborators. Record any error without replacing the folder. The script includes no training or financial account scoring.

## What was checked locally

The local smoke run captured 14 layer outputs for the canonical test batch containing origin 186/ADBE. Its full validation/test arrays and axes exactly match the earlier local GPU replay; its isolated batch exactly matches the corresponding full-call values. Local source-reference errors remain 0.0000143051 pp validation and 0.0015182495 pp test. Two trace tests and one comparator test pass; comparing the local trace to itself yields zero for both splits and the batch. The strengthened trace adds hashes of ten model/preparation modules to distinguish code differences from hardware differences. The usable local trace is `results/RISHI_SOURCE_DEVICE_TRACE_LOCAL_20261028_V2/`; the earlier smoke trace remains preserved. Both are ignored by Git.

After receiving the three original-device files in a fresh local directory, compare them using:

```powershell
.\.venv\Scripts\python.exe research/analysis/rishi_compare_device_traces.py --local results/RISHI_SOURCE_DEVICE_TRACE_LOCAL_20261028_V2 --candidate <received-trace-directory> --out results/RISHI_SOURCE_DEVICE_COMPARISON.json
```

Replace the bracketed directory with the actual path. The comparator requires identical prepared-input, axis, checkpoint/export, trace-code and ten model/preparation module hashes, matching layer names/shapes and batch IDs before reporting per-layer and forecast differences. Hashes do not by themselves prove every dependency is identical. Treat source output as untrusted until those checks pass. Source-device agreement for this one checkpoint cannot establish the full 80-checkpoint reproduction gate or authorize gated interventions by itself. No message has been sent.
