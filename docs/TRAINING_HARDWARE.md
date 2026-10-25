# Training hardware

<!-- benchmark-audit-20261010:start -->
NYSX is the passive Global X NYSE 100 ETF. It launched March 25, 2026 and has no tradable 2022–2025 history. Its index covers technology and growth companies across exchanges, not the 100 largest NYSE-listed companies. NYSX performance is unavailable here. NYSE historical charts use the simulated equal-weight cohort basket (scheduled rebalances and modeled costs), not an ETF. NASDAQ charts retain QQQ; SPY supplies broad-market context. [Issuer and inception](https://www.globalxetfs.com/funds/nysx/) and [prospectus](https://assets.globalxetfs.com/funds/documents/nysx/prospectus-regulatory/Summary-Prospectus_NYSX.pdf).

The refreshed CHARTS.md pages are the current visual record. Earlier ETF-overlay descriptions are superseded by this correction. Audited return paths and training selections remain preserved.
<!-- benchmark-audit-20261010:end -->

User preference recorded October 8, 2026: use the laptop's NVIDIA CUDA GPU for future training where feasible.

The laptop reports **NVIDIA GeForce RTX 4070 Laptop GPU**, **8,188 MiB VRAM**, driver **591.66**.
The project `.venv` now has **PyTorch 2.14.0+cu130**, CUDA runtime **13.0**,
and a verified working CUDA device. The official wheel was installed into this environment;
the existing NVIDIA driver was retained.

The completed original studies and the Huber-plus-ranking follow-up use deterministic CPU training,
two PyTorch threads per worker and three seed workers. The ranking follow-up keeps that backend to
match the existing fits, of which 72 were exactly reproduced and reused.

Both `run_robust_research.py` and `run_architecture_research.py` now support `--device auto|cpu|cuda`.
The default `auto` selects CUDA when available. GPU runs require `--workers 1`;
parallel CPU runs can use `--device cpu --workers 3`. Models, inputs, masks and training targets
move to the selected device; predictions return to CPU NumPy and saved checkpoints contain CPU tensors.
GPU/build metadata is stored in the protocol and checkpoints. Fit reuse requires matching backend/build.

Training retains float32, no automatic mixed precision, TF32 disabled, strict deterministic algorithms,
cuDNN benchmarking disabled and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Hardware changes do not change
chronological splits, validation selection or portfolio accounting. CPU/GPU floating-point results need
not be identical; compare experiments on a common backend.

Verification passed **all 19 architectures**, including Huber+ranking: CUDA forward passes, finite
backward gradients, optimizer steps and CPU/GPU inference comparison. Both actual training runners
completed a synthetic training epoch, validation, prediction and checkpoint check. All 41 installed
packages passed dependency validation. Peak tensor memory in the synthetic verification was about 100 MiB.
These checks do not establish a training speedup; benchmark the actual workload separately.

[GPU verification receipt](audit/cuda_verification.json)

## Future use

```powershell
.\.venv\Scripts\python.exe -m hypershift run run_robust_research --device cuda --workers 1 --out runs/gpu_new
```

Installation used:

```powershell
uv pip install --python .venv\Scripts\python.exe --cache-dir .cache\uv-cuda --link-mode copy --no-deps 'torch==2.14.0+cu130' --index-url https://download.pytorch.org/whl/cu130
```

`requirements/requirements-tested.txt` records the current CUDA packages.
`requirements/requirements-tested-cpu.txt` preserves the historical CPU environment;
`requirements/requirements-cuda.txt` pins the CUDA wheel.
Use the matching official wheel index when installing these local-version pins.
See [PyTorch installation](https://pytorch.org/get-started/locally/) and
[reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

No historical models were retrained, no new backtest experiments were launched and no training job
is left running. Existing study results, graphs and slide decks retain their original CPU provenance.

[Current ranking-loss comparison](HUBER_RANKING.md)
