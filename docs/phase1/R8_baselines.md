# R8: RSR-I and STHGCN baselines (implemented)

Code: `src/hypershift/models/baselines.py`; selected by `RunConfig.model` = `think` (default, nothing else changes) | `rsr_i` | `sthgcn`.
They run through `train_one_run` unchanged: same data, windows, loss (masked MSE + alpha x ranking hinge, alpha=1), metrics,
run-folder contract and validation-Sharpe epoch selection. `temporal`/`spatial`/`attn_*`/`structure` are ignored by the baselines.
Tests: `tests/test_baselines_r8.py` (8 tests: shapes, finite grads for every parameter, graph actually used via input-gradient
locality and graph-vs-empty output change, HGNN operator properties, `train_one_run` end to end for both, real-data relation
subsetting). Full suite on CPU: 229 passed.

## RSR-I (Feng et al. 2019), as implemented
Reference read: `training/relation_rank_lstm.py` (fetched, quoted lines verified).
- LSTM(5 -> hidden) over the window, last hidden state f [N, U]. U = `hidden` = 32 (README NYSE setting `-u 32`).
- Relations: RSR's raw multi-hot tensors, industry (108 channels) + wiki (33 channels) concatenated, K = 141 (NYSE), each with its
  own trailing self-relation channel. 291,704 related pairs on full NYSE. Restricted to the current universe by ticker.
- `rel_weight = leaky_relu(Linear(K -> 1)(multi-hot), 0.2)`, computed sparsely (index_add over the entries; no [N,N,K] tensor).
- `weight = (f f^T) * rel_weight`, plus -1e9 mask on unrelated pairs, `softmax(..., dim=0)` (over axis 0 of the [N,N] matrix, exactly
  as the TF code does, which normalizes over the receiving node, not the neighbours), `prop = W f`, `pred = leaky_relu(Linear([f, prop]))`.
- Parameters: 5,199 (K + LSTM + head). THINK HH has 1,890.

Deviations:
1. End-to-end training. RSR pre-trains the LSTM (`rank_lstm.py`) and loads it; we train LSTM and relational layer jointly.
2. Window length 16 (our pipeline, shared with THINK) instead of RSR's NYSE `-l 8` / NASDAQ `-l 16`. Optimizer Adam lr 1e-3, weight decay 5e-4,
   grad clip 1 (our pipeline) vs RSR's Adam lr 1e-3 without weight decay. RSR alpha=1 = ours.
3. Inputs: our normalisation (`norm: paper` for the full runs), missing days handled by our mask.
4. Dense [B,N,N] attention; full NYSE uses `micro_batch_days=2` (exactly the same gradients, LSTM/softmax have no batch coupling).
5. Not run: RSR-E (explicit), single relation family at a time (RSR uses one of industry or wiki per run; the task asked for both).

## STHGCN (Sawhney et al. 2020/21), as implemented
Reference: the midas-research/sthgcn-icdm repo layout (`TimeBlock`, `STGCNBlock`, `HGNN_conv`, standard HGNN operator) as summarized in
`R_feasibility.md`; the repo does not run and the paper text was not readable, so **everything below not marked from the repo is inferred**.
- `TimeBlock(k=3)`: `relu(conv1(x) + sigmoid(conv2(x)) + conv3(x))`, valid convolution along time [from repo].
- Block = TimeBlock(in -> 64) -> hypergraph conv `relu(G X Theta)` (64 -> 16, at every time step) -> TimeBlock(16 -> 64) -> BatchNorm over
  the node axis [STGCN layout, inferred]. Two blocks, then a TimeBlock(64 -> 64), flatten time x channels, Linear -> scalar.
  Time length 16 -> 6 (five k=3 valid convs).
- Hypergraph operator (HGNN): `G = Dv^-1/2 H W De^-1 H^T Dv^-1/2`, W = identity, H from our `Hypergraph` (industry + wiki star hyperedges,
  same as THINK, so both see the same relation information). Isolated stocks get `G_ii = 1` (HGNN alone would zero them) [inferred].
- Parameters: 105,381 on full NYSE (BN affine per node included), ~55x THINK.

Deviations / inferred: channel widths 64/16 (STGCN defaults; repo values not confirmed); the repo's per-stock GRUs are omitted (the gated
temporal convs play that role; repo hard-codes 423 GRUs); no attention, Euclidean only; MSE + ranking loss instead of the repo's
tertile classification head (ranking task); BatchNorm couples days within a micro-batch (`micro_batch_days=4` on full NYSE, so BN
statistics come from 4 days, not 8); lookback 16 instead of the repo's 50.

## Run plan (`scripts/queues/phase1_gpu_r8.sh`)
Starts after "phase1 queue 2 done", one worker, concurrent with queue 3. Before every attempt waits for >= 1500 MiB free VRAM
(nvidia-smi, 60 s poll); any failure (incl. CUDA OOM) removes `failed.json` and retries up to 6 times after 60 s. Ends with `phase1 queue r8 done`.
1. Small scale (309 stocks): `poc_sectors.py run --variant R8_rsr_i|R8_sthgcn --seeds s --set model=...`, 10 seeds each, 30 epochs, patience 10,
   level inputs, lr 1e-3, alpha 1 (same settings as `POC_sectors_eq14`); folders `results/POC_sectors_R8_rsr_i/rsr_i/`, `results/POC_sectors_R8_sthgcn/sthgcn/`.
   Compare against `POC_sectors_eq14` HH_hyper. The single-arm baseline mode in poc_sectors is triggered by `model=` in `--set`.
2. Full NYSE: `hypershift.run --config configs/think_nyse.yaml --set exp=R8_baselines norm=paper epochs=100 patience=1000 batch_days=8
   label={RSR_I|STHGCN} model=... micro_batch_days={2|4}`, seeds 0-9 interleaved; folders `results/R8_baselines/{RSR_I,STHGCN}/seed_k`.
   Compare validation-selected and `test_oracle_sr` with `R5_eq14` THINK (paper protocol).

## Cost estimate (ESTIMATE, not measured on GPU)
CPU-only forward+backward probe on full NYSE (2 days): THINK HH 0.065 s, RSR-I 0.139 s (2.1x), STHGCN 0.232 s (3.6x). THINK is launch-bound on
GPU (measured 21.6 s/epoch), the baselines are dense-matmul/conv bound, which a GPU handles better than a CPU. Estimated GPU 10-25 s/epoch for
each, i.e. 17-40 min per 100-epoch seed; 10 seeds x 2 models = 6-13 GPU-h if the GPU is not contended (more when sharing with queue 3).
Small scale: about 1-3 s/epoch, ~0.5 h total. Peak VRAM (analytic, not measured): RSR-I with micro=2 ~ 0.3-0.5 GB; STHGCN with micro=4 ~ 1 GB
(activations [4,1737,14,64] x ~30 saved tensors); the 1500 MiB gate leaves headroom.
