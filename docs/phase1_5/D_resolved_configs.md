# D (part 2): resolved configs of the Phase 1 headline arms, loss reduction, batch construction, repo defaults

*Phase 1.5 fidelity audit, 2026-10-02. Everything below is read from each run's `config.json` / `metrics.json` (not from launch commands) and from the code at HEAD (`src/hypershift/train/loop.py`, `train/loss.py`, `config.py`). For every run directory I checked that all seeds share one config (they differ only in `seed`/`exp`/`label`; 0 differences across all seeds of every arm listed). Repo defaults were fetched on 2026-10-02 from GitHub raw: `fulifeng/Temporal_Relational_Stock_Ranking` `training/relation_rank_lstm.py` (RSR) and `midas-research/sthan-sr-aaai` `README.md`, `training/train_nyse.py`, `training/hgat_nyse.py` (STHAN-SR). The THINK paper (`docs/paper/icdm22-think.pdf`, pp. 849-854) states none of these hyperparameters (paper audit P44, R_feasibility), so every value in our column is INFERRED (not in paper); the comparison below is with the two code repos only.*

## 1. Fully resolved RunConfig, headline arms

All columns are `RunConfig` fields. `device: cuda` is the field default (Kaggle T4 / laptop RTX 3050 ran the arms). `nodes / edges` and `epochs_run` come from `metrics.json`.

### 1a. Full NYSE (R5_g2: THINK/EH/EE/HE; R8_baselines_g2: RSR-I, STHGCN)

| run dir | label | n seeds | model | temporal | spatial | structure | norm | input_mode | target | attn_score | attn_dist | hidden | seq | kernel | lr | weight_decay | alpha | epochs | patience | batch_days | micro_batch_days | grad_clip | topk | sources | market | device | epochs_run (median) | nodes / edges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `R5_g2/THINK_paperProtocol` | HH (THINK) | 25 | think | hyp | hyp | hyper | paper | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 100 | 1000 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 100 | 1737 / 4350 |
| `R5_g2/EH` | EH | 25 | think | euc | hyp | hyper | paper | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 100 | 1000 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 100 | 1737 / 4350 |
| `R5_g2/EE` | EE | 10 | think | euc | euc | hyper | paper | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 100 | 1000 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 100 | 1737 / 4350 |
| `R5_g2/HE` | HE | 10 | think | hyp | euc | hyper | paper | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 100 | 1000 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 100 | 1737 / 4350 |
| `R8_baselines_g2/RSR_I` | RSR-I | 10 | rsr_i | hyp | hyp | hyper | paper | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 100 | 1000 | 8 | 2 | 1.0 | 5 | industry,wiki | NYSE | cuda | 100 | 1737 / 4350 |
| `R8_baselines_g2/STHGCN` | STHGCN | 10 | sthgcn | hyp | hyp | hyper | paper | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 100 | 1000 | 8 | 4 | 1.0 | 5 | industry,wiki | NYSE | cuda | 100 | 1737 / 4350 |

Reading the table:
- The four R5_g2 arms differ **only** in `temporal`/`spatial`; RSR-I and STHGCN differ from THINK only in `model` and `micro_batch_days` (memory only, gradient-identical: loop.py `(loss * len(chunk)/len(batch)).backward()`). All use `norm: paper` (full-series-max scaling, look-ahead kept to match the paper), `input_mode: level`, `epochs 100`, `patience 1000` (early stopping off: median epochs_run = 100), `batch_days 8`, `lr 1e-3`, `weight_decay 5e-4`, `alpha 1.0`, `grad_clip 1.0`, `hidden 32`, `seq 16`, `kernel 4`, top-5.
- HH (THINK) is stored under the label `THINK_paperProtocol` in `results/R5_g2`.
- `hidden`/`kernel` are ignored by STHGCN (fixed widths 64/16, `k = 3`) and `kernel` by RSR-I; `attn_*`, `temporal`, `spatial`, `structure` are ignored by the baselines (`config.py` comments, `build_model`).

### 1b. Small scale (309 stocks, 558-edge g2 graph)

| run dir | label | n seeds | model | temporal | spatial | structure | norm | input_mode | target | attn_score | attn_dist | hidden | seq | kernel | lr | weight_decay | alpha | epochs | patience | batch_days | micro_batch_days | grad_clip | topk | sources | market | device | epochs_run (median) | nodes / edges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `POC_sectors_g2/HH_hyper` | g2/HH_hyper | 10 | think | hyp | hyp | hyper | train | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 16 | 309 / 558 |
| `POC_sectors_g2/EH_hyper` | g2/EH_hyper | 10 | think | euc | hyp | hyper | train | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 17 | 309 / 558 |
| `POC_sectors_g2/EE_hyper` | g2/EE_hyper | 10 | think | euc | euc | hyper | train | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 15 | 309 / 558 |
| `POC_sectors_g2/HH_clique` | g2/HH_clique | 10 | think | hyp | hyp | clique | train | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 19 | 309 / 4897 |
| `POC_sectors_g2/EH_clique` | g2/EH_clique | 10 | think | euc | hyp | clique | train | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 15 | 309 / 4897 |
| `POC_sectors_g2/EE_clique` | g2/EE_clique | 10 | think | euc | euc | clique | train | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 13 | 309 / 4897 |
| `POC_sectors_g2/HH_none` | g2/HH_none | 10 | think | hyp | hyp | none | train | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 18 | 309 / 558 |
| `POC_sectors_g2/EE_none` | g2/EE_none | 10 | think | euc | euc | none | train | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 19 | 309 / 558 |
| `POC_sectors_rel_g2/HH_hyper` | rel_g2/HH_hyper | 10 | think | hyp | hyp | hyper | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 12 | 309 / 558 |
| `POC_sectors_rel_g2/EH_hyper` | rel_g2/EH_hyper | 10 | think | euc | hyp | hyper | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 14 | 309 / 558 |
| `POC_sectors_rel_g2/EE_hyper` | rel_g2/EE_hyper | 10 | think | euc | euc | hyper | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 19 | 309 / 558 |
| `POC_sectors_rel_g2/HH_clique` | rel_g2/HH_clique | 10 | think | hyp | hyp | clique | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 12 | 309 / 4897 |
| `POC_sectors_rel_g2/EH_clique` | rel_g2/EH_clique | 10 | think | euc | hyp | clique | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 13 | 309 / 4897 |
| `POC_sectors_rel_g2/EE_clique` | rel_g2/EE_clique | 10 | think | euc | euc | clique | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 15 | 309 / 4897 |
| `POC_sectors_rel_g2/HH_none` | rel_g2/HH_none | 10 | think | hyp | hyp | none | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 16 | 309 / 558 |
| `POC_sectors_rel_g2/EE_none` | rel_g2/EE_none | 10 | think | euc | euc | none | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 20 | 309 / 558 |
| `POC_sectors_rel_tuned_g2/HH_hyper` | rel_tuned_g2/HH_hyper | 10 | think | hyp | hyp | hyper | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.0005 | 0.0005 | 100.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 14 | 309 / 558 |
| `POC_sectors_rel_tuned_g2/EH_hyper` | rel_tuned_g2/EH_hyper | 10 | think | euc | hyp | hyper | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 30.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 22 | 309 / 558 |
| `POC_sectors_rel_tuned_g2/EE_hyper` | rel_tuned_g2/EE_hyper | 10 | think | euc | euc | hyper | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.0005 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 22 | 309 / 558 |
| `POC_sectors_rel_tuned_g2/HH_none` | rel_tuned_g2/HH_none | 10 | think | hyp | hyp | none | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.0005 | 0.0005 | 100.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 18 | 309 / 558 |
| `POC_sectors_rel_tuned_g2/EE_none` | rel_tuned_g2/EE_none | 10 | think | euc | euc | none | train | relative | return | eq14 | mult | 32 | 16 | 4 | 0.0005 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 21 | 309 / 558 |
| `POC_sectors_R8_rsr_i_g2/rsr_i` | R8_rsr_i_g2/rsr_i | 10 | rsr_i | hyp | hyp | hyper | train | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 17 | 309 / 558 |
| `POC_sectors_R8_sthgcn_g2/sthgcn` | R8_sthgcn_g2/sthgcn | 10 | sthgcn | hyp | hyp | hyper | train | level | return | eq14 | mult | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 13 | 309 / 558 |
| `POC_sectors_A10_g2/HH_hyper` | A10_g2/HH_hyper (attn_dist off) | 10 | think | hyp | hyp | hyper | train | level | return | eq14 | off | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 16 | 309 / 558 |
| `POC_sectors_rel_A10_g2/HH_hyper` | rel_A10_g2/HH_hyper (attn_dist off) | 10 | think | hyp | hyp | hyper | train | relative | return | eq14 | off | 32 | 16 | 4 | 0.001 | 0.0005 | 1.0 | 30 | 10 | 8 | 0 | 1.0 | 5 | industry,wiki | NYSE | cuda | 11 | 309 / 558 |

Reading the table: small-scale runs use `norm: train` (train-period max, leak-free), `epochs 30`, `patience 10` (median `epochs_run` shows where early stopping hit), `batch_days 8`, otherwise as above. `rel_tuned_g2` is the per-arm tuned variant (`lr`/`alpha` picked on validation, e.g. HH lr 5e-4 alpha 100, EH alpha 30, EE lr 5e-4). Clique arms use `micro_batch_days 1`. The small-scale RSR-I/STHGCN arms (`R8_*_g2`) are THINK's config with `model` swapped.

## 2. What is not a RunConfig field (from the code)

| item | value | where |
|---|---|---|
| optimizer | `torch.optim.Adam(params, lr, weight_decay)`, betas (0.9, 0.999), eps 1e-8; weight decay is coupled L2 (added to the gradient), not decoupled | loop.py `train_one_run` |
| LR schedule / dropout | none / none (STHGCN has BatchNorm over the node axis) | loop.py, `models/` |
| grad clipping | `clip_grad_norm_(all params, 1.0)` on the total norm after gradient accumulation, before `opt.step()` | loop.py |
| init, THINK hyperbolic FC | `z ~ N(0,1) * (2*in*out)^-1/2`, `r = 0` | `layers.py PoincareLinear` |
| init, THINK Euclidean temporal conv | `nn.Linear` PyTorch default | `layers.py EucTemporalConv` |
| init, attention | `a ~ N(0,1) * dim^-1/2`, `gamma = 0`; Euclidean attention FC = `nn.Linear` default | `attention.py` |
| init, RSR-I / STHGCN | `nn.LSTM`/`nn.Linear` defaults; STHGCN `theta ~ U(-1/sqrt(16), 1/sqrt(16))` | `baselines.py` |
| seeding | `random`, `numpy`, `torch` (+cuda) set to `cfg.seed`; shuffle RNG `default_rng(cfg.seed)` | loop.py |
| epoch selection | best **validation** Sharpe; test arrays saved at that epoch; `test_oracle_sr` = max over epochs of test Sharpe (diagnostic) | loop.py |
| steps per epoch | train targets t in [16, 755] = 740 windows; `batch_days 8` gives 93 optimizer steps per epoch (last step has 4 days) | `window_offsets` |

## 3. Loss reduction and scale

`rank_mse_loss(pred, gt, mask, alpha)` (loss.py), inputs `[B, N]` (B days):
- `reg` = mean over all B*N entries of `mask * (pred - gt)^2`: a mean divided by B*N (not by the number of observed stocks).
- `rank` = mean over all B*N*N entries of `relu(-(pred_i - pred_j)*(gt_i - gt_j) * mask_i*mask_j)`: a **mean over ordered pairs** (both (i,j) and (j,i), and the zero diagonal), divided by B*N*N. It is **averaged, not summed**.
- `loss = reg + alpha * rank`; the batch loss is the mean over days (equal weight per day). Accumulation chunks are weighted by `len(chunk)/len(batch)`, equal to the unsplit step.

Scale on real data (309-stock universe, 739 train days, `gt` sd 1.76%, zero-prediction MSE 3.08e-4; random predictions of the stated sd, CPU):

| pred sd | reg (MSE) | rank | rank / reg | norm of d reg / d pred | norm of d rank / d pred |
|---|---|---|---|---|---|
| 0 | 3.08e-4 | 0 | 0 | 7.3e-5 | 0 (relu'(0) = 0) |
| 1e-3 | 3.09e-4 | 8.4e-6 | 0.027 | 7.4e-5 | 3.9e-5 |
| 3e-3 | 3.17e-4 | 2.5e-5 | 0.079 | 7.5e-5 | 3.8e-5 |
| 1e-2 | 4.08e-4 | 8.4e-5 | 0.21 | 8.5e-5 | 3.9e-5 |
| 3e-2 | 1.2e-3 | 2.5e-4 | 0.21 | 1.5e-4 | 3.9e-5 |

With `alpha = 1` the ranking term is 3-20% of the MSE in value but carries about half of its gradient norm; its gradient norm is roughly constant in the prediction scale (a hinge on a product of differences), while the MSE gradient is dominated by the noise term `-gt` when predictions are small. The tuned `alpha = 30-100` of the small-scale `rel_tuned` arms is the validation search pushing the ranking term up. This is a measurement, not a verdict on whether `alpha = 1` is "right".

**Same as the repos?** Yes for the formula and reductions. STHAN-SR (PyTorch): `reg = torch.mean(weight * (input - target)**2)`, `rank = torch.mean(F.relu((pre_pw_dif*gt_pw_dif)*mask_pw))` with `pre = r_i - r_j`, `gt = gt_j - gt_i`, which equals our `relu(-dp*dg)`. RSR (TF): the same rank term; `reg` uses `tf.losses.mean_squared_error(weights=mask)`, which divides by the number of nonzero weights instead of N (a difference of `mean(mask) = 0.9994` here). The differences are in what feeds the loss: section 5.

## 4. Batch construction: are ranking pairs only within a day?

Yes, verified in code and by test.
- `gather_batch` returns tensors `[B, N, ...]` with B = days; `rank_mse_loss` forms `dp = pred[:, :, None] - pred[:, None, :]`, a `[B, N, N]` tensor, so pairs are (stock i, stock j) on the **same day b**. Different days interact only through the final mean over B.
- Test `tests/test_loss_batching.py` (4 tests, added here): (1) batch loss/reg/rank equal the mean of per-day values, (2) the gradient of day 0's predictions is unchanged when the other days' predictions and returns are replaced by random values (no cross-day pairs), (3) `rank` equals an explicit triple loop over `(b, i, j)` divided by `B*N*N` and `reg` divides by `B*N`, (4) a masked stock gets exactly zero gradient.
- `batch_days 8` means one optimizer step averages 8 per-day losses; the repos use one day per step (section 5).

## 5. Differences from the STHAN-SR and RSR repo defaults

| item | STHAN-SR repo (`train_nyse.py`, `hgat_nyse.py`, README) | RSR repo (`relation_rank_lstm.py`; README fetched 2026-10-02) | this repo (R5_g2 / small scale) |
|---|---|---|---|
| README run line | `python train_nyse.py -m NYSE -l 16 -u 64 -a 1 -e ...` | NYSE `python relation_rank_lstm.py -m NYSE -l 8 -u 32 -a 10 -e NYSE_rank_lstm_seq-8_unit-32_0.csv.npy`; NASDAQ `-rn wikidata -l 16 -u 64 -a 0.1`; argparse defaults `-l 4 -u 64 -r 0.001 -a 1` | `seq 16`, `hidden 32` |
| window | 16 (README) | 8 (NYSE) | 16 |
| hidden | GRU hard-coded 32 in `hgat_nyse.py` (README `-u 64` is not used by the model code I read) | 32 (NYSE) | 32 |
| lr | 1e-3 | 1e-3 | 1e-3 |
| weight decay | `Adam(..., weight_decay=5e-4)` | none (TF `AdamOptimizer`) | 5e-4 for every model incl. RSR-I (a deviation for RSR-I) |
| alpha | 1 (NYSE), 0.1 (NASDAQ, README) | **10 (NYSE README line `-a 10`)**, 0.1 (NASDAQ), argparse default 1 | 1 (NYSE), 0.1 (NASDAQ); 10-100 only in the tuned small-scale arms |
| **target** | **model outputs a price; `return_ratio = (pred - base_price)/base_price`; loss on the ratio** | same | **`target: return`: the output is the return directly.** `target: price` exists in the code, no Phase 1 headline arm uses it |
| **batch** | **1 day per optimizer step**, 740 steps per epoch | same | **`batch_days 8`: 93 steps per epoch** |
| epochs | 100 | 50 | 100 (R5), 30 (small) |
| grad clipping | none | none | `grad_clip 1.0` |
| dropout | `HypergraphConv(..., dropout=0.5)` in both conv layers | none in the code read | none |
| init | `xavier_uniform_` for matrices, `uniform_(0,1)` for 1-d params | `glorot_uniform` dense head | section 2 |
| seed | fixed 123456789 | not fixed in what I read | seeds 0-24 |
| train-window rule | each epoch: shuffle `arange(valid_index)` (756 offsets), run the first `valid_index - seq - steps + 1` = 740. Offsets >= 740 have targets in [756, 771], so **validation-period days can appear as training targets** (by code reading; I did not run it) | same | train targets strictly `< valid_index` (`test_window_offsets_no_leak`) |
| input | STHAN-SR: raw 5-feature windows (files normalised by the full-series max); RSR: pre-trained LSTM embedding, not trained jointly | embedding | `norm: paper` (R5) = the same files, level inputs; small-scale `norm: train` |
| epoch selection | evaluates validation and test every epoch; no checkpoint-selection variable found by grep, not traced further | not traced | best validation Sharpe, leak-free |

Differences that could matter and are **unquantified** here: (i) price-ratio target versus direct return, (ii) one day versus eight days per step (740 vs 93 updates per epoch), (iii) validation days entering training in the repos, (iv) gradient clipping and weight decay on every model, (v) dropout 0.5 in STHAN-SR's conv layers. None of these is stated in the THINK paper (our values are INFERRED).
