# Phase 1.5 E: the authors' ORIGINAL RSR-I code on NYSE, scored with our evaluator

*2026-10-02. Question: is "no skill" shared across implementations (protocol or data issue) or specific to THINK? Method: run Feng et al. 2019's own code (`fulifeng/Temporal_Relational_Stock_Ranking`, commit `cfbb01b`, TF1) on Kaggle GPU with their README NYSE command, save their per-day predictions, and score them with `hypershift.eval.metrics.evaluate_all` on our test days and masks. Scorer: `scripts/rsr_orig_score.py`; per-seed numbers: `docs/phase1_5/E_rsr_original_scores.json`. Kernel: https://www.kaggle.com/code/tomphamdustry/hypershift-rsr-orig (private; version 5, full run).*

## 1. Result in one paragraph

The original code ran (5 seeds, 20 min per seed on a T4, 50 epochs each). Scored by our evaluator on the 237 test days (2017), their RSR-I has **Sharpe 0.45 ± 0.46 at the epoch their own code selects (min validation loss), 0.93 ± 0.24 at the epoch our rule selects (max validation Sharpe), and 1.41 ± 0.27 at the best test epoch (oracle, the paper-style protocol)**. Hold-all is 1.531. No seed beats hold-all except two at the oracle epoch. NDCG@5 is 0.563-0.567 against random 0.5639, per-seed IC is within +-0.01 of zero. So the authors' own code, with their data and their pretrained embedding, shows **the same "no skill beyond noise" picture as our RSR-I reimplementation** (leak-free 0.879 ± 0.443, best-test 2.369, NDCG@5 0.5669, IC +0.0106). That points to a protocol / data / evaluation-window property shared by implementations, not a THINK-specific failure. It does not say the paper's 1.05 is wrong: the paper's Sharpe formula and selection are not ours (PA4), and the oracle number (1.41) is in the neighbourhood of the paper's table (1.05 to 1.18) but that is selection on test.

## 2. What was run (their code, their README command)

`python relation_rank_lstm.py -m NYSE -l 8 -u 32 -a 10 -e NYSE_rank_lstm_seq-8_unit-32_0.csv.npy -g 1` (README "NYSE"). From reading the code:
- **Model = RSR-I (implicit)** because `-ip` defaults 0: weight = head/tail dense scores plus `leaky_relu(dense(relation))` ("sum weight"), not the inner product. Note: this is the README's NYSE command, whose `-ip` is 0; our R8 reimplementation used the inner-product form `(f f^T) * rel_weight`. So the two are **different RSR variants** (in the authors' code `-ip 1` = "inner product weight"; the README does not say which one the paper calls RSR-I; UNKNOWN). This is the main model difference to keep in mind when comparing to our 0.88.
- **Relations: `sector_industry` only** (default `-rn`), 108 channels; no Wikidata relation. Our R8 used industry + wiki (K = 141).
- **Two-stage training**: the LSTM (32 units, window 8) is *pre-trained* (`rank_lstm.py`), its per-day embedding `[1737, 1245, 32]` is a fixed input; stage 2 trains only the relation layer and the output layer (Adam lr 1e-3, alpha 10, 50 epochs hard-coded, one full cross-section per step, 748 steps/epoch). The released `rank_lstm.py` does **not** save the embedding; the authors' pretrained file (Google Drive link in their README, md5 `254ec420...`) was downloaded and used unchanged. How/where that LSTM was trained and selected (train only, or with test info) is **UNKNOWN**: the repo does not say.
- **Their splits**: train offsets < 756, valid 756-1007, test 1008-1244; identical to ours (`valid_index 756`, `test_index 1008`, T = 1245). Test column c = target day 1008 + c = our test day c (237 days). Verified: their `gt` equals our saved `test_gt` to 1.2e-7 on all stock-days both mask (410,305 pairs).
- **Their mask** (min over the 9-day window, missing = -1234) is **521 stock-days larger** than ours (410,826 vs 410,305; ours is a subset, 0.13%): our loader masks something theirs keeps. Scoring uses OUR gt and mask (primary); using theirs changes Sharpe by < 0.01 (hold-all 1.528 vs 1.531).
- **Their evaluator** (`evaluator.py`): per day, top-1/5/10 sets with ticker-index sets; it **computes no Sharpe**. It returns `mse` (masked), `mrrt` (mean reciprocal rank of the top-1) and `btl` (1 + sum of the daily top-1 return, i.e. a cumulative top-1 return, not annualized). (The x15.87 Sharpe belongs to STHAN-SR's evaluator, per audit A.)
- **Their epoch-selection rule**: after each epoch they evaluate val and test; the "best" epoch is the one with the **lowest mean validation loss** (`loss` = masked MSE + alpha x ranking hinge, summed over validation days; strict `<`); the printed "Best Test performance" is the test performance at that epoch. This does not use test data (leak-free), but it does not use Sharpe either.
- **Seeds**: the code fixes seed 123456789 everywhere. The first run uses it unchanged ("authors' seed"); seeds 1-4 use patch P2. Because the embedding is fixed, seeds vary only the init of ~110 relation weights + the 33-parameter head and the batch order, so the seed spread understates what a full retrain (LSTM included) would show.

## 3. TF1 on Kaggle: what was needed

`tensorflow==1.15` needs Python <= 3.7 and CUDA 10; the Kaggle image has Python 3.12 and TF 2.20 (CUDA works). I did not try to install an old Python. Route: TF2 with `tensorflow.compat.v1` + `disable_v2_behavior()`. Every patch is in `kaggle/rsr_orig/patch_rsr.py` (applied by string replacement with asserted occurrence counts; the original is kept as `.orig`; the unmodified files come from the pinned commit with md5 checks in `build_rsr_orig.py`; the patched file is saved in the output as `relation_rank_lstm.patched.py`):

| patch | change | affects results? |
|---|---|---|
| P1 | `import tensorflow as tf` -> `import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()`; plus a drop-in `tf.layers.dense` (glorot-uniform kernel, zero bias, `tensordot` for rank 3), used only because `tf.compat.v1.layers` does not exist in TF >= 2.16 (checked: TF 2.21 locally, 2.20 on Kaggle) | No intended change. Same init and maths as TF1 `tf.layers.dense`; **not verified bit-for-bit against TF1** (no TF1 available) |
| P2 | `seed = 123456789` (3 places) -> `int(os.environ.get('RSR_SEED', '123456789'))` | Default unchanged |
| P3 | `epochs=50` -> env `RSR_EPOCHS` (default 50) | No |
| P4 | after their `evaluate()` calls, dump per-epoch val/test predictions, gt, mask and their per-epoch perf dicts to `RSR_OUT` | No (read-only) |
| P5 | save the 6 arrays that `train()` returns (their val-loss "best" epoch) | No |

No change to the model, loss, optimiser, batches, selection or evaluator. Other TF2-vs-TF1 differences I did not patch and cannot rule out: Adam epsilon/default handling, `tf.layers`-removed shim, GPU numerics. A 1-epoch smoke ran first on Kaggle CPU (558 s/epoch; GPU sessions were full) and the full run on T4: epoch 0 = 202 s (graph build + 1.3 GB constant), then 19.7 s/epoch, 20.3 min per seed.

**Data identity.** Local `data/raw/rsr/data` vs the GitHub clone: `2013-01-01/` (2,817 files) identical md5 for md5 (aggregate `d8a0e26a...`), `relation.tar.gz` and the NYSE ticker/wiki csvs identical. On Kaggle the kernel verified the 1,737 NYSE price files (aggregate md5 `276fa524...`) and `NYSE_industry_relation.npy` (md5 `c427f780...`) match the local copy before training. So **we used their data, byte for byte**.

## 4. Numbers (NYSE, 237 test days; our evaluator: top-5, Sharpe annualized sqrt(252), no costs)

Sharpe by epoch-selection rule (5 seeds; mean ± std ddof 1):

| selection | mean Sharpe | per seed (123456789, 1, 2, 3, 4) | NDCG@5 | IC (daily Spearman) | seeds > hold-all |
|---|---|---|---|---|---|
| **Their rule** (min val loss; "as printed" epoch; leak-free) | **0.447 ± 0.456** | 0.10, 0.01, 0.46, 1.17, 0.50 | 0.5629 | -0.001 | 0/5 |
| **Our rule** (max val Sharpe; leak-free) | **0.931 ± 0.240** | 1.06, 1.04, 0.93, 1.11, 0.52 | 0.5645 | +0.001 | 0/5 |
| Oracle (max test Sharpe, paper-style; diagnostic) | 1.410 ± 0.271 | 1.47, 1.05, 1.71, 1.22, 1.60 | 0.5667 | +0.003 | 2/5 |
| Last epoch (49) | 0.643 ± 0.367 | | 0.5640 | 0.000 | |

Selected epochs (their, ours, oracle) per seed: 47/39/46, 42/43/47, 45/36/21, 48/43/12, 46/35/20. Hold-all (same days, our mask) **1.531**; random NDCG@5 0.5639 (R8 doc); our reimplementation (10 seeds, 100 epochs) for comparison: leak-free 0.879 ± 0.443, best-test 2.369 ± 0.219, NDCG@5 0.5669, IC +0.0106 ± 0.0027, paper Table II p852 RSR-I 1.05 (different Sharpe formula, PA4).

What their own evaluator prints at their selected epoch (test, 2017): `btl` (1 + sum of daily top-1 returns) 2.48, 1.16, 2.75, 2.23, 1.64 for seeds 123456789, 1, 2, 3, 4 (validation `btl` 1.21, 2.31, 0.98, 2.24, 1.80), `mrrt` 0.040-0.049 (a random top-1 gives about 1/N x ln N = 0.004; this ratio is **not** evidence of skill, see below), `mse` about 2.27e-4. Top-1 `btl` is a very noisy number (one stock a day): seeds differ by 1.6x with the same data. Note that `btl` and our top-5 Sharpe at the same epoch disagree in rank across seeds (seed 123456789: btl best of the five but top-5 Sharpe 0.10).

Random tie-break (audit A request): the predictions are continuous float32 returns; on average 1,731 distinct values per day among 1,731 masked stocks, so **there are no ties** and the random tie-break Sharpe equals the stable-argsort Sharpe to 1e-16 in the smoke test and in the full run (`sr_random_tiebreak_mean` in the JSON, 20 draws). Ties are not an issue for this model (unlike near-constant THINK runs).

Further diagnostics at the epochs above: `ndcg_sthan` (the STHAN-SR last-day evaluator, reference only) 0.83 ± 0.08 (their rule), 0.78 (ours, oracle), the same 0.76-0.87 band as every no-skill model in R8; distinct daily top-5 sets over 237 days: 186 (their rule), 66 (our rule), 70 (oracle): this model rotates holdings, it is not a fixed portfolio like R8's STHGCN.

## 5. Reading (what this does and does not show)

1. **Shared, not THINK-specific.** The authors' code on the authors' data and pretrained embedding does not beat hold-all or random-NDCG leak-free under our evaluator. Selecting on validation gives Sharpe 0.45 to 0.93 against hold-all 1.53; the validation-to-test disconnect also appears here (validation Sharpe at their selected epoch: 0.99-1.49, test 0.01-1.17). The only way to get >1.2 is to pick the epoch on test (oracle 1.41), which is the paper's protocol according to R8/C1, and which C1's shuffled-label control shows is also reachable without signal. So the missing THINK advantage in Phase 1 cannot be blamed solely on our THINK implementation: the baseline from the paper's authors behaves the same way on this split.
2. **Not a replication of the paper's 1.05.** The paper's Sharpe formula is not ours; the oracle number 1.41 vs our oracle 2.37: ours is higher because 100 epochs of a fully trainable model give more epochs and more freedom to overfit the test curve; theirs trains 2 layers for 50 epochs. Leak-free numbers agree within noise (0.93 ± 0.24 with our rule, 0.88 ± 0.44 for ours).
3. **Their selection rule is worse than ours on Sharpe** (0.45 vs 0.93) but with 5 seeds (std 0.46) I do not claim a difference; both are below hold-all.
4. **Different RSR variants**: they ran sum-weight, industry-only, pretrained LSTM; our R8 ran inner-product, industry + wiki, end-to-end. The agreement therefore is across implementations and variants, which makes the shared conclusion a bit stronger, but no matched ablation was run.

## 6. Caveats and unknowns

- 5 seeds; the embedding is fixed (UNKNOWN whether its pretraining used test information; the README does not say), so seed spread is understated. A seed does not change the pretrained LSTM.
- TF2 compat run, not TF1 (patch P1 shim not bit-checked against TF1).
- Our mask differs from theirs in 521 stock-days (0.13%); cause not investigated (INFERRED: our loader drops days where any feature is missing, theirs only checks the close column; not tested).
- `mrrt` baseline for random 0.004 is my estimate (INFERRED), not computed.
- Only NYSE, only the README command; NASDAQ and `-rn wikidata`, `-ip 1` not run.
- The kernel `hypershift-rsr-orig` is private on account `tomphamdustry`. Raw predictions (about 170 MB per seed) stay in `kaggle/build/out_rsr_full/` (git-ignored via `kaggle/build`), not committed.

## 7. Reproduce

```bash
export KAGGLE_USER=tomphamdustry
python kaggle/rsr_orig/build_rsr_orig.py --repo <clone of fulifeng repo at cfbb01b> --pretrain <NYSE_rank_lstm_seq-8_unit-32_0.csv.npy> --username $KAGGLE_USER
bash kaggle/rsr_orig/launch_rsr_orig.sh smoke --build-dataset   # then: full  (GPU=false for a CPU smoke; push_when_free.sh retries while 2 GPU sessions are busy)
kaggle/.venv-kaggle/Scripts/python.exe -m kaggle.cli kernels output $KAGGLE_USER/hypershift-rsr-orig -p kaggle/build/out_rsr_full
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src .venv/Scripts/python.exe scripts/rsr_orig_score.py kaggle/build/out_rsr_full/rsr_orig_out
```
