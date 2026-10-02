# F: learnability of THINK on the known-signal benchmark (STATUS, work in progress)

*Phase 1.5 F, 2026-10-02. Checkpoint written when the laptop shut down; the final table is not done.*

## Status

No Kaggle kernels have been pushed for F. (`kernels list` shows only the older ks/ks2/ks-smoke/rsr-orig/s1-s4 kernels, none from F.)

## Tested so far (CPU diagnostic `scripts/f_diag.py`, 200 optimizer steps, `high` level, seed 0, batch 8; a diagnostic, not a training run)

1. **Weight decay collapse (strong evidence, mechanism).** In `relative` mode at init the loss gradient on each hyperbolic `z` is about 1e-5 to 1e-4 (norm), while the coupled-L2 gradient `5e-4*|w|` is about 3.6e-4: weight decay dominates by 10-40x. Adam then moves each weight by about `lr*sign(w)` toward 0. HH_hyper with wd 5e-4: `|z|` of (tconv1, tconv2, DHHAN fc) goes 0.71 -> 0.006, 0.04, 0.000 in 200 steps and pred sd 2.5e-4 -> 4e-12 (the D "1e-8 to 1e-12 collapse"). With wd 0 over the same 200 steps: `|z|` stays 0.70-0.71 (tconv2 0.48), pred sd stays about 3e-5, batch IC rises to about +0.2. Because `PoincareLinear` output is proportional to `|z|`, three stacked layers shrink multiplicatively.
2. **Init shrinks the signal ~10x per layer in relative mode** (activation norms 0.085 -> 0.024 (tconv1) -> 0.0016 (DHHAN) -> 0.0002 (tconv2)), so initial pred sd (2e-4) is 50x below the return sd (1e-2).
3. **Gradient clipping never fires** (pre-clip grad norm 4e-5 to 2e-3 in relative mode; 0.1-0.2 at the first step of level mode). It is irrelevant here.
4. **Level mode:** EE_none (no hyperbolic layers) also does not learn with wd 0 or 5e-4 (batch IC about 0 after 200 steps), so weight decay does not explain level-mode failure; the signal sits in 1% differences on top of O(1) level offsets.
5. DHHAN `a` gets gradient 2e-10 at init in relative mode (distance near 0), so attention is effectively a uniform mean (not yet measured over training).

## Code switches added (defaults unchanged; `tests/test_f_switches.py`, 9 tests pass)

`RunConfig`: `input_std` (per-channel train-window standardisation), `input_scale`, `init_gain` (PoincareLinear z init multiplier), `head_scale` (learnable output scale exp(t), excluded from weight decay), `spatial_residual` (DHHAN self path; DEPARTURE from eq. 15), `decoupled_wd` (AdamW), `log_ic` (per-epoch val/test IC and test pred sd in history.jsonl), and `grad_clip <= 0` now means clipping off. `eval.metrics.daily_ic` moved from `scripts/known_signal.py`.

## Left to do

- Kaggle grid (relative baseline, HH_hyper and EH_hyper at `high`/`group`, 3 seeds, one factor at a time: wd {5e-4, 0}, decoupled wd, lr, alpha, init_gain, head_scale, spatial_residual, batch_days 1/8, input_std/scale; level-mode input and `target=price` tested from level inputs only). 1-epoch smoke first.
- Final stage: best config on all arms incl. HH_none and EE_none, same seeds.
- Final table with fidelity labels (faithful / INFERRED / DEPARTURE), minimal config that learns, and the comparison against the oracle IC.
