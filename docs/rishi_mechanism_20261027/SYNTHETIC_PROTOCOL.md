# Synthetic learning gate — October 27, 2026

This completes an available portion of RESEARCH_PLAN §8.2: train the actual pinned dynamic adapter and an own-stock-only path on planted data. It does not replace missing financial forecasts or claim all of Phase D is complete.

Freeze before outcomes: 8 stock IDs, 16 sessions, 25 features; train 128 independent synthetic dates, evaluate 64 independent held-out dates. Data RNG 27102028, model seeds 7/19/42. Fixed pair groups {0,1}, {2,3}, {4,5}, {6,7}. Exactly 800 updates per fit, Adam LR .0001, source RMSE objective, gradient cap1, same dropout .1 in the full network. No early selection, hyperparameter search or extension based on outcome.

Synthetic input is Gaussian with SD .2. Feature0 is a stock-specific scalar broadcast over all 16 sessions, independently redrawn per stock/date. All other channels are independent Gaussian nuisance. Labels:

- Own: feature0/.2.
- Peer: paired other stock's feature0/.2, independent of the focal stock's history.
- Identity: fixed standardized ID vector, independent of every input feature. Same ID set across train/test.

Arms: full dynamic adapter unchanged; temporal-only path uses the same encoder/intensity/temporal modules and MLP with graph branches bypassed. This is a separately trained restricted path, not post-hoc component removal, but it is not parameter-equivalent and therefore not a financial architecture attribution experiment. Same initialization and optimizer recipe; each task/arm starts a fresh seed.

All labels/masks supplied explicitly; no financial data is read. Report held-out MSE, zero-predictor MSE, R² versus zero baseline and fit time. Recovery screen: R²≥.5 in all three seeds for a task designed to be representable. Temporal-only peer/identity scores should not systematically recover independently unobservable labels; R²>.2 in all three seeds triggers a data-generator/leakage audit, not a success declaration. A failed screen means this fixed learner/budget has not demonstrated recovery; do not infer the architecture cannot learn with another budget or use the failure as a financial null.

Measure the first 20 optimizer steps for the full/own/seed7 fit without evaluating held-out outputs; project 18fits×800steps from that rate. This is a resource smoke only. Cap the run at 15 minutes; save individual fit records and process progress. At a runtime cap retain incomplete status and do not change the learning recipe. Run CPU-only with two threads. Preserve existing results; use new `synthetic_gate1/` folder and refuse overwrites.

Missingness-only nuisance and changing-membership learned-recovery tasks remain separately specified follow-ups; structural mask/gradient tests already passed but are not learned-recovery tests. All synthetic recovery outcomes remain distinct from source financial claims.
