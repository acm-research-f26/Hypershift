# Study FB Task 8: learned-hyperedge collapse (stage 1, synthetic only)

**Outcome: the predeclared fix did not pass its test and was reverted. The diagnosis changed what the v3 "collapse" means.** No real-data run was made, and nothing scored any year.

## Setup

- **Data:** synthetic and fresh every training step (`scripts/featbench_soft_diag.py`). 64 stocks in 4 planted groups of 16. Each group has a white-noise factor over 64 sessions, and each stock observes its factor plus own noise σ = 1 on 5 features.
- **Target:** the group factor on the last day. It can only be estimated well by pooling the stock's own group. The own-only optimum MSE is 0.167; pooling over the true group gives about 0.0125.
- **Models:** `none` (no graph); the v3 `soft` arm; and the `soft_t` variant. `soft_t` uses temperature 0.1, a learnable membership prior initialised from a deliberately noisy partition (half the nodes reassigned), and an entropy-balance penalty of 0.1.
- **Measures:** after 600 Adam steps (lr 0.01), fresh-batch MSE, assignment entropy and the adjusted Rand index (ARI) of argmax assignments vs the planted groups.

## Results (600 steps, σ = 1)

| Model | Fresh MSE | Assignment entropy (uniform = 2.77) | ARI vs planted groups |
|---|---|---|---|
| none | 0.171 | — | — |
| soft (v3 arm) | 0.128 | 1.97 | — |
| soft_t | 0.128 | 0.19 | 0.02 (the noisy initial partition had 0.23) |

## Reading

- **The v3 soft arm is able to move off uniform** when there is group signal to exploit: entropy falls from 2.77 to 1.97 and it beats no graph. On the real S&P 500 panel its entropy stayed at log 16 in all 12 fits. That is consistent with "no exploitable group signal in the price data", not only with a broken mechanism.
- **Sharpening the assignment does not make it learn identity.** `soft_t` becomes confident but groups stocks by their current hidden state, which reflects today's values, rather than by who they are. ARI ends at 0.02, below the noisy starting partition. Both soft arms beat no graph by pooling similar-looking stocks, and both stay far from the true-group optimum (0.128 vs about 0.0125).
- **A stage-2 real-data test needs a different design.** The assignment should be identity-based, for example a learnable per-stock embedding or an assignment fixed per stock across dates, rather than a function of the day's features. It should be predeclared as its own study and pass this synthetic test first.
- **Debugging notes:** two earlier versions of the synthetic test were invalid, and both are recorded in the ledger. A smooth random-walk factor let each stock denoise from its own history. A fixed 16-day training set let the model memorise. Only fresh batches with a white-noise factor make pooling necessary.
