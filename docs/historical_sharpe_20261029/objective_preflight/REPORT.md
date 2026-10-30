# Synthetic objective preflight

Protocol `425bf03`; implementation `4a4dd4f` plus pre-run turnover correction `e1076ce`. Three linear models trained for 150 Adam steps on the same synthetic training data. This is not financial training or a historical strategy result. The planted signal and model are deliberately simple.

| Objective | Mean test rank correlation | Test surrogate loss (lower better) | Mean target turnover |
|---|---:|---:|---:|
| MSE | 0.971 | 0.00108 | 0.0207 |
| Pairwise ranking | 0.972 | -0.00250 | 0.8233 |
| Return/risk/turnover surrogate | 0.965 | -0.00235 | 0.7912 |

The equal-weight starting portfolio test surrogate loss was 0.00114 and its average turnover approximately 0.0125, counting the initial investment. Pairwise and portfolio objectives recover the planted ranking and improve the net surrogate over initialization, but their turnover is very high. The cost penalty does not ensure low turnover; its scale matters. Pairwise training loss dropped from .6931 to .1701; MSE from .000269 to .00000409. Initial all-equal scores have undefined rank correlation, so no numerical rank-correlation change from initialization is asserted.

Two unit tests checked finite gradients, ranking direction, label ties, inactive mask and penalty response. Training slices only the first 80 dates and evaluation the later 40; a separate end-to-end future-data mutation test of training has not yet been performed. An independent NumPy calculation recomputed all three saved test rank correlations, surrogate losses and turnovers to 4.96e-8. These checks support a limited implementation smoke test; before financial fits, add chronology mutation tests, address calibration/scale, freeze a matched architecture, per-fold and seed budget, and validate using actual account replays. No checkpoint gate was waived.
