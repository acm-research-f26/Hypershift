# Pre-inference portability amendment — October27,2026

The first raw replay attempt validated all1184published files, then stopped at the first parent checkpoint before any training/validation/test inference. PyTorch weights-only loading rejects NumPy scalar group IDs saved by the original checkpoint. Static checkpoint inspection identifies only `numpy.dtype` and `numpy._core.multiarray.scalar`; the dynamic checkpoint has no additional globals.

Allow only those known metadata constructors plus NumPy's int64dtype class in a scoped weights-only loading context; reject other unrecognized globals. Do not rewrite checkpoints, change weights, use unrestricted deserialization or alter forecasts/selection/account rules. A fixture test preserves NumPy integer group metadata and tensor state.

Keep the failed preflight's `execution_batch2/PUBLICATION_VERIFICATION.json` unchanged. Retry into fresh `execution_batch2_portable/` and `results/RISHI_MECHANISM_REPLAY_20261027_PORTABLE/`. Add explicit output-directory options to support preserved, reproducible attempts. The original empty large-output directory may remain empty. This is a pre-outcome environment amendment, not result-driven tuning or a changed scientific batch.
