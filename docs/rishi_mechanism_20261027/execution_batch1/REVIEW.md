# Independent review — first executed batch

Read-only subagent review covered the batch protocol, analysis implementation, two new test modules and saved numerical outputs. It did not run the compatibility launcher concurrently or access missing raw market inputs.

Verified: six input hashes; twelve workbook paths × 1,003 observations; terminal wealth/CAGR arithmetic within 1.12e-16; initial-capital pool arithmetic within 2.0e-15; synchronized stationary bootstrap, centered mean null, Holm families, source-session CAGR and consistent model-ID transport. No calculation or scientific-interpretation blocker was found.

One P2 finding: completed/partial output could be overwritten by the default runner. Added test_completed_or_partial_output_cannot_be_overwritten; observed import failure before implementing reserve_output, then implemented the guard and ran the combined suite: **31 passed**. The guard preserves nonempty destinations. No saved numerical output was rerun or overwritten as part of this fix. Test logs record the initial 30-test run and final 31-test run separately.

Lead retained source 3% hurdle versus zero-interest cash, selected-history uncertainty, distinct monthly versus daily estimands, source-only fee/turnover status, capital-pool AUM limits and issuer/exposure confounds. Broader financial learner tests and raw replay are explicitly incomplete because the original artifacts are absent.
