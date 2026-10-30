# Execution diagnostics

The `actual_resume_check` replays shared-THINK 2024 seed1868046758 from epoch7
through epoch8 and compares with the saved original epoch8 model/validation
forecasts. Both maximum absolute errors are zero. This is the check pinned by
the execution-only progress-file retry amendment.

The earlier `numerical_replay` copied a completed epoch8 checkpoint and performed
no additional training; its pass flag is not evidence of resumed computation.
It is retained as an exploratory diagnostic. Use `actual_resume_check` instead.

All scientific source runs and their frozen prefixes remain unchanged.
