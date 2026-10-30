# Extended-hours timing recheck

Using the published calendar, summary timestamps and 350 saved decision rows, `feature_timing_check_20261029.py` checks all **494,048** recorded summary rows against their same-day 20:00 Eastern after-hours or 09:15 premarket cutoff. It checks all **11,200 feature-date slots** in the declared 16-session after-hours and next-session premarket windows against each decision's cutoff, the five-minute cutoff-to-decision separation, next-exchange-session entry and recorded regular-close fill. Four entry dates are early closes. All checks pass.

The check uses the latest recorded bar end aggregated across symbols for each segment/date; a missing symbol's summary is not thereby assumed complete. It verifies chronology of the published summaries and declared calendar, not original provider publication times, revisions, transmission latency, availability of full data by 09:20, or actual execution. The five-minute lag remains an assumption. It does not alter the strict checkpoint gate or authorize prospective use.

The first script invocation failed because a Pandas column name collided with the `last` method. Bracketed column selection corrected the local checker; the successful run made no changes to source artifacts.
