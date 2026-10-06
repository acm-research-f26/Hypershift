# Frozen universe inputs

`modern_source_snapshot.json` records the original fixed twelve-name basket and
the first downloaded modern snapshot's provenance. The downloader reads only its
ticker list; it records new download hashes in the new snapshot directory.

`NYSE_source_cohort.txt` and `NASDAQ_source_cohort.txt` preserve the published
research cohorts from the pinned Temporal Relational Stock Ranking repository:

https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d/data

The research downloader chooses the first 100 SHA256-sorted tickers per cohort,
then orders those names alphabetically. This selection never uses returns or
liquidity. It still inherits the source cohorts' survival/selection bias; these
lists are not point-in-time index membership data.

The active 2024–2025 extension uses these same frozen candidate lists with
modern public prices; unavailable names are never replaced. Its per-name
coverage audit is in `runs/recent_2024_2025/universe_coverage.csv`.
`recent_identity_exclusions.json` documents two current symbols that refer to
different issuers from those in the 2017 cohorts. The raw downloads are retained,
but those mismatched price columns are masked before modeling or trading.
