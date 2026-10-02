#!/usr/bin/env bash
# Check a pushed kernel and, when finished, download results_<TAG>.zip. Usage: bash kaggle/fetch.sh [TAG=s1] [--merge]
#   --merge: after download run merge_results.sh on the zip (never overwrites local runs).
set -u
TAG="${1:-s1}"; MERGE="${2:-}"
source "$(dirname "$0")/_kaggle_cli.sh" || exit $?
cd "$ROOT" || exit 1
SLUG="$KUSER/hypershift-run-${TAG//_/-}"
ST=$(kg kernels status "$SLUG" 2>&1 | tr -d '\r'); echo "$ST"
if echo "$ST" | grep -qi "error"; then echo "WARNING: the kernel version ended in ERROR; downloading whatever output exists (may be partial or empty)." >&2
elif ! echo "$ST" | grep -qi "complete"; then echo "not complete yet (queued/running). Re-run later."; exit 3; fi
OUT=kaggle/build/out_$TAG; mkdir -p "$OUT"
kg kernels output "$SLUG" -p "$OUT" -o --file-pattern "results_.*\.zip|.*\.log" || exit 1
ls -la "$OUT"
if [ "$MERGE" = "--merge" ]; then bash kaggle/merge_results.sh "$OUT/results_$TAG.zip"; else echo "next: bash kaggle/merge_results.sh $OUT/results_$TAG.zip [--dry-run]"; fi
