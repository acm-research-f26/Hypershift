#!/usr/bin/env bash
# Status of an analysis kernel and, when COMPLETE, download its small outputs (md/json/png/log) to kaggle/build/out_<TAG>/.
#   bash kaggle/fetch_analysis.sh r5f-an F_r5f [--install]
#   --install copies <STEM>_results.md into docs/phase1_5/ and <STEM>_fig.png into docs/figures/ (never touches results/).
set -u
TAG="${1:-r5f-an}"; STEM="${2:-F_r5f}"; INSTALL="${3:-}"
source "$(dirname "$0")/_kaggle_cli.sh" || exit $?
cd "$ROOT" || exit 1
SLUG="$KUSER/hypershift-run-$TAG"
ST=$(kg kernels status "$SLUG" 2>&1 | tr -d '\r'); echo "$ST"
if echo "$ST" | grep -qi "error"; then echo "WARNING: kernel ended in ERROR; downloading logs." >&2
elif ! echo "$ST" | grep -qi "complete"; then echo "not complete yet. Re-run later."; exit 3; fi
OUT=kaggle/build/out_$TAG; mkdir -p "$OUT"
kg kernels output "$SLUG" -p "$OUT" -o || exit 1
ls -la "$OUT"
if [ "$INSTALL" = "--install" ] && [ -f "$OUT/${STEM}_results.md" ]; then
  cp "$OUT/${STEM}_results.md" "docs/phase1_5/${STEM}_results.md"; cp "$OUT/${STEM}_fig.png" docs/figures/ 2>/dev/null
  echo "installed docs/phase1_5/${STEM}_results.md"
fi
