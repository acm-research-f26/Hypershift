#!/usr/bin/env bash
# Phase 1.5 F: run scripts/r5f_analysis.py on Kaggle (CPU kernel, no GPU quota) against the r5f/r8f training-kernel outputs,
# so the laptop only downloads small md/json/png files. Does NOT touch kaggle/build (other jobs use it) and does NOT re-push
# any training kernel: uploads a small separate code dataset `hypershift-code-an` and pushes a new script kernel.
#   bash kaggle/launch_analysis.sh                  # real analysis: r5f + r8f outputs -> kernel hypershift-run-r5f-an
#   bash kaggle/launch_analysis.sh smoke            # end-to-end test on hypershift-run-r5f-smoke + r8f-smoke (tiny)
#   bash kaggle/fetch_analysis.sh r5f-an F_r5f --install     # status; when COMPLETE downloads the small outputs
# Both source kernels must be COMPLETE (a mounted kernel provides its latest finished version's output).
# Overridable env: SOURCES="slug1 slug2" (kernel slugs under $KUSER), R5_PREFIX, R8_PREFIX, NORMS, OUT_STEM, TAG, DRAWS, BOOT
set -u
MODE="${1:-real}"
if [ "$MODE" = smoke ]; then
  : "${SOURCES:=hypershift-run-r5f-smoke hypershift-run-r8f-smoke}"; : "${R5_PREFIX:=R5_f_smoke}"; : "${R8_PREFIX:=R8_f_smoke}"
  : "${NORMS:=train}"; : "${OUT_STEM:=F_r5f_smoke}"; : "${TAG:=r5f-an-smoke}"; : "${DRAWS:=3}"; : "${BOOT:=300}"
else
  : "${SOURCES:=hypershift-run-r5f hypershift-run-r8f}"; : "${R5_PREFIX:=R5_f}"; : "${R8_PREFIX:=R8_f}"
  : "${NORMS:=paper train}"; : "${OUT_STEM:=F_r5f}"; : "${TAG:=r5f-an}"; : "${DRAWS:=10}"; : "${BOOT:=5000}"
fi
source "$(dirname "$0")/_kaggle_cli.sh" || exit $?
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=-1
B=kaggle/build_an; rm -rf "$B"; mkdir -p "$B/hypershift-code-an" "$B/kernel"
D=$B/hypershift-code-an
cp -r src scripts configs pyproject.toml "$D"/ && rm -rf "$D/scripts/queues" "$D/scripts/__pycache__"
find "$D" -name __pycache__ -type d -prune -exec rm -rf {} +
echo "git HEAD: $(git rev-parse HEAD); analysis bundle $(date +%F_%H%M)" > "$D/BUNDLE_INFO.txt"
cat > "$D/dataset-metadata.json" <<META
{"title": "hypershift-code-an", "id": "$KUSER/hypershift-code-an", "subtitle": "THINK code for the CPU analysis kernel", "description": "src, scripts, configs", "licenses": [{"name": "other"}]}
META
st=$(kg datasets status "$KUSER/hypershift-code-an" 2>&1 | tr -d '\r' | tail -1)
if echo "$st" | grep -qi "ready"; then
  (cd "$B" && kg datasets version -p hypershift-code-an -m "an $(date +%F_%H%M) $(git rev-parse --short HEAD)" --dir-mode zip) || exit 1
else
  (cd "$B" && kg datasets create -p hypershift-code-an --dir-mode zip) || exit 1
fi
for i in $(seq 1 60); do
  sleep 15
  st=$(kg datasets status "$KUSER/hypershift-code-an" 2>&1 | tr -d '\r' | tail -1); echo "code-an: $st"
  echo "$st" | grep -qi "ready" && break
done
K=$B/kernel; SLUG="hypershift-run-$TAG"
NORMS_PY=""; for n in $NORMS; do NORMS_PY="$NORMS_PY${NORMS_PY:+, }\"$n\""; done
{ echo "R5_PREFIX = \"$R5_PREFIX\"; R8_PREFIX = \"$R8_PREFIX\"; NORMS = [$NORMS_PY]; OUT_STEM = \"$OUT_STEM\"; DRAWS = $DRAWS; BOOT = $BOOT"
  cat kaggle/analysis_kaggle.py; } > "$K/analysis.py"
KS=""; for s in $SOURCES; do KS="$KS${KS:+, }\"$KUSER/$s\""; done
cat > "$K/kernel-metadata.json" <<META
{"id": "$KUSER/$SLUG", "title": "$SLUG", "code_file": "analysis.py", "language": "python", "kernel_type": "script",
 "is_private": "true", "enable_gpu": "false", "enable_internet": "false", "enable_tpu": "false",
 "dataset_sources": ["$KUSER/hypershift-code-an"], "competition_sources": [], "kernel_sources": [$KS], "model_sources": []}
META
cat "$K/kernel-metadata.json"
kg kernels push -p "$K" || exit 1
echo "kernel: https://www.kaggle.com/code/$KUSER/$SLUG"
sleep 15; kg kernels status "$KUSER/$SLUG"
echo "Follow with: bash kaggle/fetch_analysis.sh $TAG $OUT_STEM --install"
