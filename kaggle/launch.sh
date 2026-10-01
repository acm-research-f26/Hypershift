#!/usr/bin/env bash
# Build bundles, upload both datasets (private), push the notebook and start it on Kaggle.
#   bash kaggle/launch.sh [SESSION=1|2|all] [TAG=s1]
# env: ACCEL=NvidiaTeslaT4 (GPU T4 x2, default; see README for P100), PRIOR_DATASET=<user>/<slug> (dataset holding results_*.zip),
#      FORCE_DATA=1 (re-version the big data dataset even if it exists), KAGGLE_USER, TIMEOUT_S (default 43200)
# Exit code 2 = no Kaggle credentials yet (nothing was uploaded).
set -u
# NOTE: datasets are uploaded with a bare relative -p from inside $B: the CLI builds its upload-state file name from the path and fails on "a/b/c" (Windows).
SESSION="${1:-1}"; TAG="${2:-s$SESSION}"; ACCEL="${ACCEL:-NvidiaTeslaT4}"
source "$(dirname "$0")/_kaggle_cli.sh" || exit $?
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=-1
B=kaggle/build
.venv/Scripts/python.exe kaggle/build_bundle.py --username "$KUSER" || exit 1

push_dataset() {  # $1 = slug, $2 = force (1 = version even if it exists)
  local slug=$1 st
  st=$(kg datasets status "$KUSER/$slug" 2>&1 | tr -d '\r' | tail -1)
  if echo "$st" | grep -qi "ready"; then
    if [ "$slug" = hypershift-rsr-data ] && [ "${FORCE_DATA:-0}" != 1 ]; then echo "$slug exists and is ready; keeping (FORCE_DATA=1 to re-version)"; return 0; fi
    (cd "$B" && kg datasets version -p "$slug" -m "bundle $(date +%F_%H%M) $(git rev-parse --short HEAD)" --dir-mode zip) || return 1
  else
    (cd "$B" && kg datasets create -p "$slug" --dir-mode zip) || return 1
  fi
  for i in $(seq 1 60); do
    st=$(kg datasets status "$KUSER/$slug" 2>&1 | tr -d '\r' | tail -1)
    echo "$slug: $st"; echo "$st" | grep -qi "ready" && return 0; sleep 20
  done
  echo "dataset $slug not ready after 20 min" >&2; return 1
}
push_dataset hypershift-code 1 || exit 1
push_dataset hypershift-rsr-data 0 || exit 1

SLUG="hypershift-run-$TAG"
K=$B/kernel; rm -rf "$K"; mkdir -p "$K"
.venv/Scripts/python.exe kaggle/make_notebook.py "$K/run_kaggle.ipynb" --session "$SESSION" --tag "$TAG" || exit 1
SRC="\"$KUSER/hypershift-code\", \"$KUSER/hypershift-rsr-data\""
[ -n "${PRIOR_DATASET:-}" ] && SRC="$SRC, \"$PRIOR_DATASET\""
cat > "$K/kernel-metadata.json" <<META
{"id": "$KUSER/$SLUG", "title": "$SLUG", "code_file": "run_kaggle.ipynb", "language": "python", "kernel_type": "notebook",
 "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "enable_tpu": "false",
 "dataset_sources": [$SRC], "competition_sources": [], "kernel_sources": [], "model_sources": []}
META
kg kernels push -p "$K" --accelerator "$ACCEL" -t "${TIMEOUT_S:-43200}" || exit 1
echo "kernel: https://www.kaggle.com/code/$KUSER/$SLUG"
sleep 15; kg kernels status "$KUSER/$SLUG"
echo "Follow with: bash kaggle/fetch.sh $TAG   (status; downloads the zip and merges when complete)"
