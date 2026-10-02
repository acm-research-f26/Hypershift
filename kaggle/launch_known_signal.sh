#!/usr/bin/env bash
# Phase 1.5 D: run the known-signal grid (notebook preset 5, or 5s = smoke) on Kaggle WITHOUT touching kaggle/build (other jobs use it).
# Uploads only a small code dataset `hypershift-code-ks` (src, scripts, configs, pyproject) and re-uses the existing `hypershift-rsr-data`.
#   KAGGLE_USER=tomphamdustry bash kaggle/launch_known_signal.sh 5s ks_smoke    # smoke test
#   KAGGLE_USER=tomphamdustry bash kaggle/launch_known_signal.sh 5  ks          # full grid
#   bash kaggle/fetch.sh ks   (fetch.sh uses slug hypershift-run-<TAG>, same as here)
set -u
SESSION="${1:-5s}"; TAG="${2:-ks_smoke}"; ACCEL="${ACCEL:-NvidiaTeslaT4}"
source "$(dirname "$0")/_kaggle_cli.sh" || exit $?
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=-1
B=kaggle/build_ks; rm -rf "$B"; mkdir -p "$B/hypershift-code-ks"
D=$B/hypershift-code-ks
cp -r src scripts configs pyproject.toml "$D"/ && rm -rf "$D/scripts/queues" "$D/scripts/__pycache__"
find "$D" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -f "$D/configs/chosen.yaml" "$D/results/tuned.json"
echo "git HEAD: $(git rev-parse HEAD); known-signal bundle $(date +%F_%H%M)" > "$D/BUNDLE_INFO.txt"
cat > "$D/dataset-metadata.json" <<META
{"title": "hypershift-code-ks", "id": "$KUSER/hypershift-code-ks", "subtitle": "THINK code for known-signal runs", "description": "src, scripts, configs", "licenses": [{"name": "other"}]}
META
st=$(kg datasets status "$KUSER/hypershift-code-ks" 2>&1 | tr -d '\r' | tail -1)
if echo "$st" | grep -qi "ready"; then
  (cd "$B" && kg datasets version -p hypershift-code-ks -m "ks $(date +%F_%H%M)" --dir-mode zip) || exit 1
else
  (cd "$B" && kg datasets create -p hypershift-code-ks --dir-mode zip) || exit 1
fi
for i in $(seq 1 60); do
  st=$(kg datasets status "$KUSER/hypershift-code-ks" 2>&1 | tr -d '\r' | tail -1); echo "code-ks: $st"
  echo "$st" | grep -qi "ready" && break; sleep 20
done
SLUG="hypershift-run-$TAG"; K=$B/kernel; mkdir -p "$K"
.venv/Scripts/python.exe kaggle/make_notebook.py "$K/run_kaggle.ipynb" --session "$SESSION" --tag "$TAG" || exit 1
cat > "$K/kernel-metadata.json" <<META
{"id": "$KUSER/$SLUG", "title": "$SLUG", "code_file": "run_kaggle.ipynb", "language": "python", "kernel_type": "notebook",
 "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "enable_tpu": "false",
 "dataset_sources": ["$KUSER/hypershift-code-ks", "$KUSER/hypershift-rsr-data"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
META
kg kernels push -p "$K" --accelerator "$ACCEL" -t "${TIMEOUT_S:-43200}" || exit 1
echo "kernel: https://www.kaggle.com/code/$KUSER/$SLUG"
sleep 15; kg kernels status "$KUSER/$SLUG"
