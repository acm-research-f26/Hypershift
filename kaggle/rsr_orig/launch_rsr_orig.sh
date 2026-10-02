#!/usr/bin/env bash
# Upload the RSR-orig code dataset (once) and push the kernel hypershift-rsr-orig (smoke | full).
#   KAGGLE_USER=tomphamdustry bash kaggle/rsr_orig/launch_rsr_orig.sh smoke|full [--build-dataset]
set -u
MODE="${1:-smoke}"
source "$(dirname "$0")/../_kaggle_cli.sh" || exit $?
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=-1
B=kaggle/build
if [ "${2:-}" = "--build-dataset" ]; then
  (cd "$B" && if kg datasets status "$KUSER/hypershift-rsr-orig-code" 2>&1 | tail -1 | grep -qi ready; then
     kg datasets version -p hypershift-rsr-orig-code -m "rsr orig" --dir-mode zip; else kg datasets create -p hypershift-rsr-orig-code --dir-mode zip; fi) || exit 1
  for i in $(seq 1 60); do
    st=$(kg datasets status "$KUSER/hypershift-rsr-orig-code" 2>&1 | tr -d '\r' | tail -1); echo "dataset: $st"
    echo "$st" | grep -qi ready && break; sleep 20
  done
fi
SLUG=hypershift-rsr-orig
K=$B/kernel_rsr; rm -rf "$K"; mkdir -p "$K"
cp kaggle/rsr_orig/run_rsr_orig.py "$K/run_rsr_orig.py"
if [ "$MODE" = smoke ]; then sed -i 's/^SMOKE = False/SMOKE = True/' "$K/run_rsr_orig.py"; fi
cat > "$K/kernel-metadata.json" <<META
{"id": "$KUSER/$SLUG", "title": "$SLUG", "code_file": "run_rsr_orig.py", "language": "python", "kernel_type": "script",
 "is_private": "true", "enable_gpu": "${GPU:-true}", "enable_internet": "true", "enable_tpu": "false",
 "dataset_sources": ["$KUSER/hypershift-rsr-data", "$KUSER/hypershift-rsr-orig-code"], "competition_sources": [], "kernel_sources": [], "model_sources": []}
META
if [ "${GPU:-true}" = true ]; then kg kernels push -p "$K" --accelerator "${ACCEL:-NvidiaTeslaT4}" -t 39600 || exit 1; else kg kernels push -p "$K" -t 39600 || exit 1; fi
echo "kernel: https://www.kaggle.com/code/$KUSER/$SLUG"
