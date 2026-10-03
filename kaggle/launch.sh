#!/usr/bin/env bash
# Build bundles, upload both datasets (private), push the notebook and start it on Kaggle.
#   bash kaggle/launch.sh <SESSION|PRESET> [TAG]
#     SESSION = 1|2|3|...|11|all (numeric presets; default TAG s<SESSION>)  or a named preset (default TAG = the preset name):
#       r8f-top   top-up of R8_f (mounts the hypershift-run-r8f output; needs r8f COMPLETE or ERROR)      -r8f-top-s = its smoke
#       p1f       Phase 1 small-scale arms rerun with the F fix (+ R7 clf, R8 small), in-kernel analysis   -p1f-s     = its smoke
#       r5f2      full NYSE alpha=0 and spatial_residual on top of the F fix (HH, EH), in-kernel analysis -r5f2-s    = its smoke
#     A full named preset refuses to start (exit 3) until its smoke kernel hypershift-run-<preset>-s is COMPLETE (SKIP_SMOKE_GATE=1 overrides).
# env: ACCEL=NvidiaTeslaT4 (GPU T4 x2, default; see README for P100), PRIOR_DATASET=<user>/<slug> (dataset holding results_*.zip),
#      PRIOR_KERNELS="slug1 slug2" (extra kernel outputs to mount, under $KUSER), FORCE_DATA=1 (re-version the big data dataset even if it exists),
#      KAGGLE_USER, TIMEOUT_S (overrides the per-preset kernel timeout), SKIP_SMOKE_GATE=1, SKIP_SOURCE_GATE=1
# Exit codes: 0 = launched and the kernel is QUEUED/RUNNING/COMPLETE; 1 = failed or refused by Kaggle (see the message; do not retry in a loop);
#             2 = no Kaggle credentials (nothing was uploaded); 3 = precondition not met yet (source kernel still running, smoke not COMPLETE): nothing uploaded, try later.
set -u
# NOTE: datasets are uploaded with a bare relative -p from inside $B: the CLI builds its upload-state file name from the path and fails on "a/b/c" (Windows).
SESSION="${1:-1}"
case "$SESSION" in ''|*[!0-9]*) DEFTAG="$SESSION";; *) DEFTAG="s$SESSION";; esac   # numeric -> s<N>; named presets use hyphenated tags (an underscore made title and slug disagree)
TAG="${2:-$DEFTAG}"; ACCEL="${ACCEL:-NvidiaTeslaT4}"
source "$(dirname "$0")/_kaggle_cli.sh" || exit $?
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=-1

# ---- per-preset settings: guard horizon (SESSION_LIMIT_H), kernel timeout, mounted kernel outputs, source gate, smoke gate
LIMIT_H=12; TIMEOUT_H=12; KSOURCES=""; GATE_SRC=""; GATE_SMOKE=""
case "$SESSION" in
  r8f-top)   LIMIT_H=12;  TIMEOUT_H=12;  KSOURCES="hypershift-run-r8f"; GATE_SRC="hypershift-run-r8f"; GATE_SMOKE="hypershift-run-r8f-top-s";;
  r8f-top-s) LIMIT_H=1;   TIMEOUT_H=1.4;  KSOURCES="hypershift-run-r8f-smoke";;
  p1f)       LIMIT_H=6;   TIMEOUT_H=6.5;  GATE_SMOKE="hypershift-run-p1f-s";;
  p1f-s)     LIMIT_H=1;   TIMEOUT_H=1.4;;
  r5f2)      LIMIT_H=8;   TIMEOUT_H=8.5;  GATE_SMOKE="hypershift-run-r5f2-s";;
  r5f2-s)    LIMIT_H=1;   TIMEOUT_H=1.4;;
esac
[ -n "${PRIOR_KERNELS:-}" ] && KSOURCES="$KSOURCES $PRIOR_KERNELS"
TIMEOUT_S="${TIMEOUT_S:-$(awk "BEGIN{printf \"%d\", $TIMEOUT_H*3600}")}"

kstat() {  # $1 slug -> COMPLETE|ERROR|RUNNING|QUEUED|CANCEL...|NOTFOUND (the CLI may be blocked by Smart App Control: retry)
  local i out st
  for i in 1 2 3; do
    out=$(kg kernels status "$KUSER/$1" 2>&1 | tr -d '\r')
    st=$(echo "$out" | grep -o 'KernelWorkerStatus\.[A-Z_]*' | head -1 | sed 's/.*\.//')
    [ -n "$st" ] && { echo "$st"; return; }
    echo "$out" | grep -qi "not found\|404\|does not exist" && { echo NOTFOUND; return; }
    sleep 6
  done
  echo UNKNOWN
}

# ---- gates (nothing is built or uploaded when one fails)
if [ -n "$GATE_SRC" ] && [ "${SKIP_SOURCE_GATE:-0}" != 1 ]; then
  st=$(kstat "$GATE_SRC"); echo "source kernel $GATE_SRC: $st"
  case "$st" in COMPLETE|ERROR) ;; *) echo "NOT READY: $SESSION mounts the output of $GATE_SRC, which is $st (a RUNNING kernel exposes no output, so every run would be repeated). Exit 3 = try later." >&2; exit 3;; esac
fi
if [ -n "$GATE_SMOKE" ] && [ "${SKIP_SMOKE_GATE:-0}" != 1 ]; then
  st=$(kstat "$GATE_SMOKE"); echo "smoke kernel $GATE_SMOKE: $st"
  [ "$st" = COMPLETE ] || { echo "NOT READY: smoke kernel $GATE_SMOKE is $st (must be COMPLETE before the full preset $SESSION; SKIP_SMOKE_GATE=1 overrides). Exit 3 = try later." >&2; exit 3; }
fi
for s in $KSOURCES; do   # a smoke preset mounts a finished smoke kernel
  st=$(kstat "$s"); echo "mounted kernel $s: $st"
  case "$st" in NOTFOUND|UNKNOWN) echo "NOT READY: kernel source $s not found / status unknown ($st)." >&2; exit 3;; esac
done

B=kaggle/build
# Code-only build when the (big) data dataset is already on Kaggle: the full build re-gzips several GB of relation tensors.
DATA_ST=$(kg datasets status "$KUSER/hypershift-rsr-data" 2>&1 | tr -d '\r' | tail -1)
CODE_ONLY=""; if echo "$DATA_ST" | grep -qi "ready" && [ "${FORCE_DATA:-0}" != 1 ]; then CODE_ONLY="--code-only"; fi
.venv/Scripts/python.exe kaggle/build_bundle.py --username "$KUSER" $CODE_ONLY || exit 1

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

SLUG="hypershift-run-${TAG//_/-}"
K=$B/kernel; rm -rf "$K"; mkdir -p "$K"
.venv/Scripts/python.exe kaggle/make_notebook.py "$K/run_kaggle.ipynb" --session "$SESSION" --tag "$TAG" --limit-h "$LIMIT_H" --timeout-h "$TIMEOUT_H" || exit 1
SRC="\"$KUSER/hypershift-code\", \"$KUSER/hypershift-rsr-data\""
[ -n "${PRIOR_DATASET:-}" ] && SRC="$SRC, \"$PRIOR_DATASET\""
KS=""; for s in $KSOURCES; do KS="$KS${KS:+, }\"$KUSER/$s\""; done
cat > "$K/kernel-metadata.json" <<META
{"id": "$KUSER/$SLUG", "title": "$SLUG", "code_file": "run_kaggle.ipynb", "language": "python", "kernel_type": "notebook",
 "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "enable_tpu": "false",
 "dataset_sources": [$SRC], "competition_sources": [], "kernel_sources": [$KS], "model_sources": []}
META
echo "preset $SESSION tag $TAG: guard ${LIMIT_H} h, kernel timeout ${TIMEOUT_S} s, mounted kernels: ${KSOURCES:-none}"
if ! kg kernels push -p "$K" --accelerator "$ACCEL" -t "$TIMEOUT_S"; then
  echo "kernels push REFUSED/FAILED (see the message above; a concurrency limit on GPU sessions is the usual reason). Not retrying." >&2
  exit 1
fi
echo "kernel: https://www.kaggle.com/code/$KUSER/$SLUG"
sleep 20; st=$(kstat "$SLUG"); echo "kernel status: $st"
case "$st" in ERROR|CANCEL*|NOTFOUND|UNKNOWN) echo "kernel is $st right after the push (the CLI can print \"Kernel push error\" and still exit 0, e.g. \"Maximum batch GPU session count of 2 reached\")" >&2; exit 1;; esac
echo "Follow with: bash kaggle/fetch.sh $TAG   (status; downloads the zip and merges when complete)"
