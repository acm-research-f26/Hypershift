#!/usr/bin/env bash
# Phase 1 GPU queue (single owner of the 4 GB GPU). Resumable: runs with metrics.json are skipped.
# Launch detached (survives the Claude session):
#   powershell -c "Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CommandLine='\"C:\Program Files\Git\bin\bash.exe\" -lc \"bash scripts/queues/phase1_gpu.sh\"'; CurrentDirectory='<repo>'}"
cd "/c/Users/Hi/Projects - Coding/Hypershift"; export PATH="/usr/bin:/mingw64/bin:$PATH"
PY=.venv/Scripts/python.exe; L=results/logs; mkdir -p $L
log() { echo "[$(date)] $*" >> $L/driver.log; }
# Windows Application Control intermittently blocks venv DLLs at import; retry each step.
retry() { local out=$1; shift; for i in 1 2 3; do "$@" >> "$out" 2>&1 && return 0; log "retry $i: $*"; sleep 30; done; log "FAILED: $*"; return 1; }

log "phase1 queue start"

# 1. Exact eq.14 attention reruns (THINK + hyp-pairwise), faithful and relative inputs
retry $L/poc_eq14.log     $PY scripts/poc_sectors.py run --variant eq14 --arms HH_hyper HH_clique --seeds 0-9 &
retry $L/poc_rel_eq14.log $PY scripts/poc_sectors.py run --variant rel_eq14 --input-mode relative --arms HH_hyper HH_clique --seeds 0-9 &
wait
$PY scripts/poc_sectors.py summarize --variant eq14 > $L/poc_eq14_summary.log 2>&1
$PY scripts/poc_sectors.py summarize --variant rel_eq14 > $L/poc_rel_eq14_summary.log 2>&1
log "eq14 reruns done"

# 2. Equal-budget tuning (HH + EE grids, val-only selection), then tuned 10-seed rerun.
#    Fresh variant: POC_sectors_rel_tuned/ was started with the pre-eq.14 attention and is superseded.
retry $L/poc_tune_eq14.log $PY scripts/poc_sectors.py tune --variant rel_tuned_eq14 --input-mode relative
log "poc tune (eq14) done: $(tr -d '\n ' < results/POC_sectors_rel_tuned_eq14/tuned.json 2>/dev/null)"
retry $L/poc_rel_tuned_eq14.log $PY scripts/poc_sectors.py run --variant rel_tuned_eq14 --input-mode relative --use-tuned --seeds 0-9
$PY scripts/poc_sectors.py summarize --variant rel_tuned_eq14 > $L/poc_rel_tuned_eq14_summary.log 2>&1
log "poc rel_tuned_eq14 done"

log "phase1 queue done"
