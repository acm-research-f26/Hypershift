#!/usr/bin/env bash
# Phase 1 GPU queue 3: R5/A5/A9 full-NYSE paper-protocol rerun with exact eq.14 attention.
# Starts once queue 2 logs "phase1 queue 2 done". Writes ONLY to the new exp results/R5_eq14 (old
# E1_main/THINK_paperProtocol and R_paperProtocol used pre-eq.14 attn_score=mobius and are superseded).
# Two parallel workers on disjoint seeds (even / odd), each running the steps in sequence. Resumable.
# Launch detached: Start-Process bash.exe -lc "bash scripts/queues/phase1_gpu_3.sh" (PowerShell, not WMI).
cd "/c/Users/Hi/Projects - Coding/Hypershift"; export PATH="/usr/bin:/mingw64/bin:$PATH"
PY=.venv/Scripts/python.exe; L=results/logs; mkdir -p $L
log() { echo "[$(date)] $*" >> $L/driver.log; }
retry() { local out=$1; shift; for i in 1 2 3; do "$@" >> "$out" 2>&1 && return 0; log "retry $i: $*"; sleep 30; done; log "FAILED: $*"; return 1; }

until grep -q "phase1 queue 2 done" $L/driver.log 2>/dev/null; do sleep 120; done
log "phase1 queue 3 start"

P="--set exp=R5_eq14 norm=paper epochs=100 patience=1000"
worker() {  # $1 = tag, $2 = THINK/EH seeds, $3 = EE/HE seeds
  retry $L/R5_THINK_$1.log $PY scripts/run_grid.py E1_main --labels THINK_paperProtocol --seeds $2 --set exp=R5_eq14
  log "R5 THINK worker $1 done"
  retry $L/R5_EH_$1.log $PY scripts/run_grid.py E2_geometry --labels EH --seeds $2 $P
  log "R5 EH worker $1 done"
  retry $L/R5_EEHE_$1.log $PY scripts/run_grid.py E2_geometry --labels EE HE --seeds $3 $P
  log "R5 EE/HE worker $1 done"
}
worker even 0,2,4,6,8,10,12,14,16,18,20,22,24 0,2,4,6,8 &
worker odd  1,3,5,7,9,11,13,15,17,19,21,23    1,3,5,7,9 &
wait
# aggregate.py not run here: it rewrites shared results/tables; aggregate R5_eq14 in a separate task.
log "phase1 queue 3 done"
