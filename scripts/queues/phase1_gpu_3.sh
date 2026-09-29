#!/usr/bin/env bash
# Phase 1 GPU queue Q2 (g2): full-NYSE paper-protocol rerun (R5/A5/A9) with exact eq.14 attention on the CORRECTED
# hypergraph (cache v2: 4350 hyperedges on NYSE; first-order wiki channels stars, second-order pairs, paper App. B).
# Starts once R7 (phase1_g2_clf.sh) logs "g2 clf done". Writes ONLY to the new exp results/R5_g2 (R5_eq14, E1_main/THINK_paperProtocol
# and R_paperProtocol used the old graph and are superseded).
# THINK_paperProtocol (HH) and EH: 25 seeds each; EE and HE: 10 seeds each. Two parallel workers on disjoint seeds (even / odd).
# Probe (seed 99, 1 epoch, GPU): 21.0 s/epoch, ~1.6 GB per process (nvidia-smi 1888 MiB total incl. ~300 MiB desktop), so two workers fit in 4 GB.
# Resumable. Launch: bash scripts/queues/launch.sh phase1_gpu_3
cd "/c/Users/Hi/Projects - Coding/Hypershift"; export PATH="/usr/bin:/mingw64/bin:$PATH"
PY=.venv/Scripts/python.exe; L=results/logs; mkdir -p $L
log() { echo "[$(date)] $*" >> $L/driver.log; }
retry() { local out=$1; shift; for i in 1 2 3; do "$@" >> "$out" 2>&1 && return 0; log "retry $i: $*"; sleep 30; done; log "FAILED: $*"; return 1; }

until grep -q "g2 clf done" $L/driver.log 2>/dev/null; do sleep 120; done
log "g2 full start"

P="--set exp=R5_g2 norm=paper epochs=100 patience=1000"
worker() {  # $1 = tag, $2 = THINK/EH seeds, $3 = EE/HE seeds
  retry $L/R5_g2_THINK_$1.log $PY scripts/run_grid.py E1_main --labels THINK_paperProtocol --seeds $2 --set exp=R5_g2
  log "R5_g2 THINK worker $1 done"
  retry $L/R5_g2_EH_$1.log $PY scripts/run_grid.py E2_geometry --labels EH --seeds $2 $P
  log "R5_g2 EH worker $1 done"
  retry $L/R5_g2_EEHE_$1.log $PY scripts/run_grid.py E2_geometry --labels EE HE --seeds $3 $P
  log "R5_g2 EE/HE worker $1 done"
}
worker even 0,2,4,6,8,10,12,14,16,18,20,22,24 0,2,4,6,8 &
worker odd  1,3,5,7,9,11,13,15,17,19,21,23    1,3,5,7,9 &
wait
# aggregate.py not run here: it rewrites shared results/tables; aggregate R5_g2 in a separate task.
log "g2 full done"
