#!/usr/bin/env bash
# Phase 1 GPU tail: rerun the last missing THINK_paperProtocol seed of R5_g2 (seed 24 had a partial folder, no metrics.json).
# EH/EE/HE run on Kaggle (s1/s2), so this does NOT run them. Resumable. Launch: bash scripts/queues/launch.sh phase1_think_tail
cd "/c/Users/Hi/Projects - Coding/Hypershift"; export PATH="/usr/bin:/mingw64/bin:$PATH"
PY=.venv/Scripts/python.exe; L=results/logs; mkdir -p $L
log() { echo "[$(date)] $*" >> $L/driver.log; }
retry() { local out=$1; shift; for i in 1 2 3; do "$@" >> "$out" 2>&1 && return 0; log "retry $i: $*"; sleep 30; done; log "FAILED: $*"; return 1; }

log "think tail start (seed 24)"
retry $L/R5_g2_THINK_tail.log $PY scripts/run_grid.py E1_main --labels THINK_paperProtocol --seeds 24 --set exp=R5_g2
log "think tail done"
