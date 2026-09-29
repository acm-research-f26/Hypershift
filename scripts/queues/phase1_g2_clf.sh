#!/usr/bin/env bash
# Phase 1 GPU queue R7 (g2): NASDAQ 3-class movement (macro and micro F1), 25 seeds x HH, EH, EE, on the corrected hypergraph
# (cache v2). Writes ONLY to the new exp results/E11_clf_g2 (old results/E11_clf, if any, used the old graph).
# Starts once Q1 (phase1_g2_small.sh) logs "g2 small done"; must finish before Q2 (phase1_gpu_3.sh) starts. Resumable per seed.
cd "/c/Users/Hi/Projects - Coding/Hypershift"; export PATH="/usr/bin:/mingw64/bin:$PATH"
PY=.venv/Scripts/python.exe; L=results/logs; mkdir -p $L
log() { echo "[$(date)] $*" >> $L/driver.log; }
retry() { local out=$1; shift; for i in 1 2 3; do "$@" >> "$out" 2>&1 && return 0; log "retry $i: $*"; sleep 30; done; log "FAILED: $*"; return 1; }

until grep -q "g2 small done" $L/driver.log 2>/dev/null; do sleep 120; done
log "g2 clf start"
retry $L/R7_clf_g2.log $PY scripts/run_clf.py --exp E11_clf_g2
log "g2 clf done"
