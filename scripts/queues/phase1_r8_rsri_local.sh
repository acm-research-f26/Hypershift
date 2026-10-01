#!/usr/bin/env bash
# Phase 1 R8 RSR-I local rerun. RSR-I failed on Kaggle (IsADirectoryError: the relation .npy.gz became a directory), so the
# missing RSR-I runs are done on the local GPU: small scale POC_sectors_R8_rsr_i_g2/rsr_i seeds 1-9 (seed 0 exists) and full NYSE
# R8_baselines_g2/RSR_I seeds 0-9. STHGCN runs on Kaggle, not here. Same commands/overrides as phase1_gpu_r8.sh.
# Waits for "think tail done" (so it never competes with the THINK tail for VRAM), then runs TWO workers (RSR-I ~0.3-0.5 GB each).
# Resumable (runs with metrics.json are skipped); retries a failed run up to 4x. Launch: bash scripts/queues/launch.sh phase1_r8_rsri_local
cd "/c/Users/Hi/Projects - Coding/Hypershift"; export PATH="/usr/bin:/mingw64/bin:$PATH"
PY=.venv/Scripts/python.exe; L=results/logs; mkdir -p $L
log() { echo "[$(date)] $*" >> $L/driver.log; }
# $1 = run dir (metrics.json => skip; failed.json cleanup), $2 = log file, rest = command
retry() {
  local rd=$1 out=$2; shift 2
  [ -f "$rd/metrics.json" ] && return 0
  for i in 1 2 3 4; do
    "$@" >> "$out" 2>&1 && [ -f "$rd/metrics.json" ] && return 0
    rm -f "$rd"/failed.json
    log "retry $i: $*"; sleep 60
  done
  log "FAILED: $*"; return 1
}
COMMON="--config configs/think_nyse.yaml --set exp=R8_baselines_g2 norm=paper epochs=100 patience=1000 batch_days=8"
small() { retry results/POC_sectors_R8_rsr_i_g2/rsr_i/seed_$1 $L/R8_g2_poc_rsr_i_local.log \
            $PY scripts/poc_sectors.py run --variant R8_rsr_i_g2 --seeds $1 --set model=rsr_i; }
full()  { retry results/R8_baselines_g2/RSR_I/seed_$1 $L/R8_g2_RSR_I_local.log \
            $PY -m hypershift.run $COMMON label=RSR_I model=rsr_i micro_batch_days=2 --seeds $1; log "R8 RSR_I local seed $1 done"; }

until grep -q "think tail done" $L/driver.log 2>/dev/null; do sleep 60; done
log "r8 rsri local start"

# worker A: small seeds 1-5, then full seeds 0 2 4 6 8.  worker B: small seeds 6-9, then full seeds 1 3 5 7 9.
( for s in 1 2 3 4 5; do small $s; done; for s in 0 2 4 6 8; do full $s; done; log "r8 rsri local worker A done" ) &
( for s in 6 7 8 9; do small $s; done; for s in 1 3 5 7 9; do full $s; done; log "r8 rsri local worker B done" ) &
wait
log "r8 rsri local done"
