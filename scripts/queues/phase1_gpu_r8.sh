#!/usr/bin/env bash
# Phase 1 GPU queue R8 (g2): paper baselines RSR-I and STHGCN (models rsr_i / sthgcn, see docs/phase1/R8_baselines.md)
# on the CORRECTED hypergraph (cache v2). Starts once R7 logs "g2 clf done", but waits for Q2 ("g2 full start") plus 5 min
# so Q2's two workers allocate their VRAM first (probe: ~1.6 GB each). Runs ONE worker next to Q2 on the 4 GB GPU, so before
# every attempt it waits for >= 1500 MiB free VRAM (in practice: once one Q2 worker has finished) and OOM/any failure is
# retried (up to 6x) after a pause.
# Writes ONLY to new folders: results/R8_baselines_g2 (full NYSE), results/POC_sectors_R8_rsr_i_g2, results/POC_sectors_R8_sthgcn_g2.
# Resumable: runs with metrics.json are skipped. Launch detached via PowerShell Start-Process (not WMI).
cd "/c/Users/Hi/Projects - Coding/Hypershift"; export PATH="/usr/bin:/mingw64/bin:$PATH"
PY=.venv/Scripts/python.exe; L=results/logs; mkdir -p $L
log() { echo "[$(date)] $*" >> $L/driver.log; }
MINFREE=1500
gpu_wait() {  # block until nvidia-smi reports >= MINFREE MiB free (poll 60 s); proceed if nvidia-smi is unavailable
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
    [ -z "$free" ] && return 0
    [ "$free" -ge $MINFREE ] && return 0
    sleep 60
  done
}
# $1 = run dir (for failed.json cleanup), $2 = log file, rest = command. Waits for VRAM before each attempt.
retry() {
  local rd=$1 out=$2; shift 2
  for i in 1 2 3 4 5 6; do
    gpu_wait
    "$@" >> "$out" 2>&1 && return 0
    rm -f "$rd"/failed.json
    grep -qi "out of memory" "$out" && log "OOM, will retry $i: $*" || log "retry $i: $*"
    sleep 60
  done
  log "FAILED: $*"; return 1
}

until grep -q "g2 clf done" $L/driver.log 2>/dev/null; do sleep 120; done
until grep -q "g2 full start" $L/driver.log 2>/dev/null; do sleep 60; done
sleep 300
log "g2 r8 start"

# a. Small scale (309 stocks, POC universe, 30 epochs, patience 10, level inputs like POC_sectors_eq14), 10 seeds.
for m in rsr_i sthgcn; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    retry results/POC_sectors_R8_${m}_g2/${m}/seed_$s $L/R8_g2_poc_$m.log \
      $PY scripts/poc_sectors.py run --variant R8_${m}_g2 --seeds $s --set model=$m
  done
  log "R8 small $m done"
done

# b. Full NYSE, paper protocol (norm=paper, 100 epochs, patience 1000), batch_days=8 as R5; 10 seeds each, interleaved.
#    micro_batch_days bounds VRAM (gradient accumulation; BatchNorm in STHGCN sees micro-batches).
COMMON="--config configs/think_nyse.yaml --set exp=R8_baselines_g2 norm=paper epochs=100 patience=1000 batch_days=8"
for s in 0 1 2 3 4 5 6 7 8 9; do
  retry results/R8_baselines_g2/RSR_I/seed_$s $L/R8_g2_RSR_I.log \
    $PY -m hypershift.run $COMMON label=RSR_I model=rsr_i micro_batch_days=2 --seeds $s
  log "R8 RSR_I seed $s done"
  retry results/R8_baselines_g2/STHGCN/seed_$s $L/R8_g2_STHGCN.log \
    $PY -m hypershift.run $COMMON label=STHGCN model=sthgcn micro_batch_days=4 --seeds $s
  log "R8 STHGCN seed $s done"
done

log "g2 r8 done"
