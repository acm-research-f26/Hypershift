#!/usr/bin/env bash
# Phase 1 GPU queue 2: starts automatically once queue 1 (phase1_gpu.sh) has logged "phase1 queue done".
# Every step writes to its OWN new exp folder (never reuses one). Resumable: finished runs are skipped.
# Launch detached exactly like phase1_gpu.sh (bash scripts/queues/phase1_gpu_2.sh).
cd "/c/Users/Hi/Projects - Coding/Hypershift"; export PATH="/usr/bin:/mingw64/bin:$PATH"
PY=.venv/Scripts/python.exe; L=results/logs; mkdir -p $L
log() { echo "[$(date)] $*" >> $L/driver.log; }
retry() { local out=$1; shift; for i in 1 2 3; do "$@" >> "$out" 2>&1 && return 0; log "retry $i: $*"; sleep 30; done; log "FAILED: $*"; return 1; }

# Gate: GPU is owned by queue 1 until it finishes.
until grep -q "phase1 queue done" $L/driver.log 2>/dev/null; do sleep 120; done
log "phase1 queue 2 start"

POC="$PY scripts/poc_sectors.py"

# a. C1 control: shuffled training labels. Leak-free expectation: test Sharpe ~ random-5 (0.34), IC ~ 0.
retry $L/C1_shuffled.log $POC run --variant C1_shuffled --arms HH_hyper EE_hyper --seeds 0-4 --set shuffle_train_labels=true
log "C1_shuffled done"

# b. A10: eq.14 attention without the distance term (compare POC_sectors_eq14 / POC_sectors_rel_eq14 HH_hyper).
retry $L/A10_nodist.log     $POC run --variant A10_nodist --arms HH_hyper --seeds 0-9 --set attn_dist=off
retry $L/rel_A10_nodist.log $POC run --variant rel_A10_nodist --input-mode relative --arms HH_hyper --seeds 0-9 --set attn_dist=off
log "A10 done"

# c. G2 hyperedge decomposition (paper Fig 3a) and G12 hub removal (Fig 3b), HH_hyper, level inputs, 5 seeds.
# Universe facts (309 stocks, 73 hyperedges, n/a bucket excluded; measured on CPU): edge sizes median 5, p90 ~20,
# max 47 (47/43/34/32 are the only edges > 30; 13 edges > 15; 30 > 5). Node degree: 246 nodes deg 1, 63 nodes deg >= 2, max 27.
# Paper NYSE axis 500,15,9,5,3 assumes a 500-stock edge; that does not exist here, so levels are re-scaled to this universe.
#  - large_first (split |e| > size into pairs): 30 splits only the 4 biggest edges (~3.1k pairs), 15 splits 13 edges (~4.7k pairs).
#    Sizes <= 10 add almost nothing (~5k pairs, ~= full clique, already covered by HH_clique), so no lower level.
#    Pair edges make these arms slow (~clique-like cost), so micro_batch_days=1 (identical gradients, less memory).
#  - small_first (split |e| <= size): 5 splits the 43 edges of size <= 5 into 132 total edges; the cheap "increasing order" arm.
#  - hub removal (drop every hyperedge touching a node of degree >= t): 12 -> 21 edges/259 nodes kept (mild), 8 -> 7 edges/180 nodes,
#    5 -> 4 edges/95 nodes (t = 3..5 give the identical graph, so 5 is the plateau; t=2 removes everything = HH_none).
retry $L/G2_large30.log $POC run --variant G2_decomp_large30 --arms HH_hyper --seeds 0-4 --set decompose_mode=large_first decompose_size=30 micro_batch_days=1
retry $L/G2_large15.log $POC run --variant G2_decomp_large15 --arms HH_hyper --seeds 0-4 --set decompose_mode=large_first decompose_size=15 micro_batch_days=1
retry $L/G2_small5.log  $POC run --variant G2_decomp_small5  --arms HH_hyper --seeds 0-4 --set decompose_mode=small_first decompose_size=5
log "G2 done"
for t in 12 8 5; do
  retry $L/G12_hub$t.log $POC run --variant G12_hub$t --arms HH_hyper --seeds 0-4 --set drop_hub_degree=$t
done
log "G12 done"

# d. R7 NASDAQ 3-class (resumable per seed under results/E11_clf; reports macro- and micro-F1).
retry $L/R7_clf.log $PY scripts/run_clf.py
log "R7 clf done"

log "phase1 queue 2 done"
