#!/usr/bin/env bash
# Phase 1 GPU queue Q1 (g2): 309-stock small scale on the CORRECTED hypergraph (cache v2, de20f8e: first-order wiki
# channels are stars, second-order are pairs; App. B of the paper). Every earlier stock result used the old graph.
# Every step writes to a NEW exp folder tagged g2 (never reuses one). Resumable: finished runs (metrics.json) are skipped.
# First queue of the chain: Q1 (this) -> R7 clf (phase1_g2_clf.sh) -> Q2 full NYSE (phase1_gpu_3.sh) + R8 (phase1_gpu_r8.sh).
# Two POC processes run in parallel where independent (the 4 GB GPU handled two before). Launch: bash scripts/queues/launch.sh phase1_g2_small
cd "/c/Users/Hi/Projects - Coding/Hypershift"; export PATH="/usr/bin:/mingw64/bin:$PATH"
PY=.venv/Scripts/python.exe; L=results/logs; mkdir -p $L
log() { echo "[$(date)] $*" >> $L/driver.log; }
# Windows Application Control intermittently blocks venv DLLs at import; retry each step.
retry() { local out=$1; shift; for i in 1 2 3; do "$@" >> "$out" 2>&1 && return 0; log "retry $i: $*"; sleep 30; done; log "FAILED: $*"; return 1; }
POC="$PY scripts/poc_sectors.py"
ALL="HH_hyper HH_clique HH_none EE_hyper EE_clique EE_none EH_hyper EH_clique"   # EH_none == EE_none, not an arm

log "g2 small start"

# a. Arms HH/EE/EH x hyper/clique/none, 10 seeds, 30 epochs: level inputs (variant g2) and relative inputs (rel_g2), in parallel.
retry $L/g2.log     $POC run --variant g2 --arms $ALL --seeds 0-9 &
retry $L/rel_g2.log $POC run --variant rel_g2 --input-mode relative --arms $ALL --seeds 0-9 &
wait
$POC summarize --variant g2     > $L/g2_summary.log 2>&1
$POC summarize --variant rel_g2 > $L/rel_g2_summary.log 2>&1
log "g2 small a done (g2, rel_g2 arms)"

# b. Equal-budget tuning, relative inputs, HH/EE/EH, hyper structure, 3 seeds, select on validation only.
#    The previous tuning picked alpha=10 (top of the old grid 0.1/1/10) for both geometries, so alpha is widened to
#    {1, 10, 30, 100}; lr stays {5e-4, 1e-3, 3e-3}. Same grid for every geometry (never tune one arm only).
#    Two workers split the 108 runs (36 per geometry; EH split by seed). Each `tune` also calls tune-select for its own
#    geoms (possible partial-seed EH write to tuned.json); the final tune-select below re-derives all three from disk.
GRID="--grid-lr 0.0005 0.001 0.003 --grid-alpha 1 10 30 100"
TV="--variant rel_tuned_g2 --input-mode relative"
( retry $L/tune_g2_w1.log $POC tune $TV $GRID --geoms HH --seeds 0-2
  retry $L/tune_g2_w1b.log $POC tune $TV $GRID --geoms EH --seeds 0-1 ) &
( retry $L/tune_g2_w2.log $POC tune $TV $GRID --geoms EE --seeds 0-2
  retry $L/tune_g2_w2b.log $POC tune $TV $GRID --geoms EH --seeds 2 ) &
wait
retry $L/tune_g2_select.log $POC tune-select $TV $GRID --geoms HH EE EH
log "g2 tune done: $(tr -d '\n ' < results/POC_sectors_rel_tuned_g2/tuned.json 2>/dev/null)"
# tuned 10-seed rerun for ALL arms (lr/alpha of each geometry applied to its hyper, clique and none arms), two seed halves.
retry $L/rel_tuned_g2_a.log $POC run $TV --use-tuned --arms $ALL --seeds 0-4 &
retry $L/rel_tuned_g2_b.log $POC run $TV --use-tuned --arms $ALL --seeds 5-9 &
wait
$POC summarize --variant rel_tuned_g2 > $L/rel_tuned_g2_summary.log 2>&1
log "g2 small b done (rel_tuned_g2)"

# c. Graph-dependent controls and ablations (two lanes in parallel).
# c1. A10 no-distance (HH_hyper, 10 seeds, level and relative) and C1 shuffled labels (HH_hyper + EH_hyper, 5 seeds, level).
# c2. G2 decomposition (Fig 3a) and G12 hub removal (Fig 3b), HH_hyper AND EH_hyper (= the paper's Fig 3 "Euclidean THINK":
#     Euclidean temporal conv + hypergraph attention, p853), level inputs, 5 seeds.
#
# Levels re-derived from the NEW 309-stock graph (measured on CPU): 558 hyperedges = 545 pairs (second-order wiki) + 13 larger
# edges (industry, sizes 3,13,15,17,18,19,20,21,30,32,34,43,47); 246 nodes have degree 1, max degree 35; full clique = 5067 pairs.
# The old levels no longer fit (they assumed 73 star-like edges).
#  - large_first S (split |e| > S into pairs): S=30 -> 3488 edges (9 big edges left), S=15 -> 4719 edges (3 left).
#    S <= 10 gives 4896 edges (1 left) ~ the full clique already covered by *_clique, so no lower level.
#    Pair-heavy => clique-like cost => micro_batch_days=1 (identical gradients, less memory).
#  - small_first S (split |e| <= S): S=5 would split nothing here (559 edges vs 558, the only small non-pair edge is size 3),
#    so the "increasing order" arm uses S=20 -> 1326 edges (6 big edges left; splits the seven size 13-20 industry edges).
#  - hub removal (drop every hyperedge touching a node of degree >= t, applied on the base graph): 35 -> 426 edges/263 nodes,
#    24 -> 194/221, 16 -> 40 edges (1 big)/34 nodes, 10 -> 16/24. t=8 leaves 3 edges (~none) and t<=4 removes everything
#    (= no relations, covered by *_none), so the ladder stops at 10.
FIG3="HH_hyper EH_hyper"
lane1() {
  retry $L/A10_g2.log     $POC run --variant A10_g2 --arms HH_hyper --seeds 0-9 --set attn_dist=off
  retry $L/rel_A10_g2.log $POC run --variant rel_A10_g2 --input-mode relative --arms HH_hyper --seeds 0-9 --set attn_dist=off
  log "A10_g2 done"
  retry $L/C1_g2.log $POC run --variant C1_g2 --arms $FIG3 --seeds 0-4 --set shuffle_train_labels=true
  log "C1_g2 done"
  for t in 35 24 16 10; do
    retry $L/G12_hub${t}_g2.log $POC run --variant G12_hub${t}_g2 --arms $FIG3 --seeds 0-4 --set drop_hub_degree=$t
  done
  log "G12_g2 done"
}
lane2() {
  retry $L/G2_large30_g2.log $POC run --variant G2_decomp_large30_g2 --arms $FIG3 --seeds 0-4 --set decompose_mode=large_first decompose_size=30 micro_batch_days=1
  retry $L/G2_large15_g2.log $POC run --variant G2_decomp_large15_g2 --arms $FIG3 --seeds 0-4 --set decompose_mode=large_first decompose_size=15 micro_batch_days=1
  retry $L/G2_small20_g2.log $POC run --variant G2_decomp_small20_g2 --arms $FIG3 --seeds 0-4 --set decompose_mode=small_first decompose_size=20
  log "G2_g2 done"
}
lane1 &
lane2 &
wait
log "g2 small c done"

log "g2 small done"
