#!/usr/bin/env bash
# v3 broad screen worker 2 (CPU). Shares units with other screen_v3 workers via claims.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=-1
.venv/Scripts/python.exe scripts/screen_queue.py --worker w2 --output results/SCREEN_SP500_20261019_v3 --years 2022 2023   --arms static hyp two two_residual rolling master pretrain long frozen ridge momentum reversal random          none mlp soft residual_soft random_graph sector_graph market_target cluster_target sector_target          pairwise listwise hybrid multiscale mix gated lowrank summary adaptive   -- --panel data/fresh/sp500_daily/screen_sp500_2015_2026.npz --threads 2 --train-stride 1 --validation-stride 1      --select rankic --frozen-reference 2022 --protocol broad-screen-v3-sp500
