#!/usr/bin/env bash
# Training-budget probe worker 2 (CPU, diagnostic; see the deep-test amendment in RESEARCH_SCREENING_PLAN.md).
# Same v3 protocol except patience 20 and lr 3e-3; laptop/torch 2.14 like the v3 static and two_residual parents.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=-1
.venv/Scripts/python.exe scripts/screen_queue.py --worker b2 --output results/SCREEN_SP500_20261019_v3_budget --years 2022 2023   --arms two_residual static -- --panel data/fresh/sp500_daily/screen_sp500_2015_2026.npz --threads 2 --train-stride 1   --validation-stride 1 --select rankic --frozen-reference 2022 --protocol broad-screen-v3-sp500 --patience 20 --lr 0.003
