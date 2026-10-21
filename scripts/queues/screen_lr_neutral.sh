#!/usr/bin/env bash
# Check that the --lr flag is behaviour-neutral at its default: rerun v3 2022/static/seed_7 and compare predictions.
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=-1
.venv/Scripts/python.exe scripts/research_screen.py --output results/SCREEN_SP500_20261019_v3_lrcheck --years 2022 --arms static \
  --seeds 7 --panel data/fresh/sp500_daily/screen_sp500_2015_2026.npz --threads 2 --train-stride 1 --validation-stride 1 \
  --select rankic --frozen-reference 2022 --protocol broad-screen-v3-sp500
