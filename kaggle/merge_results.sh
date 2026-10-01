#!/usr/bin/env bash
# Merge a Kaggle results zip into local results/ (no overwrite, complete runs only). Usage: bash kaggle/merge_results.sh results_s1.zip [--dry-run]
Z="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift   # absolute path before we cd
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=-1
exec .venv/Scripts/python.exe kaggle/merge_results.py "$(cygpath -m "$Z" 2>/dev/null || echo "$Z")" "$@"
