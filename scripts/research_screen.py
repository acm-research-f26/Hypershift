"""Run the independent, predeclared broad research screen."""
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hypershift.screening.runner import main

if __name__ == '__main__':
    main()
