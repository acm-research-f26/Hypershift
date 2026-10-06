@echo off
cd /d "C:\Users\Hi\Projects - Coding\Hypershift"
"C:\Program Files\Git\bin\bash.exe" scripts/queues/phase1_gpu_r8.sh >> results\logs\task_phase1_gpu_r8.log 2>&1
