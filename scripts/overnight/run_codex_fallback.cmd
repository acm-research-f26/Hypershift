@echo off
cd /d "C:\Users\Hi\Projects - Coding\Hypershift"
"C:\Program Files\Git\bin\bash.exe" scripts/overnight/codex_fallback.sh >> results\logs\task_codex_fallback.log 2>&1
