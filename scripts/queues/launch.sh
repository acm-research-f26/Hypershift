#!/usr/bin/env bash
# Launch a queue script via Windows Task Scheduler so it survives the Claude session ending
# (Start-Process children die with the session; WMI launches get blocked by Smart App Control).
# Usage: bash scripts/queues/launch.sh phase1_gpu_2
set -e
q=$1
repo="C:\\Users\\Hi\\Projects - Coding\\Hypershift"
cmd="scripts/queues/run_$q.cmd"
printf '@echo off\r\ncd /d "%s"\r\n"C:\\Program Files\\Git\\bin\\bash.exe" scripts/queues/%s.sh >> results\\logs\\task_%s.log 2>&1\r\n' "$repo" "$q" "$q" > "$cmd"
# SD far in the future: a ONCE task with the default start date would fire AGAIN at 23:59 today (duplicate GPU owner); we start it via /Run.
schtasks //Create //F //TN "Hypershift_$q" //SC ONCE //ST 23:59 //SD 01/01/2030 //TR "\"$repo\\scripts\\queues\\run_$q.cmd\"" > /dev/null
schtasks //Run //TN "Hypershift_$q"
