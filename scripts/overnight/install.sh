#!/usr/bin/env bash
# Register Task Scheduler task Hypershift_overnight: runs scripts/overnight/driver.sh every 30 min, starting now.
# Task Scheduler (not Start-Process) so it survives the Claude session ending; same battery settings as scripts/queues/launch.sh.
# Needs the user logged on (like the queues). Usage: bash scripts/overnight/install.sh
set -e
repo="C:\Users\Hi\Projects - Coding\Hypershift"
cmd="scripts/overnight/run_driver.cmd"
printf '@echo off\r\ncd /d "%s"\r\n"C:\Program Files\Git\bin\bash.exe" scripts/overnight/driver.sh >> results\logs\task_overnight.log 2>&1\r\n' "$repo" > "$cmd"
st=$(date -d '+1 minute' +%H:%M)
schtasks //Create //F //TN Hypershift_overnight //SC MINUTE //MO 30 //ST "$st" //TR "\"$repo\scripts\overnight\run_driver.cmd\"" > /dev/null
# Battery-safe; a hung tick is killed after 25 min and never overlaps the next (IgnoreNew); missed ticks run when the machine is back.
powershell -NoProfile -c '$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 25); Set-ScheduledTask -TaskName Hypershift_overnight -Settings $s' > /dev/null
schtasks //Run //TN Hypershift_overnight
echo "installed Hypershift_overnight (every 30 min, first run now). Log: results/logs/overnight.log. Remove: bash scripts/overnight/uninstall.sh"
