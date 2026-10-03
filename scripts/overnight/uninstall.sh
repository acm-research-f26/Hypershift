#!/usr/bin/env bash
# Remove the Hypershift_overnight scheduled task. State/log files are kept.
schtasks //End //TN Hypershift_overnight > /dev/null 2>&1 || true
schtasks //Delete //F //TN Hypershift_overnight && echo "removed Hypershift_overnight"
