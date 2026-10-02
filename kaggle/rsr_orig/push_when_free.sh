#!/usr/bin/env bash
# retry the GPU push every 5 min until a GPU session slot is free (Kaggle allows 2 concurrent GPU sessions)
for i in $(seq 1 80); do
  out=$(bash "$(dirname "$0")/launch_rsr_orig.sh" "${1:-full}" 2>&1 | tail -3)
  echo "$(date +%T) $out"
  echo "$out" | grep -q "successfully pushed" && exit 0
  sleep 300
done
exit 1
