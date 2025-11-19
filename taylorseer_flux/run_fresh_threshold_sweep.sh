#!/usr/bin/env bash
set -euo pipefail

# Run run_flux.sh multiple times while only changing TAYLORSEER_FRESH_THRESHOLD.
# Usage:
#   ./run_fresh_threshold_sweep.sh            # thresholds 3 4 5 6 7 8
#   ./run_fresh_threshold_sweep.sh 4 6 8 10   # custom thresholds

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$#" -eq 0 ]; then
  THRESHOLDS=(3 4 5 6 7 8)
else
  THRESHOLDS=("$@")
fi

for threshold in "${THRESHOLDS[@]}"; do
  echo ""
  echo "=== Running run_flux.sh with TAYLORSEER_FRESH_THRESHOLD=${threshold} ==="
  TAYLORSEER_FRESH_THRESHOLD="$threshold" bash run_flux.sh
done
