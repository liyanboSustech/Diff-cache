#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run run_flux.sh multiple times while only changing
# TAYLORSEER_MAX_ORDER. Usage:
#   ./run_max_order_sweep.sh          # runs orders 1..6
#   ./run_max_order_sweep.sh 2 4 6    # runs only specified orders

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$#" -eq 0 ]; then
  ORDERS=(7 8 9)
else
  ORDERS=("$@")
fi

for order in "${ORDERS[@]}"; do
  echo ""
  echo "=== Running run_flux.sh with TAYLORSEER_MAX_ORDER=${order} ==="
  TAYLORSEER_MAX_ORDER="$order" bash run_flux.sh
done
