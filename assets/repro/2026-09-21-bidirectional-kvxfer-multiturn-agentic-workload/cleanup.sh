#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tear down the multi-turn bi-directional KV transfer deployment.
#   ./cleanup.sh proxy    # on the proxy / benchmark driver node
#   ./cleanup.sh server   # on each GPU node (prefill and decode)
#
# To return a deployment to the standard prefill->decode path without tearing it
# down, relaunch P and D with bidirectional_kv_xfer set to false (or drop the
# key); it is read once at engine init, so a restart is required.

set -uo pipefail

TARGET="${1:-all}"

cleanup_proxy() {
  echo "== proxy =="
  pkill -f disagg_proxy_multiturn 2>/dev/null || true
  lsof -t -i:8000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  ss -ltn 2>/dev/null | grep -q ':8000' && echo "WARNING: 8000 still bound" || echo "proxy port free"
}

cleanup_server() {
  echo "== vllm server =="
  pkill -f "vllm serve" 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -f "vllm serve" >/dev/null || break
    sleep 1
  done
  ss -ltn 2>/dev/null | grep -E ':(8100|8200|5600)' || echo "ports 8100/8200/5600 free"
  # A killed engine can leave its registered KV buffer mapped; confirm the GPUs
  # are idle before relaunching for the other leg.
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used --format=csv
}

case "$TARGET" in
  proxy) cleanup_proxy ;;
  server) cleanup_server ;;
  all) cleanup_proxy; cleanup_server ;;
  *) echo "usage: $0 [proxy|server|all]" >&2; exit 1 ;;
esac

echo
echo "Copy results off the nodes, then terminate the instances and verify none remain running."
