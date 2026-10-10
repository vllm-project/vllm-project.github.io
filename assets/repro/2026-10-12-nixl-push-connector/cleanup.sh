#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tear down the push-vs-pull benchmark deployment.
#   ./cleanup.sh proxy    # on the proxy / benchmark driver node
#   ./cleanup.sh server   # on each GPU node (prefill and decode)
#
# The benchmark script's EXIT trap already kills the proxy it started; this is
# for proxies launched by hand or left behind by an interrupted run.

set -uo pipefail

TARGET="${1:-all}"

cleanup_proxy() {
  echo "== proxy =="
  pkill -f disagg_proxy_demo.py 2>/dev/null || true
  pkill -f disagg_proxy_pushconnector_demo.py 2>/dev/null || true
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
  # are actually idle before relaunching in the other mode.
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used --format=csv
}

case "$TARGET" in
  proxy) cleanup_proxy ;;
  server) cleanup_server ;;
  all) cleanup_proxy; cleanup_server ;;
  *) echo "usage: $0 [proxy|server|all]" >&2; exit 1 ;;
esac

echo
echo "Remember to terminate the EC2 instances themselves once results are copied off."
