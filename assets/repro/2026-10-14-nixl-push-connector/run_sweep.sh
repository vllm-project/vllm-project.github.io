#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Drive the push-vs-pull KV transfer sweep published in the vLLM blog
# (2026-09-21). Runs on the proxy node, against a 1P1D pair that is already
# serving. Wraps disagg_kv_push_pull_benchmark.sh from vLLM PR #57567, which
# starts the right proxy for the mode and drives it with `vllm bench serve`.
#
# Usage:
#   VLLM_SRC=~/vllm PREFILL_IP=10.0.0.1 DECODE_IP=10.0.0.2 MODE=pull ./run_sweep.sh
#   VLLM_SRC=~/vllm PREFILL_IP=10.0.0.1 DECODE_IP=10.0.0.2 MODE=push ./run_sweep.sh
#
# Keep the sweep identical across the two modes; only MODE changes.

set -euo pipefail

VLLM_SRC="${VLLM_SRC:?point VLLM_SRC at a vLLM clone with PR #57567 checked out}"
PREFILL_IP="${PREFILL_IP:?set PREFILL_IP}"
DECODE_IP="${DECODE_IP:?set DECODE_IP}"
MODE="${MODE:-pull}"

MODEL="${MODEL:-Qwen/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen}"

# The published figures sweep 1k-16k input at 1/2/4/8 QPS with 128 output
# tokens. NUM_REQS/ITERATIONS are kept high enough that per-point means are
# stable at 8 QPS.
QPS_LIST="${QPS_LIST:-1 2 4 8}"
INPUT_LENS="${INPUT_LENS:-1024 2048 4096 8192 16384}"
OUTPUT_LENS="${OUTPUT_LENS:-128}"
NUM_REQS="${NUM_REQS:-100}"
ITERATIONS="${ITERATIONS:-10}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

BENCH="$VLLM_SRC/examples/disaggregated/disaggregated_serving/disagg_kv_push_pull_benchmark.sh"
[ -x "$BENCH" ] || [ -f "$BENCH" ] || {
  echo "benchmark script not found: $BENCH" >&2
  echo "git fetch origin pull/57567/head:pd-bench && git checkout pd-bench" >&2
  exit 1
}

# Push mode needs the prefill instance's NIXL coordinates so the proxy can tell
# the decode node where to register its blocks. These must match the values the
# prefill server was launched with.
push_env=()
if [ "$MODE" = "push" ]; then
  push_env=(
    PREFILL_ENGINE_ID="${PREFILL_ENGINE_ID:-prefill-engine-001}"
    PREFILL_KV_HOST="${PREFILL_KV_HOST:-$PREFILL_IP}"
    PREFILL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNEL_PORT:-5600}"
    PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
    PREFILL_PP_SIZE="${PREFILL_PP_SIZE:-1}"
  )
fi

env \
  VLLM_SRC="$VLLM_SRC" \
  MODEL="$MODEL" \
  SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  PREFILL_URL="http://$PREFILL_IP:8100" \
  DECODE_URL="http://$DECODE_IP:8200" \
  MODES="$MODE" \
  QPS_LIST="$QPS_LIST" \
  INPUT_LENS="$INPUT_LENS" \
  OUTPUT_LENS="$OUTPUT_LENS" \
  NUM_REQS="$NUM_REQS" \
  ITERATIONS="$ITERATIONS" \
  RESULTS_DIR="$RESULTS_DIR" \
  "${push_env[@]}" \
  bash "$BENCH"

echo
echo "Results in $RESULTS_DIR as <mode>_qps<q>_in<in>_out<out>_iter<i>.json"
