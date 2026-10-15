#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Launch one leg of a 1P1D disaggregated deployment for the push-vs-pull
# KV transfer benchmark (vLLM blog, 2026-09-21).
#
# Run once on the prefill node and once on the decode node. The only
# difference between the two modes is the connector name:
#   pull -> NixlConnector       (decode READs KV from prefill)
#   push -> NixlPushConnector   (prefill WRITEs KV into decode)
#
# Usage:
#   PREFILL_IP=10.0.0.1 DECODE_IP=10.0.0.2 ROLE=prefill MODE=pull ./serve_pd.sh
#   PREFILL_IP=10.0.0.1 DECODE_IP=10.0.0.2 ROLE=decode  MODE=pull ./serve_pd.sh
#
# Then relaunch both with MODE=push, changing nothing else.

set -euo pipefail

ROLE="${ROLE:?set ROLE=prefill or ROLE=decode}"
MODE="${MODE:-pull}"
PREFILL_IP="${PREFILL_IP:?set PREFILL_IP}"
DECODE_IP="${DECODE_IP:?set DECODE_IP}"

MODEL="${MODEL:-Qwen/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
SIDE_CHANNEL_PORT="${SIDE_CHANNEL_PORT:-5600}"

case "$MODE" in
  pull) CONNECTOR="NixlConnector" ;;
  push) CONNECTOR="NixlPushConnector" ;;
  *) echo "MODE must be pull or push" >&2; exit 1 ;;
esac

# EFA / NIXL wiring. The side-channel host must be the node's own address:
# the default (localhost) is unreachable from the peer.
export FI_PROVIDER=efa
export VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_CHANNEL_PORT"

common_args=(
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype bfloat16
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
)

if [ "$ROLE" = "prefill" ]; then
  export VLLM_NIXL_SIDE_CHANNEL_HOST="$PREFILL_IP"
  PORT=8100
  KV_CONFIG=$(cat <<EOF
{"kv_connector":"$CONNECTOR","kv_buffer_device":"cuda","kv_role":"kv_producer",
 "engine_id":"prefill-engine-001","kv_rank":0,"kv_parallel_size":2,
 "kv_connector_extra_config":{"backends":["LIBFABRIC"]}}
EOF
)
  extra_args=()
else
  export VLLM_NIXL_SIDE_CHANNEL_HOST="$DECODE_IP"
  PORT=8200
  KV_CONFIG=$(cat <<EOF
{"kv_connector":"$CONNECTOR","kv_buffer_device":"cuda","kv_role":"kv_consumer",
 "engine_id":"decode-engine-001","kv_rank":1,"kv_parallel_size":2,
 "kv_connector_extra_config":{"backends":["LIBFABRIC"]}}
EOF
)
  # Every request must fetch full KV from prefill, which is what we measure.
  # With prefix caching on, overlapping prompts can hit the decode node's own
  # cache and skip the P->D transfer entirely, understating both modes.
  extra_args=(--no-enable-prefix-caching)
fi

echo "Starting $ROLE ($MODE / $CONNECTOR) on :$PORT"
exec vllm serve "$MODEL" \
  "${common_args[@]}" \
  "${extra_args[@]}" \
  --port "$PORT" \
  --kv-transfer-config "$KV_CONFIG"
