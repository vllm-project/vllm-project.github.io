#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Launch one leg of a 1P1D deployment for the bi-directional KV transfer
# multi-turn benchmark (vLLM blog, 2026-09-21).
#
# Run once on the prefill node and once on the decode node. The A/B is driven
# entirely by BIDIR; nothing else changes between the two legs.
#   BIDIR=false -> baseline: P recomputes the whole grown context every turn.
#   BIDIR=true  -> D retains its blocks and P reads them back (D->P transfer).
#
# Usage:
#   P_IP=10.0.0.1 D_IP=10.0.0.2 ROLE=prefill BIDIR=false ./serve_pd_multiturn.sh
#   P_IP=10.0.0.1 D_IP=10.0.0.2 ROLE=decode  BIDIR=false ./serve_pd_multiturn.sh
#
# For the GLM5.2 runs, set MODEL and SERVED_NAME:
#   MODEL=zai-org/GLM-5-FP8 SERVED_NAME=GLM ... ./serve_pd_multiturn.sh

set -euo pipefail

ROLE="${ROLE:?set ROLE=prefill or ROLE=decode}"
BIDIR="${BIDIR:-false}"          # literal JSON true/false, must match on P and D
P_IP="${P_IP:?set P_IP}"
D_IP="${D_IP:?set D_IP}"

MODEL="${MODEL:-Qwen/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-Qwen}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
SIDE_CHANNEL_PORT="${SIDE_CHANNEL_PORT:-5600}"

# Tunables documented in the post. Defaults match vLLM's own defaults.
KV_RECOMPUTE_THRESHOLD="${KV_RECOMPUTE_THRESHOLD:-64}"   # tokens
DECODER_KV_BLOCKS_TTL="${DECODER_KV_BLOCKS_TTL:-480}"    # seconds

case "$BIDIR" in
  true|false) ;;
  *) echo "BIDIR must be the literal string true or false (JSON), got '$BIDIR'" >&2; exit 1 ;;
esac

export FI_PROVIDER=efa
export VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_CHANNEL_PORT"

extra_config="\"backends\":[\"LIBFABRIC\"],\
\"kv_recompute_threshold\":$KV_RECOMPUTE_THRESHOLD,\
\"decoder_kv_blocks_ttl\":$DECODER_KV_BLOCKS_TTL,\
\"bidirectional_kv_xfer\":$BIDIR"

if [ "$ROLE" = "prefill" ]; then
  export VLLM_NIXL_SIDE_CHANNEL_HOST="$P_IP"
  PORT=8100
  KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_buffer_device\":\"cuda\",\
\"kv_role\":\"kv_producer\",\"engine_id\":\"prefill-engine-001\",\"kv_rank\":0,\
\"kv_parallel_size\":2,\"kv_connector_extra_config\":{$extra_config}}"
  # Prefix caching off on P ONLY: this is the cache-evicted scenario the
  # benchmark emulates. With it on, P hits its own cache and the D->P read
  # never has anything to prove.
  extra_args=(--no-enable-prefix-caching)
else
  export VLLM_NIXL_SIDE_CHANNEL_HOST="$D_IP"
  PORT=8200
  KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_buffer_device\":\"cuda\",\
\"kv_role\":\"kv_consumer\",\"engine_id\":\"decode-engine-001\",\"kv_rank\":1,\
\"kv_parallel_size\":2,\"kv_connector_extra_config\":{$extra_config}}"
  extra_args=()
fi

echo "Starting $ROLE on :$PORT  (bidirectional_kv_xfer=$BIDIR)"
exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  "${extra_args[@]}" \
  --port "$PORT" \
  --kv-transfer-config "$KV_CONFIG"
