#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Sweep prompt length with vLLM's multi-turn conversation benchmark, for one leg
# of the bi-directional KV transfer A/B (vLLM blog, 2026-09-21).
#
# Runs on the proxy node, against a 1P1D pair that is already serving and a
# disagg_proxy_multiturn.py already listening on :8000.
#
# Usage:
#   VLLM_SRC=~/vllm_src LEG=off ./run_multiturn_sweep.sh
#   VLLM_SRC=~/vllm_src LEG=on  ./run_multiturn_sweep.sh
#
# GLM5.2 runs: MODEL=zai-org/GLM-5-FP8 SERVED_NAME=GLM PFX_LIST="2000 4000 8000 12000 16000 20000"

set -euo pipefail

VLLM_SRC="${VLLM_SRC:?point VLLM_SRC at your vLLM clone}"
LEG="${LEG:?set LEG=off or LEG=on (must match how the servers were launched)}"

MODEL="${MODEL:-Qwen/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-Qwen}"
PROXY_URL="${PROXY_URL:-http://localhost:8000}"

# Prompt lengths published in the post: Qwen3-32B to 24k, GLM5.2 to 20k.
PFX_LIST="${PFX_LIST:-2000 4000 8000 12000 16000 20000 24000}"
NUM_CONVERSATIONS="${NUM_CONVERSATIONS:-10}"
NUM_TURNS="${NUM_TURNS:-10}"
INPUT_TOKENS="${INPUT_TOKENS:-150}"    # per user turn
OUTPUT_TOKENS="${OUTPUT_TOKENS:-300}"  # per assistant turn
NUM_CLIENTS="${NUM_CLIENTS:-2}"
MAX_ACTIVE_CONVERSATIONS="${MAX_ACTIVE_CONVERSATIONS:-10}"
SETTLE_SECONDS="${SETTLE_SECONDS:-20}"

OUT="${OUT:-$HOME/results/${SERVED_NAME}_${LEG}}"
mkdir -p "$OUT"

BENCH_DIR="$VLLM_SRC/benchmarks/multi_turn"
cd "$BENCH_DIR"

# One-time corpus download. Each conversation slices a fresh, non-overlapping
# window, so the corpus must hold roughly num_conversations x prefix tokens.
# pg1184 is ~600-700k tokens, which covers 10 x 24k with room to spare.
if [ ! -f pg1184.txt ]; then
  wget -q https://www.gutenberg.org/ebooks/1184.txt.utf-8
  mv 1184.txt.utf-8 pg1184.txt
fi

for PFX in $PFX_LIST; do
  # prefix_num_tokens has no CLI flag, so generate a config per point. Use a
  # constant distribution: a lognormal with average == max is clipped by max and
  # lands below the nominal length, which smears the x-axis.
  jq --argjson v "$PFX" \
     --argjson nc "$NUM_CONVERSATIONS" \
     --argjson nt "$NUM_TURNS" \
     --argjson it "$INPUT_TOKENS" \
     --argjson ot "$OUTPUT_TOKENS" '
     .num_conversations = $nc
     | .prompt_input.num_turns         = {distribution:"uniform", min:$nt, max:$nt}
     | .prompt_input.prefix_num_tokens = {distribution:"constant", value:$v}
     | .prompt_input.num_tokens        = {distribution:"constant", value:$it}
     | .prompt_output.num_tokens       = {distribution:"constant", value:$ot}
     ' generate_multi_turn.json > "/tmp/gen_${PFX}.json"

  echo "=== $SERVED_NAME  leg=$LEG  prefix=$PFX ==="
  # --send-conversation-id is off by default and is REQUIRED: it is what lets the
  # proxy key KV reuse across turns. Without it every turn is a cache MISS and
  # the ON leg silently measures the OFF leg.
  python3 benchmark_serving_multi_turn.py \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --url "$PROXY_URL" \
    --input-file "/tmp/gen_${PFX}.json" \
    --num-clients "$NUM_CLIENTS" \
    --max-active-conversations "$MAX_ACTIVE_CONVERSATIONS" \
    --send-conversation-id \
    --warmup-percentages=0%,20% \
    --stats-json-output "$OUT/stats_pfx${PFX}.json" \
    2>&1 | tee "$OUT/bench_pfx${PFX}.log"

  sleep "$SETTLE_SECONDS"   # let D's retained blocks lapse before the next point
done

echo
echo "Results in $OUT"
