---
layout: post
title: "Adaptive Verification in vLLM: DSpark confidence-scheduled verification"
author: "vLLM Team"
summary: "Sizing the DSpark draft-verification budget from per-request confidence instead of verifying every drafted token, so one configuration holds the throughput/latency frontier from batch size 1 to 256."
image: /assets/figures/2026-08-14-dspark-adaptive-verification/fig3-pareto.svg
math: true
tags:
  - performance
  - speculative-decoding
---

Speculative decoding buys fewer decode steps with more compute. At batch size 1 that is a good trade: the GPU is memory-bound with spare compute, so the extra work (draft tokens) is close to free. At batch size 256 the trade is much more delicate. Draft tokens now compete with real tokens for the same compute, and every rejected token wastes useful compute; with enough rejected tokens, throughput drops significantly.

**TL;DR**: [DSpark](https://arxiv.org/abs/2607.05147)'s confidence head scores each drafted token's chance of surviving verification, so instead of picking a speculation length per deployment, vLLM can decide per step how much of the draft to verify. With adaptive verification on (`num_speculative_tokens: 7`), speculative decoding is able to provide benefits all the way to concurrency 256 and still maintains the benefits of the longer draft length at lower concurrencies. This reduces the need for users to tune `num_speculative_tokens` to their workload and deployment, and makes DSpark an easier "on-by-default" type of win. It landed in [PR #47808](https://github.com/vllm-project/vllm/pull/47808) as `enable_adaptive_verification`.

## The problem

Per-position acceptance decays fast: on DeepSeek-V4-Pro-0813 the last drafted token of a 7-token block survives less than 10% of the time, against more than 70% for the first. That low probability token costs a slot in every verification batch. While the GPU is memory-bound the slot is effectively free and worth the gamble; once it saturates the "gamble" has a real throughput cost. The challenge is that the crossover moves with load and workload dependent acceptance rates, so no static `num_speculative_tokens` is optimal across concurrencies. DSpark tackles this by having an adaptive draft budget that takes into account both the load of the system and how confident the DSpark head thinks the target model will accept each draft token.

## Scheduling the budget

DSpark drafts a block of *k* tokens per pass (`num_speculative_tokens`) and emits a confidence per position using a learned confidence head. The scheduler turns those into survival probabilities, the running product along each request:

$$
S(r, i) = \prod_{j \le i} \mathrm{confidence}(r, j)
$$

Survival only decreases with position *i*, so given a draft token budget of *B*, allocating it to the most probable draft sequences is just a global top-*B* over survival scores; that admits a contiguous prefix of each request's draft with no extra constraint. Slots compete across requests: position 5 of a confident request can outrank position 1 of a low-confidence one.

![Fixed-length verification versus confidence-scheduled trimming](/assets/figures/2026-08-14-dspark-adaptive-verification/fig1-policy.svg)

*Figure 1. The same batch under both policies. Fixed verification pays for all 21 slots including the ones with near-zero survival; with adaptive verification we only verify the best B=11.*

*B* comes from maximizing expected tokens per unit of step time:

$$
B^* = \arg\max_B \frac{N_\mathrm{sampling} + \sum_{j < B} S_\mathrm{sorted}[j]}{\mathrm{draft\_cost}[\mathrm{num\_reqs}] + \mathrm{verify\_cost}[T + B]}
$$

The numerator is one bonus token per sampling request plus the survival of the *B* best draft slots; *N*<sub>sampling</sub> counts the requests that will actually sample this step, so a request still working through a chunked prefill contributes nothing. The denominator is a profiled cost table, indexed by the step's token count: *T* is the tokens already scheduled that are not drafts, so *T* + *B* is the whole step. Both are arrays, so the choice is an `np.argmax` over a cumulative sum and costs are in microseconds.

Sizing runs on the CPU while the GPU is still working on the previous step, from a double-buffered confidence array that is one step old. Handing those *B* slots out to individual requests runs on the GPU against current values, so the per-request allocation uses current confidences. The selection is written in PyTorch, lowered to Triton by `torch.compile`, and never reads back to the host.

## Varlen decode CUDA graphs

To properly support variable-sized verifications we also need varlen decode CUDA graphs. That requires attention kernel support: the sparse MLA kernels are naturally varlen, since each query token has an independent top-k, and DeepSeek open-sourced a varlen indexer kernel in [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), which is integrated as part of [PR #47808](https://github.com/vllm-project/vllm/pull/47808). Decode graphs are captured with `num_reqs = min(num_tokens, max_num_seqs)` and a promised `max_query_len = num_speculative_tokens + 1`, so one graph serves any mix of 1 to `num_speculative_tokens + 1` tokens per request.

## The cost model

The budget rule divides by a step cost, so that cost has to be cheap to look up and a good approximation of the real cost. At startup the engine times dummy steps across a fixed set of shapes (CUDA graph shapes plus a couple above the max cudagraph size), taking the median of five runs per shape. That becomes two flat lookup tables: the verification table is indexed by token count, and the drafter table by request count, since drafting costs the same regardless of how many tokens are verified. The two are summed.

![Measured verify and draft cost curves against the lookup tables](/assets/figures/2026-08-14-dspark-adaptive-verification/fig2-costcurve.svg)

*Figure 2. Both cost tables from a real startup profile, with the cost being the median of 5 samples.*

Inside the captured CUDA graphs cost is a staircase rather than a line, because of cudagraph padding: a batch of 121 tokens runs the 128-token graph and (mostly) pays for all 128. Past the capture limit the staircase ends and cost really is continuous. There is a notable jump where we fall out of the cudagraph region, and that transition is sharp enough in the cost curve to strongly encourage the budget algorithm to stay within the cudagraph region.

Profiling noise is handled by forcing the curve monotonic. Step cost can genuinely fall as the batch grows, because of kernel tile sizes, so enforcing monotonicity helps smooth out the cost curve. The steps are profiled against a synthetic KV context, 8192 tokens by default and tunable with `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`.

## Results

DeepSeek-V4-Pro-0813, TP=8 on 8×B300 (SM100), expert parallel, FP8 KV cache, `max_model_len` 16384, `max_cudagraph_capture_size` 4096, on vLLM `main` at `73b8394`. The benchmark is 880 prompts at temperature 1.0, up to 2048 output tokens, swept over concurrency 1 to 256.

![Aggregate throughput against interactivity for adaptive and fixed speculation lengths](/assets/figures/2026-08-14-dspark-adaptive-verification/fig3-pareto.svg)

*Figure 3. Throughput versus interactivity for different speculation schemes; adaptive verification stays on the Pareto frontier throughout.*

Adaptive verification stays on the edge of the Pareto curve for the whole sweep, and well outside no speculation at both ends. The effect is easy to read off the graph: it behaves like a long fixed block at low concurrency and a short one at high concurrency, which gives you both without having to know the shape of your workload in advance.

## Limitations

- FULL varlen decode graphs require `AttentionCGSupport.ALWAYS`, which the DSV4 sparse-MLA, sparse-SWA, and indexer backends report on SM100. Elsewhere adaptive verification is rejected at startup rather than falling back to PIECEWISE.
- `--enforce-eager` (step costs are profiled from captured graphs), LoRA, and pipeline parallelism are all not supported currently.
- Output logprobs are rejected when adaptive verification is on, because verification compacts logits after the forward pass.

## Appendix: reproducing

All the commands below are using [PR #47808](https://github.com/vllm-project/vllm/pull/47808), now merged into vLLM `main`; the numbers above were measured at `73b8394`.

**Server** (all measurements; ablations are `--speculative-config` deltas):

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro-0813 \
  --tokenizer-mode deepseek_v4 --trust-remote-code \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8 --max-model-len 16384 --max-num-seqs 256 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.8 \
  --compilation-config '{"max_cudagraph_capture_size":4096}' \
  --speculative-config '{"method":"dspark","attention_backend":"FLASH_ATTN","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}'
```

The draft defaults to the target checkpoint, so `"model"` can be omitted. `--kv-cache-dtype fp8` is required: the `fp8_ds_mla` layout rejects other KV dtypes. `--max-num-seqs` matters too — the default is 128, which would cap the batch below the top of the concurrency sweep. We increase the `max_cudagraph_capture_size` to `(num_speculative_tokens + 1) * max_num_seq` to ensure every verfication batch is inside a cudagraph. The larger capture size needs more memory for cudagraphs hence `--gpu-memory-utilization 0.8`; at the default it OOMs while capturing.

- fixed k: `"enable_adaptive_verification": false`, `"num_speculative_tokens": k`, for k ≥ `dspark_block_size` (5 on this checkpoint)
- no speculation: omit `--speculative-config`

**Throughput sweep**, per concurrency `c ∈ {1, 16, 32, 64, 128, 256}`, after one warmup pass (`--speed-bench-output-len 256 --num-prompts 64 --max-concurrency 32`):

```bash
MODEL=deepseek-ai/DeepSeek-V4-Pro-0813
for c in 256 128 64 32 16 1; do
  n=880; [ "$c" = 1 ] && n=240
  vllm bench serve \
    --backend openai-chat --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions --model "$MODEL" \
    --tokenizer "$MODEL" --tokenizer-mode deepseek_v4 \
    --dataset-name speed_bench --dataset-path <speed-bench-dir> \
    --speed-bench-dataset-subset qualitative --speed-bench-output-len 2048 \
    --num-prompts $n --max-concurrency $c --request-rate inf \
    --skip-chat-template --disable-shuffle --temperature 1.0 --seed 0 \
    --save-result --result-filename adaptive_on_c${c}.json
done
```

`--disable-shuffle` plus the fixed prompt set gives every arm identical prompts in identical order; `output_throughput` from the result JSON is the tok/s plotted above. `--speed-bench-output-len` is a cap, not a target — requests stop at EOS, so the realized average is well under 2048.

## Acknowledgments

This work was done by Lucas Wilkinson (Red Hat) and Benjamin Chislett (NVIDIA). Thanks to the [DSpark](https://arxiv.org/abs/2607.05147) authors for the drafting algorithm and the confidence head, and to DeepSeek for the DeepSeek-V4 checkpoints.