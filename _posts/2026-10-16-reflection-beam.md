---
layout: post
title: "Serving Reflection Beam on vLLM from Day 0"
author: "Inferact, Reflection, and Nvidia"
date: 2026-10-16 09:00:00 +0000
summary: "vLLM supports Reflection's Beam-1.0 (501B total, 23B active) from day 0, with agent-driven optimization that cut decode latency 2.71× and DFlash2 speculative decoding with adaptive verification."
image: /assets/figures/2026-10-16-reflection-beam/social.png
social_image: /assets/figures/2026-10-16-reflection-beam/social.png
tags:
  - reflection
  - performance
  - kernels
  - speculative-decoding
---

<p align="center">
<img src="/assets/figures/2026-10-16-reflection-beam/hero.png" alt="vLLM × Reflection" width="100%">
</p>

**TL;DR:** Reflection's Beam is a 501B-parameter MoE with 23B active parameters per token. vLLM supports it from day 0 and optimizes it on NVIDIA Blackwell:

- **Full feature support.** Reasoning, tool calling, and structured outputs work out of the box, and you can run the model with tensor, data, or expert parallelism.
- **Agent-driven optimization.** Coding agents optimized the model autonomously, making batch-1 decode 2.71× faster and prefill 1.87× faster than our first implementation.
- **DFlash2 with adaptive verification.** With Beam's DFlash2 drafter, a single request reaches **708 tokens/s** on TP8, with an acceptance length of 3.7. Adaptive verification, which decides each step how many draft tokens are worth verifying, adds **10% more throughput** at high concurrency.

To try it, use our Docker image `vllm/vllm-openai:reflection-beam`:

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:reflection-beam \
  reflection/Beam-1.0-501B-A23B-NVFP4 \
  -tp 4 \
  --max-num-seqs 128 \
  --reasoning-parser beam \
  --tool-call-parser beam --enable-auto-tool-choice \
  --kv-cache-dtype fp8 \
  --attention-config '{"backend":"FLASHINFER"}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --speculative-config '{
    "method": "dflash",
    "model": "reflection/Beam-1.0-501B-A23B-DFlash2",
    "num_speculative_tokens": 7,
    "attention_backend": "FLASH_ATTN"
  }'
```

## Model architecture

Beam is a 52-layer MoE decoder with hidden size 5120. The MoE routing follows DeepSeek-V3. The main differences are in the attention layers and in how the residual stream is normalized.

<iframe class="vllm-embed" src="/assets/interactive_pages/beam-arch.html" title="Beam architecture: layer stack and one sparse decoder layer" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 1000px; border: 0; overflow: hidden;"></iframe>

*Figure 1: Beam's layer stack and one sparse decoder layer. Gold marks the parts that differ from a standard pre-norm MoE. [Open it full-screen](/assets/interactive_pages/beam-arch.html).*

### Attention

- **Hybrid sliding-window and NoPE layers.** 43 layers use sliding-window attention over the last 512 tokens, with RoPE. The other 9 (every sixth layer, plus the last) attend over the full context with **no positional encoding (NoPE)**.
- **SSMax on NoPE layers.** Without RoPE, attention flattens as the number of keys grows. SSMax scales each query by `(1 + p/2048)^0.1`, where `p` is the 1-based position, so the softmax sharpens as the context grows.
- **GQA with per-head Q/K norm.** 48 query heads, 8 KV heads, head dim 128. Q and K are RMS-normalized per head before RoPE or SSMax.
- **Sigmoid output gate.** Before `o_proj`, each head's output is multiplied by σ(gate + bias), where the gate comes from its own full-width projection with a learned per-head bias. A head can be turned down or off token by token.
- **Small KV cache.** Only the 9 full-attention layers keep KV for the whole sequence:
  - 9 × 2 × 8 × 128 values per token, or 36 KiB in BF16 and 18 KiB in FP8.
  - About 18 GiB of FP8 KV for a full 1M-token context, before sharding.

### MoE and residual stream

- **DeepSeek-V3-style MoE.** 256 routed experts (d_ff 2,560, NVFP4), routed the same way as DeepSeek-V3: sigmoid scores plus a correction bias pick the best 4 of 8 expert groups, then the top 8 experts, and the renormalized weights are scaled by 2.5. Unlike DeepSeek-V3, there is no shared expert. Layers 0–2 use a dense SwiGLU MLP (d_ff 20,480) instead.
- **FP32 router.** The 5120 × 256 router weights, accumulation, and logits are all FP32.
- **Sandwich norm.** Every sublayer has an RMSNorm before it and another after it.
- **Depth-scaled residuals.** The *n*-th residual add computes `h ← √((n−1)/n)·h + x/√n` instead of `h + x`. With unit-variance sublayer outputs, the residual stream keeps unit variance at every depth instead of growing with it.

## Agent-driven optimization

Our initial version of implementation was an unoptimized, almost pure PyTorch version of the model. We then optimized it autonomously with coding agents: they profiled the model, wrote the kernels, and benchmarked each change. On TP8, the seven changes below brought batch-1 decode from 8.95 ms to 3.30 ms per step, and an 8 × 2K prefill from 328 ms to 176 ms.

<iframe class="vllm-embed" src="/assets/interactive_pages/beam-opt-chart.html" title="Beam TP8 forward latency after each optimization" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 760px; border: 0; overflow: hidden;"></iframe>

*Figure 2: TP8 model-forward latency on GB200 after each change, for batch-1 decode and an 8 × 2,048-token prefill at ~100K context. The two panels use different scales. Show the table for exact values, or [open it full-screen](/assets/interactive_pages/beam-opt-chart.html).*

1. **Fused norm → residual → norm** (−40% decode). The post-norm, scaled residual add, and next pre-norm at each of the 104 sublayer boundaries now run as one Triton kernel, with FP32 math and the reference's BF16 rounding points.
2. **Fused Q/K norm, RoPE/SSMax, and output gate.** Per-head Q/K RMSNorm and RoPE (or SSMax) run as one kernel, and the sigmoid gate as another.
3. **All-reduce fused into the norms.** Both TP all-reduces in each layer run inside the norm kernel, built on FlashInfer's multicast all-reduce. After the MoE, the same kernel also combines the top-8 expert outputs.
4. **Specialized GEMM kernel for the FP32 router.** The router used to fall back to a generic path. vLLM's small-batch FP32 router kernel now supports Beam's 5120 × 256 shape, keeping FP32 weights, accumulation, and logits.
5. **Merged QKV + gate projection with a low-latency GEMM.** The Q, K, V, and gate weights are packed into one matrix at load time, so the four projections run as one GEMM. For small batches it uses a low-latency CuTe DSL GEMM or FlashInfer's TinyGEMM.
6. **Fused KV-cache update.** The Q/K norm and RoPE kernel now also writes K and V into the KV cache, so there is no separate cache-write kernel.
7. **Dense-MLP activation in one kernel** for the three dense layers.

## The fused graph

After these changes, one MoE layer runs as **13 kernels** in batch-1 decode on TP8.

<iframe class="vllm-embed" src="/assets/interactive_pages/beam-fused-graph.html" title="Beam fused decode graph for one MoE layer" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 900px; border: 0; overflow: hidden;"></iframe>

*Figure 3: Kernels in launch order for one MoE layer, from a rank-0 trace of batch-1 TP8 decode. Gold kernels were written, fused, or extended for Beam. Each kernel lists the operations it computes. [Open it full-screen](/assets/interactive_pages/beam-fused-graph.html).*

- **6 kernels were written, fused, or extended for Beam.** Every small Beam-specific op is merged into a neighbor, and both all-reduces live inside norm kernels.
- **7 kernels come from libraries:** TRTLLM-gen attention, the cuBLAS output projection, and the TRTLLM NVFP4 MoE path (quantization, two routing kernels, two grouped GEMMs). Most of the remaining time is here, so they are next.

## Attention DP + MegaMoE

<p align="center">
<img src="/assets/figures/2026-10-16-reflection-beam/megamoe.png" alt="Attention DP with FlashInfer CuTe DSL NVFP4 MegaMoE" width="60%">
</p>

*Figure 4: FlashInfer CuteDSL NVFP4 MegaMoE can fuse the communications with expert computation.*

In the high concurrency region, Beam's performance can be further improved by combining Attention DP with MegaMoE, powered by FlashInfer's CuTe DSL NVFP4 MegaMoE kernel. The Finalize + norm boundary fusion remains applicable.

| Configuration | Peak input token throughput |
|:---|:---|
| Attention DP + MegaMoE | 168.8K tokens / second |
| TP (`flashinfer_trtllm`, `flashinfer_nvlink_one_sided`) | 98.7K tokens / second |

Attention DP with MegaMoE delivered 1.71× higher peak prefill throughput than TP4. Benchmark setup: 4×GB300, with an input sequence length of 32K tokens and an output sequence length of 1 token to isolate prefill performance.

To enable Attention DP + MegaMoE, replace the MoE and parallel settings with:

```bash
  --moe-backend flashinfer_moe_ep_cutedsl \
  --data-parallel-size 4 \
  --enable-expert-parallel
```

## DFlash2 speculative decoding

Beam ships with a **DFlash2** draft head, which vLLM supports from day 0. Instead of drafting one token at a time, it predicts a whole block of tokens in a single forward pass from the target model's hidden states.

- **Drafter.** 5 sliding-window layers that read hidden states from 5 target layers and share the target's embedding and LM head. A grouped convolution and a small candidate selector restore some of the left-to-right dependency a parallel drafter loses.
- **Adaptive verification.** With `"enable_adaptive_verification": true`, vLLM predicts how likely each draft token is to be accepted and verifies only the ones worth their cost, instead of a fixed number per step. DFlash2 has no confidence head, so the acceptance estimate comes from the drafter's own logits (see our [recent post on adaptive verification](https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator)).
- **Kernels.** The drafter's convolution, norms, and KV writes are fused, and the whole draft step runs in a CUDA graph. FP8 KV cache works for the drafter too, with no change in acceptance.

Serving results on TP8, with and without adaptive verification, on the low-entropy category of [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench)'s `throughput_1k` split: ~1K-token prompts, mostly code completion and text sorting, with greedy decoding and up to 4K output tokens. Low-entropy text is easier to predict, so expect lower acceptance on open-ended chat.

| Concurrency | Fixed draft (tok/s) | Adaptive (tok/s) | Change | Acceptance length |
|---:|---:|---:|---:|---:|
| 1 | 708 | 707 | −0.1% | 3.71 → 3.61 |
| 8 | 3,232 | 3,221 | −0.3% | 3.67 → 3.57 |
| 32 | 7,744 | 8,275 | +6.9% | 3.68 → 3.29 |
| 128 | 15,012 | 16,632 | +10.8% | 3.68 → 3.23 |

Adaptive verification pays off under load. At concurrency 128 it raises output throughput by 10.8% and cuts p99 TPOT by 9%, while verifying fewer draft tokens per step, which lowers acceptance length slightly but saves more compute than it loses. At low concurrency, where extra draft tokens are nearly free, it is neutral.

## AgentX benchmark

<p align="center">
<img src="/assets/figures/2026-10-16-reflection-beam/agentx.png" alt="Beam 1.0 NVFP4 AgentX on GB300: total token throughput per GPU versus P90 interactivity, with DFlash2 and with speculative decoding off" width="100%">
</p>

*Figure 5: Beam 1.0 NVFP4 AgentX benchmark showing up to 88K total tok/s per GPU and up to 350 tok/s per user on GB300.*

We also tested Beam on AgentX, which replays multi-turn coding conversations to measure serving performance. With DFlash2, vLLM reached approximately 88K total tokens/s per GPU at 96 concurrent requests. Total throughput counts input and output tokens, including cached input. Input-token cache hit rates ranged from 91.4% to 95.6% across the sweep.

AgentX benchmark configuration: Beam 1.0 NVFP4 on 4× GB300 with TP4/EP1 and FP8 KV cache for both target and drafter. DFlash2 uses real acceptance (average AL 3.6 across measured points), block size 8, and probabilistic sampling. AgentX uses trajectory ratios 0.55–0.75, maximum reasoning effort, temperature 1.0, and 20 minutes per point. Total throughput includes cached input tokens. Source commits: vLLM 8e0922.

## Getting started

The command at the top of this post is a good default: TP4 on a single GB300 node, with DFlash2 and FP8 KV cache. To enable adaptive verification, add `"enable_adaptive_verification": true` to the `--speculative-config`. Other optimizations above are on by default.

- **Choosing a parallel setup.**
  - For the lowest latency, use TP4 or TP8.
  - For higher throughput, use data-parallel attention with expert parallelism, e.g. `-dp 4 --enable-expert-parallel`.
- **Reasoning and tool calling.** The `beam` parsers read the response template that ships with the checkpoint. By default, vLLM also constrains every response to that template, so tool calls always parse and match each tool's JSON schema. `response_format` and `thinking_token_budget` work as usual. To turn the default constraint off, set `VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT=0`.

## Day 0 Support for RL with SkyRL

We also collaborated with the SkyRL team to enable Day 0 support for RL on Beam. LoRA, BF16 full fine-tuning and MXFP8 RL are supported out of the box, with an upgraded RL stack for the 500B scale.

The SkyRL team post-trained the Beam model on the DAPO recipe with vLLM for inference and Megatron for training. LoRA, full fine-tuning and MXFP8 RL achieve similar gains in performance in 50 training steps, or 100 optimizer updates (reasoning `minimal`), with an average accuracy (avg@8) jump from ~61% to ~79% on AIME25. Average absolute log-probability difference between Megatron and vLLM stays under 0.01 with Rollout Routing Replay (R3) enabled for LoRA and BF16 full fine-tuning, and under 0.03 for MXFP8 RL.

<p align="center">
<img src="/assets/figures/2026-10-16-reflection-beam/skyrl-rl.png" alt="SkyRL training reward, AIME 2025 accuracy, response length, and train-inference logprob difference for LoRA, BF16 full fine-tuning, and MXFP8 full fine-tuning" width="100%">
</p>

Check out the SkyRL [blog](https://blog.skyrl.ai/day-0-beam) for more details!

## Acknowledgements

This work was a close collaboration between the Inferact, Reflection, and Nvidia teams:

- **Inferact:** Woosuk Kwon, Ziqi Zhao, Ao Shen, Tiezhen Wang
- **Reflection:** Kaixing Wu, Yoni Gozlan, Neil Gollapudi, Thibaut Barroyer, Aymen Kallala, Matthias Reso, Florent Gbelidji
- **Nvidia:** Siyuan Fu, Jinze Xue

<script>
(function () {
  // Resize embedded interactive figures to their content height (they post it when framed).
  window.addEventListener('message', function (event) {
    var d = event.data;
    if (!d || d.type !== 'vllm-embed-resize' || typeof d.height !== 'number') return;
    var frames = document.querySelectorAll('iframe.vllm-embed');
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].contentWindow === event.source) {
        frames[i].style.height = Math.ceil(d.height) + 'px';
      }
    }
  });
})();
</script>
