---
layout: post
title: "DeepSeek-V4.1-Flash on vLLM: 5x Agentic Throughput Since Day 0"
author: "Inferact and the vLLM Team"
date: 2026-10-07 09:00:00 +0000
summary: "Within three weeks of release, vLLM made DeepSeek-V4.1-Flash 1.9x faster at low concurrency and lifted its throughput 5x on SemiAnalysis AgentX, with SWA bounded replay, CUDA graphs, DeepSeek's new kernels, and vLLM kernel fusions."
image: /assets/figures/2026-10-07-deepseek-v41-flash/agentx-results.png
social_image: /assets/figures/2026-10-07-deepseek-v41-flash/agentx-results.png
tags:
  - deepseek
  - performance
  - kernels
---

<p align="center">
<img src="/assets/figures/2026-10-07-deepseek-v41-flash/agentx-results.png" alt="Figure 1: SemiAnalysis AgentX results for vLLM from the day-0 model release to Oct 2" width="100%">
</p>

<p align="center">
<em>Figure 1: SemiAnalysis AgentX results for vLLM from the day-0 model release to Oct 2 (<a href="https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_seq=agentic-traces&i_xmode=interactivity&g_model=DeepSeek-V4.1-Flash&i_gpus=gb300_vllm&i_dstart=2026-09-11&i_dend=2026-10-02&i_metric=y_tpPerGpu">source</a>).</em>
</p>

**TL;DR:** In the three weeks after DeepSeek-V4.1-Flash's release, Inferact and the vLLM community optimized the model, achieving a 1.9× speedup at low concurrency and a 5.3× throughput improvement under a 150 TPS constraint. The performance improvement comes from:

- We implemented SWA bounded replay with CUDA graphs, achieving a ~30% TTFT reduction.

- We integrated DeepSeek's newly released kernels, including MegaAttention, Mega-mHC, Mega-Gate, and DeepSelect.

- We aggressively fused and parallelized the remaining kernels, including mHC side streams, and fused all-reduce with the preceding and following ops into a single kernel.

DeepSeek V4.1 introduces a highly efficient architecture for long-horizon agentic serving tasks: with its causal encoder-decoder (CED) architecture, the model activates 16B parameters per token during decode but only 8B parameters during prefill. The model is also extremely memory efficient. It combines several techniques to shrink the KV cache size: Compressed Sparse Attention 2 (CSA2), FP4 KV cache, and inter-layer KV cache sharing, pushing the global KV footprint to 890 bytes per token. This post shows how we combine these model-level optimizations from DeepSeek with vLLM-side system optimizations to achieve 5× throughput on the SemiAnalysis AgentX agentic serving benchmark. We highlight our optimizations in two categories: SWA bounded replay and kernel-related optimizations.

## SWA bounded replay

DeepSeek-V4.1-Flash keeps two kinds of KV caches. Global KV is compressed, shared across layers, and stored in FP4, at about 890 bytes per token ([V4.1 report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)). Sliding-window (SWA) KV is uncompressed in FP8, covering the last 128 positions in each of the 40 layers.

SWA KV creates two costs:

1. Prefix caching must store it at every possible hit boundary, costing more than 10× the storage of global KV.

1. Prefill runs layers 21–39 on every prompt token, though decode only reads their last 128 positions.

One straightforward approach is to recompute SWA KV instead of caching it. Exact recomputation, however, is expensive, because each layer's 128-token window depends on earlier positions in the layer below, so rebuilding it across L layers means replaying roughly L × 128 tokens.

DeepSeek V4.1 introduces SWA bounded replay, which trades exactness for efficiency. It reruns only the last 128 tokens and clips the SWA window at the replay start. The result isn't bit-exact, but DeepSeek reports negligible quality loss (details below). vLLM applies it in two places, one for each cost.

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-swa-replay.html" title="SWA bounded replay: exact rebuild vs. bounded replay after a prefix hit" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

<p align="center">
<em>Figure 2: Exact rebuild cascades one window per layer. Bounded replay recomputes only the last window, clipped at the replay start s. Step through it, or <a href="/assets/interactive_pages/dsv41-swa-replay.html">open it full-screen</a>.</em>
</p>

### Encoder side: rebuild the window on a cache hit

With encoder-side replay, vLLM caches only the global KV and skips SWA KV. On a prefix hit of length H, it reruns tokens [H − 128, H) to rebuild SWA KV, with windows clipped at s = H − 128.

### Decoder side: skip most of the prompt prefill

In DeepSeek V4.1's CED architecture, layer 20 computes the decoder's global KV and layers 21–39 reuse it. vLLM therefore runs layer 20 on every token to produce that global KV, and runs layers 21–39 only on each request's last 128 tokens. For long prompts, this skips nearly half the model.

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-swa-decoder.html" title="Decoder-side SWA bounded replay: layers 21–39 run only on the last 128 tokens" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

<p align="center">
<em>Figure 3: In the decoder, only layer 20's global KV is needed for every token. Layers 21–39 run only on the last 128 tokens. Step through it, or <a href="/assets/interactive_pages/dsv41-swa-decoder.html">open it full-screen</a>.</em>
</p>

### CUDA graphs for the trimmed layers

After trimming, layers 21–39 do so little GPU work that, run eagerly, kernel launch overhead dominates and the GPU sits idle. Their input shapes also differ from layers 0–20, so the two parts can't be captured in one CUDA graph. vLLM's breakable PIECEWISE graph already breaks in the middle of the model, which gives a natural split: layers 0–20 are captured on the full batch, and layers 21–39 are captured separately on the trimmed batch. This makes CUDA graphs usable for trimmed prefill, and lets layers 21–39 use their own capture sizes for better graph coverage.

SWA bounded replay is on by default for DeepSeek-V4.1 and is controlled by `--[no-]swa-bounded-replay`.

### Accuracy and performance results

Although SWA bounded replay is not exact, DeepSeek has reported only negligible quality loss. We confirmed this in vLLM on benchmarks including GSM8K and GPQA and observed no meaningful accuracy difference (gaps within about 1.5 standard errors).

On performance, the encoder side trades one window of prefill per hit for cache space, so the speedup comes from the decoder side. We measure single-request prefill TTFT in three settings: replay off; replay on without decoder CUDA graphs (layers 21–39 run eagerly, and only eager steps are trimmed); and replay on with decoder CUDA graphs.

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-prefill-ttft.html" title="Prefill TTFT with SWA bounded replay on NVIDIA GB200" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

<p align="center">
<em>Figure 4: Single-request prefill TTFT (ms) of DeepSeek-V4.1-Flash with SWA bounded replay off and on, with and without decoder CUDA graphs. Median of 3 runs on NVIDIA GB200, with prefix caching off. Percentages are relative to replay off (lower is better). Hover a bar for the absolute TTFT, or <a href="/assets/interactive_pages/dsv41-prefill-ttft.html">open it full-screen</a>.</em>
</p>

**Decoder replay with CUDA graphs cuts prefill computation time by 30–40%.** CUDA graphs matter most for short prompts, where kernel launches are the bottleneck: without them, launch overhead outweighs the GPU savings and replay is slower than the baseline (up to +12% at 1K on DEP2). For long prompts, the GPU work is large enough to hide launch overhead, so eager replay already captures most of the gain and CUDA graphs add a few more points.

## Kernels

DeepSeek released DeepSeek-V4.1-Flash together with new kernels in three of its repositories. DeepSelect is a new top-k library for DeepSeek Sparse Attention. DeepGEMM added sparse indexer kernels and several GEMM-related fused kernels. FlashMLA added NVFP4 KV cache support and a fused attention kernel, MegaAttention. We have integrated several of these open-source kernels into vLLM and track progress in [#57448](https://github.com/vllm-project/vllm/issues/57448).

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-kernel-fusion.html" title="Kernel fusion on the DeepSeek-V4.1 decode path" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

<p align="center">
<em>Figure 5: Single-layer forward pass of DeepSeek-V4.1. Gold outlines mark the kernel fusions, with their associated vLLM PRs. Step through the low-latency and high-throughput paths, click FULL LAYER for the whole layer, or <a href="/assets/interactive_pages/dsv41-kernel-fusion.html">open it full-screen</a>.</em>
</p>

**Mega-mHC ([#56962](https://github.com/vllm-project/vllm/pull/56962)).** Mega-mHC fuses the mHC chain into one kernel: the post step, the delayed-pre step, and RMSNorm. It replaces an existing TileLang fused path that DeepGEMM's implementation now outperforms. The kernel is 1.14–1.51× faster than the TileLang version on NVIDIA GB200.

**Mega-Gate ([#56266](https://github.com/vllm-project/vllm/pull/56266)).** Mega-Gate fuses the MoE router (gate GEMM, expert scoring, bias, and top-k selection) into one kernel. Previously, these operations ran as one GEMM followed by a separate top-k kernel, costing an extra launch and a round trip through memory for the scores. This fusion leads to 1.18–1.31× kernel speedups at medium batch sizes.

**mHC multistream overlap ([#57603](https://github.com/vllm-project/vllm/pull/57603)).** In V4.1, the mHC coefficients are shifted by one sublayer, so the next seam's coefficient GEMM reads only residual streams that already exist before attention or the FFN runs. Neither side needs the other's output until the next post/pre step combines them. At small batch sizes, vLLM now computes the next mHC block's coefficients on a side CUDA stream, in parallel with attention and the FFN. This hides work that would otherwise sit on the critical path of latency-bound decode. In the TP4 low-latency scenario, this reduces latency by about 4%.

**Sparse MQA logits ([#56254](https://github.com/vllm-project/vllm/pull/56254)).** In V4.1, later indexer layers pick their top-k from a fixed set of 16K candidate positions. Previous implementations compute the score for the entire context and mask all non-candidate blocks before scoring. DeepGEMM's sparse kernels score only the candidates, so cost no longer grows with context. Per layer on NVIDIA GB300, it's 1.2× faster at 8K tokens and 14–23× faster at 512K. End-to-end on 4× NVIDIA GB300, decode improves 3–6%. Prefill is 1.43× faster at 512K and 2× faster at 1M context.

**MegaAttention with NVFP4 compressed KV ([#56935](https://github.com/vllm-project/vllm/pull/56935)).** FlashMLA's MegaAttention kernel performs query RoPE, sparse attention, inverse RoPE on the output, and the FP8 cast in a single launch, writing straight into the buffer the output projection reads. That removes the separate kernels and memory round trips between attention and the next layer. It also reads a new NVFP4 compressed KV format, which is 45% smaller than the previous FP8 KV cache. MegaAttention also improves kernel efficiency by 1.45× through aggressive fusion that eliminates HBM writes between operations.

**Low-latency fused WO-A kernel ([#58634](https://github.com/vllm-project/vllm/pull/58634)).** For small decode batches on Blackwell, we fuse inverse RoPE, FP8 quantization, the WO-A batch GEMM, and MXFP8 requantization into a single CuTe-DSL kernel, reducing the pre-WO-B chain from three kernels to one. The key idea is to keep intermediate activations on-chip and pipeline data movement with compute, avoiding extra kernel launches and global-memory round trips that dominate at small batch sizes. This improves the fused WO-A path by up to ~2.1× and delivers up to ~6–7% lower inter-token latency at low concurrency.

**Engram.** V4.1's Engram layers look up rows from two large FP8 tables keyed by hashed token n-grams. Each step reads only a few rows, so table placement and lookup latency matter more than compute. We prefetch CPU-offloaded Engram lookups asynchronously, overlapping host-memory access with decoder compute to speed up low-batch decode ([#56512](https://github.com/vllm-project/vllm/pull/56512)). Engram heads are sharded with a unified TP/DP scheme, and co-located DP replicas share the same host tables, avoiding redundant copies and any DP communication on the lookup path ([#57651](https://github.com/vllm-project/vllm/pull/57651)). For these large host-resident tables, we also support transparent huge pages (THP) to reduce page-fault overhead, delivering up to 10× faster lookup kernels for prefills ([#56926](https://github.com/vllm-project/vllm/pull/56926)). We also added optimizations for cases that don't have enough huge pages available ([#59327](https://github.com/vllm-project/vllm/pull/59327)).

## Agentic performance

We measure performance with the SemiAnalysis AgentX benchmark as a representative agentic serving workload (detailed in our [previous post](https://vllm.ai/blog/2026-09-08-vllm-agentx)). Together, these optimizations give vLLM significant performance improvements over our day-0 implementation. As Figure 6 shows, our low-latency result improves 1.9× over our day-0 result, and the high-throughput result improves about 5×.

<p align="center">
<img src="/assets/figures/2026-10-07-deepseek-v41-flash/agentx-results.png" alt="Figure 6: SemiAnalysis AgentX results for vLLM from the day-0 model release to Oct 2" width="100%">
</p>

<p align="center">
<em>Figure 6: SemiAnalysis AgentX results for vLLM from the day-0 model release to Oct 2 (<a href="https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_seq=agentic-traces&i_xmode=interactivity&g_model=DeepSeek-V4.1-Flash&i_gpus=gb300_vllm&i_dstart=2026-09-11&i_dend=2026-10-02&i_metric=y_tpPerGpu">source</a>).</em>
</p>

For low-latency serving, we use TP4 with FlashInfer attention. Small-batch decode is largely memory-bandwidth bound, so sharding the model weights across four GPUs is a good fit. We also tried MegaAttention, but its main advantage is in higher-throughput settings where fusion has more room to help. At TP4, that benefit was much smaller, and FlashInfer ended up being faster in our runs.

For high throughput, we switch to DEP2, using DP attention with experts split across GPUs. Since V4.1 uses a shared KV latent across all heads, TP would duplicate the KV cache across GPUs. DP avoids this duplication: each GPU stores KV only for the requests it serves, with session affinity preserving prefix-cache locality across turns. MegaAttention further cuts the per-request KV footprint with NVFP4, nearly halving it versus FP8 and increasing per-GPU concurrency.

Notably, V4.1 is highly memory efficient and does not need KV cache offloading throughout the benchmark. We expect KV cache offloading to start helping at higher concurrency with P/D disaggregation.

SWA bounded replay, together with prefill-side kernel optimizations, greatly improved the TTFT as well.

<p align="center">
<img src="/assets/figures/2026-10-07-deepseek-v41-flash/ttft-vs-throughput.png" alt="Figure 7: TTFT vs. throughput after the optimizations" width="100%">
</p>

<p align="center">
<em>Figure 7: TTFT vs. throughput after the optimizations (<a href="https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_seq=agentic-traces&i_xmode=ttft&g_model=DeepSeek-V4.1-Flash&i_best=0&i_gpus=gb300_vllm&i_metric=y_tpPerGpu&i_dstart=2026-09-11&i_dend=2026-10-02">source</a>).</em>
</p>

Figure 7 shows the optimized TTFT–throughput trade-off. At around 100K throughput, TTFT drops by nearly 70% through three optimizations combined:

- SWA bounded replay lets the upper half of the model process only the last 128 tokens, cutting prefill computation roughly in half.

- CUDA graphs keep the small trimmed replay fast on the GPU instead of bound by CPU kernel launches, so we realize the full speedup.

- Kernel improvements accelerate the model computation.

## Acknowledgments

We thank DeepSeek for open-sourcing DeepSeek-V4.1-Flash and the associated kernels, the Inferact team for the initial model bring-up and optimizations, NVIDIA for their collaboration and support, and SemiAnalysis for the AgentX benchmark.

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
