---
layout: post
title: "vLLM Delivers Day-0 Support for MiniMax-M3.1-Flash"
author: "Inferact and the vLLM Team"
summary: "Day-0 vLLM support for MiniMax-M3.1-Flash on NVIDIA Blackwell, NVIDIA Rubin, and AMD MI350-series GPUs, with NVFP4 MSA kernels, NVFP4 MegaMoE, DSpark speculative decoding, AgentX results, and day-0 RL."
image: /assets/figures/2026-10-23-minimax-m31-flash/hero.png
social_image: /assets/figures/2026-10-23-minimax-m31-flash/hero.png
tags:
  - minimax
  - day-0-support
  - performance
  - kernels
  - hardware
---

<p align="center">
<img src="/assets/figures/2026-10-23-minimax-m31-flash/hero.png" alt="vLLM x MiniMax-M3.1-Flash day-0 support" width="100%">
</p>

MiniMax-M3.1-Flash is the latest model in the MiniMax-M3 family, with improvements across reasoning, coding, agentic, and visual tasks. It is also built for inference efficiency, with a native NVFP4 KV cache, NVFP4-quantized MoE weights, and a DSpark draft model for speculative decoding.

vLLM supports MiniMax-M3.1-Flash on day 0 on NVIDIA Blackwell GPUs, on NVIDIA Rubin GPUs with dedicated Rubin kernels, and on AMD MI350-series GPUs, with strong performance on agentic workloads.

## TL;DR

- **MiniMax Sparse Attention (MSA)** with an **NVFP4 KV cache** for both the indexer and the attention path, plus **NVFP4 MoE weights** and a **DSpark** drafter.
- **Blackwell MSA kernels** use the new CUDA 13.4 `QMUL4` instruction to dequantize NVFP4 into FP8 inside the kernel: up to **1.7× faster** indexer QK decode and **1.3× faster** sparse attention decode.
- **NVFP4 MegaMoE** with MXFP8 shared-expert fusion and early TMEM release: **20% lower kernel latency** in prefill and **7% higher prefill throughput**.
- **Indexer decode context parallelism (DCP)**: up to **3.3× faster** indexer per sparse layer at 512K context on 4× GB300.
- **QKV GEMM + norm + RoPE + KV-insert fusion**: **1.1–1.6× faster** prefill at 512+ tokens and about **19% lower P50 TTFT** at TP4.
- **FlashInfer PrimiTS** backend for the DSpark drafter attention: **1.3–2× faster** than FlashAttention 4, **5% lower** total forward time.
- **Rubin (SM107) MSA kernels** are also integrated into vLLM with state-of-the-art performance.
- **AMD MI350 series**: the NVFP4-native MSA kernel, indexer decode context parallelism, dual-batch overlap, and the mono kernel deliver a **2–3×** performance boost across the board.
- **Agentic Serving**: on NVIDIA GB300, up to **418.9 tok/s per user** at the latency-sensitive end; on NVIDIA Rubin, up to **724.9 tok/s per user** and **191.5K total tok/s per GPU**; on AMD MI355X, up to **133.3M total tokens per dollar** of TCO on the SemiAnalysis AgentX benchmark.
- **Day-0 RL support**: end-to-end RL with veRL, FSDP2, and LoRA on the NVFP4 checkpoint.

## Quick start

Prebuilt images are available for both vendors:

- NVIDIA: `vllm/vllm-openai:minimax-m31-flash`
- AMD: `vllm/vllm-openai-rocm:minimax-m31-flash`

Note that the model will only be available in the nightly PyPI wheel after the code is reviewed and merged upstream, which usually takes about a week. <!-- TODO: link the upstream vLLM PR once it is open. --> Full recipes are at [vLLM Recipes](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash), and the checkpoint is at [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash).

### TP4 on NVIDIA

We recommend the CUDA 13.4 toolkit for the best performance.

```bash
vllm serve MiniMaxAI/MiniMax-M3.1-Flash \
  -tp 4 \
  --reasoning-parser minimax_m3 \
  --block-size 128 \
  --kv-cache-dtype nvfp4 \
  --attention-config '{"indexer_kv_dtype": "nvfp4"}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_TS","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic"}' \
  --disable-hybrid-kv-cache-manager
```

### TP4 on AMD

```bash
vllm serve MiniMaxAI/MiniMax-M3.1-Flash \
  -tp 4 \
  --reasoning-parser minimax_m3 \
  --block-size 128 \
  --kv-cache-dtype nvfp4 \
  --attention-config '{"indexer_kv_dtype": "nvfp4", "indexer_decode_cp": true}' \
  --quantization-config.moe mxfp8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic"}' \
  --disable-hybrid-kv-cache-manager
```

## MSA primer

The MiniMax-M3 model family replaces full attention with a variant of block-sparse attention ([arXiv 2606.13392](https://arxiv.org/abs/2606.13392v1)) to make inference more efficient. The KV cache is split into blocks of 128 tokens. A lightweight indexer scores every key in the context with one index query per GQA group against a single shared index-key head, then max-pools those scores into one score per block. Main attention then runs exactly, but only over the top 16 blocks, and the most recent block is always kept. That caps each query at 2,048 keys, however long the context is.

The payoff is largest in decode, which is limited by memory bandwidth. Each step still scans the index keys for the whole context, but that is only 128 bytes per token per layer, an eighth of the 1,024 bytes of K/V that dense attention would read (FP8, 4 KV heads × 128 dims). The full K/V is then read for just the 2,048 selected tokens. At long-context decode, per-step HBM traffic drops to about 12.5% of what dense attention reads.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-m31-msa.html" title="MiniMax Sparse Attention: indexer, top-k block selection, and sparse attention" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 900px; border: 0; overflow: hidden;"></iframe>

*Figure 1: MiniMax Sparse Attention. Each GQA group's indexer picks its top KV blocks, and every query head in that group attends exactly over only those blocks. Step through it.*

### What is new in the MiniMax-M3.1 architecture

MiniMax-M3.1 greatly improves inference efficiency over the previous generation: it was natively quantization-aware trained for an NVFP4 KV cache, and its routed MoE weights are stored in NVFP4. The NVFP4 experts shrink the checkpoint from 451.5 GB (MiniMax-M3-MXFP8) to 250.1 GB, which is 44.6% smaller. The NVFP4 KV cache needs 34,560 bytes per token instead of 61,440 bytes in FP8, a saving of 43.75% per token. Together, on four NVIDIA B300 GPUs with DEP4 deployment and 256K-token sequences, M3.1 fits 76 concurrent sequences where M3 with an FP8 KV cache fits 28, about 2.7× as many on the same hardware, which directly raises serving throughput.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-m31-gpu-memory.html" title="How many MiniMax-M3.1-Flash sequences fit on each GPU" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 900px; border: 0; overflow: hidden;"></iframe>

*Figure 2: MiniMax-M3.1 is quantization-aware trained to use an NVFP4 KV cache for both the indexer and sparse attention. Change the KV cache dtype, parallelism, GPU, or context length to see how many sequences fit.*

## Performance optimizations

vLLM delivers strong MiniMax-M3.1 performance across multiple hardware platforms. On NVIDIA GPUs, we ship kernels optimized for the NVFP4 KV cache. Our MSA repository ([vllm-project/MSA](https://github.com/vllm-project/MSA)) contains kernels tuned for different workloads and operations. These include MSA indexer prefill and decode, sparse attention prefill and decode, and an MSA-specific top-k kernel.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-m31-layer-kernels.html" title="One MiniMax-M3.1 sparse decoder layer, kernel by kernel" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 1100px; border: 0; overflow: hidden;"></iframe>

*Figure 3: One MiniMax-M3.1 sparse decoder layer on Blackwell at decode (M ≤ 32), kernel by kernel. Numbered badges mark the optimizations in this post; click one to highlight its kernels.*

### NVIDIA Blackwell optimized MSA kernels

The KV cache is stored in NVFP4 while the query stays in FP8, so we cannot use the native NVFP4 tensor core instructions for the matrix multiply. Instead, the kernel dequantizes the NVFP4 KV cache to FP8 and runs an FP8 MMA. To keep the dequantization cost low, we use a new Blackwell-specific PTX instruction introduced in CUDA 13.4, `mul.e4m3x4.e2m1x4.e4m3x4` (SASS `QMUL4`). It scales four NVFP4 values by their scale factor and produces four FP8 values in a single instruction. This takes far fewer CUDA core instructions than the previous FP16 round-trip path. The result is up to a 1.7× speedup on the indexer QK decode kernel and 1.3× on the sparse attention decode kernel.

### NVFP4 MegaMoE

MegaMoE is a key building block for modern expert-parallel (EP) deployments. We extended DeepGEMM's MegaMoE to NVFP4 precision with fused MXFP8 shared experts, open-sourced in our fork ([vllm-project/DeepGEMM](https://github.com/vllm-project/DeepGEMM)). Adding NVFP4 support to MegaMoE is more than a dtype swap. NVFP4 halves the bytes per element and doubles tensor core throughput compared with the previous FP8×FP4 path. The rest of the kernel has to change too, or the pipeline falls out of balance.

One such change is releasing the TMEM stage early in the FC1 epilogue. Previously, the TMEM stage was released only after the epilogue finished. That wait is unnecessary because the epilogue runs entirely in registers. TMEM can be released as soon as the accumulator has been loaded into registers, before any arithmetic happens. This cuts total kernel latency by 20% in prefill, where the faster NVFP4 compute matters most. It also raises overall prefill throughput by 7%.

### NVFP4 indexer DCP (decode context parallelism)

In low-latency serving, each GPU scores its own index head over the full context. Across the four GPUs, that means every indexer layer reads the full index cache four times. Indexer DCP splits the work by context instead. Each GPU scores all four heads over a quarter of the pages. It then sends its local top-16 for each head over NVLink to the GPU that owns that head, which merges the candidates into the final top-16. The selection remains exact, and the indexer runs up to 3.3× faster per sparse layer at 512K context on 4× NVIDIA GB300 GPUs.

### QKV GEMM + norm + RoPE + KV insert fusion

MiniMax-M3.1's QKV and indexer projections used to write the full QKV tensor to HBM. A second kernel then read it back for norm, RoPE, and KV cache insertion. We fused the whole chain into a single GEMM kernel whose epilogue processes the FP32 accumulator head by head. It applies norm and RoPE to the query and index query. It also quantizes K, V, and the index key to NVFP4 and writes them directly into their caches. The QKV tensor never touches HBM. On GB300, prefill batches of 512 or more tokens run 1.1–1.6× faster, and P50 time-to-first-token (TTFT) drops by about 19% at TP4.

### FlashInfer PrimiTS for the DSpark drafter

MiniMax-M3.1 uses DSpark as its speculative decoding drafter, which runs non-causal GQA with a 4096-token sliding window. Previously, vLLM ran the drafter on FlashAttention 4, which is not optimized for decode shapes. We switched it to FlashInfer's PrimiTS backend. PrimiTS is 1.3–2× faster than FlashAttention 4 and cuts overall forward time by 5%.

### Agentic workload benchmark on NVIDIA GB300

vLLM delivers strong performance on agentic serving workloads. We benchmark our day-0 image on NVIDIA GB300 with the SemiAnalysis AgentX benchmark (detailed in our [previous post](https://vllm.ai/blog/2026-09-08-vllm-agentx)). At the latency-sensitive end of the frontier, MiniMax-M3.1-Flash holds 418.9 tokens per second per user. Further along the frontier, it keeps each user above 200 tokens per second while serving 38.6K total tokens per second per GPU. At the throughput end, it reaches 153.8K tokens per second per GPU, about 24× the most interactive point, while each user still sees about 127 tokens per second.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-m31-agentx-gb300.html" title="MiniMax-M3.1-Flash AgentX Pareto frontier on NVIDIA GB300" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 640px; border: 0; overflow: hidden;"></iframe>

*Figure 4: AgentX Pareto frontier for MiniMax-M3.1-Flash on NVIDIA GB300, with 7 speculative tokens and synthetic acceptance lengths of 3.16–3.26 tokens (including the first token). Hover a point for its metrics.*

### NVIDIA Rubin optimized MSA

We also optimized this model on NVIDIA Rubin GPUs. To mitigate the NVFP4 to FP8 conversion overhead, we convert NVFP4 directly to FP16 and run the score MMA in FP16 for the indexer. This cuts dequant instructions by about a third, and the scores become exact rather than carrying the E4M3 re-rounding to run FP8 MMA.

With ≤4 queries, the one-head indexer decode kernel goes further: it applies each block scale once per MMA result with SM107's packed `fma.rn.f32x2.f16x2`, instead of once per K value. This also enables more latency hiding by running three CTAs per SM. Rubin's larger shared memory (328 KB vs 228 KB on Blackwell) and extra Tensor Memory columns (576 vs. 512 on Blackwell) make that possible.

Finally, because Rubin has more SMs than Blackwell, the work planners size their tiles from the SM count, so the work is spread evenly across all 212 of Rubin's SMs.

Prefill sparse attention also benefits from locality domain optimizations. We allocate the partial buffers through the CUDA virtual memory API so that 2 MB pages alternate between the two domains, and the combined kernel hands each tile to a CTA whose SM sits in the domain that holds that tile's data. On Rubin the prefill & decode attention kernels use dense FP8 MMA with K=64 per instruction, which halves the MMA instruction count and runs at twice Blackwell's FP8 rate.

### Agentic workload benchmark on NVIDIA Rubin

On NVIDIA Rubin, MiniMax-M3.1-Flash pushes the frontier further on the same benchmark. At the latency-sensitive end, it holds 724.9 tokens per second per user. Further along the frontier, it keeps each user above 200 tokens per second while serving 157.3K total tokens per second per GPU. At the throughput end, it reaches 191.5K tokens per second per GPU while each user still sees about 134 tokens per second, about 1.5× GB300 at the same per-user rate.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-m31-agentx-rubin.html" title="MiniMax-M3.1-Flash AgentX Pareto frontier on NVIDIA Rubin" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 640px; border: 0; overflow: hidden;"></iframe>

*Figure 5: AgentX Pareto frontier for MiniMax-M3.1-Flash on NVIDIA Rubin. Hover a point for its configuration and metrics.*

## AMD GPU support

On AMD GPUs, we bring NVFP4 KV cache support, so AMD deployments can also use the low-precision format to cut KV cache memory. For decode, we add indexer DCP to speed up long-context decode, and a mono kernel that speeds up low-latency inference. For prefill, sequence parallelism and dual-batch overlap cut communication inside the engine, and a new KV cache layout lets each KV block move to the decode node in a single transfer. The MSA and mono kernels are built in [FlyDSL](https://github.com/rocm/flydsl), AMD's Python DSL for GPU kernels, with help from the FlyDSL team.

### NVFP4 MSA kernel

MiniMax-M3.1 is trained with an NVFP4 KV cache (QAT) for both the main attention and the indexer, but CDNA4 has no native instruction to dequantize NVFP4. To keep the MSA kernels fast, we turn dequantization into a table lookup. There are only 256 possible E4M3 scales and eight E2M1 magnitudes, so we precompute every scaled value, already rounded to FP8, into a 2 KB table in LDS. One `v_perm_b32` byte permute plus one XOR for the sign bits then converts four FP4 values to FP8. The kernel also dequantizes a shared-prefix page only once for up to 64 requests that share it. Together, these make sparse decode attention up to 2.5× faster than converting the selected blocks to FP8 and running the FP8 attention kernel.

### NVFP4 indexer DCP

We use the same indexer decode context parallelism on AMD as on NVIDIA, and on CDNA4 it brings an extra win. Without DCP, each GPU at TP4 scores its own index head over the full context, so the index cache is read four times. It also leaves the matrix cores underused. A DSpark-3 decode step has only 4 query rows per request, which fills just 4 of the 16 columns of the 16×16×128 FP8 MFMA. With DCP, each GPU scores all four heads over a quarter of the context, so 4 heads × 4 rows fill all 16 columns. A single kernel exchanges the index queries, scores its blocks, and writes each head's scores directly into the owning GPU's memory over Infinity Fabric. A device-side handshake keeps this working inside HIP graphs. The owner then picks the top 16 as before, and because the scores are bit-for-bit identical to the single-GPU version, the selection is exact. On 4× MI355X, the indexer runs up to 2.8× faster per sparse layer on 45 requests of 30K–210K tokens.

### SP + DBO prefill and one-read KV transfer

At TP4, long-prompt prefill on MI355X is bound by communication, both between GPUs inside a node and between the prefill and decode nodes. Inside the engine, dual-batch overlap (DBO) hides the tensor-parallel collectives behind the other microbatch's compute. Sequence parallelism (SP) cuts the traffic itself. Each GPU keeps a quarter of the tokens, the attention output is exchanged all-to-all in MXFP8 so each GPU runs the full `o_proj` on its own tokens, and the MoE output is reduce-scattered. This drops traffic per GPU from about 36 KB to 20 KB per token per layer, and norms, quantization, and the router run on only a quarter of the tokens.

A faster prefill engine only helps if the KV cache can also reach the decode node quickly. vLLM's layer-by-layer KV cache layout splits each KV block across 125 small pages, and each page needs its own NIXL RDMA read. We switch to a block-outer layout that stores each block in one contiguous 1.82 MB row, so NIXL moves it in a single read. Combined, the engine and KV transfer changes make prefill batches of 16K+ tokens 1.8–2.1× faster on 4× MI355X.

### Low-latency mono kernel

At low concurrency, decode on AMD GPUs is launch-bound: kernel launch overhead takes more time than the math. Mono decode fixes this by running each sparse MoE decoder layer as a single FlyDSL kernel launch. That one launch covers everything from the QKV projection and sparse attention to the tensor-parallel all-reduces and the MXFP4 MoE, replacing roughly 1,170 kernel launches. On MI350X at TP4, this makes the decode step up to 2.3× faster and doubles 8K-in/1K-out throughput at low concurrency. On an agentic-coding trace, it raises interactivity by 50–65%.

### Agentic workload benchmark on AMD MI355X

Our AMD implementation also delivers strong performance per cost. On the InferenceX AgentX benchmark, MiniMax-M3.1-Flash holds 331.0 tokens per second per user at the latency-sensitive end of the frontier. Further along the frontier, it keeps each user above 170 tokens per second while delivering 34.6M total tokens per dollar of TCO. At the cost-efficient end, it reaches 133.3M tokens per dollar, nearly 10× the most interactive point, while each user still sees about 79 tokens per second.
<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-m31-agentx-mi355x.html" title="MiniMax-M3.1-Flash AgentX Pareto frontier on AMD MI355X" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 820px; border: 0; overflow: hidden;"></iframe>

*Figure 6: AgentX Pareto frontier for MiniMax-M3.1-Flash on AMD MI355X, in total tokens per dollar of TCO. c1 and c4 use a synthetic acceptance length of 3.2 tokens with 7 speculative tokens; the other points use 2.45 tokens with 3 speculative tokens. Hover a point for its metrics.*

## Day-0 RL support

We run RL on MiniMax-M3.1 starting from its NVFP4 checkpoint and train only rank-32 LoRA adapters. The veRL trainer quantizes the base model the same way the engine serves it, with NVFP4 on the routed experts and MXFP8 on the other projections, and syncs only the adapters to the engine after each step. On the engine side, expert LoRA runs inside FlashInfer's TRTLLM NVFP4 MoE kernel, so the base experts still run on the normal serving kernel.

Before training, we checked that the trainer and the engine agree in three ways. The engine receives exactly the weights the trainer sends, both route each token to the same experts (routing replay), and both assign close logprobs to the same tokens.

The remaining gap is corrected with token-level importance sampling (TIS), with the importance ratio capped at 2. Because each batch gets a single optimizer step, the trainer uses the rollout logprobs as the old logprobs and skips the extra forward pass that would recompute them.

On DAPO-Math-17K with 16 GB300 GPUs, 200 GRPO steps raise AIME-2024 accuracy from 36.3% to 67.5%. Over the same run, training reward rises from −0.06 to 0.67, and train/rollout KL stays low, falling from 0.040 to 0.024.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-m31-rl-training.html" title="Day-0 RL on MiniMax-M3.1: AIME-2024 accuracy, train/rollout KL, and training reward over 200 GRPO steps" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 520px; border: 0; overflow: hidden;"></iframe>

*Figure 7: Day-0 RL on MiniMax-M3.1-Flash with veRL on DAPO-Math-17K (16× GB300, 200 GRPO steps): AIME-2024 accuracy, training reward, and train/rollout KL. Use the tabs to switch between them.*

## Acknowledgments

We extend our sincere gratitude to the MiniMax team for open-sourcing MiniMax-M3.1-Flash. We also thank Inferact, NVIDIA, and AMD (Hongxia Yang for the mono kernel, Felix Marty, Andy Luo, Felix Li, Hattie Wu, Gilbert Lei, and Peng Sun) for delivering day-0 support for the model in vLLM, as well as the FlyDSL team for their help with the AMD kernels and the vLLM community.

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
