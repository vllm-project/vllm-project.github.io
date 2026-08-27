---
layout: post
title: "vLLM × HPC-Ops: High-Performance Attention and MoE Backends from Tencent Hunyuan"
author: "Tencent Hunyuan AI Infra Team and vLLM Team"
summary: "How HPC-Ops integrates Hopper-optimized attention and FP8 MoE backends into vLLM for Tencent Hunyuan Hy3, improving mixed-length decode, MoE latency, TTFT, and TPOT on NVIDIA H20."
image: /assets/figures/2026-07-06-vllm-hpc-ops/dynamic-partitioning.png
social_image: /assets/figures/2026-07-06-vllm-hpc-ops/dynamic-partitioning.png
read_time_minutes: 12
tags:
  - performance
  - attention
  - moe
  - hpc-ops
---

## **TL;DR**

The Attention and MoE kernels from **HPC-Ops** — the production operator library built by the Tencent Hunyuan AI Infra team — are now in vLLM `main` branch as first-class backends ([AttentionPR \#46020](https://github.com/vllm-project/vllm/pull/46020), [MoE PR \#45924](https://github.com/vllm-project/vllm/pull/45924)). Both are optimized for NVIDIA's Hopper architecture, with the strongest results on H20:

* **Attention:** a per-step, load-balanced decode scheduler plus a fused RoPE \+ QK-Norm \+ KV-write prologue. On mixed-length decode, up to **2.95×** over a static split-KV schedule and **2.25×** on average over FlashInfer and FlashAttention.  
* **MoE:** a fully fused, low-latency FP8 MoE pipeline. On average **1.59×** at TP8 / EP1 and **1.21×** at TP1 / EP8 over Triton and CUTLASS, with matched output quality.

End-to-end on Hy3 across 8× H20, the two backends together cut TTFT by about **24%** and TPOT by about **17%** versus the vLLM default backend.

Both plug into stock vLLM through its backend interfaces — no source changes and no long-lived fork.

This post covers three things: what HPC-Ops is, how the two upstreamed backends are designed and integrated, and how they perform on H20.

## **Why This Matters**

Production LLM serving no longer looks like the uniform, single-turn batches most kernels were first tuned for. Real traffic is dynamic and mixed-length, models are increasingly MoE with long context, and agentic workloads push both harder. At this scale, much of the latency is decided by how well the kernels schedule work across the GPU and move data between stages, not by raw matmul throughput alone. In attention decode, a fixed split-KV schedule stalls on the longest request in a mixed batch while leaving compute idle on the short ones. In MoE, the per-expert GEMMs are small, and a conventional pipeline gathers tokens into per-expert buffers, pays launch overhead at every stage, and moves intermediates through HBM in between.

vLLM already gives the community a fast, flexible serving engine; the remaining latency and throughput come down to how well the attention and MoE kernels absorb this messy, real-world traffic. That is what HPC-Ops targets — an operator library hardened in Tencent's large-scale production serving, and the same kernels that serve Hy3 are now upstreamed into vLLM as first-class Attention and MoE backends.

## **A Quick Word on Hy3-series models**

Hy3 is Tencent Hunyuan's Mixture-of-Experts model for agentic execution, coding, and long-horizon reasoning. Activating just 21B of its 295B parameters, it reaches some of the strongest agent capabilities in its size class — rivaling open-source flagships 2–3× larger — while substantially cutting hallucination for more reliable multi-turn use. Under the hood it uses 192 experts with top-8 routing, GQA attention (64 heads, 8 KV heads, head dim 128), a 256K context window, and a 3.8B MTP layer for speculative decoding; it ships in BF16 and FP8 (Hy3-FP8).

This post is intentionally not about the model — it is about the kernels that serve it, which we turn to next.

## **HPC-Ops: A Production Operator Library, Now in vLLM**

[HPC-Ops](https://github.com/Tencent/hpc-ops) is an open-source operator library for LLM inference, built and maintained by the Tencent Hunyuan AI Infra team. It focuses on the hot paths that dominate real serving latency and throughput — attention, MoE, GEMM, sampling, normalization, and communication-compute fusion — with native BF16 and FP8 support and a clean Python API meant to drop into inference frameworks. The kernels are optimized for NVIDIA's Hopper architecture, with especially strong results on H20.

These kernels are proven in Tencent's own large-scale production serving of Hunyuan. In this release, two of them have been upstreamed into vLLM as first-class backends:

| vLLM backend | What it optimizes | Precision | Merged in |
| :---- | :---- | :---- | :---- |
| Attention | Load-balanced decode \+ fused RoPE/QK-Norm prologue | BF16 / FP8 | [PR \#46020](https://github.com/vllm-project/vllm/pull/46020) |
| Fused MoE | Fully fused low-latency MoE pipeline | FP8 | [PR \#45924](https://github.com/vllm-project/vllm/pull/45924) |

The rest of this post focuses on these two backends.

## **Attention Backend: Dynamic Load-Balanced Scheduling**

### **The challenge: mixed-length decode in every batch**

In decode, every token generation step runs attention over a request's full KV cache. A request that has accumulated 16K tokens of context costs roughly 16× more compute than one that just started at 1K. In production serving, output lengths are unpredictable and continuous batching keeps requests at very different stages of generation in the same kernel launch — so a single batch routinely mixes very short and very long sequences.

Existing decode kernels map work to CTAs through a fixed launch grid, keyed by KV head, request, and a split-KV chunk index — and that split-KV degree must be uniform across all requests, which forces a choice between two bad options. Fix the number of splits, and the longest sequence dominates: short-request CTAs finish in a fraction of the time and sit idle. Fix the chunk size instead, and the split count must be set to the maximum any request needs, so short requests get padded with empty chunks that launch, find no work, and exit — wasting scheduling slots. Either way, total kernel time is dictated by the heaviest CTA while the others stall, leaving SM cycles on the table.

### **The solution: a per-step, load-balanced decode scheduler**

The HPC-Ops attention backend replaces the fixed grid with a flat, persistent design that adapts to the batch's actual length distribution rather than a launch-time split policy, built in three stages.

* **Assign.** A lightweight assign kernel slices every KV sequence into uniform 64-token tiles. The total tile count across all heads and requests is divided by the number of available CTAs to determine a per-CTA budget — the bucket size. Tiles are traversed in head-major, batch-minor order and filled into CTA buckets sequentially: once a CTA's bucket is full, subsequent tiles spill into the next CTA. A long sequence is therefore split across multiple CTAs in proportion to its length, while a short sequence contributes only a handful of tiles and does not monopolize a CTA. A minimum workload floor per CTA prevents over-splitting when total work is small, ensuring the number of chunks stays manageable and the downstream combine cost does not outweigh the scheduling benefit. The resulting task map is computed once per decode step and reused by every transformer layer in that step, so its overhead is amortized to near zero.  
* **Compute and combine.** A persistent kernel grid then runs: each CTA loops over its assigned task bin, pulling a task descriptor, computing attention for that chunk, writing partial output and log-sum-exp to a split buffer, then advancing to the next task until it hits a terminator. Because the grid is persistent, there is no relaunch overhead between tasks and no idle gap between waves — every SM stays saturated for the full kernel duration. A final lightweight combine kernel reads the chunk count for each (head, request) pair and reduces the per-chunk partials into the final BF16 output.

The net effect is that all CTAs carry roughly equal workloads and finish at nearly the same time, regardless of how skewed the sequence-length distribution is. The long-tail stall inherent in static schedules is eliminated, and GPU cycles that were previously wasted on idle waiting are converted into useful compute.

![Dynamic Partitioning: Uniform Tiling and Balanced Bucketing](/assets/figures/2026-07-06-vllm-hpc-ops/dynamic-partitioning.png)

### **A fused attention prologue**

Before attention runs, each layer normally applies QK-Norm, RoPE, and a KV-cache write — plus, in FP8, a query quantization — as separate, memory-bound steps. HPC-Ops fuses them into a single op (`HpcRopeNorm`): starting from the fused QKV projection, it applies QK-Norm and RoPE in the model's required order (Hy3 normalizes before RoPE), writes K and V straight into the paged cache, and, in FP8, emits a per-token, per-head FP8 query with its scale so the attention kernel never re-quantizes. One kernel replaces those separate launches and their HBM round-trips on every layer's attention prologue, in both prefill and decode.

### **Integrating with vLLM**

The HPC-Ops attention APIs are integrated into vLLM as a native attention backend, alongside existing backends such as FlashAttention and FlashInfer. Specifically, `HpcAttentionBackend` inherits from vLLM's `AttentionBackend` base class and is registered through the standard backend registration mechanism.

## **MoE Backend: A Fused, Low-Latency FP8 MoE Pipeline**

### **The challenge: small expert GEMMs and the overhead around them**

MoE inference has two very different regimes. At high throughput with large batches, the expert GEMMs are large and compute-bound, and existing implementations generally perform fine there. Low-latency decode is the opposite: each expert receives only a handful of tokens, so the expert GEMMs are small and memory-bound. Kernels tuned for large matmuls underfill the GPU on these shapes, and because the number of tiles each expert produces varies and shifts from step to step, those small tiles are hard to spread evenly across the GPU.

The work around the GEMMs adds to that. A conventional MoE path is a chain of separate kernels: route tokens, gather them into per-expert buffers, Gate-Up GEMM, activation and quantization, Down GEMM, and a top-k weighted reduction back to token positions. The gather materializes a gathered-token tensor in HBM before any matmul starts, and every stage pays its own kernel launch and its own HBM round-trip for intermediates. In decode, where the GEMMs are already small, all of this piles up alongside the GEMM work itself.

### **The solution: a fused FP8 MoE pipeline**

The HPC-Ops MoE backend re-architects the whole MoE path: routing and index preprocessing, the Gate-Up GEMM, activation and quantization, the Down GEMM, and the top-k weighted reduction are fused into one compact execution path, removing the redundant overhead of a multi-stage design.

* **Routing and index build.** A shared-memory counting pass assigns tokens to experts with contiguous per-expert output ranges, cutting the global-atomic pressure of large-token routing, and builds the routing indices and per-tile task map that the GEMMs consume directly.  
* **Gate-Up GEMM.** The Gate-Up GEMM reads original tokens directly through the routing index, skipping the standalone gather step. Activation and FP8 quantization then run as a separate fused kernel whose output the Down GEMM reads directly.  
* **Occupancy-first, without warp specialization.** A single warp group handles both data movement and compute, shifting memory-latency hiding from an intra-CTA software pipeline to cross-CTA hardware scheduling and raising the number of resident CTAs per SM. A persistent grid, launched to keep every SM full, then consumes the task map and spreads the small, uneven set of per-expert tiles evenly across the CTAs.  
* **PDL-chained stages.** Programmatic Dependent Launch overlaps each kernel launch with the tail of the previous one, erasing the bubbles between stages, all the way to the final top-k weighted reduction, which can also fold in the shared-expert output.

Together, these keep intermediates and launches off the critical path. The experts run in FP8, with both per-tensor and block-wise scaling, and match the output quality of the baselines.

### **Integrating with vLLM**

The HPC-Ops Fused MoE APIs are integrated into vLLM as a native MoE backend, alongside existing backends such as DeepGEMM and Triton. Specifically, `HPCExperts` inherits from vLLM's `FusedMoEExpertsModular` base class and is registered through the standard backend registration mechanism.

## **Using HPC-Ops Backends in vLLM**

This guide describes how to enable the [HPC-Ops](https://github.com/Tencent/hpc-ops) backends (Attention and MoE) in vLLM.

### **Install**

Before getting started, install **HPC-Ops** from source:

```bash
git clone https://github.com/Tencent/hpc-ops.git
cd hpc-ops

# Build and install the wheel package
make wheel
python3 -m pip install dist/*.whl
```

### **Quick start**

The HPC-Ops Attention backend currently supports only the **Hy3-series** models.

To launch a vLLM server for the standard Hy3 model with the HPC-Ops Attention backend, run:

```bash
vllm serve tencent/Hy3 \
    --tensor-parallel-size 8 \
    --attention-backend HPC_ATTN
```

For the **Hy3-FP8** model, a few additional options are required:

```bash
vllm serve tencent/Hy3-FP8 \
    --tensor-parallel-size 8 \
    --attention-backend HPC_ATTN \
    --kv-cache-dtype fp8_e4m3 \
    --block-size 64
```

**Tip:** To enable the HPC-Ops Attention backend for a custom model, replace `rope_norm` with `HpcRopeNorm` in the model's `forward` method. See PR [\#46020](https://github.com/vllm-project/vllm/pull/46020) for reference.

The HPC-Ops MoE backend supports **FP8 models only**.

To launch a vLLM server with the HPC-Ops MoE backend, run:

```bash
vllm serve tencent/Hy3-FP8 \
    --tensor-parallel-size 8 \
    --moe-backend hpc
```

### **Hardware support**

The HPC-Ops backends are currently supported only on NVIDIA Hopper-architecture GPUs, and deliver the best performance on the H20.

## **Performance on H20**

### **Fused MoE: HPC-Ops vs Triton / CUTLASS**

We benchmarked the HPC-Ops MoE backend against the Triton and CUTLASS MoE backends under the Hy3 model configuration, at both TP8 / EP1 and TP1 / EP8 settings. Averaged over batch sizes, HPC-Ops is 1.59× faster than the best baseline at TP8 / EP1 and 1.21× at TP1 / EP8, with the largest gains at the small-to-mid batch sizes that dominate low-latency decode.

Table 1: FusedMoE latency (µs) across batch sizes at TP8 / EP1 (expert weights sharded across 8 ranks)

| Batch | HPC-Ops (µs) | Triton (µs) | CUTLASS (µs) |
| :---- | :---- | :---- | :---- |
| 4 | 42.0 | 56.4 | 74.5 |
| 16 | 85.7 | 124.2 | 209.2 |
| 32 | 124.0 | 184.3 | 275.6 |
| 64 | 147.2 | 374.9 | 330.3 |
| 128 | 161.5 | 302.9 | 345.3 |
| 256 | 170.1 | 310.9 | 351.6 |
| 512 | 194.5 | 331.6 | 369.2 |
| 1024 | 281.4 | 652.7 | 438.3 |
| 2048 | 491.8 | 731.5 | 794.4 |
| 4096 | 872.0 | 1366.0 | 1230.7 |
| 8192 | 1695.0 | 2216.8 | 2362.9 |
| 16384 | 3241.9 | 4329.1 | 4364.4 |

Table 2: FusedMoE latency (µs) across batch sizes at TP1 / EP8 (experts sharded across 8 ranks)

| Batch | HPC-Ops (µs) | Triton (µs) | CUTLASS (µs) |
| :---- | :---- | :---- | :---- |
| 4 | 118.6 | 147.4 | 140.4 |
| 8 | 136.7 | 192.8 | 170.7 |
| 16 | 149.8 | 198.4 | 263.5 |
| 32 | 153.6 | 214.6 | 264.4 |
| 64 | 166.5 | 358.1 | 266.8 |
| 128 | 213.5 | 251.7 | 272.6 |
| 256 | 386.2 | 454.9 | 493.5 |
| 512 | 705.5 | 691.7 | 741.7 |
| 1024 | 1342.6 | 1369.1 | 1359.1 |
| 2048 | 2513.9 | 2668.7 | 2530.4 |

![HPC-Ops FusedMoE on H20 — Hy3](/assets/figures/2026-07-06-vllm-hpc-ops/fused-moe-latency.png)

### **Decode under mixed-length batches: dynamic vs static scheduling**

The attention backend's headline win is decode over mixed-length batches. To isolate the scheduler, we sweep FP8 decode from uniform to highly skewed KV-length distributions (label A×B \= A requests at KV length B) and compare HPC-Ops dynamic scheduling against a static split-KV schedule, FlashInfer, and FlashAttention. The advantage over static grows with skew, from parity on small uniform batches to 2.95× on a 1×128K \+ 31×4K mix. Across these cases, dynamic scheduling is on average 2.25× faster than the best of FlashInfer and FlashAttention.

Table 3: Decode latency (ms) across KV-length distributions

| Decode scenario | HPC-Ops dynamic (ms) | HPC-Ops static (ms) | FlashInfer (ms) | FlashAttention (ms) | Dynamic vs static |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 64×0.5K | 0.013 | 0.013 | 0.050 | 0.025 | 1.00× |
| 64×4K | 0.033 | 0.043 | 0.221 | 0.095 | 1.32× |
| 32×0.125K \+ 32×4K | 0.020 | 0.033 | 0.119 | 0.053 | 1.59× |
| 2×32K \+ 30×4K | 0.032 | 0.056 | 0.169 | 0.094 | 1.76× |
| 1×64K \+ 15×4K | 0.042 | 0.097 | 0.118 | 0.065 | 2.32× |
| 1×128K \+ 31×4K | 0.063 | 0.186 | 0.220 | 0.097 | 2.95× |

![Decode Attention on H20 — Hy3: dynamic vs static scheduling](/assets/figures/2026-07-06-vllm-hpc-ops/decode-dynamic-vs-static.png)

### **Attention: HPC-Ops vs FlashAttention / Triton / FlashInfer**

We further benchmarked the HPC-Ops Attention backend against FlashAttention, Triton, and FlashInfer across prefill, extend, and decode shapes, using vLLM's attention benchmark. Across these shapes, HPC-Ops is at parity with or faster than the fastest of the three in nearly every case.

Table 4: Attention latency (ms) vs FlashAttention, Triton, and FlashInfer

| Batch Spec | Type | Batch Size | HPC-Ops (ms) | FlashAttention (ms) | Triton (ms) | FlashInfer (ms) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| q512 | prefill | 1 | 0.047 | 0.069 | 0.123 | 0.070 |
| q1ks2k | extend | 1 | 0.406 | 0.431 | 1.132 | 0.431 |
| q2k | prefill | 1 | 0.530 | 0.574 | 1.525 | 0.609 |
| q4k | prefill | 1 | 2.002 | 2.093 | 5.816 | 2.144 |
| q8k | prefill | 1 | 7.883 | 7.957 | 22.702 | 8.084 |
| 2q1ks4k | extend | 2 | 1.835 | 1.830 | 5.046 | 1.829 |
| 8q1s1k | decode | 8 | 0.019 | 0.031 | 0.035 | 0.021 |
| 16q1s2k | decode | 16 | 0.054 | 0.098 | 0.106 | 0.052 |
| 32q1s1k | decode | 32 | 0.057 | 0.102 | 0.080 | 0.058 |
| 64q1s4k | decode | 64 | 0.299 | 0.620 | 0.510 | 0.340 |

### **End-to-end: Hy3 on 8× H20**

Finally, we evaluated the end-to-end (E2E) performance of the Hy3 model with the HPC-Ops MoE and Attention backends against the vLLM default backend on 8× NVIDIA H20 GPUs. Across every test case, the HPC-Ops backend consistently outperforms the vLLM default backend, delivering substantial reductions in both TTFT and TPOT. TTFT drops by about 24% on average, and TPOT by about 17% on average, growing to about 30% at the largest batch size.

Table 5: TPOT across different batch sizes (output length \= 4K)

| Batch Size | Baseline TPOT (ms) | HPC TPOT (ms) | Improvement |
| :---- | :---- | :---- | :---- |
| 1 | 8.00 | 7.76 | \+3.0% |
| 4 | 11.14 | 10.67 | \+4.2% |
| 8 | 13.49 | 11.31 | \+16.2% |
| 16 | 17.98 | 13.56 | \+24.6% |
| 32 | 24.13 | 18.32 | \+24.1% |
| 64 | 31.10 | 21.90 | \+29.6% |

Table 6: TTFT across different batch sizes (input length \= 8k, disable Chunked Prefill, disable Prefix Caching)

| Batch Size | Baseline TTFT (ms) | HPC TTFT (ms) | Improvement |
| :---- | :---- | :---- | :---- |
| 1 | 565.69 | 431.00 | \+23.8% |
| 4 | 1920.15 | 1471.43 | \+23.4% |
| 8 | 3948.22 | 3035.44 | \+23.1% |
| 16 | 7807.18 | 5885.63 | \+24.6% |

Table 7: TTFT across different input lengths (batch size \= 16, disable Chunked Prefill, disable Prefix Caching)

| Input Length | Baseline TTFT (ms) | HPC TTFT (ms) | Improvement |
| :---- | :---- | :---- | :---- |
| 2k | 1792.62 | 1363.13 | \+24.0% |
| 4k | 3704.27 | 2886.40 | \+22.1% |
| 8k | 7807.12 | 5893.93 | \+24.5% |

## **What's Next**

This is just the start of a longer collaboration with the vLLM community. We'll keep working with vLLM maintainers and contributors to improve and extend these capabilities and to upstream further work as it matures. Feedback, issues, and benchmarks are very welcome, and we look forward to building open, high-performance inference together.

## **Acknowledgements**

We would like to thank the many people across teams who worked together to bring these backends to vLLM:

* **Tencent Hunyuan AI Infra** — for building and optimizing the HPC-Ops Attention and MoE kernels and contributing them to vLLM as backends. Sethran Liu, Chase Shao, Shengy Wei, Theo Cheng, Ryann Xue, Lando Jiang, Looper Zhao, Haank Lin, Aiden Ren, Lehua Ding, Chengv Jiang, Steven Kuang, Liqi He, Kipper Gong, Reedlau Liu, Raccoon Liu, Dick Zhu.  
* **Tencent Network Platform Department** — for the close collaboration on communication optimization. Xuan Zhang, Haoran Zhao, Yuanyuan Gong, Yadong Liu, Jinzhu Wang, Yinben Xia, Xiang Li, Quan Wen, Zekun He.  
* **vLLM/Inferact** — for the open backend interfaces, reviews, and design discussions. Kaichao You, Yongye Zhu, Yifan Qiao.  
* **NVIDIA** — for the close collaboration on kernel and performance optimization. Yuanhang Sun, Perkz Zheng, Yuxi Chi, Jiang Shao, Jun Gu, Meng Wang, River Liu, Gary Ji, Chandler Zhou.

We also thank the broader open-source kernel community whose work this builds on and measures against, including NVIDIA CUTLASS/CuTe, TensorRT-LLM, FlashInfer, FlashAttention, and Triton.
