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

The Attention and MoE kernels from [HPC-Ops](https://github.com/Tencent/hpc-ops), the production operator library built by the Tencent Hunyuan AI Infra team, are now available in vLLM `main` branch as first-class backends through [Attention PR #46020](https://github.com/vllm-project/vllm/pull/46020) and [MoE PR #45924](https://github.com/vllm-project/vllm/pull/45924).

Both backends are optimized for NVIDIA Hopper architecture, with the strongest measured results on H20. The Attention backend targets mixed-length decode with a per-step, load-balanced scheduler and a fused RoPE + QK-Norm + KV-write prologue. The MoE backend provides a fully fused, low-latency FP8 MoE pipeline. Together, they plug into stock vLLM through standard backend interfaces, without source changes or a long-lived fork.

## TL;DR

HPC-Ops adds two production-hardened backends to vLLM:

- **Attention:** a per-step, load-balanced decode scheduler plus a fused RoPE + QK-Norm + KV-write prologue. On mixed-length decode, it reaches up to **2.95x** speedup over a static split-KV schedule and averages **2.25x** faster than FlashInfer and FlashAttention on the tested mixed-length cases.
- **MoE:** a fully fused, low-latency FP8 MoE pipeline. On average, it is **1.59x** faster at TP8 / EP1 and **1.21x** faster at TP1 / EP8 than the best Triton or CUTLASS baseline, with matched output quality.
- **End-to-end Hy3 serving:** on 8x H20, the two backends together reduce TTFT by about **24%** and TPOT by about **17%** on average versus the vLLM default backend.
- **Integration path:** the backends are selected through vLLM's backend interfaces. The Attention backend is integrated as `HpcAttentionBackend`, and the MoE backend is integrated through `HPCExperts`.

This post covers what HPC-Ops is, how the two upstreamed backends are designed and integrated, how to enable them, and how they perform on H20.

## Why This Matters

Production LLM serving no longer looks like the uniform, single-turn batches that many kernels were first tuned for. Real traffic is dynamic and mixed-length, models are increasingly MoE with long context, and agentic workloads push both harder. At this scale, latency is often decided by how well kernels schedule work across the GPU and move data between stages, not by raw matmul throughput alone.

In attention decode, a fixed split-KV schedule stalls on the longest request in a mixed batch while leaving compute idle on shorter requests. In MoE, the per-expert GEMMs are small, and a conventional pipeline gathers tokens into per-expert buffers, pays launch overhead at every stage, and moves intermediates through HBM between stages.

vLLM already provides a fast, flexible serving engine. The remaining latency and throughput gains come from making attention and MoE kernels absorb messy production traffic. That is what HPC-Ops targets: an operator library hardened in Tencent's large-scale Hunyuan serving, now upstreamed into vLLM as first-class Attention and MoE backends.

## Hy3 and HPC-Ops

Hy3 is Tencent Hunyuan's Mixture-of-Experts model for agentic execution, coding, and long-horizon reasoning. It activates 21B of its 295B parameters and is designed for reliable multi-turn use. The model uses 192 experts with top-8 routing, GQA attention with 64 query heads and 8 KV heads, head dimension 128, a 256K context window, and a 3.8B MTP layer for speculative decoding. It ships in BF16 and FP8 variants, including Hy3-FP8.

This post is not a model announcement. It focuses on the kernels used to serve Hy3 efficiently in vLLM.

[HPC-Ops](https://github.com/Tencent/hpc-ops) is an open-source operator library for LLM inference. It focuses on the hot paths that dominate serving latency and throughput: attention, MoE, GEMM, sampling, normalization, and communication-compute fusion. The library supports BF16 and FP8 kernels and exposes a Python API designed to integrate with inference frameworks.

In this release, two HPC-Ops components have been upstreamed into vLLM:

| vLLM backend | What it optimizes | Precision | Merged in |
| --- | --- | --- | --- |
| Attention | Load-balanced decode + fused RoPE/QK-Norm prologue | BF16 / FP8 | [PR #46020](https://github.com/vllm-project/vllm/pull/46020) |
| Fused MoE | Fully fused low-latency MoE pipeline | FP8 | [PR #45924](https://github.com/vllm-project/vllm/pull/45924) |

## Attention Backend: Dynamic Load-Balanced Scheduling

### The Challenge: Mixed-Length Decode in Every Batch

In decode, every token generation step runs attention over a request's full KV cache. A request with 16K tokens of accumulated context costs roughly 16x more compute than one with 1K tokens. In production serving, output lengths are unpredictable, and continuous batching keeps requests at very different stages of generation inside the same kernel launch. A single batch routinely mixes very short and very long sequences.

Existing decode kernels usually map work to CTAs through a fixed launch grid keyed by KV head, request, and split-KV chunk index. The split-KV degree must be uniform across all requests, which forces a choice between two poor options. If the number of splits is fixed, the longest sequence dominates and short-request CTAs finish early. If the chunk size is fixed instead, the split count must be set to the maximum any request needs, so short requests receive empty chunks that launch, find no work, and exit.

In both cases, total kernel time is dictated by the heaviest CTA while other CTAs stall.

### The Solution: Per-Step Balanced Buckets

The HPC-Ops Attention backend replaces the fixed grid with a flat, persistent design that adapts to the batch's actual length distribution.

First, a lightweight assign kernel slices every KV sequence into uniform 64-token tiles. The total tile count across all heads and requests is divided by the number of available CTAs to determine a per-CTA budget, or bucket size. Tiles are traversed in head-major, batch-minor order and filled into CTA buckets sequentially. Once a CTA's bucket is full, later tiles spill into the next CTA.

A long sequence is therefore split across multiple CTAs in proportion to its length, while a short sequence contributes only a few tiles and does not monopolize a CTA. A minimum workload floor per CTA prevents over-splitting when total work is small, keeping downstream combine overhead under control. The resulting task map is computed once per decode step and reused by every transformer layer in that step, so the assignment overhead is amortized.

Next, a persistent kernel grid runs the assigned work. Each CTA loops over its task bin, pulls a task descriptor, computes attention for that chunk, writes partial output and log-sum-exp to a split buffer, then advances to the next task until it reaches a terminator. A final lightweight combine kernel reads the chunk count for each `(head, request)` pair and reduces the per-chunk partials into the final BF16 output.

The result is that CTAs carry roughly equal workloads and finish at nearly the same time, even when the batch contains heavily skewed sequence lengths.

<p align="center">
  <img src="/assets/figures/2026-07-06-vllm-hpc-ops/dynamic-partitioning.png" alt="Dynamic partitioning with uniform tiling and balanced CTA buckets for variable-length requests." width="100%">
  <br>
  <em>Figure 1: Variable-length requests are sliced into uniform tiles and packed into balanced CTA buckets so long requests are spread across CTAs while short requests consume only the work they need.</em>
</p>

### A Fused Attention Prologue

Before attention runs, each layer normally applies QK-Norm, RoPE, and a KV-cache write. In FP8, the path also needs query quantization. These are memory-bound steps when implemented as separate operations.

HPC-Ops fuses them into a single op, `HpcRopeNorm`. Starting from the fused QKV projection, it applies QK-Norm and RoPE in the model's required order, writes K and V directly into the paged cache, and, in FP8, emits a per-token, per-head FP8 query with its scale. The attention kernel therefore does not need to re-quantize queries. One kernel replaces multiple launches and HBM round trips in both prefill and decode.

### vLLM Integration

The HPC-Ops Attention APIs are integrated into vLLM as a native attention backend. `HpcAttentionBackend` inherits from vLLM's `AttentionBackend` base class and is registered through the standard backend registration mechanism, alongside existing backends such as FlashAttention and FlashInfer.

## MoE Backend: A Fused Low-Latency FP8 Pipeline

### The Challenge: Small Expert GEMMs

MoE inference has two different regimes. At high throughput with large batches, expert GEMMs are large and compute-bound, so existing implementations generally perform well. Low-latency decode is different. Each expert receives only a handful of tokens, so expert GEMMs are small and memory-bound. Kernels tuned for large matmuls can underfill the GPU on these shapes, and the number of tiles each expert produces varies from step to step.

The work around the GEMMs also adds overhead. A conventional MoE path is a chain of separate kernels: route tokens, gather them into per-expert buffers, Gate-Up GEMM, activation and quantization, Down GEMM, and top-k weighted reduction back to token positions. The gather materializes a gathered-token tensor in HBM before any matmul starts, and every stage pays its own kernel launch and HBM round trip.

### The Solution: A Fused FP8 MoE Path

The HPC-Ops MoE backend re-architects the whole MoE path. Routing and index preprocessing, the Gate-Up GEMM, activation and quantization, the Down GEMM, and the top-k weighted reduction are fused into one compact execution path.

- **Routing and index build:** a shared-memory counting pass assigns tokens to experts with contiguous per-expert output ranges, reducing global atomic pressure and building the routing indices and per-tile task map consumed by the GEMMs.
- **Gate-Up GEMM:** the Gate-Up GEMM reads original tokens directly through the routing index, skipping the standalone gather step. Activation and FP8 quantization run as a fused kernel whose output is consumed by the Down GEMM.
- **Occupancy-first scheduling:** a single warp group handles data movement and compute, shifting memory-latency hiding from an intra-CTA software pipeline to cross-CTA hardware scheduling and raising resident CTAs per SM. A persistent grid keeps every SM full while consuming the task map.
- **PDL-chained stages:** Programmatic Dependent Launch overlaps each kernel launch with the tail of the previous one, reducing bubbles between stages through the final top-k weighted reduction, which can also fold in shared-expert output.

Together, these choices remove intermediate traffic and launch overhead from the critical path. Experts run in FP8 with both per-tensor and block-wise scaling, while matching the output quality of the baselines.

### vLLM Integration

The HPC-Ops Fused MoE APIs are integrated into vLLM as a native MoE backend. `HPCExperts` inherits from vLLM's `FusedMoEExpertsModular` base class and is registered through the standard backend registration mechanism, alongside existing backends such as DeepGEMM and Triton.

## Using HPC-Ops Backends in vLLM

Before enabling the backend in vLLM, install HPC-Ops from source:

```bash
git clone https://github.com/Tencent/hpc-ops.git
cd hpc-ops
make wheel
python3 -m pip install dist/*.whl
```

The HPC-Ops Attention backend currently supports Hy3-series models. To launch a vLLM server for Hy3 with the HPC-Ops Attention backend:

```bash
vllm serve tencent/Hy3 \
  --tensor-parallel-size 8 \
  --attention-backend HPC_ATTN
```

For Hy3-FP8, pass the FP8 KV-cache and block-size settings:

```bash
vllm serve tencent/Hy3-FP8 \
  --tensor-parallel-size 8 \
  --attention-backend HPC_ATTN \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 64
```

To enable the HPC-Ops Attention backend for a custom model, replace `rope_norm` with `HpcRopeNorm` in the model's `forward` method. See [PR #46020](https://github.com/vllm-project/vllm/pull/46020) for the upstream integration pattern.

The HPC-Ops MoE backend supports FP8 models. To launch vLLM with the HPC-Ops MoE backend:

```bash
vllm serve tencent/Hy3-FP8 \
  --tensor-parallel-size 8 \
  --moe-backend hpc
```

The HPC-Ops backends are currently supported on NVIDIA Hopper-architecture GPUs and deliver the best measured performance on H20.

## Performance on H20

### Fused MoE: HPC-Ops vs Triton and CUTLASS

We benchmarked the HPC-Ops MoE backend against Triton and CUTLASS MoE backends under the Hy3 model configuration, at both TP8 / EP1 and TP1 / EP8. Averaged over batch sizes, HPC-Ops is 1.59x faster than the best baseline at TP8 / EP1 and 1.21x faster at TP1 / EP8. The largest gains appear at the small-to-mid batch sizes that dominate low-latency decode.

Table 1: FusedMoE latency in microseconds at TP8 / EP1, with expert weights sharded across 8 ranks.

| Batch | HPC-Ops | Triton | CUTLASS | HPC-Ops vs Best |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 42.0 | 56.4 | 74.5 | 100.0% |
| 16 | 85.7 | 124.2 | 209.2 | 100.0% |
| 32 | 124.0 | 184.3 | 275.6 | 100.0% |
| 64 | 147.2 | 374.9 | 330.3 | 100.0% |
| 128 | 161.5 | 302.9 | 345.3 | 100.0% |
| 256 | 170.1 | 310.9 | 351.6 | 100.0% |
| 512 | 194.5 | 331.6 | 369.2 | 100.0% |
| 1024 | 281.4 | 652.7 | 438.3 | 100.0% |
| 2048 | 491.8 | 731.5 | 794.4 | 100.0% |
| 4096 | 872.0 | 1366.0 | 1230.7 | 100.0% |
| 8192 | 1695.0 | 2216.8 | 2362.9 | 100.0% |
| 16384 | 3241.9 | 4329.1 | 4364.4 | 100.0% |

Table 2: FusedMoE latency in microseconds at TP1 / EP8, with experts sharded across 8 ranks.

| Batch | HPC-Ops | Triton | CUTLASS | HPC-Ops vs Best |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 118.6 | 147.4 | 140.4 | 100.0% |
| 8 | 136.7 | 192.8 | 170.7 | 100.0% |
| 16 | 149.8 | 198.4 | 263.5 | 100.0% |
| 32 | 153.6 | 214.6 | 264.4 | 100.0% |
| 64 | 166.5 | 358.1 | 266.8 | 100.0% |
| 128 | 213.5 | 251.7 | 272.6 | 100.0% |
| 256 | 386.2 | 454.9 | 493.5 | 100.0% |
| 512 | 705.5 | 691.7 | 741.7 | 102.0% |
| 1024 | 1342.6 | 1369.1 | 1359.1 | 100.0% |
| 2048 | 2513.9 | 2668.7 | 2530.4 | 100.0% |

<p align="center">
  <img src="/assets/figures/2026-07-06-vllm-hpc-ops/fused-moe-latency.png" alt="FusedMoE latency on H20 comparing HPC-Ops, Triton, and CUTLASS at TP8 EP1 and TP1 EP8." width="100%">
  <br>
  <em>Figure 2: HPC-Ops lowers FusedMoE latency across most tested batch sizes for Hy3 on H20, especially in small-to-mid batch decode regimes.</em>
</p>

### Mixed-Length Decode: Dynamic vs Static Scheduling

The Attention backend's headline win is decode over mixed-length batches. To isolate the scheduler, we swept FP8 decode from uniform to highly skewed KV-length distributions. Each label uses the form `A x B`, meaning `A` requests at KV length `B`.

The dynamic scheduler reaches parity on small uniform batches and grows to 2.95x speedup over static scheduling on a `1 x 128K + 31 x 4K` mix. Across these cases, dynamic scheduling is on average 2.25x faster than the best of FlashInfer and FlashAttention.

Table 3: Decode latency in milliseconds across KV-length distributions.

| Decode scenario | HPC-Ops dynamic | HPC-Ops static | FlashInfer | FlashAttention | Dynamic vs static |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 x 0.5K | 0.013 | 0.013 | 0.050 | 0.025 | 1.00x |
| 64 x 4K | 0.033 | 0.043 | 0.221 | 0.095 | 1.32x |
| 32 x 0.125K + 32 x 4K | 0.020 | 0.033 | 0.119 | 0.053 | 1.59x |
| 2 x 32K + 30 x 4K | 0.032 | 0.056 | 0.169 | 0.094 | 1.76x |
| 1 x 64K + 15 x 4K | 0.042 | 0.097 | 0.118 | 0.065 | 2.32x |
| 1 x 128K + 31 x 4K | 0.063 | 0.186 | 0.220 | 0.097 | 2.95x |

<p align="center">
  <img src="/assets/figures/2026-07-06-vllm-hpc-ops/decode-dynamic-vs-static.png" alt="Decode attention latency on H20 comparing HPC-Ops dynamic scheduling, HPC-Ops static scheduling, FlashInfer, and FlashAttention." width="100%">
  <br>
  <em>Figure 3: Dynamic scheduling removes the long-tail stall from skewed decode batches, with the largest gains when one long request is mixed with many shorter requests.</em>
</p>

### Attention: HPC-Ops vs FlashAttention, Triton, and FlashInfer

We also benchmarked the HPC-Ops Attention backend against FlashAttention, Triton, and FlashInfer across prefill, extend, and decode shapes using vLLM's attention benchmark. Across these shapes, HPC-Ops is at parity with or faster than the fastest of the three in nearly every case.

Table 4: Attention latency in milliseconds.

| Batch Spec | Type | Batch Size | HPC-Ops | FlashAttention | Triton | FlashInfer | HPC-Ops vs Best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q512 | prefill | 1 | 0.047 | 0.069 | 0.123 | 0.070 | 100.0% |
| q1ks2k | extend | 1 | 0.406 | 0.431 | 1.132 | 0.431 | 100.0% |
| q2k | prefill | 1 | 0.530 | 0.574 | 1.525 | 0.609 | 100.0% |
| q4k | prefill | 1 | 2.002 | 2.093 | 5.816 | 2.144 | 100.0% |
| q8k | prefill | 1 | 7.883 | 7.957 | 22.702 | 8.084 | 100.0% |
| 2q1ks4k | extend | 2 | 1.835 | 1.830 | 5.046 | 1.829 | 100.3% |
| 8q1s1k | decode | 8 | 0.019 | 0.031 | 0.035 | 0.021 | 100.0% |
| 16q1s2k | decode | 16 | 0.054 | 0.098 | 0.106 | 0.052 | 103.9% |
| 32q1s1k | decode | 32 | 0.057 | 0.102 | 0.080 | 0.058 | 100.0% |
| 64q1s4k | decode | 64 | 0.299 | 0.620 | 0.510 | 0.340 | 100.0% |

### End-to-End Hy3 on 8x H20

Finally, we evaluated end-to-end Hy3 performance with the HPC-Ops MoE and Attention backends against the vLLM default backend on 8x NVIDIA H20 GPUs. Across the tested cases, the HPC-Ops backends reduced both TTFT and TPOT. TPOT drops by about 17% on average and grows to about 30% improvement at the largest batch size. TTFT drops by about 24% on average.

Table 5: TPOT across batch sizes, with output length 4K.

| Batch Size | Baseline TPOT (ms) | HPC TPOT (ms) | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 8.00 | 7.76 | +3.0% |
| 4 | 11.14 | 10.67 | +4.2% |
| 8 | 13.49 | 11.31 | +16.2% |
| 16 | 17.98 | 13.56 | +24.6% |
| 32 | 24.13 | 18.32 | +24.1% |
| 64 | 31.10 | 21.90 | +29.6% |

Table 6: TTFT across batch sizes, with input length 8K, chunked prefill disabled, and prefix caching disabled.

| Batch Size | Baseline TTFT (ms) | HPC TTFT (ms) | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 565.69 | 431.00 | +23.8% |
| 4 | 1920.15 | 1471.43 | +23.4% |
| 8 | 3948.22 | 3035.44 | +23.1% |
| 16 | 7807.18 | 5885.63 | +24.6% |

Table 7: TTFT across input lengths, with batch size 16, chunked prefill disabled, and prefix caching disabled.

| Input Length | Baseline TTFT (ms) | HPC TTFT (ms) | Improvement |
| ---: | ---: | ---: | ---: |
| 2K | 1792.62 | 1363.13 | +24.0% |
| 4K | 3704.27 | 2886.40 | +22.1% |
| 8K | 7807.12 | 5893.93 | +24.5% |

## What's Next

This is the start of a longer collaboration with the vLLM community. We will continue working with vLLM maintainers and contributors to improve these backends, extend hardware and model coverage where possible, and upstream further kernel work as it matures. Feedback, issues, and benchmarks are welcome.

## Acknowledgements

We thank the teams and contributors who worked together to bring these backends to vLLM:

- **Tencent Hunyuan AI Infra** built and optimized the HPC-Ops Attention and MoE kernels and contributed them to vLLM as backends. Contributors include Sethran Liu, Chase Shao, Shengy Wei, Theo Cheng, Ryann Xue, Lando Jiang, Looper Zhao, Haank Lin, Aiden Ren, Lehua Ding, Chengv Jiang, Steven Kuang, Liqi He, Kipper Gong, Reedlau Liu, Raccoon Liu, and Dick Zhu.
- **Tencent Network Platform Department** collaborated on communication optimization. Contributors include Xuan Zhang, Haoran Zhao, Yuanyuan Gong, Yadong Liu, Jinzhu Wang, Yinben Xia, Xiang Li, Quan Wen, and Zekun He.
- **vLLM/Inferact** provided backend interface support, reviews, and design discussions. Contributors include Kaichao You, Yongye Zhu, and Yifan Qiao.
- **NVIDIA** collaborated on kernel and performance optimization. Contributors include Yuanhang Sun, Perkz Zheng, Yuxi Chi, Jiang Shao, Jun Gu, Meng Wang, River Liu, Gary Ji, and Chandler Zhou.

We also thank the broader open-source kernel community whose work this builds on and measures against, including NVIDIA CUTLASS/CuTe, TensorRT-LLM, FlashInfer, FlashAttention, and Triton.
