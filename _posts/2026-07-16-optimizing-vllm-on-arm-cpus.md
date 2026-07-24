---
layout: post
title: "Optimizing vLLM on Arm CPUs"
author: "Arm Team"
summary: "An overview of Arm CPU enablement and inference performance optimizations in vLLM."
image: /assets/figures/2026-07-16-arm-cpu/bars_all_vs_bf16_baseline.png
tags:
  - hardware
  - performance
---

## Introduction

Large language model serving on CPUs is an important deployment option because CPUs offer lower deployment cost, simpler infrastructure, and broad availability across cloud and enterprise data centers. As Arm® Neoverse™-based servers become more widely deployed, improving the usability, feature coverage, and performance of open-source serving frameworks such as vLLM on Arm CPUs has become increasingly important.

Over the last several months, we have worked with the vLLM, PyTorch, oneDNN, and KleidiAI communities to make upstream improvements across the Arm CPU serving stack. The result is improved usability, broader model and feature support, and substantial performance gains that benefit any Arm Neoverse-based server running vLLM.

In this blog, we walk through the usability and coverage improvements first, then dig into the main performance optimizations and the end-to-end serving results.

## Enablement

Alongside performance optimizations, we improved the usability and feature completeness of vLLM on Arm®-based CPUs, making it easier to deploy vLLM on Arm® servers.

Key enablement improvements include:

- Pre-built [wheels](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#arm-aarch64_2:~:text=venv/bin/activate-,Pre%2Dbuilt%20wheels,%C2%B6,-When%20specifying%20the) and [Docker images](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#arm-aarch64_4:~:text=%C2%B6-,Pre%2Dbuilt%20images,%C2%B6,-Intel/AMD%20x86).
- Bug fixes for crashes, accuracy issues, threading, and CPU utilization.
- Support for chunked prefill and prefix caching.
- Support for INT8 W8A8 and INT8 W4A8 inference.
- Model enablement for GPT-OSS, Whisper, and Qwen 3.5 / 3.6.
- Better integration with the [PyTorch](https://github.com/pytorch/pytorch) and [UXL](https://github.com/uxlfoundation) ecosystems.

With these enablement improvements in place, we turned our attention to understanding and eliminating the performance bottlenecks.

## Performance Improvements

When we first benchmarked vLLM on Arm-based CPUs in October 2025, performance was much lower than expected given that roughly 80% of model runtime was spent in dense layers dispatched to highly optimized BF16 GEMMs. The standalone GEMM kernels behind those layers were already close to expected hardware efficiency, so the biggest gains were unlikely to come from GEMM kernels alone. 

The profiles instead pointed to a broader optimization problem: allocator behavior, runtime synchronization, framework overheads, attention kernels, and quantized execution.

### Memory Allocation

LLM serving puts significant pressure on the CPU memory allocator. During prefill and decode, vLLM repeatedly allocates and releases tensors for scheduling, KV-cache management, and intermediate operator outputs. In our initial benchmarks, memory allocation showed up as a bottleneck, with poor reuse of large allocations causing a high number of page faults.

The root cause was PyTorch's use of glibc `malloc`. Large allocations were not reused effectively across repeated inference steps, and allocation/free paths became a source of contention as thread counts increased. As a workaround, we initially recommended preloading a caching allocator, but that added manual setup and made performance depend on runtime configuration.

To improve out-of-the-box performance, we enabled [mimalloc](https://github.com/microsoft/mimalloc) as the default allocator on Arm-based CPUs in PyTorch. Mimalloc is a caching allocator designed to scale under multi-threaded allocation pressure. We chose it because it delivered strong performance across a broad range of TorchBench workloads and was already integrated as a PyTorch dependency for non-Arm Linux builds.

This improved Llama 3.1 8B out-of-the-box offline throughput by 2.3× and delivered gains of approximately 7× in low-concurrency serving scenarios.

> [!NOTE]
> We exclude the allocator improvement from all performance plots in this post because its gains would dominate the scale and obscure the impact of the other optimizations. The plots therefore show improvements from the rest of the stack.

### Synchronization at High Core Counts

After improving memory allocation, the next bottleneck appeared when scaling inference to higher core counts. Beyond a certain point, adding more cores did not improve throughput and could even regress performance.

To understand where the scaling broke down, we profiled individual layers at high thread counts. One profile showed that 74% of the paged attention time was spent in OpenMP dynamic scheduling:


```text
97.94% gomp_thread_start
  90.08% paged_attention_v1_impl
    74.07% gomp_iter_dynamic_next
     7.00% reduceValueBlock::lambda(int)
```

`gomp_iter_dynamic_next` is part of libgomp's dynamic loop scheduling path. In this path, the runtime uses an atomic fetch-add to assign loop chunks to worker threads. The libgomp runtime used by the PyTorch wheels implemented that atomic update with a load-linked / store-conditional retry loop:

```c
for (;;) {
    long old = LDXR(p);
    long newv = old + delta;
    int fail = STLXR(p, newv);
    if (fail == 0) {
        DMB_ISH();
        return old;
    }
}
```

At high core counts, many worker threads contend on the same atomic update, leading to repeated failed store attempts and retry traffic.

Tracing this down to the assembly revealed a missed hardware optimization opportunity. The benchmark system used Neoverse™ V2 cores, which support [Arm Large System Extensions (LSE)](https://learn.arm.com/learning-paths/servers-and-cloud-computing/lse/example/). LSE provides hardware atomic instructions, such as `LDADDAL`, that replace the inefficient loop above. However, the OpenMP runtime used by PyTorch did not leverage LSE atomics.

We addressed this by building a libgomp runtime in PyTorch that uses LSE atomics on capable CPUs. 

This improved Llama 3.1 8B offline throughput by 9% and reduced Time Per Output Token (TPOT) latency by 15% in low-concurrency serving scenarios.

### Dense-Layer Layout Overhead

Even after the allocator and runtime improvements, dense layers still left performance on the table. High-performance GEMM kernels are sensitive to weight layout: to run efficiently, weights need to be in a blocked format that matches the kernel’s vectorization and cache-access pattern. Without prepacking, each call can pay the cost of transforming weights from the framework tensor layout into the kernel-friendly format.

This is especially expensive at low concurrency, where the packing cost is not amortized over large batches. We addressed this by enabling a fast oneDNN path for dense layers, accelerated by the Compute Library for Arm Architecture. This path lets vLLM pack BF16 weights during model warmup into the format expected by the kernel, then reuse that packed representation during inference.

This improved Llama 3.1 8B offline throughput by 16% and reduced TPOT latency by 60% in low-concurrency serving scenarios.

### Paged Attention

The CPU paged attention kernel was not optimized for Arm-based CPUs. The QK and PV matrix multiplications, along with the exponential in softmax, were falling back to reference implementations. As a result, we relied on PyTorch's Scaled Dot-Product Attention kernel for prefill, which meant chunked prefill and prefix caching were not supported on the Arm CPU path.

We optimized the QK and PV paths with custom GEMM kernels using Arm [BFMMLA](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bfloat16-processing-for-neural-networks-on-armv8_2d00_a) Advanced SIMD instructions. We also optimized the softmax exponential with a fast vectorized third-degree polynomial approximation.

These changes made paged attention up to 4× faster and improved Llama 3.1 8B offline throughput by 12%. 
Furthermore, this allowed us to enable paged attention for prefill on Arm-based CPUs, unlocking support for chunked prefill and prefix caching.

### BF16 Performance Improvements

The synchronization, weight prepacking, and paged attention optimizations combine to create a stronger BF16 serving baseline than the one we started with in October 2025.

<figure style="text-align: center;">
  <img src="/assets/figures/2026-07-16-arm-cpu/heatmap_bf16_optimized_vs_bf16_baseline.png" alt="Heatmap showing optimized BF16 serving relative to the October 2025 BF16 baseline" />
  <figcaption><em>Optimized BF16 serving relative to the October 2025 BF16 baseline.</em></figcaption>
</figure>

### INT8 W8A8 (8-bit weights and activations)

LLM inference repeatedly reads large weight matrices during prefill and decode. Storing weights in INT8 instead of BF16 reduces memory bandwidth pressure and can allow larger models to fit within the same memory budget. 

On Arm-based CPUs with I8MM, W8A8 also maps to [SMMLA](https://developer.arm.com/documentation/dui0379/e/arm-and-thumb-instructions/smmla), Arm’s signed INT8 matrix multiply-accumulate instruction, which provides twice the theoretical matrix-multiply throughput of BF16.

To take advantage of this, we accelerated the W8A8 quantization path with [oneDNN](https://github.com/uxlfoundation/oneDNN) JIT kernels that use `SMMLA` instructions on SVE128 and SVE256.

As a result, multiple Hugging Face INT8 W8A8 checkpoints, including `RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8` and `RedHatAI/whisper-large-v3-quantized.w8a8`, now perform well out of the box.

Compared to our optimized BF16 baseline, W8A8 with per-token activation quantization and channelwise weight quantization delivers as much as 88% higher throughput, 45% lower TPOT, and 54% lower TTFT, depending on concurrency.

<figure style="text-align: center;">
  <img src="/assets/figures/2026-07-16-arm-cpu/heatmap_int8_vs_bf16_optimized.png" alt="Heatmap showing INT8 W8A8 serving relative to the optimized BF16 path" />
  <figcaption><em>INT8 W8A8 serving relative to the optimized BF16 path.</em></figcaption>
</figure>

> [!NOTE]
> To learn more about INT8 W8A8 on Arm-based CPUs, try [this](https://learn.arm.com/learning-paths/servers-and-cloud-computing/vllm-benchmark-quantisation/) Arm Learning Path.

### INT8 W4A8 (4-bit weights, 8-bit activations)

W4A8 pushes the same idea further: quantize weights to INT4 to lower memory bandwidth pressure during inference. This is especially useful at low concurrency, where there is less batching to amortize the cost of reading model weights.

This path is accelerated through [KleidiAI](https://github.com/ARM-software/kleidiai)'s INT4 micro-kernels.

Compared to the W8A8 baseline above, W4A8 with per-token activation quantization and channelwise weight quantization delivers as much as 29% higher throughput, 26% lower TPOT, and 18% lower TTFT, depending on concurrency.

As expected, the biggest W4A8 speedups appear in low-concurrency scenarios where inference is mostly memory-bound.

<figure style="text-align: center;">
  <img src="/assets/figures/2026-07-16-arm-cpu/heatmap_int4_vs_int8.png" alt="Heatmap showing INT8 W4A8 serving relative to the INT8 W8A8 path" />
  <figcaption><em>INT8 W4A8 serving relative to the INT8 W8A8 path.</em></figcaption>
</figure>

> [!NOTE]
> Please refer to [these docs](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int8_w4a8/) to learn how to quantize your models to INT8 W4A8 with llm-compressor.

## Summary

vLLM on Arm-based CPUs has seen dramatic improvements in usability, robustness, model and feature coverage, and performance.

Relative to the October 2025 BF16 baseline, the optimized BF16 path delivers up to **2.7× the serving throughput**. INT8 W8A8 reaches up to **4.8× the baseline throughput** and a **5.7× TPOT speedup**, while INT8 W4A8 delivers the best results with up to **6.2× the baseline throughput**, a **7.8× TPOT speedup**, and a **2.6× TTFT speedup**.

The gains came from optimizing the full CPU inference stack: memory allocation, OpenMP synchronization, dense-layer prepacking, paged attention, and quantization.

<figure style="text-align: center;">
  <img src="/assets/figures/2026-07-16-arm-cpu/bars_all_vs_bf16_baseline.png" alt="Bar chart showing serving speedups for optimized BF16, INT8 W8A8, and INT8 W4A8 configurations relative to the October 2025 BF16 baseline" />
  <figcaption><em>Serving speedups for optimized BF16, INT8 W8A8, and INT8 W4A8 configurations relative to the October 2025 BF16 baseline.</em></figcaption>
</figure>


Beyond the measured performance gains, these improvements make vLLM a more complete and production-ready inference stack for Arm Neoverse-based servers through broader feature coverage, better out-of-the-box usability, upstream integration, and expanded model support.

## Acknowledgements

We thank the vLLM community for their continued support and collaboration.

Special thanks to [Li Jiang](https://github.com/bigPYJ1151) (Intel®) for maintaining the vLLM CPU backend and implementing much of the infrastructure this work builds on. We also thank [Sanket Kale](https://github.com/sanketkaleoss) (Fujitsu) for the initial Arm CPU enablement in vLLM, and [Shreyas](https://github.com/Shreyas-fuj) (Fujitsu) for contributing SVE256 INT8 kernels to oneDNN.

---

<small>
Arm is a registered trademark of Arm Limited (or its subsidiaries or affiliates).<br>
PyTorch is a trademark of The Linux Foundation.<br>
Intel and oneDNN are trademarks of Intel Corporation or its subsidiaries.<br><br>
This blog post is Copyright 2026 Arm Limited and/or its affiliates &lt;open-source-office@arm.com&gt;
</small>
