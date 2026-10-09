---
layout: post
title: "vLLM Support for Vera Rubin: 7.8x Throughput over GB200"
author: "vLLM Team, Inferact, Red Hat, and NVIDIA"
summary: "vLLM now runs on NVIDIA Vera Rubin with day-0 model support, Rubin-tuned FlashInfer kernels, and locality-aware MoE, delivering 7.8x the per-GPU throughput of GB200 on AgentX."
image: /assets/figures/2026-10-09-vera-rubin-preview/social-preview.png
social_image: /assets/figures/2026-10-09-vera-rubin-preview/social-preview.png
tags:
  - hardware
  - performance
---

![vLLM on NVIDIA Vera Rubin](/assets/figures/2026-10-09-vera-rubin-preview/social-preview.png)

## vLLM now supports Vera Rubin!

NVIDIA Vera Rubin is the next-generation rack-scale platform built for agentic inference. Inferact, NVIDIA, Red Hat, and the vLLM community have been bringing vLLM up on Rubin since it was announced, and vLLM runs on Rubin today with daily container builds and support for models from DeepSeek, Moonshot AI, Z.ai, and MiniMax.

This post is an early look at where things stand, and here are a few highlights from the work so far:

- **Rubin hardware:** 3.5x the dense FP4 FLOPS, about 2.4x the HBM bandwidth and 1.7x the bidirectional NVLink bandwidth of GB200, with 2-4x faster exponentials for softmax.
- **Day-0 support:** Rubin shares Blackwell's architecture family, so vLLM’s Blackwell kernels are compatible with Rubin. Thanks to this, vLLM already supports diverse models such as DeepSeek, Kimi, GLM, and MiniMax on Rubin.
- **Rubin-tuned kernels:** Through FlashInfer 0.7.0, vLLM gets Rubin-tuned attention, GEMM, and MoE kernels. We have also tuned our MiniMax Sparse Attention (MSA) prefill kernel for Rubin.
- **Locality-aware MoE:** To make full use of HBM bandwidth, we split the MoE weights across Rubin's locality domains so that the SMs read weights only from the HBM in the same domain.
- **Performance:** Early results already show impressive gains: **7.8x the throughput per GPU versus GB200** on AgentX at matched interactivity and up to **3.7x higher VLM throughput than GB300 NVL72** in MLPerf, with further optimizations underway.

## What Rubin changes for inference

<iframe class="vllm-embed" src="/assets/figures/2026-10-09-vera-rubin-preview/vera-rubin-vs-gb200-specs.html" title="Vera Rubin vs GB200, per GPU" style="display: block; width: 100%; height: 900px; border: 0; overflow: hidden;" loading="lazy" scrolling="no"></iframe>

*Figure 1. Per-GPU comparison of NVIDIA Vera Rubin NVL72 and GB200 NVL72. Hover over a metric to highlight its part of the GPU; "Show table" lists every value. Sources: NVIDIA [Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) and [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) spec pages, and NVIDIA Rubin developer blogs.*

### The Vera Rubin platform

<p align="center">
  <img src="/assets/figures/2026-10-09-vera-rubin-preview/vera-rubin-platform.png" alt="Overview of the NVIDIA Vera Rubin platform." width="100%">
  <br>
  <em>Figure 2. Overview of the NVIDIA Vera Rubin platform (source: <a href="https://www.nvidia.com/en-us/data-center/technologies/rubin/">NVIDIA Vera Rubin Platform</a>).</em>
</p>

NVIDIA Vera Rubin is a rack-scale platform consisting of multiple types of racks and hardware: Vera Rubin NVL72 rack, Vera CPU, Groq 3 LPX, Vera BlueField-4 STX, and Spectrum-6 SPX Ethernet racks. In this section, we dissect a few important features and improvements in Vera Rubin NVL72, where vLLM runs today; Figure 1 summarizes them per GPU.

**Compute units and FLOPs.** FLOPs is arguably the first metric LLM inference workloads care about, especially for their prefill phase. NVIDIA Rubin increases compute capacity for BF16, FP8 and NVFP4.

With 212 SMs (vs 152 SMs in Blackwell Ultra) and enhanced Tensor Cores, a Rubin GPU can deliver up to 17.5 PFLOPS of FP8 throughput and 35 PFLOPS of NVFP4 throughput. These capabilities provide optimized vLLM kernels with greater compute capacity for transformer linear layers and MoE expert computation, complemented by the memory and networking improvements described below.

**Softmax.** Notably, Rubin also improved the softmax performance, which is a core operation in LLM attention. Rubin increases exponential throughput, including 2x FP32 and 4x BF16/FP16 throughput versus NVIDIA GB200, helping softmax keep pace with faster matrix operations.

**Memory.** The HBM in Rubin has been upgraded from HBM3e to HBM4, delivering up to 2.4x higher bandwidth compared to GB200. Combined with its more powerful compute, Rubin GPU accelerates key LLM inference operations such as GEMM, MoE (Mixture of Experts), and attention, and delivers much higher overall throughput and lower decode latency ([details below](#performance)).

**Networking.** Inter-GPU networking is also greatly improved on the Rubin platform. The sixth-generation NVLink delivers 1.7x higher network bandwidth than the previous generation. This would speed up collectives (e.g., AllReduce and All2all) and other communication operations, improving the speed and scalability of large-scale LLM inference (e.g., prefill/decode disaggregation, wide expert parallelism).

## vLLM Rubin support status

The vLLM community has been working on Rubin enablement right after it was publicly announced. Being essential parts of the vLLM community, engineers from NVIDIA, Inferact and Red Hat have collaborated and contributed to make sure all users can deploy arbitrary models on the Rubin platform with ease.

In this section, we highlight vLLM’s ongoing Rubin-specific support, namely leveraging locality domains, and our usability improvements for out-of-the-box use on Rubin hardware.

### Locality domain support

Since Ampere, NVIDIA GPUs have featured non-uniform global memory accesses. The [locality domain feature](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/locality-domains.html) in NVIDIA CUDA 13.4 allows applications to take full advantage of non-uniform global memory access by placing computation and data within the same locality domain. SMs can access global memory within their own locality domain with higher bandwidth and lower latency than HBM in other domains. With [Green Contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html) and CUDA streams, we can launch one kernel in each locality domain, so that each kernel can access local memory. This feature primarily speeds up memory-bound workloads such as MoE decode. Locality domains remain in active design and development. In this section, we use MoE decode as an example for a deep dive.

<iframe class="vllm-embed" src="/assets/figures/2026-10-09-vera-rubin-preview/moe-split-n-locality.html" title="Split-N on two locality domains" style="display: block; width: 100%; height: 560px; border: 0; overflow: hidden;" loading="lazy" scrolling="no"></iframe>

*Figure 3. Split-N in MoE forward on two locality domains. The weights W are split by columns at N/2, and the SMs of each domain read only the half of W in their own HBM, so each domain uses its local memory bandwidth. The input X and the output C are across both domains. Use Pause, Prev/Next or the step chips to go through the steps.*

MoE decode is bound by reading weights from HBM, so our goal is optimizing memory throughput across locality domains. We use the split-N strategy in both FC1 and FC2, as shown in Figure 3. We shard the weights column-wise, place each shard into each locality domain’s global memory, and restrict each domain’s SMs to their local shard. This eliminates most cross-domain memory accesses, leading to improved kernel performance and power savings. Since activation memory is relatively small during decode, keeping it non-localized across two memory domains would incur minimal overhead.

SMs cannot always be partitioned into equal domains. Locality domain creation, by default, will not include those SMs when trying to create equal partitions. To make both partitions have equal SMs, we need to enable `cudaDevSmResourceGroupBackfill` (backfill mode) when creating domains (we refer users to the official locality domain documentation for more detail). During our performance study, we include both default mode (only 200 SMs are used across both domains) and backfill mode (all 212 SMs are used).

Figure 4 compares the MoE layer forward time (FC1 + FC2) with locality domains on and off for different parallel strategies. We use the MiniMax M3 MoE shapes as an example. With locality domains enabled, the MoE layer is up to 1.2x faster at small token counts, 1.16x on average from 32 to 1,024 tokens, and the trend stays roughly the same across the TP and EP serving strategies. Even in default mode, where only 200 of the 212 SMs are used, enabling locality domains gives a similar gain. The primary reason is that in small-token decode the forward pass is dominated by weight loading, and locality domains enable higher HBM throughput.

<iframe class="vllm-embed" src="/assets/figures/2026-10-09-vera-rubin-preview/moe-locality-latency.html" title="Locality-aware MoE latency on MiniMax M3" style="display: block; width: 100%; height: 640px; border: 0; overflow: hidden;" loading="lazy" scrolling="no"></iframe>

*Figure 4. FC1 + FC2 latency per rank of the MiniMax M3 MoE layer on Rubin, non-localized vs localized (lower is better), with the speedup (non-localized ÷ localized latency) above each pair. The tabs switch the parallel strategy (TP2, TP4, EP2, EP4); the toggle switches between backfill mode (all 212 SMs) and default mode (200 of 212 SMs). Balanced routing; communication time is not included.*

### Day-0 usability

Usability is always vLLM’s first priority. As of today, users can pull and use the nightly images built with CUDA 13.4 and PyTorch 2.15 from vLLM’s Docker Hub, namely [`vllm/vllm-openai:cu134-nightly`](https://hub.docker.com/layers/vllm/vllm-openai/cu134-nightly/images/sha256-5f74ee1fb3cec4f248e5ac3ad57d05c5a1af70c61787821795137372865e84fd), for Rubin hardware.

**Blackwell software stack compatibility.** Rubin shares Blackwell’s architecture family, with extended `tcgen05` tensor core instructions. It is a new GPU compile target (sm107), but kernels built for the Blackwell family target (sm100f) can also run on it. In practice, vLLM’s Blackwell kernels, especially the GEMM-heavy ones, like attention and MoE, can already run on Rubin without any modifications.

**Daily container builds.** Daily container builds for Rubin are already available ([#55953](https://github.com/vllm-project/vllm/pull/55953)), enabled by [#53443](https://github.com/vllm-project/vllm/pull/53443) and [#54640](https://github.com/vllm-project/vllm/pull/54640) for the Rubin build path on CUDA 13.4, and [#56545](https://github.com/vllm-project/vllm/pull/56545) for Rubin dependency updates.

**Model coverage.** With these parts in place, vLLM can now serve diverse models including DeepSeek, Kimi, GLM, and MiniMax on Rubin.

### Rubin-tuned kernels

Gradually, the kernels that leverage Rubin-specific hardware capabilities and features are being released and upstreamed to kernel libraries such as FlashInfer, vLLM’s fork of MSA (`vllm-project/MSA`), Humming (`vllm-project/humming`), etc. As of today, vLLM has integrated a few important kernels to achieve maximized Rubin performance, including dense NVFP4 or MXFP4 GEMM, NVFP4 MoE, FP8 attention, FP8 MSA prefill, and many more.

## Performance

We evaluate vLLM performance running on Rubin GPUs with two representative benchmarks: SemiAnalysis AgentX (detailed in our [previous post](https://vllm.ai/blog/2026-09-08-vllm-agentx)), and [MLPerf Inference v6.1](https://mlcommons.org/).

On AgentX, vLLM running MiniMax M3 on Vera Rubin NVL72 delivers up to 7.84x the per-GPU throughput of NVIDIA GB200 at matched interactivity, and 5.18x higher throughput under a 150 TPS constraint. This is a very early look at the platform’s inference capabilities. As we gain access to more Vera Rubin nodes, we’ll broaden testing and accelerate optimization, with further performance gains expected as that work progresses.

The MLPerf Inference v6.1 round was the first-ever testing ground for bringing up vLLM onto the NVIDIA Vera Rubin NVL72 platform. On the Vision Language Model (VLM) benchmark, deploying the Qwen3-VL-235B-A22B model via vLLM as the backend inference engine and Dynamo as the frontend router, Vera Rubin NVL72 delivers up to **3.7x** higher throughput than GB300 NVL72 across offline, server and interactive scenarios. More details on the published MLPerf Inference v6.1 results are in [NVIDIA's blog post](https://blogs.nvidia.com/blog/vera-rubin-nvl72-mlperf-inference/).

<p align="center">
  <img src="/assets/figures/2026-10-09-vera-rubin-preview/agentx-results.png" alt="SemiAnalysis AgentX results for vLLM serving MiniMax M3 on NVIDIA Vera Rubin vs GB200." width="100%">
  <br>
  <em>Figure 5. SemiAnalysis AgentX results for vLLM on NVIDIA Rubin, measured with MiniMax M3.</em>
</p>

## Next Steps

"Rome was not built in a day", and polishing the usability and performance on NVIDIA Rubin GPUs is going to be a continuing journey that is full of excitement. In the immediate future, working together as a community, we are planning to enable many more new features for NVIDIA Rubin GPUs, including but not limited to:

- Integrate the sm107 FlashInfer MegaMoE into vLLM through FlashInfer.
- Fully enable locality domains for MoE layers.
- Uncover and take advantage of more overlapping opportunities among different layers or kernels through PDL and Lamport Sync.
- Explore mega kernels for latency-sensitive use cases.
- Optimize KDA and MLA kernels on Rubin for Kimi K3.
- Integrate Rubin CSA and HCA kernels for DeepSeek-V4.1-Flash.
- Finish up and integrate the Rubin MSA decode kernels.
- Integrate the CFT counted-write MoE all-to-all kernel into vLLM through FlashInfer.

## Acknowledgements

This work is a collaborative effort across Inferact, NVIDIA, Red Hat, and the broader vLLM community. We would like to extend special thanks to:

- NVIDIA, for early access to Rubin and close collaboration throughout development.
- Inferact and NVIDIA, for leading the Rubin collaboration, driving performance tuning, integrating MSA kernels, and dissecting the performance of locality-aware MoE.
- NVIDIA and Red Hat, for setting up and enabling daily Docker builds for Rubin.
- The vLLM community, for continuous support and contributions throughout the process.

## Appendix: running vLLM on Rubin

<details markdown="1">
<summary>Show the kernel configurations for Rubin</summary>

vLLM has already integrated a few highly optimized kernels for Rubin. This section covers the corresponding configurations to enable them for maximized performance on Rubin GPUs.

- CuTe-DSL dense NVFP4 or MXFP4 GEMM. On by default for an NVFP4/MXFP4 model checkpoint, but you can set `--linear-backend flashinfer_cutedsl` for NVFP4 or `--linear-backend flashinfer_cutlass` for MXFP4 to make sure that it is enabled.
- CuTe-DSL NVFP4 MoE. You can turn it on via `--moe-backend flashinfer_cutedsl`.
- CuTe-DSL masked grouped GEMM for NVFP4 W4A4 MoE in the “batched” expert format. This is applicable to a deployment with `--enable-expert-parallel --data-parallel-size N --all2all-backend deepep_low_latency|nixl_ep` where `N>1`. In this case, `--moe-backend auto|flashinfer_cutedsl` both would resolve to this kernel.
- CuTe-DSL FP8 BMM for static per-tensor FP8 W8A8 linear layers. On by default and it can be picked up by the FlashInfer autotuner.
- Trtllm-gen FP8 attention. To turn on, you need FP8 KV cache via `--kv-cache-dtype fp8` or a checkpoint that specifies an FP8 KV cache, and also setting `--attention-backend FLASHINFER|FLASHINFER_MLA`. For DeepSeek-style MLA prefill, please also add `-ac.mla_prefill_backend=TRTLLM_RAGGED -ac.use_prefill_query_quantization=true`.
- CuTe-DSL FP8 MSA prefill. To turn on, you need FP8 KV cache via `--kv-cache-dtype fp8` or a checkpoint that specifies an FP8 KV cache, and also setting `--attention-config '{"backend":"CUTLASS_MSA"}'`.

</details>

<script>
// The interactive figures post their content height ("vllm-embed-resize"); size each iframe to fit.
addEventListener("message", (e) => {
  if (!e.data || e.data.type !== "vllm-embed-resize") return;
  for (const f of document.querySelectorAll("iframe.vllm-embed"))
    if (f.contentWindow === e.source) f.style.height = e.data.height + "px";
});
</script>
