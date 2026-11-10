---
layout: post
title: "vLLM-Omni on Vera Rubin: 4K Images in 0.87 Seconds and 3.1× Throughput over GB200"
author: "lishunyang12"
summary: "MiniMax H3-Image on four Rubin GPUs: 0.87s 4K latency, up to 3.1× GB200 throughput, and 146,755 images/day at the peak measured rate."
image: /assets/figures/minimax-h3-image-rubin/social-preview.png
social_image: /assets/figures/minimax-h3-image-rubin/social-preview.png
tags:
  - vllm-omni
  - image-generation
  - hardware
  - performance
---

![vLLM-Omni on Vera Rubin.](/assets/figures/minimax-h3-image-rubin/social-preview.png)

[vLLM now supports NVIDIA Vera Rubin NVL72](https://vllm.ai/blog/2026-10-09-vera-rubin-preview).
We put that support to work on image generation with **vLLM-Omni and MiniMax H3-Image**.
On four Rubin GPUs, a native 4K image can arrive in less than a second. The same
GPUs can also serve a larger queue of requests, trading response time for more
images per hour.

Here are the results:

- **4K in 0.87 seconds:** 0.866 s median latency and 0.944 s p95, using SP4 at concurrency 1.
- **Up to 3.13× GB200 throughput and 2.19× GB300 throughput:** 6,115 images/hour with four single-GPU replicas.
- **About 146,755 images/day from four GPUs:** at $0.06879 per image and full billable utilization, that capacity is worth **$105.16 per GPU per hour in gross revenue**.

The latency and throughput headlines use different operating points. The peak-rate
comparison uses Rubin with MXFP8 and SP1×DP4, versus GB200 and GB300 with FP8 and SP4.
It compares the best configurations we measured, with their respective response
times, rather than isolating hardware at a common latency target.

## From a prompt to a 4K image in under a second

The workload is deliberately simple: one prompt, no reference images, and a native
**4096×4096** output from [MiniMax H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image).
We use the released six-step schedule, with no upscaling. Latency includes image
generation, PNG encoding and receipt of the full base64 response over loopback.

At concurrency 1, the Rubin configuration delivers **2.22× faster generation than
GB200 and 1.59× faster than GB300**, measured as reference latency divided by Rubin
latency. All three use SP4 for this latency comparison; Rubin uses MXFP8 DiT while
the Blackwell references use FP8 DiT.

![Measured 4K image latency and peak throughput on four-GPU configurations, with Rubin speedups over GB200 and GB300.](/assets/figures/minimax-h3-image-rubin/hardware-comparison.png)

## More requests, more images

Fast responses and a busy image service need different configurations. SP4 splits
one image's DiT work across four GPUs. SP1×DP4 instead runs four independent
single-GPU servers, so several images can be generated at once.

SP4 at concurrency 2 reaches **4,407 images/hour** with a **1.421-second median**
response. That is already 99.9% of the measured SP4 peak. Moving to four replicas
raises throughput to **6,115 images/hour**, a **38.7% increase**, at concurrency 8
and a **4.411-second median** response. It is a useful choice when a queue of
requests matters more than finishing one request as quickly as possible.

<iframe class="vllm-embed" src="/assets/figures/minimax-h3-image-rubin/performance-benchmarks.html?embed=1" title="MiniMax H3-Image: throughput per GPU versus response latency" loading="lazy" style="display: block; width: 100%; height: 900px; border: 0;"></iframe>

<noscript><img src="/assets/figures/minimax-h3-image-rubin/performance-benchmarks.png" alt="H3-Image throughput per GPU versus median HTTP latency"></noscript>

C is the total number of in-flight requests across the four GPUs. Every plotted
point contains 160 measured requests after warmup. Both Rubin sweeps completed
800 requests with zero failures. Throughput includes client image validation and
saving. Each DiT forward generates one image; the replicas do not pack requests
into a shared DiT batch.

For a closer configuration comparison, the **FP8/SP4** runs give Rubin **2.06× GB200
and 1.44× GB300 peak throughput**, with **2.05× and 1.46× C1 latency speedups**.
The [interactive chart](/assets/figures/minimax-h3-image-rubin/performance-benchmarks.html)
lets readers select these matched-precision curves and inspect response times.

## What four GPUs can produce in a day

At the peak measured rate, a four-GPU Rubin system projects to **146,755 4K
images/day**. Using **$0.06879 per billable image**, that is **$105.16/GPU/hour**
in gross revenue. At 75% billable utilization, it becomes **$78.87/GPU/hour**.

![Projected daily 4K image capacity per four-GPU system and gross revenue per GPU-hour at the stated image price.](/assets/figures/minimax-h3-image-rubin/serving-economics.png)

The price is MiniMax's non-promotional standard 4K T2I example, used here as an
assumed sale price. MiniMax's actual API bill depends on token usage.
[Official pricing](https://platform.minimax.io/docs/pricing/overview).

The calculation is straightforward: **images/hour × price/image × billable
utilization ÷ 4 GPUs**. These are gross-revenue scenarios, before hardware, power,
network and other operating costs. Daily image counts extend the measured hourly
rate to 24 hours; they are not results from a day-long load test. For the subsecond
SP4 operating point, the same price gives **$57.86/GPU/hour** at full billable utilization.

## Where the speed comes from

The image pipeline needs to do two substantial jobs: the DiT produces latents,
and the VAE turns them into pixels. We optimize both:

- **DiT execution:** `TRTLLM_ATTN` attention and MXFP8 linear layers, using vLLM's online quantization and FlashInfer's CuTe-DSL kernel.
- **Parallelism:** Ulysses SP4 and encoder TP4 reduce single-request latency; four SP1/encoder-TP1 replicas increase node throughput.
- **VAE execution:** CUDA Graph replay reduces repeated launch overhead. Six streams per GPU allow tiled decoding work to overlap; SP4 also distributes tiles across GPUs.

In a separate native-precision Rubin test, enabling the combined VAE graph-and-stream
path reduced 4K HTTP latency from **2.01 to 1.04 seconds**, a **1.94× speedup**, with
pixel-identical outputs across six tested workloads. This ablation is separate
from the MXFP8 headline. Model weights stay resident, and the encoder and VAE
retain the release pipeline's precision.

In 160 paired SP4 quality checks, mean CLIP text–image scores were **34.353 for BF16
and 34.370 for MXFP8**. A separate 160-pair DP4/SP4 comparison had a score difference
of −0.065, with a 95% interval of [−0.301, +0.140]. These diagnostics cover the tested
corpus; they do not establish full-resolution quality equivalence.

The measurements use [vLLM-Omni PR #10120](https://github.com/vllm-project/vllm-omni/pull/10120)
with additional MXFP8 integration and an output-transfer correction. Together,
these changes give H3-Image two useful serving modes on Rubin: subsecond 4K
responses, or more than 146,000 images a day at the peak measured rate.
