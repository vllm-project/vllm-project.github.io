---
layout: post
title: "H3-Image Day-0 Support: 4K in 0.87s with Vera Rubin"
author: "vLLM-Omni team"
summary: "H3-Image on Vera Rubin: 0.87-second 4K generation or 6,115 images/hour on four GPUs, with projected API revenue and support across consumer and data-center hardware."
image: /assets/figures/minimax-h3-image-rubin/social-preview-compact.png
social_image: /assets/figures/minimax-h3-image-rubin/social-preview-compact.png
tags:
  - vllm-omni
  - image-generation
  - hardware
  - performance
---

![vLLM-Omni and MiniMax H3-Image.](/assets/figures/minimax-h3-image-rubin/social-preview-compact.png)


We brought [MiniMax H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
to **NVIDIA Vera Rubin** with vLLM-Omni. On four GPUs, native 4096×4096 generation
reaches **0.87 seconds median** in our short-prompt serving benchmark. A separate
throughput configuration delivers **6,115 images/hour**, projecting to
**146,755 images/day**.

At **USD 0.06879 per image**, using
[MiniMax's official Standard 4K API pricing example](https://platform.minimax.io/docs/pricing/overview)
before launch discounts as an assumed sale price, peak throughput projects to
**USD 105.16/GPU/hour in gross revenue** at full billable utilization, or
**USD 78.87/GPU/hour** at 75%. These estimates are before operating costs;
the daily capacity is an extrapolation of the measured rate.

H3-Image is an open-weight model for text-to-image generation and image editing. One checkpoint
handles both tasks in six sampling steps, generates native high-resolution images
without an upscaler, and supports up to 16 reference images.

We bring H3-Image to **consumer and data-center GPUs**, spanning **NVIDIA, AMD,
Intel and Moore Threads (MUSA)**, as well as **Huawei Ascend NPUs**.

- **Single- and multi-GPU serving.** Run on a single RTX GPU or distribute
  generation across multiple data-center GPUs, with configurations for different
  memory capacities.
- **Measured across 10 NVIDIA hardware types.** Compare latency for text-to-image
  generation and editing across output resolutions and reference-image counts.

## Quality

Oil-paint strokes, a multi-view camera drawing, Japanese poster lettering and
tiny words on a rice grain reveal what a smaller image can lose. A game
environment adds weathered stone and reflected light. Each uses one H3-Image-generated
**4K** image, resized to **1K** and **2K** with its aspect ratio preserved,
so composition stays fixed.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-resolution-detail.html?v=e08274c6090c" title="H3-Image texture and detail at 1K, 2K and 4K with synchronized zoom" loading="lazy" scrolling="no" height="850" style="display:block;width:100%;height:850px;border:0;overflow:hidden;"></iframe>

The [additional examples](/assets/interactive_pages/minimax-h3-image-workload-gallery.html?v=70db77d34bea)
retain the food, editing, poster and game outputs across hardware. The food and
original studio-edit prompts come from MiniMax's
[released examples](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/main/examples);
the product, poster and game prompts follow its
[H3-Image prompt-writing guide](https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/6ea7b9e55523a2177e4f759db372065ba6089510/skills/h3-image-prompt-writing/SKILL.md).
Prompts are prepared ahead of time; the PE checkpoint is not part of these timings.
The model's text encoder, generation, PNG encoding and response transfer are included.

## Performance

The two API modes are straightforward: **Text-to-Image** takes a prompt;
**Image Edit** takes a prompt and one or more reference images. Our main
comparisons use **2K (2048×2048)** and **4K (4096×4096)** output, with one
image returned per request. The prompts and reference files stay fixed across
machines. See MiniMax's [API guide](https://platform.minimax.io/docs/guides/image-generation)
for the request parameters.

Choose **Text-to-image** or **Image Edit** below. For editing, select
**1, 2, 4, 8 or 16 reference images**; output sizes reflect the measured cases.

For **4K text-to-image**, four GB300 GPUs average **1.48 seconds**. A **4K
image edit with one reference** averages **1.77 seconds**. A single RTX 5090
with CPU layerwise offload takes **52.29 seconds** and **61.45 seconds** for
those same requests. Each timing includes the complete PNG response.

For **2K text-to-image**, a **single H200** averages **1.49 seconds** with
BF16 and resident weights.

For the detailed coffee-product prompt used here, Rubin averages **1.09 seconds**
at 4K. The **0.87-second** headline comes from a separate short-prompt serving
benchmark in **Rubin Bringup**, not this request. Both use four GPUs, SP4 and
MXFP8; the request corpus, HTTP client and summary statistic differ.

Every result shows its GPU count. **Test setup** lists the exact GPU model,
CUDA-visible VRAM per GPU, interconnect, power limit and serving configuration.
GPU count, precision and memory placement vary, so these figures compare deployed
systems rather than isolate GPU architecture. The 1K checks
return 1024×1024 images; with the local model's default minimum canvas,
they are generated at 2048×2048 and resized. They therefore do not represent
native 1K diffusion compute. Some 1K and 2K runs also use different allocator
settings, shown in Test setup.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-workload-hardware.html?v=ef8a768c0639" title="H3-Image text-to-image and editing performance across hardware" loading="lazy" scrolling="no" height="1250" style="display:block;width:100%;height:1250px;border:0;overflow:hidden;"></iframe>

[Benchmark data](/assets/figures/minimax-h3-image-rubin/api-workload-results.csv).
Bars show three-seed means after warmup, with min–max ranges. Configurations,
timing scope and the 10-minute request limit are documented in **Test setup**.

## Inference Optimizations

H3-Image pairs a **Qwen3-VL encoder** with a **33.1B dense DiT** and an image
VAE. A native 4096×4096 output contributes **16,384 target-image tokens** across
six denoising steps, before VAE decoding. References add encoding work and
conditioning tokens. This gives us three practical targets: DiT compute,
decoding overhead and model memory. See the
[model architecture](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image#model-architecture).

### Faster DiT

On Blackwell and Rubin, **MXFP8 GEMMs** accelerate eligible DiT linears while
reducing their weight footprint. **Ulysses SP4** distributes sequence work across
four GPUs; DiT TP1 keeps the weights replicated. **AdaLN caching** reuses exact
conditioning projections when their inputs and weights match. All six denoising
steps still run. The encoder and VAE keep their configured precision; Hopper
and Ampere use BF16.

### Faster Decoding

The VAE decodes independent image tiles. We distribute those tiles across GPUs,
then combine **CUDA Graph replay with multiple CUDA streams** on each GPU:
replay reduces repeated launch overhead, while streams allow tiles to overlap.

A separate BF16 control on four GB300 GPUs shows why this matters. VAE decoding
fell from **2.635 s to 0.213 s**, bringing end-to-end latency from approximately
**4.0 s to 1.6 s**, or **2.5× faster**. Both paths already distribute tiles across
four GPUs; this is the combined graph-and-stream gain.
[Control measurements](/assets/figures/minimax-h3-image-rubin/vae-decoding-control.json).

### Fitting Smaller GPUs

**Layerwise CPU offload** trades weight-transfer time for GPU memory. RTX 5090
and RTX 5080 offload the DiT and text encoder and decode the VAE eagerly.
RTX 6000D and RTX PRO 6000 keep the DiT resident, offload the encoder and use
VAE CUDA Graph. This enables single-GPU deployment while explaining part of
the latency gap with resident data-center configurations.

## Rubin Bringup

Our starting point was vLLM-Omni's H3-Image implementation and
[vLLM's Rubin support](https://vllm.ai/blog/2026-10-09-vera-rubin-preview),
using a CUDA 13.4 runtime. H3-Image adds a dense diffusion workload to that
foundation: a native 4096×4096 image contributes 16,384 target-image tokens
across six denoising steps, followed by VAE decoding. We tested how this path
translates Rubin's capabilities into interactive latency and serving throughput.

We made two changes for these runs:

- **MXFP8 integration.** We connected Omni's diffusion linears to vLLM's online
  MXFP8 loading and kernel selection. On Rubin, eligible DiT linears use
  **FlashInfer CuTe-DSL MXFP8 GEMM**; attention uses **`TRTLLM_ATTN`**. These are
  existing library kernels. The text encoder and VAE retain native precision.
- **Reliable 4K output.** Repeated 4K requests exposed a stall in the output
  transfer path. We moved the final image-layout conversion to the CPU, after
  transferring the decoded uint8 tensor. Regression checks confirmed repeated
  requests completed and decoded pixels were unchanged.

We then compared SP4, SP2×DP2 and SP1×DP4, using the existing VAE CUDA Graph and
AdaLN caching paths. Four GPUs reached **0.87-second responses** with SP4, or
**6,115 images/hour** with four replicas. The following sweep fixes native 4K
text-to-image, no references, six steps and MXFP8 DiT linears, with a separate
160-request prompt–seed corpus at each concurrency point: 20 short prompts,
each 122–160 characters, across eight seeds. The cross-hardware coffee-product
request above uses a 1,292-character prompt and reports a three-seed mean.

### Latency

With SP4 at concurrency 1, median latency is **0.866 seconds** (p95 **0.944
seconds**): **2.21× faster than GB200** and **1.64× faster than GB300**. All
three latency configurations use four GPUs, SP4 and MXFP8.

### Throughput

Four single-GPU replicas (**SP1×DP4**) reach **6,115 images/hour** at concurrency
8, with **4.411-second median latency**. SP4 reaches **4,407 images/hour** at
concurrency 2 with a **1.421-second median**. Replicas add **38.7% throughput**;
the subsecond and peak-throughput headlines are separate operating points.

With SP4 fixed across platforms, Rubin delivers **2.20× GB200** and
**1.60× GB300** peak throughput. Each plotted point contains 160 warm requests;
replicas serve independent images without combining them into a DiT batch.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-performance-benchmarks.html?embed=1&amp;v=35f1312873e6" title="MiniMax H3-Image: throughput per GPU versus response latency" loading="lazy" height="720" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

### Economics

**6,115 images/hour** projects to **146,755 4K images/day** per four-GPU
Rubin system. We use **USD 0.06879 per image**, the
[MiniMax official API pricing example](https://platform.minimax.io/docs/pricing/overview)
for **Standard 4K text-to-image with no references, before launch discounts**.
At that assumed sale price, projected gross revenue is **USD 105.16/GPU/hour**
at 100% billable utilization, or **USD 78.87/GPU/hour** at 75%.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-serving-economics.html?v=eb6ce7318059" title="Latency, hourly throughput and gross revenue across four-GPU systems" loading="lazy" scrolling="no" height="720" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

Revenue = images/hour × price/image × billable utilization ÷ 4 GPUs.
Actual hosted-API charges vary with text-token usage. Revenue figures are before
operating costs; daily capacity extrapolates the measured rate, rather than a
24-hour load test.

## Other Hardware

H3-Image is also supported on **AMD Instinct, Intel Arc Pro, Huawei Ascend and
Moore Threads** hardware through the implementations and recipes below. These
community validations use their own workloads and configurations; the matched
performance figures above cover our NVIDIA runs.

**AMD Instinct.** The [AMD recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-Image-AMD.md)
includes text-to-image and editing measurements on **MI325X and MI350X**, from
one to eight GPUs, using BF16 and `FLASH_ATTN`. It covers Ulysses sequence
parallelism and tensor parallelism. ROCm uses a single VAE graph replica to
avoid the hangs observed with concurrent graph replays.

**Intel Arc Pro B70.** The [XPU recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-Image-XPU.md)
documents measured one-, two- and four-GPU deployments. Its recommended
four-card layout uses **DiT TP4 with text-encoder layerwise offload** to fit
32 GB cards. The smaller layouts also offload DiT weights; VAE decoding runs
eagerly on XPU.

**Huawei Ascend.** The [NPU recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-Image-NPU.md)
covers **Atlas 800I A2, Atlas 800T A3 and Atlas 350 (A5)**. Measured A2 and A5
profiles use one, two or four NPUs with BF16 and eager execution. Four-NPU
configurations use **DiT TP4 on A2** and **Ulysses SP4 with text-encoder TP4 on A5**;
smaller configurations use layerwise offload.

**Moore Threads MTT S5000.** H3-Image is supported through the
[MUSA implementation](https://github.com/vllm-project/vllm-omni/pull/10139),
which enables VAE graph decoding and includes a
[model-specific recipe](https://github.com/vllm-project/vllm-omni/blob/5b757b77c05db3562239b19707f059d282de7a62/recipes/MiniMaxAI/MiniMax-H3-Image-MUSA.md)
with measured **TP2 and TP4** layouts on 80 GB S5000 cards. The graph tests
check bitwise agreement with eager decoding.

## OpenDirector

[OpenDirector](https://github.com/ThinkFlowLab/OpenDirector) is an agent-driven
platform for creators, started by vLLM-Omni committers to make vLLM-Omni easier
to use in creative workflows.

Its Day-0 H3-Image integration currently uses **MiniMax's hosted image API**.
Creators can generate and edit images through the canvas or agent, then use the
outputs as references for the next edit. We plan to expand support for
**locally deployed H3-Image workflows powered by vLLM-Omni**.

## Acknowledgements

We thank MiniMax for releasing H3-Image and its model weights; the NVIDIA team and
the vLLM community for their work on Vera Rubin support; and the vLLM-Omni and
FlashInfer contributors for the serving and kernel infrastructure used in this
work.

We also thank the OpenDirector contributors for their work on H3-Image integration
in creative workflows.

<script>
(function () {
  window.addEventListener('message', function (event) {
    var d = event.data;
    if (!d || d.type !== 'vllm-embed-resize' || typeof d.height !== 'number' || !Number.isFinite(d.height)) return;
    document.querySelectorAll('iframe.vllm-embed').forEach(function (frame) {
      if (frame.contentWindow === event.source) frame.style.height = Math.max(200, Math.min(12000, Math.ceil(d.height))) + 'px';
    });
  });
})();
</script>
