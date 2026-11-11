---
layout: post
title: "H3-Image Day-0 Support: 4K in 0.87s with Vera Rubin"
author: "vLLM-Omni team"
summary: "H3-Image across server and desktop GPUs: matched creative examples, workload latency and the serving choices behind 0.87-second 4K generation on Vera Rubin."
image: /assets/figures/minimax-h3-image-rubin/social-preview.png
social_image: /assets/figures/minimax-h3-image-rubin/social-preview.png
tags:
  - vllm-omni
  - image-generation
  - hardware
  - performance
---

![vLLM-Omni and MiniMax H3-Image.](/assets/figures/minimax-h3-image-rubin/social-preview.png)


vLLM-Omni now supports [MiniMax H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image),
an open-weight model for text-to-image generation and image editing. One checkpoint
handles both tasks in six sampling steps, generates native high-resolution images
without an upscaler, and supports up to 16 reference images.

We put it to work on food photography, studio edits, English and Chinese posters,
and an adventure-game environment. Here are the highlights:

- **Different tasks, matched inputs.** Compare measured configurations across
  **10 hardware types**, with results organized by text-to-image generation,
  image editing, output resolution and reference-image count.
- **Desktop generation.** A single RTX 5090 generates the **5472×3072 adventure
  scene in 59.77 seconds** with CPU layerwise offload. A single RTX 6000D produces
  the **3552×4736 workshop poster in 42.31 seconds**. Both are three-seed means.
- **Subsecond 4K on Rubin.** Four Rubin GPUs reach **0.87 seconds median** for
  4096×4096 generation. A separate throughput configuration reaches
  **6,115 images/hour**. We examine both operating points at the end.

## Resolution and Detail

Seven scenes put resolution under a closer look: product photography,
portrait editing, food, English and Chinese typography, architecture, and a game
environment. Each comparison uses one measured **4K** image, resized to **1K**
and **2K** with the original aspect ratio preserved. This holds the composition
fixed so lettering and material textures are easier to compare. Measured source
images and their configurations remain available below the image.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-resolution-detail.html?v=e90821954c25" title="H3-Image texture and detail at 1K, 2K and 4K with synchronized zoom" loading="lazy" scrolling="no" height="850" style="display:block;width:100%;height:850px;border:0;overflow:hidden;"></iframe>

The [additional examples](/assets/interactive_pages/minimax-h3-image-workload-gallery.html?v=70db77d34bea)
retain the food, editing, poster and game outputs across hardware. The food and
original studio-edit prompts come from MiniMax's
[released examples](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/main/examples);
the product, poster and game prompts follow its
[H3-Image prompt-writing guide](https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/6ea7b9e55523a2177e4f759db372065ba6089510/skills/h3-image-prompt-writing/SKILL.md).
Prompts are prepared ahead of time; the PE checkpoint is not part of these timings.
The model's text encoder, generation, PNG encoding and response transfer are included.

## Performance Across Hardware and Workloads

### Text-to-Image and Image Edit

The two API modes are straightforward: **Text-to-Image** takes a prompt;
**Image Edit** takes a prompt and one or more reference images. Our main
comparisons use **2K (2048×2048)** and **4K (4096×4096)** output, with one
image returned per request. The prompts and reference files stay fixed across
machines. See MiniMax's [API guide](https://platform.minimax.io/docs/guides/image-generation)
for the request parameters.

The performance figure groups these requests as text-to-image, single-reference
editing and multi-reference editing. The two editing groups use the same API
mode; the reference-image count distinguishes them.

For **4K text-to-image**, four GB300 GPUs average **1.48 seconds**. A **4K
image edit with one reference** averages **1.77 seconds**. A single RTX 5090
with CPU layerwise offload takes **52.29 seconds** and **61.45 seconds** for
those same requests. Each timing includes the complete PNG response.

For **2K text-to-image**, a **single H200** averages **1.49 seconds** with
BF16 and resident weights.

Every result shows its GPU count; serving configurations are in **Test setup**. The 1K checks
return 1024×1024 images; with the local model's default minimum canvas,
they are generated at 2048×2048 and resized. They therefore do not represent
native 1K diffusion compute.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-workload-hardware.html?v=18de9b8de116" title="H3-Image text-to-image and editing performance across hardware" loading="lazy" scrolling="no" height="1250" style="display:block;width:100%;height:1250px;border:0;overflow:hidden;"></iframe>

[Benchmark data](/assets/figures/minimax-h3-image-rubin/api-workload-results.csv).
Bars show three-seed means after warmup, with min–max ranges. Configurations,
timing scope and the 10-minute request limit are documented in **Test setup**.

## How It Runs

H3-Image combines H3's **Qwen3-VL encoder**, a **33.1B-parameter dense DiT**
and an image VAE. Its 50-layer DiT follows the H3 design with the audio branch
removed. Text and reference images pass through Qwen3-VL; references also pass
through the VAE's CNN encoder. These conditioning tokens and the target-image
tokens form one packed sequence. The DiT denoises the target latents in **six
steps without classifier-free guidance**, then a **24-layer 2D ViT decoder**
turns them into pixels. See the [model architecture](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image#model-architecture).

The VAE compresses each spatial dimension by 16, and the DiT groups the latent
into 2×2 patches. A **4096×4096 image therefore contributes 16,384 target-image
tokens**, before adding text and reference conditioning. Larger canvases increase
both the sequence length and VAE decoding work; references add encoding work
and conditioning tokens. This explains why resolution and reference count
matter for memory use and latency, even with only six denoising steps.

**MXFP8.** Eligible DiT linears use MXFP8 through vLLM and FlashInfer. Quantizing
these matrix multiplications reduces their weight footprint; attention, the
text encoder and the VAE retain their configured precision. Hopper and Ampere configurations use BF16.

**AdaLN reuse.** Exact runtime caching reuses AdaLN projections across requests
when their conditioning inputs and weights match. All six denoising steps still
run; this optimization avoids repeating those conditioning projections.

**GB200 and GB300.** Ulysses SP4 splits sequence work across four GPUs while
DiT TP1 keeps the transformer weights replicated. Text-encoder TP4 and VAE tile
distribution also use all four GPUs. The attention backend is **`TRTLLM_ATTN`**.
VAE CUDA Graph replays captured tile computations, reducing repeated launch
overhead; multiple streams let independent tiles overlap. The GB200 and GB300 comparisons keep this layout and the model weights resident.

**Single-GPU systems.** RTX 5090, RTX 5080, RTX 6000D and RTX PRO 6000 use SP1, DiT TP1,
text-encoder TP1 and **`CUDNN_ATTN`**. On the RTX 5090 and RTX 5080, layerwise
offload moves DiT and text-encoder weights between CPU memory and the GPU;
the VAE runs eagerly. The RTX 6000D and RTX PRO 6000
keep DiT weights resident, offload only the text encoder, and use VAE CUDA
Graph.
Offload makes room for the model at the cost of weight-transfer work.

**Hopper and Ampere.** H100, H200 and A100 run
with BF16 and **`FLASH_ATTN`**. The H200 poster in the gallery uses resident
weights and SP4. The measured H100 systems use **H100 NVL**. **A100 PCIe** and
**A100 SXM4** appear separately because their interconnects differ.
These results describe the configurations we tested; GPU count,
precision and placement remain documented when comparing machines.

## Rubin Bringup

Rubin reaches **0.87-second 4K responses** or **6,115 images/hour** on four
GPUs, using different serving configurations. The workload is text-to-image
with no references, six steps and MXFP8 DiT linears. This serving sweep uses
a separate 160-request prompt–seed corpus.

Eligible DiT linears use **FlashInfer CuTe-DSL MXFP8 GEMM**
(`FlashInferCutedslMxfp8LinearKernel`); attention uses **`TRTLLM_ATTN`**.

### Latency

With SP4 at concurrency 1, median latency is **0.866 seconds** (p95 **0.944
seconds**): **2.21× faster than GB200** and **1.64× faster than GB300**. All
three latency configurations use four GPUs, SP4 and MXFP8.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-hardware-comparison.html?v=4c8980570b8d" title="4K image latency and peak throughput across Rubin, GB200 and GB300" loading="lazy" scrolling="no" height="720" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

### Throughput

Four single-GPU replicas (**SP1×DP4**) reach **6,115 images/hour** at concurrency
8, with **4.411-second median latency**. SP4 reaches **4,407 images/hour** at
concurrency 2 with a **1.421-second median**. Replicas add **38.7% throughput**;
the subsecond and peak-throughput headlines are separate operating points.

With SP4 fixed across platforms, Rubin delivers **2.20× GB200** and
**1.60× GB300** peak throughput. Each plotted point contains 160 warm requests;
replicas serve independent images without combining them into a DiT batch.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-performance-benchmarks.html?embed=1&amp;v=4aebbfdd17b8" title="MiniMax H3-Image: throughput per GPU versus response latency" loading="lazy" height="720" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

### Economics

**6,115 images/hour** projects to **146,755 4K images/day** per four-GPU
Rubin system. At an assumed **USD 0.06879 per billable image**, this yields
**USD 105.16/GPU/hour** gross revenue, or **USD 78.87/GPU/hour** at 75%
billable utilization.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-serving-economics.html?v=4eaa76533f1d" title="Hourly image capacity and gross revenue per GPU-hour" loading="lazy" scrolling="no" height="720" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

Revenue = images/hour × price/image × billable utilization ÷ 4 GPUs.
This uses a [MiniMax pricing example](https://platform.minimax.io/docs/pricing/overview)
as an assumed sale price. Figures are before operating costs; daily capacity
extrapolates the measured rate, rather than a 24-hour load test.

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
platform for creators, started by a vLLM-Omni committer to make vLLM-Omni easier
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
