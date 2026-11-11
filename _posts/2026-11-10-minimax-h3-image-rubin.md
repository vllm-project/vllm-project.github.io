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

![H3 image × vLLM-Omni.](/assets/figures/minimax-h3-image-rubin/social-preview.png)

*Cover adapted from [MiniMax’s H3-Image release artwork](https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/assets/minimax-h3-image.jpg).*

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

## Same Request, Every Machine

Food photography, image editing, English and Chinese posters, and game environments,
generated on different systems with the same prompt, reference images and seed **42**.
Each panel shows the output, serving configuration and **mean latency over three seeds**.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-workload-gallery.html" title="H3-Image examples: matched food, editing, typography and game-art requests across GPUs" loading="lazy" scrolling="no" height="900" style="display:block;width:100%;height:900px;border:0;overflow:hidden;"></iframe>

The food and podcast requests come from MiniMax's
[released examples](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/main/examples).
The poster and game prompts follow its
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

For **4K text-to-image**, four GB300 GPUs average **1.48 seconds**. A **4K
image edit with one reference** averages **1.77 seconds**. A single RTX 5090
with CPU layerwise offload takes **52.29 seconds** and **61.45 seconds** for
those same requests. Each timing includes the complete PNG response.

For **2K text-to-image**, a **single H200** averages **1.49 seconds** with
BF16 and resident weights.

Every result shows its GPU count and serving configuration. The 1K checks
return 1024×1024 images; with the local model's default minimum canvas,
they are generated at 2048×2048 and resized. They therefore do not represent
native 1K diffusion compute.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-workload-hardware.html" title="H3-Image text-to-image and editing performance across hardware" loading="lazy" scrolling="no" height="1250" style="display:block;width:100%;height:1250px;border:0;overflow:hidden;"></iframe>

### Reference Images

Keeping the output at **4096×4096**, the two-reference composition averages
**5.22 seconds on GB300** and **6.70 seconds on GB200**. The sixteen-reference
composition takes **39.84 seconds** and **60.63 seconds**, respectively. Each uses
four GPUs. The larger conditioning workload adds substantial work even though
the output size is unchanged.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-workload-hardware.html?mode=image-edit&amp;resolution=4096x4096" title="4K image-edit latency across reference counts and hardware" loading="lazy" scrolling="no" height="1100" style="display:block;width:100%;height:1100px;border:0;overflow:hidden;"></iframe>

These compositions measure execution time and memory capacity; some outputs
still miss or duplicate requested objects. Configurations that could not complete
the workload are labeled in the figure.

[Benchmark data](/assets/figures/minimax-h3-image-rubin/api-workload-results.csv)
for per-seed timings and configurations. Points show the mean of three seeds
after warmup; the bars include the measured range. We use a **10-minute
limit per generation**, including warmup. Configurations without a completed
measurement remain labeled in the figure.

These tests measure image generation in vLLM-Omni using API-supported output
sizes. Prompts are fixed before timing, so the numbers exclude prompt enhancement
and do not measure MiniMax's hosted service.

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
text encoder and the VAE retain their configured precision. The H200 gallery
configuration uses BF16.

**AdaLN reuse.** Exact runtime caching reuses AdaLN projections across requests
when their conditioning inputs and weights match. All six denoising steps still
run; this optimization avoids repeating those conditioning projections.

**GB200 and GB300.** Ulysses SP4 splits sequence work across four GPUs while
DiT TP1 keeps the transformer weights replicated. Text-encoder TP4 and VAE tile
distribution also use all four GPUs. The attention backend is **`TRTLLM_ATTN`**.
VAE CUDA Graph replays captured tile computations, reducing repeated launch
overhead; multiple streams let independent tiles overlap. The server-GPU comparisons keep this layout and the model weights resident.

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
precision and placement remain visible when comparing machines.

## Rubin Bringup

For Rubin, we focus on one serving workload: **4096×4096 text-to-image, no
references, six steps and MXFP8 DiT linears**. Each concurrency point uses the
same 160 prompt–seed pairs, giving us response-time distributions and throughput
under load in addition to the three-seed examples above.

Four Rubin GPUs reached **0.866 s median / 0.944 s p95** with SP4 at concurrency 1.
Four single-GPU replicas reached **6,115 images/hour**, at **4.411 s median**
latency. The latency and throughput headlines are different operating points.

### 4K Latency

Each request contains one prompt, no reference images, and a native
**4096×4096** output from [MiniMax H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image).
We use the released six-step schedule, with no upscaling. Latency includes image
generation, PNG encoding and receipt of the full base64 response over loopback.

At concurrency 1, the Rubin configuration delivers **2.21× faster generation than
GB200 and 1.64× faster than GB300**, measured as reference latency divided by Rubin
latency. All three use SP4 and MXFP8 DiT linears for this latency comparison.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-hardware-comparison.html" title="4K image latency and peak throughput across Rubin, GB200 and GB300" loading="lazy" scrolling="no" height="720" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

### Throughput

Fast responses and a busy image service need different configurations. SP4 splits
one image's DiT work across four GPUs. SP1×DP4 instead runs four independent
single-GPU servers, so several images can be generated at once.

SP4 at concurrency 2 reaches **4,407 images/hour** with a **1.421-second median**
response. That is already 99.9% of the measured SP4 peak. Moving to four replicas
raises throughput to **6,115 images/hour**, a **38.7% increase**, at concurrency 8
and a **4.411-second median** response. It is a useful choice when a queue of
requests matters more than finishing one request as quickly as possible.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-performance-benchmarks.html?embed=1" title="MiniMax H3-Image: throughput per GPU versus response latency" loading="lazy" height="720" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

C is the total number of in-flight requests across the four GPUs. Every plotted
point contains 160 measured requests after warmup. Both Rubin sweeps completed
800 requests with zero failures. Throughput includes client image validation and
saving. Each DiT forward generates one image; the replicas do not pack requests
into a shared DiT batch.

With **MXFP8/SP4 on all three platforms**, Rubin reaches **2.20× GB200 and 1.60× GB300
peak throughput**.

### Serving Economics

At the peak measured rate, a four-GPU Rubin system projects to **146,755 4K
images/day**. Using **USD 0.06879 per billable image**, that is **USD 105.16/GPU/hour**
in gross revenue. At 75% billable utilization, it becomes **USD 78.87/GPU/hour**.

<iframe class="vllm-embed" src="/assets/interactive_pages/minimax-h3-image-serving-economics.html" title="Daily image capacity and gross revenue per GPU-hour" loading="lazy" scrolling="no" height="720" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

The price is MiniMax's non-promotional standard 4K T2I example, used here as an
assumed sale price. MiniMax's actual API bill depends on token usage.
[Official pricing](https://platform.minimax.io/docs/pricing/overview).

The calculation is straightforward: **images/hour × price/image × billable
utilization ÷ 4 GPUs**. These are gross-revenue scenarios, before hardware, power,
network and other operating costs. Daily image counts extend the measured hourly
rate to 24 hours; they are not results from a day-long load test. For the subsecond
SP4 operating point, the same price gives **USD 57.86/GPU/hour** at full billable utilization.

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

**Huawei Ascend.** The [NPU recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-Image-NPU.md),
added in [PR #10155](https://github.com/vllm-project/vllm-omni/pull/10155),
covers **Atlas 800I A2, Atlas 800T A3 and Atlas 350 (A5)**. It includes measured
one-, two- and four-NPU profiles on A2 and A5, using BF16 and eager execution.
The four-NPU layouts use DiT TP4 on A2 and Ulysses SP4 with text-encoder TP4
on A5; smaller configurations use layerwise offload. Further VAE graph execution
optimizations are described in [PR #10184](https://github.com/vllm-project/vllm-omni/pull/10184).

**Moore Threads MTT S5000.** H3-Image is supported through the
[MUSA implementation](https://github.com/vllm-project/vllm-omni/pull/10139),
which enables VAE graph decoding and includes a
[model-specific recipe](https://github.com/vllm-project/vllm-omni/blob/5b757b77c05db3562239b19707f059d282de7a62/recipes/MiniMaxAI/MiniMax-H3-Image-MUSA.md)
with measured **TP2 and TP4** layouts on 80 GB S5000 cards. The graph tests
check bitwise agreement with eager decoding.

## OpenDirector

Day-0 support also reaches the creative workflow. OpenDirector's
[H3-Image integration](https://github.com/ThinkFlowLab/OpenDirector/pull/280)
is merged: users can generate an image, place it on the canvas, and use it as
a reference for the next edit, from either the canvas or the agent.

The integration uses **MiniMax's hosted image API**, with standard and Highspeed
presets and support for up to 16 ordered reference images. Its published
examples show generation and subsequent edits using one or two references.

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
