---
layout: post
title: "FastVideo's FastH3 on vLLM-Omni: Low-Latency, Scalable MiniMax H3 Serving"
author: "vLLM-Omni Team"
summary: "How FastVideo's FastH3 delivers low latency while vLLM-Omni's kernels, DLO, disaggregated encoding, VAE parallelism, and media path scale MiniMax H3 serving."
description: "An evidence-driven guide to low-latency FastH3 and scalable MiniMax H3 deployment with vLLM-Omni, DLO, and disaggregated encoding."
image: /assets/logos/vllm-logo-text-light.png
tags:
  - performance
  - large-scale-serving
  - multimodal
  - vllm-omni
  - fastvideo
  - fasth3
published: false
---

MiniMax H3 is a joint video-and-audio diffusion model: one request can combine
text, images, videos, and audio references, then return an MP4 containing H.264
video and synchronized stereo audio. That capability also makes production
serving unusually demanding. A deployment must coordinate a large Qwen3-VL
encoder, a long-sequence audio-video DiT, separate video and audio VAEs, and a
CPU media path without letting any one stage dominate latency or memory.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-model-pipeline.svg" alt="MiniMax H3 model pipeline from multimodal inputs through shared encoders, a packed sequence, task-specific joint audio-video diffusion, separate VAE decoders, and MP4 muxing" width="100%">
</p>

*Figure 1: Text is encoded by the H3/Qwen3-VL encoder; visual conditions use
both that encoder and the Visual VAE; audio references use the Audio VAE. Their
representations join noisy target video/audio latents in one packed sequence.
The selected FL2VA or Ref2VA DiT jointly predicts both output latents before
separate VAE decode and CPU mux. Sources: the official
[MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3), the
[vLLM-Omni recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md),
and the [Diffusers pipeline description](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3).*

This post explains how [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
turns that pipeline into a production serving system. For the low-latency path,
it integrates [FastVideo](https://github.com/hao-ai-lab/FastVideo)'s
[FastH3](https://haoailab.com/blogs/fasth3-preview/) four-step student and
combines it with optimized attention, VAE, transport, and MP4 execution. For
scalable serving, Distributed Layerwise Offload (DLO) expands the feasible
memory/topology envelope, while disaggregated encoding gives the Qwen3-VL and
diffusion stages independent placement, queues, replicas, and caching. To keep
the evidence compact and comparable, the article benchmarks one eight-GPU
NVIDIA B300 node: the 10-second request remains the canonical A/B, and FastH3
adds a 5/10/15-second generation-speed reference. Other hardware remains
covered by maintained recipes rather than additional result matrices.

## TL;DR

- **One serving contract, three H3 tasks.** vLLM-Omni serves text-to-video-and-audio
  (T2VA), first/last-frame-to-video-and-audio (FL2VA), and mixed-reference
  video-and-audio generation (Ref2VA) through `/v1/videos`.
- **Optimize the lossless lane first.** Dense attention, packed-sequence and
  Ulysses boundaries, fused DiT operators, VAE parallelism/kernels, GPU output
  packing/transport, and CPU MP4 construction are compared end to end against
  Diffusers on B300.
- **FastVideo's FastH3 is the measured low-latency path.** Its fused four-step
  student produces complete 5/10/15-second MP4s faster than playback on the
  qualified B300 profile. Other precision, sparsity, and cache optimizations
  remain separate quality decisions.
- **DLO and disaggregation scale production serving.** DLO changes the
  latency/memory/topology trade-off, while disaggregated encoding makes the
  Qwen3-VL stage independently placeable, replicable, schedulable, and cacheable.
  Step execution adds admission and abort boundaries, but its useful H3
  production case remains to be demonstrated.
- **One canonical benchmark keeps the comparison tractable.** Every feature A/B
  uses the official 10-second, 1344×768 T2VA case. The selected four-step B300
  profile also runs at 5 and 15 seconds to report complete-MP4 real-time factor;
  FL2VA and Ref2VA remain capability and recipe coverage.

## 1. Commercial workloads and serving goals

MiniMax H3 has two checkpoint partitions. `FL2VA` serves both text-only and
first/last-frame-conditioned generation; `Ref2VA` handles mixed image, video,
and audio references. Shared components include the tokenizer and processor,
the Qwen3-VL encoder, and the video and audio VAEs. The full task and input
contract is documented in the
[MiniMax H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md).

| Task | Commercial examples | Primary serving pressure |
|---|---|---|
| T2VA | Creative generation, advertising drafts, synthetic media pipelines | DiT latency and output throughput |
| FL2VA | Branded content, controlled transitions, image animation | Image encoding plus denoising latency |
| Ref2VA | Character consistency, video editing, audio-conditioned generation | Long multimodal encoder and packed-attention sequences |

This post uses **T2VA on 8× B300 as its only canonical benchmark**. It covers
the base, FastVideo's FastH3, quantization, and kernel paths without introducing
reference-media preprocessing or another hardware matrix. FL2VA, Ref2VA, and
other accelerators remain important capabilities documented by their recipes
and implementation sources.

For production users, the relevant question is not simply whether one request
completes. A useful deployment has to balance five objectives:

1. client-visible end-to-end latency;
2. sustained node throughput and tail latency under an explicit arrival model;
3. device HBM, host RAM, and checkpoint-storage requirements;
4. video, audio, and reference-conditioning quality; and
5. operational behavior, including startup, warmup, failure recovery, and
   output transport.

## 2. Benchmark contract and evidence rules

### 2.1 Frozen canonical workload

Every comparable row uses one official T2VA workload before platform-specific
tuning:

| Control | Canonical value |
|---|---|
| Task | T2VA through the FL2VA partition; no reference media |
| Output | 10.0 seconds requested; 1344×768, 243 aligned frames at 24 FPS, 10.125 seconds encoded |
| Base schedule | 50 requested sigma points and 49 expected DiT forwards; record both |
| Prompt | The official MiniMax H3 model-card `case-T2VA` H3-Context-IR output, frozen at model revision `42ed227e`; SHA-256 `98f36b879692095e099ae824c18d9e93e7006a490e082fd474a5f531769dcf06` |
| Seed | `0`, matching the official H3-Base script |
| vLLM-Omni | [`main@b81aeb7`](https://github.com/vllm-project/vllm-omni/commit/b81aeb7b86837f6fe8956f3aef83798ad26c5a26) with vLLM `v0.28.0`; dense BF16 and Fast Ulysses |
| Diffusers lane | Diffusers `v0.40.0`, PyTorch `2.13.0+cu130`, and Transformers `5.14.1`; eight replicated-weight ranks with native context parallelism, Ulysses8, and Ring1 |
| Model | [MiniMax H3 `42ed227e`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe) |
| Repetitions | One full-shape feasibility request, also recorded as the excluded compile/kernel warmup, then at least two measured repetitions per claimed A/B; the B300 Diffusers run uses five |
| Output checks | HTTP/process success; full H.264/AAC decode; 1344×768, exactly 243 frames at 24 FPS; 32 kHz stereo audio; nonzero frame variance and audio RMS; prompt-adherence review |
| Evidence | Representative MP4 outputs are included in this article and linked from the result tables |

The canonical prompt is the 380-word structured output shown in the official
[MiniMax H3 `case-T2VA`](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/README.md#case-t2va):
a two-shot space-opera sequence with a fleet jump, a captain's physical
reaction, synchronized bridge impacts and room tone, and an orchestral
rise-and-cut. Contributors retrieve that expanded prompt once, verify its
SHA-256, and pass it directly to H3-Base. The hosted H3-Context-IR call is not
part of readiness or request latency and must not be rerun independently per
platform.

The article keeps three comparison lanes separate:

1. **Lossless runtime A/B:** 50 sigma points, BF16 checkpoint weights, dense
   full attention, and no adapter, cache, sparsity, or quantization. Declare
   any capacity-required component staging; keep vLLM-Omni DLO in Section 5.
2. **Accelerated-path A/B:** change one adapter, numerical format, cache, or
   sparse-attention policy and attach same-seed quality evidence.
3. **Production-topology study:** change placement or parallelism to measure a
   latency/throughput/memory frontier rather than a kernel speedup.

The first feasibility request is the stop condition. OOM, accelerator error,
invalid media, missing audio, failed quality gates, or an unexpected backend
fallback stops the profile before repeated measurement. Tail latency requires
a separately declared arrival process and enough samples; it is never inferred
from the two single-request repetitions.

### 2.2 Measurement methodology

#### Headline latency

The cross-runtime headline is client-visible latency: synchronous request
submission through receipt of the complete MP4. Model download, checkpoint
conversion, compilation, startup, and warmup remain outside that interval.
Diffusers reports this E2E value only.

#### vLLM-Omni timing hierarchy

For diagnosis, vLLM-Omni exposes a hierarchy of native timing boundaries:

| Source label used below | Corresponding object | Scope |
|---|---|---|
| `client` | `curl` | The external HTTP request, through receipt of the complete MP4 |
| `request` | `RequestE2EStats` | One generation handled by the orchestrator; it may traverse one or more stages |
| `stage` | `StageRequestStats` | One independently scheduled execution unit with its own engine, workers, and device group |
| `engine` | `StageRequestStats.diffusion_metrics` | Request admission, execution, asynchronous output readiness, and output formatting inside a diffusion stage |
| `profiler` | `DiffusionPipelineProfiler` | Selected `MiniMaxH3Pipeline` method boundaries: prompt encode, denoise, and VAE decode |
| `server` | HTTP server timer | MP4 encode and mux after the final stage |

The standard MiniMax H3 profile has one diffusion stage. Prompt encoding, DiT
denoising, and both VAE decoders are model operations inside that stage.

```text
Timing boundary                                      Measurement
Client request: complete-response E2E                client: time_total
├─ Orchestrator request                              request: e2e_total_ms
│  ├─ Queue before stage dispatch                    stage: pipeline_timings.queue_wait_ms
│  └─ Diffusion stage: MiniMax H3 generation         stage: stage_gen_time_ms
│     ├─ Diffusion engine: execution                 engine: diffusion_engine_exec_time_s
│     │  ├─ Scheduler queue                          engine: scheduler_queue_wait_s
│     │  ├─ Model operation: prompt encode           profiler: encode_prompt
│     │  ├─ Model operation: DiT denoise loop        profiler: diffuse
│     │  ├─ Model operation: Video VAE decode        profiler: video_vae.decode_latent
│     │  └─ Model operation: Audio VAE decode        profiler: audio_vae.decode_latent
│     ├─ Diffusion engine: output-ready wait         engine: output_ready_wait_time_s
│     └─ Diffusion engine: output formatting         engine: postprocess_time_s
└─ HTTP response
   ├─ MP4 encode and mux                             server: Video response encoding (MP4 bytes)
   └─ Response delivery                              client: included only in time_total
```

These measurements are nested. `stage_gen_time_ms` contains the diffusion
engine work, and `diffusion_engine_exec_time_s` contains the model operations,
so parent and child values are never added together. Divide the `diffuse` time
by the observed DiT-forward count for the per-forward value.

#### Attention A/B boundary

Attention acceleration A/Bs use `diffusion_engine_exec_time_s`. This boundary
includes the model work surrounding attention while excluding output transfer,
response formatting, and MP4 construction. Profiler method timings such as
`diffuse` are diagnostic children of that boundary and are not used as the A/B
denominator.

#### B300 placement and reproduction

Each result also carries this compact placement manifest:

| Profile | Encoder | DiT | Video/audio VAE | Output |
|---|---|---|---|---|
| B300 attention A/B | 8 GPUs, text-encoder TP8, replicated vision tower | 8 GPUs, TP1, Ulysses8, Ring1, Fast Ulysses, BF16 `TRTLLM_ATTN` | VAE patch parallel 8, tile mode | Synchronous HTTP response, direct-planar MP4 path |

The server remains resident for one excluded full-shape warmup and the measured
requests. The external client boundary is:

```bash
curl -sS -o output.mp4 \
  -w 'client_e2e_s=%{time_total}\n' \
  -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  -F "prompt=<${PROMPT_FILE}" \
  -F 'width=1344' -F 'height=768' -F 'fps=24' \
  -F 'num_inference_steps=50' -F 'seed=0' \
  -F 'extra_params={"task":"t2va","duration":10.0,"aspect_ratio":"16:9","flow_shift":12.0,"audio_flow_shift":3.0}'
```

Enable the native pipeline profiler on the same eight-GPU configuration for
the diagnostic run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni --host 0.0.0.0 --port "${PORT}" --trust-remote-code \
  --task-type fl2va \
  --num-gpus 8 --usp 8 --ring 1 --ulysses-a2a-permute \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend TRTLLM_ATTN \
  --enable-diffusion-pipeline-profiler --log-stats \
  2>&1 | tee vllm-omni-breakdown.log
```

The log contains the synchronized `DiffusionPipelineProfiler` timers,
`RequestE2EStats`, `StageRequestStats`, and the MP4 encode-and-mux timer. The
profiled request is diagnostic; the unprofiled client requests provide the
headline latency. Nsight Systems verifies GPU work and transfers but does not
replace these request-level timers.

### 2.3 Benchmark scope and deployment recipes

The benchmark covers **8× B300 only**: one lossless Diffusers/vLLM-Omni A/B,
isolated lossless kernel/transport checks, and selected four-step or
precision-changing profiles. Only the selected four-step profile receives the
5/10/15-second duration sweep. Other platforms are deployment guidance, not
cross-hardware comparisons:

| Deployment | Maintained guidance |
|---|---|
| H200 and other high-memory CUDA systems | [Full MiniMax H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md) |
| RTX PRO 5000 Blackwell | [Dedicated RTX PRO 5000 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-5000.md) |
| RTX 4090 / 5090 | [RTX 4090](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-4090.md) / [RTX 5090](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-5090.md) |
| DGX Spark GB10 | [GB10 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-Spark-GB10.md) |
| AMD ROCm | [ROCm section of the full recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#amd-rocm-gfx942--gfx950) |
| Ascend NPU | Vendor-specific recipes and validation; no benchmark claim in this post |

## 3. Lossless runtime optimization

Before changing model precision or skipping computation, vLLM-Omni first
removes systems overhead from the reference MiniMax H3 pipeline. The
optimizations in this section keep the released BF16 weights, the 50-point
schedule, and dense attention coverage; Section 4 covers methods that trade
quality or precision for additional speed. We use **lossless** in this
practical sense, not to claim bitwise-identical floating-point execution.

### 3.1 Long-sequence attention and communication

MiniMax H3 denoises text, audio, and video tokens as one long packed sequence.
For the canonical workload, 58,758 valid tokens occupy a 58,816-token aligned
buffer, and Ulysses distributes that sequence across eight GPUs. vLLM-Omni
reduces unnecessary work at three boundaries:

- **TRTLLM attention and packed sequences ([PR #5283](https://github.com/vllm-project/vllm-omni/pull/5283),
  [PR #5779](https://github.com/vllm-project/vllm-omni/pull/5779)).**
  `TRTLLM_ATTN` is the default attention backend for MiniMax H3 on datacenter
  Blackwell GPUs. Valid sequence lengths are passed to the backend, and
  structural suffix padding is removed before attention.
- **Rank-local model boundaries ([PR #6173](https://github.com/vllm-project/vllm-omni/pull/6173)).**
  Each rank constructs only its own embedding and RoPE rows. After the
  transformer, the compact 128-channel projection is gathered instead of the
  5,376-channel hidden state.
- **Fast Ulysses transport ([PR #6340](https://github.com/vllm-project/vllm-omni/pull/6340)).**
  NCCL SymmetricMemory exchanges shards directly in the attention layout,
  eliminating the separate relayout around the all-to-all. It is enabled with
  `--ulysses-a2a-permute`.

Together, these changes preserve dense attention while reducing padding,
global tensor materialization, communication volume, and layout conversion.

### 3.2 Fused DiT operators

The DiT repeatedly applies small elementwise operations around its matrix
multiplications. Fusing those operations reduces kernel launches and avoids
writing intermediate tensors to memory:

- **Fused Q/K RMSNorm and RoPE ([PR #5990](https://github.com/vllm-project/vllm-omni/pull/5990)).**
  RMSNorm and RoPE run in one kernel for Q and one for K.
- **FP32 fused modulation ([PR #6281](https://github.com/vllm-project/vllm-omni/pull/6281)).**
  Gather, modulation, normalization, and residual work are combined while
  retaining FP32 accumulation.
- **Fused SwiGLU ([PR #6283](https://github.com/vllm-project/vllm-omni/pull/6283)).**
  Separate SiLU and multiply launches are replaced by one fused operation.

### 3.3 Parallel and fused VAE decoding

After denoising, H3 decodes video and audio latents independently. vLLM-Omni
uses VAE patch parallelism to distribute the tiled video decoder across the
eight GPUs. The **exact VAE operator path ([PR #6607](https://github.com/vllm-project/vllm-omni/pull/6607))**
accelerates repeated eager operations inside the video VAE: decoder-block
weight materialization, fused Q/K normalization and RoPE, fused SwiGLU, and
scaled residual updates. Its optimized kernels preserve the reference results
for supported tensor layouts and fall back to the original operations
elsewhere.

This combination shortens the video-decoding critical path without coupling it
to the independently executed audio VAE.

### 3.4 GPU-to-MP4 output path

A generation request is not complete until hundreds of decoded frames have
left the GPU and become an MP4. The original path transferred FP32 frames and
performed several layout and dtype conversions on the CPU. The current path
does each conversion once:

1. **GPU output packing ([PR #6824](https://github.com/vllm-project/vllm-omni/pull/6824)).**
   Decoded FP32 BCTHW frames become contiguous uint8 BTHWC on the GPU, reducing
   the video payload by 75% before transfer.
2. **Pinned transport.** The worker moves that compact payload to the server
   through pinned host memory.
3. **Parallel planar encoding ([PR #6288](https://github.com/vllm-project/vllm-omni/pull/6288),
   [PR #6499](https://github.com/vllm-project/vllm-omni/pull/6499), and
   [PR #6776](https://github.com/vllm-project/vllm-omni/pull/6776)).** Frames
   feed directly into H.264 encoding without constructing another full
   interleaved RGB video buffer.

`FP32 BCTHW on GPU → uint8 BTHWC → pinned D2H/IPC → parallel planar conversion → H.264/AAC MP4`

The HTTP response remains the same H.264/AAC MP4; the optimization removes
redundant copies and reduces the amount of data crossing the process boundary.

### 3.5 Diffusers versus vLLM-Omni A/B

Both runtimes use eight B300 GPUs, the same prompt and seed, 50 sigma points,
and the same complete-MP4 timing boundary. Diffusers keeps one resident
`ModularPipeline`, replicates the weights across eight ranks, and uses native
context parallelism with Ulysses8, Ring1, and dense BF16 attention. vLLM-Omni
uses text-encoder TP8, DiT USP8 with Ring1 and Fast Ulysses, VAE PP8 tile
decode, and dense `TRTLLM_ATTN`.

vLLM-Omni completes the request in 56.917 seconds versus 82.239 seconds for
Diffusers: 30.8% lower latency, or a 1.445× complete-response speedup.

Before using pixelwise or waveform metrics across runtimes, verify that both
implementations consume the same generator state, draw order, latent shapes,
and scheduler grid. If that contract differs, report matched-prompt perceptual
and semantic quality instead of presenting SSIM/PSNR as numerical parity.

| B300 runtime | Model execution (s) | Prompt encode (s) | DiT denoise, total / per step (s) | Video / audio VAE (s) | MP4 encode + mux (s) | Client E2E (s) | Peak reserved HBM (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diffusers | — | — | — | — | — | **82.239** | 151.699 |
| vLLM-Omni | **54.246** | 0.057 | 51.800 / 1.057 | 0.952 / 0.055 | 1.528 | **56.917** | 128.232 |

Diffusers phase timings were not isolated. The vLLM-Omni model-execution value
is the baseline for attention acceleration A/Bs. Its profiler phase values come
from a separate diagnostic request and are nested measurements rather than
terms to sum into E2E. Startup readiness is omitted because the two runtimes
did not use a comparable startup boundary. Generator draw-order parity is not
established, so the comparison makes no pixelwise claim.

## 4. Acceleration paths with explicit quality or precision trade-offs

### 4.1 Four-step adapters: Turbo and FastH3 from FastVideo

The largest denoising win comes from reducing the number of expensive DiT
forwards. [FastH3](https://haoailab.com/blogs/fasth3-preview/) is
[FastVideo](https://github.com/hao-ai-lab/FastVideo)'s four-step DMD2 student
of MiniMax H3. vLLM-Omni currently has two distinct integration models; they
should not be treated as interchangeable LoRAs.

| Path | Integration model | Current scope | Deployment status |
|---|---|---|---|
| [Turbo LoRA](https://github.com/vllm-project/vllm-omni/pull/6476) | Dynamically activated request adapter | FL2VA/T2VA, published four-forward schedule | Merged; [DLO support](https://github.com/vllm-project/vllm-omni/pull/6550) merged |
| [FastVideo FastH3](https://haoailab.com/blogs/fasth3-preview/) | Adapter fused into the checkpoint stream at load time | Dense/Data-Free T2VA | Merged in vLLM-Omni [#6714](https://github.com/vllm-project/vllm-omni/pull/6714); current integration rejects offload and VSA variants |

Turbo is the flexible serving option: a server can keep the base model and
activate the supported adapter per request. FastVideo's FastH3 is a specialized
server profile. Its artifact contains low-rank factors plus full-rank deltas
that an ordinary request-switchable LoRA layer cannot express, so the loader
fuses it before sharding. The sparse FastH3 VSA variants additionally depend on
a backend not yet implemented by this vLLM-Omni integration.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-few-step-adapters.svg" alt="Comparison of request-switchable Turbo LoRA and load-time-fused FastH3 weights" width="100%">
</p>

*Figure 2: Turbo keeps the base weights unchanged and adds request-selected A/B
sidecars, while FastVideo's FastH3 fuses low-rank and full-rank deltas into a
specialized student before sharding. Source contracts: vLLM-Omni
[#6476](https://github.com/vllm-project/vllm-omni/pull/6476),
[#6550](https://github.com/vllm-project/vllm-omni/pull/6550), and
[#6714](https://github.com/vllm-project/vllm-omni/pull/6714).*

The source PRs establish the mechanisms, but their workloads are not mixed
into the canonical comparison. For example, the merged vLLM-Omni FastH3
integration reports a 3.30× framework E2E and 8.37× diffusion-stage
improvement on a different B300 workload. Sections 4.4 and 6 report the frozen
FastH3 result; Turbo remains outside the current benchmark scope.

Any performance table must identify the adapter, number of requested sigma
points, actual DiT forward count, task, attention backend, and quality
comparison. A few-step result is not directly comparable to a 50-step baseline
unless those differences are explicit.

The canonical T2VA matrix compares base H3 and FastH3 under one output contract.
Section 6 turns the merged FastH3 path into a measured low-latency deployment
strategy. Turbo remains the request-switchable alternative when one service
must cover both T2VA and FL2VA, but it is not benchmarked here.

### 4.2 Weight and activation quantization

- **Online FP8.** The merged
  [global FP8 path](https://github.com/vllm-project/vllm-omni/pull/5910)
  starts from the released BF16 checkpoint and quantizes eligible DiT and
  Qwen3-VL text-decoder linears at load time. Embeddings, norms, RoPE, the
  vision tower, both VAEs, and the model's FP32 projections retain their
  declared precision. Resident serving and supported DLO paths are distinct
  profiles and must be reported separately.
- **SVDQuant NVFP4 W4A4.** The merged
  [offline loader](https://github.com/vllm-project/vllm-omni/pull/6162)
  combines an NVFP4 W4A4 base GEMM with a BF16 low-rank correction. The current
  SM103 FL2VA evidence establishes checkpoint and correctness compatibility;
  native fused residual-GEMM performance remains follow-up work.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-quantization-paths.svg" alt="Comparison of online FP8 and offline SVDQuant W4A4 execution paths for MiniMax H3" width="100%">
</p>

*Figure 3: Online FP8 keeps runtime-created FP8 weights plus frozen weight
scales and dynamically quantizes activations. Offline SVDQuant adds a BF16
low-rank correction to an NVFP4 W4A4 base branch. Adapted from the vLLM-Omni
cookbook [online FP8](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/_posts/2026-08-18-online-quantization-fp8.md)
and [SVDQuant](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/_posts/2026-08-16-understanding-pr-6162-svdquant-w4a4-blackwell.md)
explainers.*

Quantization claims include peak HBM, host RAM during startup, checkpoint
storage, latency, and same-seed video/audio metrics. A capacity win is not
automatically a speed win, and a loader correctness result is not a fused
kernel result.

### 4.3 Quantized and Sparse Attention with TRTLLM Attn Backend

On datacenter Blackwell GPUs, MiniMax H3 uses dense BF16 `TRTLLM_ATTN` by
default. The backend also supports two optional lossy acceleration modes:

- **SAGE** quantizes both the QK and PV paths to FP8.
- **Skip-Softmax** uses the QK result to dynamically skip unnecessary Softmax
  and PV computation.

We evaluate a conservative Skip-Softmax configuration that largely preserves
the generated video's visual quality. See the [SAGE quantization](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends/trtllm.md#sage-quantization)
and [Skip-Softmax](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends/trtllm.md#skip-softmax)
documentation for configuration details.

| Attention policy | SAGE config | Skip-Softmax config | Model execution | Speedup | LPIPS vs. dense | Sample |
|---|---|---|---:|---:|---:|---|
| Dense TRTLLM | Off | Off | 54.246 s | 1.000× | 0 | [Video](/assets/figures/2026-08-29-minimax-h3-production-serving/evidence/b300/trtllm_dense.mp4) |
| SAGE FP8 | `dtype_qk=fp8_e4m3`, `q_block_size=1`, `k_block_size=4` | Off | 46.592 s | **1.164×** | 0.4093 | [Video](/assets/figures/2026-08-29-minimax-h3-production-serving/evidence/b300/sage_fp8.mp4) |
| Skip-Softmax | Off | `threshold=0.05`, `disabled_until_timestep=0.97` | 50.029 s | **1.084×** | 0.0917 | [Video](/assets/figures/2026-08-29-minimax-h3-production-serving/evidence/b300/skip_softmax_005_gate097.mp4) |
| SAGE FP8 + Skip-Softmax | `dtype_qk=fp8_e4m3`, `q_block_size=1`, `k_block_size=4` | `threshold=0.05`, `disabled_until_timestep=0.97` | 46.073 s | **1.177×** | 0.4103 | [Video](/assets/figures/2026-08-29-minimax-h3-production-serving/evidence/b300/sage_fp8_skip_005_gate097.mp4) |

### 4.4 Other acceleration paths

The attention and caching methods below are independent of `TRTLLM_ATTN`:

| Path | Current status | Configuration and measurement boundary |
|---|---|---|
| [Sol-Attn for H3](https://github.com/vllm-project/vllm-omni/pull/5851) | Preview | Report dense guards, `tau`, sink tokens, KV splits, and quality gates; use the hardware recipe for platform-specific setup |
| [Dynamic Cache-DiT quality](https://github.com/vllm-project/vllm-omni/pull/5853) | Merged | `quality=lossless` removes the cache policy; `quality=high` installs the H3 profile and needs deployment-specific hit-rate/quality evidence |
| FastH3 VSA variants | Rejected by the current FastH3 path | Sparse student artifacts require a VSA backend not implemented by that integration |

Each row starts from the Section 3 lossless vLLM-Omni result on the same
platform and changes one declared acceleration policy:

| Platform / path | Weights / precision | Sigma points / actual forwards | Attention or cache policy | E2E / speedup | Peak HBM | Video/audio quality | Maturity / artifacts |
|---|---|---|---|---:|---:|---|---|
| B300 / FastH3 | Fused artifact | 5 / 4 | Dense only | 8.678 / 8.710 s on the frozen 10-second request; speedup TBD until the Section 3 lossless row lands | 94.1 GiB per GPU, allocator-reserved | Same-seed repetitions are byte-identical; no cross-path quality A/B run | Merged ([#6714](https://github.com/vllm-project/vllm-omni/pull/6714), `86b85c07`) / Section 6.2 sweep |
| B300 / online FP8 | Runtime FP8 | 50 / 49 | Dense TBD | TBD | TBD | TBD | Merged / TBD |
| B300 / SVDQuant | Offline W4A4 + BF16 correction | 50 / 49 | Dense TBD | TBD | TBD | TBD | Correctness baseline / TBD |

## 5. Production deployment features

### 5.1 Distributed Layerwise Offload

[Distributed Layerwise Offload (DLO)](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload)
keeps a bounded window of DiT layers in device memory and streams the remaining
weights from the host. It supports two materially different execution modes:

- **AllGather:** ranks retain weight shards and reconstruct each active layer
  collectively. This reduces aggregate host residency and can pair data
  parallel replicas with sequence parallelism.
- **Rank-local:** each rank streams the tensors produced by its ordinary model
  loader without reconstructing complete weights. It avoids AllGather but has
  a different host-memory and topology trade-off.

<p align="center">
  <img src="/assets/figures/2026-07-30-distributed-layerwise-offload/dlo_pipeline_last_frame.png" alt="DLO double-buffer pipeline overlapping compute, host-to-device copies, and AllGather" width="100%">
</p>

*Figure 4: DLO prepares layer N+1 with H2D and optional AllGather while layer N
computes, alternating between two bounded device slots. Reused from the
[official DLO post](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload).*

DLO is not a universal speed flag. Its value depends on interconnect, DP/SP
topology, resident-layer count, request concurrency, host memory, and host
bandwidth. This post explains that contract without adding another hardware
matrix; use the [full H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
or the [RTX PRO 5000 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-5000.md)
to select and qualify AllGather or rank-local offload for a target system.

### 5.2 Disaggregated encoding

MiniMax H3 retains approximately 51.5 GB of Qwen3-VL encoder weights in BF16.
The [disaggregated encoder path](https://github.com/vllm-project/vllm-omni/pull/5885)
moves that encoder into an independent vLLM stage:

- Stage 0 runs the Qwen3-VL encoder with its own tensor-parallel topology,
  scheduler, kernels, and prefix cache.
- The orchestrator receives Stage 0 output, and `text_encoder2diffusion` merges
  layer-50 hidden states and token-role tags with the original prompt/media.
- Stage 1 runs H3 diffusion inline without loading a second local encoder.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-encoder-disaggregation.svg" alt="MiniMax H3 request flow through an independently scaled vLLM-native encoder and diffusion stage" width="100%">
</p>

*Figure 5: Every data-flow arrow runs left to right. The encoder owns its
processor, TP/replicas, and prefix cache; the orchestrator combines its typed
conditioning with the original request before dispatching the separately
parallelized DiT/VAE stage. Adapted from vLLM-Omni
[#5885](https://github.com/vllm-project/vllm-omni/pull/5885) and
[RFC #5707](https://github.com/vllm-project/vllm-omni/issues/5707).*

The merged single-node deployment does **not** configure OmniConnector: Stage 0
output returns to the orchestrator, the adapter builds the enriched diffusion
prompt there, and `InlineStageDiffusionClient` owns the Stage 1 DiffusionEngine
in the orchestrator process. RFC #5707 sketches OmniConnector with SHM/RDMA as
a future or cross-node transport option, but that path is outside the current
recipe.

This boundary is primarily a production-architecture feature. It lets encoder
and diffusion capacity scale independently, enables prefix reuse for repeated
presentations, and avoids serializing a decoded raw-video payload across a
process boundary when the diffusion stage is kept inline.

### 5.3 Step execution: functional, not yet beneficial

The merged [MiniMax H3 step-execution path](https://github.com/vllm-project/vllm-omni/pull/5810)
lets the scheduler admit, retire, and abort requests between denoise steps.
For H3, this is currently a functional scheduling capability—not a production
optimization or recommendation.

Existing simultaneous and staggered-arrival tests reduced throughput and
increased latency: one H3 request already presents a long, compute-bound packed
sequence, so co-batching makes each forward more expensive. Keep request mode
as the production default. The remaining useful hypotheses—fast cancellation
and HBM reclamation, low-rate admission, and co-batching only for small
under-utilized requests—are tracked in
[roadmap issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700).
Step mode currently requires `FLASH_ATTN`, does not support cache backends, and
is incompatible with H3 DLO.

### 5.4 Compatibility stays living

Base H3 serves T2VA, FL2VA, and Ref2VA; Turbo serves T2VA/FL2VA; the current
FastH3 path is T2VA-only. Cross-feature composition changes too quickly for
a release-oriented blog snapshot, so the living
[MiniMax H3 feature×feature matrix in issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700)
is the authoritative source for supported, partial, incompatible, and
unverified combinations. The task and hardware commands remain in the
[maintained H3 recipes](https://github.com/vllm-project/vllm-omni/tree/main/recipes/MiniMaxAI),
while Section 3.4 records the merged output-layout and MP4 boundaries.

## 6. FastVideo FastH3 low-latency and real-time deployment

The merged [vLLM-Omni FastH3 integration](https://github.com/vllm-project/vllm-omni/pull/6714)
serves [FastVideo's Dense/Data-Free four-step artifact](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA),
fusing it before sharding. This is a good fit for a dedicated T2VA service where
complete-response latency is the primary objective. Turbo remains the
request-switchable choice when one service must switch adapters or cover FL2VA,
but the measured strategy below uses FastH3.

### 6.1 Recommended B300 profile

Use one resident FastH3 replica across eight B300 GPUs: DiT DP1 × TP1 × USP8
with Ring1, VAE patch-parallel 8 in tile mode, and dense `TRTLLM_ATTN`. On the
frozen `86b85c07` revision, the profiled 10-second request decomposes as:

| Encoder | DiT total / 4 / per-forward | Video + audio VAE | Derived transport | CPU MP4 | Profiled E2E | Clean E2E | Peak HBM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.052 s | 5.532 s / 4 / 1.383 s | 1.247 s combined | 0.881 s | 0.868 s | 8.629 s | 8.678 / 8.710 s | 94.1 GiB/GPU reserved |

The profiler timers are collected in a separate instrumented pass; the clean
E2E values carry the latency claim. Same-seed repetitions are byte-identical,
but production promotion still requires a cross-path, multi-seed quality A/B.

### 6.2 Complete-MP4 real-time reference

The duration sweep keeps the prompt, seed, 1344×768 resolution, 24 FPS,
artifact, four-forward schedule, topology, attention, VAE, output path, and CPU
affinity fixed. H3 aligns the 5/10/15-second requests to 124/243/362 frames; one
feasibility request per shape is excluded before two interleaved measurements.

| Requested / aligned / playback | DiT total / per-forward | Combined VAE | Transport + MP4 | Clean E2E | Client RTF | × real time |
|---|---:|---:|---:|---:|---:|---:|
| 5 s / 124 / 5.175 s | 2.806 s / 0.702 s | 0.637 s | 0.929 s | 4.602 / 4.396 s | 0.891 / 0.851 | 1.123 / 1.175 |
| 10 s / 243 / 10.125 s | 5.532 s / 1.383 s | 1.247 s | 1.749 s | 8.678 / 8.710 s | 0.857 / 0.860 | 1.167 / 1.163 |
| 15 s / 362 / 15.083 s | 9.517 s / 2.379 s | 1.861 s | 2.484 s | 14.177 / 14.059 s | 0.940 / 0.932 | 1.064 / 1.073 |

All six measured requests satisfy `RTF_client ≤ 1.0`, meaning the complete MP4
is ready faster than its playback duration. This is a real-time
**complete-response generation** result, not a live-streaming or
time-to-first-frame claim.

The representative FastH3 outputs below use the same 5/10/15-second duration
classes at 1280×736. They are visual examples, not the 1344×768 timing artifacts
used for the table above.

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;margin:1rem 0 1.5rem;">
  <figure style="margin:0;">
    <video controls preload="metadata" playsinline style="width:100%;background:#000;border-radius:6px;">
      <source src="{{ '/assets/figures/2026-08-29-minimax-h3-production-serving/fast-h3-5s.mp4' | relative_url }}" type="video/mp4">
    </video>
    <figcaption><strong>5-second request</strong><br>124 frames · 5.184-second MP4</figcaption>
  </figure>
  <figure style="margin:0;">
    <video controls preload="metadata" playsinline style="width:100%;background:#000;border-radius:6px;">
      <source src="{{ '/assets/figures/2026-08-29-minimax-h3-production-serving/fast-h3-10s.mp4' | relative_url }}" type="video/mp4">
    </video>
    <figcaption><strong>10-second request</strong><br>243 frames · 10.144-second MP4</figcaption>
  </figure>
  <figure style="margin:0;">
    <video controls preload="metadata" playsinline style="width:100%;background:#000;border-radius:6px;">
      <source src="{{ '/assets/figures/2026-08-29-minimax-h3-production-serving/fast-h3-15s.mp4' | relative_url }}" type="video/mp4">
    </video>
    <figcaption><strong>15-second request</strong><br>362 frames · 15.104-second MP4</figcaption>
  </figure>
</div>

### 6.3 Production recommendation

- Use a dedicated FastH3 T2VA service for the lowest validated complete-MP4
  latency; fuse the artifact at startup and pin its checksum with `86b85c07`.
- Prewarm the maximum serving shape before measurement or CUDA-graph capture,
  and keep dense `TRTLLM_ATTN`, USP8, VAE PP8, and the output path fixed.
- Keep quantization, cache, sparse attention, VSA, and alternative Ulysses
  transports disabled until each composition passes the same quality gates.
- Re-profile VAE, transport, and MP4 encoding after every denoising change;
  these stages are already material at four forwards.
- Use a separate Turbo service when request-time adapter switching or FL2VA
  coverage matters more than the dedicated FastH3 latency profile.

## 7. Results and deployment recommendations

The final B300 recommendation will draw from the lossless Diffusers/vLLM-Omni
A/B in Section 3.5, isolated acceleration results in Section 4.4, the four-step
critical path in Section 6.1, and the real-time reference in Section 6.2. It
will name the lowest complete-MP4 latency profile and report its playback RTF,
peak HBM/host RAM, and quality gates beside the claim.

Do not infer a recommendation from nominal FLOPS, multiply gains from unrelated
microbenchmarks, or treat a cold request as steady state. H200, RTX PRO 5000,
consumer GPU, ROCm, and NPU deployments are intentionally not ranked here; use
the recipes in Section 2.3 and validate against the local service objective.

## 8. RL integration with VeRL-Omni

vLLM-Omni also serves as the rollout engine for MiniMax H3 post-training in
[VeRL-Omni](https://github.com/verl-project/verl-omni). Current integrations
cover H3 DiffusionNFT and FlowGRPO paths, preserve joint video/audio rollouts,
and feed CLAP and ImageBind rewards before synchronizing full-weight or LoRA
updates back to the optimized rollout model. The resulting policy can return
to the same production serving stack described above.

This post treats RL as an ecosystem integration, not a serving benchmark. See
the [MiniMax H3 VeRL-Omni recipe](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/minimax_h3/README.md)
for the training architecture, data preparation, rewards, and launch commands.

## 9. Production readiness

### Promotion gate

The living
[feature×feature matrix in issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700)
describes software compatibility; it does not replace platform qualification.
Promote a profile only after its frozen workload passes the stage-level timing,
memory, quality, media-validation, and operational checks in Sections 2–7.
Ascend NPU remains intentionally unclassified until vendor-aligned evidence is
available.

### Operational and security considerations

- Report model download, weight preparation, compilation, and readiness
  separately from warmed request latency.
- Treat HBM, host RAM, checkpoint storage, and output-transport memory as
  independent capacity budgets.
- Preload or allowlist production LoRA adapters. Do not expose arbitrary
  request-supplied adapter paths to an untrusted endpoint.
- Treat the canonical results as T2VA evidence only. FL2VA and Ref2VA require
  separate validation before their recipes are promoted to production SLOs.
- Keep generated-media safety controls and abuse-reporting mechanisms outside
  the model server's performance-critical path, but inside the production
  service boundary.

### Licensing

MiniMax H3 is distributed under the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
not an unconditional permissive model license. Commercial and hosted-service
operators should review the current territorial, attribution, revenue,
acceptable-use, and safeguard requirements with their own counsel before
deployment.

## Future work

- integrate and qualify FastH3 VSA variants with a supported sparse-attention
  backend and multi-seed video/audio gates;
- qualify Sol-Attn against the released dense baseline;
- complete native fused NVFP4 W4A4/SVDQuant kernels, then validate memory,
  latency, and quality end to end;
- implement the
  [chunkwise VAE-to-transport-to-MP4 pipeline](https://github.com/vllm-project/vllm-omni/issues/6872)
  and qualify a GPU-accelerated encoder with a portable CPU fallback;
- disaggregate video and audio VAE decode into independently scalable stages
  with explicit placement, handoff, and failure-recovery contracts;
- identify and validate a useful H3 step-execution case—cancellation,
  staggered-arrival admission, or small-workload co-batching—or retain the
  explicit no-production-benefit conclusion.

## Acknowledgments

<!-- AUTHOR TODO: Add named benchmark collaborators, hardware vendors,
     MiniMax/FastH3 contributors, PR authors, and reviewers after
     the final evidence and author list are agreed. -->

This work builds on contributions across vLLM, vLLM-Omni, VeRL-Omni, MiniMax
H3, [FastVideo](https://github.com/hao-ai-lab/FastVideo), FastH3, Diffusers, and
NVIDIA. We especially thank the FastVideo team for
[open-sourcing FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA)
and collaborating with the vLLM-Omni community on the merged serving
integration. We also thank the contributors who implemented and validated the
model, serving, quantization, offload, kernel, VAE, media, hardware, and
training paths referenced throughout this post.

## References

- [vLLM-Omni repository](https://github.com/vllm-project/vllm-omni)
- [FastVideo repository](https://github.com/hao-ai-lab/FastVideo)
- [FastH3 technical overview](https://haoailab.com/blogs/fasth3-preview/)
- [FastH3 four-step adapter](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA)
- [MiniMax H3 model](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Diffusers MiniMax H3 pipeline](https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/minimax_h3)
- [MiniMax H3 serving recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
- [Diffusion execution modes](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/execution_modes.md)
- [VeRL-Omni repository](https://github.com/verl-project/verl-omni)
