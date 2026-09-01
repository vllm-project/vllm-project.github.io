---
layout: post
title: "MiniMax H3 on vLLM-Omni: From System-Wide Optimization to Real-Time Serving with FastVideo’s FastH3"
author: "vLLM-Omni Team"
summary: "How vLLM-Omni optimizes and scales the complete MiniMax H3 stack, then integrates FastVideo’s four-step FastH3 for generation faster than playback."
description: "An evidence-driven journey from system-wide MiniMax H3 optimization to real-time serving with FastVideo’s FastH3 on vLLM-Omni."
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

> A two-stage optimization story: first reduce overhead across the complete
> MiniMax H3 serving stack, then integrate FastVideo's four-step FastH3 for
> complete-MP4 generation faster than playback.

MiniMax H3 serving is a system problem. One request crosses a large Qwen3-VL
encoder, a long-sequence audio-video DiT, separate video and audio VAEs, device
and process boundaries, and finally H.264/AAC construction. Optimizing only the
DiT leaves substantial latency elsewhere.

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) therefore starts with
the complete resident pipeline: attention and communication, fused DiT
operators, parallel VAE decoding, compact output transport, and parallel MP4
construction. [FastVideo](https://github.com/hao-ai-lab/FastVideo)'s
[FastH3](https://haoailab.com/blogs/fasth3-preview/) then attacks the remaining
dominant term by replacing 49 DiT forwards with four.

On the measured eight-B300 profile, FastH3 produced a complete 10.125-second
MP4 in **8.678-8.710 seconds**. Throughout this post, **real-time** means the
complete response is ready faster than its playback duration. It does not mean
streaming delivery or time to first frame.

## 1. Why MiniMax H3 serving is a system-wide problem

MiniMax H3 jointly generates video and synchronized audio from text, images,
videos, and audio references. Its components have different compute, memory,
and placement requirements:

```text
request -> encoder -> joint audio/video DiT -> video + audio VAEs
        -> GPU output preparation -> D2H/IPC -> H.264/AAC MP4
```

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-model-pipeline.svg" alt="MiniMax H3 model pipeline from multimodal inputs through shared encoders, joint audio-video diffusion, VAE decode, and MP4 muxing" width="100%">
</p>

*Figure 1: Text uses the H3/Qwen3-VL encoder; visual and audio conditions also
use their corresponding VAEs. Conditioning and noisy target latents form one
packed sequence for joint audio-video denoising, followed by separate decode
and MP4 construction. Sources: the
[MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3),
[vLLM-Omni recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md),
and [Diffusers pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3).*

The released checkpoints cover three serving tasks:

| Task | Inputs | Typical use |
|---|---|---|
| T2VA | Text | Creative generation and synthetic media |
| FL2VA | Text plus first/last images | Controlled transitions and image animation |
| Ref2VA | Mixed image, video, and audio references | Consistent editing and reference-guided generation |

The DiT dominates the base schedule, but it is not the only bottleneck. Encoder
residency affects capacity; VAE decode becomes visible after denoising is
shortened; and raw frames must still cross process boundaries and become an
MP4. That is why the story begins with system-wide optimization.

## 2. Benchmark contract and evidence boundaries

The article keeps two evidence lanes separate:

| Evidence lane | Purpose |
|---|---|
| Base H3: Diffusers versus vLLM-Omni | Measure system-wide runtime optimization under a 50-point dense BF16 schedule |
| FastH3 duration sweep | Measure absolute low-latency and complete-response real-time behavior with four DiT forwards |

The two lanes use valid, frozen experiments, but not the same source SHA,
prompt, seed, and artifact. We therefore do **not** derive a base-to-FastH3
speedup. The article reports the absolute FastH3 latency until a matched A/B is
available.

### 2.1 Frozen controls

| Control | Base H3 system lane | FastH3 lane |
|---|---|---|
| Hardware | 8x NVIDIA B300 | 8x NVIDIA B300 |
| Task | T2VA through FL2VA partition | Dense/Data-Free T2VA only |
| Resolution / FPS | 1344x768 / 24 FPS | 1344x768 / 24 FPS |
| Source | vLLM-Omni [`b81aeb7`](https://github.com/vllm-project/vllm-omni/commit/b81aeb7b86837f6fe8956f3aef83798ad26c5a26) | vLLM-Omni [`86b85c07`](https://github.com/vllm-project/vllm-omni/commit/86b85c078bc041e04aee4c4d9167fb10fb1994c7) |
| Model | MiniMax H3 [`42ed227e`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe) | Same base model plus pinned FastH3 artifact |
| Prompt / seed | Official `case-T2VA` expanded prompt, SHA-256 `98f36b...f06`; seed 0 | Fixed FastH3 prompt; seed 1101 |
| Schedule | 50 sigma points / 49 DiT forwards | 5 sigma points / 4 DiT forwards |
| Topology | Encoder TP8; DiT USP8, Ring1; VAE PP8 tile | One replica; encoder TP8; DiT USP8, Ring1; VAE PP8 tile |
| Attention | Dense BF16 `TRTLLM_ATTN`, Fast Ulysses | Dense `TRTLLM_ATTN`, Fast Ulysses |
| Repetitions | One excluded full-shape warmup, then measured requests | One excluded feasibility request per shape, then two interleaved runs per duration |

Both lanes time from synchronous request submission through receipt of the
complete MP4. Downloads, startup, compilation, and the excluded warmup are
outside that interval. Every accepted output must decode as H.264 video plus
stereo 32 kHz AAC, contain the expected frame count and FPS, have nonzero video
variance and audio RMS, and pass prompt-adherence review.

For FastH3, retain the validated video and audio stream durations and define
`T_media = max(T_video, T_audio)`, the effective complete-MP4 playback duration:

`RTF_client = T_client / T_media`

`RTF_client <= 1.0` is the complete-response real-time criterion. A failed
media check, missing audio, OOM, accelerator error, or unexpected fallback
stops that profile before repeated measurement.

Other hardware is intentionally recipe coverage rather than another result
matrix: [H200 and datacenter CUDA](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md),
[RTX PRO 5000](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-5000.md),
[RTX 4090](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-4090.md),
[RTX 5090](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-5090.md),
[GB10](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-Spark-GB10.md),
and [ROCm](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#amd-rocm-gfx942--gfx950).

## 3. System-wide optimization with vLLM-Omni

The base H3 lane preserves released BF16 weights, 50 sigma points, and dense
attention coverage. The optimizations follow the execution path rather than a
feature catalogue.

### 3.1 Long-sequence attention and communication

H3 denoises text, audio, and video tokens as one long packed sequence. For the
canonical workload, 58,758 valid tokens occupy a 58,816-token aligned buffer.
vLLM-Omni reduces overhead at three boundaries:

- [`TRTLLM_ATTN`](https://github.com/vllm-project/vllm-omni/pull/5283) receives
  valid sequence lengths, and [packed-sequence refinement](https://github.com/vllm-project/vllm-omni/pull/5779)
  removes structural suffix padding.
- [Rank-local boundaries](https://github.com/vllm-project/vllm-omni/pull/6173)
  construct only local embedding/RoPE rows and gather the compact 128-channel
  projection rather than the 5,376-channel hidden state.
- [Fast Ulysses](https://github.com/vllm-project/vllm-omni/pull/6340) uses NCCL
  SymmetricMemory to exchange shards in the layout required by attention,
  removing a separate relayout around the all-to-all.

### 3.2 Fused DiT operators

The 49-forward loop repeatedly applies small operations around its matrix
multiplications. vLLM-Omni fuses Q/K RMSNorm with RoPE
([#5990](https://github.com/vllm-project/vllm-omni/pull/5990)), combines FP32
modulation, normalization, and residual work
([#6281](https://github.com/vllm-project/vllm-omni/pull/6281),
[#6878](https://github.com/vllm-project/vllm-omni/pull/6878)), and replaces
separate SiLU and multiply launches with fused SwiGLU
([#6283](https://github.com/vllm-project/vllm-omni/pull/6283)).

### 3.3 Parallel and fused VAE decoding

After denoising, H3 decodes video and audio independently. VAE patch
parallelism distributes the tiled video decoder across eight GPUs. The
[exact VAE operator path](https://github.com/vllm-project/vllm-omni/pull/6607)
accelerates decoder-block materialization, fused Q/K normalization and RoPE,
fused SwiGLU, and scaled residual updates, with eager fallbacks for unsupported
layouts.

### 3.4 GPU output preparation, transport, and MP4

A request is not complete until hundreds of frames have left the GPU. The
optimized path performs each conversion once:

1. [GPU output preparation](https://github.com/vllm-project/vllm-omni/pull/6824)
   converts decoded FP32 BCTHW frames to contiguous uint8 BTHWC, reducing the
   video payload by 75% before transfer.
2. Pinned D2H and worker-to-engine IPC transport the compact payload.
3. [Direct-planar encoding](https://github.com/vllm-project/vllm-omni/pull/6288),
   a [persistent parallel converter](https://github.com/vllm-project/vllm-omni/pull/6499),
   and support for [transported strided RGB planes](https://github.com/vllm-project/vllm-omni/pull/6776)
   feed H.264 without constructing another full interleaved RGB buffer.

`FP32 BCTHW -> uint8 BTHWC -> pinned D2H/IPC -> planar frames -> H.264/AAC MP4`

### 3.5 Measured base H3 result

Both runtimes use eight B300 GPUs, the same prompt and seed, 50 sigma points,
and the same complete-MP4 boundary. Diffusers uses replicated weights with
native context parallelism; vLLM-Omni uses encoder TP8, DiT USP8/Ring1 with
Fast Ulysses, VAE PP8 tile decode, and dense `TRTLLM_ATTN`.

| Runtime | Model execution (s) | Prompt (s) | DiT total / per-forward (s) | Video / audio VAE (s) | MP4 (s) | Client E2E (s) | Peak HBM/rank (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diffusers | - | - | - | - | - | **82.239** | 151.699 |
| vLLM-Omni | **54.246** | 0.057 | 51.800 / 1.057 | 0.952 / 0.055 | 1.528 | **56.917** | 128.232 |

vLLM-Omni lowers complete-response latency by **30.8%**, a **1.445x** speedup.
Diffusers phase timings were not isolated, and cross-runtime generator draw
order is not established, so this is a matched-deployment result rather than a
pixelwise-parity claim.

> These improvements reduce overhead around denoising. FastH3 attacks the
> remaining dominant term by reducing the denoising loop itself from 49
> forwards to four.

## 4. Scaling the general H3 serving architecture

The general H3 lane combines two different kinds of production controls. DLO
and disaggregated encoding change capacity and placement; optional quantized
weights and approximate attention trade numerical fidelity for memory or
latency. These paths explain how to fit, scale, and accelerate the broader
architecture. They did **not** produce the FastH3 numbers in Section 6.

### 4.1 Distributed Layerwise Offload

[DLO](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload) keeps a
bounded window of DiT layers in HBM while streaming the remainder from host
memory. AllGather mode reconstructs active layers collectively from host
shards; rank-local mode streams the tensors produced by each rank's normal
loader. The right choice depends on interconnect, host bandwidth, memory,
resident-layer count, and request concurrency.

<p align="center">
  <img src="/assets/figures/2026-07-30-distributed-layerwise-offload/dlo_pipeline_last_frame.png" alt="DLO double-buffer pipeline overlapping compute, host-to-device transfer, and AllGather" width="100%">
</p>

*Figure 2: DLO prepares the next layer while the current layer computes. See
the [dedicated DLO article](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload)
for the mechanism and deployment trade-offs.*

#### 8× B300 BF16 DLO Pareto frontier

On the official BF16 MiniMax-H3 FL2VA checkpoint (5.175 s, 1344×768,
SP8/Ulysses8/Ring1/DP1/TP1, AllGather, CUDNN attention), the first request is
excluded for lazy CUDA/cuDNN/JIT work and the remaining two requests are
averaged. The generated video and audio have the expected output shapes.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/b300-dlo-pareto.svg" alt="Scatter plot of B300 DLO steady latency against engine-reported HBM. The non-dominated policies are no offload, then 50, 35, 30, and 0 leading DiT blocks resident; 40, 20, and 10 resident blocks are dominated." width="100%">
</p>

*Figure 3: Latency–memory Pareto frontier. <em>r</em> is the number of resident
DiT blocks. Filled points are non-dominated measurements; open points are
dominated. At <em>r</em> = 35, DLO lowers reported HBM by 37.5% for a 5.1%
latency cost; <em>r</em> = 0 is the minimum-memory endpoint.*

### 4.2 Disaggregated encoding

H3 retains approximately 51.5 GB of Qwen3-VL encoder weights in BF16. The
[disaggregated encoder path](https://github.com/vllm-project/vllm-omni/pull/5885)
moves that one-shot encoder into an independent vLLM stage with its own
placement, tensor parallelism, replicas, queue, kernels, and prefix cache. The
orchestrator combines its layer-50 hidden states and token-role tags with the
original media before the DiT/VAE stage.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-encoder-disaggregation.svg" alt="MiniMax H3 request flow through independently scaled encoder and diffusion stages" width="100%">
</p>

*Figure 4: Encoder and diffusion capacity scale independently. The merged
single-node recipe returns conditioning through the orchestrator and keeps the
diffusion stage inline; it does not configure OmniConnector. SHM/RDMA remains a
future cross-node option in [RFC #5707](https://github.com/vllm-project/vllm-omni/issues/5707).*

### 4.3 Optional quantization and attention acceleration

Section 3 deliberately uses dense BF16 attention and released checkpoint
precision. General H3 deployments can select the following additional paths,
but each is a separate quality-performance profile rather than a lossless
runtime gain.

#### Weight and activation quantization

- **Online FP8.** The merged
  [global FP8 path](https://github.com/vllm-project/vllm-omni/pull/5910)
  starts from the BF16 checkpoint and quantizes eligible DiT and Qwen3-VL
  text-decoder linears at load time. Embeddings, norms, RoPE, the vision tower,
  both VAEs, and precision-sensitive projections keep their declared precision.
- **SVDQuant NVFP4 W4A4.** The merged
  [offline loader](https://github.com/vllm-project/vllm-omni/pull/6162)
  combines an NVFP4 W4A4 base GEMM with a BF16 low-rank correction. Current
  evidence establishes checkpoint and correctness compatibility; a native
  fused residual-GEMM performance path remains future work.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-quantization-paths.svg" alt="Comparison of online FP8 and offline SVDQuant W4A4 execution paths for MiniMax H3" width="100%">
</p>

*Figure 5: Online FP8 creates FP8 weights and scales at load time, then
quantizes eligible activations online. Offline SVDQuant combines an NVFP4 W4A4
base branch with a BF16 low-rank correction. Sources: vLLM-Omni
[#5910](https://github.com/vllm-project/vllm-omni/pull/5910) and
[#6162](https://github.com/vllm-project/vllm-omni/pull/6162), plus the cookbook
[online FP8](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/_posts/2026-08-18-online-quantization-fp8.md)
and [SVDQuant](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/_posts/2026-08-16-understanding-pr-6162-svdquant-w4a4-blackwell.md)
explainers.*

A quantized profile must report peak HBM, startup host RAM, checkpoint storage,
latency, and same-seed video/audio quality. A capacity win is not automatically
a latency win, and loader correctness is not evidence of a fused-kernel gain.

#### B300 Online FP8 capacity and latency

The following dense, resident result isolates Online FP8 from the released BF16
checkpoint. Both rows use 8 B300 GPUs, Ulysses8/Ring1 with Fast Ulysses, encoder
TP8, VAE PP8 tile decode, CUDNN attention, and the 10-second 1344×768 / 24 FPS
request with 50 requested sigma points (49 DiT forwards). One warmup is
excluded; each value is the mean of three measured requests. “Stage generation”
is the native diffusion-stage timer; E2E is offline client wall time through
returned video and audio tensors, excluding MP4 muxing.

| Weights | Stage generation (mean, n=3) | E2E (mean, n=3) | Peak HBM / rank | Result |
|---|---:|---:|---:|---|
| BF16 | 52.572 s | 53.118 s | 87.16 GiB | Lossless baseline |
| Online FP8 | **49.769 s** | **50.331 s** | **53.27 GiB** | 5.3% lower stage time; 38.9% lower peak HBM |

Every measured request returned 243 RGB frames at 1344×768 and 32 kHz stereo
audio. Distinct seeds across the three repetitions establish output shape and
successful generation, not pixelwise equivalence to BF16.

#### Quantized and Sparse Attention

On the canonical B300 base-H3 workload, `TRTLLM_ATTN` provides optional SAGE
FP8 and Skip-Softmax paths. SAGE quantizes QK and PV attention work; Skip-Softmax
uses the QK result to omit selected Softmax and PV computation. The following
table compares them with dense TRTLLM attention on the same B300 workload:

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/trtllm-sage-skip-softmax.jpg" alt="SAGE FP8 QK and PV paths around the BLASST Skip-Softmax main loop" width="100%">
</p>

*Figure 6: SAGE quantizes Q, K, P, and V to FP8 for Q×K and P×V, while
Skip-Softmax uses the [BLASST](https://arxiv.org/abs/2512.12087) tile-level
decision to bypass selected Softmax and P×V tiles.*

| Attention policy | SAGE configuration | Skip-Softmax configuration | Model execution | Speedup | LPIPS vs. dense | Sample |
|---|---|---|---:|---:|---:|---|
| Dense TRTLLM | Off | Off | 54.246 s | 1.000x | 0 | [Video](/assets/figures/2026-08-29-minimax-h3-production-serving/evidence/b300/trtllm_dense.mp4) |
| SAGE FP8 | `dtype_qk=fp8_e4m3`, `q_block_size=1`, `k_block_size=16` | Off | 44.787 s | **1.211x** | 0.3697 | [Video](/assets/figures/2026-08-29-minimax-h3-production-serving/evidence/b300/sage_fp8.mp4) |
| Skip-Softmax | Off | threshold 0.05; disabled until 0.97 | 50.029 s | **1.084x** | 0.0917 | [Video](/assets/figures/2026-08-29-minimax-h3-production-serving/evidence/b300/skip_softmax_005_gate097.mp4) |
| SAGE + Skip-Softmax | `dtype_qk=fp8_e4m3`, `q_block_size=1`, `k_block_size=16` | threshold 0.05; disabled until 0.97 | 43.867 s | **1.237x** | 0.3750 | [Video](/assets/figures/2026-08-29-minimax-h3-production-serving/evidence/b300/sage_fp8_skip_005_gate097.mp4) |

SAGE supplies the larger speedup but changes this prompt's composition
substantially; the **conservative** Skip-Softmax profile stays closer to dense.
Users can choose a higher threshold or enable Skip-Softmax for more denoising
steps to trade quality for additional speed. The
[TRTLLM attention guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends/trtllm.md)
documents the controls.

#### Cache-DiT

[Cache-DiT](https://github.com/vllm-project/vllm-omni/pull/5853) is a
request-level cache policy rather than an attention backend. For H3,
`quality=high` enables dynamic per-step reuse, while `quality=lossless`
restores the reference path. Its hit/miss behavior is deployment-dependent, so
it requires independent latency and quality qualification and is not included
in the attention A/B above.

### 4.4 Compatibility boundaries

| Combination | Status for this article |
|---|---|
| Base H3 + DLO | Supported through the maintained H3 recipes; qualify the selected topology locally |
| Base H3 + DLO + online FP8 | Supported, including the AllGather path through [#6279](https://github.com/vllm-project/vllm-omni/pull/6279); performance and quality still require local qualification |
| Base H3 + disaggregated encoder | Merged single-node path |
| FastH3 + DLO | **Unsupported**: FastH3 fusion occurs in `load_weights()`, while offload installs a different host-weight path |
| FastH3 + disaggregated encoder | **Not yet qualified**; it was not used for the reported FastH3 result |

> **Step execution sidebar.** H3 can admit and abort requests between denoise
> steps ([#5810](https://github.com/vllm-project/vllm-omni/pull/5810)), but
> existing co-batching tests did not improve latency. Request mode remains the
> recommendation while cancellation/reclamation and small under-utilized
> workloads are investigated in [issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700).

## 5. From system optimization to FastH3

[FastH3](https://haoailab.com/blogs/fasth3-preview/) is FastVideo's four-step
DMD2 student of MiniMax H3. It reuses the H3 encoder, video VAE, audio VAE,
tokenizers, and schedulers, but reduces the denoising loop to four transformer
forwards over five sigma positions.

The integration is a collaboration across two layers:

- **FastVideo** develops and releases the distilled student and adapter
  artifacts.
- **vLLM-Omni** validates the artifact, fuses it while the checkpoint streams
  in, shards the fused weights, and serves it through the optimized attention,
  VAE, transport, and MP4 path.

FastH3 is not a normal request-switchable LoRA. Besides low-rank factors, its
artifact carries full-rank deltas and replacement weights that an ordinary
LoRA layer cannot represent. vLLM-Omni therefore fuses the artifact before
sharding rather than activating it per request.

<p align="center">
  <img src="/assets/figures/2026-08-29-minimax-h3-production-serving/h3-few-step-adapters.svg" alt="Comparison of request-switchable Turbo LoRA and load-time-fused FastVideo FastH3" width="100%">
</p>

*Figure 7: Turbo leaves base weights unchanged and applies request-selected A/B
sidecars. FastH3 fuses low-rank and full-rank changes into a dedicated student
before sharding. Sources: Turbo [#6476](https://github.com/vllm-project/vllm-omni/pull/6476),
DLO support [#6550](https://github.com/vllm-project/vllm-omni/pull/6550), and
FastH3 integration [#6714](https://github.com/vllm-project/vllm-omni/pull/6714).*

| Profile | Activation model | Task scope | When to choose it |
|---|---|---|---|
| Base H3 | Released checkpoint | T2VA, FL2VA, Ref2VA | Full task coverage and compatibility with the general scaling lane |
| Turbo | Request-switchable adapter | T2VA and FL2VA | One service needs request-time switching or FL2VA |
| FastH3 | Load-time-fused dedicated student | Dense/Data-Free T2VA | Lowest validated latency on a dedicated T2VA endpoint |

FastH3 v1 rejects offload and VSA variants, accepts T2VA only, requires its
four-forward schedule and checkpoint flow shifts, and cannot accept another
request-time LoRA. These are serving contracts, not tuning suggestions.

## 6. Real-time FastH3 serving on B300

This section reports the absolute FastH3 result on vLLM-Omni `86b85c07`. It
does not divide by the base H3 result from a different source/prompt/seed.

### 6.1 Pin the artifact

The measured Dense/Data-Free artifact is pinned to Hugging Face revision
`bcf40ca6f457ed66f8badf13514943e390205fca`:

```bash
FASTH3_REV=bcf40ca6f457ed66f8badf13514943e390205fca
FASTH3_DIR=/models/FastH3-LoRA

hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors \
  --revision "$FASTH3_REV" \
  --local-dir "$FASTH3_DIR"

echo "4ce198c83132251b7fd0de2503823aa49c53983f068318f66cb19eaefb7fcc12  $FASTH3_DIR/dense-datafree/adapter_model.safetensors" \
  | sha256sum -c -
```

The adapter is 1,485,626,152 bytes. Pin both the repository revision and file
checksum; the repository name still contains `Preview-v1`, while the matching
vLLM-Omni integration is merged.

### 6.2 Serve and request

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "$H3_MODEL" --omni \
  --host 127.0.0.1 --port 8095 --trust-remote-code \
  --task-type fl2va --served-model-name MiniMaxAI/MiniMax-H3 \
  --num-gpus 8 --usp 8 --ring 1 --ulysses-a2a-permute \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend TRTLLM_ATTN \
  --lora-path "$FASTH3_DIR/dense-datafree/adapter_model.safetensors"
```

```bash
curl -sS -X POST http://127.0.0.1:8095/v1/videos/sync \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F 'num_inference_steps=4' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":10.0,"flow_shift":12.0,"audio_flow_shift":3.0}' \
  -o fasth3_10s.mp4
```

The service uses one FastH3 replica, encoder TP8, DiT DP1 x TP1 x USP8 with
Ring1 and Fast Ulysses, VAE PP8 tile decode, dense `TRTLLM_ATTN`, and the
standard compact output/MP4 path.

### 6.3 Ten-second critical path

Profiler timers come from a separate instrumented pass; clean E2E carries the
latency claim.

> **Raw benchmark bundle — pending publication gate.** The stable bundle has
> not yet been published. Before publication, this
> [evidence-handoff requirement](https://github.com/vllm-project/vllm-project.github.io/pull/315#issuecomment-5459581336)
> must be replaced by a bundle URL containing raw clean/profiler samples, logs,
> the environment manifest, media metadata and hashes, and topology evidence
> for both the critical-path row and duration sweep.

| Encoder | DiT total / 4 / per-forward | Video + audio VAE | Derived transport | CPU MP4 | Profiled E2E | Clean E2E | Peak HBM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.052 s | 5.532 s / 4 / 1.383 s | 1.247 s combined | 0.881 s | 0.868 s | 8.629 s | **8.678 / 8.710 s** | 94.1 GiB/GPU reserved |

### 6.4 Five-, ten-, and fifteen-second sweep

The sweep holds prompt, seed, resolution, artifact, schedule, topology,
attention, VAE, output path, and CPU affinity fixed. H3 aligns the requested
durations to 124, 243, and 362 frames.

| Requested / aligned | Video / audio duration | DiT total / per-forward | Combined VAE | Transport + MP4 | Clean E2E | Client RTF | x real time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 s / 124 | 5.167 / 5.175 s | 2.806 s / 0.702 s | 0.637 s | 0.929 s | 4.602 / 4.396 s | **0.889 / 0.849** | **1.125 / 1.177** |
| 10 s / 243 | 10.125 / 10.125 s | 5.532 s / 1.383 s | 1.247 s | 1.749 s | 8.678 / 8.710 s | 0.857 / 0.860 | 1.167 / 1.163 |
| 15 s / 362 | 15.083 / 15.083 s | 9.517 s / 2.379 s | 1.861 s | 2.484 s | 14.177 / 14.059 s | 0.940 / 0.932 | 1.064 / 1.073 |

All six measured requests satisfy `RTF_client <= 1.0`: complete-MP4 generation
is faster than playback for every tested duration.

### 6.5 Representative outputs and quality boundary

These supplied FastH3 outputs cover the same 5/10/15-second duration classes.
They are 1280x736 representative examples, not the 1344x768 timing artifacts
used in Section 6.4.

| Request | Frames | MP4 duration | Resolution / FPS | Video |
|---:|---:|---:|---:|---|
| 5 s | 124 | 5.184 s | 1280x736 / 24 FPS | [Open MP4](/assets/figures/2026-08-29-minimax-h3-production-serving/fast-h3-5s.mp4) |
| 10 s | 243 | 10.144 s | 1280x736 / 24 FPS | [Open MP4](/assets/figures/2026-08-29-minimax-h3-production-serving/fast-h3-10s.mp4) |
| 15 s | 362 | 15.104 s | 1280x736 / 24 FPS | [Open MP4](/assets/figures/2026-08-29-minimax-h3-production-serving/fast-h3-15s.mp4) |

The links above are presentation examples. The publication-grade timing and
media evidence remains subject to the raw-bundle gate in Section 6.3.

| Quality gate | Status |
|---|---|
| Repeated same-seed FastH3 output | Byte-identical in the measured runs |
| Media structure | Expected frames/FPS, H.264, stereo AAC, nonzero video/audio signal |
| Matched base-versus-FastH3 multi-seed quality | **Pending; no parity claim** |

Reducing denoising exposes the new tail: on the 10-second profile, combined
VAE, derived transport, and CPU MP4 account for roughly three seconds in the
instrumented path. [RFC #6872](https://github.com/vllm-project/vllm-omni/issues/6872)
proposes overlapping VAE chunks, D2H/IPC, and encoding rather than optimizing
these stages in isolation.

## 7. Production guidance and limitations

The deployment choice is now concrete:

| Requirement | Recommended profile |
|---|---|
| Full T2VA, FL2VA, and Ref2VA coverage | Base H3 with the system-wide stack |
| Request-time adapter switching or FL2VA with four-forward Turbo | Separate Turbo service |
| Lowest validated T2VA complete-response latency | Dedicated FastH3 service from Section 6 |
| Host-memory-driven fit or independently scaled encoder capacity | Base H3 DLO or disaggregated-encoder lane; qualify locally |

Do not combine the reported FastH3 profile with DLO, VSA, quantization, cache
policies, alternative Ulysses transports, or encoder disaggregation without a
new correctness, quality, memory, and latency qualification. The living
[feature compatibility tracker](https://github.com/vllm-project/vllm-omni/issues/5700)
records cross-feature work, but it can lag merged implementation. Verify the
linked PRs and maintained recipes before selecting a production combination.

Before promotion:

- pin model, adapter, source, container, and codec-package revisions;
- preserve one excluded full-shape warmup and raw measured samples;
- publish the stable raw benchmark bundle required beside the Section 6 tables;
- validate every MP4 and retain representative outputs plus hashes;
- complete the matched multi-seed FastH3 quality comparison;
- monitor HBM, host RAM, CPU affinity, failures, and fallback counters; and
- re-profile VAE, transport, and MP4 after every denoising change.

MiniMax H3 uses the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
Commercial and hosted-service operators should review its current territorial,
attribution, revenue, acceptable-use, and safeguard requirements with counsel.

For post-training, vLLM-Omni can also serve H3 rollouts in
[VeRL-Omni](https://github.com/verl-project/verl-omni); training is ecosystem
coverage rather than part of this serving benchmark.

## 8. Conclusion and focused future work

System-wide optimization makes the complete H3 pipeline efficient. FastVideo's
four-forward student then moves the dedicated T2VA profile into
faster-than-playback complete-response generation on the measured B300 system.

The remaining work follows directly from that progression:

- integrate and qualify FastH3 VSA variants and native fused NVFP4 kernels;
- integrate and qualify the [Sol-Attn](https://github.com/vllm-project/vllm-omni/pull/5851)
  on-the-fly sparse-attention backend across target Blackwell platforms and
  multi-seed workloads;
- complete a matched base/FastH3 multi-seed quality evaluation;
- implement the [chunkwise VAE-to-transport-to-MP4 pipeline](https://github.com/vllm-project/vllm-omni/issues/6872)
  and qualify a GPU encoder; and
- qualify FastH3 composition with encoder disaggregation or other scaling
  features rather than inferring compatibility.

## Acknowledgments

<!-- AUTHOR TODO: Add final named benchmark collaborators and reviewers after
     the evidence and author list are agreed. -->

This work builds on contributions across vLLM, vLLM-Omni, VeRL-Omni, MiniMax
H3, [FastVideo](https://github.com/hao-ai-lab/FastVideo), FastH3, Diffusers, and
NVIDIA. We especially thank the FastVideo team for
[open-sourcing FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA)
and collaborating with the vLLM-Omni community on the merged serving
integration. We also thank the contributors who implemented and validated the
model, serving, attention, kernels, VAE, transport, media, and training paths.

## Appendix A. Reproducibility

### A.1 Timing hierarchy

The vLLM-Omni measurements are nested; parent and child values must not be
added together:

| Boundary | Scope |
|---|---|
| Client | Request submission through complete MP4 receipt |
| Request | Orchestrator lifetime across stages |
| Stage | One independently scheduled engine/device group |
| Engine | Queue, model execution, output-ready wait, and formatting |
| Profiler | Prompt, DiT, and VAE method boundaries inside engine execution |
| Server | H.264/AAC encode and mux after the final stage |

Per-forward denoise time divides by the actual DiT forward count, not the
requested sigma-position count. Profiler values come from separate diagnostic
requests and do not replace unprofiled client latency.

### A.2 Base H3 vLLM-Omni reproduction

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "$H3_MODEL" --omni \
  --host 127.0.0.1 --port 8093 --trust-remote-code \
  --task-type fl2va --num-gpus 8 --usp 8 --ring 1 \
  --ulysses-a2a-permute --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend TRTLLM_ATTN
```

The canonical request uses the prompt and seed in Section 2, 50 requested
sigma points, flow shift 12, audio flow shift 3, and a 10-second target.

## References

- [vLLM-Omni repository](https://github.com/vllm-project/vllm-omni)
- [FastVideo repository](https://github.com/hao-ai-lab/FastVideo)
- [FastH3 technical overview](https://haoailab.com/blogs/fasth3-preview/)
- [FastH3 four-step adapter](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA)
- [MiniMax H3 model](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Diffusers MiniMax H3 pipeline](https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/minimax_h3)
- [MiniMax H3 serving recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
- [Distributed Layerwise Offload](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload)
- [Feature compatibility tracker](https://github.com/vllm-project/vllm-omni/issues/5700)
- [Chunkwise output pipeline RFC](https://github.com/vllm-project/vllm-omni/issues/6872)
- [VeRL-Omni repository](https://github.com/verl-project/verl-omni)
