---
layout: post
title: "Qwen3-Omni in vLLM-Omni: Staged Serving for Real-Time Multimodal Generation"
author: "vLLM-Omni Team"
summary: "How vLLM-Omni serves and optimizes Qwen3-Omni with staged Thinker-Talker-Code2Wav execution, batching, CUDA Graphs, async chunking, replicas, hot-path cleanup, and perf validation."
image: /assets/figures/2026-06-27-qwen3-omni-optimization/qwen3-omni-serving-flow.svg
social_image: /assets/figures/2026-06-27-qwen3-omni-optimization/qwen3-omni-serving-flow.svg
read_time_minutes: 12
tags:
  - performance
  - multimodal
  - vllm-omni
---

Qwen3-Omni combines multimodal understanding with speech generation. This post explains how [vLLM-Omni](https://github.com/vllm-project/vllm-omni) serves it as a staged pipeline and optimizes each stage for online workloads.

## TL;DR

vLLM-Omni's Qwen3-Omni serving stack includes:

- **A three-stage pipeline:** Thinker for multimodal reasoning, Talker for speech codec generation, and Code2Wav for waveform reconstruction.
- **OpenAI-compatible serving:** `/v1/chat/completions` is the primary endpoint for Qwen3-Omni text and audio generation.
- **Async chunking:** Thinker-to-Talker and Talker-to-Code2Wav handoffs emit partial payloads instead of waiting for full-stage completion.
- **Batching, replicas, and CUDA Graphs:** stage-level batching, Talker/Code2Wav replicas, and per-stage graph capture on Thinker, Talker, and Code2Wav improve high-concurrency throughput.
- **Performance validation:** controlled benchmark sweeps and DFX perf runs show lower TTFP, lower RTF, and higher throughput as each optimization layer is enabled.

## Quickstart

The default Qwen3-Omni deploy profile is resolved automatically when serving the model with `--omni`:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
```

For explicit configuration, pass the staged deploy profile:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/deploy/qwen3_omni_moe.yaml
```

Requests should use `/v1/chat/completions`. Set `modalities` in the request body to declare output types — e.g. `["text"]` for text only, or `["text", "audio"]` for text plus speech.

## The Qwen3-Omni Serving Model

Text-only LLM serving is one loop: prefill, decode, detokenize. Qwen3-Omni adds two speech stages after multimodal reasoning, each with a different compute profile:

```text
Thinker   -> multimodal understanding + text generation
Talker    -> hidden states and embeddings to RVQ codec codes
Code2Wav  -> codec codes to waveform audio
```

<p align="center">
  <img src="/assets/figures/2026-06-27-qwen3-omni-optimization/qwen3-omni-serving-flow.svg" alt="Qwen3-Omni serving flow in vLLM-Omni, showing multimodal inputs, Thinker, Talker, Code2Wav, and text/audio outputs." width="100%">
  <br>
  <em>Figure 1: Qwen3-Omni serving in vLLM-Omni is a staged dataflow: Thinker produces text and hidden states, Talker produces codec codes, and Code2Wav reconstructs audio.</em>
</p>

Three details drive the serving design:

1. **Stage handoffs carry tensors, not just token IDs.** Talker needs Thinker hidden states, prompt embeddings, and speech control tokens; Code2Wav needs codec codes and chunk metadata such as `left_context_size`.
2. **First-audio latency is a first-class metric.** Connector chunk size sets TTFP; decode left context preserves audio continuity. vLLM-Omni keeps chunk cadence (`codec_chunk_frames`, `initial_codec_chunk_frames`) separate from vocoder context (`codec_left_context_frames`).
3. **Output routing follows request modalities.** Each `/v1/chat/completions` call carries a `modalities` list that declares the desired outputs. vLLM-Omni reads it per request: `"text"` alone stops after Thinker, while `"audio"` (typically together with `"text"`) continues through Talker and Code2Wav — so the same deployed pipeline can serve text-only and speech-generation requests without separate server configs.

| Stage | Component | Mode | Output |
|---|---|---|---|
| 0 | Thinker | `LLM_AR` | Text + latent side outputs |
| 1 | Talker | `LLM_AR` | RVQ codec codes |
| 2 | Code2Wav | `LLM_GENERATION` | Audio waveform |

Custom processors build each handoff in full-payload or async-chunk mode; shared-memory connectors move payloads between stages. At concurrency 32, a Batch-only baseline still reaches only **35.0** output tok/s and **4975 ms** mean audio TTFP (RTF **0.98**). The sections below optimize this pipeline stage by stage.

## Optimization Overview

Different parts of the Qwen3-Omni pipeline hit different bottlenecks. vLLM-Omni does not apply one fixed recipe; each optimization targets a specific stage or handoff. The sections below follow the order we validated in the benchmark sweep. Each one answers three questions: **why** the problem exists, **what you gain**, and **why that gain follows** from the mechanism.

| Technique | Target stage / path | Problem it addresses | Primary benefit |
|---|---|---|---|
| Stage decomposition | Thinker → Talker → Code2Wav | One monolithic loop cannot tune batching, graphs, or devices per sub-path | Independent runtime policy per stage |
| AR + Code2Wav batching | Talker MTP path, Code2Wav async chunks | Single-request micro-work leaves GPU underutilized at high concurrency | Higher occupancy and output tok/s |
| CUDA Graph | Thinker / Talker / Code2Wav decode paths | Repeated CPU kernel dispatch on every decode step | Lower TPOT/RTF; ~4× throughput jump in sweep |
| Async chunk + SHM connectors | Thinker→Talker, Talker→Code2Wav | Full-payload barriers inflate TTFP | Pipelined handoffs; largest TTFP reduction |
| Async omni output | Client-facing audio emission | Synchronous client I/O stalls stage workers | Throughput recovery without TTFP regression |
| Stage replicas | Talker, Code2Wav | Speech stages queue up while Thinker has headroom | Horizontal scale on bottleneck stages only |
| Hot-path cleanup | Talker code predictor, connector payloads | Per-step Python/allocation/sync tax compounds over long utterances | Lower per-step latency; stacks with all layers above |

The rest of this section follows Qwen3-Omni through that stack in the order we validated it.

## Qwen3-Omni: A Full Optimization Path

Among omni-modality models in vLLM-Omni, Qwen3-Omni has the most complete three-stage speech path: Thinker for multimodal reasoning, Talker for codec generation, and Code2Wav for waveform reconstruction.

We validated each layer with a controlled benchmark sweep at concurrency 32 (`128` prompts, `5` warmups, three GPUs mapped as `0/1/2`: Thinker on GPU 0, Talker and Code2Wav each with 2 replicas on GPUs 1 and 2). Each configuration restarted the server with an isolated deploy profile. The charts in [Validation Results](#validation-results) start from the **Batch** baseline and add one optimization at a time.

| Step | Config added | Output tok/s | Mean audio TTFP | Mean audio RTF |
|---|---|---:|---:|---:|
| Baseline | Batch | 35.0 | 4975 ms | 0.98 |
| + CUDA Graph | Graph capture on Thinker, Talker, Code2Wav | 135.2 (+286%) | 1798 ms (−64%) | 0.38 (−61%) |
| + Async chunk | Async chunk stage handoffs | 152.5 (+13%) | 497 ms (−72%) | 0.39 |
| + Async output | Async omni output path | 137.4 | 464 ms (−7%) | 0.31 (−21%) |
| + Stage replicas | 2× Talker + 2× Code2Wav | 185.8 (+35%) | 482 ms | 0.31 |

<p align="center">
  <img src="/assets/figures/2026-06-27-qwen3-omni-optimization/qwen3-omni-optimization-stack.svg" alt="Qwen3-Omni optimization stack in vLLM-Omni, showing staged execution, batching and replicas, CUDA Graph, async chunking, and hot-path cleanup." width="100%">
  <br>
  <em>Figure 2: Qwen3-Omni performance comes from optimizing the staged dataflow, stage runtime, and decode hot path together.</em>
</p>

### 1. Stage Decomposition: Making Internal Boundaries Explicit

**Why.** Qwen3-Omni is not one homogeneous decode loop. Thinker runs multimodal AR text generation, Talker runs a codec-predictor AR path, and Code2Wav runs parallel vocoder decode. Folding all three into a single serving path forces one batching policy, one graph policy, and one device layout on workloads with very different shapes and bottlenecks.

**What you gain.** Explicit stages let vLLM-Omni treat each component as an independent runtime: separate `max_num_seqs`, sampling params, connectors, graph/eager policy, and optional replicas. That is the prerequisite for every optimization below — without it, batching and CUDA Graph would still be constrained by whichever sub-path is slowest in a monolithic loop.

**Why it works.** Stage boundaries turn model-internal handoffs into first-class serving objects. Connectors define what crosses each boundary (hidden states, embeddings, codec codes, chunk metadata), and the scheduler can schedule, batch, and scale each stage on its own critical path. Code2Wav can batch vocoder forwards while Talker still runs single-token decode; audio can surface from stage 2 while text still streams from stage 0; only stages 1 and 2 need replicas under concurrent speech load.

In practice, stage 0 owns the tokenizer and multimodal prompt; stage 1 consumes Thinker hidden states and embeddings; stage 2 consumes codec codes and emits audio. The default MoE deploy profile places Thinker on one GPU and Talker/Code2Wav on another, matching their different memory and compute profiles. Without explicit boundaries, connector payloads, async-chunk handoff state, and per-stage execution policy would remain buried inside model code.

### 2. Batching: AR Decode and Code2Wav Micro-Work

**Why.** After stage decomposition, the speech path still spent most GPU time on single-request micro-work. Each Talker decode step runs a short code-predictor forward; each Code2Wav chunk is a small vocoder forward. At concurrency 32, processing those steps one request at a time leaves SMs idle between launches and prevents the scheduler from amortizing fixed per-step costs.

**What you gain.** Batching raises GPU occupancy on the speech-generation side: more requests share the same Talker MTP invocation and the same Code2Wav forward. In reported tests, Code2Wav batching alone cut E2E from 9697 ms to 6435 ms and improved RTF by **62.36%** at high concurrency. Batching is the baseline layer in our sweep — without it, later optimizations have little to stack on.

**Why it works.** Talker and Code2Wav decode steps are largely independent across requests once stage inputs are prepared. The optimized Talker runner collects single-token non-prefill requests into a decode batch, writes `input_ids`, embeddings, last Talker hidden states, and text-step embeddings into preallocated GPU buffers, and invokes the Talker MTP path once. Batched Code2Wav returns lists of audio tensors, reuses `ubatch_slices`, and runs one forward for multiple connector chunks. Fixed kernel launch and Python dispatch costs are paid once per batch instead of once per request.

Even with batching alone, the benchmark sweep at concurrency 32 still reaches only **35.0** output tok/s, **4975 ms** mean audio TTFP, and **0.98** mean audio RTF — near realtime on the audio path. Batching improves utilization but does not remove per-step launch overhead or full-payload stage waits; the next layers address those.

### 3. CUDA Graph: Per-Stage Decode Capture

**Why.** Batching raised occupancy, but each decode step still paid repeated CPU-side kernel dispatch. Qwen3-Omni runs three decode-heavy stages; Talker alone may execute hundreds of short steps per utterance, and each step previously re-launched the same stable operator sequence from Python. At concurrency 32, that launch tax dominated TPOT and kept RTF above realtime even after batching.

**What you gain.** Turning on CUDA Graph for all three stages in the benchmark sweep raises output throughput from **35.0** to **135.2** tok/s (+286%), cuts mean audio TTFP from **4975 ms** to **1798 ms**, and drops mean audio RTF from **0.98** to **0.38**. Most of the win comes from removing launch overhead across Thinker text generation, Talker codec decode, and Code2Wav vocoder forwards together — not from Code2Wav alone.

**Why it works.** CUDA Graph captures a fixed operator sequence once and replays it with minimal CPU work. Each stage has a different capture point, but the principle is the same: decode shapes bucket into stable `(batch, seq, frames)` profiles, so the runtime records the graph at warmup and reuses it on the hot path. In the benchmark sweep, the **CUDA Graph** step sets `enforce_eager: false` on stages 0, 1, and 2 (the **Batch** baseline sets `enforce_eager: true` everywhere).

#### Stage 0 — Thinker: vLLM outer decode graph

The Thinker is an autoregressive multimodal stage (`LLM_AR`). When `enforce_eager` is false, it uses vLLM's standard CUDA Graph capture on the decode path — the same mechanism as text-only LLM serving. This removes repeated CPU-side kernel dispatch during long Thinker generations.

The default deploy profile leaves stage 0 on graph-friendly settings on CUDA. Platform overrides can disable outer capture where the backend is unsafe: for example, XPU sets `enforce_eager: true` and `max_cudagraph_capture_size: 0` on stage 0.

#### Stage 1 — Talker: outer decode graph + compiled code predictor

The Talker stage also runs through vLLM's outer CUDA Graph path when `enforce_eager: false`. Each Talker decode step additionally invokes the **code predictor** — a short re-prefill transformer that emits RVQ codec codes. That inner path is optimized separately:

- **`torch.compile`** fuses the 5-layer predictor forward (`dynamic=False`, `epilogue_fusion=False`) so RMSNorm/RoPE stay numerically aligned with the reference path while still reducing kernel count per step.
- On CUDA, the code predictor does **not** enable a second manual CUDA Graph layer by default (`use_cuda_graphs=False`), because that would conflict with vLLM's Talker `CUDAGraphWrapper`. The outer Talker graph and compiled inner forward are complementary: one captures the AR stage loop, the other fuses the codec-prediction micro-forward.
- On NPU, the same wrapper can fall back to platform graphs (`use_cuda_graphs=True`) where Inductor is unavailable.

Optional prefix-graph buckets (`code_predictor_prefix_graphs` in connector config) can capture additional stable predictor shapes when explicitly enabled.

#### Stage 2 — Code2Wav: inner vocoder graph

Code2Wav is a generation stage (`LLM_GENERATION`), not an AR loop. Its graph path is an **inner** `CUDAGraphDecoderWrapper` rather than vLLM's outer wrapper:

```python
# Enabled during weight load when stage enforce_eager is false
self.code2wav.enable_cudagraph(
    codec_chunk_frames=chunk_frames,
    codec_left_context_frames=left_frames,
)
```

**Shape bucketing from connector config.** Before warmup, the wrapper reads `codec_chunk_frames` and `codec_left_context_frames` from the stage connector config. Capture enumerates the `(batch, num_quantizers, frames)` buckets that async-chunk and full-payload decode will hit at runtime — including the smaller first chunk from `initial_codec_chunk_frames`.

**Vocoder warmup.** `precompute_snake_caches()` runs before graph capture so SnakeBeta activations do not pay repeated setup inside the captured decode loop.

**Chunk dispatch.** In async-chunk mode, `chunked_decode_streaming` delegates stable chunks to `_cudagraph_wrapper.chunked_decode_with_cudagraph`; full-payload paths use the wrapper's batched decode entry points when shapes match captured buckets.

#### Backend policy across all three stages

| Platform | Thinker / Talker | Code2Wav |
|---|---|---|
| CUDA (default) | Outer graph on (`enforce_eager: false`) | Inner graph on (`enforce_eager: false`) |
| ROCm | Outer graph on | **Eager** — `conv_transpose1d` via MIOpen is not capture-safe |
| XPU | Eager, `max_cudagraph_capture_size: 0` | Eager, `max_cudagraph_capture_size: 0` |

### 4. Async Chunk: Inter-Stage Handoffs and Overlapped Transfer

**Why.** CUDA Graph made each stage faster, but the pipeline was still **barrier-synchronized**: Talker could not start until Thinker finished, and Code2Wav could not emit audio until Talker accumulated a full payload. First-audio latency therefore tracked full Thinker generation plus full Talker prefill — even when only a few codec frames were needed to produce the first audible chunk.

**What you gain.** Async chunking is the largest TTFP win in the sweep: mean audio TTFP drops from **1798 ms** (CUDA Graph) to **497 ms**, and from **4975 ms** (Batch alone) overall. Output throughput rises to **152.5** tok/s because pipelined handoffs keep Talker and Code2Wav busy while Thinker is still decoding. In a reported long-run benchmark, enabling async put/get cut E2E from 390.5 s to 271.2 s and mean audio TTFP from 175.2 s to 4.0 s.

**Why it works.** Async chunking turns sequential stage completion into **pipelined partial handoffs**. Thinker emits embedding rows incrementally; Talker accumulates codec frames and slices on `initial_codec_chunk_frames` / `codec_chunk_frames` boundaries; Code2Wav decodes each slice with `codec_left_context_frames` for continuity. The async scheduler and shared-memory connector overlap chunk transfer with stage compute, so the next stage starts work while the previous stage is still decoding.

**Connector chunk policy.** The deploy config controls output chunk cadence independently from Code2Wav decode context:

```yaml
initial_codec_chunk_frames: 4
codec_chunk_frames: 25
codec_left_context_frames: 25
```

The connector can emit a small first codec chunk for lower TTFP while Code2Wav keeps enough left context for audio continuity across chunk boundaries.

**Incremental handoffs instead of full replay.** On the worker connector path, Thinker→Talker async chunking tracks `put_req_chunk` plus pending chunk-0 prefill state; Talker→Code2Wav async chunking accumulates codec rows in `code_prompt_token_ids` and slices them using `codec_chunk_frames`, `initial_codec_chunk_frames`, and `codec_left_context_frames`.

**Transfer overlap.** The async scheduler and shared-memory connector overlap chunk IO with stage compute. Large payloads use key-addressed POSIX SHM; small payloads can inline to avoid mmap and file-lock overhead. Pending keys are tracked and unlinked on request cleanup so aborted async chunks do not leak `/dev/shm` segments.

### 5. Async Output: Non-Blocking Client Emission

**Why.** Async chunking speeds inter-stage handoffs, but the **client-facing emission path** can still stall stage workers. If Code2Wav must synchronously serialize, buffer, and push each audio chunk through the HTTP serving loop before starting the next decode step, GPU time is lost to I/O and Python scheduling even though the stage-to-stage pipeline is already incremental.

**What you gain.** On top of async chunking in the benchmark sweep, async output keeps mean audio TTFP near **464 ms** while lowering RTF from **0.39** to **0.31** at concurrency 32 — decoupling client emission removes scheduling stalls without giving back the TTFP win from async chunking.

**Why it works.** `use_async_omni_output` decouples chunk production from chunk delivery. Code2Wav can finish a vocoder forward and hand audio to a non-blocking output path while Talker and the connector keep advancing. The serving loop surfaces packets to `/v1/chat/completions` asynchronously instead of making decode workers wait on client I/O.

### 6. Stage Replicas: Scaling Talker and Code2Wav

**Why.** Under concurrent speech generation, Thinker, Talker, and Code2Wav do not saturate equally. Thinker handles one multimodal prompt per request; Talker and Code2Wav execute many short decode steps and vocoder forwards per utterance. At concurrency 32, a single Talker/Code2Wav instance becomes the tail bottleneck even when Thinker still has headroom — replicating the entire pipeline would waste memory duplicating the multimodal stage.

**What you gain.** Adding replicas on top of async output reaches **185.8** output tok/s (+35%) at concurrency 32 — the highest throughput in the sweep — while keeping mean audio TTFP near **482 ms** and RTF near **0.31**. The tracked multi-replica perf suite separately targets **2100 ms** mean audio TTFP and **0.39** mean audio RTF for long text-plus-audio workloads at concurrency 32.

**Why it works.** Stage replication is horizontal scaling applied only where queue depth builds up. One Thinker on GPU 0 fans out to two Talker replicas on GPUs 1 and 2; two Code2Wav replicas share the same pair of GPUs. Load spreads across the speech-generation side without copying the full three-stage pipeline per replica. The benchmark deploy config keeps one Thinker and runs two replicas each for Talker and Code2Wav:

```json
{
  "stage_overrides": {
    "1": {"num_replicas": 2, "devices": "1,2"},
    "2": {"num_replicas": 2, "devices": "1,2"}
  },
  "extra_cli_args": ["--async-chunk"]
}
```

<p align="center">
  <img src="/assets/figures/2026-06-27-qwen3-omni-optimization/qwen3-omni-async-replica.svg" alt="Qwen3-Omni async chunk and stage replica serving, showing one Thinker feeding two Talker replicas and two Code2Wav replicas." width="100%">
  <br>
  <em>Figure 3: Async chunking and stage replicas target the speech-generation side of the pipeline, where Talker and Code2Wav can become the bottleneck under concurrent load.</em>
</p>

### 7. Hot-Path Cleanup: Talker Decode and Connector Payloads

**Why.** Batching, graphs, async chunking, and replicas address structural bottlenecks, but the Talker decode loop still executed hundreds of small per-step costs: Python `generate()` dispatch, repeated `torch.cat` while building connector payloads, and device-to-host reads of decode state that the next step immediately needed back on GPU.

**What you gain.** Hot-path cleanup removes overhead that scales with utterance length. In a long-context single-request test, E2E dropped from 21.28 s to 7.37 s, audio TTFP from 3197 ms to 1796 ms, and audio RTF from 0.71 to 0.28. These changes stack with the layers above and show up in the DFX perf suite baselines rather than as a separate sweep row.

**Why it works.** Each fix targets a repeated micro-cost on the critical path:

**Connector payload construction.** Large async chunks previously paid for repeated `torch.cat` while accumulating Thinker and Talker payloads. Passing decode embeddings per token and trimming redundant payload assembly removes allocation and copy work on every chunk boundary.

**Talker code predictor rewrite.** The older path used Hugging Face `generate()` for very short codec-predictor sequences. That added Python dispatch, dynamic allocation, and KV-cache overhead on every Talker decode step. The optimized path uses re-prefill with SDPA, native GQA, inline top-k sampling, cached module references, and `torch.compile` on the inner transformer (`epilogue_fusion=False` for fp32-safe RMSNorm/RoPE). On CUDA this compile path sits below vLLM's Talker CUDA Graph rather than adding a conflicting second graph layer; see §3 above.

**GPU-resident decode state.** Intermediate keys such as `hidden_states.last`, `hidden_states.trailing_text`, `embed.tts_pad_projected`, and `codes.audio` stay in `model_intermediate_buffer` so the next decode step reuses GPU-resident tensors instead of forcing device-to-host synchronization — eliminating round trips that would otherwise serialize every Talker step.

**Numerical precision on the code predictor.** Where audio quality is sensitive, RMSNorm variance and RoPE stay in fp32, attention uses SDPA with native GQA, and per-call embedding buffers avoid cross-request aliasing. Qwen3-Omni configures the wrapper with parallel embedding, stored sampling, and returned projection buffers so Talker can feed the next decode step without rebuilding state on CPU. The compile and graph layers (§3) then optimize this cleaned-up path instead of fighting legacy `generate()` overhead.

These hot-path changes stack with the batching, graph, and async-chunk optimizations above. Qwen3-Omni performance tests in the DFX perf suite — text-only serving and async-chunk workloads — give tracked baselines instead of one-off local numbers.

### 8. Validation Results

We validated the optimization stack with a controlled local benchmark on `Qwen3-Omni-30B-A3B-Instruct` (`128` prompts, concurrency `32`, `5` warmups, three GPUs mapped as `0/1/2`: Thinker on GPU 0, Talker and Code2Wav each with 2 replicas on GPUs 1 and 2). Each configuration restarted the server with an isolated deploy profile. The charts below start from the **Batch** baseline and compare Batch, CUDA Graph, Async chunk, Async output, and Stage replicas.

<p align="center">
  <img src="/assets/figures/2026-06-27-qwen3-omni-optimization/qwen3-omni-bench-throughput.svg" alt="Qwen3-Omni output throughput at concurrency 1 vs 32 across optimization configs from Batch to Replicas." width="100%">
  <br>
  <em>Figure 4: Output throughput (tok/s) at concurrency 1 (orange) vs concurrency 32 (green). CUDA Graph, async output, and stage replicas raise c=32 output tok/s from 35.0 (Batch) to 185.8 (Replicas).</em>
</p>

<p align="center">
  <img src="/assets/figures/2026-06-27-qwen3-omni-optimization/qwen3-omni-bench-reqps.svg" alt="Qwen3-Omni request throughput at concurrency 1 vs 32 across optimization configs from Batch to Replicas." width="100%">
  <br>
  <em>Figure 5: Request throughput (req/s) at concurrency 1 (orange) vs concurrency 32 (green). Stage replicas reach 11.1 req/s at c=32, up from 2.0 req/s (Batch).</em>
</p>

<p align="center">
  <img src="/assets/figures/2026-06-27-qwen3-omni-optimization/qwen3-omni-bench-rtf.svg" alt="Qwen3-Omni mean audio RTF at concurrency 1 vs 32 across optimization configs from Batch to Replicas." width="100%">
  <br>
  <em>Figure 6: Mean audio RTF at concurrency 1 (orange) vs concurrency 32 (green). Batch alone stays near realtime at c=32 (RTF 0.98); async chunking and replicas keep RTF at or below ~0.40.</em>
</p>

<p align="center">
  <img src="/assets/figures/2026-06-27-qwen3-omni-optimization/qwen3-omni-bench-ttfp.svg" alt="Qwen3-Omni mean audio TTFP at concurrency 1 vs 32 across optimization configs from Batch to Replicas." width="100%">
  <br>
  <em>Figure 7: Mean audio TTFP in milliseconds (log scale). Async chunking drops c=32 TTFP from ~4975 ms (Batch) to ~497 ms.</em>
</p>

| Config (c=32) | Output tok/s | Audio-s/s | Mean audio TTFP | Mean audio RTF |
|---|---:|---:|---:|---:|
| Batch | 35.0 | 10.8 | 4975 ms | 0.98 |
| CUDA Graph | 135.2 | 40.9 | 1798 ms | 0.38 |
| Async chunk | 152.5 | 45.9 | 497 ms | 0.39 |
| Async output | 137.4 | 42.4 | 464 ms | 0.31 |
| Stage replicas | 185.8 | 57.1 | 482 ms | 0.31 |

Stage replicas deliver the best throughput on this sweep: **185.8** output tok/s at concurrency 32, with TTFP and RTF staying in the same **~460–500 ms / ~0.31** band as async output. Async chunking is the largest latency win over Batch alone, cutting TTFP from **4975 ms** to **497 ms** and RTF from **0.98** to **0.39**.

The following results come from the Qwen3-Omni DFX perf artifacts for `Qwen/Qwen3-Omni-30B-A3B-Instruct` using the `openai-chat-omni` backend. They reflect the accumulated optimization stack above. The percentages compare the measured run against the tracked perf-suite baseline for the same workload. Lower is better for TTFT, E2EL, TTFP, and audio RTF; higher is better for throughput.

| Workload | Key result | Change vs. baseline |
|---|---:|---:|
| Long text + audio output, 128 prompts, concurrency 32, 2500 input / 900 output tokens | 80.43 audio-s/s, 0.399 mean audio RTF, 1065 ms mean audio TTFP | TTFT -20.3%, E2EL -26.3%, audio RTF -29.1% |
| Audio-input multimodal, 10 prompts at 0.1 req/s | 3.15 audio-s/s, 0.117 mean audio RTF, 396 ms mean audio TTFP | TTFT -12.7%, E2EL -11.6%, audio RTF -10.3% |
| Image + video multimodal, 40 prompts at 0.5 req/s | 13.74 audio-s/s, 0.146 mean audio RTF, 332 ms mean audio TTFP | TTFT -9.2%, E2EL -24.0%, audio RTF -19.8% |
| Image + video + audio multimodal, 100 prompts at 1.0 req/s | 29.07 audio-s/s, 0.186 mean audio RTF, 553 ms mean audio TTFP | TTFT -14.3%, E2EL -34.5%, audio RTF -31.4% |

The same infrastructure also benefits text-only routing through Qwen3-Omni. In the text-only long-output workload with 128 prompts at concurrency 32, the measured run reached 2,032 output tok/s and 7,698 total tok/s, while mean E2EL dropped from the tracked 18.15 s baseline to 14.15 s (-22.1%).

The multi-replica perf suite separately validates the deployment shape where stages 1 and 2 each run two replicas. That suite tracks long text-plus-audio serving at concurrency 8, 16, 24, and 32, with baseline targets for TTFT, TPOT, audio TTFP, and audio RTF. The Stage replicas row in the benchmark sweep above shows the same scaling idea in practice: replicating Talker and Code2Wav improves throughput and latency without duplicating the Thinker stage.

The important result is that the gains show up across different request shapes: long text-plus-audio generation, audio input, image/video input, and mixed image/video/audio input. That is what we want from a pipeline optimization. It improves the common dataflow instead of only helping one narrow prompt pattern.

## Deployment Profile

The default Qwen3-Omni-MoE deploy profile is designed for a staged layout. In the checked-in profile, stage 0 runs on one GPU, while stages 1 and 2 run on another:

```yaml
stages:
  - stage_id: 0
    devices: "0"
    max_num_seqs: 64
    default_sampling_params:
      temperature: 0.0
      max_tokens: 2048

  - stage_id: 1
    devices: "1"
    max_num_seqs: 64
    input_connectors:
      from_stage_0: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.9
      top_k: 50
      max_tokens: 4096

  - stage_id: 2
    devices: "1"
    max_num_seqs: 64
    input_connectors:
      from_stage_1: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.0
      max_tokens: 65536
```

The exact device assignment is not a universal recommendation. The useful part is the separation of concerns:

- Thinker has the multimodal prompt and deterministic text-generation profile.
- Talker has speech-generation sampling parameters.
- Code2Wav has audio reconstruction parameters and backend-specific graph policy.
- Shared-memory connectors carry structured payloads between adjacent stages.

## What This Enables

With this design, vLLM-Omni can serve Qwen3-Omni as more than a monolithic checkpoint:

- Per-request `modalities` control how far the pipeline runs: text-only requests finish at Thinker; adding `"audio"` activates Talker and Code2Wav.
- Chat serving can receive text while audio serving consumes Code2Wav output metadata such as sample rate.
- Platform-specific deployment differences can live in YAML rather than model branches.

This is the main lesson from Qwen3-Omni support in vLLM-Omni: omni-modality serving is a dataflow problem as much as a kernel problem. Efficient inference depends on carrying the right tensors across stage boundaries, preserving request-scoped async-chunk handoff state, and choosing execution policy per component.

## References

- Qwen3-Omni pipeline topology in vLLM-Omni: [`pipeline.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/pipeline.py)
- Qwen3-Omni model wrapper in vLLM-Omni: [`qwen3_omni.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py)
- Qwen3-Omni stage input processors: [`stage_input_processors/qwen3_omni.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_input_processors/qwen3_omni.py)
- Qwen3-Omni deploy profile: [`qwen3_omni_moe.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_omni_moe.yaml)
- Qwen3-Omni async-chunk perf config: [`test_qwen3_omni_async_chunk.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json)
- Qwen3-Omni multi-replica perf config: [`test_qwen3_omni_multi_replicas.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_multi_replicas.json)
- Qwen3-Omni model repository: [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)

If you are interested in Qwen3-Omni serving or omni-modality inference, join the `#vllm-omni` channel in [vLLM Slack](https://slack.vllm.ai), or open an issue in [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni).
