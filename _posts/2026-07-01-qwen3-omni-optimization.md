---
layout: post
title: "Experience and Lessons Learned from Serving Multi-Stage Qwen3-Omni in vLLM-Omni"
author: "vLLM-Omni Team and Ant Group SCT Team"
summary: "How vLLM-Omni serves and optimizes Qwen3-Omni with staged Thinker-Talker-Code2Wav execution, batching, CUDA Graphs, async chunk, async output, replicas, hot-path cleanup, and perf validation."
image: /assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-serving-flow.svg
social_image: /assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-serving-flow.svg
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
- **Batching, CUDA Graphs, async chunk, async output, replicas, and hot-path cleanup:** stage-level batching and per-stage graph capture on Thinker, Talker, and Code2Wav improve high-concurrency throughput; async-chunk handoffs and async output keep the pipeline and decode workers from stalling on full-payload barriers and synchronous payload construction; Talker/Code2Wav replicas scale the speech-generation stages; and hot-path cleanup trims the per-step model-internal overhead that scales with utterance length.
- **Performance validation:** controlled benchmark sweeps and DFX perf runs show lower audio TTFP (time to first audio packet), lower audio RTF (real-time factor), and higher throughput as each optimization layer is enabled.

## Quickstart

The default Qwen3-Omni deploy profile is resolved automatically when serving the model with `--omni`:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091
```

For explicit configuration, pass the staged deploy profile:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml
```

The bundled profile includes a `platforms:` section. vLLM-Omni detects the runtime backend (CUDA, NPU, ROCm or XPU) and merges the matching deltas automatically — no extra CLI flag is required. The same launch command works across hardware.

Requests should use `/v1/chat/completions`. Set `modalities` in the request body to declare output types — e.g. `["text"]` for text only, or `["text", "audio"]` for text plus speech.

For deployment options, async-chunk settings, and multi-replica layouts, see the [Qwen3-Omni online serving guide](https://github.com/vllm-project/vllm-omni/blob/main/examples/online_serving/qwen3_omni/README.md).

## Qwen3-Omni Serving Model

Text-only LLM serving is one loop: prefill, decode, detokenize. Qwen3-Omni adds two speech stages after multimodal reasoning, each with a different compute profile:

```text
Thinker   -> multimodal understanding + text generation
Talker    -> hidden states and embeddings to RVQ codec codes
Code2Wav  -> codec codes to waveform audio
```

<p align="center">
  <img src="/assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-serving-flow.svg" alt="Qwen3-Omni serving flow in vLLM-Omni, showing multimodal inputs, Thinker, Talker, Code2Wav, and text/audio outputs." width="100%">
  <br>
  <em>Figure 1: Qwen3-Omni serving in vLLM-Omni is a staged dataflow: Thinker produces text and hidden states, Talker produces codec codes, and Code2Wav reconstructs audio.</em>
</p>

## Optimization Overview

Different parts of the Qwen3-Omni pipeline hit different bottlenecks. vLLM-Omni does not apply one fixed recipe; each optimization targets a specific stage or handoff. The walkthrough that follows takes each one in turn — in the order we validated it — and answers three questions in order: **Why** the problem exists, **Why it works** (how the mechanism removes that problem), and **What you gain** (the measured payoff).

| Technique | Target stage / path | Problem it addresses | Primary benefit |
|---|---|---|---|
| Stage decomposition | Thinker → Talker → Code2Wav | Three stages with very different compute profiles share one loop, so a single batching/graph/device policy lets the slowest sub-path gate the rest — capping throughput and blocking per-stage tuning and scaling | Independent runtime policy per stage |
| AR + Code2Wav batching | Talker MTP path, Code2Wav async chunks | Single-request micro-work leaves SMs idle between launches at high concurrency, capping GPU occupancy and req/s | Higher occupancy and req/s |
| CUDA Graph | Thinker / Talker / Code2Wav decode paths | Repeated CPU-side kernel dispatch on every decode step inflates TPOT and keeps audio RTF above real-time | Lower TPOT and audio RTF; ~4× throughput jump in sweep |
| Async chunk | Thinker→Talker, Talker→Code2Wav | Full-payload stage barriers force Code2Wav to wait for a full Talker payload, delaying first audio and inflating audio TTFP | Pipelined handoffs; largest audio TTFP reduction |
| Async omni output | Thinker connector payloads | Synchronous payload construction blocks Thinker decode workers between chunks, wasting GPU time on allocation and lowering throughput | Throughput recovery without audio TTFP regression |
| Stage replicas | Talker, Code2Wav | Talker and Code2Wav saturate and queue while Thinker still has headroom, becoming the tail bottleneck under load | Horizontal scale on bottleneck stages only |
| Hot-path cleanup | Talker code predictor, connector payloads | Per-step Python, allocation, and sync overhead compounds over long utterances, inflating E2EL and audio TTFP | Lower per-step latency; stacks with all layers above |

We validated each layer with a controlled benchmark sweep on Seed-TTS `en` (`Qwen3-Omni-30B-A3B-Instruct`, `10`/`160`/`320`/`640` prompts at concurrency `1`/`16`/`32`/`64`, `5` warmups, three visible GPUs mapped as `0/1/2`). Each configuration restarted the server with an isolated deploy profile and added one optimization on top of the previous row. **Batch** through **Async output** pin one stage per GPU (Thinker / Talker / Code2Wav on GPUs 0 / 1 / 2, single replica each); the **Stage replicas** row keeps Thinker on GPU 0 and runs 2× Talker + 2× Code2Wav on GPUs 1 and 2. The table below summarizes concurrency 64; [Validation Results](#validation-results) charts all four concurrency levels.

| Step | Config added | Talker / Code2Wav replicas | Req/s | Mean audio TTFP | Mean audio RTF |
|---|---|---|---:|---:|---:|
| Baseline | Batch | 1 / 1 | 2.2 | 5884 ms | 1.15 |
| + CUDA Graph | Graph capture on Thinker, Talker, Code2Wav | 1 / 1 | 8.6 (+299%) | 2790 ms (−53%) | 0.59 (−49%) |
| + Async chunk | Async-chunk stage handoffs | 1 / 1 | 9.3 (+8%) | 655 ms (−77%) | 0.63 |
| + Async output | Async omni output path | 1 / 1 | 11.3 (+22%) | 631 ms (−4%) | 0.47 (−25%) |
| + Stage replicas | 2× Talker + 2× Code2Wav | 2 / 2 | 11.7 (+4%) | 632 ms | 0.47 |

<p align="center">
  <img src="/assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-optimization-stack.svg" alt="Qwen3-Omni optimization stack in vLLM-Omni, showing staged execution, batching and replicas, CUDA Graph, async chunk, and hot-path cleanup." width="100%">
  <br>
  <em>Figure 2: Qwen3-Omni performance comes from optimizing the staged dataflow, stage runtime, and decode hot path together.</em>
</p>

## Optimization Stack, Stage by Stage

Each optimization builds on the previous one, so the numbers reported in each step assume every layer above it is already enabled.

### 1. Stage Decomposition and Batching: The Baseline

**Why.** Qwen3-Omni is not one homogeneous decode loop: Thinker does multimodal AR text generation, Talker runs a codec-predictor AR path, and Code2Wav runs parallel vocoder decode. Folding these three very different workloads into a single serving path forces the same batching policy, graph policy, and device layout on all of them — and lets the slowest sub-path gate the rest. Separating the stages removes that coupling, but it exposes a second problem: the speech path still spends most of its GPU time on single-request micro-work. Each Talker decode step is a short code-predictor forward and each Code2Wav chunk is a small vocoder forward, so at concurrency 64, running them one request at a time leaves SMs idle between launches and never amortizes the fixed per-step cost.

**Why it works.** This addresses the two problems in turn. First, stage decomposition breaks the coupling: stage boundaries become first-class serving objects, connectors define what crosses each one (hidden states, embeddings, codec codes, chunk metadata), and the scheduler can schedule, batch, and graph each stage on its own critical path — so no single policy is forced on all three and the slowest sub-path no longer gates the rest. Second, per-stage batching closes the idle-SM gap: collecting concurrent requests into one Talker MTP invocation and one Code2Wav forward fills the SMs that single-request micro-work left idle, and amortizes the fixed per-step cost across the batch.

**What you gain.** Explicit stages let vLLM-Omni treat each component as an independent runtime — separate `max_num_seqs`, sampling params, connectors, graph/eager policy, and optional replicas — which is the prerequisite for every optimization below. This batched, stage-decomposed configuration is the **Batch** baseline that every later row builds on.

### 2. CUDA Graph: Per-Stage Decode Capture

**Why.** Batching raised occupancy, but each decode step still paid repeated CPU-side kernel dispatch. Qwen3-Omni runs three decode-heavy stages; Talker alone may execute hundreds of short steps per utterance, and each step previously re-launched the same stable operator sequence from Python. At concurrency 64, that launch tax dominated TPOT and kept audio RTF above real-time even after batching.

**Why it works.** CUDA Graph removes the per-step kernel dispatch that dominated TPOT: it captures a fixed operator sequence once and replays it with minimal CPU work. Each stage has a different capture point, but the principle is the same: decode shapes bucket into stable `(batch, seq, frames)` profiles, so the runtime records the graph at warmup and reuses it on the hot path.

<p align="center">
  <img src="/assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-cuda-graph-stages.svg" alt="Per-stage CUDA Graph capture in Qwen3-Omni: Thinker and Talker run under vLLM's outer decode graph, Talker's inner code predictor is torch.compiled, and Code2Wav uses an inner CUDAGraphDecoderWrapper." width="100%">
  <br>
  <em>Figure 3: Each stage captures graphs at a different point. Thinker and Talker decode under vLLM's outer graph; Talker's inner code predictor is <code>torch.compile</code>d instead of given a second graph; Code2Wav uses an inner <code>CUDAGraphDecoderWrapper</code>.</em>
</p>

#### Stage 0 — Thinker: vLLM outer decode graph

The Thinker is an autoregressive multimodal stage (`LLM_AR`). When `enforce_eager` is false, it uses vLLM's standard CUDA Graph capture on the decode path — the same mechanism as text-only LLM serving. This removes repeated CPU-side kernel dispatch during long Thinker generations.

#### Stage 1 — Talker: outer decode graph + compiled code predictor

The Talker stage also runs through vLLM's outer CUDA Graph path when `enforce_eager: false`. Each Talker decode step additionally invokes the **code predictor** — a short re-prefill transformer that emits RVQ codec codes. That inner path is optimized separately:

- **`torch.compile`** fuses the 5-layer predictor forward (`dynamic=False`, `epilogue_fusion=False`) so RMSNorm/RoPE stay numerically aligned with the reference path while still reducing kernel count per step.
- On CUDA, the code predictor does **not** enable a second manual CUDA Graph layer by default (`use_cuda_graphs=False`), because that would conflict with vLLM's Talker `CUDAGraphWrapper`. The outer Talker graph and compiled inner forward are complementary: one captures the AR stage loop, the other fuses the codec-prediction micro-forward.

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

**What you gain.** Turning on CUDA Graph for all three stages in the benchmark sweep raises req/s from **2.2** to **8.6** (+299%), cuts mean audio TTFP from **5884 ms** to **2790 ms**, and drops mean audio RTF from **1.15** to **0.59**. Most of the win comes from removing launch overhead across Thinker text generation, Talker codec decode, and Code2Wav vocoder forwards together.

### 3. Async Chunk: Pipelined Inter-Stage Handoffs

**Why.** CUDA Graph made each stage faster, but the pipeline was still **barrier-synchronized**: Talker could not start until Thinker finished, and Code2Wav could not emit audio until Talker accumulated a full payload. First-audio latency therefore tracked full Thinker generation plus full Talker prefill — even when only a few codec frames were needed to produce the first audible chunk.

**Why it works.** Async chunk replaces the full-payload barrier with **pipelined partial handoffs**. Thinker emits embedding rows incrementally; Talker accumulates codec frames and slices them on `initial_codec_chunk_frames` / `codec_chunk_frames` boundaries; The async scheduler overlaps chunk transfer with stage compute, so each stage starts work while the previous one is still decoding — first audio is ready after a few codec frames instead of a full Thinker generation plus Talker prefill.

<p align="center">
  <img src="/assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-async-chunk-timeline.svg" alt="Async chunk before/after timeline: the barrier-synchronized pipeline waits for full Thinker and Talker payloads before first audio, while async chunk overlaps partial Thinker, Talker, and Code2Wav chunks so first audio arrives much earlier." width="100%">
  <br>
  <em>Figure 4: Without async chunk, each stage waits for the previous stage's full payload, so first audio tracks full Thinker generation plus Talker prefill. Async chunk overlaps partial handoffs, so Code2Wav starts emitting audio after only a few codec frames.</em>
</p>

**What you gain.** Async chunk is the largest audio TTFP win in the sweep: mean audio TTFP drops from **2790 ms** (CUDA Graph) to **655 ms**.

### 4. Async Output: Non-Blocking Payload Construction

**Why.** Async chunk pipelines Thinker→Talker→Code2Wav, but **synchronous payload construction** can still block decode workers. If Thinker must fully assemble each connector payload — copying embeddings and hidden states on every chunk boundary — before the next decode step can start, GPU time is lost to Python scheduling even though stage handoffs are already incremental.

**Why it works.** `async_omni_output` decouples payload construction from stage handoff: Thinker hands decode state to a non-blocking output path and immediately returns to the next token, while the connector assembles and ships chunks asynchronously.

<p align="center">
  <img src="/assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-async-output-step-gap.svg" alt="Async output before/after profiler trace: before, decode steps are separated by ~2.8 ms idle gaps while the connector payload is built synchronously; after, steps pack back-to-back with ~41 us gaps and the GPU stays busy." width="100%">
  <br>
  <em>Figure 5: Decode step gap before and after async output. With synchronous payload construction, the GPU sits idle ~2.8 ms between Talker steps; moving payload assembly off the decode path packs steps back-to-back, shrinking the inter-step gap to ~41 µs.</em>
</p>

**What you gain.** On top of async chunk at concurrency 64, async output keeps mean audio TTFP near **631 ms** while lowering mean audio RTF from **0.63** to **0.47**.

### 5. Stage Replicas: Scaling Talker and Code2Wav

**Why.** The three stages do not saturate equally under load. For each request, Thinker generates text once, but Talker and Code2Wav then run hundreds of short decode steps and vocoder forwards to render that text as audio — far more sustained small-step work on the speech side. So as concurrency climbs, Talker and Code2Wav saturate first: at concurrency 64 a single replica of either becomes the tail bottleneck while Thinker still has headroom. Scaling the whole pipeline would clear it, but waste memory duplicating the large multimodal Thinker.

**Why it works.** Replication adds capacity only to the stages that saturate, not the whole pipeline. A single Thinker on GPU 0 feeds 2× Talker and 2× Code2Wav replicas spread across GPUs 1 and 2: the extra replicas absorb the speech-side backlog, while the heavy multimodal stage stays unduplicated. The benchmark deploy config enables this with:

```json
{
  "stage_overrides": {
    "1": {"num_replicas": 2, "devices": "1,2"},
    "2": {"num_replicas": 2, "devices": "1,2"}
  }
}
```

<p align="center">
  <img src="/assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-async-replica.svg" alt="Qwen3-Omni async chunk and stage replica serving, showing one Thinker feeding two Talker replicas and two Code2Wav replicas." width="100%">
  <br>
  <em>Figure 6: Async chunk and stage replicas target the speech-generation side of the pipeline, where Talker and Code2Wav can become the bottleneck under concurrent load.</em>
</p>

**What you gain.** Adding replicas on top of async output reaches **11.7** req/s at concurrency 64 — the highest throughput in the sweep — while holding mean audio TTFP near **632 ms** and mean audio RTF near **0.47**. The replica margin over a single Talker/Code2Wav widens as concurrency climbs, since the speech stages are the first to saturate.

### 6. Hot-Path Cleanup: Talker Decode and Connector Payloads

**Why.** Batching, graphs, async chunk, and replicas eliminate the structural, framework-level bottlenecks — optimizations that apply to most multimodal serving pipelines. What remains is model-internal: profiling the Talker decode loop still shows a long tail of small costs that repeat on every step and so scale with utterance length — redundant connector traffic, per-step `torch.cat` and CPU serialization while building payloads, Python dispatch in the codec predictor, and device-to-host reads of decode state that the next step needs straight back on the GPU.

**Why it works.** Each fix below removes one of those repeated costs — or trims fixed per-deployment overhead — without changing the audio output:

**Decode-only connector handoffs.** Chunk 0 still ships the full Thinker prefill; every later decode step sends only the new `embed.decode` row instead of re-transmitting the full prefill embedding and hidden-state tensors. Connector traffic stays **O(1) per step** rather than growing with prompt length, and each handoff avoids redundant CPU serialization and cross-stage copies that would otherwise repeat on every Talker step in a long utterance.

**Single-GPU executor default.** Removing the implicit `"mp"` default for `distributed_executor_backend` lets single-GPU deployments use the `uni` executor and avoid multiprocess startup, IPC, and worker-sync overhead on the low-concurrency paths where per-step latency matters most.

**Connector payload construction.** Accumulating Thinker and Talker payloads previously paid for repeated `torch.cat` on every chunk boundary. Passing decode embeddings per token and trimming redundant assembly removes that allocation and copy work. The same change set also skips building downstream pooler/multimodal CPU payloads when a request's final stage is already local, avoiding hidden-state D2H on paths that do not feed another stage.

**Talker code predictor rewrite.** The old path drove very short codec-predictor sequences through Hugging Face `generate()`, adding Python dispatch, dynamic allocation, and KV-cache overhead on every Talker step. The rewrite uses re-prefill with SDPA, native GQA, inline top-k sampling, cached module references, and `torch.compile` on the inner transformer. On CUDA this compile path sits below vLLM's Talker CUDA Graph rather than adding a conflicting second graph layer (see §2).

**GPU-resident decode state and stage-local fast paths.** Intermediate tensors such as `hidden_states.last`, `hidden_states.trailing_text`, `embed.tts_pad_projected`, and `codes.audio` stay in `model_intermediate_buffer`, so the next step reuses them on-device instead of forcing a device-to-host round trip that would serialize every Talker step. Talker and Code2Wav also skip multimodal `get_mrope_input_positions` — they only need cheap linear positions — and `_store_value` avoids redundant `.to("cpu")` work when a tensor is already on CPU.

**Numerical precision guardrail.** These rewrites must not regress audio quality, so RMSNorm variance and RoPE stay in fp32 (`epilogue_fusion=False`), and per-call embedding buffers avoid cross-request aliasing — the speed-ups above run on top of this constraint, not against it.

**What you gain.** Hot-path cleanup removes overhead that scales with utterance length. In a long-context single-request test, E2EL dropped from 21.28 s to 7.37 s, audio TTFP from 3197 ms to 1796 ms, and audio RTF from 0.71 to 0.28. These changes stack with the layers above and show up in the DFX perf suite baselines rather than as a separate sweep row.

## Validation Results

The charts below plot the same benchmark sweep summarized in [Optimization Overview](#optimization-overview) — Batch, CUDA Graph, Async chunk, Async output, and Stage replicas — across all four concurrency levels (`1`/`16`/`32`/`64`), each starting from the **Batch** baseline.

<p align="center">
  <img src="/assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-bench-reqps.svg" alt="Qwen3-Omni request throughput at concurrency 1, 8, 16, and 32 across optimization configs from Batch to Replicas." width="100%">
  <br>
  <em>Figure 7: Request throughput (req/s) at c=1 (orange), c=16 (purple), c=32 (green), and c=64 (red). Stage replicas reach 11.7 req/s at c=64 and 6.8 req/s at c=32, up from 2.2 req/s (Batch at c=64).</em>
</p>

<p align="center">
  <img src="/assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-bench-rtf.svg" alt="Qwen3-Omni mean audio RTF at concurrency 1, 8, 16, and 32 across optimization configs from Batch to Replicas." width="100%">
  <br>
  <em>Figure 8: Mean audio RTF at c=1, 16, 32, and 64. Batch sits at or above real-time under load (RTF up to 1.15 at c=64); async output and replicas keep c=32/c=64 RTF at or below ~0.47.</em>
</p>

<p align="center">
  <img src="/assets/figures/2026-07-01-qwen3-omni-optimization/qwen3-omni-bench-ttfp.svg" alt="Qwen3-Omni mean audio TTFP at concurrency 1, 8, 16, and 32 across optimization configs from Batch to Replicas." width="100%">
  <br>
  <em>Figure 9: Mean audio TTFP in milliseconds (log scale) at c=1, 16, 32, and 64. Async chunk drops c=64 TTFP from ~5884 ms (Batch) to ~655 ms.</em>
</p>


### Reading the Sweeps Together

Across the three sweeps, the story is consistent: each layer targets a different bottleneck, and together they compound.

- **Throughput (Figure 7).** Request throughput climbs from the Batch baseline of **2.2 req/s** to **11.7 req/s** at concurrency 64 (**~5.4×**), and from **1.1** to **6.8 req/s** at concurrency 32. The largest single jump is CUDA Graph (**~4×**); async output supplies the last big push at high concurrency, and stage replicas take throughput to its peak with headroom that grows as concurrency rises.
- **Real-time factor (Figure 8).** Mean audio RTF drops from an above-real-time **1.15** under Batch to **0.47** at concurrency 64 — decode moves from lagging playback under load to running comfortably ahead of it.
- **First-packet latency (Figure 9).** Mean audio TTFP falls from **~5884 ms** to **~632 ms** at concurrency 64, with async chunk contributing the largest single cut (to **~655 ms**) and the later layers holding that gain.

The takeaway is that no single layer carries the whole win: CUDA Graph and async chunk dominate the latency reductions, while async output and stage replicas add the throughput headroom at concurrency 32 and 64. Stacking them turns a pipeline that barely keeps up under load into one with real-time headroom to spare.

## Acknowledgements

We thank the Qwen3-Omni contributors in [vLLM-Omni](https://github.com/vllm-project/vllm-omni), including Haiyan Wu, Taichang Zhou, Canlin Guo, Ruirui Yang, Ziming Huang, Wengang Zheng, Lianhao Xu, Han Gao, Junhong Liu, Samit Huang, Hao Chen, Alex Brooks, Chenguang Zheng, Peiqi Yin, Wenjing Chen, Nick Cao, Shunyang Li, Yong Yang, Divyansh Singhvi, Yueqian Lin, Dayu Qiu, Roger Wang and Hongsheng Liu, for their contributions and feedback.

---

## References

**Source and configuration**

- Qwen3-Omni pipeline topology in vLLM-Omni: [`pipeline.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/pipeline.py)
- Qwen3-Omni model wrapper in vLLM-Omni: [`qwen3_omni.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py)
- Qwen3-Omni stage input processors: [`stage_input_processors/qwen3_omni.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_input_processors/qwen3_omni.py)
- Qwen3-Omni deploy profile: [`qwen3_omni_moe.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_omni_moe.yaml)
- Qwen3-Omni async-chunk perf config: [`test_qwen3_omni_async_chunk.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json)
- Qwen3-Omni multi-replica perf config: [`test_qwen3_omni_multi_replicas.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_multi_replicas.json)
- Qwen3-Omni model repository: [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)

**Optimization pull requests**

- CUDA Graph (§2): Thinker [vllm-omni#523](https://github.com/vllm-project/vllm-omni/pull/523), Talker [vllm-omni#669](https://github.com/vllm-project/vllm-omni/pull/669), Code2Wav [vllm-omni#2376](https://github.com/vllm-project/vllm-omni/pull/2376)
- Async chunk (§3): cross-stage chunked compute/communication [vllm-omni#727](https://github.com/vllm-project/vllm-omni/pull/727), async scheduling to overlap chunk IO and compute [vllm-omni#951](https://github.com/vllm-project/vllm-omni/pull/951), inter-packet latency optimization [vllm-omni#1656](https://github.com/vllm-project/vllm-omni/pull/1656)
- Async output (§4): async omni output materialization [vllm-omni#4476](https://github.com/vllm-project/vllm-omni/pull/4476)
- Stage replicas (§5): support multi-stage deployment[vllm-omni#2396](https://github.com/vllm-project/vllm-omni/pull/2396), omni stage runtime and distributed replica control plane [vllm-omni#3855](https://github.com/vllm-project/vllm-omni/pull/3855)
- Hot-path cleanup (§6): [vllm-omni#3007](https://github.com/vllm-project/vllm-omni/pull/3007), [vllm-omni#3164](https://github.com/vllm-project/vllm-omni/pull/3164), [vllm-omni#3878](https://github.com/vllm-project/vllm-omni/pull/3878)

If you are interested in Qwen3-Omni serving or omni-modality inference, join the `#vllm-omni` channel in [vLLM Slack](https://slack.vllm.ai), or open an issue in [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni).
