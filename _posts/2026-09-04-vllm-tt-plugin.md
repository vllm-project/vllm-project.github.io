---
layout: post
title: "Serving LLMs on Tenstorrent Hardware: Inside the vLLM TT Plugin"
author: "Tenstorrent Team"
summary: "Tenstorrent accelerators join vLLM as an out-of-tree platform plugin, driven by mesh-architecture choices: phase-based scheduling, single-process data parallelism on Galaxy, on-device sampling with host fallback, and async decode overlap."
image: /assets/figures/2026-09-04-vllm-tt-plugin/mesh-vs-collectives.svg
tags:
  - hardware
  - ecosystem
---

Today we are introducing [**vLLM TT Plugin**](https://github.com/tenstorrent/vllm-tt-plugin), which brings [Tenstorrent](https://tenstorrent.com/) accelerators to vLLM through the standard out-of-tree platform plugin mechanism.

Install it alongside vLLM and, whenever `ttnn` from [TT-Metal](https://github.com/tenstorrent/tt-metal) is importable, Tenstorrent hardware is discovered and registered as a vLLM platform automatically. The serving surface does not change: the same OpenAI-compatible API, the same request format, the same client code.

The more interesting part of this project is not that the backend exists. It is that a Tenstorrent device does not look much like a GPU, and vLLM's plugin interfaces turned out to be general enough that we could express those differences - a phase-constrained scheduler, a different data-parallel topology, a sampling path that partly lives on device - entirely outside vLLM core.

## Supported models

The plugin registers Tenstorrent-backed architectures under a `TT`-prefixed convention, so a checkpoint is picked up by the architecture it declares rather than by name:

| Model family | Architectures |
| --- | --- |
| Llama 3.1 / 3.2 / 3.3 | `TTLlamaForCausalLM` |
| Llama 3.2 Vision | `TTMllamaForConditionalGeneration` |
| Qwen 2.5 / Qwen 3 | `TTQwen2ForCausalLM`, `TTQwen3ForCausalLM` |
| Qwen 3.5 / Qwen 3.6 | `TTQwen3_5ForConditionalGeneration` |
| Qwen 2.5-VL / Qwen 3-VL | `TTQwen2_5_VLForConditionalGeneration`, `TTQwen3VLForConditionalGeneration` |
| Mistral / Mistral 3 | `TTMistralForCausalLM`, `TTMistral3ForConditionalGeneration` |
| Gemma 3 | `TTGemma3ForConditionalGeneration` |
| Gemma 4 | `TTGemma4ForCausalLM`, `TTGemma4ForConditionalGeneration`, `TTGemma4UnifiedForConditionalGeneration` |
| DeepSeek V3 | `TTDeepseekV3ForCausalLM` |
| GPT-OSS 20B / 120B | `TTGptOssForCausalLM` |

The classes behind those names ship in [TT-Metal](https://github.com/tenstorrent/tt-metal), alongside the runtime itself: each is a vLLM-facing generator wrapped around a hand-written TTNN implementation of the model. The plugin carries no model code - it registers the names, and tt-metal provides what they resolve to.

Because the match is on architecture, one entry can cover several releases - `TTQwen3_5ForConditionalGeneration` is what serves `Qwen/Qwen3.6-27B`, for instance.

Multimodal coverage is worth calling out, since new backends often stay text-only for a long time: Llama 3.2 Vision, Qwen-VL, Qwen 3.6, Mistral 3, and Gemma 3 all serve through the plugin today.

Models do not have to be built into the plugin. Pointing `EXTRA_MODELS_DIR` at a directory of bundle folders, each holding a `vllm_metadata.json` and an adapter class, registers architectures at startup under the `TT<HFArch>` convention. A distribution tool can ship a ready-to-serve model without a source edit, and `TT_VLLM_BUILTIN_MODELS=0` narrows the registry to only what was supplied.

That covers what you can serve today. The rest of this post is how it works: the design choices a mesh architecture forces on a GPU-shaped serving stack, and what we learned making them.

## Why a Tenstorrent backend looks different

A Tenstorrent system is a **mesh of cores and chips connected by an on-fabric network**. A single card such as n150 or n300 is already a small mesh; a [QuietBox](https://tenstorrent.com/hardware/tt-quietbox) is a larger one; a [Galaxy](https://tenstorrent.com/hardware/galaxy) is 32 Wormhole chips wired into a topology the runtime configures directly (`FABRIC_1D`, `FABRIC_2D`, `FABRIC_1D_RING`). Programs are compiled and traced against a mesh shape, and the fabric moves data between chips as part of the compiled program rather than as a collective call issued by the host.

The models served through this plugin are **hand-written [TTNN](https://github.com/tenstorrent/tt-metal) implementations** for TT mesh, from a two-chip n300 up to a 32-chip Galaxy. Within that system they run the same parallelization playbook one would use on GPUs - tensor parallelism across chips, data parallelism across submeshes - but expressed in TTNN and compiled into the mesh program rather than configured as runtime ranks. That hand-tuning is what delivers better tokens/$; we will not quote numbers here, current figures live on [tenstorrent.com](https://tenstorrent.com/) and [GitHub](https://github.com/tenstorrent/tt-metal).

<figure>
  <img
    src="/assets/figures/2026-09-04-vllm-tt-plugin/mesh-vs-collectives.svg"
    width="100%"
    alt="Diagram comparing host-issued collectives on GPUs with a compiled Tenstorrent mesh program" />
  <figcaption>Figure 1: Where cross-chip parallelism lives. In a GPU-shaped stack the host issues collectives on every layer and parallelism is a runtime choice expressed as tensor-parallel and pipeline-parallel ranks. On Tenstorrent, the mesh is compiled and traced as one program and the fabric moves data between chips inside it, so the host submits and reads once per step.</figcaption>
</figure>

That compilation model - one traced program for the whole mesh - drives nearly everything downstream:

- **There are no tensor-parallel or pipeline-parallel ranks to configure.** A 70B model on Galaxy is not "TP=32 processes"; it is one program compiled for a 32-chip mesh. `MESH_DEVICE=TG` replaces `--tensor-parallel-size`, and the plugin rejects `-tp`/`-pp` outright rather than pretending to honor them. The parallelism that best fits the (model, mesh) combination is implemented in the model code.
- **The unit of work is a whole traced step.** Device execution is dominated by replaying a captured trace for a fixed batch shape, which makes homogeneous, shape-stable batches dramatically cheaper than heterogeneous ones.
- **Sampling can happen on device.** Because the mesh program can carry sampling to the end, the token can often come back already chosen, and the host never sees the logits.

Each of those is in tension with an assumption somewhere in a GPU-shaped inference stack. The sections that follow are how we resolved them.

## Plugging in, not forking

vLLM's hardware plugin mechanism was [introduced in May 2025](https://vllm.ai/blog/2025-05-12-hardware-plugin) with `vllm-ascend` and `vllm-spyre` among its first users, and the pluggable-scheduler work that came out of the Spyre effort is what makes our approach viable at all. We depend on it heavily.

The plugin registers two entry points:

| Entry point group | Name | Target |
| --- | --- | --- |
| `vllm.platform_plugins` | `tt` | `vllm_tt_plugin.entrypoints:platform_plugin` |
| `vllm.general_plugins` | `tt_model_registry` | `vllm_tt_plugin.entrypoints:register` |

`platform_plugin()` returns `TTPlatform` **only when `ttnn` is importable**, so installing the package into an ordinary CUDA environment cannot accidentally select the Tenstorrent platform.

From there, everything flows through a single handoff. `TTPlatform.check_and_update_config()` validates the configuration, registers model architectures, and swaps in Tenstorrent-owned runtime classes through vLLM's existing extension points:

| vLLM config field | TT implementation |
| --- | --- |
| `parallel_config.worker_cls` | `vllm_tt_plugin.worker.TTWorker` |
| `scheduler_config.scheduler_cls` | `vllm_tt_plugin.scheduler.TTScheduler` or `vllm_tt_plugin.lane_scheduler.TTLaneCoordinator` |

Device-specific options ride on vLLM's generic additional-config namespace rather than through new CLI flags:

```bash
--additional-config.tt.sample_on_device_mode all
--additional-config.tt.fabric_config FABRIC_1D_RING
```

**Nothing Tenstorrent-specific lives in vLLM core.** That is the property that decides whether a backend stays usable: support tracks vLLM's release cadence rather than ours, and nobody ends up stranded on a fork three months behind upstream. We currently validate against a pinned vLLM release and are widening that window as the plugin's API surface settles.

## Phase-based scheduling: prefill-only or decode-only steps

Upstream vLLM's V1 scheduler is token-budget based, and deliberately so. A request has computed tokens and target tokens; each step hands out more token work subject to budgets. Prefill and decode are not separate modes, which is exactly what allows chunked prefill and mixed-progress batches to fall out naturally.

The Tenstorrent path is more constrained. Every scheduling step resolves to one of three outcomes:

- **prefill-only**
- **decode-only**
- **empty**

There are no mixed prefill+decode batches. Chunked prefill is supported within that constraint: a prompt that exceeds the per-step token budget is split across multiple prefill steps, and decode-only steps are interleaved between the chunks, so in-flight requests keep advancing while a long prefill is in flight. Prefill work is still admitted first by default, so then the decode steps run with bigger, more efficient batches; if no prefill can be admitted but decode requests are running, the step is decode-only, so progress continues and KV pressure can relax.

<figure>
  <img
    src="/assets/figures/2026-09-04-vllm-tt-plugin/scheduling-phases.svg"
    width="100%"
    alt="Timeline comparing upstream token-budget steps with Tenstorrent phase-homogeneous steps" />
  <figcaption>Figure 2: The same long prompt under both scheduling models. Upstream spreads it across four chunked steps and mixes decode work for other requests into those same steps. On Tenstorrent a step is still all-prefill or all-decode: the prompt runs as prefill-only chunks with decode-only steps interleaved between them, so every step keeps a stable, traceable shape while in-flight requests keep advancing.</figcaption>
</figure>

This is the design choice most likely to raise an eyebrow, so it is worth being precise about what it costs and what it does not.

**What it buys.** Traced execution rewards batch-shape stability: a step that is uniformly prefill or uniformly decode replays a trace captured for exactly that shape, while a step mixing the two would need a shape the trace was never captured for. The phase separation itself is not a Tenstorrent eccentricity: the largest GPU deployments make the same choice deliberately, running prefill and decode on entirely separate instances - [disaggregated serving](https://docs.vllm.ai/en/stable/features/disagg_prefill/). The Tenstorrent scheduler applies the same split at step granularity within one engine rather than at instance granularity across a fleet.

**What it does not cost.** Continuous batching still holds in the broad sense. Requests arrive into `waiting`, may be parked in `skipped_waiting` while structured-output grammar compiles, are admitted while other requests remain active, can be preempted back, and complete independently. The restriction is *within* a device step, not across the request lifecycle.

**What it does cost.** The interleave granularity is a whole step. Upstream mixes a prefill chunk and ongoing decode into the same step; the Tenstorrent scheduler alternates, so a decode request still waits out each prefill chunk between its own steps, and each mode switch drains the async decode overlap pipeline described below. Both are scheduling-policy costs, not fundamental limits: nothing in the hardware or in vLLM prevents capturing a mixed-shape step in the future versions.

## Single-process lane data parallelism on Galaxy

This is the part with no analogue elsewhere in vLLM, and the piece we are most interested in feedback on.

Some Tenstorrent models - Llama 3.3 70B via `TT_LLAMA_TEXT_VER=llama3_70b_galaxy`, Qwen3-32B via `TT_QWEN3_TEXT_VER=qwen3_32b_galaxy`, and GPT-OSS - are served by *single-execute* generators: one program spanning the entire Galaxy mesh, executed once per step. There is no submesh to give a second engine process. Standard multi-process data parallelism, which assigns each rank its own devices, simply has nothing to partition.

**But:** these models are single-*weights* and single-*execute*, yet they keep **four independent data-parallel KV caches**, each on its own DP submesh. So there is nothing to partition at the process level, and four things to schedule independently.

Our initial implementation was to give each DP rank its own process, just as vLLM normally does. However, given that the ranks must negotiate the prefill vs. decode step type, and there is actually only one mesh submit/readout, we needed to modify vLLM core quite a bit - far beyond the scope of the hardware plugin mechanism. The per-rank schedulers did run in parallel, but the extra inter-process scatter/gather on every step cost more than that parallelism won back.

The better answer is to put the parallelism *inside* one engine process:

`TTLaneCoordinator` owns one independent `TTScheduler` per **lane**. Each lane has its own `waiting` and `running` queues, its own admission decisions, its own KV cache manager, and its own lane-local block ID space. New requests are assigned to the least-loaded lane and stay bound to it.

Because the device executes all lanes together, the coordinator must pick one shared mode per step:

- if any lane can admit prefill, **all** lanes run a prefill step, bounded by the same decode-interleave cadence as the single-scheduler case
- otherwise, all lanes run a decode step
- a lane with no work for the selected mode contributes an empty slice of the merged batch

The coordinator then merges the per-lane `SchedulerOutput` objects, the worker builds one merged device input, and the runner splits the result back out by lane - all in one process, with **no process-level collectives anywhere** - which is exactly the scatter/gather cost that sank the multi-process attempt.

<figure>
  <img
    src="/assets/figures/2026-09-04-vllm-tt-plugin/lane-dp.svg"
    width="100%"
    alt="Diagram comparing the abandoned multi-process DP design with the shipped single-process lane-DP design" />
  <figcaption>Figure 3: The same four data-parallel KV caches, scheduled two ways. Top, the design we abandoned: four engine processes negotiate a shared prefill-or-decode mode over inter-process scatter/gather on every step, even though there is only one mesh submit and readout. Bottom, what we shipped: one engine process, a coordinator that picks the shared mode, four independent schedulers with lane-local block IDs, one merged device input, and results split back by lane.</figcaption>
</figure>

One subtlety took us a while to get right. If a forced prefill step admits zero tokens (typically because of KV pressure) while some lane still has running decode work, the step is retried in decode mode. Without that retry, KV pressure can drive the coordinator into a no-progress loop: prefill is selected because a lane *wants* to admit, admits nothing because no blocks are free, and the decode that would have freed those blocks never runs.

The user-facing surface for all of this is deliberately boring:

```bash
MESH_DEVICE=TG \
TT_LLAMA_TEXT_VER=llama3_70b_galaxy \
VLLM_RPC_TIMEOUT=900000 \
python examples/server_example_tt.py \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --data_parallel_size 4 \
  --max_num_seqs 8 \
  --async-scheduling \
  --additional-config.tt.dispatch_core_axis col \
  --additional-config.tt.sample_on_device_mode all \
  --additional-config.tt.fabric_config FABRIC_1D_RING \
  --additional-config.tt.worker_l1_size 1344544 \
  --additional-config.tt.trace_region_size 220000000
```

`--data_parallel_size 4 --max_num_seqs 8` becomes four in-process lanes of eight requests each: 32 concurrent, with `--max_num_seqs` meaning per-lane capacity. Users write the same flags they already know, and the backend maps them to whichever topology the model actually needs - in-process lanes for single-execute Galaxy models, ordinary multi-process DP with per-rank submeshes (discovered at startup and assigned via `TT_VISIBLE_DEVICES`) for everything else. The startup log states which one it chose.

## On-device sampling, with a fallback that nobody configures

When `sample_on_device_mode` is set, the mesh program carries sampling through to token selection and returns tokens rather than logits.

Plenty of requests can't use that path - logprobs, penalties, allowed-token masks, bad-word filtering, custom logits processors. The plugin does not reject them and does not ask the user to pick a mode. **It decides per batch**, falling back to vLLM's own `LogitProcessor` and sampler path whenever the batch needs something the device path cannot express, then returning to the device path when it can. Requests that need host-side sampling get correct results at the cost of a readback; everything else keeps the fast path. `always_compat_sampling` forces the host path for debugging or A/B comparison.

## Decode overlap is asynchronous readback, not an async execution model

The plugin supports decode/host overlap, gated on a per-model `supports_async_decode` declaration - if a model has not declared it, the platform disables async scheduling rather than letting a user turn on something unvalidated.

Underneath, "async" here means something narrower than it usually does, and the honest version is that it is **asynchronous host readback**, not a device-side execution thread:

1. Submit decode work with `read_from_device=False` (non-blocking).
2. Start host readback with `read_decode_output(..., async_read=True)` and keep the returned events with the submission record (also non-blocking).
3. Later, at finalization, wait on those events via `ttnn.event_synchronize(...)`.
4. Only then convert device output into host tensors and sampling results.

<figure>
  <img
    src="/assets/figures/2026-09-04-vllm-tt-plugin/async-decode.svg"
    width="100%"
    alt="Timeline showing decode overlap through asynchronous host readback" />
  <figcaption>Figure 4: Where the overlap comes from. Without it, the device waits while the host reads back and samples the previous step. With async decode the readback is left in flight, so the host schedules the next step and finalizes the previous one while the device is still busy - and the only blocking wait is <code>ttnn.event_synchronize()</code> at finalization.</figcaption>
</figure>

The engine keeps an in-flight queue of depth 2 and fills it before blocking, so the host can schedule step *N+1* while step *N*'s readback is still in flight. Overlap is kept only while the batch is *steady* - stable shape, on-device sampling, no structured-output bookkeeping, no resumed prefill. When any of those break, pending work is drained before proceeding.

So: prefill remains synchronous in practice, and decode overlap is a fast path for steady-state generation rather than a universal async pipeline. [`docs/SCHEDULING.md`](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/docs/SCHEDULING.md) in the plugin repo has the full treatment, including the finalization bookkeeping that keeps this correct when the executor's output thread and the engine thread race to the same result.

## Current limitations

`TTPlatform` rejects or adjusts unsupported combinations at configuration time, so users get a clear error before anything reaches the device rather than a failure mid-run:

- **Tensor parallel and pipeline parallel are supported, but differently.** Parallelism comes from the mesh shape (`MESH_DEVICE`) and model implementation, not from vLLM's TP/PP ranks.
- **Speculative decoding is not supported yet.**
- **LoRA is not supported yet.**
- **Prompt logprobs are not supported yet** and are rejected at request validation.
- **Prefix caching** is enabled only for models that declare support for it.
- **Async decode overlap** is enabled only for models that declare the capability.
- **Standard multi-process DP does not support MoE models.** Single-execute models needing internal data parallelism, such as GPT-OSS, fold into lane-DP instead.
- **Multi-host serving is not supported yet.** Tenstorrent hardware scales well past a single machine, but the current TT multi-host model implementation does not map directly onto vLLM's multi-host paradigm.

These are properties of the current Tenstorrent runtime and model implementations, not fundamental limits of the hardware, the software stack or of vLLM's plugin API. Each of them can be supported in the future, and the larger items are on the roadmap below.

## Try it out

Install [TT-Metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md) first and activate that environment, then clone the plugin and run its install script from the repository root:

```bash
git clone https://github.com/tenstorrent/vllm-tt-plugin.git
cd vllm-tt-plugin
source docs/install-vllm-tt.sh
```

The script builds vLLM with `VLLM_TARGET_DEVICE=empty` - the `tt` platform is supplied by the plugin at runtime - and installs the plugin. Then serve and query:

```bash
MESH_DEVICE=T3K VLLM_RPC_TIMEOUT=100000 python examples/server_example_tt.py
```

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B-Instruct", "prompt": "San Francisco is a", "max_tokens": 32}'
```

Existing OpenAI-client code needs no changes.

> [!NOTE]
> Setup currently performs a from-source vLLM build against **0.26.0** inside a tt-metal environment. Per-model commands, mesh shapes, and required environment variables are in the [plugin README](https://github.com/tenstorrent/vllm-tt-plugin) and the corresponding tt-metal model demos.

## What's next

- **Broader async decode coverage** - more model families declaring `supports_async_decode`, and fewer conditions that force a drain (especially on-device sampling modes).
- **Prefix caching across more models**, and lane-DP support for request-specific RoPE so vision models can use it.
- **Speculative decoding**, once the mesh-side draft/verify story is settled.
- **Multi-host serving** - scaling to models larger than one machine can hold.

## Acknowledgements

This work rests on the vLLM platform plugin mechanism contributed by the Ascend team and the pluggable-scheduler design contributed by the Spyre team - without the latter, a phase-based scheduler like ours would have meant a fork. Thanks to the vLLM maintainers for keeping the V1 extension points general enough that a mesh architecture fits through them.

We would like to thank the many talented people who have contributed to this work:
Viktor Puš, Tomasz Cheda, Sanjar Adylov, and Salar Hosseini.

We would especially like feedback on two things: whether folding `--data_parallel_size` into in-process lanes is the right user-facing surface for single-execute models, and which model families to prioritize next. Issues and pull requests are welcome on [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin), and we are reachable in the vLLM Slack.
